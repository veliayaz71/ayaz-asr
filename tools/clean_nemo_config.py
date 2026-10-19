#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean a trained .nemo before publishing: model_config.yaml of a model saved by train.py contains your local
manifest paths (train_ds) and possibly other local strings.

What it does (follows the procedure used for the published v0.1 .nemo):
  (a) every string value matching a sensitive pattern (--pattern; defaults: macOS/Linux home-directory paths, .jsonl,
      .wav, and your login name) -> null  (values under shard_manifests are left alone)
  (b) every non-empty manifest_filepath / tarred_audio_filepaths -> null
  The cleaned YAML is read back and rescanned (text and key tree: 0 matches required). The .nemo tar is rewritten
  STREAMING with the original member order and names, GNU format, uid/gid 0, empty user/group names, fixed mtime;
  the other members (weights, tokenizer, vocabulary) are copied byte for byte and their SHA-256 is verified. The weight
  zip's member names and data.pkl are scanned as well. Only counts, key paths and file names are printed — never values.
  --check-restore additionally loads the cleaned file with NeMo (CPU, HF_HUB_OFFLINE=1) and prints the parameter count.

Usage: clean_nemo_config.py <in.nemo> <out.nemo> [--pattern REGEX ...] [--check-restore]

TR: Yayından önce .nemo içindeki model_config.yaml'daki yerel manifest yollarını ve hassas dizgeleri null yapar,
paketi özgün üye sırasıyla, sabit başlıklarla yeniden yazar ve doğrular.
"""
import argparse
import difflib
import getpass
import hashlib
import io
import os
import re
import sys
import tarfile
import tempfile
import zipfile

PRIVATE_KEYS = ("manifest_filepath", "tarred_audio_filepaths")
SKIP_KEYS = PRIVATE_KEYS + ("shard_manifests",)
FIXED_MTIME = 1791849600
LIMIT = 40
D = []   # (label, compiled regex)


class Stop(Exception):
    """STOP condition; the message carries only counts, key paths and file names."""


def count(text):
    out = {}
    if not text:
        return out
    for label, rx in D:
        n = len(rx.findall(text))
        if n:
            out[label] = n
    return out


def summary(d):
    return " · ".join("%s %d" % (k, v) for k, v in sorted(d.items()))


def sensitive(s):
    return bool(count(s))


def empty(v):
    return v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0)


def walk(node, path, a, b):
    """(a) sensitive string value outside PRIVATE/SKIP keys -> None; (b) non-empty PRIVATE key -> None."""
    if isinstance(node, dict):
        for k in list(node.keys()):
            v = node[k]
            kp = "%s.%s" % (path, k) if path else str(k)
            if k in PRIVATE_KEYS:
                if not empty(v):
                    node[k] = None
                    b.append(kp)
                continue
            if k in SKIP_KEYS:
                continue
            if isinstance(v, str):
                if sensitive(v):
                    node[k] = None
                    a.append(kp)
            elif isinstance(v, (dict, list)):
                walk(v, kp, a, b)
    elif isinstance(node, list):
        for i in range(len(node)):
            v = node[i]
            kp = "%s[%d]" % (path, i)
            if isinstance(v, str):
                if sensitive(v):
                    node[i] = None
                    a.append(kp)
            elif isinstance(v, (dict, list)):
                walk(v, kp, a, b)


def remaining(node, path, paths, total):
    """Matches left in key names and string values."""
    if isinstance(node, dict):
        for k, v in node.items():
            kp = "%s.%s" % (path, k) if path else str(k)
            n = sum(count(str(k)).values())
            if isinstance(v, str):
                n += sum(count(v).values())
            if n:
                paths.append(kp)
                total[0] += n
            if isinstance(v, (dict, list)):
                remaining(v, kp, paths, total)
    elif isinstance(node, list):
        for i in range(len(node)):
            v = node[i]
            kp = "%s[%d]" % (path, i)
            if isinstance(v, str):
                n = sum(count(v).values())
                if n:
                    paths.append(kp)
                    total[0] += n
            elif isinstance(v, (dict, list)):
                remaining(v, kp, paths, total)


def print_paths(paths):
    for p in paths[:LIMIT]:
        print("      " + p, flush=True)
    if len(paths) > LIMIT:
        print("      … +%d more paths" % (len(paths) - LIMIT), flush=True)


def shown(m):
    return "./" if m.isdir() and m.name in (".", "./") else m.name


class HashReader:
    """Computes SHA-256 while streaming (the multi-GB weights are not loaded into memory)."""

    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
        self.n = 0

    def read(self, n=-1):
        part = self.f.read(n)
        self.h.update(part)
        self.n += len(part)
        return part


def stream_sha(f):
    h = hashlib.sha256()
    while True:
        part = f.read(4 << 20)
        if not part:
            break
        h.update(part)
    return h.hexdigest()


def clean(src, dst, work):
    from omegaconf import OmegaConf
    with tarfile.open(src, "r:") as t:
        members = t.getmembers()
        print("  original tar members: %d" % len(members), flush=True)
        for m in members:
            kind = "dir" if m.isdir() else ("%d bytes" % m.size if m.isfile() else "UNEXPECTED TYPE")
            print("    %s  %s" % (shown(m), kind), flush=True)
        dirs = [m for m in members if m.isdir()]
        files = [m for m in members if m.isfile()]
        if len(dirs) + len(files) != len(members):
            raise Stop("tar has members that are neither directories nor files")
        if len(dirs) != 1 or dirs[0].name not in (".", "./"):
            raise Stop("a single root directory member ('./') was expected")
        for m in files:
            if not m.name.startswith("./") or m.name.count("/") != 1 or ".." in m.name:
                raise Stop("unexpected member path: %s" % m.name)

        def one(cond, name):
            lst = [m for m in files if cond(m.name)]
            if len(lst) != 1:
                raise Stop("%s member not unique (%d)" % (name, len(lst)))
            return lst[0]

        cfg = one(lambda n: n == "./model_config.yaml", "model_config.yaml")
        wts = one(lambda n: n == "./model_weights.ckpt", "model_weights.ckpt")
        tok = one(lambda n: n.endswith("_tokenizer.model"), "*_tokenizer.model")
        voc = one(lambda n: n.endswith("_vocab.txt"), "*_vocab.txt")
        if len(files) != 4:
            raise Stop("unexpected file member (%d files, 4 expected)" % len(files))
        raw = t.extractfile(cfg).read()

    old_path = os.path.join(work, "model_config.original.yaml")
    with open(old_path, "wb") as f:
        f.write(raw)
    old_text = raw.decode("utf-8", "replace")
    old_lines = old_text.splitlines()
    print("  OK model_config.yaml extracted: %d bytes · %d lines" % (len(raw), len(old_lines)), flush=True)
    before = count(old_text)
    print("  sensitive matches in the original yaml: %d%s" % (sum(before.values()), (" (" + summary(before) + ")") if before else ""), flush=True)

    tree = OmegaConf.to_container(OmegaConf.load(old_path), resolve=False)
    a, b = [], []
    walk(tree, "", a, b)
    print("  (a) string values matching a sensitive pattern -> null: %d keys" % len(a), flush=True)
    print_paths(a)
    print("  (b) manifest_filepath / tarred_audio_filepaths (non-empty) -> null: %d keys" % len(b), flush=True)
    print_paths(b)
    print("  OK changed keys: %d (values are not printed)" % (len(a) + len(b)), flush=True)

    new_path = os.path.join(work, "model_config.yaml")
    OmegaConf.save(OmegaConf.create(tree), new_path)
    with open(new_path, "rb") as f:
        new_bytes = f.read()
    new_text = new_bytes.decode("utf-8")
    left_text = count(new_text)
    tree2 = OmegaConf.to_container(OmegaConf.load(new_path), resolve=False)
    print("  OK new yaml read back with OmegaConf.load", flush=True)
    paths, total = [], [0]
    remaining(tree2, "", paths, total)
    if left_text or total[0]:
        print("    key paths with remaining matches:", flush=True)
        print_paths(paths)
        raise Stop("sensitive matches left after cleaning: text %d (%s) · key tree %d"
                   % (sum(left_text.values()), summary(left_text), total[0]))
    print("  OK rescan (whole text + key tree): 0 matches", flush=True)
    new_lines = new_text.splitlines()
    minus = plus = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes():
        if tag != "equal":
            minus += i2 - i1
            plus += j2 - j1
    print("  lines: before %d · after %d · changed: -%d +%d" % (len(old_lines), len(new_lines), minus, plus), flush=True)

    tmp = dst + ".writing"
    orig_sha = {}
    with tarfile.open(src, "r:") as tin, tarfile.open(tmp, "w:", format=tarfile.GNU_FORMAT, copybufsize=4 << 20) as tout:
        for m in tin.getmembers():
            ti = tarfile.TarInfo(m.name)
            ti.mtime = FIXED_MTIME
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            if m.isdir():
                ti.type = tarfile.DIRTYPE
                ti.mode = 0o755
                tout.addfile(ti)
            elif m.name == cfg.name:
                ti.type = tarfile.REGTYPE
                ti.mode = 0o644
                ti.size = len(new_bytes)
                tout.addfile(ti, io.BytesIO(new_bytes))
            else:
                ti.type = tarfile.REGTYPE
                ti.mode = 0o644
                ti.size = m.size
                r = HashReader(tin.extractfile(m))
                tout.addfile(ti, r)
                if r.n != m.size:
                    raise Stop("copy size mismatch: %s" % m.name)
                orig_sha[m.name] = r.h.hexdigest()
                print("    streamed: %s · %d bytes" % (m.name, r.n), flush=True)

    with tarfile.open(tmp, "r:") as t:
        new = t.getmembers()
        if [m.name for m in new] != [m.name for m in members]:
            raise Stop("member order/names of the new tar differ from the original")
        print("  OK new tar: %d members · original order and names · GNU · uid/gid 0 · empty uname/gname · mtime %d"
              % (len(new), FIXED_MTIME), flush=True)
        for m in new:
            mode = 0o755 if m.isdir() else 0o644
            if m.uid or m.gid or m.uname or m.gname or m.mtime != FIXED_MTIME or m.mode != mode or m.pax_headers:
                raise Stop("member header not as expected: %s" % shown(m))
        if t.pax_headers:
            raise Stop("global pax header in the new tar")
        for m in (wts, tok, voc):
            h = stream_sha(t.extractfile(t.getmember(m.name)))
            if h != orig_sha.get(m.name):
                raise Stop("SHA-256 differs from the original member: %s" % m.name)
            print("    OK SHA-256 identical to the original member: %s · %s…" % (m.name, h[:16]), flush=True)
        y = t.extractfile(t.getmember(cfg.name)).read()
        if y != new_bytes:
            raise Stop("model_config.yaml in the package differs from the written one")
        check = os.path.join(work, "model_config.package.yaml")
        with open(check, "wb") as f:
            f.write(y)
        OmegaConf.load(check)
        if count(y.decode("utf-8")):
            raise Stop("sensitive match in the packaged model_config.yaml")
        print("    OK packaged model_config.yaml read back · 0 sensitive matches", flush=True)
        f = t.extractfile(t.getmember(wts.name))
        if not zipfile.is_zipfile(f):
            raise Stop("model_weights.ckpt is not a zip (torch.save format expected)")
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            names = z.namelist()
            n_names = sum(sum(count(nm).values()) for nm in names)
            pkl = [nm for nm in names if nm == "data.pkl" or nm.endswith("/data.pkl")]
            if len(pkl) != 1:
                raise Stop("no single data.pkl in the weight zip (%d)" % len(pkl))
            data = z.read(pkl[0])
        n_pkl = count(data.decode("utf-8", "replace"))
        print("  weight zip: %d members · data.pkl %d bytes · sensitive matches: member names %d · data.pkl %d%s"
              % (len(names), len(data), n_names, sum(n_pkl.values()), (" (" + summary(n_pkl) + ")") if n_pkl else ""), flush=True)
        if n_names or n_pkl:
            raise Stop("sensitive match in the weight file")
        print("  OK weight zip scan clean", flush=True)
    os.replace(tmp, dst)
    print("  OK written: %s · %d bytes" % (os.path.basename(dst), os.path.getsize(dst)), flush=True)


def check_restore(path):
    """NeMo restore test (CPU, offline)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.restore_from(path, map_location="cpu")
    n = sum(int(p.numel()) for p in model.parameters())
    print("  OK restore: class %s · parameters %d" % (type(model).__name__, n), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="input .nemo / girdi")
    ap.add_argument("dst", help="cleaned output .nemo / temiz çıktı")
    ap.add_argument("--pattern", action="append", default=[],
                    help="extra sensitive regex (case-insensitive), repeatable — e.g. your data directory name / ek desen")
    ap.add_argument("--no-default-patterns", action="store_true", help="do not use the default patterns")
    ap.add_argument("--no-login-name", action="store_true", help="do not add the current login name as a pattern")
    ap.add_argument("--check-restore", action="store_true", help="load the cleaned .nemo with NeMo afterwards (CPU)")
    a = ap.parse_args()
    if not a.no_default_patterns:
        D.extend((label, re.compile(rx, re.I)) for label, rx in (
            ("home directory path", r"/(?:Users|home)/"), ("jsonl", r"\.jsonl"), ("wav", r"\.wav")))
    if not a.no_login_name:
        try:
            login = getpass.getuser()
        except Exception:
            login = ""
        if len(login) >= 3:
            D.append(("login name", re.compile(re.escape(login), re.I)))
    D.extend(("pattern %d" % (i + 1), re.compile(rx, re.I)) for i, rx in enumerate(a.pattern))
    if not D:
        sys.exit("STOP: no patterns")
    print("  patterns: " + ", ".join(label for label, _ in D), flush=True)
    try:
        with tempfile.TemporaryDirectory() as work:
            clean(os.path.abspath(a.src), os.path.abspath(a.dst), work)
        if a.check_restore:
            check_restore(os.path.abspath(a.dst))
    except Stop as e:
        part = os.path.abspath(a.dst) + ".writing"
        if os.path.exists(part):
            os.remove(part)
        print("  STOP: %s" % e, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
