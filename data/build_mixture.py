#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training mixture = Turkish manifest + optional additional in-house manifests (each row repeated N times)
+ rehearsal (German and English rows repeated --priority-copies times), shuffled with a fixed seed.

Row order before shuffling (as in v0.1): turkish + extra_1 rows x N_1 + extra_2 rows x N_2 + ... + rehearsal rows.
v0.1 used two additional in-house manifests that are NOT distributed (consented volunteer voice messages x 3, then
short utterances / voice commands x 5). Without them the mixture is built from public data only.

TR: Eğitim karışımı: Türkçe manifest + isteğe bağlı dağıtılmayan veri manifestleri (her satır N kez; rızalı)
+ prova (Almanca ve İngilizce satırlar 2 kez); sabit tohumla karıştırılır.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import read_jsonl, work_dir  # noqa: E402

SEED = 2102026
PRIORITY_PROMPTS = ("de-DE", "en-US")
MAX_S = 10.0


def say(*a):
    print("  [mixture]", *a, flush=True)


def build(turkish, extras, rehearsal, priority_copies, seed):
    rows = list(turkish)
    for lst, k in extras:
        rows += [x for x in lst for _ in range(max(1, k))]
    rows += [x for x in rehearsal for _ in range(max(1, priority_copies) if x.get("target_lang") in PRIORITY_PROMPTS else 1)]
    random.Random(seed).shuffle(rows)
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    return rows, text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--turkish", dest="tr_manifest", help="Turkish manifest (default: <work>/manifests/turkish.jsonl)")
    ap.add_argument("--rehearsal", help="filtered rehearsal manifest (default: <work>/manifests/rehearsal_filtered.jsonl)")
    ap.add_argument("--extra", action="append", default=[], metavar="PATH:COPIES",
                    help="OPTIONAL additional consented in-house Turkish manifest, not distributed; repeatable, order matters "
                         "(v0.1: voice messages :3, then short utterances/commands :5) / isteğe bağlı dağıtılmayan veri")
    ap.add_argument("--priority-copies", type=int, default=2, help="copies of de-DE / en-US rehearsal rows (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED, help="shuffle seed (default: %(default)s)")
    ap.add_argument("--out", help="mixture manifest (default: <work>/manifests/mixture.jsonl)")
    ap.add_argument("--expected-sha", default="", help="stop if the mixture's SHA-256 differs")
    a = ap.parse_args()
    work = work_dir(a.work_dir)
    tr_path = a.tr_manifest or os.path.join(work, "manifests", "turkish.jsonl")
    reh_path = a.rehearsal or os.path.join(work, "manifests", "rehearsal_filtered.jsonl")
    out = a.out or os.path.join(work, "manifests", "mixture.jsonl")

    tr = read_jsonl(tr_path)
    if any(x.get("target_lang") != "tr-TR" for x in tr):
        sys.exit("STOP: the Turkish manifest has rows that are not tr-TR")
    reh = read_jsonl(reh_path)
    if not reh:
        sys.exit(f"STOP: rehearsal manifest empty: {reh_path}")
    for x in reh:
        if x.get("target_lang") in (None, "", "tr-TR") or not x.get("text") or float(x["duration"]) > MAX_S:
            sys.exit(f"STOP: rehearsal row breaks the rules: {os.path.basename(x.get('audio_filepath', '?'))}")
    extras = []
    for spec in a.extra:
        path, _, k = spec.rpartition(":")
        if not path or not k.isdigit():
            sys.exit(f"STOP: --extra must be PATH:COPIES, got {spec!r}")
        lst = read_jsonl(path)
        if not lst:
            sys.exit(f"STOP: additional manifest empty: {os.path.basename(path)}")
        for x in lst:
            if x.get("target_lang") != "tr-TR" or not x.get("text") or float(x["duration"]) > MAX_S:
                sys.exit(f"STOP: additional row breaks the rules (tr-TR, text, <= {MAX_S:g} s): "
                         f"{os.path.basename(x.get('audio_filepath', '?'))}")
            if not os.path.isfile(x["audio_filepath"]):
                sys.exit(f"STOP: audio missing: {os.path.basename(x['audio_filepath'])}")
        extras.append((lst, int(k)))

    rows, text, sha = build(tr, extras, reh, a.priority_copies, a.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(out + ".part", out)
    dur = lambda lst: sum(float(x["duration"]) for x in lst)
    total = dur(rows) or 1.0
    yodas = [x for x in tr if "video" in x]
    empty = [x for x in tr if not x.get("text")]
    langs = collections.Counter(x["target_lang"] for x in reh)
    reh_rows = sum(a.priority_copies if x.get("target_lang") in PRIORITY_PROMPTS else 1 for x in reh)
    say(f"Turkish {len(tr)} rows ({dur(tr) / 3600:.2f} h; YODAS {len(yodas)}, empty {len(empty)})"
        + "".join(f" + extra {len(lst)} x {k}" for lst, k in extras)
        + f" + rehearsal {len(reh)} rows ({dur(reh) / 3600:.2f} h, {len(langs)} languages; de/en x {a.priority_copies} -> {reh_rows} rows)"
        + f" · total {len(rows)} rows, {total / 3600:.2f} h")
    say(f"duration shares: YODAS {dur(yodas) / total:.1%} · rest of Turkish manifest {(dur(tr) - dur(yodas)) / total:.1%}"
        + "".join(f" · extra {i + 1} {dur(lst) * k / total:.1%}" for i, (lst, k) in enumerate(extras))
        + f" · rehearsal {(total - dur(tr) - sum(dur(lst) * k for lst, k in extras)) / total:.1%}")
    say(f"written {out} · SHA-256 {sha}")
    if a.expected_sha and sha != a.expected_sha:
        sys.exit(f"STOP: mixture SHA-256 {sha[:16]}… != expected {a.expected_sha[:16]}…")
    print(f"  [mixture] MIXTURE_OK {sha}", flush=True)


if __name__ == "__main__":
    main()
