#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YODAS Turkish index — the cleaned YODAS of OWSM v4 (espnet/yodas_owsmv4, yodas0.10; CC-BY-3.0).

What it does:
  1) downloads the index files (wav.scp ~2.3 GB, text.ctc ~9.4 GB; .part + HF SHA-256 + resume)
  2) keeps only Turkish utterances; computes each utterance's ark file, byte range and duration
  3) downloads 30 sample utterances (only their own byte range) and decodes them (sample rate, channels, time unit)
  4) text format: share of lines with punctuation / upper case (a hint for manually written subtitles)
  5) summary: hours, utterances, arks, bytes to download; deletes the large index files (they can be re-downloaded)
Downloads NO audio except the samples. Output: <work>/yodas/index/{tr_utterances.tsv, tr_text.ctc}, used by yodas_download.py.

TR: YODAS (OWSM v4 temizlenmiş, CC-BY-3.0) Türkçe dizini: dizin dosyalarını indirir, Türkçe sözceleri ve bayt
aralıklarını çıkarır, 30 örnek çözer. Örnekler dışında ses indirmez.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import stamp, work_dir  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/espnet/yodas_owsmv4/resolve/main/"
SUBDIR = "dump/raw/yodas0.10/"
TURKISH = {"tur", "tr"}


def say(*a):
    print(*a, flush=True)


def stop(reason):
    say(f"STOP: {reason}")
    sys.exit(1)


def curl(*arg, quiet=True):
    cmd = ["curl", "--fail", "-L", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60"]
    if quiet:
        cmd.append("-sS")
    return subprocess.run(cmd + list(arg), capture_output=True)


def head(url):
    """Hugging Face LFS headers: x-linked-etag (SHA-256) and x-linked-size. The first response is read (no redirect)."""
    r = subprocess.run(["curl", "-sSI", "--connect-timeout", "30", "-m", "60", url], capture_output=True, text=True)
    h = {}
    for line in r.stdout.replace("\r", "").split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            h[k.lower()] = v.strip().strip('"')
    return h.get("x-linked-etag"), h.get("x-linked-size") or h.get("content-length")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 22), b""):
            h.update(part)
    return h.hexdigest()


def download(base, index_dir, name):
    url = base + SUBDIR + name
    target = os.path.join(index_dir, name)
    etag, size = head(url)
    say(f"  {name}: HF SHA {etag or 'none'} · size {size or '?'}")
    if os.path.exists(target) and (etag is None or sha256(target) == etag):
        say(f"  {name}: already present" + (" and SHA matches" if etag else " (no HF SHA, size check)"))
        return target
    t0 = time.time()
    r = subprocess.run(["curl", "--fail", "-L", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
                        "--retry", "2", "-C", "-", "--progress-bar", "-o", target + ".part", url])
    if r.returncode != 0:
        stop(f"{name} could not be downloaded (curl exit {r.returncode}) — the same command resumes")
    if etag:
        s = sha256(target + ".part")
        if s != etag:
            stop(f"{name}: SHA mismatch ({s} != {etag})")
    os.replace(target + ".part", target)
    say(f"  {name}: downloaded · {os.path.getsize(target) / 2**20:.0f} MB · {time.time() - t0:.0f} s")
    return target


def lang_and_time(key):
    """<recording_id>_<start>_<end>_<language>_asr — parsed from the right (the id may contain underscores)."""
    p = key.rsplit("_", 4)
    if len(p) != 5 or p[4] != "asr":
        return None, None, None, None
    try:
        return p[3], int(p[1]), int(p[2]), p[0]
    except ValueError:
        return p[3], None, None, p[0]


def ark_path(raw):
    """wav.scp may hold a cluster-specific absolute path; the repository path starts with 'dump/raw/'."""
    i = raw.find("dump/raw/")
    return raw[i:] if i >= 0 else raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--base-url", default=os.environ.get("AYAZ_ASR_YODAS_URL", BASE_URL),
                    help="espnet/yodas_owsmv4 base URL / temel adres (default: %(default)s)")
    a = ap.parse_args()
    work = work_dir(a.work_dir)
    base = a.base_url if a.base_url.endswith("/") else a.base_url + "/"
    index_dir = os.path.join(work, "yodas", "index")
    sample_dir = os.path.join(work, "yodas", "samples")
    summary_path = os.path.join(work, "runs", f"yodas_index_{stamp()}.json")
    for d in (index_dir, sample_dir, os.path.dirname(summary_path)):
        os.makedirs(d, exist_ok=True)
    summary = dict(step="yodas_index", source=base + SUBDIR, ok=False)
    say(f"=== 1. index files ({time.strftime('%H:%M:%S')})")
    try:
        r = subprocess.run(["curl", "-sSL", "--fail", "--connect-timeout", "30", "-m", "60", base + SUBDIR + "stats.txt"],
                           capture_output=True, text=True)
        lines = [s for s in r.stdout.splitlines() if s.startswith(("tur ", "Total", "Languages"))]
        summary["stats"] = lines
        for s in lines:
            say(f"  stats.txt: {s}")
    except Exception as e:
        say(f"  stats.txt not readable: {e!r}")
    wav_scp = download(base, index_dir, "wav.scp")
    text_ctc = download(base, index_dir, "text.ctc")

    say(f"\n=== 2. Turkish selection — wav.scp, pass 1 ({time.strftime('%H:%M:%S')})")
    langs = collections.Counter()
    tr_arks = set()
    first = []
    n = 0
    with open(wav_scp, encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if len(first) < 3:
                first.append(line.rstrip()[:200])
            part = line.split(maxsplit=1)
            if len(part) != 2:
                continue
            lang = lang_and_time(part[0])[0]
            langs[lang] += 1
            if lang in TURKISH:
                tr_arks.add(ark_path(part[1].strip().rsplit(":", 1)[0]))
            if n % 5_000_000 == 0:
                say(f"  … {n:,} lines · Turkish arks {len(tr_arks)}")
    summary["wav_scp_lines"] = n
    summary["top_languages"] = langs.most_common(10)
    summary["turkish_arks"] = len(tr_arks)
    say(f"  lines {n:,} · first lines: {first}")
    say(f"  most frequent language codes: {langs.most_common(10)}")
    say(f"  ark files with Turkish utterances: {len(tr_arks)}")
    if not tr_arks:
        stop("no Turkish utterance in wav.scp — see the language code list above")

    say(f"\n=== 2b. wav.scp, pass 2: byte ranges of Turkish utterances ({time.strftime('%H:%M:%S')})")
    arks = collections.defaultdict(list)  # ark -> [(offset, key_length, key_or_None)]
    with open(wav_scp, encoding="utf-8", errors="replace") as f:
        for line in f:
            part = line.split(maxsplit=1)
            if len(part) != 2:
                continue
            path, offset = part[1].strip().rsplit(":", 1)
            ark = ark_path(path)
            if ark not in tr_arks:
                continue
            lang = lang_and_time(part[0])[0]
            arks[ark].append((int(offset), len(part[0].encode("utf-8")), part[0] if lang in TURKISH else None))
    for lst in arks.values():
        lst.sort()
    # arks whose last entry is Turkish need their size (the last utterance ends at the end of file); HEAD only those
    needed = sorted(a for a, lst in arks.items() if lst and lst[-1][2] is not None)
    sizes = {}
    for i, ark in enumerate(needed):
        _, b = head(base + ark)
        sizes[ark] = int(b) if b and b.isdigit() else None
        if (i + 1) % 25 == 0:
            say(f"  … ark size {i + 1}/{len(needed)}")
    say(f"  arks {len(arks)} · sizes queried {len(needed)}")
    rows = []
    for ark, lst in arks.items():
        for j, (offset, _, key) in enumerate(lst):
            if key is None:
                continue
            if j + 1 < len(lst):
                end = lst[j + 1][0] - lst[j + 1][1] - 1
            else:
                end = sizes.get(ark)
            if end is None or end <= offset:
                continue
            lang, start, stop_, video = lang_and_time(key)
            rows.append((key, ark, offset, end - offset, start, stop_, video))
    total_bytes = sum(k[3] for k in rows)
    summary["turkish_utterances"] = len(rows)
    summary["turkish_bytes_gb"] = round(total_bytes / 2**30, 2)
    summary["turkish_videos"] = len({k[6] for k in rows})
    say(f"  Turkish utterances {len(rows):,} · videos {summary['turkish_videos']:,} · audio bytes {total_bytes / 2**30:.1f} GB")

    say(f"\n=== 3. sample utterances: own byte range only ({time.strftime('%H:%M:%S')})")
    import numpy as np
    import soundfile as sf
    rnd = random.Random(28092026)
    samples = rnd.sample(rows, min(30, len(rows)))
    meas = []
    for i, (key, ark, offset, length, start, end, _) in enumerate(samples):
        r = curl("-r", f"{offset}-{offset + length - 1}", base + ark)
        data = r.stdout
        if r.returncode != 0 or len(data) != length:
            say(f"  [{i}] not downloaded (exit {r.returncode}, {len(data)}/{length} bytes)")
            continue
        try:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        except Exception as e:
            say(f"  [{i}] not decodable: {e!r} · first bytes {data[:16].hex()}")
            continue
        dur = len(audio) / sr
        ch = 1 if audio.ndim == 1 else audio.shape[1]
        meas.append(dict(key=key, sr=sr, channels=ch, duration=round(dur, 3), start=start, end=end,
                         format=data[:4].decode("latin-1"), bytes=length))
        if i < 5:
            sf.write(os.path.join(sample_dir, f"sample_{i:02d}.wav"), audio if audio.ndim == 1 else audio.mean(axis=1), sr,
                     subtype="PCM_16")
        say(f"  [{i}] {data[:4]!r} · {sr} Hz · {ch} ch · {dur:.2f} s · id range {start}–{end}")
    if meas:
        ratios = [o["duration"] / ((o["end"] - o["start"]) or 1) for o in meas if o["start"] is not None and o["end"] is not None]
        ratio = float(np.median(ratios)) if ratios else None
        summary["samples"] = len(meas)
        summary["sample_rates"] = sorted({o["sr"] for o in meas})
        summary["channels"] = sorted({o["channels"] for o in meas})
        summary["formats"] = sorted({o["format"] for o in meas})
        summary["id_time_unit_s"] = ratio
        say(f"  samples {len(meas)} · rates {summary['sample_rates']} · channels {summary['channels']} · "
            f"formats {summary['formats']} · id time unit ≈ {ratio} s")
        if ratio:
            durs = [(k[5] - k[4]) * ratio for k in rows if k[4] is not None and k[5] is not None]
            summary["turkish_hours"] = round(sum(durs) / 3600, 1)
            summary["duration_median_s"] = round(float(np.median(durs)), 2)
            summary["duration_p90_s"] = round(float(np.percentile(durs, 90)), 2)
            say(f"  Turkish total ≈ {summary['turkish_hours']} h · utterance median {summary['duration_median_s']} s, "
                f"p90 {summary['duration_p90_s']} s")
    else:
        say("  ! no sample could be decoded — inspect the audio format by hand")

    say(f"\n=== 4. text ({time.strftime('%H:%M:%S')}) — Turkish lines of text.ctc")
    tr_keys = {k[0] for k in rows}
    tr_text = os.path.join(index_dir, "tr_text.ctc")
    n_text = n_punct = n_upper = 0
    examples = []
    with open(text_ctc, encoding="utf-8", errors="replace") as f, open(tr_text, "w", encoding="utf-8") as g:
        for line in f:
            part = line.rstrip("\n").split(maxsplit=1)
            if not part or part[0] not in tr_keys:
                continue
            txt = part[1] if len(part) > 1 else ""
            g.write(line if line.endswith("\n") else line + "\n")
            n_text += 1
            n_punct += bool(re.search(r"[.,?!]", txt))
            n_upper += bool(re.search(r"[A-ZÇĞİÖŞÜ]", txt))
            if len(examples) < 12 and rnd.random() < 0.001:
                examples.append(txt[:160])
    summary["text_lines"] = n_text
    summary["text_punctuated_share"] = round(n_punct / max(1, n_text), 3)
    summary["text_uppercase_share"] = round(n_upper / max(1, n_text), 3)
    say(f"  Turkish text lines {n_text:,} · punctuated {100 * summary['text_punctuated_share']:.1f}% · "
        f"with upper case {100 * summary['text_uppercase_share']:.1f}%")
    for m in examples:
        say(f"    «{m}»")

    with open(os.path.join(index_dir, "tr_utterances.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "ark", "offset", "bytes", "start", "end", "video"])
        w.writerows(rows)
    with open(os.path.join(sample_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Five random utterances from the Turkish part of YODAS (OWSM v4 cleaned, yodas0.10).\n"
                "Source: espnet/yodas_owsmv4 (CC-BY-3.0). For listening to the audio quality.\n")
        for o in meas[:5]:
            f.write(f"{o['key']}\t{o['sr']} Hz\t{o['duration']} s\n")

    # the large index files can be re-downloaded; the Turkish extracts stay
    for name in ("wav.scp", "text.ctc"):
        try:
            os.remove(os.path.join(index_dir, name))
            say(f"  deleted (can be re-downloaded): {name}")
        except OSError:
            pass
    summary["ok"] = bool(rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    say("\n================ SUMMARY ================")
    for k, v in summary.items():
        say(f"  {k}: {v}")
    say(f"\nOutputs: {summary_path} · {index_dir}/tr_utterances.tsv · {index_dir}/tr_text.ctc · {sample_dir}/")


if __name__ == "__main__":
    main()
