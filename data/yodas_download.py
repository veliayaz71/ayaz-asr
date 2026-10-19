#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YODAS Turkish — audio of videos with manually written subtitles: download, convert to 16 kHz, cut into <= 12 s
clips along subtitle lines.

Sources: espnet/yodas_owsmv4 (OWSM v4's cleaned YODAS, yodas0.10; CC-BY-3.0) +
         espnet/yodas2 data/tr000/text (list of manually subtitled videos; only the ids are used)
Input:   <work>/yodas/index/{tr_utterances.tsv, tr_text.ctc}  (output of yodas_index.py)
Stages (each resumes where it stopped):
  1) manual-subtitle list (YODAS2 tr000, ~52 MB JSON) -> video ids
  2) time-stamped text (OWSM `text`, 12.3 GB) -> only Turkish lines are kept, the large file is deleted
  3) selection: manually subtitled ∩ punctuated (>= 50 %) ∩ Turkish letters (>= 15 %); clip plan from subtitle lines (<= 12 s)
  4) samples: 12 utterances are decoded, 5 clips written as WAV for listening — STOP if none decodes
  5) download: only each utterance's own byte range; FLAC -> 16 kHz mono -> clips (FLAC) + manifest
Outputs: <work>/datasets/yodas/{audio/<video>/*.flac, manifest_yodas.jsonl} · <work>/runs/yodas_download_<stamp>.json

TR: Elle alt yazılı YODAS videolarının sesini indirir, 16 kHz'e çevirir, alt yazı satırlarına göre ≤ 12 sn parçalara
böler ve manifest yazar. Her aşama kaldığı yerden sürer.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import stamp, work_dir  # noqa: E402
from ayaz_asr.text import tag_sentences  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/espnet/yodas_owsmv4/resolve/main/"
SUBDIR = "dump/raw/yodas0.10/"
YODAS2_URL = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data/tr000/text/"
TARGET_SR = 16000
CLIP_MAX = 12.0
CLIP_MIN = 1.0
CLIP_SENTENCE_S = 4.0   # past this duration, cut at a sentence end
GAP_MAX = 0.8           # a gap this long between lines closes the clip
MARGIN = 0.15           # margin added before and after a clip (without running into the neighbouring line)
TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")
TIMESTAMP = re.compile(r"<(\d+\.\d{2})>")

BASE = BASE_URL
Y2 = YODAS2_URL
AUDIO = INDEX = SAMPLES = None


def say(*a):
    print(*a, flush=True)


def stop(reason):
    say(f"STOP: {reason}")
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for p in iter(lambda: f.read(1 << 22), b""):
            h.update(p)
    return h.hexdigest()


def head(url):
    r = subprocess.run(["curl", "-sSI", "--connect-timeout", "30", "-m", "60", url], capture_output=True, text=True)
    h = {}
    for line in r.stdout.replace("\r", "").split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            h[k.lower()] = v.strip().strip('"')
    return h.get("x-linked-etag"), h.get("x-linked-size") or h.get("content-length")


def download_large(url, target):
    """Large file: .part + resume + HF SHA-256 (x-linked-etag) check. Progress line every 30 s."""
    etag, size = head(url)
    say(f"  {os.path.basename(target)}: HF SHA {etag or 'none'} · size {size or '?'}")
    if os.path.exists(target) and (etag is None or sha256(target) == etag):
        say("  already present" + (" and SHA matches" if etag else ""))
        return
    p = subprocess.Popen(["curl", "-sS", "--fail", "-L", "--connect-timeout", "30", "--speed-limit", "1024",
                          "--speed-time", "60", "--retry", "2", "-C", "-", "-o", target + ".part", url])
    t0 = time.time()
    last = t0
    while p.poll() is None:
        time.sleep(1)
        if time.time() - last < 30:
            continue
        last = time.time()
        try:
            mb = os.path.getsize(target + ".part") / 2**20
        except OSError:
            mb = 0
        total = f" / {int(size) / 2**20:.0f}" if size and size.isdigit() else ""
        say(f"  … {mb:.0f}{total} MB · {time.time() - t0:.0f} s")
    if p.returncode != 0:
        stop(f"{url} could not be downloaded (curl exit {p.returncode}) — the same command resumes")
    if etag:
        s = sha256(target + ".part")
        if s != etag:
            stop(f"SHA mismatch ({s} != {etag}) — inspect the .part file")
    os.replace(target + ".part", target)
    say(f"  downloaded · {os.path.getsize(target) / 2**20:.0f} MB · {time.time() - t0:.0f} s")


def get_small(url):
    """Small file: (http_code, bytes)."""
    r = subprocess.run(["curl", "-sS", "-L", "--connect-timeout", "30", "-m", "180", "--retry", "2",
                        "-w", "\n%{http_code}", url], capture_output=True)
    body, _, code = r.stdout.rpartition(b"\n")
    return code.decode(errors="replace").strip(), body


# ---------------------------------------------------------------- 1) manual-subtitle list
def manual_list():
    target = os.path.join(INDEX, "manual_subtitle_videos.txt")
    if os.path.exists(target):
        return {x.strip() for x in open(target, encoding="utf-8") if x.strip()}
    ids = set()
    i = 0
    while i < 2000:
        code, body = get_small(f"{Y2}{i:08d}.json")
        if code == "404":
            break
        if code != "200":
            stop(f"YODAS2 tr000 list: HTTP {code} for {i:08d}.json")
        for rec in json.loads(body.decode("utf-8")):
            ids.add(rec["audio_id"])
        i += 1
        if i % 25 == 0:
            say(f"  … {i} files · {len(ids)} videos")
    if i == 0:
        stop("no file could be fetched from the YODAS2 tr000 list")
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(ids)) + "\n")
    say(f"  YODAS2 tr000: {i} files · {len(ids)} manually subtitled videos")
    return ids


# ---------------------------------------------------------------- 2) time-stamped text
def timed_text(keys):
    target = os.path.join(INDEX, "tr_text_timed.txt")
    if os.path.exists(target) and os.path.exists(target + ".done"):
        say("  already extracted")
    else:
        big = os.path.join(INDEX, "text")
        download_large(BASE + SUBDIR + "text", big)
        n = k = 0
        with open(big, encoding="utf-8", errors="replace") as f, open(target, "w", encoding="utf-8") as g:
            for line in f:
                n += 1
                if line.split(" ", 1)[0] in keys:
                    g.write(line if line.endswith("\n") else line + "\n")
                    k += 1
                if n % 5_000_000 == 0:
                    say(f"  … {n:,} lines · Turkish {k:,}")
        open(target + ".done", "w").close()
        os.remove(big)
        say(f"  {k:,} Turkish lines out of {n:,}; the large file was deleted (can be re-downloaded)")
    timed = {}
    with open(target, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            if len(p) == 2:
                timed[p[0]] = p[1]
    return timed


def split_lines(body):
    """'<tur><asr><0.00> text<1.84><1.92> text2<3.20>' -> [(0.00, 1.84, 'text'), (1.92, 3.20, 'text2')]
    OWSM v4 format: every subtitle line is '<start> text<end>' (espnet egs2/owsm_v4/s2t1/local/utils.py)."""
    import html
    body = re.sub(r"<(?!\d+\.\d{2}>)[^<>]*>", "", body)  # language, task, <notimestamps>, <i> and similar tags
    part = TIMESTAMP.split(body)  # ['', t1, text1, t2, '', t3, text2, t4, ...]
    out = []
    for i in range(1, len(part) - 2, 4):
        start, txt, end = float(part[i]), part[i + 1], float(part[i + 2])
        txt = html.unescape(txt)
        txt = re.sub(r"^\s*[-–—]\s*", "", txt)  # dialogue dash
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt and end > start:
            out.append((start, end, txt))
    return out


def clip_plan(lines):
    """Collect consecutive subtitle lines into <= 12 s clips; past 4 s cut at a sentence end; always cut at a gap longer
    than 0.8 s (the gap may hold speech that was not subtitled). Returns [(start, end, text, prev_end, next_start)]."""
    groups, cur = [], []
    marked = 0
    for i, s in enumerate(lines):
        if re.search(r"[\[\]♪♫()]", s[2]):  # non-speech lines such as [Music], (applause): not used, act as a boundary
            marked += 1
            if cur:
                groups.append(cur)
                cur = []
            continue
        if cur and (s[1] - lines[cur[0]][0] > CLIP_MAX or s[0] - lines[cur[-1]][1] > GAP_MAX):
            groups.append(cur)
            cur = []
        cur.append(i)
        dur = s[1] - lines[cur[0]][0]
        if dur >= CLIP_SENTENCE_S and re.search(r"[.?!…][\"'”»)]*$", s[2]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out, why = [], collections.Counter()
    if marked:
        why["non-speech line"] += marked
    for g in groups:
        start, end = lines[g[0]][0], lines[g[-1]][1]
        txt = " ".join(lines[i][2] for i in g)
        dur = end - start
        if dur > CLIP_MAX:
            why["long"] += 1
            continue
        if dur < CLIP_MIN:
            why["short"] += 1
            continue
        rate = len(txt) / dur
        if rate < 3 or rate > 25:
            why["char rate"] += 1
            continue
        prev_end = lines[g[0] - 1][1] if g[0] > 0 else 0.0
        next_start = lines[g[-1] + 1][0] if g[-1] + 1 < len(lines) else None
        out.append((start, end, txt, prev_end, next_start))
    return out, why


# ---------------------------------------------------------------- 5) audio
class Resolver:
    """Hugging Face is asked once per ark; byte-range requests go straight to the CDN address."""

    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def url(self, ark, fresh=False):
        with self.lock:
            e = self.cache.get(ark)
            if e and not fresh and time.time() - e[1] < 1500:
                return e[0]
        raw = BASE + ark
        if raw.startswith("file://"):
            final = raw
        else:
            # resolved with GET (1-byte range): signed CDN URLs may depend on the method
            r = subprocess.run(["curl", "-sS", "-L", "-r", "0-0", "--connect-timeout", "30", "-m", "60", "-o", "/dev/null",
                                "-w", "%{url_effective}", raw], capture_output=True, text=True)
            final = r.stdout.strip() or raw
        with self.lock:
            self.cache[ark] = (final, time.time())
        return final


def get_range(url, start, length, follow=False):
    r = subprocess.run(["curl", "-sS"] + (["-L"] if follow else []) +
                       ["--connect-timeout", "30", "-m", "300", "-r", f"{start}-{start + length - 1}",
                        "-w", "%{http_code}", "-o", "-", url], capture_output=True)
    code = r.stdout[-3:].decode(errors="replace")
    data = r.stdout[:-3]
    return code, data


def decode_audio(data):
    """kaldiio 'AUDIO' record: b'AUDIO' + 1 byte n + n bytes length (little endian) + audio file (FLAC)."""
    import numpy as np
    import soundfile as sf
    if data[:5] == b"AUDIO":
        n = data[5]
        length = int.from_bytes(data[6:6 + n], "little")
        body = data[6 + n:6 + n + length]
        if len(body) != length:
            raise ValueError(f"short data: {len(body)}/{length}")
    elif data[:4] in (b"fLaC", b"RIFF"):
        body = data
    else:
        raise ValueError(f"unknown header {data[:8].hex()}")
    x, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    ch = x.shape[1]
    return x.mean(axis=1).astype(np.float32), sr, ch


def resample(x, sr):
    if sr == TARGET_SR:
        return x
    import soxr
    return soxr.resample(x, sr, TARGET_SR, quality="HQ")


def process(job, resolver):
    """One utterance: download -> decode -> 16 kHz -> clips. Returns (manifest rows, bytes, source_sr, channels, error)."""
    import soundfile as sf
    key, ark, offset, nbytes, video, clips = job
    last_err = None
    for attempt in range(4):
        if attempt < 2:
            code, data = get_range(resolver.url(ark, fresh=attempt > 0), offset, nbytes)
        else:  # CDN URL failed: follow the HF redirect on every request
            code, data = get_range(BASE + ark, offset, nbytes, follow=True)
        if code == "429":
            time.sleep(30 * (attempt + 1))
            last_err = "HTTP 429"
            continue
        if code not in ("206", "200", "000") or len(data) != nbytes:
            last_err = f"HTTP {code}, {len(data)}/{nbytes} bytes"
            time.sleep(2 * (attempt + 1))
            continue
        try:
            x, sr, ch = decode_audio(data)
        except Exception as e:
            last_err = f"decode: {e!r}"[:200]
            break
        y = resample(x, sr)
        d = os.path.join(AUDIO, video)
        os.makedirs(d, exist_ok=True)
        rows = []
        for i, (start, end, txt, prev_end, next_start) in enumerate(clips):
            a = max(start - MARGIN, prev_end, 0.0)
            b = min(end + MARGIN, next_start if next_start is not None else end + MARGIN)
            a, b = int(round(a * TARGET_SR)), int(round(b * TARGET_SR))
            clip = y[a:min(b, len(y))]
            if len(clip) < TARGET_SR * CLIP_MIN:
                continue
            path = os.path.join(d, f"{key}_{i:02d}.flac")
            sf.write(path, clip, TARGET_SR, format="FLAC", subtype="PCM_16")
            rows.append(dict(audio_filepath=path, duration=round(len(clip) / TARGET_SR, 3), text=tag_sentences(txt),
                             target_lang="tr-TR", lang="tr-TR", source="yodas", video=video, utterance=key,
                             raw_text=txt))
        return rows, len(data), sr, ch, None
    return None, 0, None, None, last_err or "unknown"


def main():
    global BASE, Y2, AUDIO, INDEX, SAMPLES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--base-url", default=os.environ.get("AYAZ_ASR_YODAS_URL", BASE_URL),
                    help="espnet/yodas_owsmv4 base URL / temel adres (default: %(default)s)")
    ap.add_argument("--yodas2-url", default=YODAS2_URL, help="YODAS2 tr000 text list URL (default: %(default)s)")
    ap.add_argument("--threads", type=int, default=6, help="download threads / eşzamanlı iş (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; small number for a dry run / kuru koşu için (default: 0)")
    ap.add_argument("--min-intersection", type=int, default=500,
                    help="minimum videos for the list match; smaller only for a dry run (default: %(default)s)")
    a = ap.parse_args()
    work = work_dir(a.work_dir)
    BASE = a.base_url if a.base_url.endswith("/") else a.base_url + "/"
    Y2 = a.yodas2_url if a.yodas2_url.endswith("/") else a.yodas2_url + "/"
    INDEX = os.path.join(work, "yodas", "index")
    SAMPLES = os.path.join(work, "yodas", "samples")
    data_dir = os.path.join(work, "datasets", "yodas")
    AUDIO = os.path.join(data_dir, "audio")
    manifest = os.path.join(data_dir, "manifest_yodas.jsonl")
    done_list = os.path.join(data_dir, "done.txt")
    errors = os.path.join(data_dir, "errors.txt")
    summary_path = os.path.join(work, "runs", f"yodas_download_{stamp()}.json")
    for d in (INDEX, SAMPLES, data_dir, AUDIO, os.path.dirname(summary_path)):
        os.makedirs(d, exist_ok=True)
    summary = dict(step="yodas_download", ok=False)
    tsv = os.path.join(INDEX, "tr_utterances.tsv")
    ctc = os.path.join(INDEX, "tr_text.ctc")
    if not (os.path.exists(tsv) and os.path.exists(ctc)):
        stop("index outputs missing (tr_utterances.tsv, tr_text.ctc) — run yodas_index.py first")

    say(f"=== 1. manual-subtitle list ({time.strftime('%H:%M:%S')})")
    manual = manual_list()

    say(f"\n=== 2. video statistics and selection ({time.strftime('%H:%M:%S')})")
    v = collections.defaultdict(lambda: [0, 0, 0, 0])
    with open(ctc, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            if len(p) < 2:
                continue
            s = v[p[0].rsplit("_", 4)[0]]
            s[0] += 1
            s[1] += bool(re.search(r"[.,?!]", p[1]))
            ws = p[1].split()
            s[2] += len(ws)
            s[3] += sum(1 for w in ws if any(c in TR_LETTERS for c in w))
    owsm = set(v)
    inter = owsm & manual
    punct = {k for k, s in v.items() if s[1] / s[0] >= 0.5}
    turkish = {k for k, s in v.items() if s[3] / max(1, s[2]) >= 0.15}
    summary.update(owsm_videos=len(owsm), manual_videos=len(manual), intersection=len(inter), punctuated=len(punct),
                   intersection_punctuated=len(inter & punct), punctuated_not_in_list=len(punct - manual))
    say(f"  OWSM Turkish videos {len(owsm)} · YODAS2 manually subtitled {len(manual)} · intersection {len(inter)} · "
        f"punctuated (>=50%) {len(punct)} · intersection∩punctuated {len(inter & punct)} · "
        f"punctuated but not in list {len(punct - manual)}")
    if len(inter) >= a.min_intersection:
        sel = inter & punct & turkish
        summary["selection_rule"] = "manual subtitles ∩ punctuated(>=50%) ∩ Turkish letters(>=15%)"
    else:
        say("  ! small intersection — ids may not have the same format; selecting by punctuation")
        sel = {k for k, s in v.items() if s[1] / s[0] >= 0.8} & turkish
        summary["selection_rule"] = "punctuated(>=80%) ∩ Turkish letters(>=15%) (list did not match)"
    summary["selected_videos"] = len(sel)
    say(f"  selection: {summary['selection_rule']} -> {len(sel)} videos")

    raw_jobs = []
    with open(tsv, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for key, ark, offset, nbytes, start, end, video in r:
            if video in sel:
                raw_jobs.append((key, ark, int(offset), int(nbytes), video))
    summary["selected_utterances"] = len(raw_jobs)
    summary["selected_gb"] = round(sum(x[3] for x in raw_jobs) / 2**30, 2)
    say(f"  selected utterances {len(raw_jobs):,} · audio {summary['selected_gb']} GB")

    say(f"\n=== 3. time-stamped text ({time.strftime('%H:%M:%S')})")
    all_keys = set()
    with open(tsv, encoding="utf-8") as f:
        next(f)
        for line in f:
            all_keys.add(line.split("\t", 1)[0])
    timed = timed_text(all_keys)
    jobs, reasons, plan_s = [], collections.Counter(), 0.0
    no_lines = 0
    for key, ark, offset, nbytes, video in raw_jobs:
        body = timed.get(key)
        lines = split_lines(body) if body else []
        if not lines:
            no_lines += 1
            continue
        clips, why = clip_plan(lines)
        reasons.update(why)
        if clips:
            jobs.append((key, ark, offset, nbytes, video, clips))
            plan_s += sum(p[1] - p[0] for p in clips)
    summary["no_timed_line"] = no_lines
    summary["planned_clips"] = sum(len(x[5]) for x in jobs)
    summary["planned_hours"] = round(plan_s / 3600, 1)
    summary["rejected_clips"] = dict(reasons)
    say(f"  plan: {summary['planned_clips']:,} clips from {len(jobs):,} utterances · {summary['planned_hours']} h · "
        f"utterances without timed lines {no_lines} · rejected {dict(reasons)}")
    if not jobs:
        stop("clip plan is empty — inspect the format of the time-stamped text")
    say(f"  timed line example: {timed.get(jobs[0][0], '')[:300]}")
    jobs.sort(key=lambda x: (x[1], x[2]))
    if a.limit:
        jobs = jobs[:a.limit]

    say(f"\n=== 4. sample decoding ({time.strftime('%H:%M:%S')})")
    import soundfile as sf
    resolver = Resolver()
    rnd = random.Random(28092026)
    trial = rnd.sample(jobs, min(12, len(jobs)))
    sr_count, ch_count, ok_n = collections.Counter(), collections.Counter(), 0
    written = []
    for i, job in enumerate(trial):
        rows, _, sr, ch, err = process(job, resolver)
        if err:
            say(f"  [{i}] {job[0]}: {err}")
            continue
        ok_n += 1
        sr_count[sr] += 1
        ch_count[ch] += 1
        if rows and len(written) < 5:
            x, s = sf.read(rows[0]["audio_filepath"], dtype="float32")
            p = os.path.join(SAMPLES, f"clip_{len(written):02d}.wav")
            sf.write(p, x, s, subtype="PCM_16")
            written.append((p, rows[0]["raw_text"], rows[0]["duration"]))
        say(f"  [{i}] {job[0]}: {sr} Hz · {ch} ch · {len(rows)} clips")
    summary["samples_ok"] = ok_n
    summary["source_sample_rates"] = dict(sr_count)
    summary["source_channels"] = dict(ch_count)
    with open(os.path.join(SAMPLES, "README_clips.txt"), "w", encoding="utf-8") as f:
        f.write("YODAS Turkish (OWSM v4 cleaned, manually subtitled videos) — 5 random clips, 16 kHz.\n"
                "Source: espnet/yodas_owsmv4 (CC-BY-3.0). For checking audio quality and subtitle alignment.\n\n")
        for p, txt, dur in written:
            f.write(f"{os.path.basename(p)} · {dur} s\n{txt}\n\n")
    say(f"  ok {ok_n}/{len(trial)} · source rates {dict(sr_count)} · channels {dict(ch_count)} · listening samples {SAMPLES}")
    if ok_n == 0:
        stop("no sample could be decoded")

    say(f"\n=== 5. download ({time.strftime('%H:%M:%S')}) — {len(jobs):,} utterances, {a.threads} threads")
    done = set()
    if os.path.exists(done_list):
        done = {x.strip() for x in open(done_list, encoding="utf-8") if x.strip()}
    left = [x for x in jobs if x[0] not in done]
    say(f"  previously done {len(jobs) - len(left):,} · left {len(left):,}")
    lock = threading.Lock()
    cnt = dict(utt=0, bytes=0, clips=0, sec=0.0, err=0)
    t0 = time.time()
    last = [time.time()]
    total_bytes = sum(x[3] for x in left)
    with open(manifest, "a", encoding="utf-8") as fm, open(done_list, "a", encoding="utf-8") as fd, \
            open(errors, "a", encoding="utf-8") as fe, cf.ThreadPoolExecutor(a.threads) as pool:
        futures = {pool.submit(process, job, resolver): job for job in left}
        for fut in cf.as_completed(futures):
            job = futures[fut]
            try:
                rows, b, sr, ch, err = fut.result()
            except Exception as e:
                rows, b, err = None, 0, repr(e)[:200]
            with lock:
                if err:
                    cnt["err"] += 1
                    fe.write(f"{job[0]}\t{err}\n")
                else:
                    for s in rows:
                        fm.write(json.dumps(s, ensure_ascii=False) + "\n")
                    fm.flush()
                    fd.write(job[0] + "\n")
                    fd.flush()
                    cnt["clips"] += len(rows)
                    cnt["sec"] += sum(s["duration"] for s in rows)
                cnt["utt"] += 1
                cnt["bytes"] += b
                if cnt["utt"] >= 200 and cnt["err"] > 0.2 * cnt["utt"]:
                    say(f"STOP: error share too high ({cnt['err']}/{cnt['utt']}) — last error: {err}")
                    for x in futures:
                        x.cancel()
                    break
                if time.time() - last[0] >= 30:
                    last[0] = time.time()
                    el = time.time() - t0
                    rate = cnt["bytes"] / max(1, el)
                    rest = (total_bytes - cnt["bytes"]) / max(1, rate)
                    say(f"  … {cnt['utt']:,}/{len(left):,} utterances · {cnt['bytes'] / 2**30:.2f}/{total_bytes / 2**30:.1f} GB · "
                        f"{rate / 2**20:.1f} MB/s · {cnt['clips']:,} clips ({cnt['sec'] / 3600:.1f} h) · "
                        f"errors {cnt['err']} · ~{rest / 60:.0f} min left")
    summary.update(downloaded_utterances=cnt["utt"] - cnt["err"], errors=cnt["err"], written_clips=cnt["clips"],
                   written_hours=round(cnt["sec"] / 3600, 2), download_gb=round(cnt["bytes"] / 2**30, 2),
                   minutes=round((time.time() - t0) / 60, 1))
    durs = []
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                try:
                    durs.append(json.loads(line)["duration"])
                except Exception:
                    pass
    if durs:
        durs.sort()
        summary["manifest_clips"] = len(durs)
        summary["manifest_hours"] = round(sum(durs) / 3600, 1)
        summary["clip_median_s"] = durs[len(durs) // 2]
        summary["clip_p90_s"] = durs[int(0.9 * (len(durs) - 1))]
    try:
        r = subprocess.run(["du", "-sk", data_dir], capture_output=True, text=True)
        summary["disk_gb"] = round(int(r.stdout.split()[0]) / 2**20, 1)
    except Exception:
        pass
    summary["ok"] = cnt["err"] <= 0.05 * max(1, cnt["utt"])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    say("\n================ SUMMARY ================")
    for k, val in summary.items():
        say(f"  {k}: {val}")
    say(f"\nOutputs: {summary_path} · {manifest} · listening samples {SAMPLES}/ · errors {errors}")


if __name__ == "__main__":
    main()
