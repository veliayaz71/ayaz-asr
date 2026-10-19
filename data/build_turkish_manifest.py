#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turkish part of the training mixture.

Public part (selection may differ slightly between runs, see Known gaps in README): from the YODAS pool, per video
at most --per-video random clips (0.3–10 s), seed 30092026, videos in sorted order.

Optional in-house parts (NOT distributed; v0.1 used the developer's own consented recordings here — see README):
  --speech  transcribed Turkish speech, repeated --speech-copies times (v0.1: 5)
  --empty   clips whose correct output is EMPTY (silence, noise, echo, clicks), --empty-copies copies
            (v0.1: 16); the first copy is the original, the others get a random gain (-9…+3 dB) and, if longer than
            1 s, a random crop (60–100 %), written to --variation-dir (seeded -> stable manifest)
  --noise   room-noise clips (empty label); the number of copies is chosen so that all empty examples together last
            --empty-target-hours (v0.1 target: 1.92 h; at least 1, at most 32 copies), same variation scheme
Then digits in labels are spelled out as Turkish words (v0.1: on) and everything is shuffled with the same RNG.

Row order before shuffling: yodas + speech * copies + empty rows + noise rows (as in v0.1).

TR: Eğitim karışımının Türkçe kısmı. YODAS kısmı herkese açık veriden kurulur (seçim koşudan koşuya biraz değişebilir,
README'deki Bilinen eksikler'e bakın); isteğe bağlı dağıtılmayan veri (rızalı konuşma / boş örnek / oda sesi) eklenebilir.
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
from ayaz_asr.text import digits_to_words  # noqa: E402

SEED = 30092026


def say(*a):
    print("  [turkish]", *a, flush=True)


def vary(rows, copies, prefix, seed, out_dir, enabled):
    """First copy original, the others with random gain (-9…+3 dB) and, if longer than 1 s, random crop (60–100 %)."""
    out = list(rows)
    if not (enabled and copies > 1 and rows):
        return rows * max(1, copies)
    import numpy as np
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    rc = random.Random(seed)
    for k in range(1, copies):
        for i, x in enumerate(rows):
            path = os.path.join(out_dir, f"{prefix}{i:03d}_{k:02d}.wav")
            gain_db = rc.uniform(-9.0, 3.0)
            audio, sr = sf.read(x["audio_filepath"], dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio[:, 0]
            n = len(audio)
            if n / sr > 1.0:
                m = int(n * rc.uniform(0.6, 1.0))
                start = rc.randint(0, n - m)
                audio = audio[start:start + m]
            audio = np.clip(audio * (10 ** (gain_db / 20.0)), -1.0, 1.0)
            if not os.path.exists(path):
                sf.write(path, audio, sr, subtype="PCM_16")
            out.append(dict(x, audio_filepath=path, duration=round(len(audio) / sr, 3), source=x.get("source", "empty") + "_aug"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--yodas-pool", help="YODAS pool manifest (default: <work>/manifests/yodas_pool.jsonl)")
    ap.add_argument("--out", help="output manifest / çıktı (default: <work>/manifests/turkish.jsonl)")
    ap.add_argument("--per-video", type=int, default=4, help="max clips per video / video başına en çok parça (default: %(default)s)")
    ap.add_argument("--max-duration", type=float, default=10.0, help="max clip duration in s (default: %(default)s)")
    ap.add_argument("--speech", help="OPTIONAL in-house transcribed Turkish speech manifest (not distributed) / dağıtılmayan veri: konuşma")
    ap.add_argument("--speech-copies", type=int, default=5, help="copies of --speech rows (default: %(default)s)")
    ap.add_argument("--empty", help="OPTIONAL in-house empty-label clips manifest (not distributed) / boş etiketli örnekler")
    ap.add_argument("--empty-copies", type=int, default=16, help="copies of --empty rows (default: %(default)s)")
    ap.add_argument("--noise", help="OPTIONAL in-house room-noise manifest (not distributed) / oda sesi")
    ap.add_argument("--noise-copies", type=int, default=0, help="copies of --noise rows; 0 = from --empty-target-hours (default: 0)")
    ap.add_argument("--empty-target-hours", type=float, default=1.92, help="total hours of empty examples (default: %(default)s)")
    ap.add_argument("--no-variation", action="store_true", help="plain copies instead of gain/crop variations")
    ap.add_argument("--variation-dir", help="where varied copies are written (default: <work>/datasets/empty_variations)")
    ap.add_argument("--keep-digits", action="store_true", help="do NOT spell out digits as Turkish words (v0.1 spelled them out)")
    ap.add_argument("--expected-sha", default="", help="stop if the written manifest's SHA-256 differs")
    a = ap.parse_args()
    work = work_dir(a.work_dir)
    pool_path = a.yodas_pool or os.path.join(work, "manifests", "yodas_pool.jsonl")
    out_path = a.out or os.path.join(work, "manifests", "turkish.jsonl")
    var_dir = a.variation_dir or os.path.join(work, "datasets", "empty_variations")

    speech = read_jsonl(a.speech) if a.speech else []
    for x in speech:
        if x.get("target_lang") != "tr-TR" or not x.get("text"):
            sys.exit(f"STOP: --speech row is not tr-TR with text: {os.path.basename(x.get('audio_filepath', '?'))}")
    empty = [dict(x, text="", target_lang="tr-TR", lang="tr-TR") for x in (read_jsonl(a.empty) if a.empty else [])
             if 0.3 <= float(x["duration"]) <= 20.0]
    noise = [dict(x, text="", target_lang="tr-TR", lang="tr-TR", source="noise") for x in (read_jsonl(a.noise) if a.noise else [])
             if 0.3 <= float(x["duration"]) <= 20.0]

    rnd = random.Random(SEED)
    videos = collections.defaultdict(list)
    for x in read_jsonl(pool_path):
        if 0.3 <= x["duration"] <= a.max_duration:
            videos[x["video"]].append(x)
    yodas = []
    for v in sorted(videos):
        yodas.extend(rnd.sample(videos[v], min(a.per_video, len(videos[v]))))

    empty_rows = vary(empty, a.empty_copies, "c", 30092026, var_dir, not a.no_variation)
    noise_copies = 0
    if noise:
        noise_s = sum(x["duration"] for x in noise)
        empty_s = sum(x["duration"] for x in empty_rows)
        noise_copies = a.noise_copies or max(1, min(32, round((a.empty_target_hours * 3600 - empty_s) / noise_s)))
        empty_rows = empty_rows + vary(noise, noise_copies, "o", 30092027, var_dir, not a.no_variation)
    rows = yodas + speech * a.speech_copies + empty_rows
    if not a.keep_digits:
        rows = [dict(x, text=digits_to_words(x["text"])) if x.get("text") else x for x in rows]
    rnd.shuffle(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    dur = lambda lst: sum(float(x["duration"]) for x in lst)
    total = dur(rows) or 1.0
    say(f"YODAS {len(yodas)} clips ({dur(yodas) / 3600:.2f} h, {len(videos)} videos) · in-house speech {len(speech)} × "
        f"{a.speech_copies if speech else 0} · empty {len(empty)} → {len(empty_rows) - (len(noise) * noise_copies if noise else 0)} rows"
        + (f" + noise {len(noise)} × {noise_copies}" if noise else "") + f" · total {len(rows)} rows, {total / 3600:.2f} h")
    say(f"duration shares: YODAS {dur(yodas) / total:.1%} · in-house speech {dur(speech) * a.speech_copies / total:.1%} · "
        f"empty {dur(empty_rows) / total:.1%}")
    say(f"written {out_path} · SHA-256 {sha}")
    if a.expected_sha and sha != a.expected_sha:
        sys.exit(f"STOP: manifest SHA-256 {sha[:16]}… != expected {a.expected_sha[:16]}…")


if __name__ == "__main__":
    main()
