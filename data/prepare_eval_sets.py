#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation sets and the YODAS training pool.

  fleurs_tr : FLEURS tr_tr test split (CC-BY-4.0) -> <work>/datasets/fleurs/tr_tr/test/*.wav +
              <work>/manifests/fleurs_tr_test.jsonl. General Turkish: measures regression on general Turkish.
              Expects test.tsv and audio/test.tar.gz of google/fleurs data/tr_tr in <work>/datasets/fleurs/tr_tr/
              (see README for the two download commands).
  yodas     : YODAS development split — 30 videos split off AT VIDEO LEVEL (clips of the same audio never land on
              both sides): <work>/manifests/yodas_dev.jsonl and the remaining videos <work>/manifests/yodas_pool.jsonl
              (the training subset is drawn from the pool by build_turkish_manifest.py).
Uses no GPU. Prints only counts.

TR: Değerlendirme kümeleri: FLEURS Türkçe sınav bölümü manifesti ve YODAS'ın video düzeyinde ayrılmış geliştirme
kümesi (30 video) ile eğitim havuzu.
"""
import argparse
import collections
import csv
import json
import os
import random
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import stamp, work_dir  # noqa: E402
from ayaz_asr.text import tag_sentences  # noqa: E402

SEED = 29092026
csv.field_size_limit(10 ** 7)


def say(*a):
    print(*a, flush=True)


def fleurs_tr(summary, fl_dir, out_path):
    import soundfile as sf
    tsv = os.path.join(fl_dir, "test.tsv")
    tgz = os.path.join(fl_dir, "test.tar.gz")
    wav_dir = os.path.join(fl_dir, "test")
    os.makedirs(wav_dir, exist_ok=True)
    # 1. archive: only .wav members, by file name only (directory paths inside the archive are not used)
    in_archive = 0
    if os.path.exists(tgz):
        with tarfile.open(tgz, "r:gz") as t:
            for m in t:
                if not (m.isfile() and m.name.endswith(".wav")):
                    continue
                in_archive += 1
                target = os.path.join(wav_dir, os.path.basename(m.name))
                if os.path.exists(target) and os.path.getsize(target) == m.size:
                    continue
                with t.extractfile(m) as src, open(target + ".part", "wb") as h:
                    shutil.copyfileobj(src, h)
                os.replace(target + ".part", target)
        say(f"  [fleurs_tr] {in_archive} WAV extracted -> {wav_dir}")
    # 2. manifest: FLEURS row = id, file, raw text, normalised text, characters, samples, gender
    rows, missing, broken = [], 0, 0
    rates, genders, sentences = collections.Counter(), collections.Counter(), set()
    with open(tsv, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 7:
                broken += 1
                continue
            path = os.path.join(wav_dir, r[1])
            if not os.path.exists(path):
                missing += 1
                continue
            info = sf.info(path)
            rates[f"{info.samplerate} Hz · {info.channels} ch"] += 1
            genders[r[6]] += 1
            sentences.add(r[0])
            rows.append(dict(audio_filepath=path, duration=round(info.frames / info.samplerate, 3), text=tag_sentences(r[2]),
                             target_lang="tr-TR", lang="tr-TR", source="fleurs", fleurs_id=r[0], raw_text=r[2],
                             normalized=r[3], gender=r[6]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    durs = sorted(x["duration"] for x in rows)
    summary.update(fleurs_tr_wav_in_archive=in_archive, fleurs_tr_clips=len(rows), fleurs_tr_sentences=len(sentences),
                   fleurs_tr_missing_files=missing, fleurs_tr_broken_rows=broken, fleurs_tr_hours=round(sum(durs) / 3600, 2),
                   fleurs_tr_median_s=durs[len(durs) // 2] if durs else None, fleurs_tr_max_s=durs[-1] if durs else None,
                   fleurs_tr_gender=dict(genders), fleurs_tr_sampling=dict(rates), manifest_fleurs_tr=out_path)
    say(f"  [fleurs_tr] {len(rows)} clips · {len(sentences)} sentences · {summary['fleurs_tr_hours']} h · missing {missing} · {dict(rates)}")
    return len(rows) > 0 and missing == 0


def yodas(summary, yodas_manifest, dev_path, pool_path, dev_videos):
    if not os.path.exists(yodas_manifest):
        say(f"  [yodas] ! manifest missing: {yodas_manifest}")
        return False
    with open(yodas_manifest, encoding="utf-8") as f:
        m = [json.loads(s) for s in f]
    videos = sorted({d["video"] for d in m})
    dev = set(random.Random(SEED).sample(videos, min(dev_videos, len(videos))))
    D = [d for d in m if d["video"] in dev]
    P = [d for d in m if d["video"] not in dev]
    for path, lst in ((dev_path, D), (pool_path, P)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for x in lst:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    per_video = sorted(collections.Counter(d["video"] for d in D).values())
    summary.update(yodas_dev_videos=len(dev), yodas_dev_clips=len(D), yodas_dev_hours=round(sum(d["duration"] for d in D) / 3600, 2),
                   yodas_dev_clips_per_video_median=per_video[len(per_video) // 2] if per_video else None,
                   yodas_pool_videos=len(videos) - len(dev), yodas_pool_clips=len(P),
                   yodas_pool_hours=round(sum(d["duration"] for d in P) / 3600, 2),
                   manifest_yodas_dev=dev_path, manifest_yodas_pool=pool_path)
    say(f"  [yodas] dev {len(dev)} videos · {len(D)} clips · {summary['yodas_dev_hours']} h | "
        f"pool {summary['yodas_pool_videos']} videos · {len(P)} clips · {summary['yodas_pool_hours']} h")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--yodas-manifest", help="YODAS manifest (default: <work>/datasets/yodas/manifest_yodas.jsonl)")
    ap.add_argument("--dev-videos", type=int, default=30, help="videos in the YODAS dev split / geliştirme videosu (default: %(default)s)")
    ap.add_argument("--only", choices=("fleurs_tr", "yodas"), help="run only one part / yalnız bir bölüm")
    a = ap.parse_args()
    work = work_dir(a.work_dir)
    man = os.path.join(work, "manifests")
    summary = dict(step="prepare_eval_sets", stamp=stamp())
    jobs = []
    if a.only in (None, "fleurs_tr"):
        jobs.append(("fleurs_tr", lambda s: fleurs_tr(s, os.path.join(work, "datasets", "fleurs", "tr_tr"),
                                                      os.path.join(man, "fleurs_tr_test.jsonl"))))
    if a.only in (None, "yodas"):
        jobs.append(("yodas", lambda s: yodas(s, a.yodas_manifest or os.path.join(work, "datasets", "yodas", "manifest_yodas.jsonl"),
                                              os.path.join(man, "yodas_dev.jsonl"), os.path.join(man, "yodas_pool.jsonl"),
                                              a.dev_videos)))
    ok = True
    for name, job in jobs:
        try:
            ok = job(summary) and ok
        except Exception as e:  # if one fails the other still runs; the error goes into the summary
            ok = False
            summary[f"{name}_error"] = repr(e)[:300]
            say(f"STOP {name}: {e!r}"[:300])
    summary["ok"] = ok
    out = os.path.join(work, "runs", f"prepare_eval_sets_{summary['stamp']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    say("\n================ SUMMARY ================")
    for k, v in summary.items():
        say(f"  {k}: {v}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
