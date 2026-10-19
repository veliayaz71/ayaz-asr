#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rehearsal data for the other languages (anti-forgetting) and the interim guard sets.

Why: in an earlier experiment, training on Turkish only made 32 of 34 FLEURS languages regress significantly. Observed
in that experiment: the <tr-TR> tag leaking into other languages, Turkish-style spelling and empty output for distant
languages; since every training line had the tr-TR prompt, the model appeared to learn to ignore the prompt. Rehearsal: clips of the
other languages whose LABEL IS THE BASE MODEL'S OWN OUTPUT (with language tag and punctuation), decoded with their own
prompt. The goal is not to improve the other languages but to keep them as they are (self-distillation). Training on the
true text would teach FLEURS' domain and make the FLEURS check look better than the actual preservation.

Sub-commands:
  download — FLEURS (CC-BY-4.0) train split: 32 languages (all except Turkish and Thai; the base model does not work
             for Thai either, CER ≈ 99 %), 1–10 s recordings (memory envelope on MPS), at most 2 recordings per sentence; the archive
             is streamed and cut once the target count is reached. German and English get --quota-priority, the others
             --quota (or a per-language --quota-lang). Dev split: guard languages × --guard-n recordings, 1–30 s
             (interim guard measurement during training — selecting the step on the test split would bias the verdict).
  label    — the base model (live context 560 ms) decodes every rehearsal recording with its own prompt. First a Turkish
             sanity check: the base model's WER on the 150-clip FLEURS-tr subset (digits as words) must reproduce
             --sanity-expected ±0.002, else STOP. Filter: empty / very short output, a <tr-TR> tag, a label longer (in
             tokens) than the longest Turkish label (--max-label-tokens; computed from --turkish-manifest if not given).
             Quality check: for languages whose base test error is <= 25 %, the label error against the true text may
             exceed the base test error by at most --quality-margin, and empty outputs <= 10 % — otherwise STOP
             (wrong prompt, broken audio or another decoding path). Label error is measured on NON-EMPTY outputs only.
Outputs: <work>/datasets/fleurs_rehearsal/<fl>/train/*.wav · <work>/datasets/fleurs_guard/<fl>/dev/*.wav ·
  <out>/manifest_rehearsal.jsonl · <out>/guard/<fl>.jsonl · <out>/labels/<fl>.json (cache) · <out>/summary.json
Prints only counts and open data (FLEURS). An interrupted run resumes with the same command.

TR: Öteki dillerin provası (etiket = temel modelin kendi çıktısı) ve eğitim sırasındaki ara koruma kümeleri.
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
import shutil
import sys
import tarfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import base_model_path, print_result, read_jsonl, stamp, work_dir, write_jsonl  # noqa: E402
from ayaz_asr import text as T  # noqa: E402
from ayaz_asr.fleurs import (BY_CONFIG, HF_FLEURS, LANGUAGES, OK_STATUS, PRIORITY, CountingReader, curl_get,  # noqa: E402
                             curl_head, curl_stream, fmt_points, hf_fingerprint)
from ayaz_asr.nemo_io import ATT_CONTEXT_560MS, empty_cache, transcribe  # noqa: E402

MIN_S, MAX_S = 1.0, 10.0
PER_SENTENCE = 2
SPARE = 1.15          # extra share downloaded for recordings the filter will drop
GUARD_MAX_S = 30.0    # guard sets are decoded at every measurement during training
TOLERANCE = 0.002
QUALITY_LIMIT, EMPTY_SHARE = 0.25, 0.10
# Download alarm for broken/silent recordings. Single broken recordings are always dropped; this total threshold only
# catches a systematic conversion error. FLEURS itself has silent recordings (e.g. 13 of the first 185 suitable es_419
# train recordings, rms ~8e-06), so the threshold is 20 %.
SKIP_SHARE = 0.20
SUB_SEED, SUB_N = 29092026, 150
CHUNK = 250           # decode in chunks of this size (progress line)
TAG = re.compile(r"<([a-z]{2,3})-[A-Za-z]{2,4}>")

A = None              # parsed arguments
REHEARSAL = []        # [(fleurs config, prompt, metric, main)]
QUOTA_LANG = {}
P = {}                # paths


def say(*a):
    print("  [rehearsal]", *a, flush=True)


def quota(fl):
    """Rehearsal quota of a language: per-language quota, else priority-language quota, else the default quota."""
    return QUOTA_LANG.get(fl, A.quota_priority if fl in PRIORITY else A.quota)


def set_hash(rows):
    return hashlib.sha256(json.dumps([(os.path.basename(x["audio_filepath"]), x.get("raw_text", "")) for x in rows],
                                     ensure_ascii=False).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- 1. download
def check_wav(path, expected_frames):
    """None or a short problem name. FLEURS: 16 kHz, mono, 32-bit float; frame count = num_samples in the TSV."""
    import numpy as np
    import soundfile as sf
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1:
        return f"format {info.samplerate} Hz x {info.channels} ch"
    if expected_frames and abs(info.frames - expected_frames) > 160:
        return f"frames {info.frames} (TSV {expected_frames})"
    audio, _ = sf.read(path, dtype="float32")   # read as float: dtype="int16" does not scale, every sample becomes 0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    if rms < 1e-4:
        return f"silent (rms {rms:.1e})"
    return None


def download_split(fl, split, target_n, root, max_s, per_sentence):
    """Streams the first target_n suitable recordings of a FLEURS split (train/dev); the stream stops at the target."""
    prompt = BY_CONFIG[fl][1]
    target = os.path.join(root, fl)
    wav_dir = os.path.join(target, split)
    done_path, sel_path = os.path.join(target, "done.json"), os.path.join(target, "selection.jsonl")
    if os.path.exists(done_path) and os.path.exists(sel_path):
        with open(done_path, encoding="utf-8") as f:
            t = json.load(f)
        sel = read_jsonl(sel_path)
        if t.get("target_n") == target_n and t.get("split") == split and t.get("max_s") == max_s and sel \
                and all(os.path.exists(x["audio_filepath"]) for x in sel):
            return dict(t, status="present", mb=0.0, sec=0.0)
    os.makedirs(wav_dir, exist_ok=True)
    tsv_url, tar_url = f"{A.hf}/data/{fl}/{split}.tsv", f"{A.hf}/data/{fl}/audio/{split}.tar.gz"
    t0 = time.time()
    code, h, err = curl_head(tsv_url)
    if code not in OK_STATUS:
        raise RuntimeError(f"TSV: server HTTP {code} {err}")
    data = curl_get(tsv_url)
    kind, match = hf_fingerprint(data, h.get("x-linked-etag") or h.get("etag"))
    if match is False:
        raise RuntimeError(f"TSV fingerprint mismatch ({kind})")
    with open(os.path.join(target, f"{split}.tsv"), "wb") as f:
        f.write(data)
    rows, total_rows = {}, 0
    for r in csv.reader(io.StringIO(data.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(r) < 7 or not r[5].isdigit():
            continue
        total_rows += 1
        sec = int(r[5]) / 16000
        if r[2].strip() and MIN_S <= sec <= max_s:
            rows[r[1]] = r
    if not rows:
        raise RuntimeError(f"no suitable recording in the TSV ({total_rows} rows)")
    code2, h2, err2 = curl_head(tar_url)
    if code2 not in OK_STATUS:
        raise RuntimeError(f"archive: server HTTP {code2} {err2}")
    sel, sentences, skipped, problems = [], collections.Counter(), collections.Counter(), []
    p = curl_stream(tar_url)
    cnt = CountingReader(p.stdout)
    try:
        with tarfile.open(fileobj=cnt, mode="r|gz") as t:
            for m in t:
                if not (m.isfile() and m.name.endswith(".wav")):
                    continue
                name = os.path.basename(m.name)
                r = rows.get(name)
                if r is None:
                    continue
                if sentences[r[0]] >= per_sentence:
                    skipped["sentence"] += 1
                    continue
                path = os.path.join(wav_dir, name)
                if not (os.path.exists(path) and os.path.getsize(path) == m.size):
                    with t.extractfile(m) as src, open(path + ".part", "wb") as out:
                        shutil.copyfileobj(src, out)
                    os.replace(path + ".part", path)
                problem = check_wav(path, int(r[5]))
                if problem:
                    skipped["audio"] += 1
                    problems.append(f"{name}: {problem}")
                    os.remove(path)
                    continue
                sentences[r[0]] += 1
                sel.append(dict(audio_filepath=path, duration=round(int(r[5]) / 16000, 3), raw_text=r[2].strip(), fleurs_id=r[0],
                                gender=r[6], prompt=prompt, fleurs=fl, split=split))
                if len(sel) >= target_n:
                    break
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
    if not sel:
        raise RuntimeError(f"no recording from the archive (read {cnt.n} bytes; curl: {p.stderr.read().decode(errors='replace')[:200]})")
    if skipped["audio"] > max(2, SKIP_SHARE * (len(sel) + skipped["audio"])):
        raise RuntimeError(f"too many broken or silent recordings: {skipped['audio']} / {len(sel) + skipped['audio']} "
                           f"(threshold {SKIP_SHARE * 100:.0f}%) — e.g. {problems[:2]}")
    write_jsonl(sel_path, sel)
    d = dict(fleurs=fl, split=split, status="downloaded", n=len(sel), target_n=target_n, max_s=max_s,
             exhausted=len(sel) < target_n, candidates=len(rows), tsv_rows=total_rows,
             distinct_sentences=len(sentences), skipped=dict(skipped), mb=round(cnt.n / 2**20, 1), sec=round(time.time() - t0, 1),
             tsv=f"{kind}{'' if match is None else (' ok' if match else ' MISMATCH')}",
             archive_size=h2.get("x-linked-size"), archive_etag=(h2.get("x-linked-etag") or "").strip('"')[:16],
             hours=round(sum(x["duration"] for x in sel) / 3600, 3))
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def download_job(fl, split, n, root, max_s, per_sentence):
    """One download job with three attempts -> (result or None, error record or None). Jobs may run in parallel."""
    for attempt in (1, 2, 3):
        try:
            d = download_split(fl, split, n, root, max_s, per_sentence)
            extra = " · archive ended before the target" if d.get("exhausted") else ""
            if d["status"] == "present":
                say(f"download · {fl} {split}: already present ({d['n']} recordings){extra}")
            else:
                sk = (d.get("skipped") or {}).get("audio", 0)
                say(f"download · {fl} {split}: {d['n']} recordings · {d['hours'] * 60:.0f} min audio · {d['distinct_sentences']} "
                    f"distinct sentences · {d['mb']} MB read · {d['sec']:.0f} s · TSV {d['tsv']}{extra}"
                    + (f" · broken/silent skipped: {sk}" if sk else ""))
            return d, None
        except Exception as e:
            print(f"download {fl} {split} attempt {attempt}: {e!r}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if attempt == 3:
                print(f"STOP {fl} {split} could not be downloaded: {str(e)[:160]}", flush=True)
                return None, dict(fleurs=fl, split=split, error=repr(e)[:200])
            time.sleep(5)
    return None, dict(fleurs=fl, split=split, error="no attempt")


def cmd_download():
    jobs = [(fl, "dev", A.guard_n, P["guard_data"], GUARD_MAX_S, 1) for fl in A.guard_langs] + \
           [(fl, "train", int(round(quota(fl) * SPARE)), P["rehearsal_data"], MAX_S, PER_SENTENCE) for fl, _, _, _ in REHEARSAL]
    say(f"download: guard {len(A.guard_langs)} languages x {A.guard_n} (dev split) · rehearsal {len(REHEARSAL)} languages "
        f"(train split, {MIN_S:g}–{MAX_S:g} s, priority languages {round(A.quota_priority * SPARE)}, others "
        f"{round(A.quota * SPARE)} recordings target) — streamed, cut at the target · {A.parallel} train jobs in parallel"
        + (" · per-language quota: " + ", ".join(f"{k} {round(v * SPARE)}" for k, v in sorted(QUOTA_LANG.items())) if QUOTA_LANG else ""))
    results, failed = [], []

    def collect(d, h):
        if d is not None:
            results.append(d)
        if h is not None:
            failed.append(h)

    for j in [j for j in jobs if j[1] == "dev"]:
        collect(*download_job(*j))
    train = [j for j in jobs if j[1] != "dev"]
    if A.parallel > 1 and len(train) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=A.parallel) as pool:
            for d, h in pool.map(lambda j: download_job(*j), train):
                collect(d, h)
    else:
        for j in train:
            collect(*download_job(*j))
    critical = [h for h in failed if h["split"] == "dev" or h["fleurs"] in PRIORITY]
    ok = not critical and len([h for h in failed if h["split"] == "train"]) <= 4
    s = dict(step="rehearsal_download", ok=ok, jobs=len(jobs), done=len(results), failed=failed,
             rehearsal_recordings=sum(d["n"] for d in results if d["split"] == "train"),
             guard_recordings=sum(d["n"] for d in results if d["split"] == "dev"), mb=round(sum(d["mb"] for d in results), 1))
    say(f"download finished: {len(results)}/{len(jobs)} jobs · rehearsal {s['rehearsal_recordings']} recordings · guard "
        f"{s['guard_recordings']} recordings · {s['mb']} MB read"
        + (f" · not downloaded: {', '.join(h['fleurs'] + ' ' + h['split'] for h in failed)}" if failed else ""))
    if not ok:
        print("STOP: download insufficient: all guard sets and priority languages are required; at most 4 rehearsal "
              "languages may be missing", flush=True)
    print_result(s, P["summary_log"])
    return 0 if ok else 1


# ---------------------------------------------------------------- 2. labelling
def base_test_error(fl, metric):
    """Base model's error on the FLEURS test split (from the multilingual evaluation cache; None if absent)."""
    sel, hy = os.path.join(A.eval_data_dir, fl, "selection.jsonl"), os.path.join(A.eval_hyp_dir, "base", fl + ".json")
    if not (os.path.exists(sel) and os.path.exists(hy)):
        return None
    rows = read_jsonl(sel)
    with open(hy, encoding="utf-8") as f:
        d = json.load(f)
    hyp = d.get("hyp") or []
    if len(hyp) != len(rows):
        return None
    return T.error_rate([x.get("raw_text", "") for x in rows], hyp, metric)


def load_model():
    import torch
    import nemo.collections.asr as nemo_asr
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.restore_from(restore_path=A.base_model, map_location="cpu")
    model.to(device)
    model.eval()
    model.encoder.set_default_att_context_size(ATT_CONTEXT_560MS)
    prompts = dict(model.cfg.get("model_defaults", {}).get("prompt_dictionary", {}) or {})
    return model, device, prompts


def decode(model, device, rows, prompt, fl):
    hyp = []
    for i in range(0, len(rows), CHUNK):
        hyp += transcribe(model, [x["audio_filepath"] for x in rows[i:i + CHUNK]], prompt, P["tmp"], batch_size=8)
        empty_cache(device)
        if len(rows) > CHUNK:
            print(f"    {fl}: {min(len(rows), i + CHUNK)}/{len(rows)}", flush=True)
    return hyp


def cmd_label():
    t_start = time.time()
    os.makedirs(P["tmp"], exist_ok=True)
    rehearsal = {}
    for fl, _, _, _ in REHEARSAL:
        y = os.path.join(P["rehearsal_data"], fl, "selection.jsonl")
        if os.path.exists(y):
            rehearsal[fl] = read_jsonl(y)
    missing = [fl for fl, _, _, _ in REHEARSAL if not rehearsal.get(fl)]
    if any(fl in missing for fl in PRIORITY) or len(missing) > 4:
        print(f"STOP: rehearsal recordings missing ({', '.join(missing)}) — run download first", flush=True)
        return 2
    guard = {}
    for fl in A.guard_langs:
        y = os.path.join(P["guard_data"], fl, "selection.jsonl")
        if not os.path.exists(y):
            print(f"STOP: guard set missing ({fl}) — run download first", flush=True)
            return 2
        guard[fl] = read_jsonl(y)
    if not os.path.exists(A.fleurs_tr_test):
        print(f"STOP: FLEURS-tr test manifest missing ({A.fleurs_tr_test}) — Turkish sanity set", flush=True)
        return 2
    tr_hours = None
    if A.max_label_tokens is None or A.tr_manifest:
        if not A.tr_manifest or not os.path.exists(A.tr_manifest):
            print("STOP: --turkish-manifest is required to compute the label token ceiling (or give --max-label-tokens)", flush=True)
            return 2

    say(f"loading base model ({os.path.basename(A.base_model)})")
    model, device, prompts = load_model()
    tok = model.tokenizer
    if A.tr_manifest and os.path.exists(A.tr_manifest):
        tr = read_jsonl(A.tr_manifest)
        tr_hours = sum(float(x["duration"]) for x in tr) / 3600
        u_tr = sorted(len(tok.text_to_ids(x["text"])) for x in tr if x.get("text"))
        say(f"Turkish manifest: {len(tr)} rows · {tr_hours:.2f} h · label tokens median {u_tr[len(u_tr) // 2]}, "
            f"p99 {u_tr[int(0.99 * (len(u_tr) - 1))]}, max {u_tr[-1]}")
        u_max = u_tr[-1]
    if A.max_label_tokens is not None:
        if A.tr_manifest and u_max != A.max_label_tokens:
            say(f"note: --max-label-tokens {A.max_label_tokens} overrides the ceiling computed from the Turkish manifest ({u_max})")
        u_max = A.max_label_tokens
    say(f"label token ceiling for rehearsal labels: {u_max}")

    # Turkish sanity check: same model and decoding path as in the multilingual evaluation and at training step 0?
    tr_test = read_jsonl(A.fleurs_tr_test)
    sub = random.Random(SUB_SEED).sample(tr_test, min(SUB_N, len(tr_test)))
    hyp_tr = transcribe(model, [x["audio_filepath"] for x in sub], "tr-TR", P["tmp"], batch_size=8)
    empty_cache(device)
    value = T.tr_error(sub, hyp_tr, True)
    if abs(value - A.sanity_expected) > TOLERANCE + 1e-12:
        print(f"STOP: sanity check failed: base · FLEURS-tr subset (digits as words) {value} — expected {A.sanity_expected} "
              f"(±{TOLERANCE}); model or decoding path differs. No labelling done.", flush=True)
        print_result(dict(step="rehearsal_label", ok=False, stop="sanity", sanity=value), P["summary_log"])
        return 2
    print(f"OK sanity: base · FLEURS-tr subset (digits as words) {value} — expected {A.sanity_expected}", flush=True)

    rows_out, lang_summary, quality_fail, quality_dropped = [], [], [], []
    os.makedirs(P["labels"], exist_ok=True)
    for fl, prompt, metric, main in REHEARSAL:
        lst = rehearsal.get(fl)
        if not lst:
            continue
        if prompt not in prompts:
            print(f"! {fl}: prompt '{prompt}' not in the model's dictionary — skipped", flush=True)
            continue
        h16 = set_hash(lst)
        cache = os.path.join(P["labels"], fl + ".json")
        hyp, sec, new = None, 0.0, False
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("set") == h16 and len(d.get("hyp", [])) == len(lst):
                hyp, sec = d["hyp"], d.get("sec", 0.0)
        previous = []
        if hyp is None and A.previous_labels:   # the head of the list matches an earlier run: its labels are reused
            oy = os.path.join(A.previous_labels, fl + ".json")
            if os.path.exists(oy):
                with open(oy, encoding="utf-8") as f:
                    od = json.load(f)
                oh = od.get("hyp") or []
                if 0 < len(oh) <= len(lst) and od.get("set") == set_hash(lst[:len(oh)]):
                    previous = list(oh)
        if hyp is None:
            t1 = time.time()
            try:
                rest = lst[len(previous):]
                hyp = previous + (decode(model, device, rest, prompt, fl) if rest else [])
            except Exception as e:
                if fl in PRIORITY:
                    raise
                traceback.print_exc()
                print(f"! {fl}: decoding error — language dropped from rehearsal ({str(e)[:120]})", flush=True)
                continue
            sec, new = time.time() - t1, True
            with open(cache + ".part", "w", encoding="utf-8") as f:
                json.dump(dict(fleurs=fl, set=h16, n=len(hyp), hyp=hyp, sec=round(sec, 1), previous=len(previous)), f, ensure_ascii=False)
            os.replace(cache + ".part", cache)
        q = quota(fl)
        cnt, chosen, toks, nonempty = collections.Counter(), [], [], []
        for x, h in zip(lst, hyp):
            h = re.sub(r"\s+", " ", h or "").strip()
            if len(T.units(h, metric)) < (4 if metric == "cer" else 2):
                cnt["empty"] += 1
                continue
            nonempty.append((x["raw_text"], h))   # label quality is measured on these (empty outputs never enter training)
            if "<tr-TR>" in h:
                cnt["tr_tag"] += 1
                continue
            u = len(tok.text_to_ids(h))
            if u > u_max:
                cnt["long"] += 1
                continue
            if len(chosen) < q:
                chosen.append(dict(audio_filepath=x["audio_filepath"], duration=x["duration"], text=h, target_lang=prompt,
                                   lang=prompt, source="rehearsal", fleurs=fl, fleurs_id=x.get("fleurs_id"),
                                   raw_text=x["raw_text"], tokens=u))
                toks.append(u)
        le = T.error_rate([r for r, _ in nonempty], [h for _, h in nonempty], metric)
        le_all = T.error_rate([x["raw_text"] for x in lst], hyp, metric)   # empties counted as 100 % error; info only
        bt = base_test_error(fl, metric)
        empty_share = cnt["empty"] / max(1, len(lst))
        problem = []
        if bt is not None and bt <= QUALITY_LIMIT:
            if le > bt + A.quality_margin:
                problem.append(f"label error {fmt_points(le)} > base test {fmt_points(bt)} + {fmt_points(A.quality_margin)}")
            if empty_share > EMPTY_SHARE:
                problem.append(f"empty output share {fmt_points(empty_share)} > {fmt_points(EMPTY_SHARE)}")
        dropped = bool(problem) and A.quality_drop and fl not in PRIORITY
        if problem and not dropped:
            quality_fail.append(f"{fl}: " + "; ".join(problem))
        if dropped:
            quality_dropped.append(f"{fl}: " + "; ".join(problem))
            chosen, toks = [], []
        hrs = sum(x["duration"] for x in chosen) / 3600
        o = dict(fleurs=fl, prompt=prompt, metric=metric, main=main, decoded=len(lst), used=len(chosen), quota=q,
                 hours=round(hrs, 3), label_error=round(le, 4), label_error_all=round(le_all, 4),
                 base_test=None if bt is None else round(bt, 4),
                 empty=cnt["empty"], tr_tag=cnt["tr_tag"], long=cnt["long"], sec=round(sec, 1), previous=len(previous),
                 max_tokens=max(toks) if toks else None)
        lang_summary.append(o)
        rows_out += chosen
        say(f"label · {fl}: {len(lst)} decoded · used {len(chosen)}/{q} ({hrs * 60:.0f} min) · label error "
            f"{metric.upper()} {fmt_points(le)}" + (f" (with empties {fmt_points(le_all)})" if cnt["empty"] else "")
            + (f" (base test {fmt_points(bt)})" if bt is not None else " (no base test value)")
            + f" · empty {cnt['empty']} · <tr-TR> {cnt['tr_tag']} · long {cnt['long']}"
            + (f" · {sec:.0f} s" if new else " · cache") + (f" (previous run {len(previous)})" if previous else "")
            + (" · ! " + "; ".join(problem) if problem else "") + (" — language dropped from rehearsal" if dropped else ""))

    if len(quality_dropped) > 4:   # at most 4 faulty languages; more means a systematic problem
        quality_fail += quality_dropped
    total_h = sum(x["duration"] for x in rows_out) / 3600
    share = total_h / max(1e-9, total_h + (tr_hours or 0.0))
    s = dict(step="rehearsal_label", ok=False, sanity=value, max_label_tokens=u_max, turkish_hours=None if tr_hours is None else round(tr_hours, 2),
             languages=sum(1 for o in lang_summary if o["used"]), rows=len(rows_out), hours=round(total_h, 2), share=round(share, 3),
             per_language=lang_summary, quality_fail=quality_fail, minutes=round((time.time() - t_start) / 60, 1),
             out_dir=A.out_dir, quality_margin=A.quality_margin, quality_dropped=quality_dropped)
    if quality_fail:
        print(f"STOP: label quality check failed ({len(quality_fail)} languages): {' | '.join(quality_fail)[:400]}", flush=True)
        print_result(dict(s, stop="quality"), P["summary_log"])
        return 2
    if not (A.hours_min <= total_h <= A.hours_max):
        print(f"STOP: rehearsal duration out of range: {total_h:.2f} h (expected {A.hours_min:g}–{A.hours_max:g} h)", flush=True)
        print_result(dict(s, stop="hours"), P["summary_log"])
        return 2
    write_jsonl(P["manifest"], rows_out)
    for fl in A.guard_langs:
        write_jsonl(os.path.join(P["guard"], fl + ".jsonl"),
                    [dict(audio_filepath=x["audio_filepath"], duration=x["duration"], raw_text=x["raw_text"], prompt=BY_CONFIG[fl][1],
                          fleurs=fl, metric=BY_CONFIG[fl][2], fleurs_id=x.get("fleurs_id")) for x in guard[fl]])
    s["ok"] = True
    s["manifest"] = P["manifest"]
    with open(P["summary"], "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    say(f"rehearsal: {s['languages']} languages · {len(rows_out)} rows · {total_h:.2f} h"
        + (f" · Turkish manifest {tr_hours:.2f} h -> rehearsal share {share * 100:.1f}% (unique rows)" if tr_hours else "")
        + f" · guard: {', '.join(f'{fl} {len(guard[fl])}' for fl in A.guard_langs)} · {s['minutes']} min")
    print_result(s, P["summary_log"])
    return 0


def main():
    global A, REHEARSAL, P
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("download", "label"), help="download | label")
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--out-dir", help="rehearsal output directory (default: <work>/rehearsal)")
    ap.add_argument("--base-model", help="base .nemo (default: $AYAZ_ASR_BASE_NEMO or <work>/models/...)")
    ap.add_argument("--hf", default=HF_FLEURS, help="google/fleurs base URL (default: %(default)s)")
    ap.add_argument("--quota", type=int, default=300, help="rehearsal recordings per language (default, v0.1: %(default)s)")
    ap.add_argument("--quota-priority", type=int, default=1300, help="rehearsal recordings for de_de and en_us (default, v0.1: %(default)s)")
    ap.add_argument("--quota-lang", default="lv_lv:600,lt_lt:600,et_ee:600,ro_ro:600",
                    help="per-language quota / dile özel kota (default, v0.1: %(default)s)")
    ap.add_argument("--languages", default="", help="comma list: only these rehearsal languages (dry runs) / dil alt kümesi")
    ap.add_argument("--guard-langs", default="de_de,en_us,fr_fr,ru_ru,ja_jp,ro_ro,et_ee,bg_bg",
                    help="guard languages, FLEURS dev split / ara koruma dilleri (default, v0.1: %(default)s)")
    ap.add_argument("--guard-n", type=int, default=100, help="guard recordings per language (default: %(default)s)")
    ap.add_argument("--parallel", type=int, default=3, help="parallel train-split downloads (default: %(default)s)")
    ap.add_argument("--turkish-manifest", dest="tr_manifest", help="Turkish training manifest: label token ceiling + share report (label)")
    ap.add_argument("--max-label-tokens", type=int, help="label token ceiling; overrides the value computed from --turkish-manifest (v0.1: 127)")
    ap.add_argument("--fleurs-tr-test", help="FLEURS-tr test manifest (default: <work>/manifests/fleurs_tr_test.jsonl)")
    ap.add_argument("--sanity-expected", type=float, default=0.1204,
                    help="base model WER on the 150-clip FLEURS-tr subset, digits as words (default: %(default)s)")
    ap.add_argument("--eval-data-dir", help="multilingual eval recordings, for the base test error (default: <work>/datasets/fleurs_eval)")
    ap.add_argument("--eval-hyp-dir", help="multilingual eval hypothesis cache (default: <work>/eval/hyp)")
    ap.add_argument("--quality-margin", type=float, default=0.12, help="max label error above base test error (default, v0.1: %(default)s)")
    ap.add_argument("--no-quality-drop", dest="quality_drop", action="store_false",
                    help="stop on a failing non-priority language instead of dropping it (v0.1: drop; > 4 dropped still stops)")
    ap.add_argument("--previous-labels", default="", help="label cache of an earlier run whose list head is reused")
    ap.add_argument("--hours-min", type=float, default=15.0, help="min rehearsal hours (default, v0.1: %(default)s)")
    ap.add_argument("--hours-max", type=float, default=32.0, help="max rehearsal hours (default, v0.1: %(default)s)")
    A = ap.parse_args()
    work = work_dir(A.work_dir)
    A.out_dir = os.path.abspath(A.out_dir or os.path.join(work, "rehearsal"))
    A.base_model = base_model_path(work, A.base_model)
    A.hf = A.hf.rstrip("/")
    A.guard_langs = tuple(x for x in A.guard_langs.split(",") if x)
    A.fleurs_tr_test = A.fleurs_tr_test or os.path.join(work, "manifests", "fleurs_tr_test.jsonl")
    A.eval_data_dir = A.eval_data_dir or os.path.join(work, "datasets", "fleurs_eval")
    A.eval_hyp_dir = A.eval_hyp_dir or os.path.join(work, "eval", "hyp")
    chosen = [x for x in A.languages.split(",") if x]
    REHEARSAL = [d for d in LANGUAGES if d[0] not in ("tr_tr", "th_th") and (not chosen or d[0] in chosen)]
    REHEARSAL.sort(key=lambda d: (d[0] not in PRIORITY, LANGUAGES.index(d)))   # priority languages first
    for part in [x.strip() for x in A.quota_lang.split(",") if x.strip()]:
        fl, _, n = part.partition(":")
        if not fl or not n.isdigit() or int(n) <= 0:
            sys.exit(f"STOP: --quota-lang malformed: {part!r} (expected lang:count, e.g. lv_lv:600)")
        QUOTA_LANG[fl] = int(n)
    outside = sorted(set(QUOTA_LANG) - {d[0] for d in REHEARSAL})
    if outside:
        sys.exit(f"STOP: --quota-lang names languages outside rehearsal: {', '.join(outside)}")
    P = dict(rehearsal_data=os.path.join(work, "datasets", "fleurs_rehearsal"), guard_data=os.path.join(work, "datasets", "fleurs_guard"),
             labels=os.path.join(A.out_dir, "labels"), guard=os.path.join(A.out_dir, "guard"), tmp=os.path.join(A.out_dir, "tmp"),
             manifest=os.path.join(A.out_dir, "manifest_rehearsal.jsonl"), summary=os.path.join(A.out_dir, "summary.json"),
             summary_log=os.path.join(work, "runs", f"rehearsal_{stamp()}.jsonl"))
    try:
        sys.exit(cmd_download() if A.command == "download" else cmd_label())
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"STOP {A.command}: unexpected error — {e!r}"[:300], flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
