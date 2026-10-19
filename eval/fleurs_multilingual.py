#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multilingual FLEURS regression check (34 FLEURS languages incl. Turkish).

Question: how far did the Turkish fine-tuning move the other languages of the base model?
Sub-commands: download | decode | summary
  download — FLEURS (CC-BY-4.0) test split per language: the first --n recordings (archive head streamed), or with
             --full the WHOLE test split (archive read to the end; curl exit code, archive SHA-256 against HF's
             fingerprint — a mismatch only flags the language — and >= 99 % of the TSV rows present are checked).
             Turkish is not downloaded: its set is the 150-clip FLEURS-tr subset from prepare_eval_sets.py.
  decode   — every model is loaded from disk and FIRST decodes the Turkish subset; the value must reproduce the
             model's own training-log value within ±0.002, otherwise STOP (wrong file, wrong loading or another decoding
             path). Hypotheses are cached per model and language (an interrupted run resumes).
  summary  — error per language and model; difference to the base with a 95 % bootstrap interval (utterance level);
             '!' = interval above zero (significant regression). With --verdict MODEL the acceptance verdict:
             de and en <= +1.0 point · median over the 29 main languages (excluding Turkish) <= +2.0 · no language
             except Thai > +5.0 (the base model does not work for Thai either, CER ≈ 99 %).
Decoding uses the training code's own helpers (560 ms context); Turkish is scored with the training metric.

Models: the base model is always first (expected Turkish subset value --base-expected / --base-expected-raw);
add fine-tuned models with --model "name|label|path.nemo|digits_as_words(1/0)|expected_subset_value".

TR: Çok dilli FLEURS gerileme ölçümü ve kabul hükmü.
"""
import argparse
import csv
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
from ayaz_asr import base_model_path, print_result, read_jsonl, stamp, work_dir  # noqa: E402
from ayaz_asr import text as T  # noqa: E402
from ayaz_asr.fleurs import (HF_FLEURS, LANGUAGES, OK_STATUS, CountingReader, curl_get, curl_head, curl_stream,  # noqa: E402
                             fmt_points, fmt_signed, hf_fingerprint)
from ayaz_asr.nemo_io import ATT_CONTEXT_560MS, transcribe  # noqa: E402

SUB_SEED, SUB_N = 29092026, 150   # same as the training script: random.Random(29092026).sample(fleurs_tr, 150)
VERDICT_PRIORITY, VERDICT_MEDIAN, VERDICT_MAX = 0.010, 0.020, 0.050   # points / 100
PRIORITY_EVAL = ("de_de", "en_us", "tr_tr")

A = None
MODELS = []   # (name, label, path, [(digits_as_words, expected subset value)])
P = {}


def say(*a):
    print("  [multilingual]", *a, flush=True)


def languages():
    chosen = [x for x in A.languages.split(",") if x]
    return [d for d in LANGUAGES if not chosen or d[0] in chosen]


# ---------------------------------------------------------------- 1. download
def download_lang(fl, prompt):
    import soundfile as sf
    target = os.path.join(P["data"], fl)
    wav_dir = os.path.join(target, "test")
    done_path, sel_path = os.path.join(target, "done.json"), os.path.join(target, "selection.jsonl")
    if os.path.exists(done_path) and os.path.exists(sel_path):
        with open(done_path, encoding="utf-8") as f:
            t = json.load(f)
        sel = read_jsonl(sel_path)
        if t.get("n_target") == A.n and sel and all(os.path.exists(x["audio_filepath"]) for x in sel):
            return dict(fleurs=fl, status="present", n=len(sel), mb=0.0, sec=0.0, tsv=t.get("tsv"))
    os.makedirs(wav_dir, exist_ok=True)
    tsv_url, tar_url = f"{A.hf}/data/{fl}/test.tsv", f"{A.hf}/data/{fl}/audio/test.tar.gz"
    t0 = time.time()
    code, h, err = curl_head(tsv_url)
    if code not in OK_STATUS:
        raise RuntimeError(f"TSV: server HTTP {code} {err}")
    data = curl_get(tsv_url)
    kind, match = hf_fingerprint(data, h.get("x-linked-etag") or h.get("etag"))
    if match is False:
        raise RuntimeError(f"TSV fingerprint mismatch ({kind})")
    with open(os.path.join(target, "test.tsv"), "wb") as f:
        f.write(data)
    rows = {}
    for r in csv.reader(io.StringIO(data.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(r) >= 7:
            rows[r[1]] = r
    if not rows:
        raise RuntimeError("TSV empty or format not recognised")
    code2, h2, err2 = curl_head(tar_url)
    if code2 not in OK_STATUS:
        raise RuntimeError(f"archive: server HTTP {code2} {err2}")
    sel, archive_sha = [], None
    p = curl_stream(tar_url)
    cnt = CountingReader(p.stdout)
    try:
        with tarfile.open(fileobj=cnt, mode="r|gz") as t:
            for m in t:
                if not (m.isfile() and m.name.endswith(".wav")):
                    continue
                name = os.path.basename(m.name)
                row = rows.get(name)
                if row is None:
                    continue
                path = os.path.join(wav_dir, name)
                if not (os.path.exists(path) and os.path.getsize(path) == m.size):
                    with t.extractfile(m) as src, open(path + ".part", "wb") as out:
                        shutil.copyfileobj(src, out)
                    os.replace(path + ".part", path)
                info = sf.info(path)
                sel.append(dict(audio_filepath=path, duration=round(info.frames / info.samplerate, 3), raw_text=row[2],
                                normalized=row[3], fleurs_id=row[0], gender=row[6], prompt=prompt, fleurs=fl,
                                sample_rate=info.samplerate, channels=info.channels))
                if not A.full and len(sel) >= A.n:
                    break
        if A.full:   # read the rest of the stream; check curl exit, archive SHA-256 and TSV coverage
            while cnt.read(1 << 20):
                pass
            p.wait(timeout=120)
            if p.returncode != 0:
                raise RuntimeError(f"archive stream incomplete (curl exit {p.returncode}; read {cnt.n} bytes)")
            expected = (h2.get("x-linked-etag") or "").strip()
            if expected.startswith("W/"):
                expected = expected[2:]
            expected = expected.strip('"')
            # A SHA mismatch does NOT drop the language, it flags it: the structure is checked separately (tar read to
            # the end, curl exited 0, every WAV readable, >= 99 % of the TSV in the archive). Flagged languages are
            # listed in the summary and should be re-downloaded and checked.
            if re.fullmatch(r"[0-9a-f]{64}", expected):
                archive_sha = "sha256 ok" if cnt.h.hexdigest() == expected else "sha256 MISMATCH"
            else:
                archive_sha = "not checkable"
            found = len({os.path.basename(x["audio_filepath"]) for x in sel})
            if found < 0.99 * len(rows):
                raise RuntimeError(f"only {found} of the TSV's {len(rows)} recordings found in the archive")
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
    if not sel:
        raise RuntimeError(f"no recording from the archive (read {cnt.n} bytes; curl: {p.stderr.read().decode(errors='replace')[:200]})")
    with open(sel_path, "w", encoding="utf-8") as f:
        for x in sel:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    d = dict(fleurs=fl, status="downloaded", n=len(sel), n_target=A.n, mb=round(cnt.n / 2**20, 1), sec=round(time.time() - t0, 1),
             tsv=f"{kind}{'' if match is None else (' ok' if match else ' MISMATCH')}",
             archive_size=h2.get("x-linked-size"), archive_etag=(h2.get("x-linked-etag") or "").strip('"')[:16],
             hours=round(sum(x["duration"] for x in sel) / 3600, 3), full=A.full, tsv_rows=len(rows), archive_sha=archive_sha)
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def download_job(fl, prompt):
    for attempt in (1, 2, 3):
        try:
            d = download_lang(fl, prompt)
            if d["status"] == "present":
                say(f"download · {fl}: already present ({d['n']} recordings)")
            else:
                say(f"download · {fl}: {d['n']} recordings · {d['hours'] * 60:.0f} min audio · {d['mb']} MB read · {d['sec']:.0f} s · "
                    f"TSV {d['tsv']}" + (f" · TSV rows {d['tsv_rows']} · archive {d['archive_sha']}" if d.get("full") else "")
                    + (" !" if "MISMATCH" in str(d.get("archive_sha")) else ""))
            return d, None
        except Exception as e:
            print(f"download {fl} attempt {attempt}: {e!r}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if attempt == 3:
                print(f"STOP {fl} could not be downloaded: {str(e)[:160]}", flush=True)
                return None, dict(fleurs=fl, error=repr(e)[:200])
            time.sleep(5)
    return None, dict(fleurs=fl, error="no attempt")


def cmd_download():
    targets = [d for d in languages() if d[0] != "tr_tr"]
    if A.full:
        say(f"download: {len(targets)} languages, the WHOLE test split of each (streamed to the end; SHA-256 checked; "
            f"{A.parallel} languages at once) -> {P['data']}")
    else:
        say(f"download: {len(targets)} languages x {A.n} recordings (FLEURS test split, archive head streamed) -> {P['data']}")
    results, failed = [], []
    if A.parallel > 1 and len(targets) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=A.parallel) as pool:
            for d, h in pool.map(lambda x: download_job(x[0], x[1]), targets):
                if d is not None:
                    results.append(d)
                if h is not None:
                    failed.append(h)
    else:
        for fl, prompt, _, _ in targets:
            d, h = download_job(fl, prompt)
            if d is not None:
                results.append(d)
            if h is not None:
                failed.append(h)
    s = dict(step="multilingual_download", ok=len(failed) <= len(targets) // 2, languages=len(targets), done=len(results),
             failed=failed, recordings=sum(d["n"] for d in results), mb=round(sum(d["mb"] for d in results), 1), full=A.full,
             set=A.set_name)
    s["sha_mismatch"] = [d["fleurs"] for d in results if "MISMATCH" in str(d.get("archive_sha"))]
    if s["sha_mismatch"]:
        print(f"! archive SHA mismatch: {', '.join(s['sha_mismatch'])} — structure sound (tar read to the end, TSV almost "
              f"fully in the archive); results are used, re-download and check these languages separately", flush=True)
    say(f"download finished: {len(results)}/{len(targets)} languages · {s['recordings']} recordings · {s['mb']} MB read"
        + (f" · not downloaded: {', '.join(h['fleurs'] for h in failed)}" if failed else ""))
    print_result(s, P["summary"])
    return 0 if s["ok"] else 1


# ---------------------------------------------------------------- sets and cache
def tr_set():
    tr_test = read_jsonl(A.fleurs_tr_test)
    sub = random.Random(SUB_SEED).sample(tr_test, min(SUB_N, len(tr_test)))
    return [dict(audio_filepath=x["audio_filepath"], duration=x.get("duration"), raw_text=x.get("raw_text", ""),
                 fleurs_id=x.get("fleurs_id"), prompt="tr-TR", fleurs="tr_tr") for x in sub]


def eval_set(fl):
    if fl == "tr_tr":
        return tr_set() if os.path.exists(A.fleurs_tr_test) else []
    y = os.path.join(P["data"], fl, "selection.jsonl")
    return read_jsonl(y) if os.path.exists(y) else []


def set_hash(rows):
    import hashlib
    return hashlib.sha256(json.dumps([(os.path.basename(x["audio_filepath"]), x.get("raw_text", "")) for x in rows],
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def cache_path(model, fl):
    return os.path.join(P["hyp"], model, fl + ".json")


def cache_read(model, fl, h16):
    y = cache_path(model, fl)
    if os.path.exists(y):
        with open(y, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("set") == h16 and len(d.get("hyp", [])) == d.get("n"):
            return d
    return None


def cache_write(model, fl, h16, hyp, sec):
    y = cache_path(model, fl)
    os.makedirs(os.path.dirname(y), exist_ok=True)
    with open(y + ".part", "w", encoding="utf-8") as f:
        json.dump(dict(model=model, fleurs=fl, set=h16, n=len(hyp), hyp=hyp, sec=round(sec, 1)), f, ensure_ascii=False)
    os.replace(y + ".part", y)


# ---------------------------------------------------------------- metric
def units(text, metric, prompt, digits_as_words=True):
    """Turkish: the training metric. Other languages: see ayaz_asr.text.units."""
    if prompt == "tr-TR":
        old = T.NUMBERS_TO_WORDS
        T.NUMBERS_TO_WORDS = digits_as_words
        try:
            return T.normalize(text)
        finally:
            T.NUMBERS_TO_WORDS = old
    return T.units(text, metric)


# ---------------------------------------------------------------- 2. decoding
def load(path):
    import torch
    import nemo.collections.asr as nemo_asr
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.restore_from(restore_path=path, map_location="cpu")
    model.to(device)
    model.eval()
    model.encoder.set_default_att_context_size(ATT_CONTEXT_560MS)
    prompts = dict(model.cfg.get("model_defaults", {}).get("prompt_dictionary", {}) or {})
    return dict(model=model, device=device, prompts=prompts, path=path)


def release(m):
    import gc
    import torch
    m.pop("model", None)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def decode_list(m, rows, prompt):
    hyp = transcribe(m["model"], [x["audio_filepath"] for x in rows], prompt, P["tmp"], batch_size=8)
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()   # do not let the MPS cache grow across 34 languages
    return hyp


def cmd_decode():
    os.makedirs(P["tmp"], exist_ok=True)
    target_langs = languages()
    sets = {fl: eval_set(fl) for fl, _, _, _ in target_langs}
    missing = [fl for fl, lst in sets.items() if not lst]
    if missing:
        print(f"! languages without recordings are skipped: {', '.join(missing)}", flush=True)
    if not sets.get("tr_tr"):
        print("STOP: Turkish sanity set (FLEURS-tr test manifest) missing — no sanity check possible", flush=True)
        return 2
    s = dict(step="multilingual_decode", ok=False, models={}, full=A.full, set=A.set_name)
    t_start = time.time()
    for mi, (name, label, path, checks) in enumerate(MODELS):
        if not os.path.exists(path):
            print(f"STOP: model missing: {path}", flush=True)
            return 2
        order = ["tr_tr"] + [fl for fl, _, _, _ in target_langs if fl != "tr_tr" and sets.get(fl)]
        prompt_of = {fl: prompt for fl, prompt, _, _ in target_langs}
        pending = [fl for fl in order if cache_read(name, fl, set_hash(sets[fl])) is None]
        if not pending:
            say(f"{label}: all languages decoded before (cache)")
        m, t0 = None, time.time()
        try:
            for fl in order:
                lst, prompt = sets[fl], prompt_of.get(fl, "tr-TR")
                h16 = set_hash(lst)
                d = cache_read(name, fl, h16)
                if d is None:
                    if m is None:
                        say(f"{label}: loading model ({os.path.basename(path)}, {os.path.getsize(path) / 2**30:.2f} GB)")
                        m = load(path)
                    if prompt not in m["prompts"]:
                        print(f"! {label} · {fl}: prompt '{prompt}' not in the model's dictionary — skipped", flush=True)
                        continue
                    t1 = time.time()
                    try:
                        hyp = decode_list(m, lst, prompt)
                    except Exception as e:
                        if fl == "tr_tr":   # without the sanity set the measurement is meaningless: stop
                            raise
                        print(f"{label} · {fl}: {e!r}", file=sys.stderr, flush=True)
                        traceback.print_exc()
                        print(f"! {label} · {fl}: decoding error — language skipped ({str(e)[:120]})", flush=True)
                        s.setdefault("failed", []).append(f"{name}:{fl}")
                        continue
                    cache_write(name, fl, h16, hyp, time.time() - t1)
                    d = dict(hyp=hyp, sec=time.time() - t1)
                    new = True
                else:
                    new = False
                hyp = d["hyp"]
                if fl == "tr_tr":
                    for digits, expected in checks:
                        value = T.tr_error(lst, hyp, digits)
                        ok = abs(value - expected) <= A.tolerance + 1e-12
                        s["models"].setdefault(name, {})[f"sanity_{'digits_as_words' if digits else 'raw'}"] = value
                        mode = "digits as words" if digits else "raw"
                        if not ok:
                            print(f"STOP: sanity check failed: {label} · Turkish subset ({mode}) {value} — expected {expected} "
                                  f"(±{A.tolerance}). Model or decoding path differs from training; measurement stopped.", flush=True)
                            print_result(dict(s, ok=False, stop=f"sanity {name}"), P["summary"])
                            return 2
                        print(f"OK sanity: {label} · Turkish subset ({mode}) {value} — expected {expected}", flush=True)
                else:
                    metric = next(x[2] for x in LANGUAGES if x[0] == fl)
                    e = n = 0
                    for x, h in zip(lst, hyp):
                        r = units(x.get("raw_text", ""), metric, prompt)
                        if r:
                            e += T.edits(r, units(h, metric, prompt))
                            n += len(r)
                    empty = sum(1 for h in hyp if not units(h, metric, prompt))
                    say(f"{label} · {fl}: {metric.upper()} {fmt_points(e / max(1, n))} · empty {empty}/{len(lst)}"
                        + (f" · {d['sec']:.0f} s" if new else " · cache"))
            el = time.time() - t0
            left = len(MODELS) - mi - 1
            say(f"{label} finished ({el / 60:.1f} min)" + (f" · {left} models left, ~{el * left / 60:.0f} min" if left and pending else ""))
        finally:
            if m is not None:
                release(m)
    s.update(ok=True, minutes=round((time.time() - t_start) / 60, 1))
    print_result(s, P["summary"])
    return 0


# ---------------------------------------------------------------- 3. summary
def cmd_summary():
    rng = random.Random(1)
    names = [x[0] for x in MODELS]
    label_of = {x[0]: x[1] for x in MODELS}
    rows_out, detail, missing = [], [], []
    for fl, prompt, metric, main in languages():
        lst = eval_set(fl)
        if not lst:
            missing.append(fl)
            continue
        h16 = set_hash(lst)
        hyp = {nm: (cache_read(nm, fl, h16) or {}).get("hyp") for nm in names}
        if hyp.get("base") is None:
            missing.append(fl)
            continue
        refs = [units(x.get("raw_text", ""), metric, prompt) for x in lst]
        valid = [i for i, r in enumerate(refs) if r]
        lens = [len(refs[i]) for i in valid]
        n_units = sum(lens)
        o = dict(fleurs=fl, prompt=prompt, metric=metric, main=main, priority=fl in PRIORITY_EVAL, n=len(valid), units=n_units,
                 hours=round(sum(float(x.get("duration") or 0) for x in lst) / 3600, 3))
        errs = {}
        for nm in names:
            if hyp.get(nm) is None:
                continue
            hb = [units(h, metric, prompt) for h in hyp[nm]]
            errs[nm] = [T.edits(refs[i], hb[i]) for i in valid]
            o[nm] = round(sum(errs[nm]) / max(1, n_units), 4)
            o[f"empty_{nm}"] = sum(1 for h in hb if not h)
            if prompt != "tr-TR":
                own = prompt.split("-")[0]
                o[f"tag_tr_{nm}"] = sum(1 for h in hyp[nm] if "<tr-TR>" in (h or ""))
                o[f"foreign_tag_{nm}"] = sum(1 for h in hyp[nm] if any(m.group(1) != own for m in T.LANG_TAG.finditer(h or "")))
                o[f"tr_letters_{nm}"] = sum(1 for h in hyp[nm] if re.search("[ığĞ]", T.LANG_TAG.sub(" ", h or "")))
        eb = errs["base"]
        k = len(valid)
        for nm in names[1:]:
            if nm not in errs or not k:
                continue
            em = errs[nm]
            diff = (sum(em) - sum(eb)) / max(1, n_units)
            samples = []
            for _ in range(A.bootstrap):
                idx = [rng.randrange(k) for _ in range(k)]
                nn = sum(lens[j] for j in idx)
                samples.append((sum(em[j] for j in idx) - sum(eb[j] for j in idx)) / max(1, nn))
            samples.sort()
            lo, hi = samples[int(0.025 * A.bootstrap)], samples[min(A.bootstrap - 1, int(0.975 * A.bootstrap))]
            o[f"diff_{nm}"], o[f"lo_{nm}"], o[f"hi_{nm}"] = round(diff, 4), round(lo, 4), round(hi, 4)
            o[f"call_{nm}"] = "regression" if lo > 0 else ("improvement" if hi < 0 else "no difference")
        rows_out.append(o)
        for j, i in enumerate(valid):
            detail.append([fl, os.path.basename(lst[i]["audio_filepath"]), lst[i].get("raw_text", ""), lens[j]]
                          + [x for nm in names for x in ((hyp[nm][i], errs[nm][j]) if nm in errs else ("", ""))])
    fields = ["fleurs", "prompt", "metric", "main", "priority", "n", "units", "hours"] + names \
        + [f"{p}_{nm}" for nm in names[1:] for p in ("diff", "lo", "hi", "call")] \
        + [f"empty_{nm}" for nm in names] + [f"{p}_{nm}" for nm in names for p in ("tag_tr", "foreign_tag", "tr_letters")]
    with open(P["csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for o in rows_out:
            w.writerow(o)
    with open(P["csv_utt"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fleurs", "file", "ref", "ref_units"] + [c for nm in names for c in (f"hyp_{nm}", f"err_{nm}")])
        w.writerows(detail)
    ordered = sorted(rows_out, key=lambda o: (not o["priority"], not o["main"], o["fleurs"]))
    say("summary — error in percentage points; difference to the base with 95 % bootstrap interval (utterance level); "
        "'!' = interval above zero (significant regression)")
    for o in ordered:
        part = [f"{o['fleurs']} ({o['metric'].upper()}, {o['n']} recordings): base {fmt_points(o['base'])}"]
        for nm in names[1:]:
            if f"diff_{nm}" in o:
                mark = " !" if o[f"call_{nm}"] == "regression" else (" v" if o[f"call_{nm}"] == "improvement" else "")
                part.append(f"{label_of[nm]} {fmt_signed(o[f'diff_{nm}'])} [{fmt_signed(o[f'lo_{nm}'])}, {fmt_signed(o[f'hi_{nm}'])}]{mark}")
        leak = [f"{label_of[nm]} <tr-TR> {o[f'tag_tr_{nm}']}" for nm in names[1:] if o.get(f"tag_tr_{nm}")]
        say(" · ".join(part) + (f" · tag leak: {', '.join(leak)}" if leak else ""))
    s = dict(step="multilingual_summary", ok=bool(rows_out), languages=len(rows_out), missing=missing, full=A.full, set=A.set_name)
    for nm in names[1:]:
        reg = [o["fleurs"] for o in rows_out if o.get(f"call_{nm}") == "regression"]
        imp = [o["fleurs"] for o in rows_out if o.get(f"call_{nm}") == "improvement"]
        diffs = [o[f"diff_{nm}"] for o in rows_out if f"diff_{nm}" in o and o["fleurs"] != "tr_tr"]
        s[nm] = dict(regression=reg, improvement=imp, median_diff_non_tr=round(T.median(diffs), 4) if diffs else None,
                     priority={o["fleurs"]: o.get(f"diff_{nm}") for o in rows_out if o["priority"]})
        say(f"RESULT {label_of[nm]}: significant regression in {len(reg)} languages" + (f" ({', '.join(reg)})" if reg else "")
            + f" · significant improvement in {len(imp)}" + (f" ({', '.join(imp)})" if imp else "")
            + (f" · median difference outside Turkish {fmt_signed(s[nm]['median_diff_non_tr'])} points" if diffs else ""))
    if A.verdict and A.verdict in names[1:]:
        nm = A.verdict
        prio = {fl: next((o.get(f"diff_{nm}") for o in rows_out if o["fleurs"] == fl), None) for fl in ("de_de", "en_us")}
        main_d = [o[f"diff_{nm}"] for o in rows_out if o["main"] and o["fleurs"] != "tr_tr" and f"diff_{nm}" in o]
        all_d = [(o["fleurs"], o[f"diff_{nm}"]) for o in rows_out if o["fleurs"] not in ("tr_tr", "th_th") and f"diff_{nm}" in o]
        k_prio = all(v is not None and v <= VERDICT_PRIORITY + 1e-9 for v in prio.values())
        med = T.median(main_d) if main_d else None
        k_med = med is not None and med <= VERDICT_MEDIAN + 1e-9
        worst = max(all_d, key=lambda t: t[1]) if all_d else None
        k_max = worst is not None and worst[1] <= VERDICT_MAX + 1e-9
        passed = bool(k_prio and k_med and k_max and len(main_d) >= 25)
        s["verdict"] = dict(model=nm, priority=prio, median_main=None if med is None else round(med, 4), main_languages=len(main_d),
                            worst=worst, crit_priority=k_prio, crit_median=k_med, crit_max=k_max, passed=passed,
                            thresholds=dict(priority=VERDICT_PRIORITY, median=VERDICT_MEDIAN, max=VERDICT_MAX))
        if A.verdict_info:
            s["verdict"]["info_only"] = True
        im = lambda b: "OK" if b else "FAIL"
        say(("INFO (not an official verdict) — " if A.verdict_info else "") + f"VERDICT {label_of[nm]}: de/en (<= +1.0 point) de "
            + (fmt_signed(prio['de_de']) if prio['de_de'] is not None else '-') + " · en "
            + (fmt_signed(prio['en_us']) if prio['en_us'] is not None else '-') + f" {im(k_prio)} · median over {len(main_d)} main "
            "languages " + (fmt_signed(med) if med is not None else '-') + f" (<= +2.0) {im(k_med)} · worst language "
            + (f"{worst[0]} {fmt_signed(worst[1])}" if worst else '-') + f" (<= +5.0; Thai excluded) {im(k_max)} -> "
            + ("PASSED" if passed else "NOT PASSED"))
    if missing:
        say(f"languages not measured: {', '.join(missing)}")
    with open(P["summary"], "a", encoding="utf-8") as f:
        for o in ordered:
            f.write(json.dumps(dict(step="multilingual_language", **o), ensure_ascii=False) + "\n")
    say(f"outputs: {P['summary']} (summary) · {os.path.basename(P['csv'])} (language x model) · "
        f"{os.path.basename(P['csv_utt'])} (per utterance; FLEURS open data)")
    print_result(s, P["summary"])
    return 0 if s["ok"] else 1


def main():
    global A, MODELS, P
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("download", "decode", "summary"), help="download | decode | summary")
    ap.add_argument("--work-dir", help="working directory / çalışma dizini (default: $AYAZ_ASR_WORK or ./work)")
    ap.add_argument("--full", action="store_true", help="whole FLEURS test split per language (v0.1 verdict) / sınav bölümünün tamamı")
    ap.add_argument("--n", type=int, default=100, help="recordings per language without --full (default: %(default)s)")
    ap.add_argument("--set-name", default="", help="name of the recording/cache set (default: 'full' with --full, else none)")
    ap.add_argument("--parallel", type=int, default=1, help="languages downloaded at once (v0.1 used 3) (default: %(default)s)")
    ap.add_argument("--languages", default="", help="comma list of FLEURS configs to restrict to / dil alt kümesi")
    ap.add_argument("--hf", default=HF_FLEURS, help="google/fleurs base URL (default: %(default)s)")
    ap.add_argument("--fleurs-tr-test", help="FLEURS-tr test manifest (default: <work>/manifests/fleurs_tr_test.jsonl)")
    ap.add_argument("--base-model", help="base .nemo (default: $AYAZ_ASR_BASE_NEMO or <work>/models/...)")
    ap.add_argument("--base-expected", type=float, default=0.1204, help="base Turkish subset value, digits as words (default: %(default)s)")
    ap.add_argument("--base-expected-raw", type=float, default=0.1338, help="base Turkish subset value, raw digits (default: %(default)s)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME|LABEL|PATH|1|VALUE",
                    help="fine-tuned model to compare; VALUE = its FLEURS-tr subset value from the training log at the selected step")
    ap.add_argument("--verdict", default="", help="model NAME that receives the acceptance verdict / hüküm verilecek model")
    ap.add_argument("--verdict-info", action="store_true", help="mark the verdict as information only")
    ap.add_argument("--tolerance", type=float, default=0.002, help="sanity tolerance (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (default: %(default)s)")
    A = ap.parse_args()
    work = work_dir(A.work_dir)
    if A.full:
        A.n = 0
        A.set_name = A.set_name or "full"
    if A.set_name and not re.fullmatch(r"[a-z0-9]+", A.set_name):
        sys.exit(f"STOP: --set-name may only contain lower-case letters and digits: {A.set_name!r}")
    A.hf = A.hf.rstrip("/")
    A.fleurs_tr_test = A.fleurs_tr_test or os.path.join(work, "manifests", "fleurs_tr_test.jsonl")
    suffix = f"_{A.set_name}" if A.set_name else ""
    run = stamp()
    P = dict(data=os.path.join(work, "datasets", "fleurs_eval" + suffix), hyp=os.path.join(work, "eval", "hyp" + suffix),
             tmp=os.path.join(work, "eval", "tmp"), summary=os.path.join(work, "eval", f"multilingual_{run}.jsonl"),
             csv=os.path.join(work, "eval", f"multilingual_{run}.csv"), csv_utt=os.path.join(work, "eval", f"multilingual_{run}_utterances.csv"))
    os.makedirs(os.path.join(work, "eval"), exist_ok=True)
    MODELS = [("base", "base", base_model_path(work, A.base_model), [(True, A.base_expected), (False, A.base_expected_raw)])]
    for spec in A.model:
        p = spec.split("|")
        if len(p) != 5 or not p[0] or any(p[0] == x[0] for x in MODELS) or not re.fullmatch(r"[A-Za-z0-9_.-]+", p[0]):
            sys.exit(f"STOP: --model malformed or name already used: {spec!r}")
        MODELS.append((p[0], p[1], os.path.abspath(p[2]), [(p[3] == "1", float(p[4]))]))
    try:
        sys.exit(dict(download=cmd_download, decode=cmd_decode, summary=cmd_summary)[A.command]())
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:  # model not loadable, sanity set not decodable … — trace on stderr, short line on stdout
        traceback.print_exc()
        print(f"STOP {A.command}: unexpected error — {e!r}"[:300], flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
