#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tuning of nvidia/nemotron-3.5-asr-streaming-0.6b for Turkish (Ayaz ASR v0.1 recipe), on Apple MPS.

Sub-commands:
  env  — environment check (torch / NeMo / lightning versions, MPS allocation test, memory)
  run  — training with periodic measurement, step selection, resume, L2-SP anchor and the in-training verdict

What `run` does (defaults = v0.1, see configs/v0.1.yaml):
  * trainable: the last --layers (8) encoder layers + prediction network + joint network + language-prompt module
    (prompt kernel); the rest of the encoder (including its input subsampling) is frozen and kept in eval mode. Only the
    encoder's parameters are frozen, then its last layers unfrozen (v0.1 training log: 230,370,336 trainable).
    AdamW lr 3e-5, betas (0.9, 0.98), weight decay 1e-3; fp32; batches of 30 s audio (lhotse, 6 buckets, quadratic
    duration 15); RNNT loss computed on the CPU (numba), log-softmax of the joint off.
  * L2-SP anchor (discrete form): every --anchor-interval (10) steps every trained weight is pulled towards its original
    value by k = 1 - (1 - 1/tau)^interval:  theta <- theta + k (theta0 - theta)  — the decoupled weight-decay form of AdamW
    aimed at theta0, lambda = 1/(lr*tau) (tau 4500 -> lambda 7.41). The anchor is kept in fp16 ON THE CPU and moved to
    MPS weight by weight only when applied. Every measurement logs the relative drift ||theta - theta0|| / ||theta0||.
  * measurement every --eval-every (1000) steps on: FLEURS-tr test subset (150 clips; full set at step 0 and for the
    selected step), YODAS dev (report), guard sets (FLEURS dev of 8 languages), and — if given — in-house test sets.
  * step selection: among steps meeting the interim criteria, the lowest in-domain relative error; if none meets them,
    the lowest balance score (in-domain relative + FLEURS-tr subset relative + guard relative mean).
  * after the last step the selected step's weights are measured again with the full FLEURS-tr set and saved as .nemo.
An interrupted run resumes from the last measurement with the same command (same data and settings).

In-house test sets (not distributed; v0.1 used the developer's own consented recordings) — all optional:
  --indomain-test      Turkish everyday speech (rows with raw_text); verdict: >= 10 % relative improvement
  --indomain-aux-test  second in-domain set, only averaged into the selection score
  --short-test         short utterances; verdict: <= 2 empty outputs (padded 1.0 s / 1.2 s with --short-test-padding)
  --echo-test          clips that must stay empty (echo-empty check); verdict: empty share may drop by <= 5 points
  --noise-test         room noise (empty); report only unless --noise-decides
  --extra-test         any further Turkish set; report only
Without in-house sets the YODAS dev set can be passed as --indomain-test (public substitute; not what v0.1 did).

TR: Nemotron 3.5 ASR'nin Türkçe ince ayarı (Ayaz ASR v0.1 tarifi). Varsayılanlar v0.1'in değerleridir.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ayaz_asr import base_model_path, read_jsonl, stamp, work_dir  # noqa: E402
from ayaz_asr import text as T  # noqa: E402
from ayaz_asr.nemo_io import (ATT_CONTEXT_560MS, PAD_END_S, PAD_START_S, install_cpu_rnnt_loss, load_model, memory,  # noqa: E402
                              transcribe, train_data_config, write_padded)

SUB_SEED, SUB_N = 29092026, 150      # FLEURS-tr subset used at interim measurements
RESUME_SHUFFLE_SEED = 30092026
SHORT_MAX_EMPTY = 2                  # verdict: at most this many empty outputs on the short-utterance set
EMPTY_DROP_POINTS = 5.0              # verdict: empty share on echo/noise sets may drop by at most this many points
INDOMAIN_MIN_REL = 0.10              # verdict: in-domain relative improvement
FLEURS_TR_MAX_POINTS = 0.5           # verdict: full FLEURS-tr may rise by at most this many points


def say(*a):
    print(*a, flush=True)


def self_sha():
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "?"


# ---------------------------------------------------------------- env
def cmd_env(a):
    import platform
    import torch
    r = dict(step="env", python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
             mps_available=bool(torch.backends.mps.is_available()), mps_built=bool(torch.backends.mps.is_built()))
    for name in ("nemo", "lightning", "numba", "lhotse", "omegaconf", "sentencepiece", "psutil"):
        try:
            m = __import__(name)
            r[name] = getattr(m, "__version__", "?")
        except Exception as e:
            r[name] = "MISSING: " + repr(e)[:120]
    if r["mps_available"]:
        try:
            x = torch.randn(256, 256, device="mps")
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                y = x @ x
            torch.mps.synchronize()
            r["mps_bf16_autocast"] = str(y.dtype)
        except Exception as e:
            r["mps_bf16_autocast"] = "ERROR: " + repr(e)[:200]
        # PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 alone breaks the MPS allocator ("invalid low watermark ratio 1.4": the
        # low ratio defaults to 1.4 and must not exceed the high one) — a plain allocation is tried and must succeed.
        try:
            z = torch.ones(1024, device="mps")
            torch.mps.synchronize()
            r["mps_allocation"] = "OK" if float(z.sum().item()) == 1024.0 else "ERROR: sum mismatch"
        except Exception as e:
            r["mps_allocation"] = "ERROR: " + repr(e)[:200]
        try:
            r["mps_recommended_gb"] = round(torch.mps.recommended_max_memory() / 2**30, 2)
        except Exception as e:
            r["mps_recommended_gb"] = "?: " + repr(e)[:80]
    r["mps_watermarks"] = {k: os.environ.get(k) for k in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")}
    import psutil
    r["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    r["ram_free_gb"] = round(psutil.virtual_memory().available / 2**30, 1)
    for k, v in r.items():
        say(f"  {k}: {v}")
    r["ok"] = not (r["mps_available"] and str(r.get("mps_allocation", "")).startswith("ERROR"))
    if not r["ok"]:
        say(f"STOP: MPS allocation failed: {r['mps_allocation']} · watermark settings {r['mps_watermarks']}")
    return 0 if r["ok"] else 1


# ---------------------------------------------------------------- run
def load_optional(path, what):
    if not path:
        return []
    if not os.path.exists(path):
        raise RuntimeError(f"{what} missing: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"{what} empty: {path}")
    return rows


def cmd_run(a):
    import torch
    import lightning.pytorch as pl
    from omegaconf import OmegaConf
    torch.manual_seed(0)
    random.seed(0)
    T.NUMBERS_TO_WORDS = not a.keep_digits
    run_id = stamp()
    os.makedirs(a.out_dir, exist_ok=True)
    tmp_dir = os.path.join(a.out_dir, "tmp")
    s = dict(step="train", ok=False, run=run_id)

    rows = read_jsonl(a.train_manifest)
    dur = lambda lst: sum(float(x["duration"]) for x in lst)
    total_h = dur(rows) / 3600
    by_lang = {}
    for x in rows:
        by_lang[x["target_lang"]] = by_lang.get(x["target_lang"], 0) + 1
    s.update(train_rows=len(rows), train_hours=round(total_h, 2), train_tr_rows=by_lang.get("tr-TR", 0),
             train_empty_rows=sum(1 for x in rows if not x.get("text")), train_languages=len(by_lang))
    say(f"  [train] data: {len(rows)} rows · {total_h:.2f} h · Turkish {s['train_tr_rows']} rows (empty label "
        f"{s['train_empty_rows']}) · {len(by_lang)} languages")
    del rows

    install_cpu_rnnt_loss()
    model, _ = load_model(a.base_model)
    # auto (default) = MPS if available, else CPU. Only MPS was used for v0.1; cpu and cuda are untested.
    device = a.device if a.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    if not a.joint_log_softmax:
        # NeMo's joint applies log_softmax outside CUDA and keeps its output on MPS for backprop; numba's CPU loss applies
        # the same log_softmax itself (idempotent). Switching the joint's off does not change the loss and removes one
        # copy of the largest tensor from memory.
        model.joint.log_softmax = False
    quadratic = a.quadratic_duration or None
    n_open = a.layers
    lr = a.lr
    s.update(device=device, open_layers=n_open, lr=lr, batch_s=a.batch_seconds, quadratic_duration=quadratic, buckets=a.buckets,
             max_steps=a.steps, eval_every=a.eval_every, joint_log_softmax=model.joint.log_softmax)

    def freeze():
        for p in model.encoder.parameters():
            p.requires_grad = False
        for layer in list(model.encoder.layers)[-n_open:]:
            for p in layer.parameters():
                p.requires_grad = True

    freeze()
    # For the base model this must be 230,370,336 (v0.1 training log): last 8 encoder layers 201,515,008 + prediction
    # network 14,940,160 + joint 9,455,648 + prompt kernel 4,459,520.
    s["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # anchor = original weights (trained ones only), fp16, on the CPU. Taken BEFORE the resume file is loaded, so a
    # resumed run is still anchored to the original model.
    anchor, k_anchor = {}, 0.0
    if a.anchor_tau > 0:
        if a.anchor_tau < 1 or a.anchor_interval < 1:
            raise RuntimeError(f"invalid anchor setting: tau {a.anchor_tau:g}, interval {a.anchor_interval}")
        anchor = {n: p.detach().to("cpu", torch.float16).clone() for n, p in model.named_parameters() if p.requires_grad}
        k_anchor = 1.0 - (1.0 - 1.0 / a.anchor_tau) ** a.anchor_interval
        s.update(anchor_tau=a.anchor_tau, anchor_interval=a.anchor_interval, anchor_k=round(k_anchor, 9),
                 anchor_lambda=round(1.0 / (lr * a.anchor_tau), 4), anchor_storage="fp16, cpu",
                 anchor_mb=round(sum(t.numel() * 2 for t in anchor.values()) / 2**20))
    with open(a.train_manifest, "rb") as f:
        s["mixture_sha"] = hashlib.sha256(f.read()).hexdigest()
    if a.expected_mixture_sha and s["mixture_sha"] != a.expected_mixture_sha:
        raise RuntimeError(f"mixture differs from the expected one ({s['mixture_sha'][:16]}… != {a.expected_mixture_sha[:16]}…) — "
                           "training not started")

    fleurs_tr = read_jsonl(a.fleurs_tr_test) if os.path.exists(a.fleurs_tr_test) else []
    fleurs_sub = random.Random(SUB_SEED).sample(fleurs_tr, min(SUB_N, len(fleurs_tr))) if fleurs_tr else []
    ydev = read_jsonl(a.yodas_dev) if a.yodas_dev and os.path.exists(a.yodas_dev) else []
    indomain = load_optional(a.indomain_test, "in-domain test")
    aux = load_optional(a.indomain_aux_test, "auxiliary in-domain test")
    short = load_optional(a.short_test, "short-utterance test")
    echo = load_optional(a.echo_test, "echo test")
    noise = load_optional(a.noise_test, "noise test")
    extra = load_optional(a.extra_test, "extra test")
    empty_sets = bool(short) and bool(echo) and bool(noise)
    # guard sets (FLEURS dev split) — if one is missing the run is meaningless: stop
    guard = {}
    for fl in a.guard_langs:
        y = os.path.join(a.guard_dir, fl + ".jsonl")
        if not os.path.exists(y):
            raise RuntimeError(f"guard set missing: {y}")
        guard[fl] = read_jsonl(y)
    guard_on = bool(guard)
    short_padded = []
    if a.short_test_padding:
        if not short:
            raise RuntimeError("padded short-utterance test requested but --short-test is missing")
        os.makedirs(a.pad_dir, exist_ok=True)
        short_padded = [write_padded(x["audio_filepath"], os.path.join(a.pad_dir, f"short_{i:03d}.wav")) for i, x in enumerate(short)]
    has_text = lambda x: x["has_text"] if "has_text" in x else bool((x.get("raw_text") or "").strip())
    peak = dict(mps=0.0)
    say(f"  [train] plan: {a.steps} steps · measurement every {a.eval_every} steps · batch {a.batch_seconds:g} s · "
        f"last {n_open} encoder layers · lr {lr:g} · buckets {a.buckets} · trainable {s['trainable_params']:,} params · tests: "
        f"FLEURS-tr {len(fleurs_tr)} (interim {len(fleurs_sub)}) · YODAS-dev {len(ydev)} · in-domain {len(indomain)} · "
        f"in-domain aux {len(aux)}"
        + (f" · short {len(short)} (with text {sum(1 for x in short if has_text(x))}) · echo {len(echo)} · noise {len(noise)}"
           if empty_sets else "")
        + (f" · guard: {', '.join(f'{fl} {len(v)}' for fl, v in guard.items())} (decision: "
           f"{'/'.join(a.guard_decision_langs)} <= +{a.guard_threshold:g} points; noise "
           f"{'decides' if a.noise_decides else 'report only'})" if guard_on else "")
        + (f" · extra test {len(extra)} (report only)" if extra else "")
        + f" · interim FLEURS-tr subset <= +{a.fleurs_tr_interim_points:g} points"
        + (" · interim also: short empty <= 2 and in-domain relative >= 10 %" if a.strict_interim else "")
        + (f" · L2-SP anchor tau {a.anchor_tau:g} steps (lambda {1.0 / (lr * a.anchor_tau):.2f}; every {a.anchor_interval} "
           f"steps k {k_anchor:.7f}; fp16 on CPU {s['anchor_mb']} MiB)" if anchor else "")
        + (f" · short test PADDED ({PAD_START_S:g} + {PAD_END_S:g} s zeros; unpadded reported alongside)" if short_padded else "")
        + f" · mixture {s['mixture_sha'][:16]}…" + (" (as expected)" if a.expected_mixture_sha else "")
        + f" · train.py {self_sha()[:16]}…")

    def drift():
        """Relative distance of the trained weights from the original: all, encoder, decoder + joint."""
        tot = {"enc": [0.0, 0.0], "dec": [0.0, 0.0]}
        with torch.no_grad():
            for n, p in model.named_parameters():
                a0 = anchor.get(n)
                if a0 is None:
                    continue
                a32 = a0.to(device=p.device, dtype=torch.float32)
                part = tot["enc" if n.startswith("encoder.") else "dec"]
                part[0] += float((p.detach().float() - a32).pow(2).sum())
                part[1] += float(a32.pow(2).sum())
                del a32

        def ratio(x):
            return round((x[0] / x[1]) ** 0.5, 6) if x[1] > 0 else None

        return dict(drift=ratio([tot["enc"][0] + tot["dec"][0], tot["enc"][1] + tot["dec"][1]]),
                    drift_enc=ratio(tot["enc"]), drift_dec=ratio(tot["dec"]))

    anchor_pairs = []

    def apply_anchor(module):
        """theta <- theta + k (theta0 - theta). The anchor comes from the CPU weight by weight; no permanent copy on MPS."""
        if not anchor_pairs:
            for n, p in module.named_parameters():
                if n in anchor:
                    anchor_pairs.append((p, anchor[n]))
            if len(anchor_pairs) != len(anchor):
                raise RuntimeError(f"anchor has {len(anchor)} weights, module matches {len(anchor_pairs)}")
        with torch.no_grad():
            for p, a0 in anchor_pairs:
                p.lerp_(a0.to(device=p.device, dtype=p.dtype), k_anchor)

    def error(rows_, hyp):
        e = n = 0
        for x, h in zip(rows_, hyp):
            r = T.normalize(x["raw_text"])
            if r:
                e += T.wer(x["raw_text"], h) * len(r)
                n += len(r)
        return round(e / max(1, n), 4), sum(1 for h in hyp if not T.normalize(h))

    def measure(step_no, full=False):
        state = [(p, p.requires_grad) for p in model.parameters()]
        was_training = model.training
        model.to(device)
        model.eval()
        model.encoder.set_default_att_context_size(ATT_CONTEXT_560MS)
        o = dict(step=step_no)
        if anchor:
            o.update(drift())
        say(f"  [train] step {step_no}: measurement starting" + (" (full FLEURS-tr)" if full else "")
            + (f" · drift {o['drift']}" if anchor else "") + " …")
        for name, lst in (("indomain_aux", aux), ("indomain", indomain), ("yodas_dev", ydev), ("extra", extra),
                          ("fleurs_tr", fleurs_tr if full else fleurs_sub)):
            if not lst:
                continue
            hyp = transcribe(model, [x["audio_filepath"] for x in lst], "tr-TR", tmp_dir, batch_size=8)
            o[name], o[name + "_empty"] = error(lst, hyp)
            if name == "fleurs_tr":
                o["fleurs_tr_set"] = "full" if full else "subset"
                sub_paths = {x["audio_filepath"] for x in fleurs_sub}
                picked = [(x, h) for x, h in zip(lst, hyp) if x["audio_filepath"] in sub_paths]
                o["fleurs_tr_sub"], _ = error([x for x, _ in picked], [h for _, h in picked])
        for name, lst in (("echo", echo), ("noise", noise)):
            hyp = transcribe(model, [x["audio_filepath"] for x in lst], "tr-TR", tmp_dir, batch_size=8) if lst else []
            o[name + "_empty"] = sum(1 for h in hyp if not T.normalize(h))
        if short:   # short utterances must be written; error only on those with text
            hyp = transcribe(model, [x["audio_filepath"] for x in short], "tr-TR", tmp_dir, batch_size=8)
            if short_padded:   # unpadded result reported alongside; the criterion is the padded measurement
                o["short_unpadded_empty"] = sum(1 for h in hyp if not T.normalize(h))
                wt = [(x, h) for x, h in zip(short, hyp) if has_text(x)]
                o["short_unpadded"], _ = error([x for x, _ in wt], [h for _, h in wt]) if wt else (None, 0)
                hyp = transcribe(model, short_padded, "tr-TR", tmp_dir, batch_size=8)
            o["short_empty"] = sum(1 for h in hyp if not T.normalize(h))
            wt = [(x, h) for x, h in zip(short, hyp) if has_text(x)]
            o["short"], _ = error([x for x, _ in wt], [h for _, h in wt]) if wt else (None, 0)
        for fl, lst in guard.items():   # other languages, own prompt; metric of the multilingual evaluation
            hyp = transcribe(model, [x["audio_filepath"] for x in lst], lst[0]["prompt"], tmp_dir, batch_size=8)
            o["g_" + fl] = round(T.error_rate([x["raw_text"] for x in lst], hyp, lst[0]["metric"]), 4)
            o["gempty_" + fl] = sum(1 for h in hyp if not T.units(h, lst[0]["metric"]))
            o["gtr_" + fl] = sum(1 for h in hyp if "<tr-TR>" in (h or ""))
            if device == "mps":
                torch.mps.empty_cache()
        for p, f in state:
            p.requires_grad = f   # transcribe() freezes/unfreezes the encoder; restore our setting
        if was_training:
            model.train()
            model.encoder.eval()
            for layer in list(model.encoder.layers)[-n_open:]:
                layer.train()
        if device == "mps":
            torch.mps.empty_cache()
        say(f"  [train] step {step_no}: in-domain {o.get('indomain')} (empty {o.get('indomain_empty')}) · in-domain aux "
            f"{o.get('indomain_aux')} · YODAS-dev {o.get('yodas_dev')} · FLEURS-tr-{o.get('fleurs_tr_set')} {o.get('fleurs_tr')} "
            f"(subset {o.get('fleurs_tr_sub')})" + (f" · extra test {o.get('extra')} (empty {o.get('extra_empty')})" if extra else ""))
        if empty_sets:
            say(f"  [train] step {step_no}: short: empty {o.get('short_empty')}/{len(short)}, error {o.get('short')}"
                + (f" (unpadded: empty {o.get('short_unpadded_empty')}/{len(short)}, error {o.get('short_unpadded')})" if short_padded else "")
                + f" · echo empty {o.get('echo_empty')}/{len(echo)} · noise empty {o.get('noise_empty')}/{len(noise)}")
        if guard:
            say(f"  [train] step {step_no}: guard (FLEURS dev) " + " · ".join(
                f"{fl} {lst[0]['metric'].upper()} {o['g_' + fl] * 100:.1f} (empty {o['gempty_' + fl]}, <tr-TR> {o['gtr_' + fl]})"
                for fl, lst in guard.items()))
        return o

    # resume: after every measurement the open weights + optimizer state + measurements go to disk; restarted with the
    # same data it continues from the step it reached (--from-scratch starts over)
    resume_path = os.path.join(a.out_dir, a.resume_file)
    with open(a.train_manifest, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()[:16] + f"|{n_open}|{lr}|{a.batch_seconds}|{a.buckets}|{quadratic}"
    if anchor:   # a resume file of a run with another anchor is not taken over
        fingerprint += f"|anchor{a.anchor_tau:g}/{a.anchor_interval}"
    start, opt_state, measurements, best = 0, None, [], None
    if os.path.exists(resume_path) and not a.from_scratch:
        d = torch.load(resume_path, map_location="cpu", weights_only=False)
        if d.get("fingerprint") == fingerprint:
            with torch.no_grad():
                named = dict(model.named_parameters())
                for n, t in d["open"].items():
                    named[n].copy_(t)
            start, opt_state, measurements, best = d["step"], d["opt"], d["measurements"], d["best"]
            say(f"  ! resuming: step {start}/{a.steps} (resume file {resume_path})")
        else:
            say("  ! resume file belongs to other data or settings — starting from scratch")
        del d
    s["resumed_step"] = start
    t0 = time.time()
    if start == 0:
        measurements.append(measure(0, full=True))
        best = dict(step=0, key=None, score=2.0, eligible=None, state=None)
    base = measurements[0]

    # Step selection: first the steps meeting the interim criteria, among them the lowest in-domain error (relative to
    # the base, mean over in-domain sets). If none meets them, the lowest balance score.
    def rel(o, name):
        t, v = base.get(name), o.get(name)
        return None if v is None or not t else v / t

    def indomain_rel(o):
        xs = [g for g in (rel(o, "indomain"), rel(o, "indomain_aux")) if g is not None]
        return round(sum(xs) / len(xs), 4) if xs else 1.0

    def echo_share(o):
        return o.get("echo_empty", 0) / len(echo) if echo else None

    def noise_share(o):
        return o.get("noise_empty", 0) / len(noise) if noise else None

    # guard: in the decision languages (de, en) the guard error may rise by at most --guard-threshold points; the balance
    # score includes the mean relative guard error (3.0 at the base)
    def guard_diff(o, fl):
        t, v = base.get("g_" + fl), o.get("g_" + fl)
        return None if t is None or v is None else v - t

    def guard_ok(o):
        return all(guard_diff(o, fl) is None or guard_diff(o, fl) * 100 <= a.guard_threshold + 1e-9
                   for fl in a.guard_decision_langs if fl in guard)

    def guard_rel(o):
        xs = [o["g_" + fl] / base["g_" + fl] for fl in guard if o.get("g_" + fl) is not None and base.get("g_" + fl)]
        return round(sum(xs) / len(xs), 4) if xs else 1.0

    def eligible(o):
        f_ok = o.get("fleurs_tr_sub") is None or base.get("fleurs_tr_sub") is None or \
            (o["fleurs_tr_sub"] - base["fleurs_tr_sub"]) * 100 <= a.fleurs_tr_interim_points + 1e-9
        g_ok = guard_ok(o) if guard_on else True
        if empty_sets:
            e_ok = (echo_share(base) - echo_share(o)) * 100 <= EMPTY_DROP_POINTS
            n_ok = (not a.noise_decides) or (noise_share(base) - noise_share(o)) * 100 <= EMPTY_DROP_POINTS
            strict_ok = True
            if a.strict_interim:   # selection also respects the verdict criteria (short empty <= 2, in-domain >= 10 %)
                strict_ok = (o.get("short_empty") if o.get("short_empty") is not None else 99) <= SHORT_MAX_EMPTY and (
                    not base.get("indomain") or o.get("indomain") is None or 1 - o["indomain"] / base["indomain"] >= INDOMAIN_MIN_REL)
            return bool(f_ok and e_ok and n_ok and g_ok and strict_ok)
        return bool(f_ok and g_ok)

    def score(o):
        fr = rel(o, "fleurs_tr_sub")
        return round(indomain_rel(o) + (fr if fr is not None else 1.0) + (guard_rel(o) if guard_on else 0.0), 4)

    def key(o):  # smaller is better
        return (0, indomain_rel(o)) if eligible(o) else (1, score(o))

    for o in measurements:   # for the table and the summary (also old measurements of a resumed run)
        o.update(indomain_rel=indomain_rel(o), score=score(o), eligible=eligible(o) if o is not base else None)
        if guard_on:
            o["guard_rel"] = guard_rel(o)

    class Watch(pl.Callback):
        def __init__(self):
            self.n = 0
            self.t = time.perf_counter()
            self.audio = 0.0
            self.measure_s = 0.0

        def on_train_start(self, trainer, pl_module):
            if opt_state is not None:
                trainer.optimizers[0].load_state_dict(opt_state)
                say("  [train] optimizer state restored")

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            pl_module.encoder.eval()
            for layer in list(pl_module.encoder.layers)[-n_open:]:
                layer.train()
            self.audio += float(batch[1].sum().item()) / 16000.0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.n += 1
            total = start + self.n
            if anchor and total % a.anchor_interval == 0:
                apply_anchor(pl_module)
            m = memory()
            peak["mps"] = max(peak["mps"], m["mps_driver_mb"] or 0)
            if self.n % a.progress_every == 0:
                train_s = max(1e-6, time.perf_counter() - self.t - self.measure_s)
                left_min = (a.steps - total) * train_s / self.n / 60
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                say(f"  [train] step {total}/{a.steps} · loss {float(loss.detach().float().cpu()):.2f} · "
                    f"{self.audio / train_s:.1f}x real time · MPS {m['mps_driver_mb'] or 0:.0f} MB · "
                    f"free {m['free_mb']:.0f} MB · training left ~{left_min:.0f} min")
                if device == "mps":
                    torch.mps.empty_cache()  # against fragmentation in a long run
            if total % a.eval_every == 0 or total == a.steps:
                t_m = time.perf_counter()
                o = measure(total)
                o.update(indomain_rel=indomain_rel(o), score=score(o), eligible=eligible(o))
                if guard_on:
                    o["guard_rel"] = guard_rel(o)
                measurements.append(o)
                if best["key"] is None or key(o) < tuple(best["key"]):
                    best.update(step=total, key=key(o), score=o["score"], eligible=o["eligible"],
                                state={n: p.detach().to("cpu").clone() for n, p in pl_module.named_parameters() if p.requires_grad})
                open_w = best["state"] if best["step"] == total else \
                    {n: p.detach().to("cpu").clone() for n, p in pl_module.named_parameters() if p.requires_grad}
                tmp = resume_path + ".part"
                torch.save(dict(fingerprint=fingerprint, step=total, open=open_w, opt=trainer.optimizers[0].state_dict(),
                                measurements=measurements, best=best), tmp)
                os.replace(tmp, resume_path)
                del open_w
                say(f"  [train] step {total}: interim criteria {'met' if o['eligible'] else 'NOT met'} · in-domain (relative) "
                    f"{o['indomain_rel']}" + (f" · guard (relative) {o.get('guard_rel')}" if guard_on else "")
                    + (f" · drift {o.get('drift')} (encoder {o.get('drift_enc')} · decoder {o.get('drift_dec')})" if anchor else "")
                    + f" · balance score {o['score']} -> best step {best['step']} · state saved")
                self.measure_s += time.perf_counter() - t_m

    left = a.steps - start
    if left > 0:
        train_path = a.train_manifest
        if start > 0:
            # lhotse largely keeps the manifest order within buckets; a resumed run must not start with the same clips
            with open(a.train_manifest, encoding="utf-8") as f:
                lines = [x for x in f if x.strip()]
            random.Random(RESUME_SHUFFLE_SEED + start).shuffle(lines)
            train_path = os.path.join(a.out_dir, "mixture_resume.jsonl")
            with open(train_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        cfg = train_data_config(model, train_path, a.batch_seconds, quadratic, buckets=a.buckets)
        cfg.shuffle = True
        model.setup_training_data(cfg)
        trainer = pl.Trainer(accelerator=device, devices=1, max_steps=left, max_epochs=1000, precision="32-true", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, limit_val_batches=0, log_every_n_steps=100000,
                             use_distributed_sampler=False, callbacks=[Watch()], default_root_dir=a.out_dir)
        model.set_trainer(trainer)
        model.setup_optimization(optim_config=OmegaConf.create(dict(name="adamw", lr=lr, betas=[0.9, 0.98], weight_decay=1e-3)))
        try:
            trainer.fit(model)
            s["ok"] = True
        except Exception as e:
            s["error"] = repr(e)[:600]
            say(f"  [train] TRAINING ERROR: {e!r}")
            say("  ! the state of the last measurement is in the resume file; the same command continues from there")
            traceback.print_exc()
            if device == "mps":
                torch.mps.empty_cache()
    else:
        s["ok"] = True
    s["minutes"] = round((time.time() - t0) / 60, 1)
    s["mps_peak_mb"] = round(peak["mps"])
    s["measurements"] = measurements
    s["best_step"] = best["step"]
    s["best_score"] = best["score"]
    s["best_eligible"] = best["eligible"]
    s["resume_file"] = resume_path
    final = None
    if best.get("state") is not None:
        with torch.no_grad():
            named = dict(model.named_parameters())
            for n, t in best["state"].items():
                named[n].copy_(t.to(named[n].device))
        try:
            # all tests with the selected step's weights and the FULL FLEURS-tr set (the verdict uses this measurement)
            final = measure(f"{best['step']}-full", full=True)
            final.update(indomain_rel=indomain_rel(final), score=score(final), eligible=eligible(final))
            if guard_on:
                final["guard_rel"] = guard_rel(final)
            measurements.append(final)
            s["fleurs_tr_full_best"] = final.get("fleurs_tr")
        except Exception as e:
            s["final_measure_error"] = repr(e)[:300]
            say(f"  [train] final measurement ERROR: {e!r}")
        target = os.path.join(a.out_dir, f"model_{run_id}.nemo")
        model.to("cpu")
        model.save_to(target)
        s["model"] = target
        say(f"  [train] selected step {best['step']} saved: {target}")
    # In-training verdict: in-domain relative >= 10 % · full FLEURS-tr <= +0.5 points · echo empty share drop <= 5 points ·
    # short-utterance empty <= 2 · guard de/en <= +1.5 points (noise: report only unless --noise-decides).
    b = final or next((o for o in measurements if o.get("step") == best["step"]), base)
    if base.get("indomain"):
        s["indomain_base"], s["indomain_best"] = base["indomain"], b.get("indomain")
        s["indomain_rel_gain"] = round(1 - (b.get("indomain") if b.get("indomain") is not None else base["indomain"]) / base["indomain"], 3)
        s["crit_indomain"] = s["indomain_rel_gain"] >= INDOMAIN_MIN_REL
    if base.get("indomain_aux"):
        s["indomain_aux_rel_gain"] = round(1 - (b.get("indomain_aux") if b.get("indomain_aux") is not None else base["indomain_aux"])
                                           / base["indomain_aux"], 3)
    if empty_sets:
        s["echo_empty_share_base"], s["echo_empty_share_best"] = round(echo_share(base), 3), round(echo_share(b), 3)
        s["crit_echo"] = (echo_share(base) - echo_share(b)) * 100 <= EMPTY_DROP_POINTS
        s["noise_empty_share_base"], s["noise_empty_share_best"] = round(noise_share(base), 3), round(noise_share(b), 3)
        s["crit_noise"] = (noise_share(base) - noise_share(b)) * 100 <= EMPTY_DROP_POINTS
        s["short_empty_base"], s["short_empty_best"] = base.get("short_empty"), b.get("short_empty")
        s["short_base"], s["short_best"] = base.get("short"), b.get("short")
        s["crit_short"] = (b.get("short_empty") if b.get("short_empty") is not None else 99) <= SHORT_MAX_EMPTY
    s["noise_decides"] = a.noise_decides
    if guard_on:
        s["guard_base"] = {fl: base.get("g_" + fl) for fl in guard}
        s["guard_best"] = {fl: b.get("g_" + fl) for fl in guard}
        s["guard_diff"] = {fl: (None if guard_diff(b, fl) is None else round(guard_diff(b, fl), 4)) for fl in guard}
        s["guard_empty_best"] = {fl: b.get("gempty_" + fl) for fl in guard}
        s["guard_decision_langs"] = list(a.guard_decision_langs)
        s["guard_threshold"] = a.guard_threshold
        s["crit_guard"] = guard_ok(b)
    s["fleurs_tr_interim_points"] = a.fleurs_tr_interim_points
    s["strict_interim"] = a.strict_interim
    s["train_py_sha"] = self_sha()[:16]
    if anchor:
        s["drift_best"] = b.get("drift")
        s["drift_enc_best"], s["drift_dec_best"] = b.get("drift_enc"), b.get("drift_dec")
    if extra:
        s["extra_test"] = len(extra)
        s["extra_base"], s["extra_best"] = base.get("extra"), b.get("extra")
    if short_padded:
        s["short_padded"] = True
        s["short_unpadded_empty_base"], s["short_unpadded_empty_best"] = base.get("short_unpadded_empty"), b.get("short_unpadded_empty")
        s["short_unpadded_base"], s["short_unpadded_best"] = base.get("short_unpadded"), b.get("short_unpadded")
    if base.get("fleurs_tr") is not None and s.get("fleurs_tr_full_best") is not None:
        s["fleurs_tr_full_base"] = base["fleurs_tr"]
        s["crit_fleurs_tr"] = (s["fleurs_tr_full_best"] - base["fleurs_tr"]) * 100 <= FLEURS_TR_MAX_POINTS
    if base.get("yodas_dev") is not None and b.get("yodas_dev") is not None:
        s["yodas_dev_base"], s["yodas_dev_best"] = base["yodas_dev"], b["yodas_dev"]
    s["passed"] = bool(s.get("crit_indomain") and s.get("crit_fleurs_tr", False) and empty_sets
                       and s.get("crit_echo") and (s.get("crit_noise") or not a.noise_decides) and s.get("crit_short")
                       and (not guard_on or s.get("crit_guard")))
    cols = ["step", "indomain", "indomain_empty", "indomain_aux", "indomain_aux_empty", "yodas_dev", "yodas_dev_empty",
            "fleurs_tr_set", "fleurs_tr", "fleurs_tr_empty", "fleurs_tr_sub", "indomain_rel", "score", "eligible",
            "short", "short_empty", "echo_empty", "noise_empty", "extra", "extra_empty"]
    if guard_on:
        cols += ["guard_rel"] + [f"{p}_{fl}" for fl in guard for p in ("g", "gempty", "gtr")]
    if anchor:
        cols += ["drift", "drift_enc", "drift_dec"]
    if short_padded:
        cols += ["short_unpadded", "short_unpadded_empty"]
    with open(os.path.join(a.out_dir, f"steps_{run_id}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for o in measurements:
            w.writerow([o.get(k) for k in cols])
    for o in measurements:   # keep the summary JSON-serialisable
        o.pop("state", None)
    with open(os.path.join(a.out_dir, f"summary_{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    say(f"  [train] selected step {s['best_step']} (interim criteria {'met' if s['best_eligible'] else 'not met'}; balance "
        f"{s['best_score']}) · verdict {'PASSED' if s['passed'] else 'NOT PASSED'} · {s['minutes']} min · MPS peak {s['mps_peak_mb']} MB")
    say(f"  [train] outputs: {a.out_dir}/summary_{run_id}.json · steps_{run_id}.csv" + (f" · {os.path.basename(s['model'])}" if s.get("model") else ""))
    return 0 if s["ok"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="environment check / ortam denetimi")
    r = sub.add_parser("run", help="train / eğit", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("--work-dir", help="working directory / çalışma dizini ($AYAZ_ASR_WORK or ./work)")
    r.add_argument("--base-model", help="base .nemo ($AYAZ_ASR_BASE_NEMO or <work>/models/...)")
    r.add_argument("--train-manifest", help="mixture from build_mixture.py (<work>/manifests/mixture.jsonl)")
    r.add_argument("--expected-mixture-sha", default="", help="refuse to train if the mixture's SHA-256 differs")
    r.add_argument("--out-dir", help="outputs: resume file, CSV, summary, .nemo (<work>/train)")
    r.add_argument("--steps", type=int, default=9000, help="training steps")
    r.add_argument("--eval-every", type=int, default=1000, help="measurement interval (steps)")
    r.add_argument("--progress-every", type=int, default=100, help="progress line / MPS cache release interval (steps)")
    r.add_argument("--batch-seconds", type=float, default=30.0, help="audio seconds per batch")
    r.add_argument("--layers", type=int, default=8, help="trainable last encoder layers")
    r.add_argument("--lr", type=float, default=3e-5, help="AdamW learning rate")
    r.add_argument("--buckets", type=int, default=6, help="lhotse buckets (0 = off)")
    r.add_argument("--quadratic-duration", type=float, default=15.0, help="lhotse quadratic_duration (0 = off)")
    r.add_argument("--joint-log-softmax", action="store_true", help="keep the joint's log_softmax (v0.1: off)")
    r.add_argument("--anchor-tau", type=float, default=4500.0, help="L2-SP anchor time constant in steps (0 = no anchor)")
    r.add_argument("--anchor-interval", type=int, default=10, help="apply the anchor every N steps")
    r.add_argument("--keep-digits", action="store_true", help="score without spelling digits as words (v0.1 spelled them out)")
    r.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"),
                   help="auto = MPS if available, else CPU; only MPS was used for v0.1, cpu and cuda are untested")
    r.add_argument("--resume-file", default="resume.pt", help="resume file name inside --out-dir")
    r.add_argument("--from-scratch", action="store_true", help="ignore an existing resume file")
    r.add_argument("--fleurs-tr-test", help="FLEURS-tr test manifest (<work>/manifests/fleurs_tr_test.jsonl)")
    r.add_argument("--fleurs-tr-interim-points", type=float, default=0.3, help="interim: max rise of the FLEURS-tr subset (points)")
    r.add_argument("--yodas-dev", help="YODAS dev manifest, report only (<work>/manifests/yodas_dev.jsonl)")
    r.add_argument("--guard-dir", help="guard sets from build_rehearsal.py (<work>/rehearsal/guard)")
    r.add_argument("--guard-langs", default="de_de,en_us,fr_fr,ru_ru,ja_jp,ro_ro,et_ee,bg_bg", help="guard languages")
    r.add_argument("--guard-decision-langs", default="de_de,en_us", help="guard languages that decide")
    r.add_argument("--guard-threshold", type=float, default=1.5, help="max guard error rise in decision languages (points)")
    r.add_argument("--indomain-test", help="OPTIONAL in-house in-domain Turkish test (not distributed)")
    r.add_argument("--indomain-aux-test", help="OPTIONAL second in-domain test, selection score only")
    r.add_argument("--short-test", help="OPTIONAL short-utterance test")
    r.add_argument("--short-test-padding", action="store_true", help="measure the short test padded (v0.1: on)")
    r.add_argument("--pad-dir", help="where padded copies are written (<out-dir>/padded)")
    r.add_argument("--echo-test", help="OPTIONAL echo test (must stay empty)")
    r.add_argument("--noise-test", help="OPTIONAL room-noise test (empty)")
    r.add_argument("--noise-decides", action="store_true", help="noise empty-share drop is a criterion (v0.1: off)")
    r.add_argument("--extra-test", help="OPTIONAL further Turkish test, report only")
    r.add_argument("--no-strict-interim", dest="strict_interim", action="store_false",
                   help="interim selection without the short-empty / in-domain criteria (v0.1: strict)")
    a = ap.parse_args()
    if a.command == "env":
        sys.exit(cmd_env(a))
    work = work_dir(a.work_dir)
    a.base_model = base_model_path(work, a.base_model)
    a.train_manifest = os.path.abspath(a.train_manifest or os.path.join(work, "manifests", "mixture.jsonl"))
    a.out_dir = os.path.abspath(a.out_dir or os.path.join(work, "train"))
    a.pad_dir = a.pad_dir or os.path.join(a.out_dir, "padded")
    a.fleurs_tr_test = a.fleurs_tr_test or os.path.join(work, "manifests", "fleurs_tr_test.jsonl")
    a.yodas_dev = a.yodas_dev or os.path.join(work, "manifests", "yodas_dev.jsonl")
    a.guard_dir = a.guard_dir or os.path.join(work, "rehearsal", "guard")
    a.guard_langs = tuple(x for x in a.guard_langs.split(",") if x)
    a.guard_decision_langs = tuple(x for x in a.guard_decision_langs.split(",") if x)
    sys.exit(cmd_run(a))


if __name__ == "__main__":
    main()
