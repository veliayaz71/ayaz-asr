"""NeMo helpers shared by training, rehearsal labelling and evaluation (torch / NeMo imported lazily)."""
import json
import os

# Encoder attention context [left, right] = 560 ms chunks: the streaming setting used everywhere in v0.1.
ATT_CONTEXT_560MS = [56, 6]
# Silence added before / after an utterance by a live streaming bridge (seconds); used for the padded short test.
PAD_START_S, PAD_END_S = 1.0, 1.2


def load_model(path):
    import time
    import nemo.collections.asr as nemo_asr
    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(restore_path=path, map_location="cpu")
    return model, time.time() - t0


def param_count(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def _text_of(x):
    if isinstance(x, (list, tuple)):
        return _text_of(x[0]) if x else ""
    if isinstance(x, str):
        return x
    return getattr(x, "text", str(x))


def transcribe(model, paths, lang, tmp_dir, batch_size=4):
    """In NeMo 3.0.0, transcribe(target_lang=...) with plain file paths does not set the language prompt (the cut's
    language stays empty), and the default prompt mode 'unified' picks 'auto' about half of the time. Therefore every
    input is passed as a manifest line: language in 'lang', prompt mode 'langID'."""
    import soundfile as sf
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f"transcribe_{lang}.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(json.dumps(dict(audio_filepath=p, duration=round(sf.info(p).duration, 3), text="", lang=lang,
                                    prompt_mode="langID"), ensure_ascii=False) + "\n")
    out = model.transcribe([tmp], batch_size=batch_size, target_lang=lang, verbose=False, num_workers=0)
    if isinstance(out, tuple):
        out = out[0]
    return [_text_of(o) for o in out]


def write_padded(src, dst):
    """Copy with PAD_START_S of zeros before and PAD_END_S after (32-bit float WAV); returns ``dst``.
    Written via ``.part`` so an interrupted run leaves no broken file."""
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tmp = dst + ".part"
    sf.write(tmp, np.concatenate([np.zeros(int(PAD_START_S * sr), np.float32), audio, np.zeros(int(PAD_END_S * sr), np.float32)]),
             sr, subtype="FLOAT", format="WAV")
    os.replace(tmp, dst)
    return dst


def install_cpu_rnnt_loss():
    """Moves the logits to the CPU right before the RNNT loss (numba CPU path); the gradient flows back through
    ``.to()``. NeMo's installed files are not touched — only the class method is wrapped in this process."""
    from nemo.collections.asr.losses import rnnt as rnnt_mod
    original = rnnt_mod.RNNTLoss.forward

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        device = log_probs.device
        if device.type != "cpu":
            k = original(self, log_probs=log_probs.float().to("cpu"), targets=targets.to("cpu"),
                         input_lengths=input_lengths.to("cpu"), target_lengths=target_lengths.to("cpu"))
            return k.to(device)
        return original(self, log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths)

    rnnt_mod.RNNTLoss.forward = forward


def train_data_config(model, manifest, batch_seconds=40.0, quadratic=None, buckets=0):
    """Lhotse data config derived from the model's own train_ds (tarred/bucket keys removed)."""
    from omegaconf import OmegaConf
    base = OmegaConf.to_container(model.cfg.train_ds, resolve=True) if model.cfg.get("train_ds") is not None else {}
    for k in ("tarred_audio_filepaths", "input_cfg", "bucket_duration_bins", "bucket_batch_size", "bucketing_batch_size",
              "num_buckets", "shar_path", "cuts_path", "quadratic_duration", "bucket_buffer_size"):
        base.pop(k, None)
    base.update(dict(
        manifest_filepath=manifest, is_tarred=False, use_lhotse=True, use_bucketing=False,
        batch_duration=float(batch_seconds), batch_size=None,
        shuffle=False, num_workers=0, pin_memory=False, max_duration=20.0, min_duration=0.3,
        shuffle_buffer_size=100, lang_field="target_lang", default_prompt_mode="langID", quadratic_duration=quadratic,
        sample_rate=model.cfg.get("sample_rate", 16000),
    ))
    if buckets > 0:
        # The batch tensor is B x T_max x U_max x V; bucketing puts similar lengths into the same batch.
        base.update(dict(use_bucketing=True, num_buckets=int(buckets), num_cuts_for_bins_estimate=10000,
                         bucket_buffer_size=10000, concurrent_bucketing=False))
    if "prompt_dictionary" not in base:
        base["prompt_dictionary"] = OmegaConf.to_container(model.cfg.model_defaults.prompt_dictionary, resolve=True)
    if "num_prompts" not in base:
        base["num_prompts"] = int(model.cfg.model_defaults.get("num_prompts", 128))
    if "subsampling_factor" not in base:
        base["subsampling_factor"] = int(model.cfg.encoder.get("subsampling_factor", 8))
    return OmegaConf.create(base)


def memory():
    import psutil
    d = dict(rss_mb=psutil.Process().memory_info().rss / 2**20, free_mb=psutil.virtual_memory().available / 2**20,
             mps_alloc_mb=None, mps_driver_mb=None)
    try:
        import torch
        if torch.backends.mps.is_available():
            d["mps_alloc_mb"] = torch.mps.current_allocated_memory() / 2**20
            d["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    except Exception:
        pass
    return d


def empty_cache(device):
    if device == "mps":
        import torch
        torch.mps.empty_cache()
