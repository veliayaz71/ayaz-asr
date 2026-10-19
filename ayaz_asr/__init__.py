"""Shared helpers for the Ayaz ASR training recipe.

Heavy libraries (torch, NeMo, soundfile, ...) are imported inside the functions that need them, so every
entry point can show ``--help`` without them installed.
"""
import json
import os
import time

__version__ = "0.1"

ENV_WORK = "AYAZ_ASR_WORK"
ENV_BASE_MODEL = "AYAZ_ASR_BASE_NEMO"
BASE_MODEL_FILE = "nemotron-3.5-asr-streaming-0.6b.nemo"


def work_dir(path=None):
    """Working directory: ``--work-dir`` if given, else $AYAZ_ASR_WORK, else ./work."""
    return os.path.abspath(path or os.environ.get(ENV_WORK) or "work")


def base_model_path(work, path=None):
    """Base .nemo: ``--base-model`` if given, else $AYAZ_ASR_BASE_NEMO, else <work>/models/<file>."""
    return os.path.abspath(path or os.environ.get(ENV_BASE_MODEL) or os.path.join(work, "models", BASE_MODEL_FILE))


def stamp():
    """UTC time stamp used to name run outputs."""
    return time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    """Atomic write via ``.part``."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(path + ".part", path)


def print_result(d, summary_path=None):
    """One machine-readable result line on stdout; optionally appended to a JSON-lines summary file."""
    print("RESULT_JSON " + json.dumps(d, ensure_ascii=False), flush=True)
    if summary_path:
        try:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"! could not write summary file: {e!r}", flush=True)
