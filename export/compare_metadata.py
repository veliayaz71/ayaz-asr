#!/usr/bin/env python3
"""Compares the encoder metadata of two sherpa-onnx packages (without loading weights). The keys REQUIRED by the
sherpa-onnx streaming runtime must be identical; other differences are only printed.
Usage: compare_metadata.py <our encoder.int8.onnx> <official encoder.int8.onnx>
Exit 0 = required keys identical; 3 = difference.

TR: İki paketin kodlayıcı üst verisini karşılaştırır; zorunlu 18 anahtar aynı olmalı."""
import argparse
import json
import sys

REQUIRED = ["vocab_size", "window_size", "chunk_size_ms", "chunk_shift", "normalize_type", "cache_last_channel_dim1",
            "cache_last_channel_dim2", "cache_last_channel_dim3", "cache_last_time_dim1", "cache_last_time_dim2",
            "cache_last_time_dim3", "pred_rnn_layers", "pred_hidden", "subsampling_factor", "feat_dim", "model_type",
            "prompt_dictionary", "auto_prompt_id"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("ours", help="our encoder.int8.onnx")
ap.add_argument("official", help="official encoder.int8.onnx of the base model's sherpa-onnx package")
args = ap.parse_args()
import onnx  # noqa: E402


def meta(p):
    m = onnx.load(p, load_external_data=False)
    return {x.key: x.value for x in m.metadata_props}


a, b = meta(args.ours), meta(args.official)


def same(k):
    if k == "prompt_dictionary":
        try:
            return json.loads(a.get(k, "{}")) == json.loads(b.get(k, "{}"))
        except Exception:
            return False
    return a.get(k) == b.get(k)


diff = [k for k in REQUIRED if not same(k)]
other = sorted(k for k in set(a) | set(b) if k not in REQUIRED and a.get(k) != b.get(k))
print("  metadata: %d of %d required keys identical%s · other differing keys: %s" % (
    len(REQUIRED) - len(diff), len(REQUIRED), (" — DIFFERENT: " + ", ".join(diff)) if diff else "", ", ".join(other) or "none"))
sys.exit(3 if diff else 0)
