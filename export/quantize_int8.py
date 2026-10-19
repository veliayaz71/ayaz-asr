#!/usr/bin/env python3
"""Converts the fp32 output of export_onnx.py to int8 the way sherpa-onnx does (onnxruntime quantize_dynamic, QUInt8).
Usage: quantize_int8.py <dir>   (dir holds encoder.onnx + encoder.data, decoder.onnx, joiner.onnx)
Run it in the separate export environment (export/requirements-export.txt).

TR: export_onnx.py'nin fp32 çıktısını int8'e çevirir (onnxruntime quantize_dynamic, QUInt8)."""
import argparse
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dir", help="directory with encoder.onnx (+ encoder.data), decoder.onnx, joiner.onnx / dışa aktarma dizini")
d = ap.parse_args().dir
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
for m in ("encoder", "decoder", "joiner"):
    quantize_dynamic(model_input=os.path.join(d, f"{m}.onnx"), model_output=os.path.join(d, f"{m}.int8.onnx"),
                     weight_type=QuantType.QUInt8)
    print(f"  int8: {m}.int8.onnx · {os.path.getsize(os.path.join(d, f'{m}.int8.onnx'))} bytes", flush=True)
