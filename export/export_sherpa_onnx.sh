#!/bin/bash
# Export a fine-tuned .nemo to a sherpa-onnx streaming package (560 ms chunks, int8); follows the procedure used for
# Ayaz ASR v0.1 (published package: selected step 8000, 560 ms).
# TR: İnce ayarlı .nemo'yu sherpa-onnx akışlı paketine (560 ms parça, int8) dışa aktarır — v0.1'deki yol.
#
# Usage:
#   REFERENCE=/path/to/official-sherpa-onnx-package bash export/export_sherpa_onnx.sh <model.nemo> <output-dir>
#
#   REFERENCE  the official sherpa-onnx int8 package of the base model (560 ms chunks); its encoder metadata and
#              tokens.txt are the reference (the vocabulary must not change).
#   PY_NEMO    Python of the training environment (NeMo, torch, onnx) (default: python3)
#   PY_Q       Python of the quantisation environment, export/requirements-export.txt (default: .venv-export/bin/python;
#              created with "$PY_NEMO -m venv" + pip if missing)
#   TOKENS_SHA expected SHA-256 of the official tokens.txt (default: the value seen for v0.1)
#   LABEL      text for the exporter's 'comment' metadata (default: "local fine-tune"); replaced by COMMENT in step 5
#   COMMENT    final neutral 'comment' metadata (default: finalize_package.py's default text)
#
# Steps: 1 NeMo -> ONNX fp32 (export_onnx.py, CPU) · 2 int8 (quantize_int8.py, separate environment) · 3 tokens.txt
# byte-identical to the official one · 4 the 18 required encoder metadata keys identical to the official ones
# (compare_metadata.py) · 5 finalisation: neutral 'comment', single files, ORT load, forbidden-pattern scan
# (finalize_package.py) · README.txt + SHA256SUMS.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: REFERENCE=... $0 <model.nemo> <output-dir>"; exit 2; }
NEMO_IN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; OUT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
CHUNK_MS=560
PY_NEMO="${PY_NEMO:-python3}"
PY_Q="${PY_Q:-$HERE/../.venv-export/bin/python}"
TOKENS_SHA="${TOKENS_SHA:-729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a}"
LABEL="${LABEL:-local fine-tune}"
COMMENT="${COMMENT:-}"
: "${REFERENCE:?set REFERENCE to the official sherpa-onnx package directory of the base model}"
shaof() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1; }

[ -f "$NEMO_IN" ] || { echo "STOP: model missing: $NEMO_IN"; exit 1; }
for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do
  [ -s "$REFERENCE/$f" ] || { echo "STOP: official file missing: $REFERENCE/$f"; exit 1; }
done
[ "$(shaof "$REFERENCE/tokens.txt")" = "$TOKENS_SHA" ] || { echo "STOP: official tokens.txt is not the expected one (${TOKENS_SHA:0:16}…)"; exit 1; }
if ! "$PY_Q" -c "import onnx, onnxruntime" >/dev/null 2>&1; then
  echo "--- quantisation environment ($PY_Q): onnx + onnxruntime (the training environment is not touched)"
  "$PY_NEMO" -m venv "$(dirname "$(dirname "$PY_Q")")"
  "$PY_Q" -m pip install --quiet -r "$HERE/requirements-export.txt"
fi
NEMO_SHA=$(shaof "$NEMO_IN")
WORK="$OUT.work"
PKG="$OUT.stage"
rm -rf "$WORK" "$PKG" "$OUT.part"
echo "--- 1. NeMo -> ONNX (CPU)"
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "$PY_NEMO" "$HERE/export_onnx.py" --nemo "$NEMO_IN" --output-dir "$WORK" \
  --label "$LABEL" --chunk-size-ms "$CHUNK_MS"
for f in encoder.onnx encoder.data decoder.onnx joiner.onnx; do [ -s "$WORK/$CHUNK_MS/$f" ] || { echo "STOP: export did not write $f"; exit 1; }; done
[ -s "$WORK/tokens.txt" ] || { echo "STOP: tokens.txt not written"; exit 1; }
echo "--- 2. int8"
PYTHONDONTWRITEBYTECODE=1 "$PY_Q" "$HERE/quantize_int8.py" "$WORK/$CHUNK_MS"
mkdir -p "$PKG"
for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx; do mv "$WORK/$CHUNK_MS/$f" "$PKG/$f"; done
cp "$WORK/tokens.txt" "$PKG/tokens.txt"
echo "--- 3. tokens.txt"
[ "$(shaof "$PKG/tokens.txt")" = "$TOKENS_SHA" ] || { echo "STOP: tokens.txt is NOT byte-identical to the official package — the vocabulary must not change"; exit 1; }
echo "--- 4. required encoder metadata"
"$PY_Q" "$HERE/compare_metadata.py" "$PKG/encoder.int8.onnx" "$REFERENCE/encoder.int8.onnx" \
  || { echo "STOP: required encoder metadata differs from the official package — not usable by the streaming runtime"; exit 1; }
echo "--- 5. finalisation (neutral comment, single files, ORT, forbidden patterns)"
"$PY_Q" "$HERE/finalize_package.py" --src "$PKG" --dst "$OUT.part" --reference "$REFERENCE" ${COMMENT:+--comment "$COMMENT"}
cat > "$OUT.part/README.txt" <<EOT
Fine-tuned from nvidia/nemotron-3.5-asr-streaming-0.6b (NVIDIA, OpenMDW-1.1) — sherpa-onnx streaming package
source .nemo SHA-256: $NEMO_SHA
chunk: ${CHUNK_MS} ms · int8 (onnxruntime dynamic quantisation, QUInt8) · exporter: sherpa-onnx
scripts/nemo/nemotron-3.5-asr-streaming-0.6b/export_onnx.py (99ddefaa; adapted for a local .nemo)
EOT
( cd "$OUT.part" && for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$f"; else sha256sum "$f"; fi; done > SHA256SUMS )
rm -rf "$OUT"
mv "$OUT.part" "$OUT"
rm -rf "$WORK" "$PKG"
echo "=== package: $OUT"
cat "$OUT/SHA256SUMS"
