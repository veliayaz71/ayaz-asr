#!/bin/bash
# Ayaz ASR v0.1 — end-to-end recipe with the v0.1 settings (configs/v0.1.yaml).
# TR: Ayaz ASR v0.1 tarifinin baştan sona koşusu (v0.1 ayarlarıyla).
#
#   AYAZ_ASR_WORK=/path/to/work bash reproduce.sh            # public data only
# Runs steps 1-9 of the README; steps 10-11 (export, cleaning a .nemo) are manual.
#
# Optional non-distributed data. v0.1 cannot be reproduced exactly from this repo: it also used consented volunteer voice
# messages and the developer's own recordings, which are not distributed. With public data alone, or with your own data,
# results will differ from the published numbers (especially on the Turkish tests); even with identical data, runs are
# not bit-for-bit identical (MPS, download order). Optional inputs:
#   INHOUSE_SPEECH  INHOUSE_EMPTY  INHOUSE_NOISE           -> data/build_turkish_manifest.py --speech/--empty/--noise
#   INHOUSE_EXTRA="voice_messages.jsonl:3 short_commands.jsonl:5"   -> data/build_mixture.py --extra (order matters)
#   INDOMAIN_TEST INDOMAIN_AUX_TEST SHORT_TEST ECHO_TEST NOISE_TEST EXTRA_TEST -> train/train.py test sets
#   (without INDOMAIN_TEST the YODAS dev set is used as the in-domain test — a public substitute, not what v0.1 did)
#   REHEARSAL_ARGS="..."  -> extra arguments for build_rehearsal.py (defaults are the v0.1 settings)
# Each step resumes or is cheap to repeat; run time on one Apple M4: data preparation several hours (network bound),
# training ~7.4 h, multilingual verdict ~1.8 h.
set -euo pipefail
cd "$(dirname "$0")"
export AYAZ_ASR_WORK="${AYAZ_ASR_WORK:-$PWD/work}"
W="$AYAZ_ASR_WORK"
PY="${PY:-python3}"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 \
       TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$W/numba_cache"
BASE="${AYAZ_ASR_BASE_NEMO:-$W/models/nemotron-3.5-asr-streaming-0.6b.nemo}"
BASE_SHA=210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74
GUARD=de_de,en_us,fr_fr,ru_ru,ja_jp,ro_ro,et_ee,bg_bg
shaof() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1; }
mkdir -p "$W/models" "$W/datasets/fleurs/tr_tr"

echo "=== 1. base model and environment check"
[ -f "$BASE" ] || { echo "STOP: put the .nemo of nvidia/nemotron-3.5-asr-streaming-0.6b at $BASE (or set AYAZ_ASR_BASE_NEMO)"; exit 1; }
[ "$(shaof "$BASE")" = "$BASE_SHA" ] || echo "! base model SHA-256 differs from the one used for v0.1 ($BASE_SHA)"
"$PY" train/train.py env

echo "=== 2. YODAS Turkish (index, download)"
"$PY" data/yodas_index.py
"$PY" data/yodas_download.py

echo "=== 3. FLEURS-tr test + YODAS dev/pool split"
for f in test.tsv audio/test.tar.gz; do
  [ -s "$W/datasets/fleurs/tr_tr/$(basename "$f")" ] || \
    curl -sSL --fail -o "$W/datasets/fleurs/tr_tr/$(basename "$f")" "https://huggingface.co/datasets/google/fleurs/resolve/main/data/tr_tr/$f"
done
"$PY" data/prepare_eval_sets.py

echo "=== 4. Turkish manifest"
"$PY" data/build_turkish_manifest.py ${INHOUSE_SPEECH:+--speech "$INHOUSE_SPEECH"} ${INHOUSE_EMPTY:+--empty "$INHOUSE_EMPTY"} \
  ${INHOUSE_NOISE:+--noise "$INHOUSE_NOISE"}

echo "=== 5. base model on 100 FLEURS test recordings per language (needed by the rehearsal label quality check)"
"$PY" eval/fleurs_multilingual.py download --parallel 3
"$PY" eval/fleurs_multilingual.py decode

echo "=== 6. rehearsal (download, label with the base model, filter)"
# shellcheck disable=SC2086
"$PY" data/build_rehearsal.py download --guard-langs "$GUARD" ${REHEARSAL_ARGS:-}
# shellcheck disable=SC2086
"$PY" data/build_rehearsal.py label --guard-langs "$GUARD" --turkish-manifest "$W/manifests/turkish.jsonl" \
  --max-label-tokens 127 ${REHEARSAL_ARGS:-}
"$PY" data/filter_rehearsal.py

echo "=== 7. training mixture"
EXTRA_ARGS=()
for e in ${INHOUSE_EXTRA:-}; do EXTRA_ARGS+=(--extra "$e"); done
"$PY" data/build_mixture.py ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== 8. training (9000 steps, measurement every 1000)"
"$PY" -u train/train.py run \
  --indomain-test "${INDOMAIN_TEST:-$W/manifests/yodas_dev.jsonl}" \
  ${INDOMAIN_AUX_TEST:+--indomain-aux-test "$INDOMAIN_AUX_TEST"} ${SHORT_TEST:+--short-test "$SHORT_TEST" --short-test-padding} \
  ${ECHO_TEST:+--echo-test "$ECHO_TEST"} ${NOISE_TEST:+--noise-test "$NOISE_TEST"} ${EXTRA_TEST:+--extra-test "$EXTRA_TEST"}

echo "=== 9. multilingual verdict on the whole FLEURS test split"
SUMMARY=$(ls -t "$W"/train/summary_*.json | head -1)   # newest summary (file names have no spaces)
PICK='import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
if not s.get("model"):
    sys.exit("STOP: training saved no model")
step = s["best_step"]
m = next(o for o in s["measurements"] if o.get("step") == step)   # interim measurement of the selected step
print(s["model"], step, m["fleurs_tr_sub"], sep="\t")'
OUT_LINE=$("$PY" -c "$PICK" "$SUMMARY")
IFS=$'\t' read -r MODEL STEP SUB <<< "$OUT_LINE"   # tab-separated: the model path may contain spaces
NAME="ft_$(basename "$MODEL" .nemo | sed 's/^model_//' | tr -cd 'A-Za-z0-9_.-')"
"$PY" eval/fleurs_multilingual.py download --full --parallel 3
"$PY" -u eval/fleurs_multilingual.py decode --full --model "$NAME|step $STEP|$MODEL|1|$SUB"
"$PY" eval/fleurs_multilingual.py summary --full --model "$NAME|step $STEP|$MODEL|1|$SUB" --verdict "$NAME"

echo "=== done · model: $MODEL"
echo "    export: REFERENCE=<official sherpa-onnx package> bash export/export_sherpa_onnx.sh \"$MODEL\" \"$W/export/sherpa-onnx-int8-560ms\""
