#!/usr/bin/env bash
# Run the readable-order (encoding) control on one model through OpenRouter.
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   ./code/run_openrouter_display.sh <model_slug> <outdir> <numeric_reference_csv>
#
# Examples:
#   ./code/run_openrouter_display.sh \
#     deepseek/deepseek-v4-flash-0731 \
#     data/encoding_control/openrouter_deepseek \
#     data/raw_records/DeepSeek-V4-Flash-0731/k12_LH_neutral_learned_zero_records.csv
#
#   ./code/run_openrouter_display.sh \
#     qwen/qwen3.7-max \
#     data/encoding_control/openrouter_qwen \
#     data/raw_records/qwen3.7-max/k12_LH_neutral_learned_zero_records.csv
#
# Smoke test first (two calls, then stop):
#   SMOKE=1 ./code/run_openrouter_display.sh <model> <outdir> <numeric_ref>
#   RESUME=1 ./code/run_openrouter_display.sh <model> <outdir> <numeric_ref>
#
# Environment:
#   OPENROUTER_API_KEY  required
#   WORKERS             parallel requests (default 6)
#   RESUME=1            continue a partial/smoke run
#   SMOKE=1             send only SMOKE_LIMIT prompts, then exit
#   LLM_EXTRA_JSON      provider-specific payload, e.g. to disable thinking
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:?usage: $0 <model_slug> <outdir> <numeric_reference_csv>}"
OUTDIR="${2:?usage: $0 <model_slug> <outdir> <numeric_reference_csv>}"
NUMREF="${3:?usage: $0 <model_slug> <outdir> <numeric_reference_csv>}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Set OPENROUTER_API_KEY=sk-or-v1-... (https://openrouter.ai/keys)" >&2
  exit 1
fi
[[ -f "$NUMREF" ]] || { echo "numeric reference not found: $NUMREF" >&2; exit 1; }

export LLM_PROVIDER=openai
export LLM_BASE_URL="${LLM_BASE_URL:-https://openrouter.ai/api/v1}"
export LLM_API_KEY="$OPENROUTER_API_KEY"
export LLM_MODEL="$MODEL"
# Both models default to reasoning on OpenRouter. Turn it off so the forced
# choose tool gets the whole (small) completion budget. Override if needed.
if [[ -z "${LLM_EXTRA_JSON:-}" ]]; then
  export LLM_EXTRA_JSON='{"reasoning": {"enabled": false}}'
fi

WORKERS="${WORKERS:-6}"
RESUME_ARG=""
[[ "${RESUME:-0}" == "1" ]] && RESUME_ARG="--resume"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "SMOKE: 2 calls to $MODEL ..."
  python3 code/categorical_proxy_control.py run \
    --render ordinal --outdir "$OUTDIR" --workers 1 \
    --limit "${SMOKE_LIMIT:-2}" $RESUME_ARG
  echo "smoke done. Check $OUTDIR/*/*_ordinal_records.csv and *_metadata.json."
  echo "Then run the full pass with RESUME=1."
  exit 0
fi

echo "== $MODEL: ordinal rendering (396 distinct prompts)"
python3 code/categorical_proxy_control.py run \
  --render ordinal --outdir "$OUTDIR" --workers "$WORKERS" $RESUME_ARG

echo "== $MODEL: arbitrary-label rendering (396 distinct prompts)"
python3 code/categorical_proxy_control.py run \
  --render nominal --outdir "$OUTDIR" --workers "$WORKERS" $RESUME_ARG

MDIR="${MODEL//\//_}"
echo "== analyze $MDIR"
python3 code/categorical_proxy_control.py analyze \
  --outdir "$OUTDIR" --model "$MDIR" --numeric-reference "$NUMREF"
