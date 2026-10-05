#!/usr/bin/env bash
# Run the cross-model experiment set into a new output folder.
#
#   export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...   (and LLM_EXTRA_JSON if needed)
#   OUTDIR=runs_new ./run_paper_set.sh main       # 16 cells, 9,504 calls
#   OUTDIR=runs_new ./run_paper_set.sh seedrep    # 20 cells, 11,880 calls
#   OUTDIR=runs_new ./run_paper_set.sh regen      # stochastic regeneration, 3,960 calls
#   OUTDIR=runs_new ./run_paper_set.sh t1         # temperature-1 resampling, 15,840 calls
#
# main is exactly the 16 cells reported per replication model in the paper:
#   judgment grid (4), rule learned with and without informative proxies (2),
#   dimension-by-structure interaction (4), dose-response beta levels (6).
# Every run writes to $OUTDIR, never to results/, so stored records are never
# resumed into. --resume only continues this OUTDIR.
set -euo pipefail
cd "$(dirname "$0")"

TIER="${1:?usage: OUTDIR=... ./run_paper_set.sh main|seedrep|regen|t1}"
OUTDIR="${OUTDIR:?set OUTDIR to a new folder name}"
WORKERS="${WORKERS:-8}"
MODEL_DIR="${LLM_MODEL//\//_}"

FAILED=0
cell() {
  echo ""; echo "=== $* ==="
  python3 run_cell.py --workers "$WORKERS" --resume --outdir "$OUTDIR" "$@" \
    || { echo "!! FAILED: $*"; FAILED=$((FAILED+1)); }
}
finish() {
  if [ "$FAILED" -gt 0 ]; then
    echo ""; echo "!! $FAILED cell(s) FAILED -- run the same command again to resume"; exit 1
  fi
  echo ""; echo "$TIER done -> $OUTDIR/$MODEL_DIR"
}
trap finish EXIT

case "$TIER" in
  main)
    for SEM in neutral social; do
      cell --k 12 --corner LH --semantic "$SEM" --rule absent
      cell --k 12 --corner LH --semantic "$SEM" --rule absent --informative
      cell --k 12 --corner LH --semantic "$SEM" --rule learned
      for B in 0.30 0.45 0.60; do
        cell --k 12 --corner LH --semantic "$SEM" --rule learned --beta "$B"
      done
    done
    for K in 6 18; do
      for CORNER in LH HH; do
        cell --k "$K" --corner "$CORNER" --semantic neutral --rule learned
      done
    done
    ;;
  seedrep)
    for SEED in 20001 20002; do
      for SEM in neutral social; do
        cell --k 12 --corner LH --semantic "$SEM" --rule absent --seed "$SEED"
        cell --k 12 --corner LH --semantic "$SEM" --rule learned --seed "$SEED"
        for B in 0.30 0.45 0.60; do
          cell --k 12 --corner LH --semantic "$SEM" --rule learned --beta "$B" --seed "$SEED"
        done
      done
    done
    ;;
  regen)
    for B in "" "--beta 0.60"; do
      echo ""; echo "=== stochastic regeneration $B ==="
      python3 stochastic_regeneration.py --k 12 --corner LH --semantic neutral \
        --rule learned $B --workers "$WORKERS" --resume --outdir "$OUTDIR" \
        || { echo "!! FAILED: regen $B"; FAILED=$((FAILED+1)); }
    done
    ;;
  t1)
    for B in "" "--beta 0.60"; do
      echo ""; echo "=== temperature-1 resampling $B ==="
      python3 temperature_resampling.py --k 12 --corner LH --semantic neutral \
        --rule learned $B --workers "$WORKERS" --resume --outdir "$OUTDIR" \
        || { echo "!! FAILED: t1 $B"; FAILED=$((FAILED+1)); }
    done
    ;;
  *)
    echo "unknown tier: $TIER"; exit 2 ;;
esac
