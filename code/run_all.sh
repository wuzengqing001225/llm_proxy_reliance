#!/usr/bin/env bash
# Replication battery. Runs the cells that carry the paper's three main claims.
#
#   export LLM_BASE_URL=https://api.deepseek.com/v1
#   export LLM_API_KEY=sk-...
#   export LLM_MODEL=deepseek-chat
#   ./run_all.sh                # full battery
#   ./run_all.sh core           # claim 1 + claim 2 only (6 cells, 3,564 calls)
#   ./run_all.sh interaction    # + dimension & structure axes (4 cells, 2,376 calls)
#   ./run_all.sh beta           # + proxy-strength dose-response (6 cells, 3,564 calls;
#                                 #   rule-learned prompts -> ~7.2k chars each)
#
# Every cell is 594 calls (99 pairs x 2 orders x 3 arms).
set -euo pipefail
cd "$(dirname "$0")"

TIER="${1:-full}"
WORKERS="${WORKERS:-8}"
COMMON="--workers $WORKERS --resume"

FAILED=0
run() {
  echo ""
  echo "=== $* ==="
  python run_cell.py $COMMON "$@" || { echo "!! FAILED: $* (continuing)"; FAILED=$((FAILED+1)); }
}
finish() {
  if [ "$FAILED" -gt 0 ]; then
    echo ""; echo "!! $FAILED cell(s) FAILED -- do not treat this battery as complete"
    exit 1
  fi
}
trap finish EXIT

# ---- Claim 1: justified-use benchmark -------------------------------------
# proxies informative vs not, at fixed dimension -- the judgement grid.
for SEM in neutral social; do
  run --k 12 --corner LH --semantic "$SEM" --rule absent
  run --k 12 --corner LH --semantic "$SEM" --rule absent --informative
done

# ---- Claim 2: the reversal (no cases -> cases) ----------------------------
for SEM in neutral social; do
  run --k 12 --corner LH --semantic "$SEM" --rule learned
done

if [ "$TIER" = "core" ]; then
  echo ""; echo "core tier done"; python analyze.py --results "results/${LLM_MODEL//\//_}"; exit 0
fi

# ---- interaction add-on: the minimal cross-model test of (a) the dimension
#      effect on G_ICL and (b) the dimension x structure interaction.
#      4 cells, 2,376 calls. Neutral only: the fairness heuristic is known to be
#      model-specific, so social arms would confound the interaction readout.
# ---- beta dose-response: does proxy use TRACK the justified level?
#      RULE-LEARNED is constitutive, not a choice: beta changes only the weights
#      w_P (and hence r_true / Y), never the attribute values X. A rule-absent
#      prompt shows only X, so it is BYTE-IDENTICAL across beta levels -- the
#      manipulation would not reach the model at all. The calibration set's Y
#      labels are the only channel beta has.
#
#      Uses the VARIANCE-NORMALIZED generator (modules/dgp_norm.py): total risk
#      variance is held fixed so ranking difficulty does not drift with beta.
#      Levels are evenly spaced on the IDEAL-LEARNER line (the reference the
#      model should track), not the complete-knowledge Bayes line:
#        beta 0.00 / 0.30 / 0.45 / 0.60 -> ideal line 0.0 / 4.0 / 13.1 / 23.2 pp
#        at the frozen seed 10001 (multi-seed means -0.3 / 6.7 / 15.5 / 24.6).
#      beta=0 is the core tier's rule-learned ZERO-INFO cell -- bit-identical,
#      reuse it. The old frozen beta=0.40 informative cells (Gate 1 or Gate 2)
#      are NOT points on this curve: different parameterization, do not pool.
if [ "$TIER" = "beta" ]; then
  for B in 0.30 0.45 0.60; do
    for SEM in neutral social; do
      run --k 12 --corner LH --semantic "$SEM" --rule learned --beta "$B"
    done
  done
  echo ""; echo "beta tier done (beta=0 = core tier rule-learned zero-info cell)"
  python analyze.py --results "results/${LLM_MODEL//\//_}"; exit 0
fi

# ---- seed replication: same design, fresh randomization ------------------
#      Redraws population + evaluation pairs + calibration sample together
#      (--seed S derives all three). Covers the two headline results: the
#      zero-info judgment-grid cells and the full dose-response.
#      10 cells x 594 = 5,940 calls per model per seed.
if [ "$TIER" = "seedrep" ]; then
  for SEED in 20001 20002; do
    for SEM in neutral social; do
      run --k 12 --corner LH --semantic "$SEM" --rule absent --seed "$SEED"
      run --k 12 --corner LH --semantic "$SEM" --rule learned --seed "$SEED"
      for B in 0.30 0.45 0.60; do
        run --k 12 --corner LH --semantic "$SEM" --rule learned --beta "$B" --seed "$SEED"
      done
    done
  done
  echo ""; echo "seedrep tier done"
  python analyze.py --results "results/${LLM_MODEL//\//_}"; exit 0
fi

if [ "$TIER" = "interaction" ]; then
  for K in 6 18; do
    for CORNER in LH HH; do
      run --k "$K" --corner "$CORNER" --semantic neutral --rule learned
    done
  done
  echo ""; echo "interaction tier done (run core first if you have not)"
  python analyze.py --results "results/${LLM_MODEL//\//_}"; exit 0
fi

# ---- Claim 3: dimension harms structure discovery, not rule execution -----
for K in 6 18; do
  for SEM in neutral social; do
    run --k "$K" --corner LH --semantic "$SEM" --rule absent
    run --k "$K" --corner LH --semantic "$SEM" --rule learned
  done
done

# ---- Structure axis: does a strongly dependent (non-orthogonal) structure
#      change the picture? HH = high collinearity, high protected alignment.
for K in 6 18; do
  for SEM in neutral social; do
    run --k "$K" --corner HH --semantic "$SEM" --rule learned
  done
done

echo ""
echo "battery complete"
python analyze.py --results "results/${LLM_MODEL//\//_}"
