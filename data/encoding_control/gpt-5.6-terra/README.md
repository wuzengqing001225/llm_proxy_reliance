# GPT-5.6 Terra representation control (two runs)

This folder holds the per-call records for the representation control
(ordered labels versus arbitrary labels) run on GPT-5.6 Terra.

- Model: `gpt-5.6-terra`, default sampling (temperature not sent), reasoning
  enabled, forced `choose` tool.
- Cell: k = 12, LH, neutral names, learned rule, zero-information proxies.
- Run 1: 2026-10-19, 396 distinct prompts per rendering, no missing calls.
- Run 2: 2026-10-19/20, 396 distinct prompts per rendering, one missing
  nominal call (pair i=1533, j=3021, order 1, arm cf_a1).
- Each rendering has 99 pairs, two counterfactual arms and two presentation
  orders. The two runs use the same prompts.

## Estimates

| Rendering | Run 1 (pp) | Run 2 (pp) | Pooled (pp) [95% CI] |
|---|---|---|---|
| Ordered labels | +4.04 | +10.61 | +7.32 [+4.29, +10.35] |
| Arbitrary labels | +4.55 | -0.51 | +2.02 [-1.77, +5.81] |
| Ordered minus arbitrary | -0.51 | +11.11 | +5.30 [+0.51, +10.10] |

The pooled estimate averages the two runs within pair, arm and presentation
order, then bootstraps over pairs. Excluding the pair with the missing call
gives +5.36 pp [+0.51, +10.20] on 98 pairs.

## Reproduce

From the repository root:

```
python3 code/analyze_gpt_rep.py
```

The command reads `run1/` and `run2/` in this folder and prints the pooled
effects. Add `--json out.json` to write them.

## Files

- `run1/catctl_k12_LH_neutral_learned_zero_{ordinal,nominal}_records.csv`
- `run1/catctl_k12_LH_neutral_learned_zero_{ordinal,nominal}_metadata.json`
- `run2/...` (same names, second run)
