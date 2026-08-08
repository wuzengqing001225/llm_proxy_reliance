#!/usr/bin/env python3
"""Pooled dimension-by-dependence interaction across the identically configured models.

    python pooled_interaction.py --records ../data/raw_records/claude-sonnet-4-5 \
                                 ../data/raw_records/qwen3.7-max \
                                 ../data/raw_records/deepseek-chat

The interaction asks whether the effect of attribute dimension on proxy
reliance depends on how entangled the attributes are:

    I_model = [PSE(k=18, HH) - PSE(k=6, HH)] - [PSE(k=18, LH) - PSE(k=6, LH)]

evaluated on the neutral, rule-learned, zero-information cells.  A negative
I means that adding attributes raises reliance under low dependence (LH) and
lowers it under high dependence (HH), which is the sign reversal reported in
the main text.

The pooled estimate is the unweighted mean of the per-model interactions.
The interval resamples PAIRS independently within each model-cell, which is
the level at which the observations are dependent: the two orders of one pair
share a patient draw, so pairs and not calls are the resampling unit.

SCOPE
    Only the models that run under one serving configuration belong in the
    pool.  GPT-5.6-terra does not accept a temperature setting and is reported
    as a separate channel throughout the paper, so it is excluded here as well.
    Pass it explicitly if you want to see what including it does.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from analyze import per_pair_pse  # noqa: E402

CELLS = [(18, "HH", +1), (6, "HH", -1), (18, "LH", -1), (6, "LH", +1)]


def cell_pairs(records_dir: str, k: int, corner: str) -> dict:
    """Per-pair PSE for one neutral, rule-learned, zero-information cell."""
    stem = f"k{k}_{corner}_neutral_learned_zero_records.csv"
    path = os.path.join(records_dir, stem)
    if not os.path.exists(path):
        raise SystemExit(f"missing cell: {path}")
    df = pd.read_csv(path)
    return per_pair_pse(df[df["arm"].isin(["cf_a1", "cf_a0"])])


def model_interaction(pairs_by_cell: dict) -> float:
    return sum(sign * 100 * np.mean(list(pairs_by_cell[(k, c)].values()))
               for k, c, sign in CELLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True,
                    help="one raw-records directory per model")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_model = {}
    for d in args.records:
        name = os.path.basename(os.path.normpath(d))
        per_model[name] = {(k, c): cell_pairs(d, k, c) for k, c, _ in CELLS}
        print(f"{name:>22}: interaction = "
              f"{model_interaction(per_model[name]):+.2f} pp")

    point = float(np.mean([model_interaction(p) for p in per_model.values()]))

    rng = np.random.default_rng(args.seed)
    draws = np.empty(args.B)
    for b in range(args.B):
        vals = []
        for pairs_by_cell in per_model.values():
            tot = 0.0
            for k, c, sign in CELLS:
                v = np.fromiter(pairs_by_cell[(k, c)].values(), dtype=float)
                idx = rng.integers(0, len(v), len(v))
                tot += sign * 100 * v[idx].mean()
            vals.append(tot)
        draws[b] = np.mean(vals)
    lo, hi = np.percentile(draws, [2.5, 97.5])

    print(f"\npooled over {len(per_model)} models: {point:+.2f} pp "
          f"[{lo:+.2f}, {hi:+.2f}]  (B={args.B}, seed={args.seed})")
    print("negative means the dimension effect reverses sign with dependence")


if __name__ == "__main__":
    main()
