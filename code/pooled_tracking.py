#!/usr/bin/env python3
"""Confidence bounds for the dose-response tracking slope.

    # once, builds the Sonnet canonical records from the raw files
    python prepare_canonical.py --raw ../data/raw_records/claude-sonnet-4-5 \
                                --out ../data/canonical/claude-sonnet-4-5
    python pooled_tracking.py \
        --summary  ../data/summaries/beta_four_model_summary.csv \
        --sonnet   ../data/canonical/claude-sonnet-4-5 \
        --deepseek ../data/raw_records/DeepSeek-V4-Flash-0731 \
        --qwen     ../data/raw_records/qwen3.7-max \
        --gpt      ../data/raw_records/gpt-5.6-terra

For every model and label condition the slope gamma of
PSE_model = alpha + gamma * PSE_ideal is fitted by ordinary least squares over
the four evidence levels.

Resampling. The 99 evaluation pairs are the same at every evidence level, in
every model and under both label conditions, so one bootstrap draw resamples
pair identities once and applies the same draw to all eight fits. This keeps
the dependence that shared evaluation items induce between fits, which
matters for the pooled slope. Each fit uses every pair it resolves at all four
levels, so point estimates match the per-fit values reported in the paper. A
pair a fit does not resolve (two pairs in one Sonnet fit) is simply absent
from that fit's resample.

Reported, without reference to any margin:
  * per fit: gamma and its two-sided 95% interval. Its upper end is the
    one-sided 97.5% upper confidence bound for that fit alone.
  * per fit: a simultaneous one-sided 95% upper bound, valid for all eight
    fits jointly, from the 95th percentile of the maximum standardized
    bootstrap deviation across fits.
  * pooled slope (unweighted mean of fits), over the three identically
    configured models and over all four: gamma, the two-sided 95% interval and
    the two-sided 90% interval, whose ends are the one-sided 95% bounds.
A reader can compare these bounds with any margin of practical interest. The
script also prints, for reference, whether the 90% interval lies inside
+/- --margin, which is the two-one-sided-tests equivalence criterion at
alpha = 0.05.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd

LEVELS = [0.0, 0.30, 0.45, 0.60]


def per_pair(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"missing records: {path}\n"
                         "(for Sonnet, run: python prepare_canonical.py --raw ../data/raw_records/claude-sonnet-4-5 --out ../data/canonical/claude-sonnet-4-5)")
    acc = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["arm"] not in ("cf_a1", "cf_a0") or r["picked_i"] == "":
                continue
            acc[(r["arm"], (int(r["i"]), int(r["j"])))].append(
                int(float(r["picked_i"])))
    keys = {k for _, k in acc}
    return {k: np.mean(acc[("cf_a1", k)]) - np.mean(acc[("cf_a0", k)])
            for k in keys if ("cf_a1", k) in acc and ("cf_a0", k) in acc}


def cell_path(root: str, beta: float, sem: str) -> str:
    if beta == 0.0:
        return os.path.join(root, f"k12_LH_{sem}_learned_zero_records.csv")
    return os.path.join(
        root, f"k12_LH_{sem}_learned_info_b{int(round(beta * 100)):03d}_records.csv")


def slope(ideal: np.ndarray, M: np.ndarray) -> float:
    return float(np.polyfit(ideal, M.mean(1), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--sonnet", required=True)
    ap.add_argument("--deepseek", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.15)
    args = ap.parse_args()

    S = pd.read_csv(args.summary)
    roots = {"sonnet": args.sonnet, "deepseek": args.deepseek,
             "qwen": args.qwen, "gpt5.6-terra": args.gpt}

    raw, ideals = {}, {}
    for model, root in roots.items():
        for sem in ("neutral", "social"):
            raw[(model, sem)] = [per_pair(cell_path(root, b, sem)) for b in LEVELS]
            ideals[(model, sem)] = (S[(S.model == model) & (S.semantic == sem)]
                                    .sort_values("beta").ideal_pp.to_numpy())
    universe = sorted(set.union(*[set(d) for v in raw.values() for d in v]))
    pos = {p: t for t, p in enumerate(universe)}
    mats, cols = {}, {}
    for k, v in raw.items():
        own = sorted(set.intersection(*[set(d) for d in v]))
        mats[k] = 100 * np.array([[d[p] for p in own] for d in v])
        # column of each universe pair in this fit, or -1 if unresolved
        col = np.full(len(universe), -1)
        for c, p in enumerate(own):
            col[pos[p]] = c
        cols[k] = col
    print(f"evaluation pairs: {len(universe)} shared identities, "
          f"{min(m.shape[1] for m in mats.values())} to "
          f"{max(m.shape[1] for m in mats.values())} resolved per fit")

    point = {k: slope(ideals[k], M) for k, M in mats.items()}
    rng = np.random.default_rng(args.seed)
    n = len(universe)
    boots = {k: np.empty(args.B) for k in mats}
    for b in range(args.B):
        draw = rng.integers(0, n, n)           # one draw shared by all fits
        for k, M in mats.items():
            c = cols[k][draw]
            boots[k][b] = slope(ideals[k], M[:, c[c >= 0]])

    # simultaneous one-sided 95% upper bounds across the eight fits, from the
    # maximum of the standardized bootstrap deviations (shared draws keep the
    # dependence between fits)
    sd = {k: boots[k].std(ddof=1) for k in mats}
    tmax = np.max(np.column_stack([(boots[k] - point[k]) / sd[k] for k in mats]),
                  axis=1)
    q = np.percentile(tmax, 95)

    for k in mats:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        print(f"{k[0]:13s} {k[1]:7s} gamma {point[k]:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]"
              f"  simultaneous one-sided 95% upper {point[k] + q * sd[k]:+.3f}")

    for label, keys in (("three identically configured models (6 fits)",
                         [k for k in mats if k[0] != "gpt5.6-terra"]),
                        ("all four models (8 fits)", list(mats))):
        g = np.mean([point[k] for k in keys])
        bs = np.mean([boots[k] for k in keys], axis=0)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        l90, h90 = np.percentile(bs, [5, 95])
        inside = (l90 > -args.margin) and (h90 < args.margin)
        print(f"pooled, {label}: gamma {g:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  "
              f"90% [{l90:+.3f}, {h90:+.3f}]  "
              f"(90% interval inside +/-{args.margin}: {inside})")


if __name__ == "__main__":
    main()
