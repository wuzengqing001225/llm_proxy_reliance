#!/usr/bin/env python3
"""Order-instability robustness for the proxy-specific effect.

    python order_robustness.py --results results/deepseek-chat [more dirs...]

Within-pair order inconsistency (the two presentation orders of a pair getting
different answers) mixes decoding noise with position preference. Three checks
establish that the paired PSE estimator is robust to it:

1. Position bias by arm: P(model picks the first-presented patient), per arm.
   A position preference alone cannot bias the PSE because every pair is
   presented in both orders and the estimator averages them, but the rates are
   reported so the balance argument is inspectable.
2. Consistent-pairs-only sensitivity: recompute the PSE using only pairs whose
   two orders AGREE within each counterfactual arm. If the full-sample PSE
   were driven by order-flipping pairs, it would shrink here.
3. Regression with an explicit position term: a linear probability model
   picked_i ~ arm + order on the two counterfactual arms. With the balanced
   design the arm coefficient must reproduce the paired estimator; reporting
   it makes the adjustment explicit rather than implicit.

Writes order_robustness.csv (one row per cell) next to each results dir's
summary, and prints a per-model digest.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
from run_cell import CELL_TAG_RE  # noqa: E402


def analyse_cell(path: str) -> dict | None:
    m = CELL_TAG_RE.search(os.path.basename(path))
    if not m:
        return None
    df = pd.read_csv(path)
    df = df[df.picked_i.notna()].copy()
    df["picked_i"] = df["picked_i"].astype(int)
    # picked the first-presented patient: order 0 presents i first, order 1
    # presents j first
    df["picked_first"] = np.where(df["order"] == 0, df["picked_i"], 1 - df["picked_i"])

    out = dict(cell=os.path.basename(path).replace("_records.csv", ""))

    # (1) position bias per arm
    for arm in ("factual", "cf_a1", "cf_a0"):
        a = df[df.arm == arm]
        out[f"pos_first_{arm}"] = a.picked_first.mean() if len(a) else np.nan

    cf = df[df.arm.isin(["cf_a1", "cf_a0"])]
    if cf.empty:
        return out

    g1 = cf[cf.arm == "cf_a1"].groupby(["i", "j"])["picked_i"]
    g0 = cf[cf.arm == "cf_a0"].groupby(["i", "j"])["picked_i"]
    m1, m0 = g1.mean(), g0.mean()
    keys = sorted(set(m1.index) & set(m0.index))
    out["pse_full"] = np.mean([m1[k] - m0[k] for k in keys]) * 100
    out["n_pairs"] = len(keys)

    # (2) consistent pairs only: both orders agree within each cf arm
    c1 = g1.nunique()
    c0 = g0.nunique()
    cons = [k for k in keys if c1.get(k, 2) == 1 and c0.get(k, 2) == 1]
    out["n_consistent"] = len(cons)
    out["pse_consistent"] = (np.mean([m1[k] - m0[k] for k in cons]) * 100
                             if cons else np.nan)

    # (3) linear probability model picked_i ~ arm + order (cf arms)
    X = np.column_stack([
        np.ones(len(cf)),
        (cf.arm == "cf_a1").astype(float),
        cf["order"].astype(float),
    ])
    y = cf.picked_i.values.astype(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    out["pse_lpm_position_adjusted"] = coef[1] * 100
    out["position_coef_pp"] = coef[2] * 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True)
    args = ap.parse_args()

    for rdir in args.results:
        rows = [r for p in sorted(glob.glob(os.path.join(rdir, "*_records.csv")))
                if (r := analyse_cell(p))]
        if not rows:
            print(f"{rdir}: no cells")
            continue
        S = pd.DataFrame(rows)
        out = os.path.join(rdir, "order_robustness.csv")
        S.to_csv(out, index=False)
        d = S.dropna(subset=["pse_full"])
        delta = (d.pse_consistent - d.pse_full).abs()
        lpm = (d.pse_lpm_position_adjusted - d.pse_full).abs()
        print(f"\n== {rdir} ({len(d)} cells) -> {out}")
        print(f"  P(pick first) factual: median "
              f"{S.pos_first_factual.median():.3f} "
              f"(range {S.pos_first_factual.min():.3f}-{S.pos_first_factual.max():.3f})")
        print(f"  |PSE consistent-only - full|: median {delta.median():.2f} pp, "
              f"max {delta.max():.2f} pp "
              f"(consistent pairs: median {int(d.n_consistent.median())}/99)")
        print(f"  |PSE lpm+position - full|:    median {lpm.median():.2f} pp, "
              f"max {lpm.max():.2f} pp")
        sign_flips = ((np.sign(d.pse_consistent) != np.sign(d.pse_full))
                      & (d.pse_full.abs() > 5)).sum()
        print(f"  sign flips among |PSE|>5pp cells: {sign_flips}")


if __name__ == "__main__":
    main()
