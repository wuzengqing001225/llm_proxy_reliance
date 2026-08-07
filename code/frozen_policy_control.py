#!/usr/bin/env python3
"""Frozen-policy control for the dose-response accuracy comparison.

Raising the proxies' predictive value also moves the true ranking, so a model
that never changed its decisions would score better at higher evidence levels.
This script decomposes each model's observed accuracy change between the
weakest (beta = 0) and strongest (beta = 0.60) level into

    observed improvement  = mechanical component + evidence-specific residual

where the mechanical component scores the model's FROZEN beta=0 choices
against the beta=0.60 truth, and the residual is the per-pair difference
between the frozen policy's error and the actual beta=0.60 policy's error,
both under the beta=0.60 truth. Positive residual = the policy the model
actually uses at high evidence beats its own frozen low-evidence policy.

    python frozen_policy_control.py --results results/deepseek-chat [more...]

Writes frozen_policy_residuals.csv next to the script.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k, dgp_norm, config  # noqa: E402


def choices(df: pd.DataFrame) -> dict:
    d = df[(df.arm == "factual") & df.picked_i.notna()]
    return {(int(i), int(j)): g.picked_i.astype(int).mean()
            for (i, j), g in d.groupby(["i", "j"])}


def err(p: float, ij: tuple, r_true: np.ndarray) -> float:
    i, j = ij
    t = r_true[i] >= r_true[j]
    return p * (1 - t) + (1 - p) * t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    ref0 = dgp_k.generate("low", "high", k=12, n=5000, seed=config.TARGET_SEED)
    ref60 = dgp_norm.generate_normalized("low", "high", k=12, n=5000,
                                         seed=config.TARGET_SEED, beta_info=0.60)
    rng = np.random.default_rng(9)
    rows = []
    for base in args.results:
        model = os.path.basename(base.rstrip("/"))
        for sem in ("neutral", "social"):
            try:
                c0 = choices(pd.read_csv(
                    f"{base}/k12_LH_{sem}_learned_zero_records.csv"))
                c60 = choices(pd.read_csv(
                    f"{base}/k12_LH_{sem}_learned_info_b060_records.csv"))
            except FileNotFoundError:
                continue
            keys = sorted(set(c0) & set(c60))
            obs = np.array([err(c0[k], k, ref0.r_true)
                            - err(c60[k], k, ref60.r_true) for k in keys])
            mech = np.array([err(c0[k], k, ref0.r_true)
                             - err(c0[k], k, ref60.r_true) for k in keys])
            res = np.array([err(c0[k], k, ref60.r_true)
                            - err(c60[k], k, ref60.r_true) for k in keys])
            bs = [rng.choice(res, len(res)).mean() for _ in range(args.boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(model=model, semantic=sem, n_pairs=len(keys),
                             observed_improvement_pp=obs.mean() * 100,
                             mechanical_component_pp=mech.mean() * 100,
                             residual_pp=res.mean() * 100,
                             residual_lo=lo * 100, residual_hi=hi * 100,
                             residual_excludes_0=bool(lo > 0 or hi < 0)))
            r = rows[-1]
            print(f"{model:<20}{sem:<9} observed {r['observed_improvement_pp']:+6.2f} "
                  f"= mechanical {r['mechanical_component_pp']:+6.2f} "
                  f"+ residual {r['residual_pp']:+6.2f} "
                  f"[{lo*100:+.2f}, {hi*100:+.2f}]")
    out = os.path.join(_HERE, "frozen_policy_residuals.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
