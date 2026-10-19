#!/usr/bin/env python3
"""Pool the two GPT-5.6 Terra representation-control runs.

GPT-5.6 Terra uses default sampling, so the representation control was run
twice. This script averages the two runs within pair, arm and presentation
order, then computes the ordered-label and arbitrary-label effects and their
paired difference with pair-bootstrap intervals.

    python3 code/analyze_gpt_rep.py
    python3 code/analyze_gpt_rep.py --run1 data/encoding_control/gpt-5.6-terra/run1 \
        --run2 data/encoding_control/gpt-5.6-terra/run2

The records are the released per-call files for the two runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("cf_a1", "cf_a0")
TAG = "catctl_k12_LH_neutral_learned_zero"


def load(run: Path, render: str) -> pd.DataFrame:
    d = pd.read_csv(run / f"{TAG}_{render}_records.csv")
    d = d[d.arm.isin(ARMS) & d.picked_i.notna()].copy()
    d["picked_i"] = d.picked_i.astype(float)
    return d


def per_pair(d: pd.DataFrame) -> pd.Series:
    w = d.pivot_table(index=["i", "j"], columns="arm", values="picked_i",
                      aggfunc="mean")
    return (w["cf_a1"] - w["cf_a0"]).dropna()


def pooled(run1: Path, run2: Path, render: str) -> pd.Series:
    d = pd.concat([load(run1, render), load(run2, render)], ignore_index=True)
    g = d.groupby(["i", "j", "arm", "order"], as_index=False)["picked_i"].mean()
    return per_pair(g)


def boot(v: np.ndarray, draws: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    bs = [100 * v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)]
    return 100 * v.mean(), *np.percentile(bs, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--run1", type=Path,
                    default=root / "data/encoding_control/gpt-5.6-terra/run1")
    ap.add_argument("--run2", type=Path,
                    default=root / "data/encoding_control/gpt-5.6-terra/run2")
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    ordinal = pooled(args.run1, args.run2, "ordinal")
    nominal = pooled(args.run1, args.run2, "nominal")
    common = ordinal.index.intersection(nominal.index)
    diff = (ordinal.loc[common] - nominal.loc[common]).to_numpy()

    out = {}
    for name, values in (("ordinal", ordinal), ("nominal", nominal),
                         ("ordinal_minus_nominal", pd.Series(diff, index=common))):
        est, lo, hi = boot(values.to_numpy(), args.draws)
        out[name] = {"effect_pp": round(est, 2), "ci_lo": round(lo, 2),
                     "ci_hi": round(hi, 2), "pairs": int(len(values))}
        print(f"{name:24s} {est:+7.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]  "
              f"pairs {len(values)}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
