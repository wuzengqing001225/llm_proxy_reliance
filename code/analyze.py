#!/usr/bin/env python3
"""Analyse recorded cells: capability, proxy effect, and the justified benchmark.

    python analyze.py --results results/DeepSeek-V4-Flash-0731

Recomputes every quantity from the raw records -- nothing is taken on trust from
the run. Confidence intervals are pair-level bootstrap (the pair is the
resampling unit, because both presentation orders of one pair are dependent).
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k
import dgp_norm                      # noqa: E402
import oracle                     # noqa: E402
sys.modules["dgp"] = dgp_k

import config                     # noqa: E402
from run_cell import build_cell, derived_seeds   # noqa: E402

from run_cell import CELL_TAG_RE as CELL_RE   # single source of truth


def per_pair_error(df: pd.DataFrame, r_true: np.ndarray) -> dict:
    """Pairwise error rate per pair, averaged over presentation orders."""
    out = {}
    d = df[df["picked_i"].notna()]
    for (i, j), g in d.groupby(["i", "j"]):
        truth_i = r_true[int(i)] >= r_true[int(j)]
        err = [(bool(v) != truth_i) for v in g["picked_i"].astype(int)]
        out[(int(i), int(j))] = float(np.mean(err))
    return out


def per_pair_pse(df_cf: pd.DataFrame) -> dict:
    """Directed PSE per pair: P(pick i | A=1) - P(pick i | A=0)."""
    out = {}
    d = df_cf[df_cf["picked_i"].notna()]
    for (i, j), g in d.groupby(["i", "j"]):
        a1 = g[g["arm"] == "cf_a1"]["picked_i"].astype(int)
        a0 = g[g["arm"] == "cf_a0"]["picked_i"].astype(int)
        if len(a1) and len(a0):
            out[(int(i), int(j))] = float(a1.mean() - a0.mean())
    return out


def boot_ci(vals: np.ndarray, B: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(B)])
    return tuple(np.percentile(bs, [2.5, 97.5]))


def analyse_cell(path: str) -> dict | None:
    m = CELL_RE.search(os.path.basename(path))
    if not m:
        return None
    k, corner, semantic, rule, info = int(m[1]), m[2], m[3], m[4], m[5] == "info"
    beta = None if m.lastindex is None or m[6] is None else int(m[6]) / 100.0
    seed = None if m.lastindex is None or m.lastindex < 7 or m[7] is None else int(m[7])

    df = pd.read_csv(path)
    ref, pairs, Xcal, Ycal, r_ideal = build_cell(k, corner, semantic, rule, info,
                                                 beta, seed)

    expected = len(pairs) * config.N_ORDERS * 3
    if len(df) < expected:
        print(f"  WARNING {os.path.basename(path)}: {len(df)}/{expected} calls "
              f"recorded -- truncated cell, estimates may be biased "
              f"(truncation is not random)")

    err = per_pair_error(df[df["arm"] == "factual"], ref.r_true)
    pse = per_pair_pse(df[df["arm"].isin(["cf_a1", "cf_a0"])])
    keys = sorted(set(err) & set(pse))
    if not keys:
        return None
    e = np.array([err[p] for p in keys])
    s = np.array([pse[p] for p in keys])

    # Justified level: how much a Bayes-optimal decider's ranking shifts under the
    # same intervention. Uses the DGP's own ground-truth risk (r_true), which
    # already includes the proxy contribution when proxies are informative --
    # do not try to rebuild it from ref.w (that holds legit weights only).
    # Intervention is per-pair on the target index i, matching the LLM arms.
    pse_bayes = 0.0
    pse_ideal_ref = None
    if beta is not None:
        # Two reference lines, not interchangeable. The Ideal learner (same 80
        # calibration cases the model saw) is what the model should track; the
        # complete-knowledge Bayes line is an upper bound only -- scoring against
        # it would read the posterior's finite-sample shrinkage (2-4 pp) as the
        # model failing to track.
        gc, ga = config.CORNERS[corner]
        _, _, d_seed = derived_seeds(seed)
        post = dgp_norm.fit_ideal(gc, ga, k=k, beta_info=beta,
                                  n_cal=config.N_CAL, dcal_seed=d_seed)
        pse_ideal_ref = dgp_norm.ideal_line(post, ref, pairs)
        pse_bayes = dgp_norm.bayes_line(ref, pairs)
    elif info:
        for t, sign in ((1, 1.0), (0, -1.0)):
            picks = []
            for p in pairs:
                newA = ref.A.copy()
                newA[int(p["i"])] = t
                cf = dgp_k.counterfactual(ref, newA)
                picks.append(float(cf.r_true[int(p["i"])] >= cf.r_true[int(p["j"])]))
            pse_bayes += sign * float(np.mean(picks))
        pse_bayes *= 100

    lo_e, hi_e = boot_ci(e, seed=1)
    lo_s, hi_s = boot_ci(s, seed=2)
    total = len(pairs) * config.N_ORDERS
    return dict(
        k=k, corner=corner, semantic=semantic, rule=rule, informative=info,
        n_pairs=len(keys), missing_rate=float(df["picked_i"].isna().mean()),
        R_LLM=e.mean(), R_LLM_lo=lo_e, R_LLM_hi=hi_e,
        R_Ideal=r_ideal,
        G_ICL_pp=(e.mean() - r_ideal) * 100 if r_ideal is not None else np.nan,
        PSE_pp=s.mean() * 100, PSE_lo=lo_s * 100, PSE_hi=hi_s * 100,
        beta=(config.BETA_INFO if (info and beta is None) else beta),
        # Primary reference for beta cells: the Ideal learner given the SAME 80
        # calibration cases. PSE_Bayes_pp is the complete-knowledge upper bound.
        PSE_Ideal_pp=pse_ideal_ref,
        excess_vs_Ideal_pp=(None if pse_ideal_ref is None
                            else s.mean() * 100 - pse_ideal_ref),
        PSE_Bayes_pp=pse_bayes,
        excess_PSE_pp=s.mean() * 100 - pse_bayes,
        calls_expected=total * 3, calls_recorded=len(df),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory of *_records.csv")
    ap.add_argument("--out", default=None, help="summary CSV path")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.results, "*_records.csv"))):
        r = analyse_cell(p)
        if r:
            rows.append(r)
            print(f"  parsed {os.path.basename(p)}")
    if not rows:
        raise SystemExit(f"no parseable cells in {args.results}")

    S = pd.DataFrame(rows).sort_values(["rule", "informative", "k", "corner", "semantic"])
    out = args.out or os.path.join(args.results, "summary.csv")
    S.to_csv(out, index=False)

    show = ["k", "corner", "semantic", "rule", "informative", "beta",
            "R_LLM", "R_Ideal", "G_ICL_pp", "PSE_pp", "PSE_lo", "PSE_hi",
            "PSE_Ideal_pp", "excess_vs_Ideal_pp",
            "PSE_Bayes_pp", "excess_PSE_pp", "missing_rate"]
    print("\n" + S[show].round(3).to_string(index=False))
    print(f"\nwrote {out}")

    # Semantic discount, where both semantics of a condition are present
    print("\nsemantic contrast (social - neutral), per condition:")
    for keyvals, g in S.groupby(["k", "corner", "rule", "informative"]):
        if set(g["semantic"]) == {"social", "neutral"}:
            soc = g[g.semantic == "social"].PSE_pp.iloc[0]
            neu = g[g.semantic == "neutral"].PSE_pp.iloc[0]
            print(f"  k={keyvals[0]:>2} {keyvals[1]} rule={keyvals[2]:<8} "
                  f"info={str(keyvals[3]):<5}  delta = {soc-neu:+6.2f} pp")


if __name__ == "__main__":
    main()
