#!/usr/bin/env python3
"""Dose-response readout: does the model's proxy use TRACK the evidence?

    python dose_response.py --results results/deepseek-chat

Fits, over the beta levels of one semantic condition,

    PSE_model(beta) = alpha + gamma * PSE_Ideal(beta)

and reports gamma with two tests specified in the internal analysis plan: gamma vs 0 (any tracking at
all) and gamma vs 1 (calibrated tracking). The reference is the IDEAL LEARNER
line -- the BLR posterior fitted on the same 80 calibration cases the model saw
-- not the complete-knowledge Bayes line, which a finite-sample learner cannot
reach (shrinkage 2-4 pp) and which would bias gamma downward.

Confidence intervals resample PAIRS, not levels: the evaluation pairs are shared
across beta levels by construction, so redrawing pairs and refitting the whole
curve preserves the pairing that gives this design its power.

READOUT GRID (specified in advance)
    gamma ~ 1, alpha ~ 0   fully evidence-calibrated -- the strongest form of
                           "reasonable integration"
    gamma ~ 1, alpha > 0   tracks evidence but with a constant surplus: the
                           excess is an evidence-INDEPENDENT fixed bias
    gamma > 1              over-reacts to evidence
    gamma ~ 0, alpha > 0   no reliable tracking detected -- proxy use is
                           present but no increase with predictive value is
                           detected (intervals may remain compatible with
                           weak tracking)

What gamma ~ 0 does NOT establish: overcorrection. That claim needs the
separate signature of social sitting below the Ideal line WITH an accuracy cost
(measurable on the factual arm), not a flat slope.

SCOPE
    This curve speaks to in-context evidence sensitivity under rule-learned
    prompts. It says nothing about the label-triggered suppression, which lives in the
    rule-absent condition and is already known to disappear once calibration
    cases are supplied.
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

import config                              # noqa: E402
import dgp_k                               # noqa: E402
import dgp_norm                            # noqa: E402
import oracle                              # noqa: E402
sys.modules["dgp"] = dgp_k
from analyze import per_pair_pse           # noqa: E402
from run_cell import CELL_TAG_RE, build_cell   # noqa: E402


def collect_levels(results_dir: str, semantic: str, k: int = 12, corner: str = "LH",
                   seed=None):
    """Per-pair PSE at each beta level, keyed by pair, plus the Ideal line.

    MUST filter on k and corner, not just semantic+rule: interaction-tier cells
    (k6/k18 x LH/HH, learned, zero-info) are otherwise legal beta=0 candidates,
    and because `levels` is keyed by beta the sorted-last file silently wins --
    a different population whose pair keys only coincidentally intersect the
    sweep's (observed collapse: 99 -> 19 shared pairs, curve on garbage subset).
    """
    levels = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_records.csv"))):
        m = CELL_TAG_RE.search(os.path.basename(path))
        if not m or m[3] != semantic or m[4] != "learned":
            continue
        if int(m[1]) != k or m[2] != corner:
            continue                    # e.g. interaction-tier k6/k18 or HH cells
        cell_seed = None if m.lastindex is None or m.lastindex < 7 or m[7] is None \
            else int(m[7])
        if cell_seed != seed:
            continue                    # never mix seed replications with the
                                        # frozen configuration on one curve
        beta = 0.0 if (m.lastindex is None or m[6] is None) else int(m[6]) / 100.0
        if m[5] == "zero":
            beta = 0.0                      # the reused zero-info cell is beta = 0
        elif m[6] is None:
            continue                        # frozen beta=0.40 cell: different
                                            # parameterization, never on this curve
        df = pd.read_csv(path)
        k, corner = int(m[1]), m[2]
        ref, pairs, _, _, _ = build_cell(k, corner, semantic, "learned",
                                         beta > 0, beta if beta > 0 else None,
                                         seed)
        pse = per_pair_pse(df[df["arm"].isin(["cf_a1", "cf_a0"])])
        if not pse:
            continue
        gc, ga = config.CORNERS[corner]
        from run_cell import derived_seeds
        _, _, d_seed = derived_seeds(seed)
        if beta > 0:
            post = dgp_norm.fit_ideal(gc, ga, k=k, beta_info=beta,
                                      n_cal=config.N_CAL, dcal_seed=d_seed)
        else:
            dcal = dgp_k.generate(gc, ga, k=k, n=config.N_CAL, seed=d_seed)
            post = oracle.blr_fit(dcal.X, dcal.Y, sigma=dgp_k.SIGMA_Y, tau=1.0)
        levels[beta] = dict(pse=pse, ideal=dgp_norm.ideal_line(post, ref, pairs),
                            bayes=dgp_norm.bayes_line(ref, pairs), n=len(pse))
    return levels


def fit_curve(levels: dict, B: int = 5000, seed: int = 7):
    """OLS of model PSE on the Ideal line, with pair-resampling CIs."""
    betas = sorted(levels)
    if len(betas) < 3:
        raise SystemExit(f"need >=3 beta levels, found {betas}")
    shared = set.intersection(*(set(levels[b]["pse"]) for b in betas))
    # Guard against a wrong cell slipping into a level: the levels share one
    # frozen pair set by construction, so the intersection must be essentially
    # complete. A collapsed intersection means mixed populations -- fail loudly
    # rather than fit a curve on a coincidental subset.
    min_n = min(len(levels[b]["pse"]) for b in betas)
    if len(shared) < 0.9 * min_n:
        raise SystemExit(
            f"shared pairs collapsed to {len(shared)} (smallest level has "
            f"{min_n}); levels are mixing different populations -- check that "
            f"every level file is the same k/corner cell")
    keys = sorted(shared)
    x = np.array([levels[b]["ideal"] for b in betas])
    Y = np.array([[levels[b]["pse"][p] * 100 for p in keys] for b in betas])  # levels x pairs

    def ols(y):
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coef                                   # alpha, gamma

    alpha, gamma = ols(Y.mean(axis=1))
    rng = np.random.default_rng(seed)
    draws = np.empty((B, 2))
    n = len(keys)
    for b in range(B):
        idx = rng.integers(0, n, n)                   # same pairs across all levels
        draws[b] = ols(Y[:, idx].mean(axis=1))
    a_lo, a_hi = np.percentile(draws[:, 0], [2.5, 97.5])
    g_lo, g_hi = np.percentile(draws[:, 1], [2.5, 97.5])
    # A zero-width CI means every resample gave the same slope, which happens when
    # the model's PSE does not move across levels at all. That is the flat-slope
    # case; a naive "lo > 0 or hi < 0" test would call it a significant slope.
    tol = 1e-9
    degenerate = (g_hi - g_lo) < tol
    return dict(betas=betas, ideal=x, bayes=[levels[b]["bayes"] for b in betas],
                model=Y.mean(axis=1), n_pairs=n, degenerate_ci=degenerate,
                alpha=alpha, alpha_ci=(a_lo, a_hi),
                gamma=gamma, gamma_ci=(g_lo, g_hi),
                gamma_excludes_0=bool(not degenerate and (g_lo > 0 or g_hi < 0)),
                gamma_excludes_1=bool((g_lo > 1 or g_hi < 1)
                                      and not (degenerate and abs(gamma - 1) < tol)))


def verdict(fit: dict) -> str:
    g, (g_lo, g_hi) = fit["gamma"], fit["gamma_ci"]
    a, (a_lo, a_hi) = fit["alpha"], fit["alpha_ci"]
    a_pos = a_lo > 0
    if fit.get("degenerate_ci") and abs(g) < 1e-9:
        return ("no reliable tracking detected (PSE identical at every level)"
                + (" with a positive constant surplus" if a_pos else ""))
    if not fit["gamma_excludes_0"]:
        return ("no reliable tracking detected (gamma CI spans 0)"
                + (" with a positive constant surplus" if a_pos else ""))
    if fit["gamma_excludes_1"]:
        return "over-reacts to evidence" if g > 1 else "tracks evidence but under-weights it"
    return ("fully evidence-calibrated (gamma ~ 1, alpha ~ 0)" if not a_pos
            else "tracks evidence with an evidence-independent constant surplus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--corner", default="LH")
    ap.add_argument("--seed", type=int, default=None,
                    help="analyse a seed replication instead of the frozen cells")
    args = ap.parse_args()

    rows = []
    fits = {}
    for semantic in ("neutral", "social"):
        levels = collect_levels(args.results, semantic, args.k, args.corner,
                                args.seed)
        if len(levels) < 3:
            print(f"{semantic}: only {sorted(levels)} levels present, skipping")
            continue
        f = fit_curve(levels)
        fits[semantic] = f
        print(f"\n=== {semantic} ({f['n_pairs']} shared pairs) ===")
        print(f"{'beta':>6} {'ideal line':>11} {'model PSE':>10} {'bayes (ub)':>11}")
        for b, xi, yi, bi in zip(f["betas"], f["ideal"], f["model"], f["bayes"]):
            print(f"{b:>6.2f} {xi:>11.2f} {yi:>10.2f} {bi:>11.2f}")
            rows.append(dict(semantic=semantic, beta=b, ideal_pp=xi,
                             model_pp=yi, bayes_pp=bi))
        print(f"  gamma = {f['gamma']:+.3f}  CI [{f['gamma_ci'][0]:+.3f}, {f['gamma_ci'][1]:+.3f}]"
              f"   excludes 0: {f['gamma_excludes_0']}   excludes 1: {f['gamma_excludes_1']}")
        print(f"  alpha = {f['alpha']:+.2f} pp  CI [{f['alpha_ci'][0]:+.2f}, {f['alpha_ci'][1]:+.2f}]")
        print(f"  verdict: {verdict(f)}")

    if "neutral" in fits and "social" in fits:
        gn, gs = fits["neutral"]["gamma"], fits["social"]["gamma"]
        print(f"\nsecondary: gamma_social - gamma_neutral = {gs - gn:+.3f}")
        print("  (Gate 2 hinted the social slope may be ~0; treat as secondary)")

    if rows:
        out = args.out or os.path.join(args.results, "dose_response.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
