#!/usr/bin/env python3
"""Pair-clustered logistic model and per-order estimates of the proxy effect.

Single cell (flip effect):
    python clustered_logit.py --neutral CELL_records.csv

Label contrast (suppression), neutral and social cells of the same pairs:
    python clustered_logit.py --neutral N_records.csv --social S_records.csv

Several independent generating-process draws pooled (repeat the flags once per
draw, in matching order):
    python clustered_logit.py --neutral N0.csv N1.csv N2.csv \
                              --social  S0.csv S1.csv S2.csv

Model. Each counterfactual-arm call is one binary observation (was the target
patient picked):

    logit P(pick target) = b0 + b_arm*[do(A=1)] + b_order*[target shown second]
                           (+ b_arm_order*[do(A=1)]*[target second], --arm-x-order)
                           (+ b_social*[social] + b_int*[do(A=1)]*[social], label contrast)
                           (+ one intercept per draw, when several draws are given)

fitted by generalized estimating equations with an independence working
correlation and the evaluation pair (within its draw) as the cluster, so the
calls that share a pair are not treated as independent.

Reported:
  * the average marginal effect of the flip (or of the flip-by-label
    interaction) in pp, with a pair-cluster bootstrap interval that resamples
    pairs within each draw;
  * the robust z-test of the corresponding log-odds coefficient;
  * the order coefficient, and with --arm-x-order the arm-by-order term;
  * direct per-order estimates, computed separately for the calls in which the
    target was shown first and second, each with its own pair bootstrap
    interval and no model: the neutral flip effect and, for label contrasts,
    the social flip effect and the social-minus-neutral contrast.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load(path: str, social: int, draw: int) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.arm.isin(["cf_a1", "cf_a0"]) & d.picked_i.notna()].copy()
    key = ["arm", "i", "j", "order"]
    dup = d.duplicated(key).sum()
    if dup:
        raise SystemExit(f"{path}: {dup} duplicated calls; refusing to fit")
    d["y"] = d.picked_i.astype(float).astype(int)
    d["arm"] = (d.arm == "cf_a1").astype(int)
    d["order"] = d["order"].astype(int)          # 1 = target shown second
    d["pair"] = (f"d{draw}_" + d.i.astype(int).astype(str) + "_"
                 + d.j.astype(int).astype(str))
    d["social"] = social
    d["draw"] = draw
    return d[["y", "arm", "order", "pair", "social", "draw"]]


def design(d: pd.DataFrame, label: bool, arm_x_order: bool, n_draws: int):
    X = pd.DataFrame({"const": 1.0, "arm": d.arm, "order": d.order})
    if arm_x_order:
        X["arm_x_order"] = d.arm * d.order
    if label:
        X["social"] = d.social
        X["arm_x_social"] = d.arm * d.social
    for k in range(1, n_draws):
        X[f"draw{k}"] = (d.draw == k).astype(float)
    return X


def fit(d, label, arm_x_order, n_draws):
    X = design(d, label, arm_x_order, n_draws)
    return sm.GEE(d.y, X, groups=pd.factorize(d.pair)[0],
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Independence()).fit()


def ame(res, d, label, arm_x_order, n_draws):
    def flip(sub):
        X1 = design(sub.assign(arm=1), label, arm_x_order, n_draws)
        X0 = design(sub.assign(arm=0), label, arm_x_order, n_draws)
        return 100 * float(np.mean(res.predict(X1) - res.predict(X0)))
    if not label:
        return flip(d)
    return flip(d[d.social == 1]) - flip(d[d.social == 0])


def resample(d, rng):
    parts = []
    for _, g in d.groupby("draw"):
        pairs = g.pair.unique()
        by = {p: x for p, x in g.groupby("pair")}
        for k, p in enumerate(rng.choice(pairs, len(pairs), replace=True)):
            parts.append(by[p].assign(pair=f"{p}#{k}"))
    return pd.concat(parts, ignore_index=True)


def _flip_by_pair(d, order, social):
    g = d[(d.order == order) & (d.social == social)]
    w = g.pivot_table(index="pair", columns="arm", values="y")
    return (w[1] - w[0]).dropna()


def per_order(d, B, rng, label):
    """Paired effects within each display order, no model.

    Returns, per order, the neutral flip effect and, for label contrasts, the
    social flip effect and the social-minus-neutral contrast computed on the
    pairs present in both cells. Intervals resample pairs (within draw).
    """
    out = {}
    for o in (0, 1):
        series = {"neutral flip": _flip_by_pair(d, o, 0)}
        if label:
            n, s = _flip_by_pair(d, o, 0), _flip_by_pair(d, o, 1)
            common = n.index.intersection(s.index)
            series["social flip"] = s
            series["social - neutral"] = s.loc[common] - n.loc[common]
        res = {}
        for name, ser in series.items():
            v = ser.to_numpy()
            # pair ids start with "d<draw>_"; resample within each draw so a
            # pooled interval keeps every draw's sample size, as in the
            # model-based interval
            draws = np.array([p.split("_", 1)[0] for p in ser.index])
            groups = [np.flatnonzero(draws == g) for g in np.unique(draws)]
            bs = []
            for _ in range(B):
                idx = np.concatenate([g[rng.integers(0, len(g), len(g))]
                                      for g in groups])
                bs.append(100 * v[idx].mean())
            res[name] = (100 * v.mean(), *np.percentile(bs, [2.5, 97.5]), len(v))
        out[o] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral", nargs="+", required=True)
    ap.add_argument("--social", nargs="+")
    ap.add_argument("--arm-x-order", action="store_true")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    label = args.social is not None
    if label and len(args.social) != len(args.neutral):
        ap.error("--social needs one file per --neutral file")
    nd = len(args.neutral)
    parts = []
    for k, path in enumerate(args.neutral):
        parts.append(load(path, 0, k))
        if label:
            parts.append(load(args.social[k], 1, k))
    d = pd.concat(parts, ignore_index=True)

    res = fit(d, label, args.arm_x_order, nd)
    est = ame(res, d, label, args.arm_x_order, nd)
    rng = np.random.default_rng(args.seed)
    bs = []
    for _ in range(args.B):
        bd = resample(d, rng)
        bs.append(ame(fit(bd, label, args.arm_x_order, nd), bd, label,
                      args.arm_x_order, nd))
    lo, hi = np.percentile(bs, [2.5, 97.5])

    term = "arm_x_social" if label else "arm"
    what = "suppression (social - neutral)" if label else "flip effect"
    print(f"draws: {nd}  pairs: {d.pair.nunique()}  calls: {len(d)}")
    print(f"{what} (AME): {est:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
    print(f"{term} log-odds: {res.params[term]:+.3f}  "
          f"z = {res.tvalues[term]:+.2f}  P = {res.pvalues[term]:.2e}")
    print(f"order log-odds: {res.params['order']:+.3f}  P = {res.pvalues['order']:.3g}")
    if args.arm_x_order:
        print(f"arm x order log-odds: {res.params['arm_x_order']:+.3f}  "
              f"P = {res.pvalues['arm_x_order']:.3g}")
    po = per_order(d, args.B, rng, label)
    for o, name in ((0, "target shown first "), (1, "target shown second")):
        for quantity, (e, l, h, n) in po[o].items():
            print(f"per-order, {name}, {quantity}: {e:+.2f} pp "
                  f"[{l:+.2f}, {h:+.2f}]  ({n} pairs)")


if __name__ == "__main__":
    main()
