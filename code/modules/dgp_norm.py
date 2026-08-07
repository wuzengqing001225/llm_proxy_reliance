"""Variance-normalized proxy-strength sweep, layered on the frozen dgp_k.

WHY THIS MODULE EXISTS
----------------------
`dgp_k.generate(proxy_informative=True, beta_info=b)` raises the proxy loading
without renormalizing the legitimate signal, so the TOTAL risk variance grows
with b (1.011 -> 1.279 over b in [0, 0.8]) and the |dr| ranking-difficulty scale
drifts with it. For a dose-response design that is fatal: ranking difficulty and
the quantity being measured move together.

This module fixes total risk variance while letting the proxy's SHARE of the
risk signal vary -- which is exactly what the justified level measures.

    r_true = a * r_legit + c * (P @ w_P)
    Var(r_true) = V  (held constant),  Var(c * P@w_P) / V = beta^2 / V

WHAT MUST NOT BE "FIXED" INSTEAD
--------------------------------
Holding the *legit share* constant looks like the natural normalization and is
wrong: Var(P@w_P) = beta^2 and Var(r_legit) == 1 by construction, so legit
share, proxy variance share, and the justified level are all functions of one
another. Pinning the share pins the justified level and the manipulation
vanishes (measured: justified level = 0.00 pp at every beta).

The design therefore accepts a documented trade-off. These three cannot hold at
once; this module keeps the first two:
  1. proxy carries a non-zero share of the risk signal (manipulation exists)
  2. total risk variance constant (ranking difficulty comparable)
  3. legitimate fields strictly dominant (scenario stays "proxies carry SOME
     diagnostic value") -- at the top level beta=0.60 the legit share is 0.385
     (measured, seed 10001): proxies slightly EXCEED the legitimate fields.
     The curve therefore covers "proxies carry 0 -> roughly 60% of the risk
     signal"; do NOT describe the top level as "legit fields still dominant".
     State this range limit in the paper.

COMMON-TARGET DISCIPLINE
------------------------
`dgp_k.make_pairs` strata on `sample.r`, which this module rescales. Building
pairs per level would therefore shift the evaluation set with beta (measured
overlap drops to 76/99) and destroy the paired comparison. Always take pairs
from ONE reference sample via `reference_pairs()` and reuse them at every level.
"""
from __future__ import annotations
import dataclasses
import numpy as np

import dgp_k

V_DEFAULT = 1.10          # total risk variance; > Var(r_legit)=1 to leave room
BETA_LEVELS = (0.0, 0.30, 0.45, 0.60)   # even steps on the IDEAL line


def generate_normalized(gamma_C, gamma_A, k: int = 12, n: int = 5000, seed: int = 0,
                        beta_info: float = 0.45, V: float = V_DEFAULT, **kw):
    """Sample with proxy strength `beta_info` at fixed total risk variance `V`.

    beta_info == 0 returns dgp_k.generate(...) unchanged, so zero-info cells
    already collected remain valid and bit-identical.
    """
    s = dgp_k.generate(gamma_C, gamma_A, k=k, n=n, seed=seed,
                       proxy_informative=(beta_info > 0), beta_info=beta_info, **kw)
    if beta_info <= 0:
        return s

    contrib = s.r_true - s.r                      # = P @ w_P (unstandardized P)
    proxy_var = min(beta_info ** 2, 0.95 * V)     # cap so the legit part stays real
    c = np.sqrt(proxy_var / np.var(contrib))
    a = np.sqrt(max(1e-12, (V - proxy_var) / np.var(s.r)))
    r_true = a * s.r + c * contrib

    # Scaling r and w_P consistently is what keeps dgp_k.counterfactual correct:
    # it recomputes r_true_new = sample.r + P_new @ sample.w_P, which with the
    # scaled fields evaluates to a*r + c*(P_new @ w_P_orig) as intended.
    return dataclasses.replace(s, r=a * s.r, w_P=s.w_P * c,
                               r_true=r_true, Y=r_true + s.y_noise)


def reference_pairs(gamma_C, gamma_A, k: int = 12, n: int = 5000,
                    seed: int = 10001, n_per_stratum: int = 33,
                    pair_seed: int = 30001):
    """The single evaluation pair set, reused at every beta level.

    Built from the beta=0 sample so it is independent of proxy strength.
    """
    ref0 = dgp_k.generate(gamma_C, gamma_A, k=k, n=n, seed=seed)
    return dgp_k.make_pairs(ref0, n_per_stratum=n_per_stratum, seed=pair_seed)


def justified_level(sample, pairs) -> float:
    """Deprecated alias for `bayes_line` -- kept so older cells still run.

    The earlier implementation intervened on all target indices in ONE batch,
    which silently corrupts any pair whose comparator is another pair's target.
    Use `bayes_line` (complete-knowledge upper bound) or `ideal_line` (the
    primary reference) instead.
    """
    return bayes_line(sample, pairs)


def structure_report(sample, pairs) -> dict:
    """Manipulation-check quantities for one level."""
    k = sample.X.shape[1]
    L_, M_ = dgp_k.K_LADDER[k]
    C = np.corrcoef(sample.X.T)
    ev = np.linalg.eigvalsh(C)
    cmc = ev.max() / np.trace(C)
    P = sample.X[:, L_:]
    Pd = np.hstack([P, np.ones((len(P), 1))])
    beta_A, *_ = np.linalg.lstsq(Pd, sample.A.astype(float), rcond=None)
    a_mc = 1 - np.var(sample.A - Pd @ beta_A) / np.var(sample.A.astype(float))
    r_leg = np.mean([abs(np.corrcoef(sample.X[:, j], sample.r_true)[0, 1])
                     for j in range(L_)])
    r_pr = np.mean([abs(np.corrcoef(sample.X[:, L_ + m], sample.r_true)[0, 1])
                    for m in range(M_)])
    return dict(
        C_MC_norm=(cmc - 1 / k) / (1 - 1 / k),
        A_MC=float(a_mc),
        sd_r_true=float(sample.r_true.std()),
        legit_share=float(r_leg / (r_leg + r_pr)),
        median_abs_dr=float(np.median([abs(sample.r_true[int(p["i"])]
                                           - sample.r_true[int(p["j"])]) for p in pairs])),
        justified_pp=justified_level(sample, pairs),
    )


# ---------------------------------------------------------------------------
# Reference lines for the dose-response readout
# ---------------------------------------------------------------------------
# There are TWO reference lines and they are not interchangeable.
#
#   bayes_line()  -- a decider with COMPLETE knowledge of the generating rule.
#                    Report it as an UPPER BOUND only.
#   ideal_line()  -- the conjugate-BLR learner given exactly the same 80
#                    calibration cases the model sees. This is the line the
#                    model "should" track: with 80 cases the posterior shrinks,
#                    so a finite-sample learner cannot reach the Bayes line
#                    (measured shrinkage: 2-4 pp). Scoring the model against the
#                    Bayes line would read that shrinkage as the model failing to
#                    track, biasing the tracking slope downward.


def _directed_pse(score_fn, sample, pairs) -> float:
    """Directed PSE in pp, intervening on ONE pair's target index at a time.

    Per-pair (not batched) on purpose: batching writes A for every target index
    at once, which corrupts a pair whose comparator j happens to be another
    pair's target i. Whether that collision exists depends on the pair draw --
    at seed 10001 there is none, but that is luck, not a guarantee, and target
    indices already repeat once.
    """
    out = {}
    for target in (1, 0):
        picks = []
        for p in pairs:
            i, j = int(p["i"]), int(p["j"])
            newA = sample.A.copy()
            newA[i] = target                     # target only; comparator untouched
            cf = dgp_k.counterfactual(sample, newA)
            picks.append(float(score_fn(cf, i, j)))
        out[target] = np.array(picks)
    return float((out[1] - out[0]).mean() * 100)


def bayes_line(sample, pairs) -> float:
    """Directed PSE of a complete-knowledge decider (upper bound reference)."""
    return _directed_pse(lambda cf, i, j: cf.r_true[i] >= cf.r_true[j], sample, pairs)


def ideal_line(posterior, sample, pairs) -> float:
    """Directed PSE of the BLR Ideal learner -- the primary reference line.

    `posterior` must come from oracle.blr_fit on the SAME calibration set the
    model was shown (same k, corner, beta, DCAL seed).
    """
    import oracle
    return _directed_pse(
        lambda cf, i, j: oracle.ideal_pairwise_decision(
            posterior, cf.X[i], cf.X[j], i, j) == i,
        sample, pairs)


def fit_ideal(gamma_C, gamma_A, k: int = 12, beta_info: float = 0.45,
              n_cal: int = 80, dcal_seed: int = 517, sigma_y=None, tau: float = 1.0):
    """Posterior for the Ideal learner on the normalized calibration set."""
    import oracle
    dcal = generate_normalized(gamma_C, gamma_A, k=k, n=n_cal, seed=dcal_seed,
                               beta_info=beta_info)
    return oracle.blr_fit(dcal.X, dcal.Y,
                          sigma=dgp_k.SIGMA_Y if sigma_y is None else sigma_y, tau=tau)
