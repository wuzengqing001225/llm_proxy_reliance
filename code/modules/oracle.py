"""oracle.py — Bayes-known oracle and Ideal learner (conjugate Bayesian
linear regression) for the pilot infrastructure.

Both operate on the same X representation the LLM sees (standardized within-sample
concatenation of [X_legit, P], shape (n, K)). The LEGIT-vs-PROXY partition is
known only to the DGP: the Ideal learner regresses on the full k = K = 6
attributes and does not know which columns are legit.

Bayes-known oracle
------------------
Has full knowledge of the true generative rule. For the frozen DGP:
    - Zero-info condition: r_true = X_legit @ w, ranking on r_true is optimal.
    - Informative condition: r_true = X_legit @ w + P @ w_P.
Pairwise decision: argmax_{i∈{a,b}} r_true(i).

If r_true(i) ≠ r_true(j), the Bayes decision is deterministic and its pairwise
error is exactly 0. The reported R_Bayes is thus 0 on any deterministic pair set.
(For a Y-based comparison one can also define R_Bayes^Y using the noisy Y, but
by construction Y = r_true + noise is a monotonic-in-expectation ranking; the
finite-sample Bayes error rate on noisy Y is computable but not used as the
target rate here.)

Ideal learner (conjugate Bayesian linear regression)
----------------------------------------------------
Predictive model:
    Y | X, beta ~ N(X_aug @ beta, sigma^2),   X_aug = [1, X] of shape (n, K+1)
    beta ~ N(0, tau^2 * I_{K+1})
Posterior:
    beta | D_cal ~ N(m_N, S_N)
    S_N = (tau^{-2} I + sigma^{-2} X'X)^{-1}
    m_N = sigma^{-2} S_N X' y
Posterior predictive mean at x_star: x_star_aug @ m_N.
Pairwise decision: argmax posterior-predictive mean.

Verification targets (see infrastructure_validation_report.md):
    (a) |D_cal| → ∞  ⇒  R_Ideal → R_Bayes = 0  (learner recovers truth)
    (b) On the SAME test pair set, R_any − R_Bayes = (R_Ideal − R_Bayes)
        + (R_any − R_Ideal), numerically exact.
    (c) |D_cal| sweep ∈ {5, 10, 20, 40, 80} produces a monotone-decreasing
        Ideal-learner regret without saturation at either end.
    (d) Same latent pairs re-surfaced at the four corners → oracle pairwise
        error difference < 2 pp (target: exactly 0 on r_true-defined pairs).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Bayes-known oracle
# ---------------------------------------------------------------------------

def bayes_pairwise_decision(sample, pair) -> int:
    """Return the index (i or j) that the Bayes-known oracle selects.

    Ranks on sample.r_true. Deterministic; ties break to the first index.
    """
    i, j = int(pair["i"]), int(pair["j"])
    return i if sample.r_true[i] >= sample.r_true[j] else j


def bayes_pairwise_error_rate(sample, pairs) -> float:
    """R_Bayes = P(oracle picks the wrong patient). Ground truth is the pair
    tuple: i > j is the label if r_true[i] > r_true[j]. Pairs from make_pairs
    always have i as the higher-r individual, so the correct answer is i.
    """
    n_err = 0
    n_tot = 0
    for p in pairs:
        pred = bayes_pairwise_decision(sample, p)
        # ground-truth answer is the one with the higher r_true value
        truth = p["i"] if sample.r_true[p["i"]] >= sample.r_true[p["j"]] else p["j"]
        n_err += int(pred != truth)
        n_tot += 1
    return n_err / max(n_tot, 1)


# ---------------------------------------------------------------------------
# Conjugate Bayesian linear regression — Ideal learner
# ---------------------------------------------------------------------------

@dataclass
class BLRPosterior:
    m: np.ndarray   # posterior mean of beta, shape (K+1,)
    S: np.ndarray   # posterior covariance, shape (K+1, K+1)
    sigma: float
    tau: float


def blr_fit(X: np.ndarray, y: np.ndarray, sigma: float, tau: float) -> BLRPosterior:
    """Conjugate Bayesian linear regression with known noise sigma and
    isotropic prior N(0, tau^2 I) on beta (including intercept)."""
    n, p = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    d = p + 1
    prior_prec = np.eye(d) / (tau ** 2)
    lik_prec = X_aug.T @ X_aug / (sigma ** 2)
    S_inv = prior_prec + lik_prec
    S = np.linalg.inv(S_inv)
    m = S @ (X_aug.T @ y / (sigma ** 2))
    return BLRPosterior(m=m, S=S, sigma=sigma, tau=tau)


def blr_predict_mean(post: BLRPosterior, X_new: np.ndarray) -> np.ndarray:
    """Posterior-predictive mean at rows of X_new (n, K)."""
    n = X_new.shape[0]
    X_aug = np.hstack([np.ones((n, 1)), X_new])
    return X_aug @ post.m


def ideal_pairwise_decision(post: BLRPosterior, X_row_i: np.ndarray,
                            X_row_j: np.ndarray, idx_i: int, idx_j: int) -> int:
    """Pick argmax posterior-predictive mean between two rows."""
    pred_i = blr_predict_mean(post, X_row_i[None, :])[0]
    pred_j = blr_predict_mean(post, X_row_j[None, :])[0]
    return idx_i if pred_i >= pred_j else idx_j


def ideal_pairwise_error_rate(sample, pairs, post: BLRPosterior) -> float:
    """R_Ideal on the given pair set, using the posterior-predictive mean."""
    X = sample.X
    n_err = 0
    for p in pairs:
        i, j = int(p["i"]), int(p["j"])
        pred = ideal_pairwise_decision(post, X[i], X[j], i, j)
        truth = i if sample.r_true[i] >= sample.r_true[j] else j
        n_err += int(pred != truth)
    return n_err / max(len(pairs), 1)


# ---------------------------------------------------------------------------
# D_cal construction (calibration set drawn from the same generative distribution)
# ---------------------------------------------------------------------------

def make_calibration_set(sample_gen_fn, gamma_C, gamma_A, n_cal: int, seed: int,
                         proxy_informative: bool = False):
    """Draw a calibration set of size n_cal from the same DGP, using an
    isolated seed (offset from pilot seeds).

    sample_gen_fn: dgp.generate function (passed in to avoid an import cycle).
    Returns a `Sample` object.
    """
    return sample_gen_fn(gamma_C, gamma_A, n=n_cal, seed=seed,
                         proxy_informative=proxy_informative)


# ---------------------------------------------------------------------------
# End-to-end regret decomposition
# ---------------------------------------------------------------------------

def regret_decomposition(sample_test, pairs, decision_fn,
                         ideal_post: BLRPosterior):
    """Return (R_any, R_ideal, R_bayes, gap_stat, gap_llm) on the same test set.

    decision_fn(sample_test, pair) -> chosen index (i or j). This is the "LLM"
    or fake-LLM under test; pairs' ground truth is sample_test.r_true.
    """
    R_any   = _error_of_decision(sample_test, pairs, decision_fn)
    R_ideal = ideal_pairwise_error_rate(sample_test, pairs, ideal_post)
    R_bayes = bayes_pairwise_error_rate(sample_test, pairs)
    return dict(
        R_any=R_any, R_ideal=R_ideal, R_bayes=R_bayes,
        gap_statistical=R_ideal - R_bayes,   # finite-sample statistical difficulty
        gap_llm_specific=R_any - R_ideal,    # ICL regret
        total=R_any - R_bayes,
        identity_check=abs((R_any - R_bayes) - ((R_ideal - R_bayes) + (R_any - R_ideal))),
    )


def _error_of_decision(sample, pairs, decision_fn) -> float:
    n_err = 0
    for p in pairs:
        pred = decision_fn(sample, p)
        truth = p["i"] if sample.r_true[p["i"]] >= sample.r_true[p["j"]] else p["j"]
        n_err += int(pred != truth)
    return n_err / max(len(pairs), 1)


__all__ = [
    "BLRPosterior", "blr_fit", "blr_predict_mean",
    "bayes_pairwise_decision", "bayes_pairwise_error_rate",
    "ideal_pairwise_decision", "ideal_pairwise_error_rate",
    "make_calibration_set", "regret_decomposition",
]
