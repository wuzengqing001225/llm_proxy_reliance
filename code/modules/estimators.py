"""estimators.py — Pairwise PSE estimator, cluster bootstrap, transport gap
and density-ratio weighting for the pilot infrastructure.

Randomization protocol Ω (per pair):
    order       : (i,j) vs (j,i)             (>= 2 versions)
    template    : template_id ∈ {0,1,…}      (>= 2 versions)
    repeats     : per (order, template) draw >= 1 sample

A "call record" is one invocation:  (pair_id, order, template_id, repeat_id,
factual_or_counterfactual, decision ∈ {i, j}).

π_f(x_i, x_j) = P_Ω(f_Ω picks i in the factual world).
PSE(pair) = π_f_factual − π_f_counterfactual_on_A.

Cluster bootstrap: resample latent pairs (with replacement), aggregate π's
within-pair over the randomization draws that already exist for that pair,
then compute the estimand and take the sampling distribution across bootstrap
replicates.

Transport gap:
    TE_f = E_P[τ(Z)] − E_Q[τ(Z)]  ≈  Cov_Q(w, τ)  when E_Q[w] = 1.
    Density-ratio weighting: E_P̂[τ] = mean(w · τ) over Q-sampled data
    (self-normalized when w may drift from 1 in sample).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any


# ---------------------------------------------------------------------------
# Randomization protocol
# ---------------------------------------------------------------------------

@dataclass
class Randomization:
    n_orders: int = 2
    n_templates: int = 2
    n_repeats: int = 1

    def enumerate(self):
        for o in range(self.n_orders):
            for t in range(self.n_templates):
                for r in range(self.n_repeats):
                    yield dict(order=o, template=t, repeat=r)

    def __iter__(self):
        return self.enumerate()

    @property
    def n_calls_per_pair(self):
        return self.n_orders * self.n_templates * self.n_repeats


# ---------------------------------------------------------------------------
# Call generation: driver runs a decision_fn under the Ω protocol
# ---------------------------------------------------------------------------

def run_decisions(sample, pairs, decision_fn: Callable,
                  omega: Randomization, factual: bool,
                  seed: int = 0) -> List[Dict[str, Any]]:
    """Run the decision function over pair × Ω. Returns list of call records.

    decision_fn(sample, pair, order, template, repeat, rng) -> int (chosen index).
    """
    rng = np.random.default_rng(int(seed))
    records = []
    for pair_id, p in enumerate(pairs):
        for opts in omega.enumerate():
            # An i.i.d. sub-rng per call so reproducibility is invariant to loop order
            sub_seed = rng.integers(0, 2**32 - 1)
            sub_rng = np.random.default_rng(sub_seed)
            chosen = decision_fn(sample, p, opts["order"], opts["template"],
                                 opts["repeat"], sub_rng)
            records.append(dict(
                pair_id=pair_id, i=int(p["i"]), j=int(p["j"]),
                order=opts["order"], template=opts["template"],
                repeat=opts["repeat"],
                factual=bool(factual), chosen=int(chosen),
                picked_i=int(chosen == int(p["i"])),
            ))
    return records


def per_pair_counterfactual_records(sample, pairs, decision_fn: Callable,
                                    omega: "Randomization", seed: int = 0,
                                    target_A=None,
                                    _dgp_module=None) -> List[Dict[str, Any]]:
    """Generate counterfactual records where A is flipped on the TARGET (index i)
    of each pair, leaving the comparator j unchanged.

    target_A:
        None → flip: new_A_i = 1 - A_i (symmetric SCM counterfactual)
        int  → force: new_A_i = target_A  (directed counterfactual; useful when
               you want an unambiguous "A→1 vs A→0" sign convention).

    _dgp_module: optionally pass in the dgp module reference (avoids import cycle
    when this file is exec'd directly). Defaults to importing 'dgp' from sys.modules.
    """
    if _dgp_module is None:
        import sys as _sys
        _dgp_module = _sys.modules.get("dgp")
        if _dgp_module is None:
            import dgp as _dgp_module

    rng = np.random.default_rng(int(seed))
    records = []
    for pair_id, p in enumerate(pairs):
        new_A = sample.A.copy()
        idx_i = int(p["i"])
        if target_A is None:
            new_A[idx_i] = 1 - new_A[idx_i]
        else:
            new_A[idx_i] = int(target_A)
        cf = _dgp_module.counterfactual(sample, new_A)
        for opts in omega.enumerate():
            sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            chosen = decision_fn(cf, p, opts["order"], opts["template"],
                                 opts["repeat"], sub_rng)
            records.append(dict(
                pair_id=pair_id, i=idx_i, j=int(p["j"]),
                order=opts["order"], template=opts["template"], repeat=opts["repeat"],
                factual=False, chosen=int(chosen), picked_i=int(chosen == idx_i),
            ))
    return records


def pi_per_pair(records) -> Dict[int, float]:
    """Aggregate π = P_Ω(choose i) per pair_id."""
    out = {}
    counts = {}
    for r in records:
        out[r["pair_id"]] = out.get(r["pair_id"], 0) + r["picked_i"]
        counts[r["pair_id"]] = counts.get(r["pair_id"], 0) + 1
    return {k: out[k] / counts[k] for k in out}


# ---------------------------------------------------------------------------
# PSE estimator (per-pair and average) + cluster bootstrap
# ---------------------------------------------------------------------------

def pse_per_pair(records_f, records_cf) -> Dict[int, float]:
    """PSE(pair) = π_factual(pair) − π_counterfactual(pair)."""
    pi_f = pi_per_pair(records_f)
    pi_cf = pi_per_pair(records_cf)
    keys = set(pi_f) | set(pi_cf)
    return {k: pi_f.get(k, 0.5) - pi_cf.get(k, 0.5) for k in sorted(keys)}


def pse_average(records_f, records_cf) -> float:
    """Average PSE = mean over pairs."""
    per = pse_per_pair(records_f, records_cf)
    return float(np.mean(list(per.values())))


def cluster_bootstrap_pse(records_f, records_cf, B: int = 2000,
                          seed: int = 0, ci: float = 0.95):
    """Cluster-bootstrap the average PSE over latent pairs (pair_id is cluster).

    Signed by the observed A: PSE(pair) = π_factual − π_counterfactual, where
    the counterfactual flips A_i → 1 − A_i.
    """
    per = pse_per_pair(records_f, records_cf)
    pair_ids = np.array(sorted(per.keys()))
    values = np.array([per[p] for p in pair_ids])
    n = len(pair_ids)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return dict(estimate=float(values.mean()), se=float(boot.std(ddof=1)),
                ci_lo=float(lo), ci_hi=float(hi), n_pairs=int(n), B=B)


def directed_pse(records_a1, records_a0):
    """Directed PSE = π(target A=1) − π(target A=0), per pair.

    records_a1 and records_a0 are counterfactual record sets generated with
    target_A=1 and target_A=0 respectively.
    """
    pi_a1 = pi_per_pair(records_a1)
    pi_a0 = pi_per_pair(records_a0)
    keys = sorted(set(pi_a1) & set(pi_a0))
    return {k: pi_a1[k] - pi_a0[k] for k in keys}


def cluster_bootstrap_directed_pse(records_a1, records_a0, B: int = 2000,
                                   seed: int = 0, ci: float = 0.95):
    per = directed_pse(records_a1, records_a0)
    keys = np.array(sorted(per.keys()))
    values = np.array([per[k] for k in keys])
    n = len(keys)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return dict(estimate=float(values.mean()), se=float(boot.std(ddof=1)),
                ci_lo=float(lo), ci_hi=float(hi), n_pairs=int(n), B=B)


# ---------------------------------------------------------------------------
# Transport gap and density-ratio weighting
# ---------------------------------------------------------------------------

def transport_gap_analytic(tau_Q, w_Q, tau_P=None):
    """Return (E_P[τ] − E_Q[τ], Cov_Q(w, τ)).

    Given τ evaluated at Q-samples, and importance weights w = p(Z)/q(Z) at those
    samples, the covariance identity yields E_P[τ] − E_Q[τ] = Cov_Q(w, τ) when
    E_Q[w] = 1. If Q- and P-samples of τ are provided separately, the LHS is the
    plain difference of means. This function returns both under the same set of
    samples for direct comparison.
    """
    tau_Q = np.asarray(tau_Q); w_Q = np.asarray(w_Q)
    cov = float(np.cov(w_Q, tau_Q, bias=True)[0, 1])
    if tau_P is None:
        E_P_hat = float((w_Q * tau_Q).mean())
        gap = E_P_hat - float(tau_Q.mean())
    else:
        gap = float(np.mean(tau_P) - np.mean(tau_Q))
    return dict(gap=gap, cov=cov)


def density_ratio_weight_gaussian(x_Q, mu_P, cov_P, mu_Q, cov_Q):
    """Analytic w = p(x)/q(x) for multivariate Gaussian p and q, evaluated at x_Q.

    Used for the transport-gap validation where P and Q are known Gaussians (the
    audit vs deployment distributions differ only in their attribute correlation).
    """
    x_Q = np.asarray(x_Q)
    dP = x_Q.shape[1]
    inv_P = np.linalg.inv(cov_P)
    inv_Q = np.linalg.inv(cov_Q)
    log_det_P = np.linalg.slogdet(cov_P)[1]
    log_det_Q = np.linalg.slogdet(cov_Q)[1]

    diff_P = x_Q - mu_P
    diff_Q = x_Q - mu_Q
    logp = -0.5 * log_det_P - 0.5 * np.einsum('ij,jk,ik->i', diff_P, inv_P, diff_P)
    logq = -0.5 * log_det_Q - 0.5 * np.einsum('ij,jk,ik->i', diff_Q, inv_Q, diff_Q)
    logw = logp - logq
    w = np.exp(logw - logw.max())
    w = w / w.mean()  # self-normalize
    return w


__all__ = [
    "Randomization", "run_decisions", "per_pair_counterfactual_records",
    "pi_per_pair",
    "pse_per_pair", "pse_average", "cluster_bootstrap_pse",
    "directed_pse", "cluster_bootstrap_directed_pse",
    "transport_gap_analytic", "density_ratio_weight_gaussian",
]
