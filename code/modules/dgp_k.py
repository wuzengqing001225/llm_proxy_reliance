"""
dgp_k.py — k-generalized version of dgp.py.

WHY THIS EXISTS
    dgp.py hard-freezes k = L + M = 4 + 2 = 6 (`if k != K: raise ValueError`).
    Every result in the project to date was produced at k=6. The core research
    question asks how ATTRIBUTE DIMENSION interacts with dependency structure,
    which requires varying k. This module unfreezes that axis.

    dgp.py is left untouched so the lineage of all existing artifacts stays valid.

BACKWARD COMPATIBILITY
    At (L=4, M=2) this module reproduces dgp.py BIT-FOR-BIT in the zero-info
    condition (verified by test_backward_compat()). Two facts make this work:
      * np.linspace(0.6, 0.3, 4) == [0.6, 0.5, 0.4, 0.3] == dgp.W_RAW exactly,
        so the weight profile generalizes without changing the k=6 case.
      * The RNG draw ORDER is preserved (A, eta_C, u_A, S, proxy_noise, y_noise),
        so identical seeds give identical draws.

ONE DELIBERATE DIFFERENCE (informative condition only)
    dgp.py sets w_P = beta_info / sqrt(M) per proxy. Because proxies share
    eta_A, Var(P @ w_P) = beta_info^2 * [1 + (M-1) * gA^2] — this GROWS with M,
    so the informative proxy signal would get mechanically stronger at larger k
    and confound the dimension effect. This module normalizes by the true
    correlated variance so Var(P @ w_P) = beta_info^2 for every M.

    This changes the k=6 INFORMATIVE config relative to dgp.py. That condition
    was validated in infrastructure but NEVER run in any main experiment, so no
    existing result is affected. Pass legacy_wp=True to recover the old formula.
    The zero-info condition is unaffected (w_P = 0 either way).

DIFFICULTY MATCHING ACROSS k
    r = Xleg @ w is normalized to unit variance at every L (see _compute_w), and
    make_pairs stratifies on |Δr| measured in units of r's SD. So the Bayes-optimal
    ranking difficulty of a pair at a given stratum is matched across k BY
    CONSTRUCTION. What legitimately changes with k is how many fields a learner
    must integrate to recover r — which is the effect under study, not a confound.

READING C_MC ACROSS k
    C_MC = lambda_max(Sigma_{X|A}) / tr(Sigma_{X|A}) has floor 1/k, so its raw
    value is NOT comparable across dimensions. Use the normalized form
        C_MC_norm = (C_MC - 1/k) / (1 - 1/k)  in [0, 1]
    which manipulation_check() reports alongside the raw value.
"""


from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

L: int = 4          # DEFAULT number of legit attributes (k=6 frozen config)
M: int = 2          # DEFAULT number of proxy attributes
K: int = L + M      # DEFAULT total attribute dimension

# Weight profile, generalized. linspace(0.6, 0.3, 4) == [0.6,0.5,0.4,0.3] exactly,
# so L=4 reproduces the frozen W_RAW.
W_HI, W_LO = 0.6, 0.3

def w_raw(L_: int = L) -> np.ndarray:
    """Legit-attribute raw weights for dimension L_ (decaying profile)."""
    if L_ < 2:
        raise ValueError(f"L must be >= 2; got {L_}")
    return np.linspace(W_HI, W_LO, L_)

W_RAW: np.ndarray = w_raw(L)      # frozen k=6 value, kept for compatibility

# Dimension ladder used by the k sweep. Ratio L:M held at 2:1 so the proxy
# FRACTION is invariant — otherwise PSE would shrink with k for the purely
# mechanical reason that proxies carry less relative weight.
K_LADDER = {6: (4, 2), 12: (8, 4), 18: (12, 6)}

# A_MC = R^2(A | P) rises mechanically with M: each extra proxy carries more
# information about A, so a fixed mu_A would confound "attribute dimension"
# with "total information about the protected attribute". mu_A is therefore
# re-solved per dimension (brentq on A_MC, n=40k, seed=999) to hold A_MC at the
# k=6 anchor values: low = 0.0682, high = 0.4245. gA is left at its frozen value
# so the per-proxy A-loading is unchanged; only the latent alignment moves.
MU_A_BY_K = {
    (6,  "low"):  0.5500,   (6,  "high"): 0.7700,   # frozen anchors
    (12, "low"):  0.4409,   (12, "high"): 0.7159,
    (18, "low"):  0.3940,   (18, "high"): 0.6981,
}
SIGMA_Y: float = 0.5    # residual noise on Y | r

# gCp = proxy loading on eta_C. Kept at 0 so proxies are pure A-signal + noise;
# leaves A_MC as a function of (gA, mu_A) alone and prevents cross-contamination
# of A_MC by gamma_C.
GCP_DEFAULT: float = 0.0

# Frozen (gC, gA, mu_A) for the four corners + center. Keys use the (γ_C, γ_A)
# convention of the study plan.
FROZEN_PARAMS = {
    ("low",  "low"):  dict(gC=0.05, gA=0.35, mu_A=0.55),
    ("low",  "high"): dict(gC=0.05, gA=0.75, mu_A=0.77),
    ("high", "low"):  dict(gC=0.70, gA=0.35, mu_A=0.55),
    ("high", "high"): dict(gC=0.70, gA=0.75, mu_A=0.77),
    ("mid",  "mid"):  dict(gC=0.37, gA=0.55, mu_A=0.66),
}


def resolve_params(gamma_C, gamma_A):
    """Map (γ_C, γ_A) labels to (gC, gA, mu_A) numeric parameters."""
    key = (gamma_C, gamma_A)
    if key not in FROZEN_PARAMS:
        raise KeyError(f"unknown corner {key}; valid keys: {list(FROZEN_PARAMS)}")
    return FROZEN_PARAMS[key].copy()


# ---------------------------------------------------------------------------
# Core sample container
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """A generated sample. Every exogenous draw is preserved so that
    counterfactual(new_A) can regenerate the endogenous quantities."""
    # Endogenous
    A: np.ndarray          # (n,) int {0,1}
    X: np.ndarray          # (n, K) = [X_legit | P] standardized within-sample
    Xleg: np.ndarray       # (n, L)
    P: np.ndarray          # (n, M)
    r: np.ndarray          # (n,) legit risk score = Xleg @ w
    r_true: np.ndarray     # (n,) Bayes-optimal ranking score:
                           #      r_legit                 (zero-info condition)
                           #      r_legit + P @ w_P       (informative condition)
    Y: np.ndarray          # (n,) = r_true + N(0, sigma_y^2)
    # Exogenous / bookkeeping
    S: np.ndarray          # (n, L)
    eta_C: np.ndarray      # (n,)
    u_A: np.ndarray        # (n,)
    proxy_noise: np.ndarray  # (n, M)
    y_noise: np.ndarray    # (n,)
    # Parameters used
    gC: float
    gA: float
    mu_A: float
    gCp: float
    proxy_informative: bool
    beta_info: float
    w: np.ndarray          # (L,) legit-attribute risk weights
    w_P: np.ndarray        # (M,) proxy weights (all zero in zero-info condition)

    def __len__(self):
        return self.A.shape[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_w(gC: float, L_: int = L) -> np.ndarray:
    """Weights giving r = Xleg @ w unit variance in expectation, at dimension L_.

    Unit variance at every L_ is what matches Bayes ranking difficulty across k:
    |Δr| strata then mean the same thing in every dimension.
    """
    wr = w_raw(L_)
    var_r = (1 - gC**2) * np.sum(wr**2) + gC**2 * np.sum(wr)**2
    return wr / np.sqrt(var_r)


def _compute_wP(beta_info: float, M_: int, gA: float,
                legacy: bool = False) -> np.ndarray:
    """Proxy weights for the informative condition.

    Proxies share eta_A, so Cov(P_m, P_m') = gA^2 for m != m'. The naive
    beta_info/sqrt(M_) gives Var(P @ w_P) = beta_info^2 * [1 + (M_-1) gA^2],
    which grows with M_ and would confound dimension with signal strength.
    Normalizing by the true correlated variance keeps it at beta_info^2.
    """
    if legacy:
        return np.full(M_, beta_info / np.sqrt(M_))
    var_unit = M_ + M_ * (M_ - 1) * gA**2      # Var(sum P_m) for unit coefs
    return np.full(M_, beta_info / np.sqrt(var_unit))


def _build_legit(S: np.ndarray, eta_C: np.ndarray, gC: float) -> np.ndarray:
    return np.sqrt(1 - gC**2) * S + gC * eta_C[:, None]


def _build_proxy(eta_A: np.ndarray, eta_C: np.ndarray, proxy_noise: np.ndarray,
                 gA: float, gCp: float) -> np.ndarray:
    noise_var = 1.0 - gA**2 - gCp**2
    if noise_var <= 0:
        raise ValueError(f"invalid params: 1 - gA^2 - gCp^2 = {noise_var} ≤ 0")
    return (gA * eta_A[:, None]
            + gCp * eta_C[:, None]
            + np.sqrt(noise_var) * proxy_noise)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(gamma_C, gamma_A, k: int = K, n: int = 5000, seed: int = 0,
             proxy_informative: bool = False, beta_info: float = 0.20,
             sigma_y: float = SIGMA_Y, gCp: float = GCP_DEFAULT,
             standardize: bool = True, legacy_wp: bool = False) -> Sample:
    """Generate n samples under (γ_C, γ_A) at attribute dimension k.

    k must be a key of K_LADDER (6, 12, 18); the L:M split is taken from there
    so the proxy fraction stays constant as k grows.
    """
    if k not in K_LADDER:
        raise ValueError(f"k must be one of {sorted(K_LADDER)}; got {k}")
    L_, M_ = K_LADDER[k]
    params = resolve_params(gamma_C, gamma_A)
    # Hold A_MC constant across k (see MU_A_BY_K).
    if (k, gamma_A) in MU_A_BY_K:
        params["mu_A"] = MU_A_BY_K[(k, gamma_A)]
    elif k != K:
        raise ValueError(f"no calibrated mu_A for k={k}, gamma_A={gamma_A!r}")
    return _generate_raw(params["gC"], params["gA"], params["mu_A"],
                         n=n, seed=seed, L_=L_, M_=M_,
                         proxy_informative=proxy_informative,
                         beta_info=beta_info, sigma_y=sigma_y,
                         gCp=gCp, standardize=standardize, legacy_wp=legacy_wp)


def generate_from_params(gC: float, gA: float, mu_A: float, *, n: int = 5000,
                         seed: int = 0, L_: int = L, M_: int = M,
                         proxy_informative: bool = False,
                         beta_info: float = 0.20, sigma_y: float = SIGMA_Y,
                         gCp: float = GCP_DEFAULT,
                         standardize: bool = True,
                         legacy_wp: bool = False) -> Sample:
    return _generate_raw(gC, gA, mu_A, n=n, seed=seed, L_=L_, M_=M_,
                         proxy_informative=proxy_informative,
                         beta_info=beta_info, sigma_y=sigma_y,
                         gCp=gCp, standardize=standardize, legacy_wp=legacy_wp)


def _generate_raw(gC, gA, mu_A, *, n, seed, proxy_informative, beta_info,
                  sigma_y, gCp, standardize, L_=L, M_=M, legacy_wp=False):
    rng = np.random.default_rng(int(seed))

    # Exogenous draws
    A = rng.integers(0, 2, size=n).astype(int)
    A_signed = 2 * A - 1
    eta_C = rng.standard_normal(n)
    u_A = rng.standard_normal(n)
    S = rng.standard_normal((n, L_))
    proxy_noise = rng.standard_normal((n, M_))
    y_noise = rng.standard_normal(n) * sigma_y

    eta_A = mu_A * A_signed + np.sqrt(1 - mu_A**2) * u_A
    Xleg = _build_legit(S, eta_C, gC)
    P = _build_proxy(eta_A, eta_C, proxy_noise, gA, gCp)

    w = _compute_w(gC, L_)
    r = Xleg @ w  # legit risk score, unit variance at every L_

    if proxy_informative:
        # Independent Y-signal loaded on proxies. In the counterfactual (flip A),
        # eta_A changes → P changes → E[Y|X] changes → PSE ≠ 0 (as intended for
        # the informative condition, which is a robustness comparison).
        w_P = _compute_wP(beta_info, M_, gA, legacy=legacy_wp)
    else:
        w_P = np.zeros(M_)

    r_true = r + P @ w_P
    Y = r_true + y_noise

    X_unstd = np.hstack([Xleg, P])
    if standardize:
        col_mu = X_unstd.mean(axis=0, keepdims=True)
        col_sd = X_unstd.std(axis=0, keepdims=True)
        col_sd = np.where(col_sd > 0, col_sd, 1.0)
        X = (X_unstd - col_mu) / col_sd
    else:
        X = X_unstd

    return Sample(
        A=A, X=X, Xleg=Xleg, P=P, r=r, r_true=r_true, Y=Y,
        S=S, eta_C=eta_C, u_A=u_A, proxy_noise=proxy_noise, y_noise=y_noise,
        gC=gC, gA=gA, mu_A=mu_A, gCp=gCp,
        proxy_informative=proxy_informative, beta_info=beta_info,
        w=w, w_P=w_P,
    )


def counterfactual(sample: Sample, new_A, freeze_scaler: bool = False) -> Sample:
    """SCM nested counterfactual on A.

    Regenerate eta_A and proxies with new A (using the SAME exogenous noise);
    legit attributes / r / Y unchanged (they never depend on A).

    freeze_scaler=False reproduces the frozen historical behaviour: X is
    re-standardized over the counterfactual sample, which can shift a rendered
    comparator value by ~1/n (at 2-decimal rendering, 5/99 comparators at
    k=12; sensitivity analysis shows no headline estimate moves > 1.5 pp).
    freeze_scaler=True reuses the FACTUAL sample's standardization constants,
    so untouched rows render identically. Use for future runs.
    """
    n = len(sample)
    if np.isscalar(new_A):
        new_A_arr = np.full(n, int(new_A), dtype=int)
    else:
        new_A_arr = np.asarray(new_A, dtype=int)
        if new_A_arr.shape != (n,):
            raise ValueError(f"new_A shape {new_A_arr.shape} != ({n},)")

    A_signed_new = 2 * new_A_arr - 1
    eta_A_new = sample.mu_A * A_signed_new + np.sqrt(1 - sample.mu_A**2) * sample.u_A
    P_new = _build_proxy(eta_A_new, sample.eta_C, sample.proxy_noise,
                         sample.gA, sample.gCp)

    # In the zero-info condition (w_P=0), r_true = r_legit is unchanged → the
    # oracle PSE truth is exactly 0. In the informative condition, r_true
    # changes because P changes (this is intended: informative proxies carry
    # genuine signal, so the Bayes-optimal ranking should shift with A).
    r_true_new = sample.r + P_new @ sample.w_P
    Y_new = r_true_new + sample.y_noise

    X_unstd = np.hstack([sample.Xleg, P_new])
    if freeze_scaler:
        base = np.hstack([sample.Xleg, sample.P])
        col_mu = base.mean(axis=0, keepdims=True)
        col_sd = base.std(axis=0, keepdims=True)
    else:
        col_mu = X_unstd.mean(axis=0, keepdims=True)
        col_sd = X_unstd.std(axis=0, keepdims=True)
    col_sd = np.where(col_sd > 0, col_sd, 1.0)
    X = (X_unstd - col_mu) / col_sd

    return Sample(
        A=new_A_arr, X=X, Xleg=sample.Xleg, P=P_new,
        r=sample.r, r_true=r_true_new, Y=Y_new,
        S=sample.S, eta_C=sample.eta_C, u_A=sample.u_A,
        proxy_noise=sample.proxy_noise, y_noise=sample.y_noise,
        gC=sample.gC, gA=sample.gA, mu_A=sample.mu_A, gCp=sample.gCp,
        proxy_informative=sample.proxy_informative, beta_info=sample.beta_info,
        w=sample.w, w_P=sample.w_P,
    )


# ---------------------------------------------------------------------------
# Manipulation-check statistics
# ---------------------------------------------------------------------------

def compute_C_MC(sample: Sample) -> float:
    """C_MC = λ_max(Σ_{X|A}) / trace(Σ_{X|A})."""
    A = sample.A
    Xc = sample.X.copy()
    for a in (0, 1):
        mask = (A == a)
        if mask.any():
            Xc[mask] -= Xc[mask].mean(axis=0, keepdims=True)
    Sigma = np.cov(Xc, rowvar=False, bias=False)
    eig = np.linalg.eigvalsh(Sigma)
    return float(eig.max() / eig.sum())


def compute_A_MC(sample: Sample) -> float:
    """A_MC = R^2(A | P) via OLS."""
    P = sample.P
    A = sample.A.astype(float)
    X = np.hstack([np.ones((P.shape[0], 1)), P])
    beta, *_ = np.linalg.lstsq(X, A, rcond=None)
    Ahat = X @ beta
    ss_res = float(((A - Ahat) ** 2).sum())
    ss_tot = float(((A - A.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Pair construction (common target evaluation)
# ---------------------------------------------------------------------------

DEFAULT_STRATA = ((0.3, 0.6), (0.6, 1.0), (1.0, 1.5))  # r-SD = 1 by construction


def make_pairs(sample: Sample, n_per_stratum: int = 10,
               strata=DEFAULT_STRATA, seed: int = 0):
    """Stratify latent pairs by |r_i - r_j|.

    Returns list of dicts: {i, j, stratum, d}, where i is the index of the higher-r
    individual. The pair indices refer to positions in the original sample; the
    same latent pair can be re-surfaced under other structural conditions by
    calling _generate_raw with the same seed and the alternate (gC, gA, mu_A).
    """
    rng = np.random.default_rng(int(seed))
    n = len(sample)
    r = sample.r

    pairs_by_stratum = [[] for _ in strata]
    max_draws = n_per_stratum * len(strata) * 500
    draws = 0
    seen = set()
    while any(len(b) < n_per_stratum for b in pairs_by_stratum) and draws < max_draws:
        i, j = rng.integers(0, n, size=2)
        draws += 1
        if i == j:
            continue
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key in seen:
            continue
        d = abs(r[i] - r[j])
        for k_s, (lo, hi) in enumerate(strata):
            if lo <= d < hi and len(pairs_by_stratum[k_s]) < n_per_stratum:
                a, b = (i, j) if r[i] > r[j] else (j, i)
                pairs_by_stratum[k_s].append(
                    dict(i=int(a), j=int(b), stratum=k_s, d=float(d))
                )
                seen.add(key)
                break

    return [p for bucket in pairs_by_stratum for p in bucket]


__all__ = [
    "L", "M", "K", "W_RAW", "SIGMA_Y", "FROZEN_PARAMS",
    "K_LADDER", "MU_A_BY_K", "w_raw",
    "resolve_params", "generate", "generate_from_params", "counterfactual",
    "compute_C_MC", "compute_A_MC", "make_pairs",
    "Sample",
]
