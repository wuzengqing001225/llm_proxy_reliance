"""
coupling_analysis.py — formal test of the capability / proxy-use coupling claim.

PAPER CLAIM UNDER TEST (three-layer framework, layer 1, third sub-question):
    "(k, dependency structure) -> { how does capability change? how does proxy
     use change? ARE THE TWO COUPLED? }"

The coupling claim is a claim about CONDITIONS: do conditions that degrade
capability also raise impermissible proxy reliance? It is NOT a claim that a
given decision is wrong *because* the model leaned on a proxy. So the primary
test is between-condition; a pair-level probe is secondary and mechanistic.

WHY THE EARLIER NUMBER WAS NOT A TEST
    Gate 2 gave 6 cells per informativeness level, all at ONE structural corner
    (LH). Between-cell variation was therefore k + semantics only, and k drives
    BOTH G_ICL and PSE, so the raw correlation is confounded by a common cause.
    Partialling k out of n=6 leaves 3 df — uninterpretable either way.

DESIGN THIS PROGRAM ANALYSES
    12 cells = 3 k-levels x 4 structural corners, rule-learned, neutral labels,
    zero-information proxies. Common-target: the evaluation pairs are FIXED (the
    LH reference sample, seed 10001, make_pairs seed 30001) for a given k, and
    only the corner of D_cal varies. Capability differences therefore come from
    what the model induced from the calibration set, not from different test
    items -- and the LH cells already run in Gate 2 pool in unchanged.

TWO STATISTICAL TRAPS, AND HOW THEY ARE HANDLED
  (1) Correlating two NOISILY ESTIMATED quantities across 12 cells.
      A naive Pearson CI treats each cell's G_ICL and PSE as exact. They are
      not; measurement error attenuates the correlation and mis-states its CI.
      Handled by `coupling_between_cells`: a cluster bootstrap that resamples
      the COMMON pair set once per replicate, recomputes every cell's G_ICL and
      PSE from that same resample, then recomputes the correlation. Using one
      shared resample preserves the common-target pairing across cells (which
      is the design's precision advantage) and propagates estimation error into
      the correlation's CI.

  (2) Response instability inflates BOTH error and |PSE| at the pair level.
      A pair the model answers 50/50 looks error-prone AND A-sensitive purely
      from response variance, with no coupling whatsoever. Handled by
      `coupling_within_cell`: significance comes from a WITHIN-CELL PERMUTATION
      null that shuffles the pse<->err pairing while leaving both marginal
      distributions exactly intact. Any noise-driven inflation of the marginals
      is present in the null too, so it cannot manufacture a positive result.
      (An order-disagreement covariate is NOT used as a control: with 2 orders,
      factual disagreement is 1 iff err_p == 0.5, i.e. collinear with the
      outcome.)

CEILING CHECK
    G_ICL = R_LLM - R_Ideal has room to move only while R_Ideal sits well below
    chance. At k=18 the high-gamma_C corners give R_Ideal ~0.19-0.23 because the
    calibration covariance no longer matches the LH targets. `ceiling_report`
    flags any cell whose headroom (0.5 - R_Ideal) is under 0.30 so it can be
    excluded from, or reported separately in, the primary fit.

Every estimate is recomputed from the raw per-call records; no sub-agent's
self-reported scalar is trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

# ----------------------------------------------------------------------------
# Frozen design constants
# ----------------------------------------------------------------------------
CORNERS = {"LL": ("low", "low"), "LH": ("low", "high"),
           "HL": ("high", "low"), "HH": ("high", "high")}
K_LEVELS = (6, 12, 18)
TARGET_GC, TARGET_GA = "low", "high"   # fixed target sample = LH, all corners
TARGET_N, TARGET_SEED, PAIR_SEED = 5000, 10001, 30001
DCAL_N, DCAL_SEED = 80, 517
N_PER_STRATUM = 33
CHANCE = 0.5
HEADROOM_MIN = 0.30                    # 0.5 - R_Ideal must exceed this


# ----------------------------------------------------------------------------
# Per-pair quantities from raw records
# ----------------------------------------------------------------------------
def per_pair_error(records: pd.DataFrame, r_true: np.ndarray) -> dict:
    """Factual arm -> {(i,j): error rate over orders}. Correct = higher r_true."""
    df = records[(records["arm"] == "factual") & records["picked_i"].notna()]
    acc: dict = {}
    for row in df.itertuples():
        i, j, chosen = int(row.i), int(row.j), int(row.chosen)
        wrong = int(r_true[chosen] < max(r_true[i], r_true[j]))
        acc.setdefault((i, j), []).append(wrong)
    return {p: float(np.mean(v)) for p, v in acc.items()}


def per_pair_pse(records: pd.DataFrame) -> dict:
    """cf_a1 / cf_a0 arms -> {(i,j): pi(A->1) - pi(A->0)} averaged over orders."""
    out = {}
    for arm in ("cf_a1", "cf_a0"):
        df = records[(records["arm"] == arm) & records["picked_i"].notna()]
        acc: dict = {}
        for row in df.itertuples():
            acc.setdefault((int(row.i), int(row.j)), []).append(float(row.picked_i))
        out[arm] = {p: float(np.mean(v)) for p, v in acc.items()}
    common = set(out["cf_a1"]) & set(out["cf_a0"])
    return {p: out["cf_a1"][p] - out["cf_a0"][p] for p in common}


def build_cell_table(cells: dict, r_true_by_k: dict) -> pd.DataFrame:
    """cells: {(k, corner): records DataFrame} -> tidy per-pair frame.

    Returns one row per (k, corner, pair) with `err` and `pse`, restricted to
    pairs present in every arm of that cell.
    """
    rows = []
    for (k, corner), rec in cells.items():
        missing = {"arm", "i", "j", "picked_i", "chosen"} - set(rec.columns)
        if missing:
            raise ValueError(f"cell {(k, corner)} missing columns {sorted(missing)}")
        err = per_pair_error(rec, r_true_by_k[k])
        pse = per_pair_pse(rec)
        for p in sorted(set(err) & set(pse)):
            rows.append(dict(k=k, corner=corner, i=p[0], j=p[1],
                             err=err[p], pse=pse[p]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Cell-level estimates
# ----------------------------------------------------------------------------
def cell_estimates(tidy: pd.DataFrame, r_ideal: dict) -> pd.DataFrame:
    """Point estimates of R_LLM, G_ICL and PSE for every cell (pp for G/PSE)."""
    rows = []
    for (k, corner), g in tidy.groupby(["k", "corner"]):
        ri = r_ideal[(k, corner)]
        rows.append(dict(k=k, corner=corner, n_pairs=len(g),
                         R_LLM=g.err.mean(), R_Ideal=ri,
                         G_ICL_pp=(g.err.mean() - ri) * 100,
                         PSE_pp=g.pse.mean() * 100,
                         headroom=CHANCE - ri))
    return pd.DataFrame(rows).sort_values(["k", "corner"]).reset_index(drop=True)


def ceiling_report(est: pd.DataFrame) -> pd.DataFrame:
    """Flag cells where G_ICL has too little room to move to be informative."""
    out = est.copy()
    out["ceiling_ok"] = out.headroom >= HEADROOM_MIN
    return out[["k", "corner", "R_Ideal", "headroom", "ceiling_ok", "G_ICL_pp", "PSE_pp"]]


# ----------------------------------------------------------------------------
# PRIMARY TEST — between-cell coupling, k controlled, error propagated
# ----------------------------------------------------------------------------
def _corr_and_partial(g_vals, p_vals, k_vals):
    """Pearson r(G, PSE) and the same partialling out k."""
    r12 = sps.pearsonr(g_vals, p_vals)[0]
    r1c = sps.pearsonr(g_vals, k_vals)[0]
    r2c = sps.pearsonr(p_vals, k_vals)[0]
    denom = np.sqrt((1 - r1c ** 2) * (1 - r2c ** 2))
    pr = (r12 - r1c * r2c) / denom if denom > 1e-12 else np.nan
    return r12, pr


def coupling_between_cells(tidy: pd.DataFrame, r_ideal: dict,
                           B: int = 5000, seed: int = 101,
                           cells_ok: set | None = None) -> dict:
    """Bootstrap CI for the between-cell G_ICL <-> PSE correlation.

    One shared resample of the common pair set per replicate: this preserves the
    common-target pairing across cells and pushes each cell's estimation error
    into the correlation's sampling distribution.
    """
    t = tidy if cells_ok is None else tidy[
        [(k, c) in cells_ok for k, c in zip(tidy.k, tidy.corner)]]
    if t.empty:
        raise ValueError("no cells left after the ceiling filter")

    keys = sorted({(k, c) for k, c in zip(t.k, t.corner)})
    # pair index shared across cells, per k (targets are common within a k)
    pairs_by_k = {k: sorted({(i, j) for kk, i, j in
                             zip(t.k, t.i, t.j) if kk == k}) for k in sorted(set(t.k))}
    lut = {(k, c): dict(zip(zip(g.i, g.j), zip(g.err, g.pse)))
           for (k, c), g in t.groupby(["k", "corner"])}

    est = cell_estimates(t, r_ideal)
    r_obs, pr_obs = _corr_and_partial(est.G_ICL_pp.values, est.PSE_pp.values,
                                      est.k.values.astype(float))

    rng = np.random.default_rng(seed)
    boot_r, boot_pr = [], []
    for _ in range(B):
        draw = {k: [pairs_by_k[k][x] for x in
                    rng.integers(0, len(pairs_by_k[k]), len(pairs_by_k[k]))]
                for k in pairs_by_k}
        gs, ps, ks = [], [], []
        for (k, c) in keys:
            d = lut[(k, c)]
            vals = [d[p] for p in draw[k] if p in d]
            if not vals:
                break
            e = np.mean([v[0] for v in vals])
            s = np.mean([v[1] for v in vals])
            gs.append((e - r_ideal[(k, c)]) * 100)
            ps.append(s * 100)
            ks.append(float(k))
        if len(gs) != len(keys):
            continue
        r_b, pr_b = _corr_and_partial(np.array(gs), np.array(ps), np.array(ks))
        boot_r.append(r_b)
        boot_pr.append(pr_b)

    boot_r = np.asarray(boot_r)
    boot_pr = np.asarray(boot_pr)
    boot_pr = boot_pr[np.isfinite(boot_pr)]
    return dict(n_cells=len(keys),
                r_raw=r_obs, r_ci=tuple(np.percentile(boot_r, [2.5, 97.5])),
                r_excludes_zero=bool(np.percentile(boot_r, 2.5) > 0
                                     or np.percentile(boot_r, 97.5) < 0),
                r_partial_k=pr_obs,
                pr_ci=tuple(np.percentile(boot_pr, [2.5, 97.5])),
                pr_excludes_zero=bool(np.percentile(boot_pr, 2.5) > 0
                                      or np.percentile(boot_pr, 97.5) < 0),
                n_boot=len(boot_r))


def structural_movement_at_fixed_k(est: pd.DataFrame) -> pd.DataFrame:
    """Does the structural manipulation move BOTH quantities at fixed k?

    This is the design's cleanest coupling evidence: within a k-level the test
    items are identical, so any co-movement across corners cannot be a k effect.
    """
    rows = []
    for k, g in est.groupby("k"):
        rows.append(dict(k=k, n_corners=len(g),
                         G_spread_pp=g.G_ICL_pp.max() - g.G_ICL_pp.min(),
                         PSE_spread_pp=g.PSE_pp.max() - g.PSE_pp.min(),
                         r_within_k=(sps.pearsonr(g.G_ICL_pp, g.PSE_pp)[0]
                                     if len(g) > 2 else np.nan),
                         argmax_G=g.loc[g.G_ICL_pp.idxmax(), "corner"],
                         argmax_PSE=g.loc[g.PSE_pp.idxmax(), "corner"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# SECONDARY TEST — pair-level, permutation null (noise-immune)
# ----------------------------------------------------------------------------
def coupling_within_cell(tidy: pd.DataFrame, B: int = 5000,
                         seed: int = 202) -> pd.DataFrame:
    """Within each cell: does |pse_p| covary with err_p across pairs?

    p-value from permuting the pse<->err pairing WITHIN the cell. Both marginals
    are preserved exactly, so response instability -- which inflates |pse| and
    err together -- is present in the null and cannot produce a false positive.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (k, corner), g in tidy.groupby(["k", "corner"]):
        e = g.err.to_numpy(float)
        a = np.abs(g.pse.to_numpy(float))
        if e.std() < 1e-12 or a.std() < 1e-12:
            rows.append(dict(k=k, corner=corner, n_pairs=len(g), r_obs=np.nan,
                             p_perm=np.nan, note="degenerate (no variance)"))
            continue
        r_obs = sps.pearsonr(e, a)[0]
        null = np.array([sps.pearsonr(e, rng.permutation(a))[0] for _ in range(B)])
        p = (1 + np.sum(np.abs(null) >= abs(r_obs))) / (1 + B)   # two-sided
        rows.append(dict(k=k, corner=corner, n_pairs=len(g), r_obs=r_obs,
                         p_perm=p, note=""))
    return pd.DataFrame(rows)


def stouffer_combine(p_values, r_values) -> dict:
    """Combine per-cell permutation p-values, sign-aware (Stouffer's Z)."""
    p = np.asarray([x for x in p_values if np.isfinite(x)], float)
    r = np.asarray([x for x, q in zip(r_values, p_values) if np.isfinite(q)], float)
    if len(p) == 0:
        return dict(z=np.nan, p=np.nan, k=0)
    z = np.sign(r) * sps.norm.isf(p / 2)
    z_comb = z.sum() / np.sqrt(len(z))
    return dict(z=float(z_comb), p=float(2 * sps.norm.sf(abs(z_comb))), k=len(z))
