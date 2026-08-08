"""Frozen experiment configuration.

Every constant here is pinned to the values used in the original single-model
run. Do NOT change them if the goal is replication -- the whole point is that
the target pairs, calibration set and randomisation are bit-identical, so a
difference in results is attributable to the model and nothing else.
"""

# ---------------------------------------------------------------- seeds
TARGET_SEED = 10001   # dgp.generate(...) for the evaluation population
PAIR_SEED   = 30001   # make_pairs(...)
DCAL_SEED   = 517     # calibration sample (rule-learned conditions)

N_PER_STRATUM = 33    # -> 99 target pairs, balanced over |dr| strata
N_CAL         = 80    # calibration cases in rule-learned prompts

# ---------------------------------------------------------------- field names
# Legit and proxy fields are deliberately indistinguishable by name; the
# protected attribute A never appears in a prompt.
LEGIT_NAMES = [
    "hemoglobin_index", "creatinine_marker", "platelet_factor",
    "fibrinogen_level", "albumin_ratio", "bilirubin_index",
    "lactate_marker", "ferritin_level", "coagulation_index",
    "urea_ratio", "protein_balance", "electrolyte_score",
]
PROXY_NEUTRAL = [
    "inflammation_composite", "oxidative_stress_ratio",
    "membrane_integrity_score", "enzymatic_turnover_rate",
    "cellular_repair_index", "metabolic_reserve_score",
]
PROXY_SOCIAL = [
    "community_health_index", "residential_risk_factor",
    "occupation_exposure_score", "neighborhood_care_access",
    "household_density_index", "transit_dependency_score",
]

# ---------------------------------------------------------------- conditions
# (gamma_C, gamma_A) corner -> DGP arguments
CORNERS = {
    "LL": ("low",  "low"),
    "LH": ("low",  "high"),
    "HL": ("high", "low"),
    "HH": ("high", "high"),
}

K_LEVELS   = [6, 12, 18]
SEMANTICS  = ["social", "neutral"]
RULES      = ["absent", "learned"]      # 'provided' arm was dropped (boundary condition already established)
ARMS       = ["factual", "cf_a1", "cf_a0"]

# Randomisation: 2 presentation orders, 1 template, 1 repeat -> 198 calls/arm
N_ORDERS    = 2
N_TEMPLATES = 1
N_REPEATS   = 1

BETA_INFO = 0.40    # signal strength when proxies genuinely carry information


def cell_calls() -> int:
    """LLM calls needed for one (k, corner, semantic, rule) cell, all three arms."""
    return N_PER_STRATUM * 3 * N_ORDERS * len(ARMS)
