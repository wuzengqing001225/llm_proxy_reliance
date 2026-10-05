"""Clinical field definitions and conditional category substitution utilities."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LEGIT = [("num_lab_procedures", "Lab procedures this stay"),
         ("num_procedures", "Other procedures this stay"),
         ("num_medications", "Medications this stay"),
         ("number_outpatient", "Outpatient visits, prior year"),
         ("number_emergency", "Emergency visits, prior year"),
         ("number_inpatient", "Inpatient stays, prior year"),
         ("time_in_hospital", "Days in hospital"),
         ("number_diagnoses", "Number of diagnoses"), ("age_mid", "Age")]
PAYER_LABEL = {"MC": "Medicare", "MD": "Medicaid", "SP": "Self-pay", "BC": "Blue Cross/Blue Shield"}
ADM_GROUP = {7: "Emergency room", 1: "Referral", 2: "Referral", 3: "Referral", 4: "Transfer",
             5: "Transfer", 6: "Transfer", 10: "Transfer", 18: "Transfer", 22: "Transfer",
             25: "Transfer", 26: "Transfer"}
ADM_EXPECTED = {1: "physician referral", 2: "clinic referral", 3: "hmo referral",
                4: "transfer from a hospital", 5: "transfer from a skilled nursing facility",
                6: "transfer from another health care facility", 7: "emergency room"}
NOT_RECORDED = "Not recorded or other"


def set_scheme(panel, scheme):
    if scheme != "full":
        raise ValueError("The released experiment uses the frozen full category scheme")
    panel["joint"] = panel[["payer", "adm"]].astype(str).agg(" | ".join, axis=1)


@dataclass
class Design:
    panel: pd.DataFrame
    cats: list
    Lz: np.ndarray
    s: np.ndarray
    s_of: dict
    wL: np.ndarray
    corr: float
    model: LogisticRegression
    U: np.ndarray
    cal_idx: np.ndarray
    fit_idx: np.ndarray
    pairs: list = field(default_factory=list)
    changes: np.ndarray | None = None


def conditional_dist(design, idx, group):
    features = np.column_stack([np.full(len(idx), group), design.Lz[idx]])
    probabilities = design.model.predict_proba(features)
    order = [list(design.model.classes_).index(c) for c in design.cats]
    return probabilities[:, order]


def category_under(design, t, group):
    cumulative = np.cumsum(conditional_dist(design, np.array([t]), group)[0])
    k = int(np.searchsorted(cumulative, design.U[t], side="right"))
    return design.cats[min(k, len(design.cats) - 1)]


def ideal_features(design, idx, cats):
    levels = [sorted({c.split(" | ")[f] for c in design.cats}) for f in range(2)]
    blocks = [design.Lz[idx]]
    for f in range(2):
        blocks.append(np.array([[c.split(" | ")[f] == v for v in levels[f][1:]] for c in cats], float))
    return np.column_stack(blocks)
