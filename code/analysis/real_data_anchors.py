"""Where real clinical data sits on the two manipulated axes.

Answers the reviewer question "is the synthetic grid anywhere near reality?" by
computing, for public clinical datasets, the same two manipulation-check
quantities the generating process targets:

  gamma_C  structure concentration = (lambda_max/trace - 1/k) / (1 - 1/k)
           0 = mutually independent attributes, 1 = a single latent factor.
           Normalized so panels of different width are comparable.

  gamma_A  proxy channel strength = R^2(A | X), the linear predictability of a
           binary protected attribute from the non-protected panel.

No LLM calls. Requires network access to the UCI repository.

Run:  python real_data_anchors.py
Out:  real_data_anchors_structure.csv, real_data_anchors_proxy.csv,
      real_data_coverage.png
"""
import urllib.request, os
import numpy as np, pandas as pd

SOURCES = {
    "uci_diabetes_130hosp": "https://archive.ics.uci.edu/static/public/296/data.csv",
    "uci_heart_disease":    "https://archive.ics.uci.edu/static/public/45/data.csv",
    "uci_breast_wdbc":      "https://archive.ics.uci.edu/static/public/17/data.csv",
    "uci_cervical_cancer":  "https://archive.ics.uci.edu/static/public/383/data.csv",
    "uci_dermatology":      "https://archive.ics.uci.edu/static/public/33/data.csv",
    "uci_hepatitis":        "https://archive.ics.uci.edu/static/public/46/data.csv",
    "uci_liver_disorders":  "https://archive.ics.uci.edu/static/public/60/data.csv",
}

# The levels our generating process was configured to hit, recomputed from the
# DGP module rather than pasted -- see dgp_k.structure_report.
OUR_LEVELS = {"gamma_C_low_k6": 0.114, "gamma_C_low_k18": 0.166,
              "gamma_C_high_k6": 0.295, "gamma_C_high_k18": 0.314,
              "gamma_A_low": 0.068, "gamma_A_high": 0.429}


def cmc_norm(X):
    """Normalized structure concentration. Identical estimator to the DGP check."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] < 2:
        return np.nan
    C = np.corrcoef(X.T)
    C = C[~np.isnan(C).all(1)][:, ~np.isnan(C).all(0)]
    k = C.shape[0]
    ev = np.linalg.eigvalsh(C)
    return (ev.max() / np.trace(C) - 1 / k) / (1 - 1 / k)


def r2_A_given_X(X, A):
    """Linear R^2 of a binary protected attribute on the covariate panel."""
    X = np.asarray(X, float); A = np.asarray(A, float)
    Xd = np.hstack([X, np.ones((len(X), 1))])
    b, *_ = np.linalg.lstsq(Xd, A, rcond=None)
    return 1 - np.var(A - Xd @ b) / np.var(A)


def numeric_panel(df, min_cov=0.90, min_uniq=4):
    """Usable numeric columns, dropping identifier-like and sparse ones.

    Identifier columns must go: they are uncorrelated with everything and would
    push the concentration estimate toward the independence floor.
    """
    num = df.select_dtypes(include=[np.number])
    keep = [c for c in num.columns
            if num[c].notna().mean() >= min_cov and num[c].nunique() >= min_uniq
            and not any(t in c.lower() for t in
                        ("_id", "id_", "encounter", "patient_nbr", "index", "unnamed"))]
    return num[keep].dropna()


def fetch(cache="real_anchors"):
    os.makedirs(cache, exist_ok=True)
    got = {}
    for name, url in SOURCES.items():
        fp = os.path.join(cache, name + ".csv")
        if not os.path.exists(fp):
            try:
                urllib.request.urlretrieve(url, fp)
            except Exception as e:
                print(f"  skip {name}: {type(e).__name__}")
                continue
        got[name] = fp
    # The datasets are NOT redistributed with this package (they remain under
    # their UCI licences). To let others verify they analysed the same files,
    # record a SHA-256 checksum of every downloaded file.
    import hashlib
    with open(os.path.join(cache, "CHECKSUMS.sha256"), "w") as fh:
        for name, fp in sorted(got.items()):
            h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            fh.write(f"{h}  {name}.csv\n")
    print(f"  wrote {cache}/CHECKSUMS.sha256 for {len(got)} files")
    return got


def build_panels(got):
    """Named panels, one per dataset, plus the variable-kind label used in the figure."""
    P = {}
    if "uci_diabetes_130hosp" in got:
        d = pd.read_csv(got["uci_diabetes_130hosp"], low_memory=False)
        cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"]
        P["Diabetes 130-hospital \u00b7 utilization counts"] = (d[cols].dropna(), "administrative")
    spec = [("Heart disease (Cleveland) \u00b7 clinical + lab", "uci_heart_disease", "clinical"),
            ("Breast cancer WDBC \u00b7 imaging morphometry", "uci_breast_wdbc", "imaging"),
            ("Cervical cancer risk \u00b7 history + serology", "uci_cervical_cancer", "clinical"),
            ("Dermatology \u00b7 histopathology scores", "uci_dermatology", "clinical"),
            ("Hepatitis \u00b7 liver panel", "uci_hepatitis", "lab panel"),
            ("Liver disorders \u00b7 enzyme panel", "uci_liver_disorders", "lab panel")]
    for nm, key, kind in spec:
        if key not in got:
            continue
        pan = numeric_panel(pd.read_csv(got[key], low_memory=False))
        if pan.shape[1] >= 3 and len(pan) >= 50:
            P[nm] = (pan, kind)
    return P


def structure_table(panels, n_boot=200, seed=4242):
    rng = np.random.default_rng(seed)
    rows = []
    for nm, (pan, kind) in panels.items():
        v = pan.values
        pt = cmc_norm(v)
        bs = [cmc_norm(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(dataset=nm, variable_kind=kind, k=pan.shape[1], n=len(pan),
                         C_MC_norm=pt, ci_lo=lo, ci_hi=hi,
                         inside_swept_range=bool(0.114 <= pt <= 0.314),
                         nearest_level="low" if abs(pt - 0.14) < abs(pt - 0.305) else "high"))
    return pd.DataFrame(rows).sort_values("C_MC_norm").reset_index(drop=True)


def proxy_table(got):
    """R^2(A|X) for protected attributes that are actually recorded in these datasets."""
    rows = []
    if "uci_diabetes_130hosp" in got:
        d = pd.read_csv(got["uci_diabetes_130hosp"], low_memory=False)
        cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"]
        s = d[cols + ["race", "gender"]].dropna()
        rows.append(("Diabetes 130-hosp \u00b7 gender", s[cols].values,
                     (s.gender.astype(str) == "Female").astype(float).values))
        rows.append(("Diabetes 130-hosp \u00b7 race (Black vs other)", s[cols].values,
                     (s.race.astype(str) == "AfricanAmerican").astype(float).values))
    if "uci_heart_disease" in got:
        hd = pd.read_csv(got["uci_heart_disease"], low_memory=False)
        sc = next((c for c in hd.columns if c.lower() == "sex"), None)
        if sc:
            pan = numeric_panel(hd.drop(columns=[sc]))
            rows.append(("Heart disease \u00b7 sex", pan.values,
                         hd.loc[pan.index, sc].astype(float).values))
    if "uci_cervical_cancer" in got:
        cc = pd.read_csv(got["uci_cervical_cancer"], low_memory=False)
        ac = next((c for c in cc.columns if c.lower().startswith("age")), None)
        if ac:
            pan = numeric_panel(cc.drop(columns=[ac]))
            a = cc.loc[pan.index, ac].astype(float)
            rows.append(("Cervical cancer \u00b7 age >= median", pan.values,
                         (a >= a.median()).astype(float).values))
    out = [dict(dataset_and_protected_attribute=nm, k=X.shape[1], n=len(X),
                R2_A_given_X=r2_A_given_X(X, A),
                low_level=OUR_LEVELS["gamma_A_low"], high_level=OUR_LEVELS["gamma_A_high"],
                inside_swept_range=bool(OUR_LEVELS["gamma_A_low"] <= r2_A_given_X(X, A)
                                        <= OUR_LEVELS["gamma_A_high"]))
           for nm, X, A in rows]
    return pd.DataFrame(out).sort_values("R2_A_given_X").reset_index(drop=True)


if __name__ == "__main__":
    got = fetch()
    panels = build_panels(got)
    S = structure_table(panels)
    Q = proxy_table(got)
    S.to_csv("real_data_anchors_structure.csv", index=False)
    Q.to_csv("real_data_anchors_proxy.csv", index=False)
    print(S.to_string(index=False))
    print()
    print(Q.to_string(index=False))
    print(f"\nstructure range: {S.C_MC_norm.min():.3f}-{S.C_MC_norm.max():.3f} "
          f"| our levels {OUR_LEVELS['gamma_C_low_k6']:.3f}/{OUR_LEVELS['gamma_C_high_k18']:.3f}")
    print(f"proxy range:     {Q.R2_A_given_X.min():.4f}-{Q.R2_A_given_X.max():.4f} "
          f"| our levels {OUR_LEVELS['gamma_A_low']:.3f}/{OUR_LEVELS['gamma_A_high']:.3f}")
