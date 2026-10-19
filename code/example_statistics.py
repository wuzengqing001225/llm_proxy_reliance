#!/usr/bin/env python3
"""Same-example learner references and example statistics for the twelve-field
LH cells. No model calls.

Writes data/summaries/example_statistics.csv with one row per example set:
  design            example_count, dose_response or informative_grid
  k, n_examples, beta
  learner_pse_pp    same-example learner effect on the evaluation pairs
  learner_error     same-example learner pairwise error rate
  proxy_outcome_r   mean sample correlation between each proxy and the
                    example outcomes
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "modules"))
import config  # noqa: E402
import dgp_k  # noqa: E402
import dgp_norm  # noqa: E402
import oracle  # noqa: E402
sys.modules["dgp"] = dgp_k
from run_cell import build_cell  # noqa: E402

M_BY_K = {6: 2, 12: 4, 18: 6}


def row(design, k, n, beta, ref, pairs, x, y):
    post = oracle.blr_fit(x, y, sigma=dgp_k.SIGMA_Y, tau=1.0)
    m = M_BY_K[k]
    r = [np.corrcoef(x[:, -m + j], y)[0, 1] for j in range(m)]
    return dict(design=design, k=k, n_examples=n, beta="" if beta is None else beta,
                learner_pse_pp=round(dgp_norm.ideal_line(post, ref, pairs), 4),
                learner_error=round(oracle.ideal_pairwise_error_rate(ref, pairs, post), 4),
                proxy_outcome_r=round(float(np.mean(r)), 4))


def main() -> None:
    rows = []
    gc, ga = config.CORNERS["LH"]
    ref, pairs, _, _, _ = build_cell(12, "LH", "neutral", "learned", False)
    for n in (12, 20, 80):
        d = dgp_k.generate(gc, ga, k=12, n=n, seed=config.DCAL_SEED)
        rows.append(row("example_count", 12, n, None, ref, pairs, d.X, d.Y))
    for beta in (0.0, 0.30, 0.45, 0.60):
        if beta == 0.0:
            ref, pairs, x, y, _ = build_cell(12, "LH", "neutral", "learned", False)
        else:
            ref, pairs, x, y, _ = build_cell(12, "LH", "neutral", "learned", True, beta)
        rows.append(row("dose_response", 12, config.N_CAL, beta, ref, pairs, x, y))
    for k in (6, 12, 18):
        ref, pairs, x, y, _ = build_cell(k, "LH", "neutral", "learned", True)
        rows.append(row("informative_grid", k, config.N_CAL, config.BETA_INFO,
                        ref, pairs, x, y))
    out = HERE.parent / "data/summaries/example_statistics.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
