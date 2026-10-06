#!/usr/bin/env python3
"""Same-example learner effect in the zero-information, eighty-example cells.

Writes data/summaries/learner_zero_info.csv (k, corner, learner_pse_pp) for the
LH cells at six, twelve and eighteen fields and the HH cells at six and
eighteen fields. No model calls.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "modules"))
import config  # noqa: E402
import dgp_k  # noqa: E402
import dgp_norm  # noqa: E402
import oracle  # noqa: E402
sys.modules["dgp"] = dgp_k
from run_cell import build_cell  # noqa: E402


def main() -> None:
    out = HERE.parent / "data/summaries/learner_zero_info.csv"
    rows = []
    for k, corner in ((6, "LH"), (12, "LH"), (18, "LH"), (6, "HH"), (18, "HH")):
        ref, pairs, xcal, ycal, _ = build_cell(k, corner, "neutral", "learned", False)
        post = oracle.blr_fit(xcal, ycal, sigma=dgp_k.SIGMA_Y, tau=1.0)
        rows.append(dict(k=k, corner=corner,
                         learner_pse_pp=round(dgp_norm.ideal_line(post, ref, pairs), 4)))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
