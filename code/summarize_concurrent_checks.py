#!/usr/bin/env python3
"""Write data/concurrent_checks/analysis/summary.json from the stored answers.

Uses the same estimators and bootstrap seeds as concurrent_checks.py. No model
calls and no clinical panel are needed (learner references are omitted here
and reported by `concurrent_checks.py analyze-neartie`).
"""
import csv
import json
from pathlib import Path

import numpy as np

import concurrent_checks as cc

ROOT = Path(__file__).resolve().parent.parent / "data/concurrent_checks"
M = ROOT / cc.MODEL


def interval(values):
    m, lo, hi, l9, h9 = cc.boot(np.asarray(values, float))
    return dict(effect_pp=m, lo_pp=lo, hi_pp=hi, lo90_pp=l9, hi90_pp=h9, n=len(values))


def main() -> None:
    out = {}
    rows = list(csv.DictReader(open(ROOT / "links/neartie_links.csv")))
    ans = cc.read_answers([M / "neartie_jobs_answers.jsonl"])
    for sample in ("neartie", "wide"):
        pids, c, ch, _, missing = cc.pair_contrasts(rows, ans, sample)
        out[sample] = dict(all=interval(c), changed=interval(c[ch]), missing=missing)
    _, cn, _, _, _ = cc.pair_contrasts(rows, ans, "neartie")
    _, cw, _, _, _ = cc.pair_contrasts(rows, ans, "wide")
    rng = np.random.default_rng(2)
    dd = [cn[rng.integers(0, len(cn), len(cn))].mean()
          - cw[rng.integers(0, len(cw), len(cw))].mean() for _ in range(cc.B)]
    out["neartie_minus_wide"] = dict(effect_pp=100 * (cn.mean() - cw.mean()),
                                     lo_pp=100 * np.percentile(dd, 2.5),
                                     hi_pp=100 * np.percentile(dd, 97.5))
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis/summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
