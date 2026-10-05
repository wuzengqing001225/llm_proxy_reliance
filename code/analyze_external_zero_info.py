#!/usr/bin/env python3
"""Analyze complete real-record zero-information call records without API use.

Reports the population-pair proxy-substitution effect, separate display
orders, the effect among targets whose proxy changed, and pair-bootstrap
intervals. It never replaces missing responses with a new call or a zero.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from external_zero_info import ideal_pair_effects, prepare
from run_external_zero_info import file_sha256, selected_pairs


def interval(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if not n:
        return float("nan"), float("nan")
    indices = rng.integers(0, n, size=(draws, n))
    return tuple(float(x) for x in np.quantile(values[indices].mean(1),
                                                (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ids-mapping", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    if args.draws <= 0:
        parser.error("--draws must be positive")
    meta = json.loads(args.metadata.read_text())
    if meta.get("status") != "complete":
        raise SystemExit("Run is partial; do not analyze it as a complete sample")
    served = meta.get("served", [])
    if len(served) != 1 or served[0].get("model") != meta.get("expected_served"):
        raise SystemExit("Mixed or unexpected provider-reported model IDs")
    if meta.get("requested_channel") != "tools=yes reasoning_budget=no temperature=0.0":
        raise SystemExit("Unexpected request channel in metadata")
    if (file_sha256(args.panel) != meta.get("panel_sha256")
            or file_sha256(args.ids_mapping) != meta.get("mapping_sha256")):
        raise SystemExit("Dataset or official mapping differs from run metadata")

    rows = {}
    with args.records.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["pair_id"]), row["arm"], int(row["order"]))
            if key in rows:
                raise SystemExit(f"Duplicate logical record: {key}")
            rows[key] = row
    total_pairs = int(meta["pairs"])
    expected = total_pairs * 4
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} logical records, found {len(rows)}")
    grouped = defaultdict(dict)
    for (pair_id, arm, order), row in rows.items():
        grouped[pair_id][arm, order] = row
    required = {(arm, order) for arm in ("cf_a1", "cf_a0")
                for order in (0, 1)}
    if set(grouped) != set(range(total_pairs)) or any(set(g) != required for g in grouped.values()):
        raise SystemExit("A pair is missing an arm or presentation order")

    pair_effects = []
    complete_pair_ids = []
    order_effects = {0: [], 1: []}
    changed_effects = []
    unchanged_effects = []
    missing = 0
    for pair_id in range(total_pairs):
        g = grouped[pair_id]
        missing += sum(row["picked_i"] == "" for row in g.values())
        if any(row["picked_i"] == "" for row in g.values()):
            continue
        changed = {row["proxy_changed"] for row in g.values()}
        if len(changed) != 1:
            raise SystemExit(f"Proxy change flag differs within pair {pair_id}")
        by_order = [int(g["cf_a1", order]["picked_i"])
                    - int(g["cf_a0", order]["picked_i"])
                    for order in (0, 1)]
        effect = sum(by_order) / 2
        pair_effects.append(effect)
        complete_pair_ids.append(pair_id)
        for order in (0, 1):
            order_effects[order].append(by_order[order])
        (changed_effects if changed == {"1"} else unchanged_effects).append(effect)
    if not pair_effects:
        raise SystemExit("No pair has all four parsed responses")
    if any(x != 0 for x in unchanged_effects):
        raise SystemExit("Identical prompts produced a nonzero arm contrast")

    def report(name: str, values: list[float], seed: int) -> None:
        arr = np.asarray(values, dtype=float)
        lo, hi = interval(arr, args.draws, seed)
        print(f"{name}: n={len(arr)}, effect={100 * arr.mean():+.2f} pp, "
              f"pair-bootstrap 95% [{100 * lo:+.2f}, {100 * hi:+.2f}] pp")

    print(f"run set={meta['run_set']}, pairs planned={total_pairs}, "
          f"pairs complete={len(pair_effects)}, missing logical responses={missing}")
    report("all eligible pairs", pair_effects, 7)
    report("display order 0", order_effects[0], 8)
    report("display order 1", order_effects[1], 9)
    if changed_effects:
        report("proxy-changed targets only", changed_effects, 10)
    print(f"proxy-unchanged complete pairs: {len(unchanged_effects)} "
          "(all have zero contrast by exact-prompt reuse)")
    prepared = prepare(str(args.panel),
                       n_primary=int(meta.get("design_primary_pairs", total_pairs)),
                       ordering=meta["ordering"])
    pairs = selected_pairs(prepared, meta["run_set"])
    if len(pairs) != total_pairs:
        raise SystemExit("Reconstructed pair count differs from the call record")
    for pair_id, pair in enumerate(pairs):
        row = grouped[pair_id]["cf_a1", 0]
        if int(row["i"]) != pair["i"] or int(row["j"]) != pair["j"]:
            raise SystemExit(f"Reconstructed patient pair differs at {pair_id}")
    ideal = ideal_pair_effects(prepared, pairs)[complete_pair_ids]
    report("ideal learner given the same examples", ideal.tolist(), 11)
    excess = np.asarray(pair_effects) - ideal
    report("model minus ideal learner, paired", excess.tolist(), 12)
    print("true-risk Bayes reference: +0.00 pp by the checked zero-proxy rule")
    if missing:
        print("Missing responses were excluded, not replaced; inspect their pattern "
              "before interpreting the estimate.")


if __name__ == "__main__":
    main()
