#!/usr/bin/env python3
"""Recompute all published encoding and real-record results without API calls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from categorical_proxy_control import per_pair, boot
from external_zero_info import prepare, unique_counterfactual_prompts, SEED
from run_external_zero_info import selected_pairs

MODEL = "claude-sonnet-4-5-20250929"
ROOT = Path(__file__).resolve().parents[1]


def report(series):
    e, lo, hi = boot(series.to_numpy())
    return dict(n=len(series), effect_pp=float(e), lo_pp=float(lo), hi_pp=float(hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--panel", type=Path, default=Path(__file__).resolve().parent / "analysis/real_anchors/uci_diabetes_130hosp.csv")
    ap.add_argument("--ids-mapping", type=Path, default=Path(__file__).resolve().parent / "analysis/real_anchors/IDS_mapping.csv")
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "data/new_evidence_manifest.json").read_text())
    for name, sha in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise SystemExit(f"Evidence checksum failed: {name}")
    result = {"encoding": {}, "external": {}, "paired_sensitivity": {}}
    enc = root / "data/encoding_control" / MODEL
    for rule in ("learned", "absent"):
        series = {}
        for render in ("ordinal", "nominal"):
            path = enc / f"catctl_k12_LH_neutral_{rule}_zero_{render}_records.csv"
            df = pd.read_csv(path)
            if len(df) != 396 or df.duplicated(["pair_id", "arm", "order"]).any():
                raise SystemExit(f"Incomplete or duplicate encoding records: {path}")
            series[render] = per_pair(df)
        if rule == "learned":
            series["numeric"] = per_pair(pd.read_csv(enc / "catctl_k12_LH_neutral_learned_zero_numeric_remeasure_records.csv"))
        series["numeric_reference"] = per_pair(pd.read_csv(root / f"data/encoding_control/reference/sonnet_k12_LH_neutral_{rule}_zero_records.csv"))
        result["encoding"][rule] = {key: report(value) for key, value in series.items()}
        if rule == "learned":
            # call-level agreement between the two numeric measurements
            keys = ["i", "j", "arm", "order"]
            def calls(path):
                df = pd.read_csv(path)
                df = df[df.arm.isin(["cf_a1", "cf_a0"]) & df.picked_i.notna()]
                return df.assign(order=df.order.astype(int)).set_index(keys).picked_i.astype(int)
            second = calls(enc / "catctl_k12_LH_neutral_learned_zero_numeric_remeasure_records.csv")
            battery = calls(root / "data/encoding_control/reference/sonnet_k12_LH_neutral_learned_zero_records.csv")
            common = second.index.intersection(battery.index)
            result["agreement"] = {"numeric_second_vs_battery": dict(
                agree=int((second.loc[common] == battery.loc[common]).sum()), n=len(common))}
        for a, b in (("ordinal", "nominal"), ("numeric", "ordinal"), ("numeric", "nominal"), ("numeric_reference", "ordinal"), ("numeric_reference", "nominal"), ("numeric", "numeric_reference")):
            if a not in series:
                continue
            common = series[a].index.intersection(series[b].index)
            result["encoding"][rule][f"{a}_minus_{b}"] = report(series[a].loc[common] - series[b].loc[common])
    ext = root / "data/external_zero_info" / MODEL
    specs = [("primary", "fixed", 1200), ("random_target", "fixed", 300),
             ("primary", "reverse", 300), ("primary", "seeded", 300), ("gap_stratified", "fixed", 1200)]
    all_series = {}
    for run_set, ordering, n in specs:
        stem = f"external_zero_{run_set}_{ordering}_p{n}"
        records, meta = ext / f"{stem}_records.csv", ext / f"{stem}_metadata.json"
        metadata = json.loads(meta.read_text())
        prepared = prepare(str(args.panel.resolve()), n_primary=n, ordering=ordering)
        pairs = selected_pairs(prepared, run_set)
        jobs, links = unique_counterfactual_prompts(prepared, pairs)
        design_hash = hashlib.sha256(json.dumps(dict(jobs=sorted(jobs), run_set=run_set,
                       ordering=ordering, pairs=n, seed=SEED), sort_keys=True).encode()).hexdigest()
        if design_hash != metadata["design_hash"]:
            raise SystemExit(f"Rebuilt prompts differ from acquisition design: {stem}")
        calls = pd.read_csv(ext / f"{stem}_calls.csv", keep_default_na=False)
        if calls.job_id.duplicated().any() or set(calls.job_id) != set(jobs):
            raise SystemExit(f"Calls do not match the frozen job set: {stem}")
        choices = dict(zip(calls.job_id, calls.patient.astype(str)))
        for row in pd.read_csv(records, keep_default_na=False).to_dict("records"):
            group = {"cf_a1": 1, "cf_a0": 0}[row["arm"]]
            if row["job_id"] != links[(int(row["pair_id"]), group, int(row["order"]))] or str(row["patient"]) != choices[row["job_id"]]:
                raise SystemExit(f"Logical record differs from recorded call: {stem}")
        proc = subprocess.run([sys.executable, str(Path(__file__).parent / "analyze_external_zero_info.py"),
                               "--records", str(records), "--metadata", str(meta), "--panel", str(args.panel.resolve()),
                               "--ids-mapping", str(args.ids_mapping.resolve())], capture_output=True, text=True, check=True)
        parsed = {}
        for line in proc.stdout.splitlines():
            match = re.match(r"(.+): n=(\d+), effect=([+-][\d.]+) pp, pair-bootstrap 95% \[([+-][\d.]+), ([+-][\d.]+)\] pp", line)
            if match:
                key, count, effect, lo, hi = match.groups()
                parsed[key] = dict(n=int(count), effect_pp=float(effect), lo_pp=float(lo), hi_pp=float(hi))
        result["external"][stem] = parsed
        all_series[stem] = per_pair(pd.read_csv(records)).sort_index()
        print(stem + ": " + proc.stdout.splitlines()[1])
    # Pair IDs, rather than sorted patient IDs, align the same unordered pairs
    # across target orientations and category couplings.
    def by_id(path):
        df = pd.read_csv(path)
        wide = df.pivot_table(index="pair_id", columns="arm", values="picked_i")
        return wide.cf_a1 - wide.cf_a0
    baseline = by_id(ext / "external_zero_primary_fixed_p1200_records.csv").iloc[:300]
    result["external"]["primary_fixed_first300"] = {"all eligible pairs": report(baseline)}
    for stem in ("external_zero_random_target_fixed_p300", "external_zero_primary_reverse_p300", "external_zero_primary_seeded_p300"):
        result["paired_sensitivity"][stem] = report(by_id(ext / f"{stem}_records.csv") - baseline)
    out = root / "data/summaries/new_evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "condition", "contrast", "n", "effect_pp", "lo_pp", "hi_pp"])
        writer.writeheader()
        for experiment in ("encoding", "external"):
            for condition, rows in result[experiment].items():
                for contrast, values in rows.items():
                    writer.writerow(dict(experiment=experiment, condition=condition, contrast=contrast, **values))
    print("Encoding learned:", json.dumps(result["encoding"]["learned"]))
    print("Agreement:", json.dumps(result["agreement"]))
    print(f"Verified {len(manifest)} source files. Saved {out}")


if __name__ == "__main__":
    main()
