#!/usr/bin/env python3
"""Assemble canonical per-cell Sonnet records from the raw shards.

1. gate2/ stores each k-sweep cell in shards (expK_g2_*, expK_g2fix_*,
   expK_g2fix2_*). Shards are merged on (arm, i, j, order) with precedence
   g2fix2 > g2fix > g2.
2. beta_sweep/ stores one file per arm. Two files code the arm column and
   presentation order differently, so the arm is taken from the file name
   and order is normalized to 0/1. One gate2 shard stores pair_id as a
   string, so the merge key uses the patient indices (i, j).
3. The script fails on duplicate keys and reports completeness for every
   assembled cell.

Usage
-----
    python prepare_canonical.py --raw ../data/raw_records/claude-sonnet-4-5 \
                                --out ../data/canonical/claude-sonnet-4-5

Outputs cell files named like k12_LH_neutral_learned_zero_records.csv, the
same format as the replication models, ready for analyze.py / dose_response.py.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys

import pandas as pd

FIELDS = ["arm", "pair_id", "i", "j", "order", "patient", "chosen", "picked_i"]
GEN_RANK = {"g2": 0, "g2fix": 1, "g2fix2": 2}


def generation(path: str) -> int:
    m = re.search(r"expK_(g2(?:fix2?)?)_", os.path.basename(path))
    if not m:
        raise SystemExit(f"cannot read generation from {path}")
    return GEN_RANK[m.group(1)]


def read_normalized(path: str) -> pd.DataFrame:
    """Read one raw file, normalizing the two known schema variants."""
    df = pd.read_csv(path)
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    # arm from filename when the column carries a run tag (beta_sweep variants)
    m = re.search(r"_(factual|cf_a[01])_records\.csv$", path)
    if m:
        df["arm"] = m.group(1)
    bad_arms = set(df["arm"].unique()) - {"factual", "cf_a1", "cf_a0"}
    if bad_arms:
        raise SystemExit(f"{path}: unrecognized arm values {bad_arms} and the "
                         f"filename does not name the arm")
    # order encodings in the raw files: {0,1}, {1,2}, {'ij','ji'}
    df["order"] = df["order"].map({"ij": 0, "ji": 1}).fillna(
        pd.to_numeric(df["order"], errors="coerce"))
    if df["order"].max() > 1:
        df["order"] = df["order"] - 1
    df["order"] = df["order"].astype(int)
    return df[FIELDS]


def merge_with_precedence(paths: list[str]) -> pd.DataFrame:
    """Higher-precedence shards overwrite lower ones on (arm, i, j, order)."""
    rows: dict = {}
    for p in sorted(paths, key=generation):
        for r in read_normalized(p).itertuples(index=False):
            rows[(r.arm, int(r.i), int(r.j), r.order)] = r
    return pd.DataFrame(rows.values(), columns=FIELDS)


def check_and_report(df: pd.DataFrame, tag: str) -> None:
    dup = df.duplicated(subset=["arm", "i", "j", "order"]).sum()
    if dup:
        raise SystemExit(f"{tag}: {dup} duplicate (arm,pair,order) keys")
    arms = df["arm"].value_counts().to_dict()
    n_missing = int(df["picked_i"].isna().sum())
    complete = all(arms.get(a, 0) == 198 for a in ("factual", "cf_a1", "cf_a0"))
    status = "complete" if complete else f"INCOMPLETE {arms}"
    print(f"  {tag}: {len(df)} rows, {status}, unparseable {n_missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="raw_records/claude-sonnet-4-5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- gate2: k-sweep, rule-learned cells ----
    print("gate2 (canonical merge g2fix2 > g2fix > g2):")
    for k in (6, 12, 18):
        for info in ("zero", "info"):
            for sem in ("neutral", "social"):
                paths = glob.glob(os.path.join(
                    args.raw, "gate2", f"expK_g2*_k{k}_{info}_{sem}*records.csv"))
                if not paths:
                    continue
                df = merge_with_precedence(paths)
                tag = f"k{k}_LH_{sem}_learned_{info}"
                check_and_report(df, tag)
                df.to_csv(os.path.join(args.out, tag + "_records.csv"), index=False)

    # ---- beta_sweep: dose-response cells (per-arm files, k12 LH learned) ----
    print("beta_sweep (arm from filename, order normalized):")
    for b in ("030", "045", "060"):
        for sem in ("neutral", "social"):
            paths = glob.glob(os.path.join(
                args.raw, "beta_sweep", f"expK_beta_b{b}_{sem}_*_records.csv"))
            if not paths:
                continue
            df = pd.concat([read_normalized(p) for p in paths])
            tag = f"k12_LH_{sem}_learned_info_b{b}"
            check_and_report(df, tag)
            df.to_csv(os.path.join(args.out, tag + "_records.csv"), index=False)

    print("\nThe beta=0 level of the dose response is the k12 zero cell "
          "assembled above.")


if __name__ == "__main__":
    main()
