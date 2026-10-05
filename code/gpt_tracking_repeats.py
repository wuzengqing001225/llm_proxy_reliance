#!/usr/bin/env python3
"""Recompute the GPT-5.6 Terra repeated-measurement dose response.

Two independent neutral-label runs per evidence level, analyzed separately
from the battery records. GPT-5.6 Terra uses its default sampling (the
metadata channel string records the client request settings). Makes no
model calls and writes no files.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "modules"))
import config
import dgp_k
import dgp_norm
import oracle
from analyze import per_pair_pse
from dose_response import fit_curve
from run_cell import build_cell

BETAS = (0.0, 0.30, 0.45, 0.60)
MODEL = "gpt-5.6-terra"
TAG = "k12_LH_neutral_learned_"
DEFAULT_DATA = (Path(__file__).resolve().parent.parent / "data" / "raw_records"
                / MODEL / "repeated_measurement")


def record_path(data: Path, beta: float, run: str) -> Path:
    suffix = "zero" if beta == 0 else f"info_b{round(beta * 100):03d}"
    return data / f"run_{run}" / f"{TAG}{suffix}_records.csv"


def load_pse(path: Path, expected_channel: str) -> dict:
    df = pd.read_csv(path)
    key = ["arm", "pair_id", "order"]
    if len(df) != 594 or df.duplicated(key).any() or df["picked_i"].isna().any():
        raise ValueError(f"Incomplete, duplicated or unparseable calls: {path}")
    if set(df["arm"]) != {"factual", "cf_a1", "cf_a0"}:
        raise ValueError(f"Unexpected arms: {path}")
    meta_path = path.with_name(path.name.replace("_records.csv", "_metadata.json"))
    meta = json.loads(meta_path.read_text())
    if (meta.get("channel") != expected_channel or meta.get("extra_json") != ""
            or meta.get("served") != [{"model": MODEL,
                                       "system_fingerprint": "", "calls": 594}]):
        raise ValueError(f"Unexpected serving metadata: {meta_path}")
    pse = per_pair_pse(df[df["arm"].isin(("cf_a1", "cf_a0"))])
    if len(pse) != 99:
        raise ValueError(f"Expected 99 complete pairs: {path}")
    return pse


def ideal_reference(beta: float) -> tuple[float, float]:
    ref, pairs, _, _, _ = build_cell(12, "LH", "neutral", "learned",
                                     beta > 0, None if beta == 0 else beta)
    gc, ga = config.CORNERS["LH"]
    if beta == 0:
        dcal = dgp_k.generate(gc, ga, k=12, n=config.N_CAL,
                              seed=config.DCAL_SEED)
        post = oracle.blr_fit(dcal.X, dcal.Y, sigma=dgp_k.SIGMA_Y, tau=1.0)
    else:
        post = dgp_norm.fit_ideal(gc, ga, k=12, beta_info=beta,
                                  n_cal=config.N_CAL,
                                  dcal_seed=config.DCAL_SEED)
    return dgp_norm.ideal_line(post, ref, pairs), dgp_norm.bayes_line(ref, pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    levels = {}
    observed = {}
    print("beta  ideal(pp)  run_a(pp)  run_b(pp)  mean(pp)")
    for beta in BETAS:
        first = load_pse(record_path(args.data, beta, "a"),
                         "tools=yes reasoning_budget=no temperature=0.0")
        second = load_pse(record_path(args.data, beta, "b"),
                          "tools=yes reasoning_budget=no temperature=0.0")
        if set(first) != set(second):
            raise ValueError(f"Evaluation pairs differ at beta={beta}")
        mean = {pair: (first[pair] + second[pair]) / 2 for pair in first}
        ideal, bayes = ideal_reference(beta)
        levels[beta] = {"pse": mean, "ideal": ideal, "bayes": bayes}
        observed[beta] = {"a": first, "b": second}
        print(f"{beta:4.2f} {ideal:10.2f} "
              f"{100 * np.mean(list(first.values())):10.2f} "
              f"{100 * np.mean(list(second.values())):10.2f} "
              f"{100 * np.mean(list(mean.values())):9.2f}")

    fit = fit_curve(levels, B=5000, seed=7)
    print(f"gamma = {fit['gamma']:+.3f} "
          f"[{fit['gamma_ci'][0]:+.3f}, {fit['gamma_ci'][1]:+.3f}]")
    print(f"alpha = {fit['alpha']:+.2f} pp "
          f"[{fit['alpha_ci'][0]:+.2f}, {fit['alpha_ci'][1]:+.2f}]")
    x = np.array([levels[beta]["ideal"] for beta in BETAS])
    design = np.column_stack((np.ones(len(x)), x))
    slopes = []
    for choices in itertools.product(("a", "b"), repeat=len(BETAS)):
        y = np.array([100 * np.mean(list(observed[beta][choice].values()))
                      for beta, choice in zip(BETAS, choices)])
        slopes.append(np.linalg.lstsq(design, y, rcond=None)[0][1])
    print(f"one-run-per-level slope range (16 combinations): "
          f"{min(slopes):+.3f} to {max(slopes):+.3f}")


if __name__ == "__main__":
    main()
