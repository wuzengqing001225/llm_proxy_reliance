#!/usr/bin/env python3
"""Preflight: verify the package reproduces the frozen design before spending API budget.

    python verify.py            # offline checks only
    python verify.py --api      # additionally make 2 live calls

Checks:
  1. DGP reference values match the values recorded in the original run
  2. Target pairs are stratum-balanced and identical across semantic/arm
  3. Counterfactual intervention moves proxy fields but not legit fields
  4. Prompts contain no protected attribute and no field-name leakage
  5. Ideal-learner reference values reproduce
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k
import oracle
sys.modules["dgp"] = dgp_k

import config
import prompts
from run_cell import build_cell, pair_sample

# Values recorded in the original single-model run (frozen; do not edit).
EXPECTED_R_IDEAL = {
    (6,  "LH", False): 0.0101,
    (12, "LH", False): 0.0202,
    (18, "LH", False): 0.0202,
    (12, "HH", False): None,     # not run originally; no reference to check
}

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true", help="also make 2 live API calls")
    args = ap.parse_args()

    print("1. DGP reference values")
    for (k, corner, info), exp in EXPECTED_R_IDEAL.items():
        ref, pairs, Xcal, Ycal, r_ideal = build_cell(k, corner, "neutral", "learned", info)
        if exp is None:
            print(f"  [ -- ] k={k} {corner}: R_Ideal={r_ideal:.4f} (no stored reference)")
            continue
        check(f"k={k} {corner} R_Ideal", abs(r_ideal - exp) < 5e-3,
              f"got {r_ideal:.4f}, expected {exp:.4f}")

    print("\n2. Pair construction")
    ref, pairs, Xcal, Ycal, _ = build_cell(12, "LH", "neutral", "learned", False)
    strata = {}
    for p in pairs:
        strata[p["stratum"]] = strata.get(p["stratum"], 0) + 1
    check("99 pairs", len(pairs) == 99, f"got {len(pairs)}")
    check("balanced strata", set(strata.values()) == {33}, f"got {strata}")

    print("\n3. Counterfactual: target-only intervention on proxies")
    n_legit, n_proxy = dgp_k.K_LADDER[12]
    p0 = pairs[0]
    ti, tj = int(p0["i"]), int(p0["j"])
    cf1, cf0 = pair_sample(ref, "cf_a1", p0), pair_sample(ref, "cf_a0", p0)
    check("legit fields invariant under A",
          np.allclose(cf1.Xleg, cf0.Xleg))
    check("target proxies move with A",
          not np.allclose(cf1.P[ti], cf0.P[ti]))
    check("comparator proxies untouched",
          np.allclose(cf1.P[tj], cf0.P[tj]))
    n_moved = int((~np.isclose(cf1.P, cf0.P).all(axis=1)).sum())
    check("exactly one row intervened", n_moved == 1, f"rows moved: {n_moved}")

    print("\n4. Prompt hygiene")
    p_learn = prompts.build_prompt(12, "social", "learned",
                                   ref.X[pairs[0]["i"]], ref.X[pairs[0]["j"]], Xcal, Ycal)
    p_abs = prompts.build_prompt(12, "neutral", "absent",
                                 ref.X[pairs[0]["i"]], ref.X[pairs[0]["j"]])
    banned = ["protected", "race", "gender", "ethnic", "A=", "attribute A"]
    check("no protected-attribute leakage",
          not any(b.lower() in p_learn.lower() for b in banned))
    n_cases = sum(1 for ln in p_learn.splitlines() if " -> " in ln and not ln.startswith("TRAINING"))
    check(f"rule-learned prompt has {config.N_CAL} cases", n_cases == config.N_CAL,
          f"got {n_cases}")
    check("rule-absent prompt has no cases", " -> " not in p_abs)
    check("no scoring rule disclosed",
          "weight" not in p_learn.lower() and "coefficient" not in p_learn.lower())
    print(f"  prompt sizes: rule-learned {len(p_learn):,} chars, rule-absent {len(p_abs):,} chars")

    print("\n5. Semantic labels differ only in proxy names")
    a = prompts.field_names(12, "social")
    b = prompts.field_names(12, "neutral")
    check("legit names identical", a[:n_legit] == b[:n_legit])
    check("proxy names differ", a[n_legit:] != b[n_legit:])

    if args.api:
        print("\n6. Live API")
        from client import Client, ModelConfig
        cli = Client(ModelConfig.from_env())
        got = [cli.choose(p_abs), cli.choose(p_learn)]
        check("both calls parsed", all(g in (1, 2) for g in got), f"got {got}")
        check("tool calling", not cli._tools_rejected,
              "using tool calls" if not cli._tools_rejected else "fell back to text parsing")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
