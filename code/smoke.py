#!/usr/bin/env python3
"""Six live calls against a new model, before committing to a full tier.

    export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
    python smoke.py

Answers the four questions that decide whether a model is usable at all:
  1. does it respond, and how fast?
  2. does it accept tool-calling, or fall back to text parsing?
  3. is it a reasoning model (needs max_completion_tokens + a real budget)?
  4. does it answer both a short prompt and the 7k-char rule-learned prompt?

Costs six calls. Run it before every new model. A model that fails here will
produce a cell of 594 missing observations if you skip straight to run_all.sh.
"""
from __future__ import annotations
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k                                   # noqa: E402
sys.modules["dgp"] = dgp_k

import config                                  # noqa: E402
from client import Client, ModelConfig         # noqa: E402
from run_cell import build_cell                # noqa: E402

ok = True


def check(label: str, passed: bool, note: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {note}" if note else ""))


def warn(label: str, fine: bool, note: str = "") -> None:
    """Advisory only -- never fails the smoke test."""
    print(f"  [{'ok  ' if fine else 'WARN'}] {label}" + (f"  {note}" if note else ""))


def main() -> None:
    cfg = ModelConfig.from_env()
    print(f"model: {cfg.model}\nendpoint: {cfg.base_url}\n")
    cli = Client(cfg)

    # build_cell only constructs the calibration set when rule == "learned",
    # so build the learned cell for the long prompt (same population + pairs;
    # only Xcal/Ycal differ).
    ref, pairs, _, _, _ = build_cell(12, "LH", "neutral", "absent", False)
    _, _, Xcal, Ycal, _ = build_cell(12, "LH", "neutral", "learned", False)
    import prompts
    p_short = prompts.build_prompt(12, "neutral", "absent",
                                   ref.X[pairs[0]["i"]], ref.X[pairs[0]["j"]], None, None)
    p_long = prompts.build_prompt(12, "neutral", "learned",
                                  ref.X[pairs[0]["i"]], ref.X[pairs[0]["j"]], Xcal, Ycal)

    print(f"prompt sizes: rule-absent {len(p_short):,} chars, "
          f"rule-learned {len(p_long):,} chars\n")

    print("1. Short prompt (rule-absent)")
    t0 = time.time()
    got = [cli.choose(p_short) for _ in range(3)]
    dt = (time.time() - t0) / 3
    check("all three parsed", all(g in (1, 2) for g in got), f"got {got}")
    warn("latency", dt < 30, f"{dt:.1f}s per call")

    print("\n2. Long prompt (rule-learned, 80 calibration cases)")
    t0 = time.time()
    got_l = [cli.choose(p_long) for _ in range(3)]
    dtl = (time.time() - t0) / 3
    check("all three parsed", all(g in (1, 2) for g in got_l), f"got {got_l}")
    warn("latency", dtl < 60, f"{dtl:.1f}s per call")
    if dt > 15 or dtl > 45:
        print("       note: this is slow for a standard chat model — a default")
        print("       'thinking' mode may be on. Try:")
        print("         export LLM_EXTRA_JSON='{\"enable_thinking\": false}'")
        print("       and rerun smoke.py; if latency drops sharply, keep it set")
        print("       for the whole run (also removes a hidden-reasoning channel).")

    print("\n3. Call channel actually used")
    print(f"       {cli.channel()}")
    if cli._tools_rejected:
        print("       note: tool-calling unavailable; answers parsed from text.")
        print("       This is a CHANNEL DIFFERENCE vs models measured with tools.")
        print("       Record it, and watch missing_rate — text parsing fails more")
        print("       often than forced tool output.")
    if cli._reasoning:
        print("       note: reasoning model detected; using max_completion_tokens "
              f"= {cfg.reasoning_budget}. Raise LLM_REASONING_BUDGET if you see "
              "missing observations.")

    cell_s = config.cell_calls() * dt / 3600
    cell_l = config.cell_calls() * dtl / 3600
    print(f"\n4. Projected wall-clock at --workers 8")
    print(f"       rule-absent cell:  {cell_s/8:.1f} h")
    print(f"       rule-learned cell: {cell_l/8:.1f} h")
    print(f"       core tier (6 cells, 3,564 calls): "
          f"~{(4*cell_s + 2*cell_l)/8:.1f} h")
    print(f"       full tier (18 cells, 10,692 calls): "
          f"~{(8*cell_s + 10*cell_l)/8:.1f} h")

    print("\n" + ("SMOKE TEST PASSED — safe to run a tier"
                  if ok else
                  "SMOKE TEST FAILED — fix before spending a tier's budget"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
