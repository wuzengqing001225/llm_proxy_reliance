#!/usr/bin/env python3
"""Stochastic-regeneration sensitivity of the directed proxy-specific effect.

Reviewer request (Scientific Reports revision, R1.6). The frozen protocol
computes a unit-level counterfactual: flipping the target's protected
attribute regenerates the proxies through the structural equations with the
SAME exogenous noise (dgp_k.counterfactual). This script estimates the
interventional analogue. For each evaluation pair and each of --draws
independent redraws, the TARGET patient's proxy-channel noise (u_A and
proxy_noise) is replaced with a fresh draw, and both counterfactual arms are
run under that shared redraw (common random numbers across the two arms, so
each redraw is itself a well-formed contrast). Averaging over redraws gives
the noise-marginalized interventional contrast the reviewer asks about.
The comparator patient and all other patients keep their original noise, and
the factual arm is not needed.

Run (same environment variables as run_cell.py):

    python stochastic_regeneration.py --k 12 --corner LH --semantic neutral \
        --rule learned                          # zero-information cell
    python stochastic_regeneration.py --k 12 --corner LH --semantic neutral \
        --rule learned --beta 0.60              # high-evidence dose cell

Calls per cell: 99 pairs x 2 orders x 2 arms x --draws (default 5) = 1980.

Writes  results/<model>/<cell>_stochregen_records.csv  with schema
    arm,draw,pair_id,i,j,order,patient,chosen,picked_i

Analysis (offline, no API calls), comparing against the frozen records:

    python stochastic_regeneration.py --analyze \
        --records results/<model>/<cell>_stochregen_records.csv \
        --frozen  <path to the frozen <cell>_records.csv>
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k                      # noqa: E402
import config                     # noqa: E402
import prompts                    # noqa: E402
from client import Client, ModelConfig      # noqa: E402
from run_cell import build_cell             # noqa: E402

FIELDS = ["arm", "draw", "pair_id", "i", "j", "order", "patient", "chosen",
          "picked_i"]
NOISE_BASE_SEED = 260914   # revision date; independent of every frozen seed


def fresh_noise_sample(ref, target: int, pid: int, draw: int):
    """Copy of `ref` whose TARGET row carries freshly drawn proxy noise.

    The redraw is deterministic in (pid, draw) so interrupted runs resume
    reproducibly, and it is shared by the two counterfactual arms.
    """
    rng = np.random.default_rng(np.random.SeedSequence(
        [NOISE_BASE_SEED, int(pid), int(draw)]))
    u_A = ref.u_A.copy()
    pn = ref.proxy_noise.copy()
    u_A[target] = rng.standard_normal()
    pn[target, :] = rng.standard_normal(pn.shape[1])
    return replace(ref, u_A=u_A, proxy_noise=pn)


def run(args) -> None:
    informative = args.informative or args.beta is not None
    ref, pairs, Xcal, Ycal, _ = build_cell(
        args.k, args.corner, args.semantic, args.rule, informative, args.beta)

    tag = (f"k{args.k}_{args.corner}_{args.semantic}_{args.rule}_"
           f"{'info' if informative else 'zero'}")
    if args.beta is not None:
        tag += f"_b{int(round(args.beta * 100)):03d}"

    cfg = ModelConfig.from_env()
    cli = Client(cfg)
    model_dir = os.path.join(args.outdir, cfg.model)
    os.makedirs(model_dir, exist_ok=True)
    out = os.path.join(model_dir, f"{tag}_stochregen_records.csv")

    done = set()
    if args.resume and os.path.exists(out):
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["arm"], int(row["draw"]), int(row["pair_id"]),
                          int(row["order"])))

    jobs = []
    for pid, p in enumerate(pairs):
        i, j = int(p["i"]), int(p["j"])
        for draw in range(args.draws):
            base = fresh_noise_sample(ref, i, pid, draw)
            for arm, aval in (("cf_a1", 1), ("cf_a0", 0)):
                newA = ref.A.copy()
                newA[i] = aval
                sample = dgp_k.counterfactual(base, newA)
                for order in range(config.N_ORDERS):
                    if (arm, draw, pid, order) in done:
                        continue
                    a, b = (i, j) if order == 0 else (j, i)
                    pr = prompts.build_prompt(args.k, args.semantic, args.rule,
                                              sample.X[a], sample.X[b],
                                              Xcal, Ycal)
                    jobs.append((arm, draw, pid, i, j, order, a, pr))

    print(f"{tag}: {len(jobs)} calls to run -> {out}")
    if not jobs:
        return

    new_file = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(FIELDS)

    n_written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cli.choose, pr): (arm, draw, pid, i, j, order, first)
                for arm, draw, pid, i, j, order, first, pr in jobs}
        for fut in as_completed(futs):
            arm, draw, pid, i, j, order, first = futs[fut]
            patient = fut.result()
            if patient is None:
                w.writerow([arm, draw, pid, i, j, order, "", "", ""])
            else:
                chosen = first if patient == 1 else (j if first == i else i)
                w.writerow([arm, draw, pid, i, j, order, patient, chosen,
                            int(chosen == i)])
            n_written += 1
            if n_written % 20 == 0:
                fh.flush()
                print(f"    {n_written}/{len(jobs)}", flush=True)
    fh.flush()
    fh.close()
    print(f"wrote {n_written} records")


# ---------------------------------------------------------------- analysis
def _pair_means(rows, key_draw=False):
    """{(pid) or (pid, draw): mean picked_i} per arm from record dicts."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        if r["picked_i"] == "":
            continue
        k = (int(r["pair_id"]), int(r["draw"])) if key_draw else int(r["pair_id"])
        acc[(r["arm"], k)].append(int(r["picked_i"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def analyze(args) -> None:
    with open(args.records) as f:
        srows = list(csv.DictReader(f))
    with open(args.frozen) as f:
        frows = [dict(r, draw="0") for r in csv.DictReader(f)
                 if r["arm"] in ("cf_a1", "cf_a0")]

    def pse_by_pair(rows):
        m = _pair_means(rows)
        pids = sorted({k for a, k in m if (("cf_a1", k) in m and ("cf_a0", k) in m)})
        return {p: m[("cf_a1", p)] - m[("cf_a0", p)] for p in pids}

    stoch = pse_by_pair(srows)
    froz = pse_by_pair(frows)
    common = sorted(set(stoch) & set(froz))
    s = np.array([stoch[p] for p in common])
    z = np.array([froz[p] for p in common])

    md = _pair_means(srows, key_draw=True)
    draws = sorted({d for a, (p, d) in md})
    per_draw = []
    for d in draws:
        vals = [md[("cf_a1", (p, d))] - md[("cf_a0", (p, d))]
                for p in common
                if ("cf_a1", (p, d)) in md and ("cf_a0", (p, d)) in md]
        per_draw.append(100 * float(np.mean(vals)))

    rng = np.random.default_rng(0)
    B = 5000
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(common), len(common))
        diffs[b] = 100 * (z[idx].mean() - s[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    print(f"pairs used: {len(common)}")
    print(f"frozen (counterfactual) PSE : {100*z.mean():+.2f} pp")
    print(f"stochastic-regeneration PSE : {100*s.mean():+.2f} pp "
          f"(per-draw: {', '.join(f'{v:+.1f}' for v in per_draw)})")
    print(f"difference frozen - stochastic: {100*(z.mean()-s.mean()):+.2f} pp "
          f"[{lo:+.2f}, {hi:+.2f}]  (pair bootstrap, B={B})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--records", help="stochregen records (analysis mode)")
    ap.add_argument("--frozen", help="frozen cell records (analysis mode)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--corner", default="LH")
    ap.add_argument("--semantic", default="neutral")
    ap.add_argument("--rule", default="learned")
    ap.add_argument("--informative", action="store_true")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.analyze:
        if not (args.records and args.frozen):
            ap.error("--analyze requires --records and --frozen")
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
