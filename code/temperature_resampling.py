#!/usr/bin/env python3
"""Temperature-1 resampling check of the proxy-specific effect estimator.

Reviewer request (Scientific Reports revision, R2.3). The frozen protocol
elicits one deterministic choice per prompt at temperature 0, and estimates
the PSE as a mean of binary choices over the evaluation-pair population. The
reviewer proposes reading a per-prompt choice distribution instead. This
script estimates that distribution directly, by sampling each prompt
--samples times (default 20) at temperature 1, and then compares the
resulting cell-level PSE, computed from empirical per-pair choice
probabilities, with the deterministic estimate from the frozen records.

Run (same environment variables as run_cell.py):

    python temperature_resampling.py --k 12 --corner LH --semantic neutral \
        --rule learned                          # zero-information cell
    python temperature_resampling.py --k 12 --corner LH --semantic neutral \
        --rule learned --beta 0.60              # high-evidence dose cell

Calls per cell: 99 pairs x 2 orders x 2 arms x --samples (default 20) = 7920.

Writes  results/<model>/<cell>_t1resample_records.csv  with schema
    arm,sample_id,pair_id,i,j,order,patient,chosen,picked_i

Analysis (offline, no API calls):

    python temperature_resampling.py --analyze \
        --records results/<model>/<cell>_t1resample_records.csv \
        --frozen  <path to the frozen <cell>_records.csv>
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace as dc_replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k                      # noqa: E402
import config                     # noqa: E402
import prompts                    # noqa: E402
from client import Client, ModelConfig      # noqa: E402
from run_cell import build_cell, pair_sample   # noqa: E402

FIELDS = ["arm", "sample_id", "pair_id", "i", "j", "order", "patient",
          "chosen", "picked_i"]


def run(args) -> None:
    informative = args.informative or args.beta is not None
    ref, pairs, Xcal, Ycal, _ = build_cell(
        args.k, args.corner, args.semantic, args.rule, informative, args.beta)

    tag = (f"k{args.k}_{args.corner}_{args.semantic}_{args.rule}_"
           f"{'info' if informative else 'zero'}")
    if args.beta is not None:
        tag += f"_b{int(round(args.beta * 100)):03d}"

    cfg = dc_replace(ModelConfig.from_env(), temperature=args.temperature)
    cli = Client(cfg)
    model_dir = os.path.join(args.outdir, cfg.model)
    os.makedirs(model_dir, exist_ok=True)
    out = os.path.join(model_dir, f"{tag}_t1resample_records.csv")

    done = set()
    if args.resume and os.path.exists(out):
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["arm"], int(row["sample_id"]),
                          int(row["pair_id"]), int(row["order"])))

    jobs = []
    for pid, p in enumerate(pairs):
        i, j = int(p["i"]), int(p["j"])
        for arm in ("cf_a1", "cf_a0"):
            sample = pair_sample(ref, arm, p)
            for order in range(config.N_ORDERS):
                a, b = (i, j) if order == 0 else (j, i)
                pr = prompts.build_prompt(args.k, args.semantic, args.rule,
                                          sample.X[a], sample.X[b], Xcal, Ycal)
                for sid in range(args.samples):
                    if (arm, sid, pid, order) in done:
                        continue
                    jobs.append((arm, sid, pid, i, j, order, a, pr))

    print(f"{tag}: {len(jobs)} calls to run at temperature "
          f"{args.temperature} -> {out}")
    if not jobs:
        return

    new_file = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(FIELDS)

    n_written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cli.choose, pr): (arm, sid, pid, i, j, order, first)
                for arm, sid, pid, i, j, order, first, pr in jobs}
        for fut in as_completed(futs):
            arm, sid, pid, i, j, order, first = futs[fut]
            patient = fut.result()
            if patient is None:
                w.writerow([arm, sid, pid, i, j, order, "", "", ""])
            else:
                chosen = first if patient == 1 else (j if first == i else i)
                w.writerow([arm, sid, pid, i, j, order, patient, chosen,
                            int(chosen == i)])
            n_written += 1
            if n_written % 50 == 0:
                fh.flush()
                print(f"    {n_written}/{len(jobs)}", flush=True)
    fh.flush()
    fh.close()
    print(f"wrote {n_written} records")


# ---------------------------------------------------------------- analysis
def _pair_prob(rows):
    """{(arm, pid): empirical P(pick target)} over orders and samples."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        if r["picked_i"] == "":
            continue
        acc[(r["arm"], int(r["pair_id"]))].append(int(r["picked_i"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def analyze(args) -> None:
    with open(args.records) as f:
        trows = list(csv.DictReader(f))
    with open(args.frozen) as f:
        frows = [r for r in csv.DictReader(f)
                 if r["arm"] in ("cf_a1", "cf_a0")]

    def per_pair(rows):
        m = _pair_prob(rows)
        pids = sorted({k for a, k in m
                       if (("cf_a1", k) in m and ("cf_a0", k) in m)})
        return {p: m[("cf_a1", p)] - m[("cf_a0", p)] for p in pids}

    temp = per_pair(trows)
    froz = per_pair(frows)
    common = sorted(set(temp) & set(froz))
    t = np.array([temp[p] for p in common])
    z = np.array([froz[p] for p in common])

    rng = np.random.default_rng(0)
    B = 5000
    pse_t = np.empty(B)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(common), len(common))
        pse_t[b] = 100 * t[idx].mean()
        diffs[b] = 100 * (z[idx].mean() - t[idx].mean())
    tlo, thi = np.percentile(pse_t, [2.5, 97.5])
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])

    print(f"pairs used: {len(common)}")
    print(f"deterministic (temperature 0) PSE : {100*z.mean():+.2f} pp")
    print(f"temperature-1 probability PSE     : {100*t.mean():+.2f} pp "
          f"[{tlo:+.2f}, {thi:+.2f}]")
    print(f"difference deterministic - resampled: "
          f"{100*(z.mean()-t.mean()):+.2f} pp [{dlo:+.2f}, {dhi:+.2f}]")
    print(f"per-pair correlation (resampled vs deterministic): "
          f"{np.corrcoef(t, z)[0,1]:+.3f}")
    support = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    n_graded = int(np.sum(
        np.min(np.abs(t[:, None] - support[None, :]), axis=1) > 1e-9))
    print(f"pairs whose empirical per-pair value falls outside the "
          f"five-point deterministic support: {n_graded}/{len(common)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--records", help="t1resample records (analysis mode)")
    ap.add_argument("--frozen", help="frozen cell records (analysis mode)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--corner", default="LH")
    ap.add_argument("--semantic", default="neutral")
    ap.add_argument("--rule", default="learned")
    ap.add_argument("--informative", action="store_true")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
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
