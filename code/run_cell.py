#!/usr/bin/env python3
"""Run one experimental cell (all three arms) against any OpenAI-compatible model.

    python run_cell.py --k 12 --corner LH --semantic neutral --rule learned

Writes  results/<model>/<cell>_records.csv  with the frozen schema:
    arm,pair_id,i,j,order,patient,chosen,picked_i

Each arm is 198 calls (99 pairs x 2 presentation orders); three arms per cell.
Records are flushed to disk incrementally, so an interrupted run loses at most
one call -- resume with --resume.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))
import dgp_k
import dgp_norm                      # noqa: E402
import oracle                     # noqa: E402
import estimators                 # noqa: E402
sys.modules["dgp"] = dgp_k        # estimators imports the DGP under this name

import config                     # noqa: E402
import prompts                    # noqa: E402
from client import Client, ModelConfig   # noqa: E402
from codex_client import CodexClient, CodexConfig   # noqa: E402

FIELDS = ["arm", "pair_id", "i", "j", "order", "patient", "chosen", "picked_i"]

# k, corner, semantic, rule, info -- shared with analyze.py and calibrate_wrapper.py
CELL_TAG_RE = re.compile(
    r"k(\d+)_(LL|LH|HL|HH)_(social|neutral)_(absent|learned)_(zero|info)"
    r"(?:_b(\d{3}))?(?:_s(\d+))?_records\.csv$")


def derived_seeds(seed):
    """One replication seed derives all three randomization seeds.

    None -> the frozen configuration (config.TARGET_SEED / PAIR_SEED /
    DCAL_SEED). Any other value S -> (S, S+20000, S+40000), so a seed
    replication redraws the population, the evaluation pairs and the
    calibration sample together.
    """
    if seed is None:
        return config.TARGET_SEED, config.PAIR_SEED, config.DCAL_SEED
    return seed, seed + 20000, seed + 40000


def build_cell(k: int, corner: str, semantic: str, rule: str, informative: bool,
               beta: float | None = None, seed: int | None = None):
    """Everything that does not depend on the model: population, pairs, calibration."""
    gc, ga = config.CORNERS[corner]
    t_seed, p_seed, d_seed = derived_seeds(seed)
    b = config.BETA_INFO if beta is None else beta
    if beta is None:
        ref = dgp_k.generate(gc, ga, k=k, n=5000, seed=t_seed,
                             proxy_informative=informative, beta_info=b)
    else:
        # Dose-response levels use the variance-normalized generator so ranking
        # difficulty stays fixed while the proxy's share of the risk signal moves.
        ref = dgp_norm.generate_normalized(gc, ga, k=k, n=5000,
                                           seed=t_seed, beta_info=b)
    if beta is None:
        pairs = dgp_k.make_pairs(ref, n_per_stratum=config.N_PER_STRATUM,
                                 seed=p_seed)
    else:
        # make_pairs strata on sample.r, which the normalized generator rescales.
        # Building pairs from `ref` would shift the evaluation set with beta
        # (measured overlap falls to 76/99) and break the paired comparison.
        # Always take the reference (beta=0) pair set and reuse it at every level.
        pairs = dgp_norm.reference_pairs(gc, ga, k=k, n=5000,
                                         seed=t_seed,
                                         n_per_stratum=config.N_PER_STRATUM,
                                         pair_seed=p_seed)

    Xcal = Ycal = None
    r_ideal = None
    if rule == "learned":
        if beta is None:
            dcal = dgp_k.generate(gc, ga, k=k, n=config.N_CAL, seed=d_seed,
                                  proxy_informative=informative, beta_info=b)
        else:
            dcal = dgp_norm.generate_normalized(gc, ga, k=k, n=config.N_CAL,
                                                seed=d_seed, beta_info=b)
        Xcal, Ycal = dcal.X, dcal.Y
        post = oracle.blr_fit(Xcal, Ycal, sigma=dgp_k.SIGMA_Y, tau=1.0)
        r_ideal = oracle.ideal_pairwise_error_rate(ref, pairs, post)
    return ref, pairs, Xcal, Ycal, r_ideal


def pair_sample(ref, arm: str, pair):
    """The sample a given pair is rendered from.

    Matches the frozen estimator exactly: the directed counterfactual forces A
    on the TARGET index i only, leaving the comparator j untouched. Flipping A
    for the whole population instead would move the comparator's proxies too
    and change what the proxy effect measures.
    """
    if arm == "factual":
        return ref
    newA = ref.A.copy()
    newA[int(pair["i"])] = 1 if arm == "cf_a1" else 0
    return dgp_k.counterfactual(ref, newA)


def run_arm(cli: Client, ref, pairs, k, semantic, rule, Xcal, Ycal, arm,
            out_path: str, workers: int, done: set) -> int:
    jobs = []
    # stratum-descending so any truncation costs the easiest items, not the hardest
    for pid, p in sorted(enumerate(pairs), key=lambda t: -t[1]["stratum"]):
        i, j = int(p["i"]), int(p["j"])
        sample = pair_sample(ref, arm, p)   # per-pair: A forced on target i only
        for order in range(config.N_ORDERS):
            if (arm, pid, order) in done:
                continue
            a, b = (i, j) if order == 0 else (j, i)
            pr = prompts.build_prompt(k, semantic, rule,
                                      sample.X[a], sample.X[b], Xcal, Ycal)
            jobs.append((pid, i, j, order, a, pr))

    if not jobs:
        return 0

    new_file = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(FIELDS)

    n_written = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cli.choose, pr): (pid, i, j, order, a)
                for pid, i, j, order, a, pr in jobs}
        for fut in futs:
            pass  # submitted; consumed below in completion order
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            pid, i, j, order, first = futs[fut]
            patient = fut.result()
            if patient is None:
                # unparseable -> recorded as missing, never substituted
                w.writerow([arm, pid, i, j, order, "", "", ""])
            else:
                chosen = first if patient == 1 else (j if first == i else i)
                w.writerow([arm, pid, i, j, order, patient, chosen, int(chosen == i)])
            n_written += 1
            if n_written % 20 == 0:
                fh.flush()
                print(f"    {arm}: {n_written}/{len(jobs)}", flush=True)
    fh.flush()
    fh.close()
    return n_written


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((row["arm"], int(row["pair_id"]), int(row["order"])))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, required=True, choices=config.K_LEVELS)
    ap.add_argument("--corner", required=True, choices=list(config.CORNERS))
    ap.add_argument("--semantic", required=True, choices=config.SEMANTICS)
    ap.add_argument("--rule", required=True, choices=config.RULES)
    ap.add_argument("--informative", action="store_true",
                    help="proxies genuinely carry outcome information")
    ap.add_argument("--arms", nargs="*", default=config.ARMS, choices=config.ARMS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--beta", type=float, default=None,
                    help="override BETA_INFO (proxy predictive strength) for a "
                         "dose-response sweep; implies --informative. The frozen "
                         "value is %.2f — only pass this for the beta sweep."
                         % config.BETA_INFO)
    ap.add_argument("--via", choices=["api", "codex"], default="api",
                    help="api = bare single-turn chat completion (default); "
                         "codex = shell out to `codex exec` (agent-wrapped — see "
                         "codex_client.py before interpreting)")
    ap.add_argument("--seed", type=int, default=None,
                    help="replication seed: redraws population, pairs and "
                         "calibration together (frozen config when omitted); "
                         "cells are tagged _s<seed>")
    ap.add_argument("--resume", action="store_true",
                    help="skip calls already present in the output CSV")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and print one prompt; make no API calls")
    args = ap.parse_args()

    if args.beta is not None:
        args.informative = True     # a beta level only means anything when informative
        if args.rule != "learned":
            # beta changes w_P (and r_true / Y), never the attribute values X.
            # A rule-absent prompt shows only X, so it is byte-identical across
            # beta levels and the manipulation never reaches the model.
            ap.error("--beta requires --rule learned: with rule=%s the prompt is "
                     "byte-identical across beta levels (verified), so the sweep "
                     "would measure a null manipulation." % args.rule)

    ref, pairs, Xcal, Ycal, r_ideal = build_cell(
        args.k, args.corner, args.semantic, args.rule, args.informative,
        args.beta, args.seed)

    tag = (f"k{args.k}_{args.corner}_{args.semantic}_{args.rule}"
           f"_{'info' if args.informative else 'zero'}")
    if args.beta is not None:
        # keep sweep cells in their own files; the frozen beta keeps the plain tag
        tag += f"_b{int(round(args.beta*100)):03d}"
    if args.seed is not None:
        tag += f"_s{args.seed}"
    print(f"cell {tag}: {len(pairs)} pairs x {config.N_ORDERS} orders "
          f"x {len(args.arms)} arms = {len(pairs)*config.N_ORDERS*len(args.arms)} calls")
    if r_ideal is not None:
        print(f"  R_Ideal (no LLM needed) = {r_ideal:.4f}")

    if args.via == "codex":
        print("  via: codex exec  (AGENT-WRAPPED — not comparable to an --via api "
              "arm without the wrapper calibration; see codex_client.py)")

    if args.dry_run:
        p = prompts.build_prompt(args.k, args.semantic, args.rule,
                                 ref.X[pairs[0]['i']], ref.X[pairs[0]['j']], Xcal, Ycal)
        print("\n--- example prompt (first 1200 chars) ---")
        print(p[:1200])
        print(f"--- prompt length: {len(p)} chars ---")
        return

    if args.via == "codex":
        ccfg = CodexConfig.from_env()
        cli = CodexClient(ccfg)
        model_label = "codex_" + ccfg.model.replace("/", "_")
        if args.workers > 4:
            print(f"  note: --via codex spawns a process per call; "
                  f"lowering workers {args.workers} -> 4")
            args.workers = 4
    else:
        cli = Client(ModelConfig.from_env())
        model_label = os.environ["LLM_MODEL"].replace("/", "_")
    model_dir = os.path.join(args.outdir, model_label)
    os.makedirs(model_dir, exist_ok=True)
    out = os.path.join(model_dir, f"{tag}_records.csv")
    done = load_done(out) if args.resume else set()
    if done:
        print(f"  resuming; {len(done)} calls already recorded")

    for arm in args.arms:
        n = run_arm(cli, ref, pairs, args.k, args.semantic, args.rule,
                    Xcal, Ycal, arm, out, args.workers, done)
        print(f"  {arm}: wrote {n} records")

    # run metadata: capture what the records themselves cannot show
    import json, datetime
    meta_path = out.replace("_records.csv", "_metadata.json")
    meta = dict(
        cell=tag, model=model_label,
        base_url_host=os.environ.get("LLM_BASE_URL", "").split("//")[-1].split("/")[0],
        date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        channel=cli.channel() if hasattr(cli, "channel") else "codex",
        extra_json=os.environ.get("LLM_EXTRA_JSON", ""),
        seed=args.seed, beta=args.beta,
        workers=args.workers, resume=bool(done),
    )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"done -> {out}  (+metadata)")


if __name__ == "__main__":
    main()
