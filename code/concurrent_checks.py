#!/usr/bin/env python3
"""Two supplementary Sonnet checks with a locked protocol.

A. Concurrent regeneration comparison (synthetic task).
   The frozen counterfactual (target proxies regenerated with the same
   exogenous noise) and stochastic regeneration (five fresh redraws of the
   target's proxy noise, shared by both arms) are rendered from the same
   frozen cells by the same code and sent in one randomly interleaved queue.
   Cells: k = 12, LH, neutral names, eighty examples, zero-information and
   beta = 0.60. The comparison is the within-pair difference between schemes.

B. Near-tie real-record check (follow-up-priority task).
   An independent, patient-disjoint sample of pairs whose readmission-anchored
   risk scores differ by less than 0.3 standard deviations of the training
   score, selected by a fixed rule that uses only the risk score. A concurrent
   control of primary pairs with gap at least 0.3 is re-sent in the same
   randomly interleaved queue, so the two groups are compared at the same
   time.

Subcommands (none makes a model call):

    python concurrent_checks.py export-regen --outdir ../bundle/jobs
    python concurrent_checks.py export-neartie --panel P --ids-mapping M \
        --outdir ../bundle/jobs
    python concurrent_checks.py analyze-regen --jobs J --answers ANS
    python concurrent_checks.py analyze-neartie --panel P --jobs J --answers ANS

Answers are JSONL lines {"job_id", "patient", "served_model", "stop_reason",
"utc"} written by run_jobs.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "modules"))

MODEL = "claude-sonnet-4-5-20250929"
QUEUE_SEED = 20261005          # interleaving order of the regeneration queue
NEARTIE_SEED = 20261006        # near-tie pair selection and queue order
NEARTIE_GAP = 0.3              # |risk_i - risk_j| < 0.3 training-score SD
NEARTIE_PAIRS = 600
WIDE_PAIRS = 300               # concurrent control: primary pairs with gap >= 0.3
MARGIN_REGEN = 10.0            # pp, equivalence margin for frozen - stochastic
MARGIN_NEARTIE = 5.0           # pp, margin for a meaningful average effect
B = 5000
REGEN_CELLS = (("zero", None), ("b060", 0.60))
DRAWS = 5


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jobs(path: Path, jobs: list[tuple[str, str]], seed: int) -> list[str]:
    """Write unique jobs in a seeded random order. Returns the order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(jobs))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for seq, k in enumerate(order):
            job_id, prompt = jobs[k]
            fh.write(json.dumps(dict(seq=seq, job_id=job_id,
                                     prompt_sha256=sha(prompt),
                                     prompt=prompt)) + "\n")
    return [jobs[k][0] for k in order]


def read_answers(paths) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["job_id"] in out:
                    raise SystemExit(f"duplicate answer for {r['job_id']}")
                out[r["job_id"]] = r
    return out


def boot(values: np.ndarray, seed: int = 0) -> tuple[float, float, float, float, float]:
    """Mean, 95% and 90% percentile intervals over resampled units (in pp)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.array([values[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
    lo90, hi90 = np.percentile(draws, [5, 95])
    return (100 * values.mean(), 100 * lo95, 100 * hi95, 100 * lo90, 100 * hi90)


# ------------------------------------------------------------------ A: regen
def regen_records():
    import config
    import dgp_k
    import prompts
    from run_cell import build_cell, pair_sample
    from stochastic_regeneration import fresh_noise_sample

    records, jobs = [], {}
    pair_sets = {}
    for cell, beta in REGEN_CELLS:
        ref, pairs, Xcal, Ycal, _ = build_cell(12, "LH", "neutral", "learned",
                                               beta is not None, beta)
        pair_sets[cell] = [(int(p["i"]), int(p["j"])) for p in pairs]
        for pid, p in enumerate(pairs):
            i, j = int(p["i"]), int(p["j"])
            variants = [("frozen", -1, None)] + [("stochastic", d, d)
                                                 for d in range(DRAWS)]
            for scheme, draw, d in variants:
                base = ref if d is None else fresh_noise_sample(ref, i, pid, d)
                for arm, aval in (("cf_a1", 1), ("cf_a0", 0)):
                    if d is None:
                        sample = pair_sample(ref, arm, p)
                    else:
                        newA = ref.A.copy()
                        newA[i] = aval
                        sample = dgp_k.counterfactual(base, newA)
                    for order in range(config.N_ORDERS):
                        a, b = (i, j) if order == 0 else (j, i)
                        pr = prompts.build_prompt(12, "neutral", "learned",
                                                  sample.X[a], sample.X[b],
                                                  Xcal, Ycal)
                        h = sha(pr)
                        if h in jobs and jobs[h] != pr:
                            raise SystemExit("SHA-256 collision")
                        jobs[h] = pr
                        records.append(dict(cell=cell, scheme=scheme, draw=draw,
                                            pair_id=pid, i=i, j=j, order=order,
                                            arm=arm, first=a, job_id=h))
    return records, jobs, pair_sets


def export_regen(args) -> None:
    out = Path(args.outdir)
    records, jobs, pair_sets = regen_records()
    root = _HERE.parent / "data/canonical/claude-sonnet-4-5"
    for cell, beta in REGEN_CELLS:
        tag = "k12_LH_neutral_learned_zero" if beta is None else \
            "k12_LH_neutral_learned_info_b060"
        arch = root / f"{tag}_records.csv"
        if arch.exists():
            with open(arch) as f:
                ap = {(int(r["pair_id"]), int(r["i"]), int(r["j"]))
                      for r in csv.DictReader(f)}
            mine = {(pid, i, j) for pid, (i, j) in enumerate(pair_sets[cell])}
            print(f"{cell}: pairs identical to battery cell: {ap == mine}")
    n_dup = len(records) - len({r["job_id"] for r in records})
    order = write_jobs(out / "regen_jobs.jsonl", list(jobs.items()), QUEUE_SEED)
    with open(out / "regen_links.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    by = defaultdict(int)
    for r in records:
        by[(r["cell"], r["scheme"])] += 1
    print(f"logical records {len(records)}; unique prompts {len(jobs)}; "
          f"duplicated logical prompts {n_dup}")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:5s} {k[1]:10s} {v}")
    first = order[:40]
    schemes = [next(r["scheme"] for r in records if r["job_id"] == h) for h in first]
    print("first 40 queued jobs by scheme:", "".join("F" if s == "frozen" else "s"
                                                     for s in schemes))


def analyze_regen(args) -> None:
    records = list(csv.DictReader(open(args.links)))
    ans = read_answers(args.answers)
    acc = defaultdict(list)
    missing = 0
    for r in records:
        a = ans.get(r["job_id"])
        if a is None or a.get("patient") not in (1, 2):
            missing += 1
            continue
        first, i, j = int(r["first"]), int(r["i"]), int(r["j"])
        chosen = first if a["patient"] == 1 else (j if first == i else i)
        acc[(r["cell"], r["scheme"], int(r["draw"]), int(r["pair_id"]),
             r["arm"])].append(int(chosen == i))
    served = {a.get("served_model") for a in ans.values()}
    print(f"answers {len(ans)}; logical records without a valid answer "
          f"{missing}; served models {sorted(map(str, served))}")

    def contrast(cell, scheme):
        draws = [-1] if scheme == "frozen" else list(range(DRAWS))
        out = {}
        for pid in range(99):
            vals = []
            for d in draws:
                k1 = (cell, scheme, d, pid, "cf_a1")
                k0 = (cell, scheme, d, pid, "cf_a0")
                if acc.get(k1) and acc.get(k0):
                    vals.append(np.mean(acc[k1]) - np.mean(acc[k0]))
            if len(vals) == len(draws):
                out[pid] = float(np.mean(vals))
        return out

    diffs, effects = {}, {}
    for cell, _ in REGEN_CELLS:
        f, s = contrast(cell, "frozen"), contrast(cell, "stochastic")
        common = sorted(set(f) & set(s))
        fz = np.array([f[p] for p in common])
        st = np.array([s[p] for p in common])
        effects[cell] = (dict(zip(common, fz)), dict(zip(common, st)))
        diffs[cell] = dict(zip(common, fz - st))
        for name, v in (("frozen", fz), ("stochastic", st),
                        ("frozen - stochastic", fz - st)):
            m, lo, hi, l9, h9 = boot(v)
            print(f"{cell:5s} {name:20s} {m:+6.2f} pp  95% [{lo:+.2f}, {hi:+.2f}]"
                  f"  90% [{l9:+.2f}, {h9:+.2f}]  pairs {len(v)}")
    common = sorted(set(diffs["zero"]) & set(diffs["b060"]))
    pooled = np.array([(diffs["zero"][p] + diffs["b060"][p]) / 2 for p in common])
    m, lo, hi, l9, h9 = boot(pooled)
    equiv = (l9 > -MARGIN_REGEN) and (h9 < MARGIN_REGEN)
    print(f"pooled frozen - stochastic {m:+.2f} pp 95% [{lo:+.2f}, {hi:+.2f}] "
          f"90% [{l9:+.2f}, {h9:+.2f}] on {len(common)} pairs; "
          f"within +/-{MARGIN_REGEN:g} pp by 90% interval: {equiv}")
    # two-point tracking slope under each scheme, reference 0 and 23.2 pp
    for k, name in ((0, "frozen"), (1, "stochastic")):
        z = np.array([effects["zero"][k][p] for p in common])
        b6 = np.array([effects["b060"][k][p] for p in common])
        rng = np.random.default_rng(1)
        sl = []
        for _ in range(B):
            idx = rng.integers(0, len(common), len(common))
            sl.append((b6[idx].mean() - z[idx].mean()) / 0.232)
        print(f"two-point slope, {name}: {(b6.mean() - z.mean()) / 0.232:+.3f} "
              f"95% [{np.percentile(sl, 2.5):+.3f}, {np.percentile(sl, 97.5):+.3f}]")
    # interleaving check: call times by scheme
    times = defaultdict(list)
    for r in records:
        a = ans.get(r["job_id"])
        if a and a.get("utc"):
            times[r["scheme"]].append(a["utc"])
    for k, v in times.items():
        print(f"{k}: first call {min(v)}, last call {max(v)}")
    # agreement with other collections of the same frozen prompts (descriptive)
    canon = _HERE.parent / "data/canonical/claude-sonnet-4-5"
    second = _HERE.parent / ("data/encoding_control/claude-sonnet-4-5-20250929/"
                             "catctl_k12_LH_neutral_learned_zero_numeric_remeasure_records.csv")
    for cell, label, arch in (
            ("zero", "battery", canon / "k12_LH_neutral_learned_zero_records.csv"),
            ("b060", "battery", canon / "k12_LH_neutral_learned_info_b060_records.csv"),
            ("zero", "second numeric measurement", second)):
        if not arch.exists():
            continue
        old = {}
        for r in csv.DictReader(open(arch)):
            if r["arm"] in ("cf_a1", "cf_a0") and r["picked_i"] != "":
                old[(int(float(r["i"])), int(float(r["j"])), r["arm"],
                     int(float(r["order"])))] = int(float(r["picked_i"]))
        agree = n = 0
        for r in records:
            if r["cell"] != cell or r["scheme"] != "frozen":
                continue
            key = (int(r["i"]), int(r["j"]), r["arm"], int(r["order"]))
            a = ans.get(r["job_id"])
            if key in old and a and a.get("patient") in (1, 2):
                first, i, j = int(r["first"]), int(r["i"]), int(r["j"])
                chosen = first if a["patient"] == 1 else (j if first == i else i)
                agree += int(int(chosen == i) == old[key])
                n += 1
        print(f"{cell}: frozen choices agree with the {label} on {agree}/{n} calls")


# ---------------------------------------------------------------- B: near tie
def neartie_pairs(prepared) -> list[dict]:
    """Fixed selection rule. Uses only the risk score and patient indices."""
    d = prepared.design
    risk = d.wL
    used = set(int(x) for x in d.cal_idx)
    for p in prepared.primary + prepared.stratified:
        used |= {int(p["i"]), int(p["j"])}
    pool = np.array(sorted(int(x) for x in prepared.study if int(x) not in used))
    rng = np.random.default_rng(NEARTIE_SEED)
    order = rng.permutation(pool)
    unpaired = set(order.tolist())
    pairs = []
    for a in order:
        a = int(a)
        if a not in unpaired:
            continue
        unpaired.discard(a)
        cand = sorted(b for b in unpaired if abs(risk[a] - risk[b]) < NEARTIE_GAP)
        if not cand:
            unpaired.add(a)
            continue
        b = int(cand[rng.integers(0, len(cand))])
        unpaired.discard(b)
        i, j = (a, b) if risk[a] >= risk[b] else (b, a)
        pairs.append(dict(i=i, j=j, gap=float(abs(risk[a] - risk[b]))))
        if len(pairs) == NEARTIE_PAIRS:
            break
    if len(pairs) < NEARTIE_PAIRS:
        raise SystemExit(f"only {len(pairs)} near-tie pairs could be formed")
    return pairs


def neartie_prepare(panel):
    import external_zero_info as ez
    prepared = ez.prepare(panel, n_primary=1200, n_stratified=150,
                          ordering="fixed")
    return ez, prepared


def wide_pairs(prepared) -> list[dict]:
    """Fixed random subset of primary pairs with gap >= 0.3 (risk score only)."""
    risk = prepared.design.wL
    eligible = [(pid, p) for pid, p in enumerate(prepared.primary)
                if abs(risk[p["i"]] - risk[p["j"]]) >= NEARTIE_GAP]
    rng = np.random.default_rng(NEARTIE_SEED + 1)
    pick = sorted(rng.choice(len(eligible), WIDE_PAIRS, replace=False).tolist())
    return [dict(i=int(eligible[k][1]["i"]), j=int(eligible[k][1]["j"]),
                 gap=float(abs(risk[eligible[k][1]["i"]] - risk[eligible[k][1]["j"]])),
                 primary_pair_id=int(eligible[k][0])) for k in pick]


def export_neartie(args) -> None:
    ez, prepared = neartie_prepare(args.panel)
    if args.ids_mapping:
        ok, report = ez.mapping_gate(args.ids_mapping)
        print(f"official mapping gate: {ok} ({report})")
        if not ok:
            raise SystemExit("mapping gate failed")
    d = prepared.design
    A = d.panel.A.to_numpy()
    joint = d.panel.joint.to_numpy()
    groups = (("neartie", neartie_pairs(prepared)), ("wide", wide_pairs(prepared)))
    all_jobs, rows = {}, []
    for group_name, pairs in groups:
        jobs, links = ez.unique_counterfactual_prompts(prepared, pairs)
        prefix = "nt" if group_name == "neartie" else "wd"
        for job, prompt in jobs.items():
            all_jobs[f"{prefix}:{job}"] = prompt
        changed = [int(ez.prior.category_under(d, int(p["i"]), 1 - int(A[p["i"]]))
                       != joint[p["i"]]) for p in pairs]
        for (pid, arm_group, order), job in sorted(links.items()):
            p = pairs[pid]
            rows.append([group_name, pid, p.get("primary_pair_id", ""), p["i"],
                         p["j"], f"{p['gap']:.6f}", changed[pid], arm_group,
                         order, f"{prefix}:{job}"])
        gaps = np.array([p["gap"] for p in pairs])
        ref = ez.reference(prepared, [dict(i=p["i"], j=p["j"]) for p in pairs])
        print(f"{group_name}: pairs {len(pairs)}; gap median {np.median(gaps):.3f}, "
              f"range [{gaps.min():.3f}, {gaps.max():.3f}]; changed targets "
              f"{sum(changed)}/{len(pairs)} ({np.mean(changed):.1%}); unique prompts "
              f"{len(jobs)}; full-information {ref[0]:+.2f} pp; same-example "
              f"learner {ref[1]:+.2f} pp")
        if group_name == "neartie":
            print(f"planning: conservative 95% half-width "
                  f"{196 * np.sqrt(np.mean(changed) / len(pairs)):.1f} pp; with the "
                  f"variance of near-tie pairs in the primary sample (0.0293) "
                  f"{196 * np.sqrt(0.0293 / len(pairs)):.1f} pp")
    patients = [r[3] for r in rows] + [r[4] for r in rows]
    nt = {x for r in rows if r[0] == "neartie" for x in (r[3], r[4])}
    wd = {x for r in rows if r[0] == "wide" for x in (r[3], r[4])}
    print(f"near-tie and wide-gap patients disjoint: {not (nt & wd)}")
    prim = _HERE.parent / ("data/external_zero_info/claude-sonnet-4-5-20250929/"
                           "external_zero_primary_fixed_p1200_records.csv")
    if prim.exists():
        old = {row["job_id"].split(":")[-1] for row in csv.DictReader(open(prim))}
        mine = {j.split(":")[-1] for j in all_jobs if j.startswith("wd:")}
        print(f"wide-gap prompts identical to the primary prompts: "
              f"{len(mine & old)}/{len(mine)}")
    out = Path(args.outdir)
    write_jobs(out / "neartie_jobs.jsonl", list(all_jobs.items()), NEARTIE_SEED)
    with open(out / "neartie_links.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "pair_id", "primary_pair_id", "i", "j", "gap",
                    "proxy_changed", "group", "order", "job_id"])
        w.writerows(rows)
    print(f"total unique prompts {len(all_jobs)}")


def pair_contrasts(rows, ans, sample):
    """Per-pair PSE for one sample. An arm averages its valid orders. A pair
    needs a valid answer in both arms."""
    picks = defaultdict(list)
    pairs = {}
    missing = 0
    for r in rows:
        if r["sample"] != sample:
            continue
        pid = int(r["pair_id"])
        pairs[pid] = dict(i=int(r["i"]), j=int(r["j"]),
                          changed=int(r["proxy_changed"]),
                          primary_pair_id=r["primary_pair_id"])
        a = ans.get(r["job_id"])
        if a is None or a.get("patient") not in (1, 2):
            missing += 1
            continue
        order = int(r["order"])
        picks[(pid, int(r["group"]))].append(int(a["patient"] == (1 if order == 0 else 2)))
    pids = sorted(p for p in pairs if picks.get((p, 1)) and picks.get((p, 0)))
    c = np.array([np.mean(picks[(p, 1)]) - np.mean(picks[(p, 0)]) for p in pids])
    ch = np.array([pairs[p]["changed"] for p in pids], bool)
    return pids, c, ch, pairs, missing


def analyze_neartie(args) -> None:
    ez, prepared = neartie_prepare(args.panel)
    rows = list(csv.DictReader(open(args.links)))
    ans = read_answers(args.answers)
    res = {}
    for sample in ("neartie", "wide"):
        pids, c, ch, pairs, missing = pair_contrasts(rows, ans, sample)
        res[sample] = (c, ch)
        print(f"[{sample}] logical records without valid answer {missing}; "
              f"resolving pairs {len(pids)} of {len(pairs)}")
        m, lo, hi, *_ = boot(c)
        line = f"[{sample}] PSE {m:+.2f} pp 95% [{lo:+.2f}, {hi:+.2f}]"
        if sample == "neartie":
            line += (f"; 95% interval inside +/-{MARGIN_NEARTIE:g} pp: "
                     f"{(lo > -MARGIN_NEARTIE) and (hi < MARGIN_NEARTIE)}")
        print(line)
        m, lo, hi, *_ = boot(c[ch])
        print(f"[{sample}] changed targets only ({ch.sum()}): {m:+.2f} pp "
              f"[{lo:+.2f}, {hi:+.2f}]")
        ideal = ez.ideal_pair_effects(prepared, [dict(i=pairs[p]["i"], j=pairs[p]["j"])
                                                 for p in pids])
        m, lo, hi, *_ = boot(ideal)
        print(f"[{sample}] same-example learner {m:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
        m, lo, hi, *_ = boot(c - ideal)
        print(f"[{sample}] model minus learner {m:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
        if sample == "wide":
            res["wide_pids"] = (pids, pairs)
    (cn, _), (cw, _) = res["neartie"], res["wide"]
    rng = np.random.default_rng(2)
    dd = [cn[rng.integers(0, len(cn), len(cn))].mean()
          - cw[rng.integers(0, len(cw), len(cw))].mean() for _ in range(B)]
    print(f"near-tie minus concurrent wide-gap: {100 * (cn.mean() - cw.mean()):+.2f} pp "
          f"95% [{100 * np.percentile(dd, 2.5):+.2f}, {100 * np.percentile(dd, 97.5):+.2f}]")
    # agreement of the wide-gap control with the primary-sample answers (descriptive)
    prim = _HERE.parent / ("data/external_zero_info/claude-sonnet-4-5-20250929/"
                           "external_zero_primary_fixed_p1200_records.csv")
    if prim.exists():
        import pandas as pd
        r = pd.read_csv(prim)
        old_pick = {(row.job_id.split(":")[-1]): row.patient for row in
                    r.drop_duplicates("job_id").itertuples()}
        agree = n = 0
        for row in rows:
            if row["sample"] != "wide":
                continue
            a = ans.get(row["job_id"])
            key = row["job_id"].split(":")[-1]
            if a and a.get("patient") in (1, 2) and key in old_pick:
                agree += int(a["patient"] == old_pick[key])
                n += 1
        print(f"[control] wide-gap answers agree with the primary-sample answers "
              f"on {agree}/{n} logical records (descriptive)")
    times = defaultdict(list)
    for row in rows:
        a = ans.get(row["job_id"])
        if a and a.get("utc"):
            times[row["sample"]].append(a["utc"])
    for k, v in times.items():
        print(f"{k}: calls from {min(v)} to {max(v)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-regen")
    e.add_argument("--outdir", required=True)
    e = sub.add_parser("export-neartie")
    e.add_argument("--panel", required=True)
    e.add_argument("--ids-mapping")
    e.add_argument("--outdir", required=True)
    e = sub.add_parser("analyze-regen")
    e.add_argument("--links", required=True)
    e.add_argument("--answers", nargs="+", required=True)
    e = sub.add_parser("analyze-neartie")
    e.add_argument("--panel", required=True)
    e.add_argument("--links", required=True)
    e.add_argument("--answers", nargs="+", required=True)
    args = ap.parse_args()
    {"export-regen": export_regen, "export-neartie": export_neartie,
     "analyze-regen": analyze_regen, "analyze-neartie": analyze_neartie}[args.cmd](args)


if __name__ == "__main__":
    main()
