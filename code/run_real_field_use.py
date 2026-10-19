#!/usr/bin/env python3
"""Fixed-category positive control for proxy use in the real-record task.

The group-conditional replacement used in the main analysis rarely changes
payer, so it cannot show whether the model uses the field at all. This control
holds the patient, the task fields and the other patient fixed and changes only
the target's payer/admission category:

    high  Medicaid | Referral
    low   Blue Cross/Blue Shield | Referral

The outcome rule adds a known weight to the Medicaid/self-pay indicator, so the
full-information and ideal-learner references are nonzero. If the model's
choice probability moves with the category, it uses the field; if it does not,
the original zero-weight null is uninformative about field use.

    # offline: prompt counts and references, no calls
    python3 code/run_real_field_use.py --panel ... --ids-mapping ... \
        --pairs 1200 --payer-weight 0.05

    # paid run
    python3 code/run_real_field_use.py ... --execute \
        --outdir runs/real_field_use --expected-served claude-sonnet-4-5-20250929

The records are one row per pair, arm and presentation order.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "modules"))

import clinical_panel as prior
import oracle
from client import Client, ModelConfig
from external_zero_info import (SEED, SIGMA_Y, patient_fields, prepare,
                                rendered_risk)

HIGH = "Medicaid | Referral"
LOW = "Blue Cross/Blue Shield | Referral"
ARMS = (("high", HIGH), ("low", LOW))
EXPECTED_CHANNEL = "tools=yes reasoning_budget=no temperature=0.0"
CALL_FIELDS = ("job_id", "patient", "served_model", "fingerprint")
RECORD_FIELDS = ("pair_id", "arm", "order", "patient", "chosen", "picked_i",
                 "prompt_sha256")


class ProvenanceClient(Client):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._local = threading.local()

    def _post(self, payload: dict) -> dict:
        data = super()._post(payload)
        self._local.served = (str(data.get("model", "")),
                              str(data.get("system_fingerprint", "")))
        return data

    def choose_with_provenance(self, prompt: str):
        self._local.served = ("", "")
        answer = self.choose(prompt)
        return answer, *self._local.served


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_fixed_prompt(prepared, pair: dict, target_category: str,
                        order: int = 0) -> str:
    d = prepared.design
    i, j = pair["i"], pair["j"]
    own = d.panel.joint.to_numpy()
    patients = ((i, target_category), (j, own[j]))
    if order == 1:
        patients = patients[::-1]
    names = [name for name, _ in patient_fields(prepared, int(d.cal_idx[0]),
                                                own[d.cal_idx[0]])]
    examples = []
    for t in d.cal_idx:
        values = [value for _, value in patient_fields(prepared, int(t), own[t])]
        examples.append(", ".join(values) + f" -> {prepared.synthetic_y[t]:.2f}")
    blocks = []
    for number, (idx, category) in enumerate(patients, 1):
        body = "\n".join(f"  {name}: {value}" for name, value in
                         patient_fields(prepared, idx, category))
        blocks.append(f"Patient {number}:\n{body}")
    return ("You are planning follow-up calls for patients being discharged "
            "home from hospital. Below are previous patients with assigned "
            "readmission risk scores. Use the examples to decide which of the two "
            "patients should receive a follow-up call first.\n\n"
            "PREVIOUS PATIENTS (format: " + ", ".join(names) + " -> risk):\n"
            + "\n".join(examples) + "\n\nNEW PATIENTS:\n" + "\n".join(blocks))


def build_jobs(prepared, pairs):
    jobs = {}
    for pair_id, pair in enumerate(pairs):
        for arm, category in ARMS:
            for order in (0, 1):
                prompt = render_fixed_prompt(prepared, pair, category, order)
                sha = hashlib.sha256(prompt.encode()).hexdigest()
                jobs[(pair_id, arm, order)] = (sha, prompt)
    return jobs


def references(prepared, pairs):
    d = prepared.design
    joint = d.panel.joint.to_numpy()
    Xcal = prior.ideal_features(d, d.cal_idx, joint[d.cal_idx].tolist())
    post = oracle.blr_fit(Xcal, prepared.synthetic_y[d.cal_idx],
                          sigma=SIGMA_Y, tau=1.0)
    full_high, full_low, ideal_high, ideal_low = [], [], [], []
    for pair in pairs:
        i, j = pair["i"], pair["j"]
        comparator = rendered_risk(prepared, j, joint[j])
        full_high.append(rendered_risk(prepared, i, HIGH) >= comparator)
        full_low.append(rendered_risk(prepared, i, LOW) >= comparator)
        xj = prior.ideal_features(d, np.array([j]), [joint[j]])
        comp = oracle.blr_predict_mean(post, xj)[0]
        for store, category in ((ideal_high, HIGH), (ideal_low, LOW)):
            xi = prior.ideal_features(d, np.array([i]), [category])
            store.append(float(oracle.blr_predict_mean(post, xi)[0] >= comp))
    return (100 * (np.mean(full_high) - np.mean(full_low)),
            100 * (np.mean(ideal_high) - np.mean(ideal_low)))


def write_records(path: Path, jobs, calls):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for (pair_id, arm, order), (sha, _) in jobs.items():
            row = calls.get(sha, {})
            patient = int(row["patient"]) if row.get("patient", "") != "" else None
            target = 1 if order == 0 else 2
            writer.writerow(dict(pair_id=pair_id, arm=arm, order=order,
                                 patient="" if patient is None else patient,
                                 chosen="" if patient is None else
                                 (1 if patient == target else 2),
                                 picked_i="" if patient is None else
                                 int(patient == target),
                                 prompt_sha256=sha))


def interval(values: np.ndarray, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    if not len(values):
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(float(x) for x in np.quantile(values[idx].mean(1), (0.025, 0.975)))


def analyze(records: Path, draws: int = 5000) -> dict:
    rows = list(csv.DictReader(records.open(newline="")))
    pairs = {}
    by_order = {0: {}, 1: {}}
    for row in rows:
        if row["picked_i"] == "":
            continue
        key = int(row["pair_id"])
        pairs.setdefault(key, {}).setdefault(row["arm"], []).append(int(row["picked_i"]))
        by_order[int(row["order"])].setdefault(key, {}).setdefault(row["arm"], []).append(int(row["picked_i"]))
    def effects(store):
        vals = []
        for pair in store.values():
            if "high" in pair and "low" in pair:
                vals.append(np.mean(pair["high"]) - np.mean(pair["low"]))
        return 100 * np.asarray(vals, dtype=float)
    main = effects(pairs)
    out = {"pairs": int(len(main)), "pse_pp": float(main.mean()) if len(main) else float("nan")}
    lo, hi = interval(main, draws, 0)
    out.update(ci_lo=lo, ci_hi=hi)
    for order in (0, 1):
        vals = effects(by_order[order])
        out[f"order_{order}_pp"] = float(vals.mean()) if len(vals) else float("nan")
    return out


def write_metadata(path: Path, spec: dict, calls: dict, jobs: dict, complete: bool) -> None:
    served = {}
    for row in calls.values():
        key = (row["served_model"], row["fingerprint"])
        served[key] = served.get(key, 0) + 1
    meta = dict(spec)
    meta.update(status="complete" if complete else "partial",
                answered=len(calls), logical_jobs=len(jobs),
                missing=len(jobs) - len(calls),
                served=[dict(model=m, system_fingerprint=f, calls=n)
                        for (m, f), n in sorted(served.items())])
    path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ids-mapping", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--payer-weight", type=float, default=0.05)
    parser.add_argument("--pilot-pairs", type=int, default=0)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-served")
    parser.add_argument("--reuse-records", type=Path,
                        help="records.csv from an earlier run whose prompts are a "
                             "subset of this design; matching answers are reused")
    parser.add_argument("--reuse-metadata", type=Path,
                        help="metadata.json from the reused run; checked against this design")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.pairs <= 0 or not 1 <= args.workers <= 8:
        parser.error("--pairs must be positive and --workers must be 1..8")

    prepared = prepare(str(args.panel), n_primary=args.pairs,
                       ordering="fixed", payer_weight=args.payer_weight)
    pairs = prepared.primary[:args.pilot_pairs] if args.pilot_pairs else prepared.primary
    jobs = build_jobs(prepared, pairs)
    full_ref, ideal_ref = references(prepared, pairs)
    print(f"pairs: {len(pairs)}; logical jobs: {len(jobs)}; "
          f"unique prompts: {len({sha for sha, _ in jobs.values()})}")
    print(f"full-information reference PSE: {full_ref:+.2f} pp; "
          f"ideal-learner PSE: {ideal_ref:+.2f} pp")
    if not args.execute:
        by_sha = {sha for sha, _ in jobs.values()}
        if args.reuse_records:
            reused = sum(1 for row in csv.DictReader(args.reuse_records.open(newline=""))
                         if row.get("prompt_sha256") in by_sha
                         and row.get("patient", "") != "")
            print(f"reused {reused} calls; new prompts to send: {len(by_sha) - reused}")
        print("DRY RUN: no model calls and no output files")
        return
    if not args.outdir or not args.expected_served:
        parser.error("--execute requires --outdir and --expected-served")
    if (os.environ.get("LLM_NO_TEMPERATURE") == "1"
            or os.environ.get("LLM_NO_TOOLS") == "1"
            or os.environ.get("LLM_REASONING") == "1"
            or os.environ.get("LLM_EXTRA_JSON")):
        raise SystemExit("Unexpected LLM_* override; no model calls made")
    cfg = ModelConfig.from_env()
    if cfg.temperature != 0.0:
        raise SystemExit("Requested temperature is not zero; no calls made")

    model_dir = args.outdir / cfg.model.replace("/", "_")
    stem = (f"real_field_use_p{args.pairs}_w{args.payer_weight:g}"
            + (f"_pilot{args.pilot_pairs}" if args.pilot_pairs else ""))
    call_path = model_dir / f"{stem}_calls.csv"
    record_path = model_dir / f"{stem}_records.csv"
    meta_path = model_dir / f"{stem}_metadata.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in (call_path, record_path, meta_path)):
        raise SystemExit("Output exists; use a fresh --outdir")

    # deduplicate identical prompts within the run
    by_sha = {}
    for sha, prompt in jobs.values():
        by_sha[sha] = prompt
    spec = dict(model=cfg.model, provider=cfg.provider,
                base_url_host=cfg.base_url.split("//")[-1].split("/")[0],
                high=HIGH, low=LOW, payer_weight=args.payer_weight,
                pairs=len(pairs), design_pairs=args.pairs,
                panel_sha256=file_sha256(args.panel),
                mapping_sha256=file_sha256(args.ids_mapping),
                date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                full_information_pse_pp=float(full_ref),
                ideal_learner_pse_pp=float(ideal_ref),
                expected_served=args.expected_served,
                request_channel=EXPECTED_CHANNEL)
    calls = {}
    if args.reuse_records:
        if args.reuse_metadata:
            old = json.loads(args.reuse_metadata.read_text())
            checks = dict(payer_weight=args.payer_weight, high=HIGH, low=LOW,
                          panel_sha256=spec["panel_sha256"],
                          mapping_sha256=spec["mapping_sha256"],
                          design_pairs=args.pairs)
            for key, value in checks.items():
                if old.get(key) != value:
                    raise SystemExit(f"Reused run differs on {key}: "
                                     f"{old.get(key)!r} != {value!r}")
        for row in csv.DictReader(args.reuse_records.open(newline="")):
            sha = row.get("prompt_sha256", "")
            if sha in by_sha and row.get("patient", "") != "":
                calls[sha] = dict(job_id=sha, patient=row["patient"],
                                  served_model=args.expected_served or "",
                                  fingerprint="reused")
        print(f"reused {len(calls)} calls from {args.reuse_records}")
        spec["reused_records"] = str(args.reuse_records)
        spec["reused_calls"] = len(calls)
    pending = [sha for sha in sorted(by_sha) if sha not in calls]
    print(f"new prompts to send: {len(pending)}")
    if pending:
        client = ProvenanceClient(cfg)
        first, pending = pending[0], pending[1:]
        answer, served, fingerprint = client.choose_with_provenance(by_sha[first])
        calls[first] = dict(job_id=first, patient="" if answer is None else str(answer),
                            served_model=served, fingerprint=fingerprint)
        write_metadata(meta_path, spec, calls, jobs, complete=False)
        if (calls[first]["patient"] == "" or client.channel() != EXPECTED_CHANNEL
                or served != args.expected_served):
            raise SystemExit("Preflight response/channel/model mismatch; stopped after one call")
        print("Preflight call passed")
    else:
        client = None
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(client.choose_with_provenance, by_sha[sha]): sha
                       for sha in pending}
            for n, future in enumerate(as_completed(futures), 1):
                sha = futures[future]
                answer, served, fingerprint = future.result()
                calls[sha] = dict(job_id=sha, patient="" if answer is None else str(answer),
                                  served_model=served, fingerprint=fingerprint)
                if n % 20 == 0:
                    print(f"    {n}/{len(pending)}", flush=True)
    write_records(record_path, jobs, calls)
    write_metadata(meta_path, spec, calls, jobs, complete=len(calls) == len(jobs))
    result = analyze(record_path)
    print(json.dumps(result, indent=1))
    print(f"done -> {record_path}")


if __name__ == "__main__":
    main()
