#!/usr/bin/env python3
"""Run the real-record zero-information audit with an explicit paid-call gate.

Without --execute this command performs offline checks and prints the call
budget only. With --execute it makes API calls. --pilot-pairs 99 runs only
the first 99 of the same frozen primary sample; a later --resume without
that limit completes the sample without changing its prompts or pair IDs.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from client import Client, ModelConfig
from external_zero_info import (SEED, evaluate, prepare,
                                unique_counterfactual_prompts)

CALL_FIELDS = ("job_id", "patient", "served_model", "fingerprint")
RECORD_FIELDS = ("arm", "pair_id", "i", "j", "order", "patient", "chosen",
                 "picked_i", "job_id", "proxy_changed")
EXPECTED_CHANNEL = "tools=yes reasoning_budget=no temperature=0.0"


class ProvenanceClient(Client):
    """Attach provider-returned identifiers to each parsed answer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._local = threading.local()

    def _post(self, payload: dict) -> dict:
        data = super()._post(payload)
        self._local.served = (str(data.get("model", "")),
                              str(data.get("system_fingerprint", "")))
        return data

    def choose_with_provenance(self, prompt: str) -> tuple[int | None, str, str]:
        self._local.served = ("", "")
        answer = self.choose(prompt)
        model, fingerprint = self._local.served
        return answer, model, fingerprint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_pairs(prepared, run_set: str) -> list[dict]:
    return {"primary": prepared.primary,
            "random_target": prepared.random_orientation,
            "gap_stratified": prepared.stratified}[run_set]


def load_calls(path: Path, expected_jobs: set[str]) -> dict[str, dict]:
    calls = {}
    if not path.exists():
        return calls
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            job_id = row["job_id"]
            if job_id in calls or job_id not in expected_jobs:
                raise ValueError(f"Duplicate or foreign job in {path}: {job_id}")
            calls[job_id] = row
    return calls


def summarize_served(calls: dict[str, dict]) -> list[dict]:
    counts = Counter((row["served_model"], row["fingerprint"])
                     for row in calls.values())
    return [dict(model=model, system_fingerprint=fingerprint, calls=n)
            for (model, fingerprint), n in sorted(counts.items())]


def write_metadata(path: Path, specification: dict, calls: dict[str, dict],
                   complete: bool) -> None:
    content = dict(specification)
    content.update(status="complete" if complete else "partial",
                   recorded_unique_calls=len(calls),
                   served=summarize_served(calls),
                   updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(content, indent=2) + "\n")


def write_logical_records(path: Path, pairs: list[dict], links: dict,
                          calls: dict[str, dict], prepared) -> None:
    A = prepared.design.panel.A.to_numpy()
    joint = prepared.design.panel.joint.to_numpy()
    from clinical_panel import category_under
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for pair_id, pair in enumerate(pairs):
            i, j = pair["i"], pair["j"]
            other = category_under(prepared.design, i, 1 - int(A[i]))
            changed = int(other != joint[i])
            for order in (0, 1):
                first = i if order == 0 else j
                for group, arm in ((1, "cf_a1"), (0, "cf_a0")):
                    job_id = links[(pair_id, group, order)]
                    answer = calls[job_id]["patient"]
                    chosen = ""
                    picked_i = ""
                    if answer in ("1", "2"):
                        chosen = first if answer == "1" else (j if first == i else i)
                        picked_i = int(chosen == i)
                    writer.writerow(dict(arm=arm, pair_id=pair_id, i=i, j=j,
                                         order=order, patient=answer, chosen=chosen,
                                         picked_i=picked_i, job_id=job_id,
                                         proxy_changed=changed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ids-mapping", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--ordering", choices=("fixed", "reverse", "seeded"),
                        default="fixed")
    parser.add_argument("--run-set", choices=("primary", "random_target",
                                               "gap_stratified"), default="primary")
    parser.add_argument("--pilot-pairs", type=int, default=0,
                        help="limit paid calls to the first N pairs; resume later")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-served",
                        help="provider-returned model ID required for paid calls")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly authorize API calls after offline checks")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.pairs <= 0 or not 1 <= args.workers <= 8:
        parser.error("--pairs must be positive and --workers must be 1..8")

    prepared = prepare(str(args.panel), n_primary=args.pairs,
                       ordering=args.ordering)
    gates = evaluate(prepared, str(args.ids_mapping))
    pairs = selected_pairs(prepared, args.run_set)
    jobs, links = unique_counterfactual_prompts(prepared, pairs)
    if args.pilot_pairs and not 1 <= args.pilot_pairs <= len(pairs):
        parser.error("--pilot-pairs must be within the selected pair set")
    eligible_jobs = {job_id: prompt for job_id, prompt in jobs.items()
                     if not args.pilot_pairs or int(job_id.split(":", 1)[0]) < args.pilot_pairs}
    print(f"{args.run_set}: {len(pairs)} pairs, {4 * len(pairs)} logical records, "
          f"{len(jobs)} unique paid prompts; this invocation permits "
          f"{len(eligible_jobs)} of them")

    if not args.execute:
        print("DRY RUN: no model calls and no output files")
        return
    if not all(gates.values()):
        raise SystemExit("Offline gates failed; no model calls made")
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
    stem = f"external_zero_{args.run_set}_{args.ordering}_p{args.pairs}"
    call_path = model_dir / f"{stem}_calls.csv"
    record_path = model_dir / f"{stem}_records.csv"
    meta_path = model_dir / f"{stem}_metadata.json"
    request_path = model_dir / f"{stem}_requests.jsonl"
    design_hash = hashlib.sha256(json.dumps(
        dict(jobs=sorted(jobs), run_set=args.run_set, ordering=args.ordering,
             pairs=args.pairs, seed=SEED), sort_keys=True).encode()).hexdigest()
    specification = dict(design_hash=design_hash, panel_sha256=file_sha256(args.panel),
                         mapping_sha256=file_sha256(args.ids_mapping),
                         requested_model=cfg.model,
                         provider=cfg.provider,
                         system_sha256=hashlib.sha256(Path(os.environ["LLM_SYSTEM_PROMPT_FILE"]).read_bytes()
                                        if os.environ.get("LLM_SYSTEM_PROMPT_FILE") else b"").hexdigest(),
                         base_url_host=cfg.base_url.split("//")[-1].split("/")[0],
                         expected_served=args.expected_served,
                         requested_channel=EXPECTED_CHANNEL,
                         run_set=args.run_set, ordering=args.ordering,
                         pairs=len(pairs), design_primary_pairs=args.pairs,
                         seed=SEED)
    if not args.resume and any(p.exists() for p in (call_path, record_path, meta_path, request_path)):
        raise SystemExit("Output already exists; use --resume with identical settings")
    if args.resume:
        if not meta_path.exists():
            raise SystemExit("Cannot resume without existing metadata")
        old = json.loads(meta_path.read_text())
        if any(old.get(key) != value for key, value in specification.items()):
            raise SystemExit("Resume metadata differs from current design")
    model_dir.mkdir(parents=True, exist_ok=True)
    if cfg.provider == "anthropic":
        os.environ["LLM_REQUEST_LOG"] = str(request_path)
    calls = load_calls(call_path, set(jobs))
    pending = [(job_id, prompt) for job_id, prompt in eligible_jobs.items()
               if job_id not in calls]
    if not pending:
        print("No new calls in the selected range")
        return

    client = ProvenanceClient(cfg)
    new_file = not call_path.exists()
    with call_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CALL_FIELDS)
        if new_file:
            writer.writeheader()

        def record(job_id: str, result: tuple[int | None, str, str]) -> None:
            answer, served_model, fingerprint = result
            row = dict(job_id=job_id, patient="" if answer is None else str(answer),
                       served_model=served_model, fingerprint=fingerprint)
            writer.writerow(row)
            stream.flush()
            calls[job_id] = row

        first_id, first_prompt = pending.pop(0)
        record(first_id, client.choose_with_provenance(first_prompt))
        write_metadata(meta_path, specification, calls, complete=False)
        if (calls[first_id]["patient"] == ""
                or client.channel() != EXPECTED_CHANNEL
                or calls[first_id]["served_model"] != args.expected_served):
            raise SystemExit("Preflight response/channel/model mismatch; stopped after one call")
        print("Preflight call passed; continuing with the selected job range")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            batch_size = max(8, 4 * args.workers)
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                futures = {executor.submit(client.choose_with_provenance, prompt): job_id
                           for job_id, prompt in batch}
                for future in as_completed(futures):
                    record(futures[future], future.result())
                write_metadata(meta_path, specification, calls, complete=False)
                if (client.channel() != EXPECTED_CHANNEL
                        or any(calls[job_id]["served_model"] != args.expected_served
                               for job_id, _ in batch)):
                    raise SystemExit("Serving channel/model changed; stopped after a batch")
    complete = len(calls) == len(jobs)
    write_metadata(meta_path, specification, calls, complete=complete)
    if complete:
        write_logical_records(record_path, pairs, links, calls, prepared)
    if client.channel() != EXPECTED_CHANNEL:
        raise SystemExit("Serving channel changed during the run; inspect metadata")
    if any(row["served_model"] != args.expected_served for row in calls.values()):
        raise SystemExit("Served model changed during the run; inspect metadata")
    print(f"recorded {len(calls)}/{len(jobs)} unique calls; "
          f"{'complete' if complete else 'paused'} -> {model_dir}")


if __name__ == "__main__":
    main()
