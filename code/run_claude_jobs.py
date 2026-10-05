#!/usr/bin/env python3
"""Run exported, frozen prompts through the native Claude API.

Without --execute, validate prompts and print the number of calls only.
Output answers can be ingested by categorical_proxy_control.py. Every
response attempt is retained. No hosted agent or orchestration API is used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from client import Client, ModelConfig


class RecordedClient(Client):
    def _post(self, payload):
        data = super()._post(payload)
        self.last = data
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", type=Path)
    ap.add_argument("--expected-served", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--invalid-attempts", type=int, default=1)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.invalid_attempts <= 5 or Path(args.name).name != args.name:
        ap.error("name must be a filename, invalid-attempts must be 1 to 5")
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    needed = {row["prompt_sha256"] for row in jobs}
    if len(needed) != len(jobs) or any(hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"] for row in jobs):
        raise SystemExit("Duplicate jobs or invalid prompt hashes")
    print(f"{args.name}: {len(jobs)} unique prompts")
    if not args.execute:
        print("DRY RUN, no model calls")
        return
    if args.outdir is None:
        ap.error("--execute requires --outdir")
    os.environ["LLM_PROVIDER"] = "anthropic"
    cfg = ModelConfig.from_env()
    if cfg.model != args.expected_served or any(os.environ.get(x) for x in ("LLM_EXTRA_JSON", "LLM_REASONING", "LLM_NO_TOOLS", "LLM_NO_TEMPERATURE")):
        raise SystemExit("Unexpected model or request override")
    args.outdir.mkdir(parents=True, exist_ok=True)
    answer_path = args.outdir / f"{args.name}.jsonl"
    meta_path = args.outdir / f"{args.name}_metadata.json"
    log_path = args.outdir / f"{args.name}_attempts.jsonl"
    system_file = os.environ.get("LLM_SYSTEM_PROMPT_FILE")
    system_text = Path(system_file).read_text() if system_file else ""
    spec = dict(model=cfg.model, provider=cfg.provider, base_url=cfg.base_url,
                jobs_sha256=hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
                system_sha256=hashlib.sha256(system_text.encode()).hexdigest(), invalid_attempts=args.invalid_attempts)
    if not args.resume and any(p.exists() for p in (answer_path, meta_path, log_path)):
        raise SystemExit("Output exists, use --resume with identical settings")
    if args.resume:
        if not meta_path.exists() or any(json.loads(meta_path.read_text()).get(k) != v for k, v in spec.items()):
            raise SystemExit("Resume specification differs")
    seen = set()
    if answer_path.exists():
        for line in answer_path.read_text().splitlines():
            row = json.loads(line)
            sha = row["prompt_sha256"]
            if sha in seen or sha not in needed or row.get("served_model") not in ("", args.expected_served):
                raise SystemExit("Duplicate or foreign stored answer")
            seen.add(sha)
    os.environ["LLM_REQUEST_LOG"] = str(log_path)
    cli = RecordedClient(cfg)
    def metadata():
        meta_path.write_text(json.dumps(dict(spec, recorded=len(seen), planned=len(jobs),
                             status="complete" if seen == needed else "partial",
                             updated_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    metadata()
    with answer_path.open("a") as stream:
        for job in jobs:
            if job["prompt_sha256"] in seen:
                continue
            patient = None
            for attempt in range(args.invalid_attempts):
                cli.last = {}
                patient = cli.choose(job["prompt"])
                served = cli.last.get("model", "")
                if served and served != args.expected_served:
                    raise SystemExit("Unexpected served model, inspect attempt log")
                if patient is not None:
                    break
            row = dict(prompt_sha256=job["prompt_sha256"], patient=patient,
                       served_model=cli.last.get("model", ""),
                       stop_reason=cli.last.get("native_stop_reason", ""))
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            seen.add(job["prompt_sha256"])
            metadata()
    print(f"Saved {len(seen)} answers to {answer_path}")


if __name__ == "__main__":
    main()
