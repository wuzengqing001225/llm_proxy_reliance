#!/usr/bin/env python3
"""Answer exported jobs in their fixed queue order, resumably.

Each jobs file is a JSONL queue ({"seq", "job_id", "prompt_sha256", "prompt"}).
Jobs are sent strictly in file order, in consecutive batches. The caller must
return exactly one result per request, in request order. Any other return
(wrong length, non-list, exception) stops the round with outcome
`caller_error` and consumes no attempt. Every attempt is
logged with its UTC time. A job with an accepted answer is never re-sent. An
invalid answer (error, no tool call, missing patient) is retried up to
MAX_ATTEMPTS model attempts. Requests rejected before reaching the model
(rate or quota limits) are logged and re-sent later. The run stops cleanly
when such a limit is hit and resumes from the saved state. Final outcomes:
`complete` (every job has a valid answer) or `complete_with_failures` (some
jobs exhausted their attempts and are recorded with patient null).

Request settings: model claude-sonnet-4-5-20250929, one user message with the
prompt text exactly, temperature 0, max_tokens 64, forced tool `choose`, no
system prompt and no thinking field.

Direct Claude API use:

    export ANTHROPIC_API_KEY=...
    python run_jobs.py --bundle . --state state --api

From Python with another caller that takes a list of request dicts and
returns one result dict per request ({"tool_use": {...}, "model", "stop_reason"}
or {"error": "..."}):

    import run_jobs
    run_jobs.run(call_batch, bundle=".", state="state")
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

MODEL = "claude-sonnet-4-5-20250929"
TOOL = {"name": "choose", "description": "Select which patient to prioritize.",
        "input_schema": {"type": "object",
                         "properties": {"patient": {"type": "integer", "enum": [1, 2]}},
                         "required": ["patient"]}}
QUEUE = ("regen_jobs", "neartie_jobs")    # execution order
MAX_ATTEMPTS = 5
BATCH = 8
LIMIT_WORDS = ("ceiling", "rate limit", "rate_limit", "quota", "overloaded",
               "429", "529", "budget")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def request(prompt: str) -> dict:
    return dict(model=MODEL, max_tokens=64, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                tools=[TOOL], tool_choice={"type": "tool", "name": "choose"})


def valid_patient(res) -> int | None:
    if not isinstance(res, dict) or res.get("error"):
        return None
    tu = res.get("tool_use") or {}
    if tu.get("name") != "choose":
        return None
    p = (tu.get("input") or {}).get("patient")
    return p if p in (1, 2) and not isinstance(p, bool) else None


def read_jsonl(path):
    return [json.loads(x) for x in open(path) if x.strip()] if os.path.exists(path) else []


def load_queue(bundle: str, name: str) -> list[dict]:
    path = os.path.join(bundle, "jobs", f"{name}.jsonl")
    jobs = read_jsonl(path)
    seen = set()
    for k, r in enumerate(jobs):
        if r["seq"] != k or r["job_id"] in seen or \
                hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]:
            raise SystemExit(f"corrupt queue {name} at line {k}")
        seen.add(r["job_id"])
    return jobs


def check_bundle(bundle: str) -> None:
    sums = os.path.join(bundle, "SHA256SUMS.txt")
    for line in open(sums):
        h, rel = line.split()
        with open(os.path.join(bundle, rel), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != h:
                raise SystemExit(f"checksum mismatch: {rel}")


def status(bundle: str, state: str) -> dict:
    out = {}
    for name in QUEUE:
        jobs = load_queue(bundle, name)
        ans = read_jsonl(os.path.join(state, f"{name}_answers.jsonl"))
        att = read_jsonl(os.path.join(state, f"{name}_attempts.jsonl"))
        ids = {a["job_id"] for a in ans}
        out[name] = dict(jobs=len(jobs), answered=len(ids),
                         duplicate_answers=len(ans) - len(ids),
                         valid=sum(a["patient"] in (1, 2) for a in ans),
                         attempts=len(att),
                         limit_rejections=sum(bool(a.get("limit")) for a in att),
                         caller_errors=sum(bool(a.get("caller_error")) for a in att),
                         failed_jobs=sum(a["patient"] not in (1, 2) for a in ans),
                         served=sorted({str(a.get("served_model")) for a in ans}))
    return out


def run(call_batch, bundle: str = ".", state: str = "state",
        max_new: int | None = None) -> dict:
    check_bundle(bundle)
    os.makedirs(state, exist_ok=True)
    new = 0
    for name in QUEUE:
        jobs = load_queue(bundle, name)
        ans_path = os.path.join(state, f"{name}_answers.jsonl")
        att_path = os.path.join(state, f"{name}_attempts.jsonl")
        answered = {a["job_id"] for a in read_jsonl(ans_path)}
        tries: dict[str, int] = {}
        for a in read_jsonl(att_path):
            if not a.get("limit") and not a.get("caller_error"):
                tries[a["job_id"]] = tries.get(a["job_id"], 0) + 1
        pending = [r for r in jobs if r["job_id"] not in answered]
        while pending:
            batch = pending[:BATCH]
            sent = now()
            try:
                results = call_batch([request(r["prompt"]) for r in batch])
                problem = None if isinstance(results, list) and \
                    len(results) == len(batch) else \
                    f"caller returned {type(results).__name__} of length " \
                    f"{len(results) if hasattr(results, '__len__') else 'n/a'} " \
                    f"for {len(batch)} requests"
            except Exception as exc:
                problem = f"caller raised {exc!r}"[:300]
            if problem:
                with open(att_path, "a") as fa:
                    for r in batch:
                        fa.write(json.dumps(dict(job_id=r["job_id"], seq=r["seq"],
                                                 sent_utc=sent, utc=now(),
                                                 caller_error=problem)) + "\n")
                return dict(outcome="caller_error", detail=problem,
                            new_answers=new, status=status(bundle, state))
            stop = None
            with open(att_path, "a") as fa, open(ans_path, "a") as fn:
                for r, res in zip(batch, results):
                    res = res if isinstance(res, dict) else {"error": "non-dict result"}
                    err = (res.get("error") or "")[:300]
                    limit = bool(err) and any(w in err.lower() for w in LIMIT_WORDS)
                    p = valid_patient(res)
                    rec = dict(job_id=r["job_id"], seq=r["seq"], sent_utc=sent,
                               utc=now(), limit=limit, patient=p,
                               served_model=res.get("model"),
                               stop_reason=res.get("stop_reason"),
                               tool_use=res.get("tool_use"), usage=res.get("usage"),
                               error=err or None)
                    fa.write(json.dumps(rec) + "\n")
                    if limit:
                        stop = stop or "limit"
                        continue
                    tries[r["job_id"]] = tries.get(r["job_id"], 0) + 1
                    if res.get("model") not in (None, MODEL):
                        stop = "served_model_mismatch"
                        continue
                    if p is not None or tries[r["job_id"]] >= MAX_ATTEMPTS:
                        fn.write(json.dumps(dict(job_id=r["job_id"], seq=r["seq"],
                                                 patient=p,
                                                 served_model=res.get("model"),
                                                 stop_reason=res.get("stop_reason"),
                                                 utc=rec["utc"])) + "\n")
                        answered.add(r["job_id"])
                        new += 1
            pending = [r for r in pending if r["job_id"] not in answered]
            if stop:
                return dict(outcome=stop, new_answers=new, status=status(bundle, state))
            if max_new is not None and new >= max_new:
                return dict(outcome="max_new", new_answers=new,
                            status=status(bundle, state))
    st = status(bundle, state)
    failed = sum(v["answered"] - v["valid"] for v in st.values())
    return dict(outcome="complete_with_failures" if failed else "complete",
                failed_jobs=failed, new_answers=new, status=st)


def api_call_batch(reqs: list[dict]) -> list[dict]:
    """Direct Claude Messages API, one request at a time."""
    key = os.environ["ANTHROPIC_API_KEY"]
    out = []
    for body in reqs:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                msg = json.loads(resp.read())
            tool = next((b for b in msg.get("content", [])
                         if b.get("type") == "tool_use"), None)
            out.append(dict(model=msg.get("model"), stop_reason=msg.get("stop_reason"),
                            usage=msg.get("usage"),
                            tool_use=dict(name=tool["name"], input=tool["input"])
                            if tool else None))
        except urllib.error.HTTPError as exc:
            out.append(dict(error=f"HTTP {exc.code}: {exc.read()[:200]!r}"))
            if exc.code in (429, 529):
                time.sleep(20)
        except Exception as exc:
            out.append(dict(error=str(exc)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=".")
    ap.add_argument("--state", default="state")
    ap.add_argument("--api", action="store_true", help="call the Claude API directly")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(json.dumps(status(args.bundle, args.state), indent=1))
        return
    if not args.api:
        ap.error("use --api, or import run_jobs and pass a call_batch function")
    while True:
        r = run(api_call_batch, args.bundle, args.state)
        print(json.dumps(r, indent=1))
        if r["outcome"] != "limit":
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
