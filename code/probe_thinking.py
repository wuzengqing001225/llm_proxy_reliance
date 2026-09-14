#!/usr/bin/env python3
"""Find the payload key that disables the provider's thinking mode.

Sends one real forced-tool-choice request per candidate and reports which
one the provider accepts. Uses the same env vars as run_cell.py
(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL). About 5 calls total.

    python probe_thinking.py
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import prompts  # noqa: E402

BASE = os.environ.get("LLM_BASE_URL", "").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "")
if not (BASE and KEY and MODEL):
    raise SystemExit("set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL first")

CANDIDATES = [
    ("enable_thinking false", {"enable_thinking": False}),
    ("thinking type=disabled", {"thinking": {"type": "disabled"}}),
    ("reasoning enabled=false", {"reasoning": {"enabled": False}}),
    ("reasoning_effort none", {"reasoning_effort": "none"}),
    ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
]


def post(payload):
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


base_payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content":
                  "Patient 1 or patient 2? Answer with the tool."}],
    "tools": [prompts.CHOOSE_TOOL],
    "tool_choice": {"type": "function", "function": {"name": "choose"}},
    "max_tokens": 64,
    "temperature": 0.0,
}

print(f"model {MODEL} at {BASE}\n")
for name, extra in CANDIDATES:
    payload = {**base_payload, **extra}
    try:
        data = post(payload)
        msg = data["choices"][0]["message"]
        has_tool = bool(msg.get("tool_calls"))
        print(f"  ACCEPTED  {name:24s} tool_call returned: {has_tool}")
        if has_tool:
            print(f"\nuse:  export LLM_EXTRA_JSON='{json.dumps(extra)}'")
            break
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:160]
        print(f"  rejected  {name:24s} HTTP {e.code}: {body}")
    except Exception as e:  # noqa: BLE001
        print(f"  error     {name:24s} {e}")
else:
    print("\nNo candidate worked. Check the provider's API docs for the "
          "thinking switch, or list models:")
    print(f"  curl -s {BASE}/models -H 'Authorization: Bearer $LLM_API_KEY'")
