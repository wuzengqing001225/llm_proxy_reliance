#!/usr/bin/env python3
"""Send one display-control prompt to OpenRouter and print the raw result.

Use this when the smoke run records no patient. It shows finish_reason,
usage, content, tool_calls and whether a reasoning field came back.

  export OPENROUTER_API_KEY=sk-or-v1-...
  python3 code/openrouter_probe.py --model deepseek/deepseek-v4-flash-0731
  python3 code/openrouter_probe.py --model qwen/qwen3.7-max --text
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file",
                    default=os.path.join(HERE, "..", "data", "encoding_control",
                                         "prompts", "learned_ordinal.jsonl"))
    ap.add_argument("--reasoning", action="store_true",
                    help="leave reasoning on (default: disable)")
    ap.add_argument("--text", action="store_true",
                    help="use the one-character fallback channel, no tools")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("set OPENROUTER_API_KEY")
    prompt = json.loads(open(a.prompt_file, encoding="utf-8").readline())["prompt"]

    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
    }
    if not a.reasoning:
        payload["reasoning"] = {"enabled": False}
    if a.text:
        payload["messages"][0]["content"] = prompt + prompts.FALLBACK_INSTRUCTION
        payload["max_tokens"] = 8
    else:
        payload["tools"] = [prompts.CHOOSE_TOOL]
        payload["tool_choice"] = {"type": "function", "function": {"name": "choose"}}

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "X-Title": "proxy-audit-probe"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code)
        print(e.read().decode(errors="replace")[:2000])
        return
    if a.raw:
        print(json.dumps(raw, indent=1)[:6000])
        return
    ch = (raw.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    print("served:", raw.get("model"))
    print("finish_reason:", ch.get("finish_reason"))
    print("usage:", raw.get("usage"))
    print("content:", repr(msg.get("content")))
    print("tool_calls:", json.dumps(msg.get("tool_calls"), indent=1)[:1200])
    print("reasoning present:", "reasoning" in msg)
    print("message keys:", sorted(msg))

if __name__ == "__main__":
    main()
