"""Model client.

OpenAI-compatible chat-completions, which covers DeepSeek, OpenAI, Together,
Groq, vLLM and most local servers. Set the base URL and key via environment
variables; nothing is hardcoded.

    export LLM_BASE_URL=https://api.deepseek.com/v1
    export LLM_API_KEY=...              # your key, read from env only
    export LLM_MODEL=deepseek-chat

Tool-calling is used when the provider supports it (guarantees a parseable
answer). Providers that reject the `tools` field fall back to a
one-character text answer; set LLM_NO_TOOLS=1 to skip the tool attempt.
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import prompts


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    use_tools: bool = True
    max_retries: int = 5
    timeout: float = 120.0
    temperature: float = 0.0
    # reasoning models spend hidden tokens from the completion budget
    reasoning_budget: int = 2048

    @classmethod
    def from_env(cls) -> "ModelConfig":
        base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        key = os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("LLM_MODEL", "")
        missing = [n for n, v in
                   (("LLM_BASE_URL", base), ("LLM_API_KEY", key), ("LLM_MODEL", model))
                   if not v]
        if missing:
            raise SystemExit(
                "Missing environment variable(s): " + ", ".join(missing) +
                "\nExample:\n"
                "  export LLM_BASE_URL=https://api.deepseek.com/v1\n"
                "  export LLM_API_KEY=sk-...\n"
                "  export LLM_MODEL=deepseek-chat"
            )
        return cls(base_url=base, api_key=key, model=model,
                   reasoning_budget=int(os.environ.get("LLM_REASONING_BUDGET", "2048")),
                   use_tools=os.environ.get("LLM_NO_TOOLS", "") != "1")


class Client:
    """Minimal OpenAI-compatible client (stdlib only, no SDK dependency)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._tools_rejected = not cfg.use_tools
        self._reasoning = os.environ.get("LLM_REASONING", "") == "1"
        self._temp_rejected = False

    _EXTRA = None

    def _extra(self) -> dict:
        """Provider-specific payload keys from LLM_EXTRA_JSON, e.g.
        '{"enable_thinking": false}' to disable a default thinking mode."""
        if Client._EXTRA is None:
            raw = os.environ.get("LLM_EXTRA_JSON", "")
            Client._EXTRA = json.loads(raw) if raw else {}
        return Client._EXTRA

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.cfg.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
            return json.loads(r.read().decode())

    def _budget(self) -> dict:
        """Completion-token budget, adapted to reasoning vs. standard models.

        Reasoning models spend hidden reasoning tokens out of the SAME budget as
        the visible answer, so a 64-token cap returns an empty message and every
        call is recorded as missing. They also reject `max_tokens` in favour of
        `max_completion_tokens`, and reject any temperature other than the
        default. Both quirks are detected from the provider's 400 body and
        latched, so at most a couple of calls are spent discovering them.
        """
        cap = 64 if not self._tools_rejected else 8
        if self._reasoning:
            # room for hidden reasoning plus the answer
            return {"max_completion_tokens": max(cap, self.cfg.reasoning_budget)}
        out = {"max_tokens": cap}
        if not self._temp_rejected:
            out["temperature"] = self.cfg.temperature
        return out

    def choose(self, prompt: str) -> int | None:
        """Return 1, 2, or None if the response could not be parsed.

        None is recorded as a missing observation, never retried into a
        substitute value -- silently replacing unparseable calls would bias the
        estimate toward whatever the model does on easy items.
        """
        for attempt in range(self.cfg.max_retries):
            try:
                if not self._tools_rejected:
                    payload = {
                        "model": self.cfg.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "tools": [prompts.CHOOSE_TOOL],
                        "tool_choice": {"type": "function",
                                        "function": {"name": "choose"}},
                        **self._budget(),
                    }
                else:
                    payload = {
                        "model": self.cfg.model,
                        "messages": [{"role": "user",
                                      "content": prompt + prompts.FALLBACK_INSTRUCTION}],
                        **self._budget(),
                    }
                payload.update(self._extra())
                data = self._post(payload)
                got = self._parse(data)
                if got is None and not self._reasoning and self._truncated(data):
                    # ran out of budget mid-answer: almost always a reasoning
                    # model that accepted max_tokens but spent it thinking
                    self._reasoning = True
                    continue
                return got

            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:400]
                # Provider is in a thinking mode that rejects forced tool_choice
                # (e.g. Qwen). Do NOT fall back to the text channel: cells
                # already recorded for this model used the tool channel, and a
                # curve must not mix elicitation channels. Fail with the fix.
                if e.code == 400 and re.search(r"thinking", body, re.I):
                    raise RuntimeError(
                        "provider rejects tool_choice in thinking mode. "
                        "Disable thinking instead of falling back, e.g.\n"
                        "  export LLM_EXTRA_JSON='{\"enable_thinking\": false}'\n"
                        f"(HTTP 400: {body})") from None
                # NOTE on the three latches below: no `not self._flag` guard.
                # With concurrent workers the first 400 flips the flag while
                # sibling calls are in flight; their 400s arrive with the flag
                # already set and must retry (continue), not crash. The retry
                # loop is bounded by max_retries either way.
                # reasoning model: wants max_completion_tokens, not max_tokens
                if e.code == 400 and \
                        re.search(r"max_completion_tokens|max_tokens.*not supported"
                                  r"|unsupported_parameter.*max_tokens", body, re.I):
                    self._reasoning = True
                    continue
                # reasoning model: only the default temperature is allowed
                if e.code == 400 and \
                        re.search(r"temperature", body, re.I):
                    self._temp_rejected = True
                    continue
                # provider does not support tool calling -> permanent fallback
                if e.code == 400 and \
                        re.search(r"tool|function", body, re.I):
                    self._tools_rejected = True
                    continue
                if e.code in (408, 409, 425, 429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(min(2 ** attempt, 30))
        return None

    @staticmethod
    def _truncated(data: dict) -> bool:
        try:
            return data["choices"][0].get("finish_reason") == "length"
        except (KeyError, IndexError):
            return False

    def channel(self) -> str:
        """One-line description of how calls were actually made. Record this."""
        return (f"tools={'no' if self._tools_rejected else 'yes'} "
                f"reasoning_budget={'yes' if self._reasoning else 'no'} "
                f"temperature={'default' if self._temp_rejected or self._reasoning else self.cfg.temperature}")

    @staticmethod
    def _parse(data: dict) -> int | None:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            return None
        for call in (msg.get("tool_calls") or []):
            try:
                args = json.loads(call["function"]["arguments"])
                p = int(args["patient"])
                if p in (1, 2):
                    return p
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                pass
        text = (msg.get("content") or "").strip()
        m = re.search(r"\b([12])\b", text)
        return int(m.group(1)) if m else None
