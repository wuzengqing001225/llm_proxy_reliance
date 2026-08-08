"""Codex CLI as a *subject* endpoint (not as a driver).

Use this only when your access to a GPT model is through Codex CLI rather than
a plain API key. It exposes the same `.choose(prompt) -> 1 | 2 | None`
interface as `client.Client`, so `run_cell.py --via codex` is a drop-in swap.

------------------------------------------------------------------------------
READ THIS BEFORE INTERPRETING ANY RESULT FROM THIS PATH
------------------------------------------------------------------------------
`codex exec` is an agent, not a completions endpoint. Every call it makes is
wrapped in Codex's own system prompt and tool definitions, and the model runs a
short agent loop before emitting a final message. The DeepSeek arm, by
contrast, was a bare single-turn chat completion.

So a GPT-via-Codex arm differs from the DeepSeek arm in **two** ways at once —
the model changed, and the framing changed. A raw difference between them is
therefore not a model effect and must not be reported as one.

Two ways to deal with this, in order of preference:

  1. Get an OpenAI API key and use `client.py` instead. Then both arms are
     bare single-turn calls and the comparison is clean. This is the right
     answer if it is available to you at all.

  2. If Codex is your only GPT access, run the *wrapper calibration* in
     `calibrate_wrapper.py`: replay one already-completed DeepSeek cell through
     `codex exec` with Codex configured to call DeepSeek. Model held fixed,
     framing varied — that isolates the wrapper effect, and the GPT arm can
     then be read against a known offset rather than an unknown one.

Never skip step 2 and report the raw GPT-vs-DeepSeek difference.
------------------------------------------------------------------------------

Setup:

    export CODEX_SUBJECT_MODEL=gpt-5.1        # whichever model you have access to
    python run_cell.py --via codex --k 12 --corner LH --semantic neutral --rule absent

Throughput is much lower than the API path — each call spawns a process and
runs an agent turn. Budget accordingly and keep `--workers` small (2-4).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"patient": {"type": "integer", "enum": [1, 2]}},
    "required": ["patient"],
    "additionalProperties": False,
}

# Appended to every prompt. Kept minimal and identical across arms; its only job
# is to stop the agent from exploring the filesystem or narrating.
EXEC_SUFFIX = (
    "\n\nAnswer from the information above only. Do not run any commands, read "
    "any files, or explain your reasoning. Reply with exactly one character: "
    "1 or 2."
)


@dataclass
class CodexConfig:
    model: str
    binary: str = "codex"
    timeout: float = 180.0
    max_retries: int = 3
    use_schema: bool = True

    @classmethod
    def from_env(cls) -> "CodexConfig":
        model = os.environ.get("CODEX_SUBJECT_MODEL", "")
        if not model:
            raise SystemExit(
                "Set CODEX_SUBJECT_MODEL to the model you want measured, e.g.\n"
                "  export CODEX_SUBJECT_MODEL=gpt-5.1"
            )
        binary = os.environ.get("CODEX_BIN", "codex")
        if shutil.which(binary) is None:
            raise SystemExit(f"{binary!r} not found on PATH. Install Codex CLI first.")
        return cls(model=model, binary=binary,
                   use_schema=os.environ.get("CODEX_NO_SCHEMA", "") != "1")


class CodexClient:
    """Same interface as client.Client, backed by `codex exec`."""

    def __init__(self, cfg: CodexConfig):
        self.cfg = cfg
        self._schema_path: str | None = None
        self._schema_rejected = not cfg.use_schema
        # Run in an empty scratch dir so no AGENTS.md or repo content is picked
        # up as context. Every call must see exactly the same environment.
        self._cwd = tempfile.mkdtemp(prefix="codex_subject_")
        if cfg.use_schema:
            fd, p = tempfile.mkstemp(suffix=".json", prefix="answer_schema_")
            with os.fdopen(fd, "w") as f:
                json.dump(ANSWER_SCHEMA, f)
            self._schema_path = p

    def _argv(self, with_schema: bool) -> list[str]:
        argv = [
            self.cfg.binary, "exec",
            "--model", self.cfg.model,
            "--sandbox", "read-only",   # exec's default; stated so it cannot drift
            "--ephemeral",              # do not persist session rollout files
            "--skip-git-repo-check",
            "-",                        # read the whole prompt from stdin
        ]
        if with_schema and self._schema_path:
            argv[-1:] = ["--output-schema", self._schema_path, "-"]
        return argv

    def choose(self, prompt: str) -> int | None:
        """Return 1, 2, or None if unparseable. None is recorded as missing."""
        body = prompt + EXEC_SUFFIX
        for attempt in range(self.cfg.max_retries):
            with_schema = not self._schema_rejected
            try:
                proc = subprocess.run(
                    self._argv(with_schema),
                    input=body, capture_output=True, text=True,
                    timeout=self.cfg.timeout, cwd=self._cwd,
                )
            except subprocess.TimeoutExpired:
                continue
            if proc.returncode != 0:
                err = (proc.stderr or "")[-400:]
                # older builds may not know --output-schema
                if with_schema and re.search(r"output-schema|unexpected argument", err, re.I):
                    self._schema_rejected = True
                    continue
                if attempt == self.cfg.max_retries - 1:
                    return None
                continue
            got = self._parse(proc.stdout)
            if got is not None:
                return got
        return None

    @staticmethod
    def _parse(stdout: str) -> int | None:
        text = (stdout or "").strip()
        if not text:
            return None
        # structured output arrives as a JSON object on the final line
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    p = int(json.loads(line)["patient"])
                    if p in (1, 2):
                        return p
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    pass
                break
        # otherwise take the last standalone 1 or 2 the agent emitted
        hits = re.findall(r"\b([12])\b", text)
        return int(hits[-1]) if hits else None
