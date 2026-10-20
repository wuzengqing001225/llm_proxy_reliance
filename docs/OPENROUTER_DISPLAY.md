# Running the display control on DeepSeek and Qwen through OpenRouter

This runs the same readable-order control used for Sonnet 4.5 and GPT-5.6
Terra: one frozen zero-information cell (k = 12, LH, neutral names, learned
rule), 99 pairs, eighty labeled examples, and two renderings of the same proxy
bins — **ordered labels** (`level 1`–`level 4`) and **arbitrary two-letter
codes**. Everything else is held fixed.

Each rendering has **396 distinct prompts**. The control therefore costs
**792 calls per model** (396 ordered + 396 arbitrary). The paired difference
`ordered − arbitrary` is the display effect.

## Models and OpenRouter prices

Checked against `https://openrouter.ai/api/v1/models` on 2026-10-21. Prices
are USD per token; multiply by 1e6 for the per-million rate.

| Paper name | OpenRouter model id | Input / M | Output / M |
|---|---|---|---:|---:|
| DeepSeek-V4-Flash-0731 | `deepseek/deepseek-v4-flash-0731` | $0.1298 | $0.3894 |
| Qwen3.7-max | `qwen/qwen3.7-max` | $1.475 | $4.425 |

The prompts are about **4,650 input tokens** each (tiktoken estimate; the
provider's count can differ by a few percent), and a forced `choose` tool call
returns about 16–32 output tokens.

## Cost estimate

| Model | Calls | Input cost | Output cost | Total |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash-0731 | 792 | ~$0.48 | ~$0.01 | **~$0.49** |
| Qwen3.7-max | 792 | ~$5.43 | ~$0.11 | **~$5.54** |
| **Both** | 1,584 | ~$5.91 | ~$0.12 | **~$6.03** |

Range across 4,300–5,000 input tokens and 16–64 output tokens: **$5.5–$6.6**
for both models. Qwen dominates the bill. If you repeat each model for
sampling stability, double it (~$12); DeepSeek and Qwen at temperature zero
should not need a repeat.

The one-call probes give a lower, cache-aware estimate: DeepSeek
$0.00013/call, Qwen $0.0035/call with a cached 80-example prefix. At 792 calls
that is **~$0.10 for DeepSeek and ~$2.8 for Qwen (~$2.9 total)**. If the cache
does not hold, the no-cache estimate is the ~$6 upper bound. Check
`usage.cost` in the probe output for the current rate.

## Setup

```bash
cd /Users/wuzengqing/Desktop/第一工作台/科学研究/SABM/社会判断属性/experiments/github_full
export OPENROUTER_API_KEY=sk-or-v1-...
```

Get the key at <https://openrouter.ai/keys>. The runner uses OpenRouter's
OpenAI-compatible endpoint:

```
LLM_BASE_URL=https://openrouter.ai/api/v1
```

The shell script sets the base URL, key and model; you only provide the key.
It also disables reasoning by default (`reasoning.enabled=false`): both models
default to thinking on OpenRouter, and a small completion budget can otherwise
be spent on hidden reasoning before the `choose` call is emitted.

## Smoke test first (two calls)

Do not spend the full amount before checking that the model id, tool calling
and parser work.

```bash
SMOKE=1 ./code/run_openrouter_display.sh \
  deepseek/deepseek-v4-flash-0731 \
  data/encoding_control/openrouter_deepseek \
  data/raw_records/DeepSeek-V4-Flash-0731/k12_LH_neutral_learned_zero_records.csv
```

Inspect:

```bash
ls data/encoding_control/openrouter_deepseek/deepseek_deepseek-v4-flash-0731/
head -3 data/encoding_control/openrouter_deepseek/deepseek_deepseek-v4-flash-0731/*_ordinal_records.csv
cat data/encoding_control/openrouter_deepseek/deepseek_deepseek-v4-flash-0731/*_ordinal_metadata.json
```

You want `patient`/`picked_i` non-empty for the two calls, the served model
id in the metadata, and `channel` to say `tools=yes reasoning_budget=no
temperature=0.0` (or `tools=no` if the provider rejected tools; the runner
falls back to the one-character channel).

`missing: 396` in the smoke metadata is expected: `--limit 2` sends only two
of the 396 distinct prompts, and the other 394 are written as empty rows.
Look at whether the two attempted rows have a patient, not at the total.

## Qwen tool calling is less reliable than DeepSeek's

The DeepSeek probe returns a `choose` tool call. On Qwen, the first probe
prompt returned a long written analysis with `finish_reason: length` and no
tool call, although the two smoke prompts happened to answer. Tool choice is
therefore prompt- and routing-dependent for Qwen on OpenRouter.

Before committing to the full Qwen run, send a 30-prompt pilot:

```bash
SMOKE_LIMIT=30 SMOKE=1 RESUME=1 ./code/run_openrouter_display.sh   qwen/qwen3.7-max   data/encoding_control/openrouter_qwen   data/raw_records/qwen3.7-max/k12_LH_neutral_learned_zero_records.csv
```

Count answered rows (`$6 != ""`). If any are missing, run Qwen in the
one-character text channel instead, which the probe confirms works:

```bash
LLM_NO_TOOLS=1 RESUME=1 ./code/run_openrouter_display.sh   qwen/qwen3.7-max   data/encoding_control/openrouter_qwen_text   data/raw_records/qwen3.7-max/k12_LH_neutral_learned_zero_records.csv
```

The text channel changes the elicitation relative to Qwen's battery
(`tools=yes`), so the `numeric (battery)` comparison is channel-confounded.
The ordinal-minus-arbitrary paired difference is still within the text channel
and is the number to use; state the channel in the SI.

## Full run

DeepSeek:

```bash
RESUME=1 ./code/run_openrouter_display.sh \
  deepseek/deepseek-v4-flash-0731 \
  data/encoding_control/openrouter_deepseek \
  data/raw_records/DeepSeek-V4-Flash-0731/k12_LH_neutral_learned_zero_records.csv
```

Qwen:

```bash
RESUME=1 ./code/run_openrouter_display.sh \
  qwen/qwen3.7-max \
  data/encoding_control/openrouter_qwen \
  data/raw_records/qwen3.7-max/k12_LH_neutral_learned_zero_records.csv
```

`RESUME=1` continues after the smoke test; for a fresh directory omit it.
`WORKERS=6` is the default; lower it if OpenRouter rate-limits.

## What the analysis prints

For each model:

```
ordinal: effect +X.XX pp [lo, hi], pairs 99, missing calls 0
nominal: effect +Y.YY pp [lo, hi], pairs 99, missing calls 0
paired difference ordinal - nominal: +Z.ZZ pp [lo, hi] on 99 pairs
```

The number to compare with the paper is the **paired difference**:

- Sonnet 4.5: +10.10 pp [3.03, 17.68]
- GPT-5.6 Terra: +5.30 pp [0.51, 10.10] (two-run pooled)

The `numeric (battery)` line uses that model's own battery records (passed as
`--numeric-reference`), so it is the right model-specific numeric baseline,
not Sonnet's.

## Troubleshooting

**HTTP 400 mentions “thinking” / forced tool choice.** The model is in a
reasoning mode that rejects `tool_choice`. Disable it through
`LLM_EXTRA_JSON`, for example:

```bash
# Qwen-style
export LLM_EXTRA_JSON='{"enable_thinking": false}'
# OpenRouter unified reasoning switch
export LLM_EXTRA_JSON='{"reasoning": {"enabled": false}}'
# DeepSeek V4.1-style
export LLM_EXTRA_JSON='{"thinking": {"type": "disabled"}}'
```

Then rerun with `RESUME=1`. Do not mix a tool-channel run with a text-channel
run in the same records directory; if the provider permanently rejects tools,
start a fresh directory and record `channel=tools=no`.

**HTTP 400 mentions “tool” or “function”.** The runner flips to the
one-character text channel after a retry. This is usable, but it is a different
elicitation channel from Sonnet/GPT. Prefer a provider that supports tools, or
report the channel.

**HTTP 429/5xx.** The client retries with backoff. Reduce `WORKERS` to 3–4.

**Smoke run records no patient.** First check `channel` and `extra_json` in
the metadata. If `reasoning_budget=yes` and `temperature=default`, the client
latched into a reasoning model and the 2,048-token budget was likely spent
before the answer. The script now sets `reasoning.enabled=false`; rerun into a
fresh directory (or with `RESUME=1`) and check the two attempted rows.

If it is still empty, send one prompt with the probe and keep its output:

```bash
python3 code/openrouter_probe.py --model deepseek/deepseek-v4-flash-0731
python3 code/openrouter_probe.py --model qwen/qwen3.7-max
python3 code/openrouter_probe.py --model qwen/qwen3.7-max --text
```

The probe prints `finish_reason`, `usage`, `content`, `tool_calls` and whether
a reasoning field came back. A `finish_reason: length` with a large
`completion_tokens` means reasoning is still on; an empty `tool_calls` with
`content` text means the model chose the text channel.

**Missing calls.** The records keep `patient` empty and the metadata reports
`missing`; rerun with `RESUME=1`. The client never substitutes a value for an
unparseable call.

## Files written

```
<outdir>/<model_with_underscores>/
  catctl_k12_LH_neutral_learned_zero_ordinal_records.csv
  catctl_k12_LH_neutral_learned_zero_ordinal_metadata.json
  catctl_k12_LH_neutral_learned_zero_nominal_records.csv
  catctl_k12_LH_neutral_learned_zero_nominal_metadata.json
```

The directory name is the model id with `/` replaced by `_`, e.g.
`deepseek_deepseek-v4-flash-0731`.
