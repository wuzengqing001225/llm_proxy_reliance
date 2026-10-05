# Acquisition settings

## Models and request settings

| Model | Identifier | Sampling | Notes |
|---|---|---|---|
| Claude Sonnet 4.5 | `claude-sonnet-4-5` (battery), `claude-sonnet-4-5-20250929` (representation, real-record and concurrent checks) | temperature 0 | 64-token output cap, forced `choose` tool, no thinking field |
| DeepSeek-V4-Flash-0731 | `DeepSeek-V4-Flash-0731` (API model `deepseek-chat`) | temperature 0 | forced tool choice |
| Qwen3.7-max | `qwen3.7-max` | temperature 0 | forced tool choice, `{"enable_thinking": false}` |
| GPT-5.6 Terra | `gpt-5.6-terra` | model default | forced tool choice; see below |
| DeepSeek-V4.1-Flash | `deepseek-flash` | temperature 0, or 1 for resampling | thinking disabled; protocol checks only |

GPT-5.6 Terra runs at its default reasoning effort (medium) and does not accept
a temperature setting at that effort, so both GPT datasets (the battery and the
repeated measurement) use the model's default sampling. The `channel` string in
the GPT metadata records the client request settings. GPT results are reported
separately where models are pooled.

DeepSeek-V4.1-Flash is a later release than DeepSeek-V4-Flash-0731. All
DeepSeek battery estimates use `data/raw_records/DeepSeek-V4-Flash-0731/`.

## Sonnet requests

All stored Sonnet records were collected through the Claude API from a client
environment that adds a fixed system message, whose text is not visible to the
caller. The message is identical across conditions, arms and display orders.
The released runners reproduce the prompts and request settings and send no
system message unless `LLM_SYSTEM_PROMPT_FILE` is set. Collection dates are in
Supplementary Note 15 and, where present, in the metadata files.

## Missing and retried responses

The 80-example representation conditions have no invalid responses. In the
no-example code condition, twelve prompts were resolved under the prespecified
rule of up to five attempts, and 124 accepted answers with a valid patient
value ended with a refusal stop. Primary and refusal-excluded results are both
released (`data/encoding_control/posthoc_refusal_excluded/`).

The real-record experiment has 5,992 valid unique prompt responses. Requests
that returned no model response were repeated, and answered requests were never
repeated.

## New calls

Write any new API calls to a separate output directory. Stored answers and
their recorded settings are not modified.
