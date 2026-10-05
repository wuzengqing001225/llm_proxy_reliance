# Auditing proxy reliance in language model decisions against predictive evidence

Code and data for the paper. The release covers the synthetic patient-ranking
tasks with known risk, the matched proxy representation control, conditional
proxy substitution on real patient covariates from the UCI Diabetes 130-US
Hospitals dataset, and two concurrent protocol checks.

## Reproduce the results without model calls

Install Python 3.12 and `code/requirements_external.txt`. From `code/`:

```bash
python verify.py
python analysis/fetch_diabetes_data.py
python reproduce_new_evidence.py
python pooled_tracking.py --summary ../data/summaries/beta_four_model_summary.csv \
    --sonnet ../data/canonical/claude-sonnet-4-5 \
    --deepseek ../data/raw_records/DeepSeek-V4-Flash-0731 \
    --qwen ../data/raw_records/qwen3.7-max --gpt ../data/raw_records/gpt-5.6-terra
python gpt_tracking_repeats.py
python concurrent_checks.py analyze-regen --links ../data/concurrent_checks/links/regen_links.csv \
    --answers ../data/concurrent_checks/claude-sonnet-4-5-20250929/regen_jobs_answers.jsonl
python concurrent_checks.py analyze-neartie --panel analysis/real_anchors/uci_diabetes_130hosp.csv \
    --links ../data/concurrent_checks/links/neartie_links.csv \
    --answers ../data/concurrent_checks/claude-sonnet-4-5-20250929/neartie_jobs_answers.jsonl
```

Then run `python figures/code/make_paper_figures.py` from the repository root.
The UCI retrieval scripts download the official files and check their SHA-256
hashes. The dataset is https://archive.ics.uci.edu/dataset/296, DOI
10.24432/C5230J, distributed by UCI under CC BY 4.0. Downloaded files stay
local and are not redistributed.

## Running new model calls

Synthetic cells use `code/run_cell.py` (one cell) or `code/run_paper_set.sh`
(the cross-model set) with an OpenAI-compatible endpoint:

```bash
export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
OUTDIR=runs_new ./run_paper_set.sh main
```

Sonnet runs use the native Claude API. Store the key in `ANTHROPIC_API_KEY`
and set `LLM_PROVIDER=anthropic` and `LLM_MODEL=claude-sonnet-4-5-20250929`.
Requests use `/messages`, the forced `choose` tool, temperature zero and no
thinking field.

```bash
cd code
python run_claude_jobs.py --jobs ../data/encoding_control/prompts/learned_ordinal.jsonl --name learned_ordinal
```

Without `--execute` the command only validates prompts. Add
`--execute --outdir results/new_encoding` to make API calls. `--resume`
requires the same job-file, model and system-prompt hashes, and
`--invalid-attempts 5` applies the representation control's retry rule.
Real-record runs use `run_external_zero_info.py` with `--panel`,
`--ids-mapping`, `--pairs`, `--ordering` and `--run-set`. Without `--execute`
it performs offline checks only.

Request settings for the stored data are in `docs/PROVENANCE.md`.

## Repository layout

- `code/`: generating process (`modules/`), runners, analyses and tests.
- `data/`: per-call records, condition summaries and analysis outputs
  (`docs/DATA.md`).
- `figures/`: figure code and generated figures.

Per-pair confidence intervals condition on the recorded choices and do not
include variation over future model calls.

## License

Code is released under the MIT License (`LICENSE`). The UCI datasets remain
under their own licenses.
