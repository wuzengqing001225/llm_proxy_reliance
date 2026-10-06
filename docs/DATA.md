# Data layout

Per-call record files share the schema `arm,pair_id,i,j,order,patient,chosen,picked_i`
(`arm` is `factual`, `cf_a1` or `cf_a0`; `picked_i` is 1 when the target `i`
is chosen). Cell tags follow `k<fields>_<corner>_<names>_<rule>_<info>[_b<beta>][_s<seed>]`.

## Synthetic task

- `raw_records/claude-sonnet-4-5/`: Sonnet per-call records.
  - `gate2/`: the example-supplied grid at six, twelve and eighteen fields,
    stored in shards (`expK_g2_*`, `expK_g2fix_*`, `expK_g2fix2_*`).
    `code/prepare_canonical.py` merges them on `(arm, i, j, order)` with
    precedence `g2fix2 > g2fix > g2`.
  - `expK_g1_*`: six cells of the no-example grid.
  - `beta_sweep/`: dose-response levels, one file per arm.
  - `dimension_by_structure/`, `coupling/`, `dcal_sweep/`, `mechanism/`:
    dependence-structure, coupling, example-count and example-correlation cells.
- `canonical/claude-sonnet-4-5/`: merged Sonnet cells used by all analyses.
- `raw_records/DeepSeek-V4-Flash-0731/`, `raw_records/qwen3.7-max/`,
  `raw_records/gpt-5.6-terra/`: battery cells; `seed_replication/` holds the
  two additional populations (seeds 20001 and 20002).
- `raw_records/gpt-5.6-terra/repeated_measurement/`: two independent
  neutral-label dose-response runs (`run_a`, `run_b`). Analyze with
  `code/gpt_tracking_repeats.py`. Analyzed separately from the battery.
- `raw_records/deepseek-flash/`: DeepSeek-V4.1-Flash protocol checks for the
  zero-information and beta = 0.60 cells: frozen counterfactual
  (`*_frozenbaseline_records.csv`), stochastic regeneration
  (`*_stochregen_records.csv`, column `draw`) and temperature-1 resampling
  (`*_t1resample_records.csv`, column `sample_id`). Analyze with the
  `--analyze` mode of `code/stochastic_regeneration.py` and
  `code/temperature_resampling.py`.

`summaries/learner_zero_info.csv` gives the same-example learner's effect in the
zero-information cells with eighty examples (`code/learner_zero_info.py`).

Conditions released as condition summaries only:

- `summaries/acore/`: rule-provided boundary condition and the four
  dependence corners (`*_rule_absent_*`, `*_rule_provided_*`, `*_mechanism`).
- `summaries/expB/`: audit-population transport and proxy-omission analyses.
- `summaries/gate_summaries/`: Sonnet grid without examples
  (`expK_gate1_summary.csv`) and with eighty examples (`expK_gate2_summary.csv`).
  Per-call records exist for the cells listed above.

## Representation control

`encoding_control/` holds the ordered-level, code and second numeric
measurement records for Sonnet, exact prompt exports (`prompts/`), the battery
numeric records of the same cells (`reference/`) and the refusal-excluded
sensitivity analysis (`posthoc_refusal_excluded/`).

## Real-record experiment

`external_zero_info/` holds call files, logical records and metadata for the
five real-record conditions, and `analysis/` holds their analysis outputs and
the derived first-300 baseline. UCI inputs are not redistributed. Retrieve
them with `code/analysis/fetch_diabetes_data.py`.

## Concurrent checks

`concurrent_checks/` holds the prespecified protocol (`PROTOCOL.md`), job
links, answers and records for the concurrent regeneration comparison and the
near-tie check with its wide-gap control. Analyze with
`code/concurrent_checks.py analyze-regen` and `analyze-neartie`, and summarize
with `code/summarize_concurrent_checks.py`.

## Checksums

`new_evidence_manifest.json` lists SHA-256 hashes of the representation and
real-record source files. `code/reproduce_new_evidence.py` verifies them before
recomputing `summaries/new_evidence.json`.
