Per-call records for the protocol checks, on the
current serving version of the DeepSeek flash line (API identifier
`deepseek-flash`). The provider has retired the pinned
DeepSeek-V4-Flash-0731 snapshot used for the paper's reported estimates,
so these checks rerun their own frozen-protocol baseline on the current
version and every comparison stays within one model. The paper's estimates
are computed from the archived records under
`data/raw_records/DeepSeek-V4-Flash-0731/` and are unaffected.

Serving configuration: temperature 0 unless noted, forced single-token
tool choice, thinking mode disabled
(`LLM_EXTRA_JSON='{"thinking": {"type": "disabled"}}'`), both
presentation orders. See the `*_metadata.json` files. Two cells, both
k = 12, corner LH, neutral labels, rule learned: the zero-information
cell and the beta = 0.60 dose cell.

Files per cell:

- `*_frozenbaseline_records.csv` — the frozen protocol rerun on this
  version (`run_cell.py`, all three arms).
- `*_stochregen_records.csv` — stochastic-regeneration arms
  (`code/stochastic_regeneration.py`), five redraws of the target's
  proxy-channel noise per pair, `draw` column indexes the redraw.
- `*_t1resample_records.csv` — temperature-1 resampling
  (`code/temperature_resampling.py`), twenty samples per prompt at
  temperature 1, `sample_id` column indexes the sample.

Analysis: the two scripts' `--analyze` mode, with `--frozen` pointing at
the `*_frozenbaseline_records.csv` of the same cell.
