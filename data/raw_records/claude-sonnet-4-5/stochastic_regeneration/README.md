Per-call records for the protocol checks added. Two cells,
both k = 12, corner LH, neutral labels, rule learned: the zero-information
cell and the beta = 0.60 dose cell.

Files:

- `*_frozenbaseline_records.csv` — the frozen-protocol counterfactual arms
  (`cf_a1`, `cf_a0`), rerun in the same session as the check on the same
  serving model, used as the same-model baseline. Schema matches the main
  records: `arm,pair_id,i,j,order,patient,chosen,picked_i`.
- `*_stochregen_records.csv` — stochastic-regeneration arms produced by
  `code/stochastic_regeneration.py`. For each pair, five independent
  redraws of the target patient's proxy-channel noise, each shared by the
  two counterfactual arms. The `draw` column indexes the redraw.

Model: `claude-sonnet-4-5-20250929`, the dated snapshot recorded in the
frozen records. Temperature 0, forced single-token tool choice, both
presentation orders. Unparseable responses are recorded with empty
`patient,chosen,picked_i` fields and never substituted. One pair (the last
in job order) is missing one arm in both cells because a client-side token
budget truncated each run slice at the same deterministic point, so
analyses resolve on 98 of 99 pairs.

Analysis:

    python stochastic_regeneration.py --analyze \
        --records k12_LH_neutral_learned_zero_stochregen_records.csv \
        --frozen  k12_LH_neutral_learned_zero_frozenbaseline_records.csv
