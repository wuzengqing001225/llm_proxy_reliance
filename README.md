# Proxy reliance in large language model decisions is uncalibrated to predictive evidence

Code and data for the paper "Proxy reliance in large language model decisions is uncalibrated to predictive evidence".

## Layout

```
code/                 experiment package (run new models)
  config.py           frozen seeds and conditions -- do not edit for replication
  modules/            dgp_k.py (generating process), dgp_norm.py (variance-normalized
                      proxy-strength sweep), oracle.py (Bayes + BLR Ideal learner),
                      estimators.py
  run_cell.py         run one cell (3 arms x 99 pairs x 2 orders = 594 calls)
  analyze.py          recompute capability / PSE / justified benchmark from records
  dose_response.py    fit PSE = alpha + gamma * IdealLine, pair-resampled CIs
  verify.py           offline design checks (no API key needed)
  analysis/           real_data_anchors.py (clinical-data anchoring), coupling_analysis.py
data/                 recomputed cell summaries, dose-response curves, anchors
```

## Reproduce

```bash
export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
cd code && ./run_all.sh core        # then: interaction, beta
```

Regenerate the real-data anchors (downloads ~20 MB from UCI, writes the
anchor tables and `real_anchors/CHECKSUMS.sha256`):

```bash
cd code/analysis && python real_data_anchors.py
```

`real_data_anchors.py` downloads seven public UCI clinical datasets
at run time. The datasets are not redistributed here (they remain under their
UCI licences); the script records SHA-256 checksums of the downloaded files in
`real_anchors/CHECKSUMS.sha256` so the exact inputs can be verified, and the
derived anchor statistics are included in `data/summaries/`.
