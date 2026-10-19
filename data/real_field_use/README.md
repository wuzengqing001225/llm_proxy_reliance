# Real-record field-use positive control

The group-conditional substitution in the main real-record analysis rarely
changes payer, so its near-zero estimate cannot show whether the model uses the
field at all. This control fixes the target's admission source at Referral and
changes only its payer:

- high: `Medicaid | Referral`
- low: `Blue Cross/Blue Shield | Referral`

The outcome rule adds a weight of 1.0 to the standardized Medicaid/self-pay
indicator. On the 1,200 primary pairs the full-information reference is
+10.42 pp and the ideal-learner reference is +11.25 pp.

## Result

| Quantity | Estimate (pp) | 95% interval |
|---|---|---|
| Sonnet effect | +40.33 | [+37.96, +42.75] |
| Presentation order 0 | +41.33 | |
| Presentation order 1 | +39.33 | |

The model uses the field when it carries information and uses it more than the
reference supports. The 300-pair pilot was reused in the full run (1,200 of
4,800 calls reused).

## Files

- `claude-sonnet-4-5-20250929/real_field_use_p1200_w1_records.csv`
- `claude-sonnet-4-5-20250929/real_field_use_p1200_w1_metadata.json`
- `pilot300/` holds the 300-pair pilot records and metadata referenced by the
  full run's `reused_records` field.

## Reproduce

The runner is `code/run_real_field_use.py`. From the repository root:

```
python3 code/run_real_field_use.py \
  --panel code/analysis/real_anchors/uci_diabetes_130hosp.csv \
  --ids-mapping code/analysis/real_anchors/IDS_mapping.csv \
  --pairs 1200 --payer-weight 1.0
```

Omit `--execute` for the offline prompt/reference check. The panel and mapping
are downloaded UCI files and are not redistributed.
