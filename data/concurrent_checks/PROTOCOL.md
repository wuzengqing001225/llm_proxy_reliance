# Locked protocol: concurrent regeneration and near-tie checks

Fixed on 5 October 2026, before any answer to these jobs was seen.
Model: `claude-sonnet-4-5-20250929`, temperature 0, max_tokens 64, forced
tool `choose`, one user message, no system prompt, no thinking field.

## A. Concurrent frozen versus stochastic regeneration (priority)

Cells: k = 12, LH, neutral names, eighty labelled examples, zero-information
(`zero`) and beta = 0.60 (`b060`). The same 99 evaluation pairs as the
archived cells (verified offline).

Conditions, rendered now by the same code from the same frozen materials:
- frozen: target proxies regenerated with the same exogenous noise
  (99 pairs x 2 arms x 2 orders = 396 prompts per cell);
- stochastic: five fresh redraws of the target's proxy noise, each shared by
  both arms (1,980 prompts per cell).

All 4,752 prompts are unique and sent in one queue in a random order fixed by
seed 20261005, so frozen and stochastic calls are interleaved in time. Every
call time is recorded. The frozen zero-cell prompts are identical to those of
earlier numeric runs of the same cell (396/396 hashes). The frozen material
was originally generated before this run, and only the model calls are
concurrent.

Estimand: per pair, frozen contrast (cf_a1 minus cf_a0, averaged over orders)
minus stochastic contrast (averaged over orders and the five draws).

Primary analysis, per cell and pooled over the two cells on the same pairs:
mean difference with 95% and 90% pair-bootstrap intervals (5,000 resamples
of pairs, shared across cells for the pooled value).

Missing data, fixed in advance. Only valid answers enter the analysis. Within
a scheme and draw, an arm's value is the mean over its valid display orders,
so a missing order leaves the other order. A draw contributes only if both
arms have at least one valid order. A pair's stochastic contrast requires all
five draws to contribute, and its frozen contrast requires both arms. A pair
failing either requirement is dropped from both schemes in that cell, and
from the pooled value. The number of dropped pairs and of invalid records is
reported.

Reading rules:
- 95% interval containing zero: no resolved difference between schemes.
- 90% interval inside plus or minus 10 pp: the schemes agree within the
  prespecified tolerance of 10 pp. This is not a claim that they are
  identical.
- Also reported: each scheme's effect, the two-point tracking slope under
  each scheme (reference 0 and 23.2 pp), call-time ranges by scheme, and
  agreement of the concurrent frozen choices with the archived records.
All results are reported. Nothing is rerun or excluded because of its value.

## B. Near-tie real-record check (supplementary)

Population and task identical to the real-record experiment (UCI Diabetes
130-US Hospitals, first eligible discharge-home encounter, same risk score,
same eighty examples, fixed category coupling).

Gap scale: absolute difference of the readmission-anchored linear risk score,
standardized by the training-set mean and SD (the scale of the gap strata in
the paper). Near tie: gap < 0.3.

Selection (uses only the risk score and patient indices): study-set patients
not used as examples, in the 1,200 primary pairs or in the 150 gap-stratified
pairs. Patients are visited in a random order (seed 20261006). Each unpaired
patient is paired with a uniformly random unpaired patient whose gap is below
0.3. The first 600 pairs formed are used. The higher-risk member is the
target. Result: 600 patient-disjoint pairs, median gap 0.154, 235 targets
(39.2%) whose categories change, 1,670 unique prompts (identical arm prompts
are called once within a pair and order). In 365 pairs both arms show the
same categories, so these pairs contribute zero by construction.

Concurrent wide-gap control: 300 pairs drawn at random (seed 20261007) from
the 952 primary pairs with gap at least 0.3. Their prompts are identical to
the original primary prompts (794/794). Median gap 0.949, 97 changed targets
(32.3%), 794 unique prompts. Its patients are disjoint from the near-tie
pairs. Near-tie and control prompts are sent in one queue in a random order
(seed 20261006), so the two groups are collected at the same time.

Planning: with the variance of the 248 near-tie pairs in the primary sample
(0.0293), the 95% half-width is about 1.4 pp. Under the conservative bound
(variance at most the change fraction) it is about 5.0 pp. These are
approximations, not guarantees.

Primary analysis: PSE over the 600 near-tie pairs with a 95% pair-bootstrap
interval. The changed-target subset is always reported next to it, with its
own interval, so that dilution by unchanged pairs can be judged. Missing
data: an arm averages its valid orders, and a pair needs a valid answer in
both arms.

Reading rules:
- 95% interval entirely inside plus or minus 5 pp: the average effect among
  near-tie pairs is small. This weakens, but does not exclude, the
  explanation that wide risk gaps produced the near-zero primary effect.
- 95% interval excluding zero with a point estimate of at least 5 pp in
  absolute value: a meaningful near-tie effect.
- Otherwise: inconclusive, reported with the interval.
Secondary: same-example learner reference and model-minus-learner for each
group, and the direct difference between the near-tie pairs and the
concurrent wide-gap control (independent patients, separate bootstrap). A
difference between groups is judged only from this direct comparison.
Agreement of the control's answers with its original calls is descriptive.
