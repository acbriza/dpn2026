# Discussion — defense notes

Running record of the reasoning behind specific claims in `manuscript/main.tex`'s Discussion
section. Purpose: if a reviewer challenges a statement, the supporting argument and the numbers
behind it are already worked out here rather than needing re-derivation.

Each entry names the claim, where it lives in the manuscript, the reasoning, and (where relevant)
caveats or nuances that were deliberately left out of the manuscript prose.

---

## Why specificity is the most unstable metric across the 40 runs

**Claim in the manuscript** (Section 4.2.1, CatBoost Repeated Cross-Validation):

> By contrast, specificity showed the greatest instability across runs (mean = 0.527, STD =
> 0.141, 95% CI: 0.483--0.571), confirming that the model's ability to correctly exclude
> DPN-negative patients was more sensitive to partition composition. This reflects the
> relatively smaller size of the Unconfirmed-DPN class (n = 57) and, as the per-fold analysis
> below shows, fold-by-fold variability in both Bayesian threshold selection and tuned
> hyperparameters.

### 1. Is "greatest instability" actually true?

Yes. Comparing all 8 metrics' STDs in `tab:optimization_metrics_40repeats`:

| Metric | STD |
|---|---|
| Precision | 0.044 |
| F1 | 0.046 |
| AUPRC | 0.049 |
| Accuracy | 0.060 |
| ROC-AUC | 0.073 |
| Sensitivity | 0.083 |
| Youden Index | 0.138 |
| **Specificity** | **0.141** |

Specificity's raw STD is the largest of the eight, edging out even the Youden Index — which is
itself partly driven by specificity, since Youden = sensitivity + specificity − 1.

### 2. Why class size drives this (the primary, most defensible mechanism)

Specificity = TN/(TN+FP), computed **only over the negative-class (Unconfirmed DPN) patients** in
each fold's test set:

- Unconfirmed DPN: n = 57 total → ~14 per fold test set (4 stratified folds)
- Confirmed DPN: n = 130 total → ~32 per fold test set

Sensitivity is computed over the Confirmed-DPN class, so it draws on roughly 2.3× more patients
per fold than specificity does. For a proportion estimated from n trials, sampling variance
scales as p(1−p)/n — smaller n mechanically inflates variance. **So even if the model were
equally reliable at both tasks, specificity would look noisier than sensitivity purely from
having fewer patients per fold to estimate it from.** This is the statistically strongest part
of the argument and should be the first thing cited if challenged.

### 3. What the threshold/hyperparameter clause adds

This clause refers to the **full 40-run picture** (all 10 repeats), where the threshold itself
has STD = 0.042 (`tab:optimization_metrics_40repeats`, Threshold row). Each of the 40 runs picks
its own F1-maximizing threshold independently from its own inner out-of-fold split.

Specificity is directly and sharply sensitive to where that threshold lands. With only ~14
negative-class patients per fold, shifting the threshold slightly can flip 2–3 patients from
false-positive to true-negative — and 2–3 patients out of 14 is a **15–20 percentage point**
swing in specificity. Compounding this, each run's independently tuned `scale_pos_weight`
(range 0.633–0.954 in repeat 1) shifts the entire predicted-probability distribution *before*
the threshold is even applied.

### 4. Nuance deliberately left out of the manuscript (potential reviewer catch)

The very next subsubsection (4.2.2) notes that **repeat 1's own four thresholds are tightly
clustered** (0.483–0.500, STD 0.008) — far tighter than the full 40-run threshold STD of 0.042.

This is **not a contradiction**: repeat 1 is one draw of 10, and it happens to be an unusually
threshold-stable one. The two statements describe different populations (all 40 runs vs. one
specific repeat's 4 folds).

However, a reader skimming both subsections back-to-back could read "thresholds vary" in 4.2.1
and "thresholds are tightly clustered" in 4.2.2 as inconsistent. **Open item:** a short clause
distinguishing the two populations could be added to 4.2.1 if this is judged to be a real risk;
not currently in the text.

---

## Validation: the first repeat is representative of the full 40-run distribution

**Claim in the manuscript** (Section 4.2.2, Model Selection, Performance, and Clinical
Suitability for Pre-Diagnostic Screening):

> the first repeat's per-fold mean estimates (Table~\ref{tab:catboost_metrics_firstrepeat}) fall
> within roughly one standard deviation of the full 40-repeat pooled estimates
> (Table~\ref{tab:optimization_metrics_40repeats}) for every metric

This claim is load-bearing: it is the second of the two arguments defending the use of repeat 1's
four fold models as the representative set on which *all* downstream Explainability and
Counterfactual analysis depends. (The first argument — that the choice is fixed by
`random_state=42` at pipeline-design time, not selected after seeing results — stands on its own
regardless of these numbers.)

### Source of truth

Validated against the raw per-run data in
`manuscript/references/hyperparameter_optimization/optimization_results.json` (40 runs), **not**
the rounded table values. Sanity check performed first: runs 0--3 of that JSON match
`catboost_first_repeat_optimization_metrics.csv` exactly (AUPRC, sensitivity, specificity all
identical to 6 dp), confirming that runs are ordered such that each consecutive block of 4 is one
repeat.

### What `|z|` means here

`|z|` is a **descriptive standardized distance** — how far repeat 1's mean sits from the center of
a reference distribution, in units of that distribution's standard deviation:

```
|z| = |repeat-1 mean − reference mean| / reference SD
```

"Within one SD" means `|z| <= 1`.

It is deliberately **not** a z-test statistic: no p-value is attached, no significance is claimed,
and no normality assumption is made (the value is never converted into a probability). Formal
inference would in fact be inappropriate here, because the 10 repeats are not independent samples
— they are 10 reshufflings of the *same* 187 patients, with heavily overlapping training data
across repeats. That dependence would violate the independence assumption behind a real z- or
t-test. Using `|z|` descriptively sidesteps the problem, and is why the manuscript's framing is
"not an atypical draw" rather than anything statistically stronger.

### Result 1 — as the manuscript states it (vs. 40-run pooled estimates)

| Metric | Repeat-1 mean | Pooled mean | Pooled SD | \|diff\| | \|z\| |
|---|---|---|---|---|---|
| threshold | 0.4935 | 0.4814 | 0.0420 | 0.0122 | 0.29 |
| AUPRC | 0.8926 | 0.8864 | 0.0494 | 0.0062 | 0.13 |
| ROC-AUC | 0.7994 | 0.7985 | 0.0730 | 0.0008 | 0.01 |
| sensitivity | 0.8698 | 0.8864 | 0.0834 | 0.0166 | 0.20 |
| specificity | 0.5607 | 0.5268 | 0.1414 | 0.0339 | 0.24 |
| F1 | 0.8420 | 0.8452 | 0.0464 | 0.0032 | 0.07 |
| Youden | 0.4305 | 0.4132 | 0.1383 | 0.0173 | 0.12 |
| precision | 0.8210 | 0.8127 | 0.0438 | 0.0083 | 0.19 |
| accuracy | 0.7757 | 0.7765 | 0.0599 | 0.0009 | 0.01 |

**All 9 metrics pass, max |z| = 0.29.** Not marginal.

### Result 2 — the stricter, more defensible test (vs. the 10 repeat-means)

Result 1 is a **lenient** test, and a methodologically-minded reviewer may well say so: it
compares a *mean of 4 folds* against the SD of *individual runs*. A mean-of-4 is inherently less
variable than a single run, so this is an easy bar to clear. The appropriate comparison is
repeat 1's mean against the distribution of all 10 repeat-means.

Worked example (specificity — where the two yardsticks differ most). The numerator is identical
in both; only the denominator changes:

- vs. pooled: 0.0339 / 0.1414 (SD of 40 individual runs) = **0.24**
- vs. repeat-means: 0.0339 / 0.0411 (SD of 10 repeat-means) = **0.83**

Averaging 4 folds smooths out fold-to-fold noise, so the spread of repeat-means (0.0411) is much
tighter than the spread of individual runs (0.1414) — making the same absolute gap look ~3.5x
larger against the correct yardstick.

| Metric | Repeat-1 mean | Mean of 10 repeat-means | SD of 10 repeat-means | \|z\| | Rank among 10 |
|---|---|---|---|---|---|
| threshold | 0.4935 | 0.4814 | 0.0187 | 0.65 | 8/10 |
| AUPRC | 0.8926 | 0.8864 | 0.0088 | 0.70 | 8/10 |
| ROC-AUC | 0.7994 | 0.7985 | 0.0161 | 0.05 | 5/10 |
| sensitivity | 0.8698 | 0.8864 | 0.0271 | 0.61 | 3/10 |
| specificity | 0.5607 | 0.5268 | 0.0411 | 0.83 | 8/10 |
| F1 | 0.8420 | 0.8452 | 0.0188 | 0.17 | 4/10 |
| Youden | 0.4305 | 0.4132 | 0.0503 | 0.34 | 8/10 |
| precision | 0.8210 | 0.8127 | 0.0138 | 0.61 | 8/10 |
| accuracy | 0.7757 | 0.7765 | 0.0237 | 0.04 | 5/10 |

**The stricter test also passes for all 9 metrics, max |z| = 0.83.** So the claim survives the
comparison a reviewer is more likely to demand.

### Honest nuance: repeat 1 is not dead-center typical

The rank column above matters. Repeat 1 sits **8th of 10** on threshold, AUPRC, specificity,
Youden, and precision (mildly favorable side), and **3rd of 10** on sensitivity (mildly
unfavorable side).

No metric is an outlier, and — importantly for defending against a cherry-picking accusation —
the direction is **not uniformly self-serving**: repeat 1 is slightly *pessimistic* on
sensitivity, the metric this paper repeatedly identifies as the most clinically important for a
screening instrument. Combined with the fact that the repeat was fixed by `random_state=42`
before any results were seen, this is a strong position.

The defensible framing remains "not an atypical draw from the pipeline's overall performance
distribution" — **not** "perfectly average," which the rank data would not support.

### Open items (optional manuscript tightenings, not yet applied)

1. The hedge "roughly" does no work against the pooled comparison (max |z| = 0.29) and could be
   dropped — though it is a fair hedge if it is meant to implicitly cover the stricter
   repeat-level test, where the max reaches 0.83.
2. The claim would be materially stronger if it cited the **repeat-level** test (Result 2)
   instead of the pooled test (Result 1), since that is the comparison a methodologically-minded
   reviewer would actually ask for. This would require adding the repeat-level numbers to the
   manuscript or an appendix.
