# AGE / NS collinearity: implications for modeling and counterfactuals

This note summarizes the pros/cons of keeping both **AGE** and **NS** (Sudoscan
neuropathy score) as features, given the collinearity finding in **Figure S3**
(`module/experiments/binary/eda/s3_redundancy_caption.txt`), and its specific
consequences for CatBoost and for the counterfactual analysis.

## The finding (Figure S3)

- Spearman ρ(AGE, NS) = **−0.88**, the strongest pair among the 22 modelling
  features.
- VIF flags exactly these two features at the threshold=5 cutoff: NS = 10.5,
  AGE = 7.2.
- NS is age-adjusted by construction, so the pair carries overlapping but not
  identical information.
- The `NoCol` ablation (`module/selreport.py:100`, dropping every high-VIF
  feature — i.e. AGE and NS) performs **no better** than the full feature set
  (`module/edareport.md:64-68`). Removing the pair discards two informative
  predictors rather than redundant noise.

## Which config actually governs the counterfactual run

The counterfactual stage that matters is
[`module/experiments/bin_cf_final_202608.yml`](../../module/experiments/bin_cf_final_202608.yml)
(tag `final_202608`, "after code review" — the current run, not an earlier
draft config). Its `dice.cf_features` block:

```yaml
actionable: INSULIN,HBA1C,HPN,PAOD,DSLPDMIA,CKD
unactionable: SEX,AGE,SUBJ,DM_DUR,GBS,FEET_MEAN_ESC,FEET_PCT_ASYM,HAND_MEAN_ESC,
              HAND_PCT_ASYM,NS,CAS,DEC_VS,DEC_PPS,DEC_LTS,DEC_AR,MNSI
```

**Both AGE and NS are `unactionable`** — both held fixed at the patient's
observed value during counterfactual search. (An earlier config,
`module/configs/bin_cf_final_auprc_f125.yml`, put NS in the actionable list;
that is not the config that produced the final counterfactual results, and
the asymmetric-actionability concern that config would have raised does not
apply here.)

## Modeling: pros/cons of keeping both

**Pros**
- ρ = −0.88 still leaves ~1 − 0.88² ≈ 22% of NS's variance unexplained by
  AGE. That residual is plausibly the clinically meaningful part —
  sudomotor dysfunction in excess of what age alone predicts — which is
  consistent with the pair together outperforming either dropped alone in
  the ablation.
- CatBoost is largely insensitive to this kind of redundancy at the
  *prediction* level. Ordered boosting on trees splits on whichever feature
  gives the best split at each node; two correlated features don't produce
  the coefficient-variance blowup that VIF is actually designed to flag in
  linear/logistic models. VIF is being used here as a screening heuristic on
  a tree ensemble, not a diagnostic of an actual failure mode — which is
  exactly why dropping the pair didn't help.

**Cons**
- **Attribution instability**, not prediction instability. Even though
  CatBoost's outputs are robust to the redundancy, feature importance/SHAP
  credit can still be routed somewhat arbitrarily between two correlated
  features across different trees, bootstrap folds, or the `catboost_split0
  /1/2` runs in `module/experiments/binary/explainability/catboost/`. The
  *combined* importance of the pair should be stable; the *split* between
  NS and AGE individually is a weaker claim and worth a fold-to-fold
  sensitivity check before asserting one dominates the other in the
  manuscript text.
- Because NS is age-adjusted by construction, a SHAP dependence plot for NS
  is implicitly a residualized effect, but AGE is also in the model — the
  two curves can look like they disagree about the same underlying biology
  unless captioned carefully (the EDA report already does this at
  `module/edareport.md:64-68`).
- Minor: two correlated axes give a ~187-row dataset two channels to fit the
  same signal, a small overfitting surface area — second-order next to the
  actionable-feature-weakness issue already flagged in the report.

## Counterfactuals: revised assessment

Because **both** AGE and NS are `unactionable` in the config that actually
produced the results, the collinearity does **not** create the
achievability problem it would if one of the pair were searchable while the
other was fixed (e.g. DiCE proposing an NS change that isn't reachable
without an AGE change it isn't allowed to make). That risk doesn't apply
here.

What remains relevant:

- **AGE and NS jointly anchor a single, largely redundant "non-modifiable
  risk axis"** at two correlated positions. For a given patient, both values
  are held fixed, so the actionable search (INSULIN, HBA1C, HPN, PAOD,
  DSLPDMIA, CKD — six features, all comorbidity/treatment variables) has to
  do all the work of moving the model's output past `threshold_delta` (0.08)
  from wherever this fixed pair anchors it. Patients at the extremes of the
  AGE/NS axis may need disproportionately large actionable-feature
  displacements to flip class, not because the actionable features are
  individually weak (already flagged in `module/edareport.md:70-76`) but
  because the fixed axis is encoded twice rather than once. Worth checking
  empirically: does CF displacement magnitude in the six actionable features
  correlate with how extreme a patient's AGE/NS values are?
- **Interaction sensitivity, not achievability.** CatBoost can learn split
  patterns that condition on AGE in one branch and NS in a correlated branch
  for overlapping subpopulations. Because both are fixed inputs during CF
  search, measurement noise or small differences in either correlated value
  could put a patient on a different local decision surface (different
  learned interaction with the actionable features) without the patient's
  actual clinical state differing meaningfully — a source of run-to-run CF
  variability that traces back to the same redundancy, not to the
  actionable/unactionable split itself.

## Bottom line

- **Prediction (CatBoost):** keep both — empirically supported by the
  `NoCol` ablation, and collinearity is a linear-model artifact that doesn't
  hurt a tree ensemble's accuracy.
- **Explainability:** treat the individual NS vs. AGE SHAP split with more
  caution than their combined contribution; check fold-to-fold stability
  before making claims about which one "matters more."
- **Counterfactuals:** the achievability concern that would arise from an
  actionable/unactionable split across the collinear pair does not apply
  under `bin_cf_final_202608.yml`, since both are fixed. The remaining
  effect is that the fixed AGE/NS pair may inflate the actionable-feature
  displacement needed to flip class for patients at the extremes of that
  axis — worth a quick correlation check against the local CF reports.
