# P14 — Methods vs. Code Audit

**Scope:** `manuscript/main.tex` Methods §Machine Learning Modeling, §Explainability Analysis,
§Counterfactual Generation (plus the supplementary subsections that carry their parameter tables),
audited against `module/dataload.py`, `module/utils2/{selection,optimization,explainability,counterfactuals}.py`,
the drivers `selreport.py` / `optreport.py` / `expreport.py` / `cfreports.py`, and the five
`module/experiments/bin_*_final_202608.yml` configs actually read by those drivers.
**Date:** 2026-09-01 · **Branch:** `codereviewed` · **Mode:** report only, no manuscript edits.

Findings marked **[reproduced]** were confirmed by executing code against the real dataset and the
stored `final_202608` artifacts, not by inspection alone.

---

## Discrepancy table

| # | Manuscript says | Code does | Sev |
|---|---|---|---|
| 1 | "a classification threshold was then selected per split as the value maximizing F1 on the precision–recall curve of those **out-of-fold probabilities**" (§hpo) | Each OOF refit passes its own validation fold as CatBoost's `eval_set`, so the tree count is chosen on the fold being scored. The refits stop at **1–40 trees of 500**, collapsing the OOF probabilities into a ~0.45–0.55 band; the threshold is read off that band. **[reproduced]** | **BLOCKER** |
| 2 | "The winning trial's parameters were **refit on the full training set** and separately used for three out-of-fold refits… All other settings held constant: …at most 500 boosting iterations, and early stopping after 50 rounds" (§hpo) | The threshold is calibrated on 1–40-tree models (proba range 0.45–0.55) and then applied to a **500-tree** model whose test proba range is 0.04–0.99. Different probability scales. **[reproduced]** | **BLOCKER** |
| 3 | "Per-fold thresholds were tightly clustered (0.483–0.500), so this narrow range cannot explain the considerable spread in specificity" (Results §HPO) | The clustering is an artifact of finding 1, not a property of the tuned models. The stated inference rests on a mechanism that does not hold. | **BLOCKER** |
| 4 | "**Up to 20** counterfactuals targeting the opposite class were requested per instance" (§Local CF); "Counterfactuals per instance: 20" (Table `tab:local_cf`) | `config.dice.local_cf.nrepeats: 3` → three seeded DiCE runs of `total_CFs: 20` each, pooled and de-duplicated: **up to 60**. The manuscript's own tables report 21, 22, 24, 24 CFs for four patients. **[reproduced]** | **FAIL** |
| 5 | "Fourteen algorithms… (logistic regression, LDA, **QDA**, SGD, GaussianNB, linear SVM, kNN, RBF SVM, decision tree, RF, extra trees, gradient boosting, XGBoost, LightGBM, CatBoost)" — 15 names for a count of 14 (§Models, §Candidate models) | `QDA` is commented out in `selection.py:55` (`# produces errors on some runs`). 14 algorithms + Naive baseline ran. **[reproduced]** | **FAIL** |
| 6 | "fold-by-fold variability in both **Bayesian threshold selection** and tuned hyperparameters" (Results §HPO) | Threshold selection is `argmax` of F1 over `precision_recall_curve` points. Nothing Bayesian. | **FAIL** |
| 7 | Nothing in Methods on how many DiCE runs, their seeds, or that `config.experiment.random_seed` is unused by the CF stage | `np.random.seed(s)`/`random.seed(s)` for `s in (0,1,2)` inside `generate_diverse_cfs`; the config seed is never read by this stage | **FAIL** |
| 8 | "Sparsity Mean… L1 and L2 means… both computed **only over the six actionable features**" (Table `tab:localcf-model-level`) | `get_local_cf_distances` diffs over **all 22 shared feature columns**. Numerically identical only because the unactionable-violation guard runs first. | ADVISORY |
| 9 | §General Setup does not say what data DiCE sees (the supplementary table does) | `dice_ml.Data(dataframe=dfXy_test)` — the fold's **held-out test set** supplies the MADs (`feature_weights='inverse_mad'`) and the kdtree population init | ADVISORY |
| 10 | Nothing on the post-generation actionability guard in Methods (it appears only in Discussion §cf-plausibility) | `drop_cfs_outside_features_to_vary` silently removes CFs that moved a pinned feature, before any csv or figure is written | ADVISORY |
| 11 | Nothing on the per-patient wall-clock budget in Methods (Results states "1-hour search budget") | `TIMEOUT_PRESETS['normal'] = 3600 s`, default `--gen-timeout normal`; on exhaustion the patient gets `error.txt` and is dropped from the reported set. The 8-of-61 yield is **hardware- and load-dependent**. | ADVISORY |
| 12 | Nothing on the `NoCol` feature set's derivation timing | `get_high_vif` is computed once on the **whole cohort** before any CV split (`selreport.py:95`). Unsupervised (no `y`), and `NoCol` is not the selected set. | ADVISORY |
| 13 | "SHAP attributions are computed… wrapping the model's positive-class probability as a black-box function" | Correct, but the resulting explainer is SHAP's **Permutation** explainer with no seed set — run-to-run variation is unbounded and unreported | ADVISORY |
| 14 | Explainability Methods describes only CatBoost importances and SHAP | `expreport.py` also produces per-fold Brier curves, pooled AUROC/AUPRC, pooled calibration and decision-curve analysis (bootstrap n from `config.experiment.random_seed`) | ADVISORY |
| 15 | "Code availability: Not applicable"; no seed value, no library versions anywhere | Seed 42; dice-ml 0.9, scikit-learn 1.4.2, catboost 1.2.10, shap 0.49.1, optuna 4.8.0, numpy 1.26.4, pandas 2.2.2, scipy 1.13.0 | ADVISORY |
| 16 | "a further 3 were dropped for incomplete measurements" (§cohort) | Correct and correctly attributed: `nan_strategy` defaults to `"drop"` and no driver overrides it. Codes 36 / 46 / 173, on `NS,CAS` / `DM_DUR,HBA1C` / `NS`. **[reproduced]** | **OK** |
| 17 | "the NCS variables are excluded from every modeling stage, before selection begins" (§leakage) | Verified on every path. No NCS column reaches any model, any SHAP call, or `features_to_vary`. **[reproduced]** | **OK** |
| 18 | "CatBoost's positive-class weight (`scale_pos_weight`) was tuned per train/test split over 0.3–1.0… SMOTE was deliberately not used" | Exactly matches `param_space.scale_pos_weight: {min: 0.3, max: 1.0}`. No SMOTE / imblearn / `class_weight` anywhere in the repo. **[reproduced]** | **OK** |
| 19 | "outer 4×10 repeated stratified CV, inner 3-fold, Optuna TPE, 100 trials, single scoring on the held-out test set" | Exactly matches `bin_opt_final_202608.yml` and `nested_cv_optimization`. The outer test fold is touched once, at `optimization.py:214`. | **OK** |
| 20 | "the four fold models of the *first* repeat… are additionally saved for reuse without refitting" | `if fold_idx < n_splits_outer` on a `RepeatedStratifiedKFold` that yields repeat-major, asserted to tile all 187 rows. Both downstream stages load that joblib and use each fold's own stored threshold. **[reproduced]** | **OK** |

---

## Headline result

**Two BLOCKERS, both in the same defect (findings 1–3), and both narrower than they first look.**

- The defect is **inside the outer training fold**. It is *not* test-set leakage. The outer test
  set is split off at `optimization.py:97` and read exactly once, at line 214. Nothing about the
  4×10 outer structure, the inner 3-fold structure, the 100-trial TPE search, or the
  train/test separation is misdescribed. Findings 17–20 confirm the manuscript's leakage,
  label-purity, model-provenance and threshold-provenance claims as literally written.
- **Threshold-free metrics survive intact.** AUPRC 0.886 (95% CI 0.871–0.902) and ROC-AUC 0.799
  (0.776–0.821) — the paper's headline discrimination claims, its no-skill comparison, its
  model-selection argument and its feature-importance results — are computed from
  `predict_proba` on the untouched test fold and are unaffected.
- **Every threshold-dependent number is affected**: sensitivity, specificity, accuracy,
  precision, F1, F1.25–F2 and Youden in Tables `tab:optimization_metrics_40repeats` and
  `tab:catboost_metrics_firstrepeat`; the confusion-matrix-derived discussion of the
  operating point; and — because `get_instances_of_interest` selects on
  `|p − threshold| ≤ 0.08` — *which 61 patients became counterfactual candidates*.

Finding 4 (CF counts) is independently checkable from the manuscript's own tables and should be
corrected regardless. Findings 5 and 6 are one-word factual errors.

---

## BLOCKER 1 — the "out-of-fold" probabilities that set each threshold are not out-of-fold

`optimization.py:177-193`, the OOF threshold loop:

```python
for inner_train_idx, inner_val_idx in inner_splits:
    fold_model = model_class(**best_params, random_state=random_state)
    fit_kwargs = {"verbose": 0}
    if "early_stopping_rounds" in best_params:
        fit_kwargs["eval_set"] = (X_outer_train[inner_val_idx], y_outer_train[inner_val_idx])
    fold_model.fit(X_outer_train[inner_train_idx], y_outer_train[inner_train_idx], **fit_kwargs)
    oof_proba[inner_val_idx] = fold_model.predict_proba(X_outer_train[inner_val_idx])[:, 1]
```

`best_params` always contains `early_stopping_rounds: 50` (verified in the stored
`catboost_first_repeat_optimization_metrics.csv`), so the branch always fires. The model's tree
count is therefore chosen to maximise AUC **on `inner_val_idx`**, and `inner_val_idx` is then
scored by that same model and written into `oof_proba`.

The in-code comment at `optimization.py:131-135` asserts this is safe — *"inner_val_idx is never
trained on, so using it here doesn't leak into the fold_scores computed from it below"*. That is
true of the gradient updates and false of the stopping decision. `optreport_refactor.md` records
this `eval_set` as a deliberate fix ("the code already computes and reuses this exact held-out
inner split for scoring, so this doesn't introduce any new leakage") — the reasoning was applied
to the Optuna objective, where the consequence is only a biased trial ranking, and carried over to
the threshold loop, where the consequence is a biased threshold. The file's own
`train_final_model` (unused by this pipeline) goes the other way, building a `threshold_cv` with
`random_state + 1000` and commenting *"Reusing inner_cv would produce the same folds as
hyperparameter tuning… leading to an optimistically biased threshold."*

### Reproduced

Re-running each fold's inner loop from the stored `X_train`/`y_train`/`best_params`
(`StratifiedKFold(3, shuffle=True, random_state=42+fold)`) reproduces all four stored thresholds
to 4 decimal places, then re-runs them with `eval_set` removed:

| fold | stored thr | reproduced (with ES) | without ES | Δ | trees kept by ES (of 500) |
|---|---|---|---|---|---|
| 0 | 0.4986 | 0.4986 | 0.3060 | −0.193 | 3, 16, 8 |
| 1 | 0.5000 | 0.5000 | 0.2561 | −0.244 | 23, 18, **1** |
| 2 | 0.4831 | 0.4831 | 0.5505 | +0.067 | 40, 16, 23 |
| 3 | 0.4923 | 0.4923 | 0.4437 | −0.049 | 20, 13, 30 |

One of fold 1's three OOF models is a **single tree**. Applying the no-ES thresholds to the
*same* stored fold models on the *same* test sets moves the reported operating point materially:

| fold | sens (ES) | sens (no ES) | spec (ES) | spec (no ES) |
|---|---|---|---|---|
| 0 | 0.727 | 0.909 | 0.643 | 0.429 |
| 1 | 0.939 | 0.970 | 0.357 | 0.214 |
| 2 | 0.906 | 0.906 | 0.600 | 0.600 |
| 3 | 0.906 | 0.969 | 0.643 | 0.571 |

Removing the leak is not a cosmetic correction: it shifts Model 0's sensitivity — the manuscript's
named "fold-specific weak point" (Results §HPO, Discussion §foldchoice) — from 0.727 to 0.909, and
that fold is one of the three that supplied the reported counterfactuals.

**Scope note.** This is contained within the outer training fold. It does not contradict the
manuscript's §leakage claim, which is about the *test* set and is accurate as written.

---

## BLOCKER 2 — the threshold is calibrated on a different probability scale from the model it is applied to

The final refit at `optimization.py:174-175` deliberately omits `eval_set`, so it trains the full
500 iterations. `optreport_refactor.md` records this as an accepted, discussed design choice. The
consequence was not traced: the model that is *scored* has 500 trees, while the models whose
probabilities *set its threshold* have 1–40.

Probability distributions, reproduced:

| fold | OOF proba (what the threshold is read off) | final model on its test set (what the threshold is applied to) |
|---|---|---|
| | min / p25 / med / max | min / p25 / med / max |
| 0 | 0.450 / 0.497 / 0.507 / 0.551 | 0.098 / 0.316 / 0.670 / 0.949 |
| 1 | 0.351 / 0.500 / 0.511 / 0.703 | 0.053 / 0.743 / 0.906 / 0.992 |
| 2 | 0.410 / 0.505 / 0.543 / 0.655 | 0.044 / 0.515 / 0.813 / 0.979 |
| 3 | 0.480 / 0.499 / 0.505 / 0.531 | 0.242 / 0.482 / 0.583 / 0.829 |

Fold 3's OOF probabilities occupy a band 0.05 wide; the model it is calibrating for spans 0.59.
A threshold chosen on the first is not a meaningful operating point on the second.

The manuscript's §hpo reads as though both refits are the same procedure — *"refit on the full
training set and separately used for three out-of-fold refits over the same tuning folds"*, under
settings *"held constant"* including early stopping. A reader has no way to learn from the
manuscript that the two refits differ in the one respect that determines whether the threshold
transfers.

---

## BLOCKER 3 — an interpretive claim rests on the artifact

Results §HPO argues:

> Per-fold thresholds were tightly clustered (0.483–0.500), so this narrow range cannot explain
> the considerable spread in specificity (mean 0.561, range 0.357–0.643). That spread more
> plausibly reflects two compounding sources of per-fold variability: which Unconfirmed patients
> happened to fall into each fold's test set, and the substantially different hyperparameters each
> fold's independent search converged on…

The clustering is not evidence about the tuned models. Near-constant 1–40-tree models emit
probabilities piled around the base rate, so the F1-optimal cut necessarily lands near 0.5 in
every fold. Both the observation and the inference drawn from it are downstream of the defect.
Discussion §foldchoice and §limitations inherit the same reasoning.

---

## FAIL 4 — "up to 20 counterfactuals per instance" contradicts the manuscript's own tables

`bin_cf_final_202608.yml` sets `local_cf.nrepeats: 3` and `local_cf.total_CFs: 20`.
`generate_diverse_cfs` (`counterfactuals.py:600-660`) loops `for s in seeds` with
`seeds = list(range(nrepeats))`, seeding `np.random.seed(s)` / `random.seed(s)`, calls
`generate_counterfactuals(total_CFs=20)` each time, concatenates all three results with the query
instance, and de-duplicates. The ceiling is 60, not 20.

The manuscript states 20 in two places (§Local Counterfactual Analysis; Table `tab:local_cf`),
and then reports counts above 20 in three more (`tab:localcf-listing`, `tab:localcf-changed`, and
the case narratives — *"the 22 counterfactuals generated for her"*, *"The 24 generated
counterfactuals"*). Verified against the artifacts (`split*/nofiltering/*/…_local_cf.csv`, rows
minus the query row): 17, 24, 21, 15, 24, 22, 21, 14 → 158 total, matching
`tab:localcf-changed`. Four of eight exceed 20.

Also undescribed from the same call: `stopping_threshold` is set to the fold's tuned threshold
(this one *is* in `tab:local_cf`), while `maxiterations` (DiCE default 500), `sparsity_weight`
(default 0.2) and `initialization` (default `kdtree`) are left at library defaults and unstated.

---

## FAIL 5 — QDA is named among the fourteen algorithms but was never run

`selection.py:55`:

```python
# "QDA": QuadraticDiscriminantAnalysis(), # produces errors on some runs
```

`All_benchmarking_scores.joblib` contains exactly 15 entries — Naive, Logistic Regression, LDA,
SGDClassifier, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM,
CatBoost, kNN, Naive Bayes, Linear SVM, RBF SVM. No QDA.

The *count* "Fourteen algorithms" is correct (14 + the Naive baseline), and so is
"$14 \times 12 \times 40$ scored runs". The *enumeration* is not: §Models lists 15 algorithm
names, and Additional file 1 §Candidate models devotes a clause to QDA's class-specific
covariance. Dropping QDA from both lists reconciles them.

---

## FAIL 6 — "Bayesian threshold selection"

Results §HPO attributes fold-to-fold specificity variation to *"fold-by-fold variability in both
Bayesian threshold selection and tuned hyperparameters."* Threshold selection is
`np.argmax` over F1 computed on `precision_recall_curve` output (`optimization.py:199-208`) — a
grid scan of observed probabilities, with no prior, surrogate or acquisition function. Methods
§hpo describes it correctly; only this Results sentence does not.

(The related caption on `tab:catboost_metrics_firstrepeat`, *"tuned using Bayesian optimization
targeting the AUPRC"*, is defensible — TPE is sequential model-based optimization — but reads
oddly next to Methods' precise "Tree-structured Parzen Estimator". Note the codebase does contain
genuine `skopt.BayesSearchCV` code in `train_final_model`; it is not on this pipeline's path.)

---

## FAIL 7 — counterfactual seeding is undocumented

`cfreports_refactor.md` records it explicitly: *"`config.experiment.random_seed` is not used by
this pipeline. Reproducibility rests on `np.random.seed(s)` / `random.seed(s)` per repeat inside
`generate_diverse_cfs`."* The manuscript says nothing about CF seeding at all, and a reader who
noticed `random_seed: 42` in a released config would draw the wrong conclusion.

---

## Verification of the four checks that came back clean

### Check 2 — the label is not in the predictors  ✅

The outcome is derived from `Confirmed` (Toronto criteria, NCS-based) in
`dataload._build_dpn_status`. NCS exclusion was traced on every path, not just the asserted one:

| path | mechanism |
|---|---|
| selection | `Xnoncs = X.drop(columns=D.ncs_cols)`; every one of the 12 feature sets is built from `Xnoncs.columns` or from `D.{profile,comorbidity,neuro,mnsi,sudo}_cols` (`selreport.py:80-113`) |
| optimization | `no_ncs_datacols = [c for c in data_cols if c not in D.ncs_cols]` (`optreport.py:60`) — so the persisted `X_train`/`X_test` in the joblib are already NCS-free |
| explainability | same filter, **plus** `assert list(X.columns) == persisted_cols` (`expreport.py:119`), which pins the refit and every importance/SHAP label to the persisted 22 |
| counterfactuals | `dfXy` is built from the NCS-free `X`; `features_to_vary` is derived from `dfXy.columns`, then triple-asserted (`cfreports.py:197-199`) |
| eda | `edareport.py:354-378` audits the exclusion against `config.analysis.excluded_groups` rather than assuming it |

**[reproduced]** Loading the real dataset yields exactly 22 predictors, no NCS column among them;
the four stored fold models each carry `X_train` of shape (140–141, 22).

The `cfreports.py` assertion is therefore belt-and-braces — `features_to_vary` is drawn from a
frame that never held an NCS column. The load-bearing guard is the `no_ncs_datacols` filter in
each driver. One robustness gap worth knowing: `features_to_vary` is a *complement*
(`[c for c in dfXy.columns if c not in columns_not_to_vary]`), so any predictor added to the
dataset in future becomes actionable by default unless it is also added to the config's
`unactionable` list. The NCS assertion would still catch NCS specifically.

The manuscript states the exclusion twice and prominently — §cohort in bold, and §leakage as its
first substantive point. Both are accurate.

**One presentational note, not a leak.** `plot_local_cf_heatmap2` renders the patient's raw NCS
values in the per-patient report's metadata table, from `Xfull` (`counterfactuals.py:903-922`).
This is a display-only context table; `Xfull` reaches no model, no DiCE object and no distance
computation. The per-patient report figures are not among the manuscript's figures.

### Check 3 — threshold provenance  ✅ (within the limits of Blockers 1–2)

Each fold's threshold is stored alongside its model in `first_repeat_results[fold_idx]`
(`optimization.py:262-271`) and both downstream stages read it from that same record:
`cfreports.py:242` → `cf.CatBoostWrapper(model, threshold)`, and the candidate selection at
`cfreports.py:286-291` passes the same value as both `threshold` and the centre of the
`±0.08` band. No cross-fold threshold reuse, and no shared threshold, anywhere.
`expreport.py:143` reads the threshold but the explainability outputs it produces (importances,
SHAP, ROC/PR, calibration, Brier) are all threshold-free.

Methods §hpo and §General Setup describe this correctly (*"each wrapped with **that fold's own**
tuned decision threshold rather than one shared across folds"*).

### Check 4 — which models are reported  ✅

The first-repeat selection is correct: `RepeatedStratifiedKFold` yields repeat-major, so
`fold_idx < n_splits_outer` captures repeat 1, and the code asserts those four test folds tile all
187 rows (`optimization.py:298-300`). **[reproduced]** — the joblib holds keys 0–3, test shapes
47/47/47/46 = 187.

The manuscript is unusually careful here, and I could not find a place where a reader would
mistake the four for the study: §hpo states it, Results §HPO states it twice, Discussion
§foldchoice devotes a paragraph to *why* (no meaningful average of 40 independently tuned
ensembles) and pre-empts the cherry-picking objection, and Additional file 1 §Hyperparameter
optimization opens by restating that the headline claim is the 40-split mean.

Two soft spots, both caption-level:

- `tab:catboost_metrics_firstrepeat`'s caption never says "first repeat" — it says *"4 CatBoost
  models independently trained and tested over 4 data splits… together constituting evaluation
  across the entire dataset."* True of one repeat, but a reader who reaches the table from the
  List of Tables sees no pointer to the other 36 splits.
- `fig:explainability` ("across the four fold models") and `fig:globalcf` ("by the fold model that
  produced them") likewise stand alone. Results §Feature Importance opens "Across all four folds"
  without re-anchoring.

None of these is wrong; adding "first-repeat" to the three captions would close the gap.

### Check 8 — the missing-data story  ✅

`nan_strategy` defaults to `"drop"` in `DPN_data.load`, and **no driver passes it** — all four
call `D.load(classification=config.experiment.classification_type)` and no config file contains a
`nan_strategy` key. So the drop path is what produced every reported number.

**[reproduced]** 190 → 187; missing codes exactly `[36, 46, 173]`; 130 Confirmed / 57 Unconfirmed;
prevalence 0.6952 (manuscript: 0.695, and 130/187 in the `fig:catboost_auprc` caption).
`dataset/cleaning_report.txt` gives the reasons: row 35 → `NS, CAS`; row 45 → `DM_DUR, HBA1C`;
row 172 → `NS`. Mean-imputation report: 0 cells. The manuscript's "3 were dropped for incomplete
measurements" is accurate and describes the strategy actually used.

Two latent issues that do **not** affect this run but bear on the released code:

- `tests/test_dataload.py::test_impute_mean_strategy_leaves_no_nan_anywhere` is a strict xfail
  documenting that NaNs in `categorical_cols` bypass `nan_strategy` entirely. Patient 46 has a
  missing `INSULIN`; under `drop` it is removed only because `DM_DUR` is *also* missing. Under
  `impute_mean` it would survive as a live NaN. Since the final configs use `drop`, no reported
  number is affected — but a reader re-running with `impute_mean` would get a silently different
  cohort.
- `self.patient_codes` is set **only** in the `drop` branch (`dataload.py:170`), so
  `index_to_patient_code` — which names every counterfactual output folder and figure — would
  raise under `impute_mean`.
- Row inclusion is in principle sensitive to NCS completeness, since `ncs_cols ⊆
  initial_numeric_cols` drives the drop mask. In this dataset no NCS NaN survives (`NR` and
  `NO F WAVE` are mapped to 0 first), so all three drops are driven by predictor columns.

---

## Check 1, remainder — the leakage audit in full

Beyond Blockers 1–2, everything the manuscript claims about separation holds.

| Step | Where it happens | Verdict |
|---|---|---|
| Imputation | Never runs (0 cells imputed). Row-drop on missingness precedes CV but estimates nothing from other rows. | clean |
| Scaling | Selection stage only, inside `build_smart_pipeline` → `Pipeline` → `cross_validate`, so fitted per training fold. Optimization/explainability/CF stages use CatBoost unscaled. | clean |
| Encoding | `one_hot_encode=False` on every driver; categoricals stay as 0/1 numerics. | clean |
| Feature selection | The 12 sets are fixed *a priori* by clinical group. `NoCol` alone derives from a whole-cohort VIF (finding 12) — unsupervised, and not the selected set. | advisory |
| Threshold selection | Confined to the outer training fold, but see Blockers 1–2. | **blocker** |
| Test-fold scoring | `optimization.py:214-236`, once per fold, after tuning. | clean |
| Downstream stages | Both load the persisted split and never re-partition. The explainability refit uses the persisted `X_train` only. | clean |

Two further points the manuscript should state but which are not defects:

- **Threshold folds are the tuning folds.** `inner_splits` is materialised once and reused by both
  the Optuna objective and the OOF loop (`optimization.py:100-110`). The manuscript is honest
  about this ("over the same tuning folds"); it just does not flag that the hyperparameters were
  themselves selected on those folds, so the OOF estimate is optimistic on a second, independent
  count. `train_final_model` in the same file uses a different seed precisely to avoid this.
- **DiCE's reference distribution is the test fold** (finding 9). `dice_ml.Data` is built from
  `dfXy_test`, and DiCE 0.9's genetic method defaults to `feature_weights='inverse_mad'` and
  `initialization='kdtree'` — both computed from that frame. This is a post-hoc explanation of
  test-fold patients using test-fold statistics, not a prediction leak, and Table `tab:dice_setup`
  does state *"that fold's own held-out test set"*. §General Setup, which is what most readers
  will read, does not.
- **CF permitted ranges use whole-cohort σ**, in both `get_global_permitted_range` and
  `get_local_permitted_range` (documented in both docstrings and in `cfreports_refactor.md`). The
  manuscript states this for the global range ("over the whole cohort") and implies it for the
  local one ("scaled to a magnitude seen in the cohort"). Adequate.

The manuscript's §leakage paragraph makes one further claim worth confirming: *"CatBoost's feature
importances are the one exception: they are read off the fitted trees rather than from held-out
predictions."* Correct — `model.get_feature_importance()` with no data argument returns
PredictionValuesChange from the fitted ensemble (`expreport.py:187`). And SHAP is computed on
`X_test` with `X_test` as masker (`collect_shap`, `explainability.py:961-963`), matching *"the
same test fold as background distribution."*

---

## Check 5, remainder — class imbalance

The account is accurate. Two things a reader would benefit from that are not said:

1. **The positive class is the majority class.** Confirmed DPN is 130/187, so tuning
   `scale_pos_weight` over 0.3–1.0 *down-weights the majority* — the range is bounded at 1.0, i.e.
   the search can never up-weight the positive class. That is a deliberate and defensible
   asymmetry for a screening task that already over-predicts positives, but §classimbalance
   describes it only as "CatBoost's positive-class weight… was tuned per train/test split over
   0.3–1.0", which reads as neutral tuning.
2. **Interaction with the threshold.** All four tuned weights land in 0.633–0.954 and all four
   thresholds in 0.483–0.500, which the manuscript reads as two independent sources of variation.
   Given Blocker 1, the thresholds carry very little information about the tuned models, so the
   Results §HPO argument that the weight "directly governs how the model trades sensitivity
   against specificity" while the threshold range "cannot explain" the spread is not supported by
   the evidence presented.

No SMOTE, `imblearn`, `RandomOverSampler` or `class_weight` appears anywhere in the repository —
the deliberate-omission claim is true. **[reproduced]**

---

## Check 6, remainder — counterfactual setup, item by item

| Manuscript claim | Code | ✓ |
|---|---|---|
| DiCE backend `sklearn`, `model_type` classifier | `dice_ml.Model(model=wrapped_model, backend="sklearn", model_type="classifier")` | ✓ |
| method `genetic` | `dice_ml.Dice(d, m, method=config.dice.method)`, `method: genetic` | ✓ |
| background data = that fold's test set | `dice_ml.Data(dataframe=dfXy_test, …)` | ✓ (supplementary only — finding 9) |
| six features permitted to vary: HBA1C, INSULIN, HPN, DSLPDMIA, CKD, PAOD | `dice.cf_features.actionable: INSULIN,HBA1C,HPN,PAOD,DSLPDMIA,CKD` | ✓ |
| global range `[max(0, min−σ), max+σ]` over the whole cohort | `get_global_permitted_range`; the `minval==0 → 0` special case is equivalent to `max(0, …)` | ✓ |
| instance range `[max(0, xᵢ−σ), xᵢ+σ]` | `get_local_permitted_range`; categoricals skipped, `monotonic_cols` empty (`progressive: none`) | ✓ |
| 20 CFs requested per instance | 3 seeded runs × 20, pooled | ✗ finding 4 |
| desired class = opposite | `desired_class="opposite"` | ✓ |
| proximity 0.5, diversity 1.0, categorical penalty 0.1 | all three passed from config | ✓ |
| post-hoc sparsity: binary search, param 0.05 | `posthoc_sparsity_algorithm='binary'`, `posthoc_sparsity_param=0.05` | ✓ |
| stopping threshold = fold's tuned threshold | `stopping_threshold=threshold` | ✓ |
| candidate rule: misclassified **or** within 0.08 of the fold threshold | `misclassified_mask \| (np.abs(y_proba − threshold) <= delta)`, `threshold_delta: 0.08` | ✓ |
| 61 candidates (19/13/12/17) | **[reproduced]** from the four `instances_of_interest.csv` | ✓ |
| **sufficiency / necessity checks** | **`check_sufficiency` and `check_necessity` are never called.** | see below |

### Sufficiency and necessity — dead code, and correctly absent from the manuscript

`utils2/counterfactuals.py:1161` and `:1187` define `check_sufficiency` and `check_necessity`, and
`bin_cf_final_202608.yml` carries `sufficiency.maxiterations: 200` and
`necessity.{maxiterations: 500, total_CFs: 2, nrepeats: 5}`. Neither function is called from
`cfreports.py`, `generate_local_cf_reports`, `postreports.py`, or anywhere else in `module/`
(only from `legacy/`). No sufficiency or necessity artifact exists under
`experiments/binary/counterfactuals/`.

**The manuscript does not claim them**, which is the correct state. The only near-occurrences are
rhetorical — Discussion §catboost_vs_rf's *"'minimal sufficient change' logic"* and
§cf-plausibility's *"necessary but not sufficient"* — neither of which asserts a computed check.
No action needed on the manuscript; flagged so the two config blocks are not mistaken for a
described procedure, and so the dead code is not read as evidence of an unreported analysis.

### Also done but not described

- **`drop_cfs_outside_features_to_vary`** (finding 10). DiCE's genetic search violates
  `features_to_vary` on this dataset in a small fraction of CFs; the guard removes them before
  anything is written, and saves them to `*_local_cf_unactionable.csv`.
  `cfreports_refactor.md` documents an extensive investigation showing the cause is DiCE, not the
  calling code, and that it is **not confined to categoricals** (a LogisticRegression control
  moved the continuous `NS` in 19 of 20 CFs for one patient). Discussion §cf-plausibility covers
  this well and the case narratives report the individual removals; Methods does not mention it.
- **The 1-hour per-patient budget** (finding 11) appears in Results but not Methods. Because the
  budget is wall-clock and the timeout runs the work in a child process, *which* candidates
  succeed is hardware- and load-dependent. `cfreports_refactor.md` notes that difficulty is not
  predicted by margin: patient 74 (margin 0.0141) exhausted an hour, patient 40 (margin 0.0143)
  finished in 5.5 minutes. The "8 of 61" figure is honest about the *observed* run but is not a
  property of the method.
- **DiCE's global feature importance** (`get_global_importance`, and `dice.global_cf` in the
  config) was **not** run for `final_202608` — no output exists. The manuscript correctly
  describes no such analysis; `fig:globalcf` is an aggregation of the local CFs by `postreports.py`.
  Noted only so the unused config block is not mistaken for an unreported stage.
- **`postreports.py`** is a sixth, undocumented stage (`bin_postreport_final_202608.yml`) that
  builds `tab:localcf-*`, `global_cf_counts.png` and the legend-free explainability panels.
  Presentation only, but it means several manuscript tables have a code provenance the Methods
  does not mention.

---

## Check 7, remainder — randomness and reproducibility

`config.experiment.random_seed: 42` in all five configs. Actual seeding:

| Stage | Seeded | Not seeded |
|---|---|---|
| selection | `RepeatedStratifiedKFold(random_state=42)` | **Every stochastic estimator.** `DecisionTree`, `RandomForest`, `ExtraTrees`, `GradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost`, `SGDClassifier` and both `SVC(probability=True)` are constructed with no `random_state` (`selection.py:51-71`). Re-running `selreport.py` will not reproduce Table `tab:selection_metrics`. |
| optimization | outer CV 42; inner CV `42+fold_idx`; `TPESampler(seed=42+fold_idx)`; CatBoost `random_state=42` | — **[reproduced: all four stored thresholds and fold models regenerate exactly]** |
| explainability | CatBoost refit `random_seed=42`; bootstrap seeds from config | **SHAP.** `shap.Explainer(fn, X_test)` resolves to the Permutation explainer with no seed; mean-\|SHAP\| values vary run to run. The manuscript quotes them to 3 decimals (e.g. "SHAP 0.047–0.079"). |
| counterfactuals | `np.random.seed(s)`/`random.seed(s)`, `s ∈ {0,1,2}` per instance | `config.experiment.random_seed` unused; wall-clock timeout makes the successful set machine-dependent |

**What a reader would need, and currently has none of.** "Code availability: Not applicable", no
seed value in the text, and no software versions anywhere. The pipeline's own environment is
pinned (`installation/dpncf.yml`, `environment_manual_install.txt`) and was verified in place:

> Python 3.12 · dice-ml 0.9 · scikit-learn 1.4.2 · catboost 1.2.10 · shap 0.49.1 ·
> optuna 4.8.0 · numpy 1.26.4 · pandas 2.2.2 · scipy 1.13.0 · xgboost 3.2.0 ·
> lightgbm 4.6.0 · scikit-optimize 0.10.2 · statsmodels 0.14.6

Minimum additions for reproducibility: the seed (42), the version block above, an explicit
statement that the selection-stage estimators are unseeded (so its table is reproducible in
ranking but not in digits), and a note that the counterfactual yield depends on a wall-clock
budget. Given that the data are restricted, releasing the code is the only remaining
reproducibility lever — "Not applicable" forecloses it.

---

## Recommended actions

**Must fix before submission**

1. Re-run `optreport.py` with the OOF threshold loop's `eval_set` removed (a one-line change at
   `optimization.py:182-185`), so the threshold is calibrated on genuinely held-out probabilities
   from models trained the same way as the final refit. Regenerate the two metrics tables and, if
   the thresholds move, the counterfactual candidate sets. *If a re-run is not feasible*, the
   alternative is to report threshold-free metrics only (AUPRC, ROC-AUC) and withdraw the
   threshold-dependent columns and the operating-point discussion — but that removes the paper's
   sensitivity claim, which is its clinical argument.
2. Correct "up to 20 counterfactuals" to "up to 60 (three seeded runs of 20, pooled and
   de-duplicated)" in §Local Counterfactual Analysis and Table `tab:local_cf`. This reconciles the
   Methods with four numbers the manuscript already prints.
3. Drop QDA from §Models and Additional file 1 §Candidate models.
4. Replace "Bayesian threshold selection" in Results §HPO with the actual procedure.

**Should fix**

5. State in §General Setup that DiCE's reference distribution is the fold's held-out test set
   (currently supplementary-only), and that permitted-range σ is a whole-cohort statistic.
6. Move the actionability guard and the 1-hour budget into Methods, and note the budget makes the
   yield hardware-dependent.
7. Correct the sparsity/L1/L2 caption to "over all 22 features; only the six actionable features
   can differ, by construction of the guard".
8. Add "first repeat" to the captions of `tab:catboost_metrics_firstrepeat`, `fig:explainability`
   and `fig:globalcf`.
9. Add the seed, the version block, and a note on unseeded selection-stage estimators and SHAP.
   Reconsider "Code availability: Not applicable".

**Consider**

10. Note in §classimbalance that the positive class is the majority class, so a `scale_pos_weight`
    range capped at 1.0 can only down-weight it.
11. Note that threshold selection reuses the tuning folds, so the OOF estimate is optimistic
    independently of Blocker 1.
12. Additional file 1 §Pooled model diagnostics promises per-fold ROC and PR curves and discusses
    calibration, Brier and DCA, but only the pooled AUPRC figure is included.
