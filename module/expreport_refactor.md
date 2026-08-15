# `expreport.py` & `explainability.py` Review & Refactor Summary

_2026-08-15_

Session summary of a correctness review and documentation/refactoring pass on
[expreport.py](expreport.py) and [utils2/explainability.py](utils2/explainability.py) (the
final-models / explainability reporting pipeline), intended to run as:

```
python module/expreport.py bin_exp_final_202608.yml overwrite
```

## Applied changes

### Correctness bugs (confirmed via reproduction, not just inspection)

- **`expreport.py` could not complete a fresh run at all.** The per-split refit called
  `CatBoostClassifier(**best_params, verbose=0)`, but `best_params` — loaded from the
  optimization stage — already contains `'verbose': 0`, because `param_space` declares it as
  `Categorical([0])`. Python raises `TypeError: got multiple values for keyword argument
  'verbose'` before CatBoost is reached. Reproduced end-to-end against the real config:
  the script died at `Retrainining model 0...`. It had gone unnoticed because a stale
  `retrained_models.joblib` short-circuited the entire retrain block on existence alone, so
  every figure in the output directory had been produced by a revision of the code that
  could no longer execute. Fixed by merging (`**{**best_params, 'verbose': 0, ...}`), which
  also survives a future `param_space` that stops setting `verbose`.
- **The refit silently dropped `random_state`, so the explainability figures were built on
  different models than the optimization stage measured.** `opt.best_params_` returns only
  the *searched* parameters; `random_state` was set on the base estimator, not in
  `param_space`, so it never reaches `best_params` and the refit fell back to CatBoost's
  default seed of `0`. Verified against the persisted splits: predicted probabilities
  diverged from the stored models by up to **0.086** (per fold: 0.0800, 0.0759, 0.0857,
  0.0300). This matters beyond reproducibility — the per-fold `threshold` values reused for
  DCA were tuned against the *stored* models. Fixed by passing
  `random_seed=config.experiment.random_seed`; the refit now reproduces the stored models
  exactly (`0.000e+00` on all four splits).
- **`brier_curve()` did not compute a Brier curve.** It weighted the two class losses by `c`
  and `(1 - c)` instead of thresholding the scores at `c`, making the result *exactly linear
  in c* (verified: max deviation from a straight line `5.6e-17`) and making its area the
  **class-balanced** Brier score rather than the Brier score. The two coincide only when
  `n_pos == n_neg`; this cohort is ~70% positive, so all four per-split figures overstated
  the loss by 8–46%:

  | split | reported area (old) | corrected area | `brier_score_loss` |
  |---|---|---|---|
  | 0 | 0.2034 | 0.1880 | 0.1881 |
  | 1 | 0.2808 | 0.1922 | 0.1917 |
  | 2 | 0.2037 | 0.1585 | 0.1581 |
  | 3 | 0.1733 | 0.1591 | 0.1593 |

  Fixed to the Hernández-Orallo et al. (2011) definition,
  `BC(c) = 2·[π_pos·(1-c)·FNR(c) + π_neg·c·FPR(c)]`, whose defining property is that the area
  equals the Brier score — now verified on every split to within trapezoid discretisation
  error. The legend's `AUC = …` therefore became an interpretable quantity (the Brier
  score), which is what the docstring's "lower is better" always implied.
- **The pooled DCA figure reported a mislabelled uncertainty.** The annotation read
  `Net Benefit Mean: {mean_nb} ± {np.std(net_benefits)}`, where `mean_nb` is the net benefit
  *at* the mean fold threshold (not a mean of anything) and the `±` was the standard
  deviation of the net-benefit curve across all 199 swept thresholds — a property of the
  curve's shape, not an uncertainty. On the real pooled data it printed `0.473 ± 0.249`,
  implying ~50% uncertainty on a quantity whose actual bootstrap 95% CI is `[0.377, 0.579]`.
  Fixed to report that CI, using the `nb_lower`/`nb_upper` arrays the function already
  computed for its CI band and had simply never used in the annotation.
- **`plot_pooled_decision_curve_analysis` raised `NameError` whenever `pooled_df` was not
  supplied.** It called `_pool_fold_predictions()` and `_print_distribution_audit()` —
  underscore-prefixed names that do not exist in the module (`hasattr` → `False` for both).
  Dormant only because `expreport.py` always passes `pooled_df=`. The same broken block
  appeared verbatim in the `_referenced` twin. Fixed to the real names.
- **`get_colors()` raised `KeyError: 'Nerve Conduction Studies'`** for any label in
  `D.ncs_cols`: the key exists only in `COLOR_GROUP_MAP_ORIG`, where it is itself commented
  out. Dormant only because `expreport.py` filters the NCS columns out of `X`. Fixed by
  commenting out that branch, so NCS labels fall through to `palette['Gray']` — matching how
  NCS is already handled in `COLOR_GROUP_MAP_ORIG`, and leaving every existing figure legend
  unchanged (adding the key would have inserted an NCS entry into four legends for features
  those figures don't contain).
- **The pooled-DCA CSV write sat outside its `if savedir:` guard**, so `savedir=None` — the
  parameter's own default — raised `TypeError` after all the plotting and bootstrap work had
  completed. Every other save in the file is guarded.
- **The categorical→`str` conversion in the refit loop was inert, and mutated the loaded
  data in place.** The comments claimed it was "needed in CatBoost for use in DiCE", but
  `cat_features` is commented out on the fit, so CatBoost parses the strings straight back to
  floats: the fitted models report `get_cat_feature_indices() == []`, and predictions are
  bit-identical with and without the cast (`0.000e+00` on all four splits). Worse,
  `X_train = split_results[midx]['X_train']` is an alias, not a copy, so the assignment
  edited the frames inside the loaded joblib — leaking `object`-dtype columns into SHAP and
  the pooled plots, but *only* on runs where the refit loop actually executed, making the
  cached and uncached paths disagree. DiCE is never imported in this script. Removed, with a
  note recording what was removed and why.
- **The `overwrite` command-line argument did nothing.** `overwrite_reports` was parsed and
  never read anywhere in the file — its only historical consumer,
  `get_ksplit_trained_models`, is in the commented-out block. So the documented invocation
  regenerated reports whether or not the flag was passed, and omitting it silently
  overwrote existing reports. Wired up as a set-level guard on the output directory
  (`outputdir.glob(f'{config.model.code}_*')`), scoped so it sees report files but not
  `retrained_models.joblib` or the copied config; returning early skips the data load, the
  refit, the SHAP loop and the four bootstrap passes.
- **`split_results` is a `dict` but was indexed as a sequence.** The joblib stores
  `{0: …, 1: …, 2: …, 3: …}`, while `expreport.py` and five functions in
  `explainability.py` do `for s in range(len(split_results))` and pair the result
  positionally against `trained_models`, a list. That works only because the keys happen to
  be `0..n-1` in insertion order; string fold names, `(repeat, fold)` tuples, or a
  non-contiguous set after dropping a fold would give `KeyError` or silently mis-pair models
  with test sets. Normalised to a list once at load time, which also makes reality match the
  `split_results : list of dict` contract every docstring in `explainability.py` declares.
- **Feature-importance labels were aligned by assumption.** `model.get_feature_importance()`
  returns a bare array ordered by the model's training columns; `expreport.py` labelled it
  with `X.columns`, assembled from the dataset in *this* script while the models come from a
  joblib written by a separate stage. Nothing connected the two orderings, so a change to
  the selection stage would silently attach every importance to the wrong feature name.
  Verified aligned today (22 features, exact match) and now asserted explicitly at load time.
- **Target column was hardcoded.** `y = dfdpn['Confirmed_Binary_DPN']` ignored
  `config.experiment.classification_type`, which is read generically and passed to
  `D.load()`. Fixed to `D.get_target_column()`, matching how
  [optreport_refactor.md](optreport_refactor.md) and
  [selreport_refactor.md](selreport_refactor.md) resolved the same bug.
- **Dataset path resolution was CWD-dependent** — the same bug pattern as the two sibling
  scripts. `DPN_data(config.data.dataset_path[3:])` sliced the config's `../` prefix off and
  resolved the remainder against the process's working directory; verified that the result
  exists from the repo root and *not* from `module/`. Fixed to resolve against `script_dir`,
  the way `config_path` already was.

### Documentation & cleanup

- Deleted **235 lines of superseded code**, both of which wrote to the *same output
  filenames* as their live counterparts — so calling both in one session silently overwrote
  one figure with the other, with no indication on disk which had won:
  - `plot_pooled_decision_curve_analysis_referenced` (155 lines) — superseded by the live
    version, which added a bootstrap CI band, clinical-range shading and a ~3000× faster
    vectorised net-benefit computation (0.04 ms vs 122 ms per curve, measured). Before
    deleting, ported back the one thing the live version had lost: validation that
    thresholds stay below 1.0, since net benefit divides by `(1 - threshold)`.
  - `plot_importances_heatmap2` (80 lines) — a colour-strip layout variant, unreferenced in
    any script or notebook. Its removal also made `matplotlib.colors as mcolors` dead.
- Gave `plot_pooled_decision_curve_analysis` the docstring it was missing (the only one of
  the four pooled plotters without one), documenting the two parameters neither sibling has,
  `thresholds` and `clinical_threshold_range`.
- Added a **"Note on pooling"** to `plot_pooled_auroc`, `plot_pooled_auprc` and
  `plot_pooled_calibration_curve` recording that the pooled curves aggregate fold models with
  independently-tuned hyperparameters — see "out of scope" below for the finding these
  document.
- Removed the blanket `warnings.filterwarnings('ignore')`. Measured first, by running the
  full pipeline with the filter neutralised: it was suppressing **0 warnings, 0 unique**. Its
  only live effect was to guarantee that any future warning would be swallowed unseen — the
  same blind spot the in-place mutation bug above lived in.
- Removed `sys.path.append('..')`, which resolves against the working directory and so
  appended **`/home/toni_briza`** under the documented invocation, putting the user's home
  directory on the import path. All three imports (`dataload`, `ymlconfig`, `utils2`) resolve
  from `module/`, which Python places on `sys.path[0]` automatically.
- Moved `matplotlib.use('Agg')` above `import matplotlib.pyplot`, so it precedes the first
  pyplot import — most importantly the one inside `explainability.py`.
- `'$\pm$'` → `r'$\pm$'`: `\p` is an invalid escape sequence, a `SyntaxWarning` today and a
  `SyntaxError` in a future Python. Both files now compile warning-free.
- Closed an unbalanced parenthesis in the pooled-AUROC legend (`95% CI (bootstrap` →
  `(bootstrap)`), bringing it in line with the other three pooled figures.
- `confusion_matrix(y, y_pred, labels=[0, 1])` so `.ravel()` into four values cannot fail on
  a degenerate single-class slice.
- Sourced the bootstrap seed from `config.experiment.random_seed` in all four pooled plots,
  replacing a hardcoded `seed=42` that was a silent second source of truth (identical
  behaviour today, since the config declares 42).
- Import hygiene in `explainability.py`: removed a duplicated `auc` in the `sklearn.metrics`
  import, removed the unused `import json`, removed two function-local imports that shadowed
  module-level ones, and hoisted `brier_score_loss` (the one that was actually load-bearing)
  to module level. Removed a function-local `confusion_matrix` import left over from the
  loop implementation the vectorised version replaced.
- `expreport.py` cleanup: removed a dead `config_path = Path(r'experiments')` (overwritten
  nine lines later), the unused `numpy`/`pyplot` imports, and the unread `dfXy`; gave the
  `tag` assertion the same both-values message its `rundate` sibling already had; renamed the
  four unused `*_stats` returns to `_`-prefixed names.
- Annotated, rather than silently corrected, comments that contradicted their code (per the
  review convention): the `cat_features`/DiCE comments on the refit, and a note on
  `np.trapz` recording that `np.trapezoid` does **not** exist in this environment's NumPy
  1.26.4, so the swap must wait for a NumPy 2 pin.

### Tests

Added [../tests/test_explainability_correctness.py](../tests/test_explainability_correctness.py) —
45 tests in 11 classes, following the conventions of `test_optimization_correctness.py`
(stdlib `unittest`, runnable directly or under pytest, each class documenting the bug it
guards). Covers every correctness bug above, plus the pooling invariant that each patient
contributes exactly one out-of-sample prediction. Tests requiring the dataset or the
persisted joblib are `skipUnless`-guarded.

## Verification

- Both files compile cleanly after every change; `explainability.py` and `expreport.py` are
  free of `SyntaxWarning` (asserted by tests).
- **Ran the full pipeline end-to-end after each change**, against a throwaway
  `experiment.tag` so the real output directory was never touched — `exit=0`, all 17 outputs
  produced, and the printed metrics diffed identical to the previous step every time. The
  throwaway config and its output directory were deleted at the end of the session.
- Confirmed the seed fix works at the level that matters: the refit models now reproduce the
  optimization stage's stored models to `0.000e+00` on all four splits.
- Confirmed the corrected `brier_curve` area reproduces `sklearn.metrics.brier_score_loss` on
  all four splits (max deviation `5e-4`, pure trapezoid discretisation), and that the
  degenerate all-positive / all-negative guards still return the correct `0.04`.
- Confirmed the `overwrite` guard in both directions: blocks with a message when reports
  exist and the flag is absent, regenerates identically when it is present.
- Full test suite: **73 passed, 1 xfailed, 13 subtests, 6.2s** — the new file plus the three
  pre-existing ones, which are unaffected.
- **Mutation-tested the new tests**: temporarily reverted each of six fixes in the real
  source and confirmed the guarding test goes red — 6/6 detected. Originals were restored in
  a `finally` block and the file verified byte-identical afterwards. A regression test that
  passes against the buggy code is worthless, so this step is the actual evidence the suite
  works.
- Did **not** re-run the pipeline against the real `bin_exp_final_202608.yml`; the user had
  already deleted the `final_202608` output directory, and regenerating manuscript figures
  was left as their call. Expect the four `*_brier.png` and the DCA annotation to differ from
  the pre-fix versions — those are the corrections.

## Found but intentionally left unchanged (discussed with user)

- **Per-fold probability calibration before pooling was evaluated and declined as
  over-engineering.** Prototyped and measured: a per-fold Platt map, fitted leakage-free on
  out-of-fold predictions from each fold's own training data, recovers pooled AUROC
  0.7559 → 0.7960 (fold-wise mean 0.7994), AUPRC 0.8577 → 0.8880, pooled ECE 0.0856 → 0.0602,
  Brier 0.1744 → 0.1603, and improves *per-fold* ECE from a mean of 0.167 to 0.134 with
  per-fold AUROC provably unchanged (the map is monotone; rankings verified identical).
  Declined because it adds a fitted component to the deployed model to pre-empt a critique
  reviewers are unlikely to raise, and would require re-expressing the tuned thresholds on
  the new scale. **The one claim this makes unsupportable** is quoting the pooled ECE of
  0.086 as "the model's calibration" — no model achieves it; the fold models sit at
  0.11–0.20, mean 0.167. Reporting calibration fold-wise costs nothing and avoids it.
- **`cat_features` remains disabled.** Re-enabling it for DiCE would change every model, SHAP
  value and figure, and DiCE is not used in this script. The contradictory comments are now
  annotated rather than acted on; the decision belongs with `cfreports.py`.
- **A concern about the i.i.d. bootstrap was investigated and retracted.** The initial review
  flagged that the four pooled plots resample pooled predictions i.i.d. across patients,
  suggesting the CIs were optimistically narrow. The user correctly pointed out that pooling
  is fold-matched. Verified: 187 unique patients, disjoint folds, one out-of-sample
  prediction each — so the usual failure mode (a patient contributing several correlated
  rows) does not exist here. Testing the residual within-fold-correlation concern directly,
  a fold-stratified bootstrap gives `[0.6781, 0.8314]` (width 0.1533) against i.i.d.'s
  `[0.6752, 0.8307]` (width 0.1555) — a 1.4% difference, and *narrower*, the opposite of the
  claim. No change warranted.
- **Unused-looking names kept deliberately**: `from datetime import datetime` and
  `model_threshold` are read only by intentionally-preserved commented-out blocks
  (`model_threshold` was commented out alongside its consumer); the four `_*_stats` returns
  document what each plotter yields. The 14 `plt.show()` calls are no-ops under `Agg` but are
  what makes these functions usable from `explainability.ipynb`.

## Found, out of scope for this session

- **Pooled AUROC sits below every individual fold** — `0.7559` pooled against per-fold
  `[0.7814, 0.7576, 0.7812, 0.8772]`, mean `0.7994 ± 0.046`. Diagnosed, not patched. The
  cause is score-scale heterogeneity: each fold ran an independent Bayesian search and
  landed on different hyperparameters (`scale_pos_weight` 0.633–0.954, depth 4–10, learning
  rate 0.0035–0.0200), so mean predicted probability ranges 0.582–0.786 across folds while
  true prevalence is flat at 0.681–0.702. Rank-normalising within fold before pooling
  recovers `0.7985`, i.e. the entire 0.043 gap is scale mismatch rather than performance.
  Effects differ by metric family, and the directions are not the same:
  - **Brier is unaffected** (pooled 0.1744 vs fold-mean 0.1743) — it is a per-observation
    average, so pooling is exactly the size-weighted mean of the fold values.
  - **AUROC/AUPRC are understated** (−0.0435, −0.0349) — conservative, so safe but
    self-defeating if quoted as the headline result.
  - **Calibration is flattered** (pooled ECE 0.0856 vs fold-mean 0.1671) — opposing per-fold
    biases partially cancel, so the pooled reliability diagram looks roughly twice as well
    calibrated as any deployable model. This is the optimistic direction, and it propagates
    into DCA, which reads absolute probabilities against thresholds.

  Recorded in the three pooled docstrings. The reporting fix is free — lead with the
  fold-wise `mean ± std` these functions already compute and that
  `plot_roc_auc_overlapping` / `plot_cv_auprc` already plot. The root-cause fix (select
  hyperparameters once rather than per fold) is a change to the optimization stage.
- **`train_final_model`'s `eval_set` gap, flagged in
  [optreport_refactor.md](optreport_refactor.md), still stands.** `BayesSearchCV` calls
  `estimator.fit(X, y)` with no `eval_set`, so `early_stopping_rounds` in its `search_spaces`
  is inert. It reaches this file only through `get_ksplit_trained_models`, which is commented
  out in `expreport.py`'s execution path (though live in `explainability.ipynb`), so it was
  again not exercised by the specified invocation.
- **Multiclass is not supported end-to-end.** The target-column fix above makes `y` follow
  the config, but `predict_proba(X)[:, 1]` appears in `pool_fold_predictions` and five
  plotting functions, alongside binary `roc_curve`/`precision_recall_curve` usage,
  `prevalence` as `np.sum(labels) / N`, and hardcoded `'DPN positive'`/`'DPN negative'` rug
  labels. Supporting `classification_type: multiclass` is real work in `explainability.py`,
  not a rename.
- **Functions live only in `explainability.ipynb`** — `get_ksplit_trained_models`,
  `plot_importances`, `plot_roc_auc`, `plot_decision_curve_analysis`, `plot_shap`,
  `plot_cv_auprc` are called from uncommented notebook cells but are commented out in
  `expreport.py`. They were not deleted, and beyond the fixes above were not deeply audited
  against the specified invocation.

## Files changed

- `module/expreport.py`
- `module/utils2/explainability.py`
- `tests/test_explainability_correctness.py` (new)
