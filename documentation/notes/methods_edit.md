# Methods section rewrite — session notes

## Original prompt (verbatim)

> 1. Update the methodology section so that it is faithful to the experiments done. Specifically, running the pipeline using the yml configs in the `experiments/` folder
>
> 2. Create illustrations of the following and add to `main.tex` (the illustrations can be placed in `documentation/illustration` folder)
> 	- general pipeline
> 	- repeated stratified k-fold and optimization
> 	- counterfactual generation
>
> You may find these files helpful: `nested_cv_optimization.md` and `nested_cv_optimization.svg` in `manuscript/illustration`
>
> 3. based on the experiment design explain clearly and emphasize:
> - the prevention of data leakage
> - how the small dataset was handled; explain why SMOTE was not used and deferred to class imbalance handling in CatBoost.
> - preempt the possible confusion of "4 models" with "4 experiments," raising questions about which result to trust. Emphasize: _one modeling pipeline evaluated via 4-fold CV, yielding aggregated performance metrics_, not four separate studies.
>
> 4. The following tables and figures will be presented in the Discussion section. Make sure that the methodology covers the details necessary for discussing them:
> - manuscript/references/hyperparameter_optimization/optimization_metrics_ci.csv
> - manuscript/references/selection/summaries/auprc_summary_table_mean.png
> - manuscript/references/selection/violins/All/auprc_violin.png
> - manuscript/references/explainability/catboost_all_splits_auprc.png
> - manuscript/references/explainability/catboost_all_splits_feature_importances.png
> - manuscript/references/explainability/catboost_all_splits_shap.png
> - manuscript/references/explainability/catboost_pooled_auprc.png
> - manuscript/references/explainability/catboost_pooled_calibration.png
> - manuscript/references/explainability/catboost_pooled_dca.png
>
> Make a plan for executing these tasks and confirm with me first before executing

Process directions given after the plan was drafted: work on branch `codereviewed_methods`;
save this plan to `documentation/notes/methods_edit.md` (this file) with the original
prompt at the top, commit it; then execute the plan in auto-accept mode, committing after
each completed task.

Clarifications resolved via AskUserQuestion before finalizing the plan below:
- Illustrations live in `manuscript/references/illustrations/` (matches the existing
  `nested_cv_optimization.svg`/`.md` and every current `\includegraphics{references/...}`
  path in `main.tex`), not the nonexistent `documentation/illustration/` named in the
  prompt above.
- The nested-CV figure is a **new, compact** diagram sized for the journal's column
  width, not a reuse of the existing full-width 9-panel infographic (its verified
  numbers/structure remain the source of truth).

---

## Plan

### Context

`manuscript/main.tex` §Methods (lines 392–611) was written before/independently of the
final pipeline runs and has drifted from what `module/experiments/*.yml` (the configs
`optreport.py`/`selreport.py`/`expreport.py`/`cfreports.py` actually read, per CLAUDE.md)
describe. Concretely, verified against the code and configs:

- Hyperparameter Optimization subsection describes "Repeated Stratified k-Fold... with 3
  splits" for the outer loop — the real outer loop is `RepeatedStratifiedKFold(k=4,
  n_repeats=10)` = 40 iterations (`bin_opt_final_202608.yml`), with a *separate* 3-fold
  inner loop for Optuna search. These are being conflated.
- No subsection describes the Explainability stage at all, even though its outputs
  (`catboost_all_splits_*`, `catboost_pooled_*`, `optimization_metrics_ci.csv`) are
  slated for the Discussion section per the task.
- The DiCE local-CF parameter table (Table `tab:local_cf`) has `diversity_weight: 0.1`;
  the executed config (`bin_cf_final_202608.yml`) has `1.0` for local CF (0.1 is only the
  `categorical_penalty`).
- The borderline-instance window is documented as "±0.2 of the mean decision threshold";
  the executed config and `counterfactuals.py:509` (`get_instances_of_interest`) use
  `threshold_delta: 0.08`.
- Nothing states explicitly that SMOTE was not used, why, or that class imbalance is
  instead handled via a *tuned* CatBoost `scale_pos_weight` (searched 0.3–1.0, confirmed
  in `bin_opt_final_202608.yml` and `explainability.py:109-111`).
- Nothing preempts "4 models" being read as 4 separate experiments — the four per-fold
  CatBoost refits used later in Explainability/Counterfactuals are the first repeat's
  four outer-fold models from the *same* nested-CV procedure whose pooled result is
  reported as mean ± 95% CI over 40 outer iterations, not four independent studies.
- No leakage-prevention framing is stated outright, even though the code already
  enforces it (threshold selected only on inner OOF of outer-train; outer test touched
  once; runtime assertion that the 4 outer test folds tile all 187 patients each repeat).

Goal: rewrite Methods to be faithful to the five configs in `module/experiments/`, add
the three requested illustrations, and make sure the text pre-establishes everything the
Discussion section will need to reference for the 9 named tables/figures.

### 1. Verify remaining unconfirmed facts before writing numbers

A couple of details need one more code check before they go in the manuscript text, per
CLAUDE.md's "re-derive from the pipeline, not the manuscript":

- `module/utils2/explainability.py`: confirm exactly what "first repeat" retraining does
  (which hyperparameters/threshold it pulls from the optimization stage's saved
  `catboost_first_repeat_trained_models.joblib`), and confirm the pooled-metric caveats
  already found in code comments (pooled AUROC/AUPRC more conservative than fold-wise
  mean; pooled calibration looks better than fold-wise mean — both explicitly noted at
  `explainability.py:1192-1193` and `:1518-1520`) so the methodology text pre-explains
  the discrepancy instead of leaving it for Discussion to justify from scratch.
- `module/utils2/counterfactuals.py`: confirm `check_sufficiency`/`check_necessity`
  (lines ~1161, ~1187) semantics so the General Setup / Local Counterfactual Analysis
  text describes them correctly (Discussion will reference "10 of 53 candidates" which
  depends on this).
- `module/utils2/selection.py`: confirm how `auprc_summary_table_mean.png` and the
  per-feature-set violin plots are produced (mean AUPRC per model × feature set over the
  40 repeated-CV runs; violin = distribution of per-run AUPRC), so the Feature Sets
  subsection states this precisely.

### 2. Rewrite `\section{Methods}` (main.tex ~L392–611)

Keep the existing Models subsection (already accurate) and the Feature Sets table
(already accurate), but fix/add the following, each grounded in a specific config:

- **Feature Sets / selection subsection**: state explicitly that this stage
  (`bin_sel_final_202608.yml`) evaluates 14 algorithms × 12 feature sets under
  `RepeatedStratifiedKFold(k=4, n_repeats=10)` = 40 runs, ranked by mean AUPRC
  (`sort_by: auprc`), with VIF threshold 5 for the NoCol set.
- **Repeated Stratified k-Fold Cross-Validation subsection**: correct to describe the
  *actual* nested structure used in optimization (outer 4×10=40 vs inner 3-fold), not a
  generic restatement — cross-reference the new nested-CV figure here instead of
  duplicating its content in prose.
- **New: "Preventing Data Leakage" subsection** (short, explicit): inner hyperparameter
  search and threshold selection never see the outer test fold; the outer test fold is
  touched exactly once, after tuning; runtime-asserted fold-coverage guarantee. State
  this is what licenses reporting the outer-test metrics as an unbiased estimate.
- **New: "Handling Class Imbalance and a Small Sample Size" subsection**: repeated
  stratified CV as the variance-control strategy for n=187; explicit statement that
  SMOTE was deliberately not used, with reasoning — synthetic minority oversampling in a
  22-feature clinical space from 57 minority patients risks clinically implausible
  synthetic patients and can leak information across CV folds if resampling isn't done
  strictly fold-internally; instead, class imbalance is handled inside CatBoost's loss
  via `scale_pos_weight`, tuned as a hyperparameter (0.3–1.0) alongside `depth`,
  `learning_rate`, `l2_leaf_reg` — reweighting the existing 187 real patients rather than
  fabricating new ones, which also keeps every training point valid as an anchor for the
  later counterfactual permitted-range calculations.
- **Rewrite Hyperparameter Optimization subsection** to match `bin_opt_final_202608.yml`
  exactly: outer `RepeatedStratifiedKFold(k=4, n_repeats=10)`; inner `StratifiedKFold(k=3)`
  on outer-train only; Optuna TPE, 100 trials/outer-iteration, objective = mean AUPRC over
  3 inner folds; searched params (depth 4–10 int, learning_rate 0.003–0.3 log-uniform,
  l2_leaf_reg 1.0–10.0, scale_pos_weight 0.3–1.0) vs. fixed params
  (`loss_function=Logloss`, `eval_metric=AUC`, `iterations=500`,
  `early_stopping_rounds=50`); the refit → 3 OOF refits → threshold (max F1 on inner OOF
  PR curve) sequence; outer-test evaluation recording the 13 metrics that appear in
  `optimization_metrics_ci.csv` (threshold, accuracy, precision, sensitivity,
  specificity, f1/1.25/1.5/1.75/2, youden, roc-auc, auprc), pooled as mean ± 95% CI
  (normal approximation, z=1.96, NaN-aware) over the 40 outer iterations.
- **New, prominent: "One Modeling Pipeline, Evaluated by Cross-Validation" clarifying
  paragraph** (end of the Hyperparameter Optimization subsection, or its own short
  subsection): states plainly that this is one modeling pipeline (one algorithm, one
  hyperparameter-search procedure) evaluated via 4-fold CV repeated 10 times, and that
  its primary performance claim is the pooled mean ± 95% CI over those 40 outer
  iterations — not four separate studies. Explicitly bridges to Explainability/CF: the
  "four models" referenced there are the four outer-fold refits of the *first* repeat
  only, retained as representative per-fold snapshots for interpretability and as the
  basis for counterfactual generation, not four independently-designed experiments.
- **New: "Explainability Analysis" subsection** (currently entirely missing, needed for
  6 of the 9 Discussion artifacts): describes the `bin_exp_final_202608.yml` retraining
  step — for each of the first repeat's 4 outer folds, refit CatBoost with that fold's
  tuned hyperparameters and threshold; report per-fold ("all splits") feature importance,
  SHAP, and ROC-AUC/AUPRC to demonstrate consistency across folds; separately report
  "pooled" diagnostics (AUROC, AUPRC, calibration/reliability, decision curve analysis)
  computed by concatenating the 4 out-of-sample fold predictions from repeat 1 — one
  prediction per patient across the full cohort, with no patient ever scored by a model
  trained on them. State the known pooled-vs-fold-mean divergence pattern up front (from
  §1's code-comment verification) so Discussion doesn't need to re-derive it.
- **Counterfactual Generation subsections**: fix Table `tab:local_cf`
  (`diversity_weight` → 1.0) and the borderline-window description (±0.2 → the actual
  `threshold_delta = 0.08`, and clarify it is a margin around the *fold-specific*
  threshold, not a single "mean threshold"; verify exact wording against
  `counterfactuals.py:496-524` in step 1). Add a short description of the sufficiency
  and necessity checks (`check_sufficiency`/`check_necessity`) since the Discussion's
  "10 of 53 candidates" figure depends on understanding what these checks reject.

### 3. Three new illustrations (`manuscript/references/illustrations/`)

Build each as hand-authored SVG (reusing the validated 4-role colour palette, Arial,
discrete `<rect>/<circle>/<line>/<polygon>` shapes — same conventions documented in
`nested_cv_optimization.md` §5), sized to a journal single/double-column width rather
than the existing figure's full-width 9-panel layout. For each: rasterize a PNG via
`convert -density 96` for a visual QA pass (checking label collisions/overflow, same
step the existing SVG's `.md` documents), then produce a print-quality **vector** PDF via
headless Chrome (`google-chrome --headless --print-to-pdf`, wrapping the SVG in a
minimal HTML shell) since `rsvg-convert`/Inkscape are not installed and ImageMagick
`convert` would rasterize. Keep `.svg` + `.pdf` side by side per figure, matching the
existing file's pattern, plus a short companion `.md` build-notes file per figure
(mirroring `nested_cv_optimization.md`) documenting the source-of-truth trace.

- **`pipeline_overview.svg`** — general pipeline: Data (EAMC dataset, `dataload.py`) →
  Selection (14 models × 12 feature sets, repeated 4×10 CV, `bin_sel_final_202608.yml`)
  → Optimization (nested CV, CatBoost, `bin_opt_final_202608.yml`) → Explainability
  (per-fold + pooled diagnostics, `bin_exp_final_202608.yml`) → Counterfactuals (DiCE,
  `bin_cf_final_202608.yml`), each stage box naming its driving config and its key
  output artifact feeding the next stage. Placed near the start of §Methods.
- **`nested_cv_optimization_compact.svg`** — new compact version distilling panels B–D
  of the existing infographic: outer 4×10 loop, inner 3-fold Optuna search (objective =
  AUPRC), refit + threshold selection, with the leakage-prevention property (outer test
  never touched during search) as the visual focal point. Placed inside the
  Hyperparameter Optimization subsection.
- **`counterfactual_generation.svg`** — patient selection (misclassified ∪ borderline via
  `threshold_delta`) → global vs. local permitted range → DiCE genetic-algorithm setup
  (`dice_ml.Data`/`Model`/`Dice`) → sufficiency/necessity checks → per-patient CF output.
  Placed inside the Counterfactual Generation subsection.

### 4. Wire figures into `main.tex`

Add each as a `figure*`/`figure` environment (matching the existing `\includegraphics`
style, e.g. `references/eda/fig1a_participant_flow.pdf`) with a full, self-contained
caption and a `\label` for cross-referencing, at the three locations named above.

### 5. Verification

- Re-check every new numeric claim against the five `module/experiments/*.yml` files and
  the specific code lines found in step 1 — not against the old manuscript text.
- Visually inspect each rasterized PNG for label collisions/overflow before finalizing
  the SVG.
- Attempt a local `pdflatex`/`latexmk` compile of `main.tex` if the toolchain is
  available in this environment; otherwise do a careful manual syntax review (balanced
  braces/environments, no duplicate labels, all new `\includegraphics` paths resolve).
