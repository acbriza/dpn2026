# Nested CV Optimization (Compact) Illustration — Build Notes

Companion document for `nested_cv_optimization_compact.svg` / `.pdf`.
Generated 2026-08-21 on branch `codereviewed_methods`, as part of the Methods
rewrite recorded in `documentation/notes/methods_edit.md`.

## Status — reference for the code, not for the manuscript

This figure is **not used in the manuscript**: `main.tex` does not include it, and it was
superseded by `detailed_pipeline` (see `detailed_pipeline.md`, which records it as
"retained for reference only"). It is kept as a reference for the *pipeline code* — its
panels describe what `module/utils2/optimization.py` actually computes.

It is therefore deliberately faithful to the pipeline rather than to the manuscript's
reporting. In particular, panel 4 and the pooling panel say **13 metrics**, which is the
true number recorded per outer-test iteration in `optimization_metrics_ci.csv`
(threshold, accuracy, precision, sensitivity, specificity, f1, f1.25, f1.5, f1.75, f2,
youden, roc-auc, auprc). The manuscript reports only eight of these, dropping the four
extra F-beta variants — see "Reporting decision — metric count" in
`documentation/notes/methods_edit.md`. **Do not edit this figure to match the
manuscript's counts**; the divergence is intentional.

One consequence: the last bullet under "What was condensed or dropped" below states that
this figure "is now wired into the manuscript". That was true when written and is no
longer; the pointer to `optimization_metrics_ci.csv` it describes now stands on its own
as code documentation.

## Relationship to the existing full-size figure

This is a new, smaller diagram sized for a `figure*` (double-column) journal
page, not a resize of `nested_cv_optimization.svg` (the earlier 9-panel,
1660×1260 infographic). It distills that figure's panels B–D (outer loop, inner
loop, refit/threshold) plus a compact version of panels F/G/H (searched vs.
constant hyperparameters, what each stage optimises) into four panels at
1600×600. All verified numbers are unchanged from the original and were not
re-derived — see `nested_cv_optimization.md` §2–3 for the full derivation
(dataset shape, exact split sizes via `RepeatedStratifiedKFold`/`StratifiedKFold`
enumeration, fit-count accounting). This file only records what changed in the
retelling.

## What was condensed or dropped versus the original figure

- Panel A (cohort/class-balance detail) and panel I (fit-count accounting,
  12,160 CatBoost fits) are dropped — both are already covered by the manuscript
  text (Data Collection §, and the Methods prose accompanying this figure).
- Panels F/G/H (searched hyperparameters, constant settings, per-stage
  optimisation criteria) are condensed into the footer strip and the "What gets
  reported" panel, rather than kept as three separate full panels.
- The per-repeat 4-row fold-tiling diagram is reduced from 4 rows to 2 rows plus
  a caption noting the other two, since the point (stratified, non-overlapping,
  full coverage) is made by 2 rows as well as 4.
- Added, not in the original: an explicit "Outer test is touched exactly once,
  after search + threshold selection" callout, and a pointer to
  `optimization_metrics_ci.csv` as the artifact the pooled panel produces —
  both added because this figure is now wired into the manuscript (the original
  was not) and needs to connect directly to Table `tab:optimization_metrics_summary`
  in the Discussion.

## Design

Same conventions and colour roles as `nested_cv_optimization.svg` §5 (used to
fit `#2a78d6`, inner validation `#1baf7a`, outer test `#eb6834`, search/decision
accent `#4a3aa7`). Arial, discrete shapes, no `<style>`/`<defs>`/gradients.

## Regenerating

Edit the `.svg` directly, then run `./render.sh nested_cv_optimization_compact`
from this directory. See `pipeline_overview.md` for the render toolchain note
(headless Chrome + `pdftoppm`, not ImageMagick `convert`).