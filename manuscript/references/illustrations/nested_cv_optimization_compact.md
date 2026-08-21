# Nested CV Optimization (Compact) Illustration — Build Notes

Companion document for `nested_cv_optimization_compact.svg` / `.pdf`.
Generated 2026-08-21 on branch `codereviewed_methods`, as part of the Methods
rewrite recorded in `documentation/notes/methods_edit.md`.

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