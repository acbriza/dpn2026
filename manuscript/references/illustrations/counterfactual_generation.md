# Counterfactual Generation Illustration — Build Notes

Companion document for `counterfactual_generation.svg` / `.pdf`.
Generated 2026-08-21 on branch `codereviewed_methods`, as part of the Methods
rewrite recorded in `documentation/notes/methods_edit.md`.

## Sources of truth

| What | Source |
|---|---|
| Instance-of-interest selection (misclassified ∪ borderline, `delta`) | `module/utils2/counterfactuals.py:492-524` (`get_instances_of_interest`) |
| Per-split, fold-specific threshold passed into instance selection | `module/cfreports.py:241-289` |
| Global / local permitted-range formulas | `module/utils2/counterfactuals.py:152-177` (`get_global_permitted_range`) and `:420-461` (`get_local_permitted_range`) |
| DiCE Data/Model/Dice setup, `dataframe=dfXy_test` | `module/cfreports.py:276-282` |
| Local CF generation parameters (`total_CFs`, weights, sparsity) | `module/experiments/bin_cf_final_202608.yml` (`dice.local_cf`) |
| Actionable feature list | `module/experiments/bin_cf_final_202608.yml` (`dice.cf_features.actionable`) |
| Sufficiency / necessity check semantics | `module/utils2/counterfactuals.py:1161-1220` (`check_sufficiency`, `check_necessity`) |

Two corrections made here relative to the manuscript text as it stood before this
session (both fixed in the same pass in `main.tex`, not only in this figure):
`diversity_weight` is `1.0` for local CF generation (not `0.1`, which is
`categorical_penalty`), and the borderline window is `threshold_delta = 0.08`
around *that fold's own* tuned threshold (not "±0.2 of the mean threshold" —
there is no single mean threshold; each of the four persisted fold models keeps
its own).

The permitted-range formulas use `max(0, x - σ)`, not `min(0, x - σ)` as the
pre-existing manuscript text stated: the code floors the lower bound at 0 (so a
naturally non-negative quantity like HbA1c is never pushed negative by the `-σ`
margin), which is what `max(0, ...)` computes; `min(0, ...)` would instead have
allowed negative values whenever `x - σ < 0`.

## Design

Same conventions as the other two new figures (Arial, discrete shapes, no
`<style>`/`<defs>`/gradients). Canvas `1600 x 560`, four sequential panels.
Panel 3 (DiCE genetic search) is outlined in the search-accent colour (`#4a3aa7`),
consistent with its role in `nested_cv_optimization_compact.svg`. Panel 4's three
output boxes use the inner-validation green (`#1baf7a`) to mark them as
*produced* artefacts, distinct from the orange (`#eb6834`) used for the
held-out/exclusion criteria in panel 1.

## Regenerating

Edit the `.svg` directly, then run `./render.sh counterfactual_generation` from
this directory. See `pipeline_overview.md` for the render toolchain note
(headless Chrome + `pdftoppm`, not ImageMagick `convert`).