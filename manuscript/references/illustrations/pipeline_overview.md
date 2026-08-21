# Pipeline Overview Illustration — Build Notes

Companion document for `pipeline_overview.svg` / `.pdf`.
Generated 2026-08-21 on branch `codereviewed_methods`, as part of the Methods
rewrite recorded in `documentation/notes/methods_edit.md`.

## Sources of truth

Each of the five stage boxes traces to one `module/experiments/*.yml` config and
its driving script, per CLAUDE.md's pipeline-stage description:

| Stage | Script | Config |
|---|---|---|
| Data | `module/dataload.py` (`DPN_data`) | n/a |
| Selection | `module/selreport.py` | `bin_sel_final_202608.yml` |
| Optimization | `module/optreport.py` | `bin_opt_final_202608.yml` |
| Explainability | `module/expreport.py` | `bin_exp_final_202608.yml` |
| Counterfactuals | `module/cfreports.py` | `bin_cf_final_202608.yml` |

The "14 algorithms × 12 feature sets" and "40 runs" figures on the Selection box,
and the "outer 4×10 / inner k=3" figures on the Optimization box, are the same
numbers verified for `nested_cv_optimization_compact.md`. The "4 persisted fold
models" note on Optimization/Explainability/Counterfactuals reflects
`expreport.py:104-112` and `cfreports.py:241-265`, which both load
`first_repeat_trained_models_filename` from the optimization stage rather than
refitting from scratch.

## Design

Same conventions as `nested_cv_optimization.svg` (Arial, discrete
`<rect>/<circle>/<line>/<polygon>` shapes, no `<style>`/`<defs>`/gradients, so the
file imports cleanly into Canva). Canvas `1600 x 440`. The optimization box is
outlined in the search-accent colour (`#4a3aa7`) to mark it as the one stage that
performs a hyperparameter search; the NCS-exclusion note on the Data box uses the
outer-test orange (`#eb6834`) to flag it as an exclusion/caveat, consistent with
that colour's role elsewhere.

## Regenerating

Edit the `.svg` directly, then run `./render.sh pipeline_overview` from this
directory (headless Chrome → vector PDF, `pdftoppm` → PNG for a visual QA pass —
delete the `_qa-1.png` before committing). Requires `google-chrome` and
`pdftoppm` (poppler-utils); ImageMagick's `convert` cannot rasterize PDFs in this
environment (its PDF coder is blocked by policy), which is why `pdftoppm` is used
for the QA step instead of the `convert -density 96` approach documented in
`nested_cv_optimization.md`.