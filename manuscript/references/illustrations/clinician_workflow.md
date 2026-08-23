# Clinician Workflow Illustration — Build Notes

Companion document for `clinician_workflow.svg` / `.pdf` / `.png`.
Generated 2026-08-22, as part of the Discussion-section rewrite recorded in
`manuscript/references/prompts/mss_edits.md` ("Edit for main.tex", item 4: suggest how
clinicians could use this pipeline and illustrate it).

## Sources of truth

| Step | Content | Source |
|---|---|---|
| 1. Routine Clinic Visit | Feature groups collected (profile, comorbidities, MNSI, neuro exam, Sudoscan); NCS explicitly not required | `module/dataload.py` (`DPN_data`) — same feature-group split documented in `pipeline_overview.md` |
| 2. CatBoost Risk Score | Predicted probability vs. the model's own fold-calibrated threshold | `module/expreport.py` — retrains and evaluates the 4 persisted fold models from the optimization stage |
| 3. Review Counterfactual | DiCE genetic search over the 6 actionable features (HbA1c, insulin, HPN, PAOD, dyslipidemia, CKD); "smallest realistic change" framing | `module/cfreports.py` / `module/experiments/bin_cf_final_202608.yml` (`dice.cf_features.actionable`) |
| 4. Clinician Decision | Two branches (high-confidence → NCS referral priority; borderline + actionable CF → risk-factor management + re-screen) | Derived from the case-study framing in the Discussion's Counterfactual Analysis section (Patients 20, 40, 123), not from a separate script — this is the proposed usage pattern, not a pipeline output |

The decision-support callout is a direct restatement of the Discussion's explicit framing
(decision-support research, not a deployment-ready tool; proof-of-concept status; single-clinic
n=187; no external validation or prospective testing) — see the Limitations subsection of
`manuscript/main.tex`.

## Design

Same conventions as `pipeline_overview.svg` (Arial, discrete `<rect>/<circle>/<line>/<polygon>`
shapes, no `<style>`/`<defs>`/gradients, so the file imports cleanly into Canva). Canvas
`1600 x 560`. The counterfactual-review box (Step 3) is outlined in the same purple accent
(`#4a3aa7`) `pipeline_overview.svg` uses to mark its one "special" stage, since counterfactual
review is this pipeline's central contribution. The decision-support disclaimer uses a filled
orange callout (border `#eb6834`, fill `#fdf1ea`) rather than the thin orange caption text
`pipeline_overview.svg` uses for its NCS-exclusion note, because this message is a load-bearing
caveat for the whole illustration, not a secondary annotation, and needed more visual weight.

## Regenerating

Edit the `.svg` directly, then run `./render.sh clinician_workflow` from this directory
(headless Chrome → vector PDF, `pdftoppm` → PNG). Requires `google-chrome` and `pdftoppm`
(poppler-utils). Unlike the other three illustrations in this folder, the PNG here is a
committed deliverable (the manuscript's `\includegraphics` in the clinician-workflow paragraph
points at `clinician_workflow.png`), not a QA-only artifact — after regenerating, rename
`clinician_workflow_qa-1.png` to `clinician_workflow.png` rather than deleting it.
