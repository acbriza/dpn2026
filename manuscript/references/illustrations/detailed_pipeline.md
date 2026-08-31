# Detailed Pipeline Illustration — Build Notes

Companion document for `detailed_pipeline.svg` (single full-page version) and
`detailed_pipeline_part1.svg` / `detailed_pipeline_part2.svg` (two-figure split).
Generated 2026-08-31 on branch `codereviewed`.

Both versions carry the same content. Pick one; do not use both.

## Purpose

Consolidates the whole model lifecycle into one figure: how CatBoost is trained
(nested CV), tuned (Optuna), thresholded, evaluated, and then reused for feature
importance / SHAP and for DiCE counterfactuals. Derived from
`nested_cv_optimization_compact.svg` but not a resize of it — panels were rebuilt.

## Sizing (single-column sn-jnl, `\textwidth`=372pt, `\textheight`=553pt)

Rendered text size is `px_size × 372 / canvas_width_px` at `width=\textwidth`.

| File | Canvas | Height on page | Body text | Caption budget |
|---|---|---|---|---|
| `detailed_pipeline.svg` | 650×898 | 514pt (93% page) | 7.4pt | ~39pt ≈ 4 lines |
| `detailed_pipeline_part1.svg` | 540×698 | 481pt (87% page) | 9.0pt | ~72pt ≈ 7 lines |
| `detailed_pipeline_part2.svg` | 540×436 | 300pt (54% page) | 9.0pt | ample |

**The manuscript ships the single-page version**, as `fig:pipeline_overview`
(`main.tex`, `\begin{figure}[p]` at `width=0.95\textwidth`) — body text 7.1pt,
graphic ~488pt, leaving ~65pt of caption. The current 8-line caption fits with
slack. The two-figure split is kept built and maintained as an alternative if the
single page ever has to give ground, but is not referenced by `main.tex`.

`nested_cv_optimization_compact.svg` is **retained for reference only and is not
used in the manuscript** — no `\ref` or `\includegraphics` points at it. It was
never re-sized: at a 1600px canvas with 11.5px text it would render at **2.7pt**
in this column, far below the ~6pt floor, so it must not be dropped into the
single-column layout as-is.

## Content decisions (this session)

- **Panels renamed away from code vocabulary**: "Outer loop —
  RepeatedStratifiedKFold" → "Which patients are held out"; "Inner loop — per
  outer iteration" → "Which training patients tune the model".
- **The outer/inner vocabulary was retired everywhere**, in the figures *and* in
  `main.tex`, on the grounds that it is sklearn's internal naming rather than
  something a clinical reader can decode. The two must stay in sync — changing one
  alone desynchronizes the figure from the Methods text that describes it.

  | was | is now |
  |---|---|
  | outer iteration | train/test split |
  | outer-training set / outer training fold | training set |
  | outer test fold | test set |
  | inner splits / inner folds | tuning folds |
  | inner validation | tuning validation |
  | outer-fold refits | fold models |

  `main.tex` now contains zero occurrences of "outer" or "inner" in this sense.
  Two things deliberately survive the rename: the splitter names in the panel
  subtitles (`Repeated Stratified K-Fold`, `Stratified K-Fold`, "4 folds", "3-fold")
  because those are the sklearn classes actually used and are worth naming; and
  "fold model" / "per fold" for the four saved models of repeat 1, which the
  manuscript also keeps. The inner-fold tiles stay labelled `val` — a normal
  abbreviation of the legend's "tuning validation".
- **All four fold rows** (outer) and **all three fold rows** (inner) are drawn,
  rather than 2 rows plus a caption noting the rest.
- **Panel subtitles name the sklearn splitter** (`Repeated Stratified K-Fold`,
  `Stratified K-Fold`) on their own line under each panel title; they no longer fit
  inline beside the title.
- **The inner panel's `outer-training set` bar is exactly as long as the three
  training folds above it** (3 tiles + 2 gaps), left-aligned to the same column
  grid, so the two panels read as one partition of the same cohort.
- **Per-split sample counts removed** (`n=140–141`, `n=46–47`, `n=93–94`, the
  "≈46–47 of the 187 patients" line, and the "touched exactly once" callout).
  The top bar in the inner panel keeps the label `outer-training set`; only the
  count was dropped, matching every other count removal.
- **Panel 3 uses the corrected dependency graph**: step 1 (refit) and step 2
  (OOF scores) are *parallel* branches off the search result — step 2 builds
  fresh models from `best_params`
  ([optimization.py:180](../../../module/utils2/optimization.py)), not from step
  1's fitted object. Steps 1 and 3 converge on step 4. The old 1→2→3→4 chain in
  `nested_cv_optimization_compact.svg` asserted a dependency that does not exist.
- **One arrow per panel transition, drawn in the gutter between panels** (x=40),
  never inside them. Panel 3's boxes carry no inbound arrows of their own: the
  numbering `1 · 2 · 3 · 4` plus the box2→box3 and 1,3→4 arrows already state the
  dependency graph, and an extra fan-out bracket into boxes 1 and 2 read as a
  bidirectional link between them. Because the gutter is now outside every panel,
  all panel titles sit at x=34 -- no indent special case.
- **The "carried forward: winning hyperparameters + the same 3 inner splits" label
  sits beside the panel 2 → panel 3 arrow**, not inside panel 3. It describes what
  crosses the boundary, so it belongs on the boundary.
- **Feature importance/SHAP and counterfactuals are nested inside one
  "Explaining the model" panel** rather than being two siblings fed by a forked
  spine. They are parallel consumers of the same saved models, so a single arrow
  into a container states that more cleanly than two arrows into two panels — and
  it removes the left-margin spine that previously had to route around panel 4 to
  reach panel 5. Nesting is shown by fill depth: wrapper `#eeede8`, sub-panels
  `#f9f9f7`, content boxes white.
- **part2 carries no colour key.** Its content uses only the magenta/teal
  explainability colours; the four train/val/test/search chips belong to part1's
  fold tiles and were orphaned decoration in part2.
- **Dropped**: the "What gets reported" panel, the pooled 40-iteration CI line,
  and the constant-hyperparameter footer strip. The four *searched*
  hyperparameter names moved inline into the Optuna box.
- **Kept**: only the four-colour key, on one row under the title.

## Verified facts behind the two new panels

- Both explainability and counterfactuals load the same
  `catboost_first_repeat_trained_models.joblib`, written only for `fold_idx < 4`
  (`optimization.py`), with an assertion that those 4 test folds tile the dataset.
- **Feature importance / SHAP refits.** `expreport.py` rebuilds each fold model
  from stored `best_params` with `random_seed` restored — omitting the seed
  drifts predictions by up to 0.086. Purpose: attach feature names, since the
  optimization stage fit on `X_train.values`. SHAP runs over each fold's own
  `X_test`.
- **Counterfactuals do not refit.** `cfreports.py` uses
  `split_results[midx]['model']` directly; a refit with `cat_features` was
  measured to move probabilities by up to 0.113 and flip up to 3 labels, which
  would change which patients are selected as instances of interest. The stored
  model is wrapped in `CatBoostWrapper(model, threshold)` so DiCE decides at the
  tuned cut-off rather than 0.5.
- Instances of interest = misclassified **or** `|p − threshold| ≤ 0.08`
  (`threshold_delta` in `bin_cf_final_202608.yml`).
- `features_to_vary` is everything except NCS columns and the configured
  unactionable list — 6 actionable features.

## Design

Same colour roles as `nested_cv_optimization.svg` §5 (`#2a78d6` used to fit,
`#1baf7a` inner validation, `#eb6834` outer test, `#4a3aa7` search/decision).
The two downstream panels (feature importance / SHAP, counterfactuals) deliberately
use colours **outside** the four data-role hues, because their boxes denote analysis
stages rather than train/validation/test partitions: magenta `#a3197a` for feature
importance and SHAP, teal `#0b7285` for counterfactuals. Do not reuse the legend
colours there.

Arial, discrete shapes, no `<style>`/`<defs>`/gradients, numeric entities only.

## Regenerating

```
./render.sh detailed_pipeline
./render.sh detailed_pipeline_part1
./render.sh detailed_pipeline_part2
```
