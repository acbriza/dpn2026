# Detailed Pipeline Illustration (cleaned pipeline) — Build Notes

Companion document for `detailed_pipeline_cleaned.svg`. Generated 2026-09-04 on branch
`codereviewed`. Derived from `detailed_pipeline.svg` by editing panel 2 only; panels 1, 3
and 4 are byte-identical apart from the legend re-spacing described below.

## When this file is the right one to ship

`detailed_pipeline.svg` describes the pipeline as it stands before the threshold-selection
fix. This file describes it **after**, and is only correct if the fix adopted includes the
separate threshold cross-validation (`threshold_cv` at `current_seed + 1000`, matching the
precedent already in `train_final_model`, `utils2/optimization.py:461-465`).

- Fix **without** `threshold_cv` (drop `early_stopping_rounds`, de-leak the Optuna
  objective, nothing else): **`detailed_pipeline.svg` needs no change at all** and should
  be kept. Every claim it makes stays true, and box 2's "One held-out score per patient"
  becomes true for the first time. Do not ship this file in that case.
- Fix **with** `threshold_cv`: `detailed_pipeline.svg` becomes wrong at one label and
  incomplete at one panel (see below). Ship this file.

Canvas is unchanged at 650&times;898, so the sizing table in `detailed_pipeline.md` still
applies: 514pt on the page (93%), body text 7.4pt at `\textwidth`, ~39pt of caption budget.
Swapping the two files in `main.tex` is a one-line `\includegraphics` change with no
reflow.

## What changed, and why

### 1. Panel 2 now shows both partitions of the training set

`detailed_pipeline.svg` draws one 3-fold split of the training set and lets it serve two
consumers (the Optuna objective and the out-of-fold scores). Under the fix those are two
independent partitions with different seeds, so the panel draws both:

- Title: "Which training patients tune the model" &rarr; "Which training patients tune the
  model, and which set its threshold" (font 14.5 &rarr; 14 to fit the panel width).
- Subtitle: "Stratified K-Fold: 3-fold, training set only" &rarr; "Stratified K-Fold
  &times;2 &middot; the training set is split 3-fold twice, with different seeds".
- The `training set` bar now spans the full panel width (582px) and feeds **two** arrows,
  one into each group, so the two partitions visibly come from the same patients.
- Two 3-fold groups, each 3 tiles &times; 88px, left at x=34, right at x=344, with a 38px
  gutter between them.
- Group captions name the consumer: "tuning folds &rarr; objective below" and
  "threshold folds &rarr; step 2 below". The Optuna box still spans the full width, so the
  captions plus the box's own "across the 3 tuning folds" line are what disambiguate which
  group it consumes.

Two deliberate departures from `detailed_pipeline.md`'s content decisions:

- **The panel-1 / panel-2 column grid is no longer shared.** `detailed_pipeline.md` records
  that the training-set bar was made exactly as long as the three tuning folds (332px, on
  panel 1's 108px grid) "so the two panels read as one partition of the same cohort". With
  two groups side by side, tiles had to shrink to 88px and the bar had to span both. The
  grid alignment was traded for a worse misreading: unequal group widths (an earlier draft
  used 108px left, 74px right) implied the threshold folds hold fewer patients than the
  tuning folds. They hold the same patients, split differently. Equal widths under one
  full-width bar states that; grid alignment does not.
- **The gutter is 38px, not the 4px used inside a group.** At an earlier 14px the six tiles
  in a row read as a single 6-fold partition.

### 2. The carried-forward label was falsified by the fix

`detailed_pipeline.svg` labels the panel 2 &rarr; panel 3 connector "carried forward:
winning hyperparameters + **the same 3 tuning folds**". Under the fix the threshold loop
draws its own splits, so this is now "carried forward: winning hyperparameters + the 3
threshold folds". This is the one label in the original that becomes actively wrong rather
than merely incomplete.

### 3. Legend

"tuning validation" &rarr; "held out within training", because the green tiles now cover
two roles (tuning validation and threshold scoring) that are both held-out-within-training
slices. `val` on the tiles still abbreviates it. The four legend items were re-spaced
(swatches at x=20 / 118 / 298 / 448) to fit the longer label without collision. No fifth
colour was introduced.

## What did **not** change, and why

- **Early stopping is nowhere in either figure.** `detailed_pipeline.md` records that the
  constant-hyperparameter footer strip was dropped and only the four *searched*
  hyperparameters kept, so removing `early_stopping_rounds` from `param_space_fn` is
  invisible here. Nothing to edit.
- **The Optuna box's "Tunes:" line is unchanged.** If `iterations` is added to the search
  space (an open decision), append "&middot; boosting rounds" to that line.
- **Panel 3 is untouched.** "2 &middot; Out-of-fold scores / One held-out score per
  patient" and "3 &middot; Decision threshold / Maximize F1 on the OOF curve" are both
  already accurate descriptions of the fixed code — the figure was a specification the
  code failed to meet, not a misdescription of intent.
- **Panels 1 and 4 are untouched.** No counts appear in either figure (they were removed
  deliberately), so none of the numbers that the re-run will move are drawn anywhere.

## Manuscript caption

`fig:pipeline_overview`'s caption (`main.tex`, ~line 443) says "within each, a 3-fold split
of the training set drives a 100-trial Optuna TPE search. The winning configuration is refit
on the full training set, thresholded on the out-of-fold precision--recall curve". That
remains true but no longer complete — it should name the second split, e.g. "&hellip;
thresholded on out-of-fold probabilities from a second, independently seeded 3-fold split of
the same training set". The rest of the caption stands.

## Design

Inherits every colour role and constraint from `detailed_pipeline.md` §Design: Arial,
discrete shapes, no `<style>`/`<defs>`/gradients, numeric entities only.

## Regenerating

```
./render.sh detailed_pipeline_cleaned
```
