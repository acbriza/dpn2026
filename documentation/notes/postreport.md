# Outputs of the postreport stage

Reference for everything `module/postreports.py` writes, what each file contains, and which
manuscript float consumes it. There was no such note before; the stage postdates the
`*_refactor.md` records kept for the other stages, and unlike them this is a description of
the artifacts rather than a session log.

```
cd module
python postreports.py bin_postreport_final_202608.yml
```

The stage takes no `overwrite` argument and no CPU count: it reads already-generated
artifacts from the selection and counterfactual runs named in its config and rewrites its
whole output tree every run.

## What the stage is for

Neither `selreport.py` nor `cfreports.py` produces the cross-cutting views the manuscript
needs. `selreport.py` writes one summary table per metric, not the single algorithm x metric
table; `cfreports.py` writes one report per patient instance, not the cross-patient
aggregates. `postreports.py` reads both stages' per-run outputs and combines them. It fits
no model and generates no counterfactual — every number it emits is a re-presentation of
something an earlier stage already computed, so a discrepancy between this stage and an
earlier one is always a bug here or a stale input, never a new result.

## Output layout

```
module/experiments/binary/postreport/<tag>/
    bin_postreport_<tag>.yml              copy of the config that produced the run
    selection/
        selection_metrics_summary.csv
        selection_metrics_summary.latex
    counterfactuals/
        cf_fulltable.{csv,latex}
        ioi_summary_per_model.{csv,latex}
        cf_changed_features.{csv,latex}
        global_cf_counts.png
        global_cf_percent.png
        case_study_counterfactuals.png
        remaining_counterfactuals.png
```

Every `.csv`/`.latex` pair holds the same numbers; the CSV is for reading and re-checking,
the `.latex` is what gets pasted into the manuscript.

## The files

| File | Written by | Manuscript float |
|---|---|---|
| `selection_metrics_summary.*` | `consolidate_selection` | Table~`tab:selection_metrics` |
| `cf_fulltable.*` | `consolidate_counterfactuals` | Table~`tab:localcf-listing` |
| `ioi_summary_per_model.*` | `consolidate_counterfactuals` | Table~`tab:localcf-model-level` |
| `cf_changed_features.*` | `consolidate_counterfactuals` | Table~`tab:localcf-changed` |
| `global_cf_counts.png` | `consolidate_counterfactuals` | Figure~`fig:globalcf` |
| `global_cf_percent.png` | `consolidate_counterfactuals` | — (not used) |
| `case_study_counterfactuals.png` | `plot_case_study_counterfactuals` | Figure~`fig:localcf-cases` |
| `remaining_counterfactuals.png` | `plot_remaining_counterfactuals` | Figure~`fig:s5` (Appendix) |

### `selection_metrics_summary`

One row per algorithm, one column per metric in `selection.metrics`, each cell
`mean ± std` over the repeated stratified k-fold runs, sorted by AUPRC descending. Read from
the selection stage's `<feature_set>_benchmarking_stats.joblib` (`stats['mean']` and
`stats['std']`), for the single feature set named in `selection.feature_set`.

The `.latex` is a `sidewaystable`, not a `tabular`: at three decimals of mean ± std over
eight metrics the table is about 188pt wider than the manuscript's 372pt text width at
`\footnotesize`, and rotating it is the only arrangement confirmed to fit. It also carries
its own `\caption` and `\label{tab:selection_metrics_meanstd}`; `main.tex` keeps the body and
supplies its own caption and `\label{tab:selection_metrics}`, so the generated label is not
the one to search for in the manuscript.

### `cf_fulltable`

One row per patient instance that produced counterfactuals (8 of the 61 candidates in the
final run): `Model` (which fold's held-out set the patient was in), `Patient Code`,
`Probability`, `Margin` (absolute distance from that fold's own tuned threshold), `Outcome`
(TP/FP/TN/FN), `CF Count`, and the mean `Sparsity`/`L1`/`L2` over that patient's
counterfactuals.

### `ioi_summary_per_model`

One row per fold: `Candidates`, `Misclassified`, `Borderline (correct)`,
`Instances with CF`, and the mean sparsity/L1/L2 over that fold's instances that produced
counterfactuals. `Misclassified` and `Borderline (correct)` partition `Candidates` — a
candidate that was not misclassified can only have qualified via the borderline rule — so
`Borderline (correct)` is *not* the count of all low-confidence candidates, since a
misclassified candidate can be borderline too.

Model 2 appears with zero instances: it produced no successful counterfactuals.

### `cf_changed_features`

One row per instance giving, per actionable feature, how many of that instance's
counterfactuals altered it, plus a `Total` row. Only the six actionable features
(`INSULIN, HBA1C, HPN, PAOD, DSLPDMIA, CKD`) appear; NCS features are excluded from this
stage as they are from every other.

### `global_cf_counts.png` / `global_cf_percent.png`

Feature x model heatmaps of the same counts, raw and as a fraction of that model's
counterfactuals. Only the counts version is used in the manuscript. The percent version is
still written because the fraction is what makes models with very different counterfactual
volumes comparable, and it is the quicker of the two to read when checking a claim about
which feature dominates a fold.

### `case_study_counterfactuals.png` / `remaining_counterfactuals.png`

Both come from `_cf_panel_figure`, one panel per patient: HbA1c as a signed magnitude on a
quantitative axis shared across the panels of that figure, binary features as up/down
direction glyphs. Rows within a panel are ordered by number of features changed, then by
size of the HbA1c change.

The two figures partition the successful instances. `case_studies` in the config names the
patients the main text discusses individually; `plot_remaining_counterfactuals` takes every
other instance from the instance table, so moving a patient into or out of `case_studies`
moves it between the two figures rather than dropping it from both or showing it twice.

Three properties worth knowing before editing either:

- **The HbA1c axis is shared within a figure, not between them.** Magnitudes are comparable
  across panels of one figure only. Both captions say so.
- **The appendix figure is drawn tighter** (`row_height=0.048`, `hspace=0.75`) so five panels
  plus a caption fit one text height. `hspace` is a fraction of mean panel height, so
  shortening the rows without raising it collapses the gap between a panel's x-axis label and
  the next panel's title, which do not shrink with the panels.
- **Figure width in `main.tex` is set by what fits, not by taste.** At `0.89\textwidth` the
  appendix figure plus caption overran the text block by 71pt (`Float too large for page`);
  `0.83\textwidth` clears it. A longer caption or a taller figure needs that number
  rechecked — the warning is only a warning, and the page still typesets.

## Cross-stage checks this stage performs

Two assumptions reach across file formats written by other stages, and both are checked
rather than trusted:

- **Threshold recovery.** The per-fold threshold is not stored next to an instance, but
  margin is its absolute distance from the prediction, so it is recoverable as
  `pred_proba ∓ margin`. `_load_fold_thresholds` reads the optimization stage's published
  per-fold thresholds — anchored on the trained-models path the counterfactual stage was
  actually pointed at — and the figure code raises if the recovered value disagrees.
- **Borderline rule.** `borderline_delta` is not restated in this stage's config;
  `_load_cf_stage_config` reads `dice.threshold_delta` back from the copy of its own config
  that the counterfactual stage left in its output directory. Re-running that stage with a
  different value therefore changes the labels here instead of silently disagreeing with
  them.

## The query-instance row

`generate_diverse_cfs` pools its counterfactuals into a frame whose first row is the query
instance itself, and that row survives into `*_local_cf_distances.csv` as `cf_row 0` with
sparsity/L1/L2 all zero. It is not a counterfactual. `_read_cf_distances` drops it here
rather than at the source, so the per-patient artifacts keep showing the baseline they are
read against.

This was a real defect, not a precaution: before it was fixed every published CF Count was
one too high and every distance mean was pulled 4–8% toward zero. Any number carried over
from an older draft of the manuscript should be checked against a current `cf_fulltable.csv`
rather than trusted.

## Getting the outputs into the manuscript

`manuscript/references/postreport/` mirrors the stage's output tree and is a **manual copy** —
nothing copies it automatically, and the manuscript compiles against whatever was last
copied there. After re-running the stage, copy the files the manuscript actually includes and
re-paste the table bodies; `main.tex` holds table bodies inline, so a regenerated `.latex`
changes nothing until it is pasted in. Note that `module/experiments/` is gitignored, so the
copy under `manuscript/references/` is the only version of these outputs in version control.
