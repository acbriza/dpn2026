# P2 — Figure Integrity Audit

**Scope:** every `figure`/`figure*` environment in `manuscript/main.tex`, audited against the
artifacts it claims to display.
**Date:** 2026-09-01 · **Branch:** `codereviewed` · **Mode:** report only, no manuscript edits.

## Scope correction

The brief anticipated "~16 figure environments in the main text plus supplementary ones in the
appendix". The actual manuscript contains **15 figure environments carrying 16 images**:
7 in the main text, 8 in the appendix. Figure 4 (`fig:explainability`) is a single environment
holding two `subfigure` panels, which accounts for the 15-vs-16 difference.

The brief also referred to canonical outputs under `module/experiments/binary/<stage>/...`.
The real layout inserts a model code and/or a tag level below the stage
(`binary/<stage>/[<model_code>/]<tag>/`), with `binary/eda/` writing flat with no tag directory.
Canonical locations were resolved from the stage configs
(`module/experiments/bin_*_final_202608.yml`, all `tag: final_202608`) rather than assumed.

---

## Headline result

- **No BLOCKERS.** All 16 `\includegraphics` paths resolve, and **all 14 pipeline-derived
  figures are byte-identical (MD5) to the canonical `final_202608` outputs.** No figure is
  from a superseded run.
- **3 FAIL findings**, all caption-vs-source defects, none requiring a re-run:
  one caption contradicts a number printed inside its own figure, one states a numeric bound
  that its own data violates, one misdescribes how the figure is sorted.
- **5 advisory findings** (rounding, uncited labels, an over-strong verbal claim).

---

## FAIL 1 — Fig 14 `fig:catboost_auprc`: caption std contradicts the number printed in the figure
**[WONTFIX 2026-09-01 — author reviewed and retained the caption as written.]**

The caption states the mean-of-folds AUPRC is **0.893 ± 0.043**. The annotation box drawn
*inside* `catboost_pooled_auprc.png` reads **`Mean & Std : 0.893 ± 0.037`**.

Recomputed from `references/hyperparameter_optimization/catboost_first_repeat_optimization_metrics.csv`
(per-fold AUPRC = 0.9008, 0.8916, 0.8372, 0.9408):

| Quantity | Value | Where it appears |
|---|---|---|
| mean | 0.8926 → **0.893** | caption **and** figure — agree |
| std, `ddof=1` (sample) | 0.0426 → **0.043** | caption + Table 8 |
| std, `ddof=0` (population) | 0.0369 → **0.037** | **printed inside the figure** |

Both values are arithmetically correct; they use different `ddof`. The defect is that a reader
comparing the caption to the image sees two different numbers for one labelled quantity. The
plotting code and the table must be reconciled to one convention.

The pooled statistics in the same caption are **correct**: recomputing average precision from
`catboost_pooled_auprc.csv` gives **0.857729 → 0.858**, and the figure's own box reads
`AUPRC : 0.858 [0.795–0.916]`, matching the caption's CI exactly. (The CSV omits sklearn's
terminal `(recall=0, precision=1)` point; adding it back reproduces 0.858 exactly. Without it a
naive recomputation yields 0.850 — noted so a later auditor does not misread this as an error.)

The no-skill baseline is **exact**: precision at recall = 1.0 in the CSV is
`0.6951871657754011` = **130/187**, matching the caption's "prevalence = 0.70, i.e., 130/187"
and the figure's dashed-line legend.

## FAIL 2 — Fig 15 `fig:s5`: "within 0.07" is violated by Patient 172
**[RESOLVED 2026-09-01 — bound corrected to 0.08 in main.tex]**

Caption: *"All five lie within 0.07 of their fold's decision threshold."*

From `references/postreport/counterfactuals/cf_fulltable.csv`:

| Panel | Patient | Margin | Within 0.07? |
|---|---|---|---|
| a | 29 | 0.0450 | yes |
| b | 76 | 0.0650 | yes |
| c | 125 | 0.0264 | yes |
| d | 23 | 0.0662 | yes |
| e | **172** | **0.0747** | **no** |

The figure contradicts its own caption on its face: panel (e) is headed
`p = 0.567, threshold 0.492` → 0.075. The correct bound is **0.08**, which is also the
candidate-selection criterion stated in the methods (δ = 0.08) and drawn in the pipeline
diagram (`|p − threshold| ≤ 0.08`). Changing "0.07" to "0.08" fixes it.

Every other claim in this caption verifies:
outcomes (29 TP, 76 TP, **125 FP**, 23 TP, 172 TP) match the "all correctly classified except
Patient 125" claim; HbA1c changed in all CFs for 29/125/23/172 and **13 of 21** for Patient 76;
dyslipidemia **20 of 21** for Patient 76; insulin in **all 17** for Patient 29 and the INSULIN
columns for Patients 125 and 172 are visibly **empty** (0 and 0 in the table).

## FAIL 3 — Fig 8 `fig:univariable_discrimination`: caption misstates the sort key
**[RESOLVED 2026-09-01 — caption corrected in main.tex; confirmed against source code]**

Caption: *"Univariable discrimination of each candidate predictor, **ranked by area under the
receiver operating characteristic curve**…"*

The figure is **not** ranked by AUC. It is ranked by discrimination *strength*, |AUC − 0.5| —
the `strength` column of `references/eda/feature_statistics.csv`. Verified programmatically:

```
plotted order == sorted by |AUC-0.5| (strength) : True
plotted order == sorted by AUC descending       : False
  AUC-desc would start: DEC_AR, DEC_VS, DM_DUR, MNSI, DEC_PPS
  figure actually starts: DEC_AR, DEC_VS, NS,   DM_DUR, FEET_MEAN_ESC
```

**Confirmed from the code that produced the figure**, at the user's request:

- `module/edareport.py:432` — `stats["strength"] = (stats["auc"] - 0.5).abs()`
- `module/edareport.py:838` (docstring) — *"Forest of per-feature AUC, ordered by discrimination strength."*
- `module/edareport.py:840` — `modelled = stats[stats.modelled].sort_values("strength")`

The sort key is unambiguously `strength`, never `auc`. Decisively, the **pipeline's own
auto-generated caption** (`references/eda/fig2_univariable_discrimination_caption.txt`, written
by `write_caption()` at `edareport.py:911`) states it correctly: *"features are ordered by
**discrimination strength**."* The error was introduced when that caption was hand-rewritten for
the manuscript; the generator was right all along.

The tell is visible in the rendered figure: **NS sits 3rd from the top with AUC 0.281** — the
single most *inverted* predictor. Under a true AUC ranking it would be last. Likewise
FEET_MEAN_ESC (0.321) is placed 5th. The ordering is the defensible one; only the caption's
description of it is wrong.

Note the companion caption for Fig 9 (`fig:s1`) says "ordered by univariable **discrimination**",
which is accurate for the same sort key — so the two captions describe the same convention
differently, and Fig 8's wording is the incorrect one.

---

## Advisory findings

**A1 — Table 8 threshold mean rounds the wrong way.** Per-fold thresholds
(0.498627, 0.5, 0.483125, 0.492284) have mean **0.49351**, which rounds to **0.494**;
Table 8 prints **0.493** (truncation). Std 0.0077 → 0.008 is correct. Every other cell in
Table 8 was recomputed and is correct to 3 dp (all 4 models × 13 rows). This table is cited by
the Fig 14 caption, hence its inclusion here.

**A2 — Fig 14 "exceeds across the recall range" is marginally over-strong.** At recall = 1.0
the PR curve equals the prevalence baseline by definition (precision = 130/187 = 0.6952), and
the bootstrap CI band dips *below* the dashed line for recall ≳ 0.95. "Exceeds across the
recall range" is true over the bulk of the curve but not at its right endpoint.

**A3 — Fig 10 `fig:s2` Haldane–Anscombe correction never fired.** Caption: *"a
Haldane–Anscombe correction applied where a cell count was zero"*. No cell count is zero in the
data — the sparsest are GBS (1 unconfirmed / 1 confirmed) and PAOD (1 / 7) — and
`or_corrected` is `False` for **every** feature in `feature_statistics.csv`. The phrasing is
conditional so not strictly false, but it implies to a reader that some intervals were
corrected. None were.

**A4 — Uncited figure labels.** **[RESOLVED 2026-09-01 for `fig:s4` — two explanatory
paragraphs added to the "Cohort and candidate predictors" appendix subsection, citing the figure
twice.]** `fig:s4` (the actionable-features figure, file
`s5_actionable_features.pdf`) is **defined but never `\ref`'d anywhere in main.tex**.
`fig:s2` is reachable only through the range "Figures~\ref{fig:s1}--\ref{fig:s3}".
Also note the filename/label offset: the file is named `s5_*` while its label is `fig:s4`,
because the EDA stage counts the provenance *table* as its s4. Harmless to LaTeX numbering,
but a trap for anyone matching files to figure numbers by name.
(`fig:feature_importances` and `fig:shap` are also uncited, but that is normal — they are
subfigure labels whose parent `fig:explainability` is cited and whose panels the caption
addresses as "(a)"/"(b)".)

**A5 — Fig 15's axis caveat is unnecessary.** Caption warns the HbA1c axis "is not the axis of
that figure, so magnitudes should be compared within a figure rather than across the two".
Both figures in fact span exactly −3 to +2, so the two axes coincide. The caution is
conservative rather than wrong.

---

## 1. FILE EXISTS — 16/16 PASS, no BLOCKERS

Every `\includegraphics` path resolves relative to `manuscript/`.

## 2. PROVENANCE — 14/14 pipeline figures byte-identical to canonical

MD5 comparison of each manuscript copy against its canonical `final_202608` output:

| Manuscript copy | Canonical | MD5 |
|---|---|---|
| `eda/fig1a_participant_flow.pdf` | `binary/eda/` | identical |
| `eda/fig2_univariable_discrimination.pdf` | `binary/eda/` | identical |
| `eda/s1_continuous_rainclouds.pdf` | `binary/eda/` | identical |
| `eda/s2_categorical_features.pdf` | `binary/eda/` | identical |
| `eda/s3_redundancy.pdf` | `binary/eda/` | identical |
| `eda/s5_actionable_features.pdf` | `binary/eda/` | identical |
| `selection/summaries/auprc_summary_table.png` | `binary/selection/final_202608/summaries/` | identical |
| `selection/violins/All/auprc_violin.png` | `binary/selection/final_202608/violins/All/` | identical |
| `explainability/catboost_pooled_auprc.png` | `binary/explainability/catboost/final_202608/` | identical |
| `postreport/explainability/…_feature_importances_nolegend.png` | `binary/postreport/final_202608/explainability/` | identical |
| `postreport/explainability/…_shap_nolegend.png` | `binary/postreport/final_202608/explainability/` | identical |
| `postreport/counterfactuals/case_study_counterfactuals.png` | `binary/postreport/final_202608/counterfactuals/` | identical |
| `postreport/counterfactuals/global_cf_counts.png` | `binary/postreport/final_202608/counterfactuals/` | identical |
| `postreport/counterfactuals/remaining_counterfactuals.png` | `binary/postreport/final_202608/counterfactuals/` | identical |

The two `illustrations/` PDFs (`detailed_pipeline.pdf`, `clinician_workflow_flow_compact.pdf`)
are **hand-authored diagrams, not pipeline outputs** — they have no canonical counterpart under
`module/experiments/`. Their sources (`.md` + `.svg` + `render.sh`) live beside them in
`references/illustrations/`. Provenance for these two is marked N/A rather than PASS; their
*content* was checked under §4 and passes.

Config check: `bin_postreport_final_202608.yml` draws from `tag: final_202608` for all three
upstream stages (selection / explainability / counterfactuals) and declares
`case_studies: 20,40,123` and the 6 `actionable_features`, all of which match what the figures
render.

## 3. CAPTION NUMBERS — every claim recomputed from source

**All 6 aggregate counterfactual claims verified** against `cf_changed_features.csv`
(n = 158 counterfactuals):

| Claim | Caption | Recomputed | |
|---|---|---|---|
| HbA1c changed | 144 (91.1%) | 144/158 = 91.139% | PASS |
| Insulin | 81 (51.3%) | 81/158 = 51.27% | PASS |
| Dyslipidemia | 59 (37.3%) | 59/158 = 37.34% | PASS |
| Hypertension | 49 (31.0%) | 49/158 = 31.01% | PASS |
| PAOD | 32 (20.3%) | 32/158 = 20.25% | PASS |
| CKD | 22 (13.9%) | 22/158 = 13.92% | PASS |

**All per-model counts verified** (Table 4 rows summed by model, cross-checked against the
`global_cf_counts.png` heatmap cells — see §5):
Model 0 → HbA1c **69**, insulin **43**, dyslipidemia **41**; Model 3 → HbA1c **51**,
insulin **31**; Model 1 → **24** CFs with hypertension **15** and PAOD **11**. All PASS.

**Sparsity range 2.16–2.65** — from `ioi_summary_per_model.csv`: Model 3 = 2.1616,
Model 1 = 2.5, Model 0 = 2.6510. PASS. Also PASS: Model 1 L1/L2 **3.40/2.29** (3.405/2.2917,
and it is the highest of the three), Model 3 **sparsity 2.16, L1 2.95** (2.1616/2.9541).

**Discrimination figures** — from `feature_statistics.csv`:
model AUROC **0.799** (per-fold mean 0.7994; the 40-run mean in
`optimization_metrics_ci.csv` is also 0.7985 → 0.799) PASS;
strongest single feature **DEC_AR 0.748** (0.7480) PASS;
actionable-feature AUC range **0.514–0.585** PASS;
**only INSULIN q = 0.046** below 0.05 (others 0.106, 0.114, 0.379, 0.482, 0.712) PASS.

**Pooled PR curve** — AUPRC **0.858** and baseline **0.70 = 130/187** both exact (see FAIL 1).
**Mean-of-folds 0.893 ± 0.043** — mean PASS, std FAIL (see FAIL 1).

**Candidate accounting** — `ioi_summary_per_model.csv`: 19+13+12+17 = **61** candidates,
14+11+9+8 = **42** misclassified, 5+2+3+9 = **19** borderline, 4+1+0+3 = **8** with CFs,
Model 2 = **12** candidates / **0** CFs. All PASS.

**Case-study probabilities/margins/thresholds** — Patient 20 p 0.4984 → 0.498, margin 0.0061 →
0.006, threshold 0.492; Patient 40 p 0.5129 → 0.513, margin 0.0143 → 0.014, threshold 0.499;
Patient 123 p 0.2695 → 0.269, margin 0.2305 → 0.231, threshold 0.500. All PASS, and all three
are printed in the figure's own panel headers where they match.

**Fig 3 heatmap claims** — from `summaries_latex/*/auprc_stats.tex` (CatBoost row):
All 0.8892; NoNeuro **0.8578** is the largest drop of any `No*` ablation (−0.0314, vs −0.0012
NoCol, +0.0097 NoProf, −0.0027 NoCom, +0.0078 NoMsi, −0.0021 NoSudo) PASS;
SudoNeuro 0.8954 is **0.006** from All, i.e. "within 0.01" PASS.

**Fig 4 panel (a) normalization** — the four importance columns sum to 99.99, 99.99, 99.98,
99.98 (rounding of displayed 2-dp cells), consistent with "normalized to sum to 100" PASS.

**Fig 13 / Table 7 selection claims** — from `selection_metrics_summary.csv`: RF 0.892 highest,
CatBoost 0.889, Extra Trees 0.885; CatBoost 0.913/0.512 and RF 0.916/0.529; RBF SVM highest
sensitivity 0.938 / lowest specificity 0.453 among top models; Naive 0.695 = prevalence with
sensitivity 1.000 / specificity 0.000. All PASS.

**Fig 11 `fig:s3`** — `s3_variance_inflation_factors.csv`: NS = 10.466 → **10.5**,
AGE = 7.198 → **7.2**, and **exactly 2** features exceed VIF 5 (next is FEET_MEAN_ESC at 3.21).
PASS. AGE–NS ρ = −0.88 as the strongest pair, PASS.

**Fig 1** — 201 screened → 190 enrolled → 187 analyzed, 130 Confirmed / 57 Unconfirmed, all
drawn in the figure and consistent with the provenance table's 3 dropped records (36, 46, 173).
PASS.

## 4. CAPTION DESCRIBES THE IMAGE — all 16 images opened and inspected

Every image was rendered and read (PDFs rasterized at 150 dpi via `pdftoppm`). **No sub-check
was left UNVERIFIED for want of being able to open a file.**

- **Fig 1** — box/arrow flow with the four stated counts and both exclusion annotations. PASS.
- **Fig 2 `detailed_pipeline`** — every structural claim present and in order: 4×10 = 40 splits;
  3-fold inner split; "Optuna TPE — 100 trials per train/test split"; refit on full training set;
  "Decision threshold — Maximize F1 on the OOF curve"; scored once on the held-out test set;
  "repeat 1 only: its 4 fold models and thresholds, saved and reused". The F1-on-OOF wording
  also matches the methods text at main.tex:426. PASS.
- **Fig 3** — 15 model rows × 12 feature-set columns; displayed 2-dp cells match the
  `auprc_stats.tex` means exactly for the CatBoost row (spot-checked all 12). PASS.
- **Fig 4** — panels (a) importances then (b) SHAP, in the caption's stated order. **Both
  panels are independently sorted by their own cross-fold mean** — verified by recomputing row
  means for all 22 features in each panel (both strictly descending, and the orders differ:
  DEC_VS is 5th in (a) but 2nd in (b)). **Domain color key verified feature by feature for all
  22 labels**: Sudoscan orange, profile blue, comorbidity green, neuro exam red, MNSI purple.
  PASS.
- **Fig 5 `localcf-cases`** — panels (a) Pt 20/Model 3/borderline FP, (b) Pt 40/Model 0/
  borderline TP, (c) Pt 123/Model 1/confident FN, in caption order. Shared HbA1c axis (−3…+2)
  across all three PASS. Baselines in parentheses match the narrative exactly: Pt 20
  INSULIN(1) HPN(0) PAOD(0) DSLPDMIA(1) CKD(0); Pt 40 (1)(1)(1)(1)(0); Pt 123 (0)(1)(0)(0)(0).
  **Red = decrease / blue = increase confirmed** by the legend (blue ▲ "0 → 1 introduced /
  started", red ▼ "1 → 0 resolved / stopped") *and* by the HbA1c half, where red bars run left
  (negative) and blue right (positive). Row counts on the y-axis (22, 24, 24 CFs) match Table 4.
  PASS.
- **Fig 6 `globalcf`** — heatmap over exactly Models 0, 1, 3; **Model 2 absent** as the caption
  states. PASS.
- **Fig 7 `clinician_workflow`** — **node color key verified node by node**: teal for data
  (Data Collection, Calibrated Threshold), purple for automated computation (Trained CatBoost
  Model, Generate Counterfactuals), **white diamonds** for the 3 decision points, orange for
  clinician steps (screens suggestions, targeted/general management), blue for the 4 outcome
  states. The "borderline" branch is the only edge reaching Generate Counterfactuals, and
  "way above" routes directly to High Priority NCS referral — both exactly as claimed. The 6
  actionable features listed in the node match the 6 CF columns. PASS.
- **Fig 8** — dashed no-discrimination line at 0.5 PASS; dotted line labelled "CatBoost model
  0.799" PASS; **diamond markers on exactly the 6 counterfactual-actionable features**, all
  clustered near 0.5, PASS; filled-vs-open marker fill tracks BH q < 0.05 (INSULIN filled;
  DSLPDMIA/CKD/HBA1C/PAOD/HPN/SEX/HAND_PCT_ASYM/GBS open). Sort key **FAIL** — see FAIL 3.
- **Fig 9 `fig:s1`** — 10 continuous features PASS; raincloud geometry as described (density
  right, box centre, raw observations left) PASS; per-panel AUC + q annotations PASS; ordering
  by discrimination strength PASS.
- **Fig 10 `fig:s2`** — panel **a** prevalence with CIs, panel **b** odds ratios on a log scale
  with the **dashed null line at 1**, in the stated order. PASS. 12 binary features PASS.
  **Filled markers = BH q < 0.05 verified against the q column for all 12** (filled: DEC_AR,
  DEC_VS, DEC_PPS, DEC_LTS, SUBJ, INSULIN; open: DSLPDMIA 0.106, CKD 0.114, SEX 0.230,
  PAOD 0.482, GBS 0.543, HPN 0.712) PASS. **Diamonds = counterfactual-actionable** PASS —
  5 diamonds, correctly excluding HBA1C, which is continuous and so absent from this
  binary-feature figure. GBS and PAOD do carry the widest intervals as claimed. See A3 for the
  Haldane–Anscombe wording.
- **Fig 11 `fig:s3`** — panels a (clustered correlation matrix, domain-colored ticks),
  b (AGE vs NS, ρ = −0.88), c (VIF with dashed threshold at 5, NS and AGE the only two bars
  past it, both highlighted red) — all three present and in order. PASS.
- **Fig 12 `fig:s4`** — 6 panels for the 6 actionable features; HBA1C rendered as
  density/box/raw and the 5 binary features as prevalence with CIs, exactly as the caption
  splits them; each panel annotated with AUC and q. PASS.
- **Fig 13 `auprc_violins`** — **left-to-right order matches descending mean AUPRC exactly**
  against `selection_metrics_summary.csv` (RF, CatBoost, Extra Trees, Naive Bayes, LogReg,
  RBF SVM, LDA, Linear SVM, XGBoost, GBM, LightGBM, kNN, SGD, Decision Tree, Naive). White
  median line inside the box PASS; RF highest median PASS; **CatBoost's lower tail does reach
  ≈0.745**, matching "extending toward 0.75" PASS; Naive violin degenerate at 0.695 PASS.
- **Fig 14** — dashed prevalence line at 0.70 with matching legend PASS; bootstrap 95% CI band
  PASS; in-figure AUPRC 0.858 [0.795–0.916] PASS. Std **FAIL** (FAIL 1); "exceeds across the
  recall range" advisory (A2).
- **Fig 15 `fig:s5`** — panels (a)–(e) for Patients 29, 76, 125, 23, 172 in caption order PASS;
  same legend and construction as Fig 5 PASS; INSULIN columns visibly empty for Patients 125
  and 172 PASS. "Within 0.07" **FAIL** (FAIL 2); axis-caveat advisory (A5).

**Row-ordering claim resolved (initially UNVERIFIED).** Both counterfactual figures claim rows
are *"ordered by number of features changed then by HbA1c change size"*. This is not
recoverable from the published CSVs, so it was resolved from the plotting code:
`module/postreports.py:425` — `np.lexsort((changes[:, hba1c_col], (changes != 0).sum(axis=1)))`.
`np.lexsort` keys on the **last** array first, so the primary key is the count of changed
features and the secondary key is the HbA1c change. **PASS**, with one nuance: the secondary
key is the *signed* change, not its magnitude, so "change size" is loose wording for a
signed sort.

**One genuine visual-resolution limit, resolved from data rather than assumed.** In Fig 15
panel (a) several Patient 29 rows appear to have a zero-length HbA1c bar, which would
contradict "HbA1c is changed in every counterfactual". Reading the raw
`counterfactuals/029/catboost_split0_patient029_local_cf.csv` shows the baseline is 11.11% and
the smallest counterfactual moves are to 11.1 and 11.0 — changes of 0.01 and 0.11 pp, which are
sub-pixel on a −3…+2 axis. **All 17 do change HbA1c; the caption is correct** and the apparent
contradiction is a rendering artifact.

## 5. INTERNAL CONSISTENCY ACROSS FIGURES — exact agreement, no disagreements found

**Fig 6 heatmap ↔ Table 4 ↔ aggregate percentages.** Every one of the 18 heatmap cells equals
the corresponding per-model sum of Table 4's per-patient rows:

| Feature | Model 0 (29,40,76,125) | Model 1 (123) | Model 3 (20,23,172) | Row total | Table 4 total |
|---|---|---|---|---|---|
| HBA1C | 17+24+13+15 = **69** | **24** | 16+21+14 = **51** | 144 | 144 ✓ |
| INSULIN | 17+18+8+0 = **43** | **7** | 11+20+0 = **31** | 81 | 81 ✓ |
| DSLPDMIA | 10+0+20+11 = **41** | **2** | 7+6+3 = **16** | 59 | 59 ✓ |
| HPN | 5+4+8+8 = **25** | **15** | 2+5+2 = **9** | 49 | 49 ✓ |
| PAOD | 2+6+2+3 = **13** | **11** | 1+3+4 = **8** | 32 | 32 ✓ |
| CKD | 1+2+7+1 = **11** | **1** | 3+4+3 = **10** | 22 | 22 ✓ |

CF counts likewise: 77 + 24 + 57 = **158**, matching Table 4's total and the "158
counterfactuals" in the running text.

**Fig 5 / Fig 15 per-patient row counts ↔ Table 4.** The y-axis labels ("22 CFs", "24 CFs",
"24 CFs" in Fig 5; "17", "21", "15", "21", "14" in Fig 15) match Table 4's CF Count column for
all 8 patients, and sum to 158.

**Figures ↔ narrative text.** Patient 20's 16/22, 11/22, 7/22, and 2/1/3; Patient 40's 24/24,
18/24, 6/24, 4 and 2; Patient 123's 24/24, 15, 11, 7, 2, 1 — all reconcile with both Table 4
and the marker directions drawn in Fig 5.

**Figures ↔ raw counterfactual data.** Table 4's Patient 29 row was recomputed from the raw
`catboost_split0_patient029_local_cf.csv` (baseline row excluded): HBA1C 17, INSULIN 17, HPN 5,
PAOD 2, DSLPDMIA 10, CKD 1 — an exact match to Table 4 and to Fig 15 panel (a).

**Thresholds are consistent across three places.** Model 0 = 0.499, Model 1 = 0.500,
Model 3 = 0.492 agree between Table 8, the Fig 5 panel headers, and the Fig 15 panel headers.

## 6. STALE-RUN SWEEP — no stale figures

All 16 referenced files, oldest first, with the newest file of the same stage for comparison:

| mtime | Stage | File | Newest in stage | Older? |
|---|---|---|---|---|
| 2026-08-10 21:48:29 | selection | `selection/summaries/auprc_summary_table.png` | 2026-08-10 21:48:40 | −11 s |
| 2026-08-10 21:48:40 | selection | `selection/violins/All/auprc_violin.png` | 2026-08-10 21:48:40 | newest |
| 2026-08-15 17:56:43 | explainability | `explainability/catboost_pooled_auprc.png` | 2026-08-15 17:56:43 | newest |
| 2026-08-21 13:21:44 | eda | `eda/fig1a_participant_flow.pdf` | 2026-08-21 13:21:56 | −12 s |
| 2026-08-21 13:21:45 | eda | `eda/fig2_univariable_discrimination.pdf` | ″ | −11 s |
| 2026-08-21 13:21:47 | eda | `eda/s1_continuous_rainclouds.pdf` | ″ | −9 s |
| 2026-08-21 13:21:50 | eda | `eda/s2_categorical_features.pdf` | ″ | −6 s |
| 2026-08-21 13:21:53 | eda | `eda/s3_redundancy.pdf` | ″ | −3 s |
| 2026-08-21 13:21:56 | eda | `eda/s5_actionable_features.pdf` | ″ | newest |
| 2026-08-22 14:51:58 | postreport-cf | `postreport/counterfactuals/global_cf_counts.png` | 2026-09-01 07:49:53 | **−10 days** |
| 2026-08-31 15:48:59 | illustrations | `illustrations/detailed_pipeline.pdf` | 2026-08-31 23:26:47 | −7.6 h |
| 2026-08-31 23:26:47 | illustrations | `illustrations/clinician_workflow_flow_compact.pdf` | ″ | newest |
| 2026-09-01 02:02:06 | postreport-cf | `postreport/counterfactuals/case_study_counterfactuals.png` | 2026-09-01 07:49:53 | −5.8 h |
| 2026-09-01 07:49:53 | postreport-cf | `postreport/counterfactuals/remaining_counterfactuals.png` | ″ | newest |
| 2026-09-01 18:09:26 | postreport-exp | `postreport/explainability/…_feature_importances_nolegend.png` | 2026-09-01 18:09:26 | newest |
| 2026-09-01 18:09:26 | postreport-exp | `postreport/explainability/…_shap_nolegend.png` | ″ | newest |

Within-stage spreads of seconds to hours are just per-figure write order inside a single run
and carry no staleness signal.

**The one candidate — `global_cf_counts.png`, 10 days behind its stage — is NOT stale.**
Its canonical counterpart was regenerated on 2026-09-01 07:49:42, but the two files are
**byte-identical (same MD5)**: the plot is deterministic and re-rendered to the same bytes.
Mtime alone would have flagged this as a superseded run; content comparison clears it.
This is also why MD5 rather than size/mtime lineage was used throughout §2.

---

## PASS/FAIL matrix

Legend: **N/A** = hand-authored diagram with no pipeline counterpart. Figure numbers are
sequence in `main.tex` (Fig 1–7 main text, Fig 8–15 appendix).

| # | Label | File | 1. Exists | 2. Provenance | 3. Numbers | 4. Describes image | 5. Consistency | 6. Not stale | Overall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `fig:patient_flow_classification` | `eda/fig1a_participant_flow.pdf` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 2 | `fig:pipeline_overview` | `illustrations/detailed_pipeline.pdf` | PASS | N/A | PASS | PASS | PASS | PASS | **PASS** |
| 3 | `fig:auprc_heatmaps` | `selection/summaries/auprc_summary_table.png` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 4 | `fig:explainability` (a+b) | `postreport/explainability/…_feature_importances_nolegend.png` + `…_shap_nolegend.png` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 5 | `fig:localcf-cases` | `postreport/counterfactuals/case_study_counterfactuals.png` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 6 | `fig:globalcf` | `postreport/counterfactuals/global_cf_counts.png` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 7 | `fig:clinician_workflow` | `illustrations/clinician_workflow_flow_compact.pdf` | PASS | N/A | PASS | PASS | PASS | PASS | **PASS** |
| 8 | `fig:univariable_discrimination` | `eda/fig2_univariable_discrimination.pdf` | PASS | PASS | PASS | **FAIL** (sort key) | PASS | PASS | **FAIL** |
| 9 | `fig:s1` | `eda/s1_continuous_rainclouds.pdf` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 10 | `fig:s2` | `eda/s2_categorical_features.pdf` | PASS | PASS | PASS | PASS (A3) | PASS | PASS | **PASS w/ advisory** |
| 11 | `fig:s3` | `eda/s3_redundancy.pdf` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 12 | `fig:s4` | `eda/s5_actionable_features.pdf` | PASS | PASS | PASS | PASS | PASS (A4) | PASS | **PASS w/ advisory** |
| 13 | `fig:auprc_violins` | `selection/violins/All/auprc_violin.png` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 14 | `fig:catboost_auprc` | `explainability/catboost_pooled_auprc.png` | PASS | PASS | **FAIL** (std 0.043 vs 0.037) | PASS (A2) | PASS | PASS | **FAIL** |
| 15 | `fig:s5` | `postreport/counterfactuals/remaining_counterfactuals.png` | PASS | PASS | **FAIL** (0.07 vs 0.0747) | PASS (A5) | PASS | PASS | **FAIL** |

**Totals:** 16/16 files exist · 14/14 provenance-checked figures byte-identical ·
12/15 figures fully PASS · 3 FAIL, all caption-text defects · 0 BLOCKERS · 0 stale figures ·
0 sub-checks left UNVERIFIED.

## Disposition of recommended fixes (as of 2026-09-01)

| # | Item | Status |
|---|---|---|
| 1 | **Fig 14** std convention (0.043 vs 0.037) | **WONTFIX** — author retained the caption as written |
| 2 | **Fig 15** "within 0.07" → "within 0.08" | **APPLIED** |
| 3 | **Fig 8** sort-key wording → "ranked by univariable discrimination, $\|\mathrm{AUC}-0.5\|$" | **APPLIED** (verified against `edareport.py`) |
| 4 | **Table 8** (`tab:catboost_metrics_firstrepeat`) threshold mean 0.493 → 0.494 | **APPLIED** |
| 5 | **Fig 14** "exceeds across the recall range" | **WONTFIX** — author reviewed and retained |
| 6 | **Fig 10** Haldane–Anscombe clause | **APPLIED** — reworded to state no correction was required |
| 7 | **Fig 12** uncited `fig:s4` | **APPLIED** — brief explanatory appendix text added |

Five of seven items applied to `manuscript/main.tex`; two retained by author decision. The
document compiles with `pdflatex` with no errors and no undefined references. Net manuscript
change: **+15 / −6 lines**, page count unchanged at 48.

### Item 5 — why it was closed as WONTFIX

Recomputation showed the point-estimate curve never falls below the no-skill line: **0 of 187
points are below it**, exactly **1** sits on it (recall = 1.0, precision = 0.695187, difference
+0.000000), and the median margin above it is **+0.142**. The right endpoint of *any* PR curve
lands on the prevalence line by construction — at recall = 1 every positive is predicted
positive, so precision = P/N — which makes it uninformative about this model. The caption's
claim is therefore a one-point technical overreach, not a misdescription. An earlier draft of
this report also asserted the bootstrap CI band dips below the line near recall ≈ 0.95; that
came from visual inspection and is **not verifiable** from `catboost_pooled_auprc.csv`, which
carries only `recall,precision` and no CI columns. It should be disregarded.

### Item 4 — where Table 8 comes from

Table label: **`tab:catboost_metrics_firstrepeat`** (`main.tex`, caption at the
"Hyperparameter optimization" appendix subsection). Source of its numbers:
`references/hyperparameter_optimization/catboost_first_repeat_optimization_metrics.csv`,
one row per fold, written by the optimization stage (`optreport.py` /
`utils2/optimization.py`) for the first repeat only. The `Mean` and `Std` columns are computed
across those four rows. Only the threshold **mean** cell is affected: 0.49351 rounds to 0.494,
the table prints 0.493. Std 0.0077 → 0.008 is correct, and all other cells verify to 3 dp.
