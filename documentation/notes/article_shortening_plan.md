# Plan: cutting `manuscript/main.tex` to 20 published pages

Companion to [`article_shortenning.md`](article_shortenning.md), which established the
target. That note answered *how long*; this one answers *what to cut*. Measured against
`main.tex` as of 2026-09-01 (1,498 lines, 62 A4 pages under `sn-jnl`).

**No edits have been made.** This is the plan only.

---

## 1. The budget

"20 pages" means 20 pages of BMC's published PDF, not of the local `sn-jnl` A4 build.
Using the conversion fit in `article_shortenning.md`:

```
pages ~= 3.73 + 0.686/1k body words + 0.673/figure + 0.343/table + 0.334/10 refs
```

Appendix floats do not count — at submission the `appendices` block becomes Additional
file 1, a separate PDF.

| | Now | Target | Change |
| --- | ---: | ---: | ---: |
| Body words (excl. floats) | 16,101 | **9,600–10,000** | −38% |
| Main-text figures | 10 | **7** | −3 |
| Main-text tables | 14 | **5** | −9 |
| Main-text caption words | 3,295 | **≤650** | −80% |
| References | 57 | 55 | −2 (dedup) |
| **Projected published pages** | **28** | **19** | |

Projection at target: `3.73 + 6.59 + 4.71 + 1.72 + 1.84 = 18.6`. The fit's worst observed
residual is +0.86 pages, so the plan lands at ~19.5 in the bad case — inside 20 with
margin. Going to 8 figures and 6 tables at the same word count projects to 20.0 exactly,
with no margin, which is why the float budget is 7/5 and not 8/6.

**Local proxy checkpoint:** the `sn-jnl` A4 build should fall from 62 pages to roughly
**28–32 pages** (main text ~19–22, Additional file ~9–11). Use this only as a progress
signal; the 20-page target is the projection above, not the local page count.

---

## 2. What the cut protects

The manuscript states its own contribution (Discussion opening, line 1028): *"the
counterfactual analysis, rather than the underlying classifier's raw discriminative
performance, is the primary contribution of this work."* Four things follow from taking
that seriously, and they are the triage rule for everything below:

1. **Counterfactuals as a DPN pre-screening tool for low-resource settings** — the central claim.
2. **The reading framework for clinical counterfactuals** — the two axes (flip direction ×
   prediction correctness) and the action-vs-evidence register, which the Conclusion
   explicitly claims generalizes beyond DPN.
3. **NCS-free screening is feasible, and a Sudoscan + bedside-exam panel nearly matches
   the full feature set** — the practical finding with immediate clinical value.
4. **Philippine cohort + Sudoscan predictors** — diversity contribution.

Everything that serves these keeps its place in the main text. Everything that documents
*how the pipeline was built* — model taxonomies, CV bookkeeping, DiCE parameter tables,
per-fold metric dumps — is reproducibility material and belongs in Additional file 1.

The three largest cuts all fall on material that defends methodological choices at length
rather than reporting findings.

---

## 3. Float triage

### 3.1 Keep in main text (7 figures, 5 tables)

| Float | Current caption | Target | Why it stays |
| --- | ---: | ---: | --- |
| `fig:patient_flow_classification` | 189 w | 40 w | STROBE-style flow is expected in a clinical paper. **Keep panel (a) only**; panel (b) is three numbers already in the text — drop it and make this a single-column `figure`, not `figure*`. |
| `fig:pipeline_overview` | 106 w | 45 w | Earns its keep by letting Methods shed prose. Change `[p]` → `[tbp]` (see §6). |
| `fig:auprc_heatmaps` | 165 w | 55 w | Carries contribution 3 (NoNeuro / SudoNeuro). |
| `fig:feature_importances` + `fig:shap` | 84 + 91 w | 55 w | **Merge into one two-panel figure.** Together they support the "top predictors are not levers" hinge into the counterfactual analysis. Separately they are two near-identical heatmaps. |
| `fig:localcf-cases` | 256 w | 90 w | Contribution 1. Allow a longer caption than the rest — it explains a non-obvious chart encoding (shared HbA1c axis vs. binary flip markers) that the reader cannot infer. |
| `fig:globalcf` | 169 w | 45 w | Contribution 1, aggregate view. The caption's last two sentences are argument, not description — move them to Discussion prose or cut. |
| `fig:clinician_workflow` | 135 w | 45 w | The "so what". Change `[p]` → `[tbp]` and delete the following `\clearpage`. |
| `tab:cohort` | 39 w | 35 w | Table 1. Clinical reviewers expect it. Caption is already short; trim the 120-word footnote block instead. |
| `tab:cf_features` | 127 w | 45 w | Defines the counterfactual scope. **Merge in the modeling-variable rows of `tab:clinical_data`** so one table gives code + description + actionability. |
| `tab:optimization_metrics_40repeats` | 117 w | 40 w | The headline performance result. |
| `tab:cf-direction-reading` | 208 w | 50 w | Contribution 2. This table is the framework; with it in place the Discussion prose around it can shrink hard (§4.4). |
| `tab:localcf-changed` | 157 w | 40 w | Contribution 1, the aggregate feature-change result quoted in the Abstract. |

Retained caption total: 1,708 w → **~585 w**.

### 3.2 Move to Additional file 1 (12 floats)

| Float | Current caption | Reason |
| --- | ---: | --- |
| `fig:univariable_discrimination` | 226 w | Same message as `tab:cohort`'s effect-size column. Highest caption-to-value ratio in the manuscript. |
| `fig:auprc_violins` | 130 w | Stability of *unoptimized baseline* estimates; secondary to every contribution. |
| `fig:catboost_auprc` | 151 w | Per-fold ROC/PR curves. Reproducibility evidence. |
| `tab:clinical_data` | 39 w | Full codebook incl. NCS variables. Modeling subset merges into `tab:cf_features`; the full version (with NCS) goes to Additional file. |
| `tab:feature_sets` | 95 w | The 12 feature sets are legible from the heatmap axis labels + two sentences of Methods. |
| `tab:selection_metrics` (sidewaystable) | 141 w | 14 models × 8 metrics. Text reports the top three. A landscape table costs a full page for numbers nobody reads in sequence. |
| `tab:catboost_metrics_firstrepeat` | 88 w | Per-fold metrics of the representative repeat. |
| `tab:dice_setup` | 111 w | DiCE configuration. **Merge with `tab:local_cf`** into one supplementary config table. |
| `tab:local_cf` | 73 w | (merged, as above) |
| `tab:localcf-model-level` | 216 w | Per-model CF yield/sparsity. The numbers that matter are quoted in text. |
| `tab:localcf-listing` | 182 w | Per-patient CF listing for all 8 instances. |

Mechanically this is cheap: cut the float, paste it inside the existing `\begin{appendices}`
block, trim its caption. `\ref` targets keep resolving — numbering just shifts to A1…An.
Only the surrounding prose needs rewording ("Figure 5" → "Additional file 1, Figure A3").

Additional file 1 ends up holding ~17 floats (existing S1–S5 + `tab:s1_provenance` +
the 11 moved after the dice_setup/local_cf merge). That is fine: Additional files carry
no page cost in the published article.

### 3.3 Caption rules

Current main-text captions average 137 words across 24 floats. New rule: **a caption
states what the reader is looking at and nothing else.** Specifically —

- Delete every sentence that restates a number already in the body text. `fig:globalcf`'s
  caption recites six counts that appear verbatim two paragraphs below it.
- Delete every sentence that argues. `fig:globalcf` ends with 50 words defending the
  decision to show a per-model breakdown; that is Discussion material.
- Delete every cross-reference chain. `fig:localcf-cases` sends the reader to two other
  sections from inside the caption.
- Keep encoding explanations (what a color or an axis means) — these are the one thing a
  caption must carry, and they are why `fig:localcf-cases` gets a 90-word allowance.
- `fig:patient_flow_classification`'s caption is 189 words, of which ~120 argue for the
  Confirmed/Unconfirmed collapse. That argument belongs in Methods §Study population;
  a version of it is already there.

---

## 4. Prose budget, section by section

| Section | Now | Target | Change |
| --- | ---: | ---: | ---: |
| Introduction → **Background** | 837 | 800 | −4% |
| Data Collection *(folds into Methods)* | 721 | 250 | −65% |
| Methods | 3,567 | 2,300 | −36% |
| Results | 2,835 | 2,700 | −5% |
| **Discussion** | **6,552** | **3,100** | **−53%** |
| **Conclusion → Conclusions** | **1,204** | **300** | **−75%** |
| Backmatter / Declarations | 160 | 200 | — |
| **Total** | **15,876** | **9,650** | **−39%** |

Discussion and Conclusions supply 3,650 of the 6,200 words cut — 59% of the reduction —
without touching a single result.

### 4.1 Background (837 → 800)

Nearly right already. One trim: paragraphs 6–8 (lines ~198–202) spend ~250 words
surveying prior XAI-in-medicine work at a level of generality that does not bear on this
study's design. Compress to ~150, keeping the citations. Everything else stays — the
underdiagnosis framing, the Sudoscan rationale, and the NCS-exclusion justification are
load-bearing.

### 4.2 Data Collection (721 → 250, folded into Methods)

BMC's structure has no standalone Data Collection section. Move to
`\subsection{Study population and data collection}` at the head of Methods, keeping only:
cohort size and recruitment window, exclusion criteria, Toronto 2009 classification, the
187/130/57 split, the Confirmed-vs-Unconfirmed collapse and why, and ethics approval.

Move to Additional file 1 (as a "Detailed clinical assessment protocol" text section):

- **The entire NCS acquisition protocol** (line 246–248: skin temperature, room
  temperature, Nihon Kohden equipment, 20 Hz–3 kHz filter, 200 µV/div, 10 ms/div) and the
  itemized nerve-attribute list. NCS is *excluded from every predictor set* — its
  acquisition parameters are not needed to reproduce the model. This is the clearest
  example in the manuscript of detail retained out of completeness rather than necessity.
- SWME / 128-Hz tuning fork / pinprick / Achilles reflex procedural detail (line 242).
- The Sudoscan measurement procedure (electrodes, <4 V, 2 minutes, reverse iontophoresis).
  Keep one sentence in Methods saying what Sudoscan measures; move the how.

### 4.3 Methods (3,567 → 2,300)

| Subsection | Now | Target | Action |
| --- | ---: | ---: | --- |
| §Models | 423 | 120 | The five-paragraph taxonomy explaining what LDA, kNN, Random Forest and SVM *are* is textbook material. Replace with one sentence naming the 14 algorithms and the majority-class baseline, plus citations. Full taxonomy → Additional file. |
| §Model and Feature Set Selection | 351 | 200 | Keep the 14×12×40 design and the VIF/NoCol rule. `tab:feature_sets` is leaving, so name the twelve sets inline in one sentence. |
| §Performance Metrics | 173 | 130 | **Keep — this is the one place AUPRC gets justified.** Two other passages currently duplicate this justification (§4.4); they go, this stays. |
| §Preventing Data Leakage | 307 | 150 | Keep the NCS-exclusion bullet essentially intact — it is a contribution, not bookkeeping. Compress the other three bullets and the closing paragraph. |
| §Class Imbalance | 220 | 130 | Keep the SMOTE-refusal rationale (a real decision with a counterfactual-specific reason: every training row stays a real patient). Compress the rest. |
| §Hyperparameter Optimization | 636 | 300 | Keep the nested-CV mechanics and the threshold rule. The italic *"One modeling pipeline, evaluated by cross-validation"* paragraph (~200 w) is defensive clarification — compress to two sentences. *"Reuse of the saved models"* → one sentence. |
| §Pooled Model Diagnostics | 309 | 0 | **Delete from main text.** It is the methodological caveat for a figure (`fig:catboost_auprc`) that is moving to Additional file 1. Move the whole subsection with it. |
| §Explainability Analysis | 335 | 180 | Keep the fold-by-fold-never-averaged rule and the importance-vs-SHAP unit distinction. Compress the rest. |
| §Counterfactual Generation (all subsections) | 680 | 450 | Keep: DiCE choice, actionability stratification, global/local permitted-range formulas, and the δ=0.08 borderline rule. Move: the two-paragraph justification of the genetic algorithm and both parameter tables. |

### 4.4 Results (2,835 → 2,700)

Light touch — this is the contribution. Two adjustments:

- **§Per-Fold Model Performance: 463 → 150.** `tab:catboost_metrics_firstrepeat` is moving
  out, so this text loses its float. Keep only: which four models are retained, the
  threshold range (0.483–0.500), sensitivity >0.90 in three of four folds with Model 0 the
  outlier at 0.727, and the positive-class-weight spread as the explanation for specificity
  variance. Drop the per-metric walkthrough.
- **§Instance-Level Counterfactuals: 726 → 550.** Compress the three `\textit{Patient
  Profile}` blocks — the figure panel headers already carry predicted probability,
  threshold and margin. Keep every counterfactual count; they are the result.

Everything else in Results stays. If space is tight later, Results is the *last* place to cut.

### 4.5 Discussion (6,552 → 3,100) — the main cut

The Discussion currently shadows Results subsection-for-subsection, and a large share of
it re-argues Methods choices rather than interpreting findings.

| Subsection | Now | Target | Action |
| --- | ---: | ---: | --- |
| Opening framing | 196 | 150 | Keep the scope/contribution statement; it does real work. |
| §Baseline Model Evaluation: Rationale and Metric Selection | 204 | 0 | **Delete.** Pure Methods rationale (why AUPRC over ROC-AUC), and it is stated a *third* time under §Clinical Suitability. One copy survives, in Methods §Performance Metrics. |
| §Interpreting the Baseline Model Comparison | 109 | 40 | One sentence: the top three are within noise, so the choice rests on §CatBoost-vs-RF. |
| §Interpreting the Feature Set Ablation | 97 | 90 | **Keep** — contribution 3. |
| §Reproducibility and Stability of Estimates | 134 | 50 | One sentence; `fig:auprc_violins` is moving out. |
| §Study Limitations at the Baseline Evaluation Stage | 79 | 0 | **Merge into §Limitations.** Stage-scattered limitations should consolidate. |
| §Selected Feature Set and Model | 93 | 40 | One sentence. |
| §CatBoost versus Random Forest | 417 | 160 | Genuine methodological contribution — keep both arguments (ordered boosting suits small *n*; smoother sigmoid surface suits a genetic CF search), drop the elaboration. Full version → Additional file. |
| §CatBoost Training and Optimization preamble | 49 | 0 | Delete (pure signposting). |
| §Interpreting the Repeated CV Estimates | 332 | 120 | Restates numbers already in Results and re-argues why sensitivity matters. Keep the sensitivity-stability point, cut the per-metric commentary on F1 and Youden. |
| §Choice of Representative Fold Models | 314 | 80 | Defensive digression. Two sentences: the first repeat is fixed by seed (so not cherry-picked) and its means sit within ~1 SD of the pooled estimates. Full argument → Additional file. |
| §Clinical Suitability for Pre-Diagnostic Screening | 157 | 0 | **Delete.** Four disconnected paragraphs, each restating something already said (AUPRC rationale ×3rd time, 4-fold CV rationale, generic CatBoost properties). |
| §Feature Importance and Model Interpretability | 345 | 200 | Keep the hinge argument: the top predictors measure *established* neuropathy and are not levers — this is what motivates the whole counterfactual analysis. Drop the closing paragraph on low-importance comorbidities. |
| §Counterfactual Analysis preamble | 642 | 300 | The opening paragraph re-defines DiCE and counterfactuals for a third time (Background, Methods, here) — cut. Keep §Feature Actionability short (the table carries it). **Keep §Why the Counterfactual Yield Was Low** — 8-of-61 is the study's most important honest limitation — but strip the DiCE parameter recitation from it (that belongs in Methods/Additional file). |
| **§Instance Level Counterfactual Analysis** | **2,122** | **900** | The single largest block in the manuscript (13% of all body words). See below. |
| §Aggregate-level Counterfactual Analysis | 510 | 350 | Keep the multifactorial reading and the model-heterogeneity caveat. Compress the literature-alignment paragraphs. |
| §A Proposed Clinical Usage Workflow | 448 | 200 | The figure carries the flow; the prose currently walks every node of it in sequence. Keep the framing sentence, the clinician-filter step, and the decision-support disclaimer. |
| §Limitations | 304 | 350 | **Grows** — absorbs the stage-specific limitations deleted above. Consolidating them here is a readability gain, not just a length one. |

**Breaking down §Instance Level Counterfactual Analysis (2,122 → 900):**

- Framework preamble (two axes + two rules), ~700 w → **250 w**. `tab:cf-direction-reading`
  is staying in the main text precisely so this prose can shrink; right now the table and
  the text say the same thing twice.
- Patient 20 (~290), Patient 40 (~330), Patient 123 (~430) → **~180 w each = 540**. Each
  currently re-states the patient profile (already in Results), then re-derives the
  direction logic (already in the framework and the table) before reaching its finding.
  Keep the finding and the case-specific reading; cut the re-derivation.
- §Synthesis of the Three Case Studies (~640 w) → **~110 w** or delete outright. It
  re-narrates all three cases in aggregate immediately after presenting them. At most it
  needs a short paragraph on the one genuinely new point: HbA1c is the most sensitive
  feature across all three, and the artifacts a reader must filter fall into three kinds.

### 4.6 Conclusions (1,204 → 300)

BMC Conclusions run 150–250 words. Rewrite as one or two paragraphs: the question asked,
the headline performance with its specificity caveat, the counterfactual finding
(HbA1c in 91.1% but rarely alone), and the proof-of-concept boundary.

Redistribute the rest:

- The "added value in three places" paragraph → one clause in the Discussion opening,
  which already makes this claim.
- The limitations recap → already covered by §Limitations; delete.
- The future-directions paragraph (~230 w) → ~80 w at the end of §Limitations, keeping
  external validation, prospective clinician-in-the-loop evaluation, and the CF-yield
  problem. Drop the rest.

---

## 5. Order of work

Sequenced so that each step's saving is measurable before the next begins, and so the
cheap, low-risk cuts land first.

1. **Housekeeping** (~15 min, no judgment calls). Delete the live template abstract at
   line 152 — it currently precedes the real one. Merge the duplicate bib entries
   (`mothilal2020` / `mothilal2020explaining`, `tesfaye2010` / `tesfaye2010diabetic`; the
   `(<-- Verify)` variants are placeholders) and settle on one key for each. 57 → 55 refs.
2. **Caption pass** (§3.3). Independent of every other cut, mechanical, and immediately
   visible in the PDF. −2,600 caption words.
3. **Float moves** (§3.2). Cut-and-paste into `\begin{appendices}`, reword the referring
   sentences. `\ref` keeps resolving throughout.
4. **Conclusions rewrite** (§4.6). Smallest section, largest proportional cut, no
   dependency on anything above. −900 words.
5. **Discussion cut** (§4.5). The big one. Work top-down; §Instance Level Counterfactual
   Analysis last, since it is the most delicate. −3,450 words.
6. **Methods compression + Data Collection fold-in** (§4.2, §4.3). −1,700 words.
7. **Results trim** (§4.4). Last, and lightest. −135 words.
8. **Float placement pass** (§6), then recompile and re-measure.

---

## 6. LaTeX-level items

- `fig:pipeline_overview` and `fig:clinician_workflow` both use `[p]` (float page), each
  consuming a near-empty page. With fewer floats competing, both should be `[tbp]`.
- Delete the `\clearpage` after `fig:clinician_workflow` (line 1227).
- `fig:patient_flow_classification` becomes a single-panel `figure`, not `figure*`, once
  panel (b) is dropped.
- `tab:selection_metrics` is the only `sidewaystable`; once it moves to Additional file 1
  the `rotating` package may become unnecessary.
- At submission, extract the `\begin{appendices}`…`\end{appendices}` block into a separate
  Additional file 1 PDF. BMC has no in-article appendix. The `\bmhead{Supplementary
  information}` text needs rewording to point at Additional file 1 rather than
  `Appendix~\ref{secA1}`.
- `\section*{Declarations}` is still the template's placeholder itemize list — it needs
  real content before submission, but it costs nothing in length.

---

## 7. How to verify progress

Re-run the measurement from `article_shortenning.md` after each numbered step. Worth
saving as a script so the count is one command:

- Strip LaTeX comments; take the region between `\maketitle` and `\bmhead{Supplementary`;
  remove `figure`/`table`/`sidewaystable` environments wholesale; strip macros; count
  alphanumeric tokens.
- Count `\begin{figure}`/`\begin{table}` occurrences *before* `\begin{appendices}` only.
- Feed body words, figure count, table count and reference count into the §1 formula.

Checkpoints, in the order of §5:

| After step | Body words | Fig | Tab | Projected pages |
| --- | ---: | ---: | ---: | ---: |
| — (now) | 16,101 | 10 | 14 | 28 |
| 3 (floats moved) | 16,101 | 7 | 5 | 25 |
| 4 (Conclusions) | 15,200 | 7 | 5 | 24 |
| 5 (Discussion) | 11,750 | 7 | 5 | 22 |
| 6 (Methods) | 10,050 | 7 | 5 | 20 |
| 7 (Results) | 9,650 | 7 | 5 | **19** |

Steps 2 and 3 alone take the manuscript from 28 to 25 projected pages without cutting a
word of argument, and step 5 does more work than steps 6 and 7 combined.

---

## 8. Open questions for the author

1. **Merging the importance and SHAP figures.** §3.1 assumes the two heatmaps can become
   one two-panel figure. If the source plots do not compose well, the fallback is to keep
   `fig:feature_importances` in the main text and move `fig:shap` to Additional file 1 —
   same float count, slightly weaker argument.
2. **The Synthesis subsection** (§4.5). The plan reduces it to ~110 words. Deleting it
   entirely saves another 110 and, in my reading, loses nothing the three cases have not
   already said — but it is the passage that most explicitly ties the framework
   (contribution 2) back to the data, so the call is the author's.
3. **`fig:univariable_discrimination` vs. `tab:cohort`.** The plan keeps the table and
   moves the figure. If reviewers are expected to prefer the visual, the reverse swap
   holds the budget exactly — but the table also serves as the conventional Table 1, which
   the figure cannot.
4. **BMC section names.** The plan assumes Background / Methods / Results / Discussion /
   Conclusions. Confirm against the SpringerLink submission guidelines, which
   `article_shortenning.md` flags as not machine-fetchable — this needs a browser check,
   along with the "no length limit" prior and any figure-legend word cap.

---

## 9. Implementation record (2026-09-01)

Plan executed. `manuscript/main.tex` compiles clean — no undefined or multiply-defined
references, no overfull boxes above 20 pt.

| | Before | After | Target |
| --- | ---: | ---: | ---: |
| Body words (excl. floats) | 16,101 | **9,239** | 9,600–10,000 |
| Main-text figures | 11 | **7** | 7 |
| Main-text tables | 13 | **5** | 5 |
| Main-text caption words | 3,295 | **608** | ≤650 |
| References cited | 40 | **38** | — |
| **Projected published pages** | **27.8** | **17.8** | ≤20 |
| Local `sn-jnl` A4 build | 62 pp | **49 pp** (main 1–27, Additional file 28–49) | — |

Measurement is reproducible: `documentation/notes/measure_length.py`, run from
`manuscript/`.

### Decisions taken

- **Explainability figure** — merged as side-by-side subfigures, with the legend panel
  stripped from both and the color key moved into the caption (using the figures' actual
  Tableau-10 values, sampled from the PNGs). Dropping the legend panel also removed the
  wasted left third of each image, so the merge cost less legibility than a naive
  side-by-side would have. Each panel keeps its own row ordering.

  The strip is a pipeline step, not a one-off edit: `strip_explainability_legends()` in
  `module/postreports.py`, driven by the new `explainability:` block in
  `module/experiments/bin_postreport_final_202608.yml`, writes
  `*_nolegend.png` into `postreport/<tag>/explainability/` (copied to
  `manuscript/references/postreport/explainability/`). It *locates* the blank gutter
  between the legend and heatmap axes rather than hardcoding a pixel offset, so the crop
  survives regeneration with a different feature count or longer labels; the search is
  restricted to the left 45% of the image so the heatmap/colorbar gap is not mistaken for
  it. Source figures are untouched.
- **Synthesis subsection** — cut to ~110 words, retaining only the two points not already
  made by the individual cases.
- **Table 1** — `tab:cohort` kept, `fig:univariable_discrimination` moved.
- **BMC structure** — applied: Background / Methods / Results / Discussion / Conclusions,
  with Data Collection folded in as Methods §2.1.

### Deviations from the plan

- **Results §Instance-Level Counterfactuals was left at full length** (~726 words) rather
  than cut to ~550. The word budget was already met with 360 words to spare, and the plan
  itself designates Results the last place to cut. The three patient profiles are the
  paper's central evidence, so the clinical detail stays. Reverse this first if the count
  needs to come down further.
- **Float counts in §3 were off by one in each direction** — the manuscript had 11 figures
  and 13 tables, not 10 and 14. The total (24) and the triage lists were correct.
- **References**: `bibfile.bib` holds 57 entries but only 40 were cited; the dedup brought
  cited references to 38. Nineteen uncited entries remain in the `.bib` and are harmless.
- **Four citations were dropped and restored** during the Discussion rewrite —
  `saito2015precision` and `davis2006relationship` (AUPRC over ROC-AUC, now in Methods
  §2.2.3), `youden1950index` (Youden index definition), and `ada2024` (guideline targets,
  Conclusions). Worth re-checking after any further prose cut.

### Added, not in the original plan

The main text now promises material that had to be written rather than merely relocated.
Additional file 1 gained three prose subsections: the detailed clinical assessment
protocol (NCS acquisition settings, SWME/tuning-fork/Sudoscan procedure), the candidate
model descriptions cut from Methods §Models, and the pooled model diagnostics caveat cut
from Methods §Pooled Model Diagnostics.

### Post-implementation fixes

- **Three main-text figures were escaping into the appendix.** `fig:localcf-cases` (p39),
  `fig:globalcf` (p40) and `fig:clinician_workflow` (p41) rendered past the appendix start
  (p28) despite being in the main-text source. Cause: `fig:localcf-cases` exceeds
  `\topfraction`, so LaTeX refused it on a text page; because figures must appear in source
  order, the two behind it queued up and all three flushed at the end. Fixed by giving the
  three `[!tbp]` — the `!` overrides the default placement fractions. All twelve main-text
  floats now land beside their discussion (`fig:localcf-cases` p18, `fig:clinician_workflow`
  p28). A `\clearpage` before the Conclusions also worked but cost a page of whitespace and
  was unnecessary once the placement was fixed; one barrier before `\backmatter` remains.
- **Appendix moved after the bibliography.** It previously sat between `\section*{Declarations}`
  and `\bibliography{}`; it is now the last block before `\end{document}`.

### Still outstanding

1. `\section*{Declarations}` is still the template's placeholder itemize list.
2. The abstract is ~349 words; the journal's IQR is 264–351, so it fits, but it is at the
   top of the range.
3. At submission, extract `\begin{appendices}`…`\end{appendices}` into a separate
   Additional file 1 PDF — BMC has no in-article appendix.
4. Confirm BMC section names and any figure-legend word cap against the SpringerLink
   submission guidelines in a browser (see §8.4).
