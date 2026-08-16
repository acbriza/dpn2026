# `edareport.py` — EDA Report Design & Build Summary

_2026-08-16_

Session summary of the design and implementation of [edareport.py](edareport.py), a new
exploratory data analysis report for the BMC Medical Informatics and Decision Making
submission, intended to run as:

```
python module/edareport.py bin_eda_final_202608.yml
```

It replaces [eda.py](eda.py), which was written against an earlier version of the study and
produces tables and figures that no longer match what the pipeline actually does. `eda.py`
is left in place; the new script writes to the same directory under different filenames, so
nothing is overwritten.

Statistics quoted below were computed by executing the code against the dataset, not
asserted from the manuscript.

## What the study actually is

The session began with a plan based on [../overleaf/main.tex](../overleaf/main.tex). Two
corrections from the author invalidated it, and the plan was re-derived from the code and
the current run outputs in `experiments/binary/`:

- **`main.tex` is stale.** It describes a 41-feature model reaching ROC-AUC 0.972 with nerve
  conduction studies as the dominant predictors. That is a superseded run.
- **Nerve conduction studies were never used for modelling or counterfactuals.** They are
  dropped in every stage: [selreport.py:80](selreport.py#L80), [optreport.py:60](optreport.py#L60),
  [expreport.py:75](expreport.py#L75), [cfreports.py:182](cfreports.py#L182), with an
  assertion at [cfreports.py:197](cfreports.py#L197) that no NCS feature can ever be varied.

The study as implemented:

| | |
|---|---|
| Analysis cohort | 187 patients (201 screened, 190 enrolled, 3 dropped for incomplete numerics) |
| Outcome | Confirmed 130 vs Unconfirmed 57, prevalence 69.5% |
| Candidate predictors | **22** — Profile 6, Comorbidity 5, Neuro exam 4, MNSI 1, Sudoscan 6 |
| Events per variable | 2.6 |
| Reference model | CatBoost, ROC-AUC **0.799** ± 0.073, sensitivity 0.886, specificity 0.527, Youden 0.413 |
| Counterfactual-actionable | 6 — INSULIN, HBA1C, HPN, PAOD, DSLPDMIA, CKD |

`main.tex` also reports the cohort as 190/131/59 in its tables while the modelling runs on
187/130/57, and its categorical table's column headers are mislabelled (the blocks are
feature-value blocks, captioned as outcome groups).

## Findings that shaped the report

Four results, all verified by execution, determined what the report contains.

- **NCS exclusion is outcome circularity.** Under the 2009 Toronto consensus criteria,
  *Confirmed* DPN requires an abnormal nerve conduction study, so NCS is constitutive of the
  label. Univariable AUCs of 0.86–0.93 (SSA_L 0.927, CMAPKNE_L 0.925) are the signature of
  definitional leakage rather than predictive power. The author confirmed this as the reason.
  Stating it up front converts the obvious reviewer question into a methodological strength,
  so it became a main-text table rather than a methods sentence.

- **A single bedside sign nearly matches the model.** DEC_AR (decreased ankle reflex) alone
  reaches AUC 0.748; the tuned 22-feature CatBoost reaches 0.799. This is the honest measure
  of what the machine learning adds, and a reviewer will compute it if the paper does not.

- **NS is largely an age transform**: ρ(AGE, NS) = **−0.88**, ρ(AGE, CAS) = +0.59. The
  Sudoscan neuropathy score is age-adjusted by construction. VIF > 5 flags **exactly two**
  features — NS (10.5) and AGE (7.2), i.e. both halves of that pair — which is why the
  collinearity-pruned `NoCol` feature set performs no better than the full set: it discards
  two informative predictors rather than redundant noise.

- **The counterfactual-actionable features are the weakest in the dataset.** INSULIN 0.586,
  DSLPDMIA 0.570, CKD 0.554, HBA1C 0.545, PAOD 0.518, HPN 0.514. Only INSULIN survives
  Benjamini-Hochberg correction across the 22 predictors (q = 0.047); 14 of 22 features do.
  The counterfactual engine is therefore constrained to move the variables carrying the least
  marginal signal, which is the expected origin of the large feature displacements the
  generated explanations require. Reporting it explicitly frames the CF magnitudes as a
  consequence of the actionable set rather than leaving them to be questioned.

## Design decisions

Settled with the author before implementation:

| Decision | Choice |
|---|---|
| Venue | BMC Medical Informatics and Decision Making |
| Main-text budget | 2 tables + 2 figures; everything else to Additional file 1 |
| NCS exclusion rationale | Outcome circularity / leakage |
| Zeros encoding absent NCS responses | Retained in all summaries; counted as an annotation line |
| Actionable-feature weakness | Reported in the main text, marked in Figure 2, with a paragraph |
| Four-category Toronto analysis | Out of scope; binary outcome only |

## Deliverables

All output goes to `experiments/binary/eda/`. Every artefact ships a standalone
`_caption.txt` at BMC length, with all quoted numbers interpolated from the computed
statistics so a caption cannot drift from its own figure.

**Main text**

- **Table 1 — cohort characteristics.** 22 predictors grouped by domain; Overall /
  Unconfirmed / Confirmed; effect size, p, q; ◇ marks the actionable six.
- **Table 2 — feature eligibility and target-leakage audit.** All 40 features × role ×
  univariable AUC. The 18 NCS variables appear with their AUCs as the evidence for exclusion,
  carrying the non-recordable annotation. Replaces the stale feature and CF-suitability
  tables in `main.tex`.
- **Figure 1 — cohort overview.** Participant flow, outcome composition with prevalence and
  EPV, structural zeros.
- **Figure 2 — univariable discrimination.** AUC forest with bootstrap CIs, coloured by
  domain, shape marking actionability, fill marking q < 0.05, reference lines at 0.5 and at
  the model's 0.799.

**Additional file 1** — S1 rainclouds for the 10 continuous predictors; S2 categorical
prevalence with Wilson CIs plus an odds-ratio forest; S3 clustered correlation matrix, the
AGE~NS scatter and VIF bars; S4 data provenance; S5 the actionable six in detail; S6 NCS
descriptives (worth publishing since the dataset is newly released).

**Cross-artefact** — `numbers.tex` (22 `\newcommand` macros for manuscript prose),
`captions.txt`, `eda_manifest.json`, `feature_statistics.csv`, `cleaning_report.txt`.

## Statistical conventions

- Non-parametric throughout: median [IQR], Mann-Whitney U. Distributions are skewed and
  zero-inflated, so normality diagnostics are not reported as table columns.
- Discrimination as **directional AUC** for every feature, so continuous and binary
  predictors sit on the scale the models are later reported on. Values below 0.5 mean lower
  in the Confirmed group and are never folded to ≥ 0.5. CIs are stratified percentile
  bootstrap (2000 resamples, seeded), stratification mattering because the groups are 130 vs 57.
- Binary features additionally get an odds ratio, the form clinical readers expect.
- **Fisher's exact test** where any expected cell falls below 5 — live for PAOD (8 positives)
  and GBS (2). `eda.py` used chi-square uniformly, which is invalid there.
- **Haldane-Anscombe** correction on empty cells, which keeps GBS estimable.
- **Benjamini-Hochberg** q-values computed within the 22 modelled features and, separately,
  within the 18 excluded NCS features. Pooling would deflate the predictors' q-values, since
  the excluded set is significant by construction and pushes every predictor to a later rank.

## Dropped from the previous script

Q-Q grid, pairplot, z-score outlier heatmap, mean ± SE bar charts, and the paired
density/boxplot figures per modality. The density and box figures showed the same
information twice across six full-width figures; bar-of-means on zero-inflated skewed data
is discouraged in medical journals; and the outlier heatmap has no analytic role once the
tests are rank-based.

## Verification

- **LaTeX**: all four tables compile under `pdflatex` with no errors and no severe overfull
  boxes; `numbers.tex` macros expand.
- **Figures**: each rendered PNG was inspected and layout collisions fixed — overlapping tick
  labels in the raincloud grids, annotations sitting on the top row of Figure 2, exclusion
  text bleeding across panels in Figure 1.
- **Tests**: [../tests/test_edareport_correctness.py](../tests/test_edareport_correctness.py),
  23 tests, all passing. Coverage: AUC against `roc_auc_score` including all-ties binary
  predictors; direction preservation; bootstrap reproducibility; Fisher-vs-chi-square
  selection; Haldane-Anscombe; the modelling/excluded split matching what `cfreports.py`
  trains on; separate BH families; the provenance parser including the patient-code
  off-by-one; and non-recordable counts against the stored zeros.

One real defect was found by the tests: `wilson_ci` returned an upper bound of
0.99999999999999978 when k = n, an interval excluding the observed proportion. Fixed by
clamping against p, which removes only the rounding error.

## Open items for the manuscript

- `main.tex` contradicts this report throughout — cohort size, feature count, headline AUC,
  and the NCS-dominates narrative. It needs rewriting against the current run, not patching.
- `youden_summary_table.png` is a table shipped as an image; BMC requires tables as text.
- The coherence check implied by Figure 2 is worth reporting once the CF results are
  summarised: INSULIN is the one actionable feature with genuine univariable signal, so it
  would be expected among the most frequently altered features in the generated
  counterfactuals. If it is not, that is worth a sentence.
