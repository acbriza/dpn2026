# Methods section review — session notes (2026-08-30)

Companion to `methods_edit.md`. That file records the *planned rewrite* of the Methods
section; this one records the *review pass over the result* — a question-by-question audit
in which each claim in Methods was checked against the pipeline code and configs, per
CLAUDE.md's directive to "re-derive study facts from the pipeline and its output reports,
not from the manuscript."

Manuscript under review: `manuscript/main.tex` (**not** the legacy
`module/legacy/202608/overleaf/main.tex` named in CLAUDE.md).
Branch: `codereviewed`. Commits produced: `9dfd147`..`828afa0` (11 commits).

The chat interactions are reproduced at length below, because in almost every case the
question itself is what exposed the error, and several of the user's framing decisions
(what to argue, what not to quantify) are editorial choices that are not recoverable from
the diff alone.

---

## Working agreements established during the session

These held for the whole session and should carry into any further Methods work.

1. **Validation requests are not edit requests.** Stated explicitly:
   > "revalidate and justify the claim in line 512. **This is not an edit request but a
   > revalidation/check request**"

   The pattern the user settled into is a three-step loop, with a separate turn for each
   step: *validate* → *suggest edits* → *apply*. Nothing was edited until an explicit
   "apply".

2. **Prose must be compact.**
   > "can you suggest more compact explanations. I find these verbose"

   The user also trimmed drafted text after applying it (see the line-562 entry), so the
   applied wording is a floor, not a target.

3. **Do not quantify an approximate claim.**
   > "no need to be specific about actual count or percentages. Just edit the language to
   > reflect the effort to maintain the split ratio"

   Numbers are welcome where the underlying quantity is exact (0.3–1.0 search range,
   57/130 = 0.44 were both kept); they are rejected where they would dress up a
   best-effort property as a guarantee.

4. **Never put an unverified number or name into the manuscript.** Applied twice: the
   "up to 0.086" prediction drift from `expreport.py`'s comment was reported in chat but
   kept out of the text (no `dpncf` env here to reproduce it), and a concrete NCS quantity
   was dropped from a draft because `D.ncs_cols` had not been read.

---

## The review, question by question

### 1. AUPRC vs ROC-AUC — the justification was aimed at the wrong thing

> **User:** "explain further this text from manuscript/main.tex line 486: ROC-AUC can look
> favourable even when precision on the minority (Unconfirmed) class is poor, whereas
> AUPRC is sensitive to exactly that failure mode."

**Finding (factual error).** The sentence justified AUPRC by its sensitivity to poor
precision on the *minority* (Unconfirmed) class. But nothing in `module/` sets
`pos_label`, so sklearn's default of `1` applies and AUPRC is computed with
**positive = Confirmed = the majority class** (prevalence 0.695). The prevalence effect
therefore runs *opposite* to the sentence's implication: the PR chance baseline here is
0.695, which makes AUPRC the more generous metric in this study, not the
imbalance-stringent one.

The user then supplied the reframing that fixed it:

> **User:** "can't the argument be based more on the purpose of the ML model which is for
> pre-screening, more than class imbalance"

That is the stronger argument, and it dissolves the mismatch rather than patching it:
under a screening framing, positive = Confirmed is the *correct* choice, because recall is
then the fraction of confirmed patients referred onward for NCS and precision is the PPV
of that referral — the fraction of referrals to a resource-intensive, often geographically
inaccessible test that prove justified. A side question — "what is ppv?" — was answered in
chat (precision = PPV; PPV is prevalence-dependent, sensitivity/specificity are not).

> **User:** "apply your Draft replacement for line 485 except for the last sentence"
> **User:** "also update main.tex line 451 as per your suggestion"

**Applied** (`9dfd147`, `3837b49`, `81eac73`): §Performance Metrics rewritten around the
screening trade-off, stating the 0.695 no-skill baseline explicitly and noting that
ROC-AUC's false-positive-rate axis is not the quantity acted on at referral. §Model
Selection's AUPRC rationale was repointed from `sec:classimbalance` to `sec:metrics`.
A stale paragraph about NCS-only and Sudoscan-only models was replaced with the
Sudoscan-anchored additive feature sets that were actually run.

**Incidental fix:** the line-451 rewrite referenced `\ref{sec:metrics}`, a label that did
not exist. `\label{sec:metrics}` was added to the Performance Metrics heading in the same
edit.

### 2. Stratification — guarantee or best effort?

> **User:** "in line 510, is the split ratio assured or is it just a best effort? Check
> actual code for parameter that is set to try to keep this split ratio: *stratification
> keeps the 69.5\%/30.5\% split intact in every fold*"

**Finding (overstatement).** Best effort. sklearn's contract is "approximately the same
percentage of samples of each class" — a rounding guarantee (within one sample per class),
not exactness. And exactness is arithmetically impossible here: 187, 130 and 57 are none
of them divisible by 4 (ideal quarters 46.75 / 32.50 / 14.25). Simulating the actual
splitter produced three distinct test-fold shapes with prevalence spanning 0.6809–0.7021.

The first draft of the fix cited that 68.1–70.2% range and "within one patient per class",
and was rejected — this is where working agreement 3 was set. **Applied** (`e70fdfc`):
"stratification **constrains every fold to reproduce the cohort's
class distribution as closely as the fold sizes allow**."

### 3. Class imbalance, `scale_pos_weight`, and SMOTE

> **User:** "revalidate and justify the claim in line 512. This is not an edit request but
> a revalidation/check request"

Four findings came out of reading `optimization.py` and `counterfactuals.py`:

- **`scale_pos_weight` was described backwards in effect.** It multiplies the loss weight
  of the **positive** class only — not "the contribution of each class". Since positive =
  Confirmed = the *majority* here, values **below 1** are what correct the imbalance. The
  inverse-prevalence value 57/130 = 0.44 sits inside the searched 0.3–1.0, which is why
  that range is capped at 1.0.
- **"400 selection folds" is unsupported.** Selection uses one seeded
  `RepeatedStratifiedKFold(4, 10)` = 40 partitions, reused across 14 models × 12 feature
  sets = 6720 fits. 400 is stale arithmetic from when the feature-set table had 10 rows.
- **The SMOTE argument's closing clause named the wrong mechanism.** It claimed synthetic
  patients would compromise the counterfactual permitted ranges. They would not: those
  ranges are computed from `dfXy` (observed data) regardless of what the model trained on.
  The real exposure is the decision boundary the counterfactuals are generated against.
- **A proposed edit was withdrawn as redundant** — a clause for `sec:generalsetup` noting
  the permitted range is computed over the full cohort; main.tex:602 already says so.

**Self-correction during drafting:** a compact draft said the permitted ranges are "read
off observed feature minima and maxima". The code (`counterfactuals.py:152-183`) is
`[max(0, min − sd), max + sd]`. Caught before applying; reworded to "derived from each
feature's observed range."

> **User:** "can you suggest more compact explanations. I find these verbose"
> **User:** "apply these changes"

**Applied** (`10676ff`): `scale_pos_weight` redescribed as "scales the loss weight of
positive-class examples", with the below-1 explanation and the bracketing 57/130 = 0.44
spelled out; "400 selection folds" replaced by "all 6720 selection fits"; and the SMOTE
closing clause repointed from the permitted ranges to the decision boundary.

**Then compressed by the user** (`59bdaa3`, "class imbalance: personal edit") to roughly
half its length — the 57/130 sentence, the 6720-fits leakage clause, and the
counterfactual-boundary explanation were all cut, leaving the sparse-feature-space argument
and "keeps every training example a real patient, something that matters for the grounding
of the downstream task of generating counterfactuals." Same lesson as the line-562 entry:
verified detail earns its place in chat, not automatically in the manuscript.

**Separately in the leakage section** (`5523e53`): the bullet's claim that "a runtime
assertion verifies, for every repeat, that the four outer test folds jointly and exactly
tile all 187 patients" was removed, leaving the bullet to state only what the nested-CV
procedure itself guarantees.

**Also in this cluster** — an F-beta discrepancy: the threshold-selection metric is
`fscore_beta: 1` (F1), not F2, and the recorded metric list had drifted from what
`optimization.py` writes. `d1efc26` and `81eac73` removed the phantom F2/F1.25/F1.5/F1.75
entries and the "$\beta=1$" gloss.

### 4. Pooled vs fold-wise diagnostics — opposite-signed biases

> **User:** "validate and explain further line 542"

The claim under review: pooling the four fold models' predictions biases the diagnostics.
Validated, and made precise:

- **Rank statistics (AUROC, AUPRC) → conservative when pooled.** Each fold model comes
  from its own hyperparameter search and has its own probability scale; pooling forces
  cross-fold pairs onto incomparable scales, and with 4 equal folds only ~¼ of patient
  pairs are within-fold.
- **Binned calibration (ECE) → optimistic when pooled.** Fold-wise ECE is mean|·|, pooled
  ECE is |mean|; by the triangle inequality
  |Σ wᶠ(ȳᶠ − p̄ᶠ)| ≤ Σ wᶠ|ȳᶠ − p̄ᶠ|, so pooled ≤ fold-wise is *guaranteed*, not incidental.
- **Per-observation averages (Brier, DCA net benefit) → unaffected.** Net benefit
  `(tp/n) − (fp/n)·w` credits each patient +1/N if TP, −w/N if FP, 0 otherwise — additive
  over observations exactly like squared error, so pooled = size-weighted mean of the fold
  values.

**Self-correction, reported to the user before the edit:** the first pass of this
validation put DCA in the *optimistic* family. That was wrong for the reason just given;
the correction was stated at the top of the next message and the applied text groups DCA
with Brier as unaffected.

**Environment limitation, disclosed rather than papered over:** there is no `dpncf` conda
env on this machine and base `python3` (anaconda py3.9) lacks `catboost`, so
`retrained_models.joblib` could not be loaded. The AUROC claim was verified instead by
integrating the saved `catboost_pooled_auroc.csv` (trapezoid → 0.756, exact match); ECE
could not be recomputed and rests on the docstring.

> **User:** "can you suggest the necessary edits?" → "apply edit to line 542"

**Applied** (`9b54da3`): the pooled-diagnostics paragraph now names Brier *and* DCA as
unaffected and spells out the net-benefit decomposition.

### 5. Restructure — are pooled diagnostics "explainability"?

> **User:** "execute these in order:
> 1. rename \subsubsection{Explainability Analysis} as \subsubsection{Pooled Model Diagnostics}
> 2. create new \subsection{Explainability Analysis}
> 3. move and enhance text related to SHAP and feature importance to \subsection{Explainability Analysis}"

Then, as a check on the result:

> **User:** "are pooled diagnostics explainability tools also? They seem to be more model
> assessment tools"

Confirmed — the user's instinct was right, and it is the reason the split is worth making.
AUROC/AUPRC/calibration/DCA answer *how well does it perform*; SHAP and feature importance
answer *what is it using*. Note that the per-fold ROC/PR curves stay in Pooled Model
Diagnostics despite the section name, because they are performance curves; they are
cross-referenced from the Explainability section rather than moved.

**Applied** (`7b554d6`): `\subsubsection{Pooled Model Diagnostics}` with
`\label{sec:pooleddiagnostics}`; a new sibling `\subsection{Explainability Analysis}`
carrying `\label{sec:explainability}` and expanded paragraphs on
`get_feature_importance()` and SHAP.

### 6. Which stages refit, and why — the name-blindness problem

> **User:** "apply the cleaner alternative then suggest a fix for line 502"

**Finding A (my own error, caught before it was written).** The draft was about to say
"Both stages that consume those four persisted models begin by retraining each one."
Reading `cfreports.py` first showed this is false: the **counterfactual stage uses the
stored models directly** (`split_results[midx]['model']`); only diagnostics/explainability
refits. The deviation from a plain hoist was reported to the user rather than applied
silently.

**Finding B (factual error at line 502).** The leakage section claimed feature importance
is an out-of-sample estimate. CatBoost's default `PredictionValuesChange` is read off the
fitted trees — no test data enters the computation — so this is false by construction.
SHAP, by contrast, *is* computed on held-out test folds and does belong in that sentence.

**Related detail, deliberately hedged:** `explainability.py` passes a *callable*
(`predict_proba[:,1]`) to `shap.Explainer`, not the model, so auto-dispatch selects a
**model-agnostic** explainer rather than CatBoost's tree-path algorithm. The manuscript
says "wraps the model's positive-class probability output as a black-box function rather
than exploiting CatBoost's internal tree paths" instead of naming TreeSHAP or a specific
explainer class, since the dispatch depends on the installed `shap` version.

Then, on the refit explanation:

> **User:** "explain this further and more simply: *The diagnostics and explainability
> analyses that follow instead refit each one from its own stored outer-training partition
> under that fold's tuned hyperparameters (\texttt{best\_params}) -- the fold-specific
> configuration selected above, not a single shared configuration -- for one narrow
> reason: the search fit on a bare array, so the stored models carry only positional
> feature names, and per-feature reporting needs the real ones.*"

The underlying mechanic, unpacked: optimization fits on `dfX.values` (bare NumPy, no
headers) but persists the splits as DataFrames (real headers). The stored models therefore
identify inputs by column position only, while the downstream SHAP and importance tools
hand named tables back. The refit — same partition, same `best_params`, **same restored
`random_seed`** — is a *relabelling*, not a second fit: predictions are identical to the
stored model's.

> **User:** "apply this fix and that of line 502"

**Applied** (`7b554d6`): a "Reuse of the persisted models" paragraph at the end of
`sec:hpo` stating plainly that counterfactual generation uses the stored models directly
while diagnostics/explainability refit them to attach names; and line 502 amended with
"CatBoost's feature importances are the one exception: they are read off the fitted trees
rather than from held-out predictions, and so describe each fold's training partition by
construction."

### 7. Counterfactual feature suitability — NCS and Sudoscan

> **User (selecting line 562):** "using Table~\ref{tab:cf_features}, edit line 562 noting
> that NCS and Sudoscan are not listed as suitable for counterfactual analysis"

Framed as a wording addition; checking first showed the existing sentence was
**affirmatively wrong**, not merely incomplete. It read:

> "We note that nerve conduction studies and Sudoscan features are marked as *conditional*
> counterfactuals. These are biomarker outcomes of neuropathy. While these are not directly
> actionable inputs, they can be treated as counterfactuals that answer 'what must change'.
> Intervention is left to the physician."

Both halves fail against the table and the code:

- **NCS has no row in Table~\ref{tab:cf_features} at all.** NCS columns are dropped via
  `D.ncs_cols` before every modelling stage, and `cfreports.py:179-199` carries three
  disjointness assertions guaranteeing no NCS feature can reach `features_to_vary`.
- **Sudoscan is rated `No`, not `Conditional`.** The four Sudoscan columns
  (`FEET_MEAN_ESC`, `FEET_PCT_ASYM`, `HAND_MEAN_ESC`, `HAND_PCT_ASYM`) are listed under
  `unactionable` in `bin_cf_final_202608.yml:26`.
- **The only `Conditional` entries are PAOD and CKD.** The six features actually varied are
  INSULIN, HBA1C, HPN, PAOD, DSLPDMIA, CKD — matching the "six actionable features" in the
  `fig:counterfactual_generation` caption and the "only 6 features are eligible to vary" in
  the `tab:local_cf` caption.

The old paragraph's rhetorical move — that these could "be treated as counterfactuals that
answer what must change, intervention left to the physician" — describes something the
pipeline never does. It was preserved as a *reason for unsuitability* rather than as a
description of practice.

**Applied**, then **compressed by the user** (`828afa0`). The final text drops the NCS
sentence entirely and keeps only:

> "Sudoscan readings are not considered suitable considering that they are biomarker
> outcomes of neuropathy rather than inputs a clinician can act on. Two features are marked
> *conditional* -- peripheral arterial occlusive disease and chronic kidney disease --
> implying that they are partially manageable but not fully reversible."

This is a deliberate editorial choice worth recording: since NCS never appears in the
table, the Methods text does not need to explain its absence there — the exclusion is
already stated in the data section — and the user preferred the shorter paragraph.

---

## Commits (oldest first)

| Commit | Subject |
|---|---|
| `9dfd147` | fix auprc vs. roc-auc defense |
| `3837b49` | update sec:modelselection |
| `5523e53` | update sec:leakage for runtime reference |
| `e70fdfc` | edit class imabalance text *(sic)* — actually the stratification sentence |
| `10676ff` | edit class imbalance edit (claude's) |
| `59bdaa3` | class imbalance: personal edit |
| `d1efc26` | edit F-betas in hyperparameter section of methods |
| `9b54da3` | edit explainability analysis of methods - brier and dca |
| `81eac73` | remove f2 in sec:metrics |
| `7b554d6` | apply edits to explainability section and hpo |
| `828afa0` | edit intro to subsubsection{Feature Suitability as Counterfactuals} |

## Errors of mine, for calibration

Recorded because they show where reading the code *before* drafting was what saved the
text, and where it did not happen early enough:

- Asserted DCA belonged with the optimistic (calibration) family in a validation response;
  wrong — net benefit is a per-observation average. Corrected in the next message.
- Wrote "read off observed feature minima and maxima" for the DiCE permitted ranges; the
  code applies a ±sd margin with a 0 floor. Caught while re-reading before applying.
- Nearly wrote that both downstream stages refit the persisted models; the counterfactual
  stage does not. Caught by reading `cfreports.py` before applying.
- Drafted an edit for `sec:generalsetup` that duplicated an existing sentence at
  main.tex:602. Dropped, and reported.
- A structure grep over-escaped its pattern (`'^\\\\\(sub\)*section...'`) and silently
  returned nothing — the failure mode looks like "no matches", not like an error.

## Open items

- **`manuscript/main-claude.tex:477`** still carries the stale "marked as *conditional*"
  sentence verbatim. It was left untouched because the request named line 562 of
  `main.tex`. Decide whether that sibling file should track `main.tex`.
- **Nothing was compiled with LaTeX at any point.** The new `\label{sec:metrics}`,
  `\label{sec:pooleddiagnostics}`, the relocated `\label{sec:explainability}`, and the
  Methods→Discussion forward references to `fig:feature_importances` and `fig:shap` are
  unverified in a build.
- **ECE's pooled-vs-fold-wise direction** was argued analytically and from the docstring,
  not recomputed — `catboost` is not importable in this environment. Worth confirming in
  the `dpncf` env if the claim is challenged in review.
