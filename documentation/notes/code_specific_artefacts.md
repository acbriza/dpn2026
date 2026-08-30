# Removing code-specific artefacts from the manuscript — session notes (2026-08-30)

Sibling to `methods_edit_review.md`, which reviewed the *substance* of Methods against the
pipeline. This session audited the *register*: `manuscript/main.tex` was largely drafted by
reading the codebase, and that provenance had leaked into the prose as config keys, library
identifiers, filenames and runtime settings. The brief:

> check manuscript/main.tex thoroughly for any references to runtime settings and files.
> Since the text was generated based on reading from code base, it may contain remnants of
> references to runtime settings and files. As a journal article, it should be written in
> the standard way, referencing only to code and runtime when these are important for
> reproducibility. Do not make any changes. Just produce a numbered list of your findings
> and I'll instruct you what to do with each one.

Manuscript: `manuscript/main.tex`. Branch: `codereviewed`.
Commits produced: `d6ff2b9`, `fb2c3fe`, `292ed19`, `0a00f35` (the user committed; the
item-23 edits at the end of the session were still uncommitted when this note was written).

---

## Working agreements (consistent with the previous session)

1. **Audit first, edit only on instruction.** The three-step loop from the last session
   held again: *find* → *suggest* → *apply*. The opening request ended "Do not make any
   changes", and nothing was touched until "apply the fixes".
2. **The user triages the list, not the assistant.** The findings were numbered
   specifically so items could be accepted or deferred individually — and they were:
   23 and 29 deferred, 27/28/30 declined, the rest applied.
3. **"Suggest a fix" means show the replacement text, not apply it.** Item 23 was
   suggested in one turn and applied in the next, on a separate instruction.

---

## The governing distinction

Not every mention of a tool is an artefact. The test that ended up separating the two:

> **Would a reader reproducing this study need it, or does it only make sense to someone
> reading our source tree?**

Three outcomes followed from that test, and they need different treatment:

| Class | Example | Disposition |
|---|---|---|
| Pure runtime artefact | `sort_by: auprc`, `backend: sklearn`, `_unactionable.csv` | Delete |
| Real content in code clothing | search ranges, seeds, `iterations=500` | Keep the fact, change the typography |
| Genuine methodological constraint stated mechanically | the per-patient search budget | Keep, restate in prose |

The second row is the one worth remembering. Hyperparameter search spaces *are*
reproducibility content and must survive; what changes is `\texttt{learning\_rate}
(0.003--0.3, log-uniform)` → "learning rate (0.003--0.3, log-uniform)". Deleting these
would have been over-correction.

---

## Findings (30 items, grouped as reported)

**A. Config keys quoted verbatim (1–4).** `sort_by: auprc`, `optimization_metric: auprc`,
`threshold_selection_metric: fscore`, `threshold_delta`. In every case the surrounding
prose already stated the fact, so only the parenthetical was removed — no information lost.

**B. Library/API identifiers as prose nouns (5–13).** `RepeatedStratifiedKFold`,
`StratifiedKFold`, `random_state=42`, `best_params`, `get_feature_importance()`,
`PredictionValuesChange`, `predict_proba` (×4), `DiverseCF`, `features_to_vary`,
`generate_counterfactuals` (×2), the `dice_ml.*` object names, and the CatBoost
hyperparameter names.

**C. Filesystem and persistence (14–18).** `_unactionable.csv`; "persisted to disk"; six
occurrences of "the four persisted fold models"; "the audit trail written by the data
loader".

**D. Runtime/execution (19–21).** "each independently configured and executed" / "its
corresponding code"; the 1-hour per-patient wall-clock budget; "over 3 repeated search
attempts".

**E. Authoring remnants (22, 24, 25, 26).** A live `%% TODO-CITE` comment; "(bold in the
discussion below)" pointing at nonexistent bolding; a dangling
`\ref{sec:sufficiencynecessity}`; an abstract text corruption.

**F. Zero-indexed / pipeline-derived naming (27–29).** "Model 0"–"Model 3"; output paths
embedding `split0..split3` and `postreport/`; Figure S4 loading `s5_actionable_features.pdf`.

**G. (30)** The empty **Code availability** declaration — the standard-journal home for
everything removed in A–D, and what licenses removing it from the body.

### Disposition

- **Applied:** 1–22, 24, 25 (26 the user had already fixed independently mid-session).
- **Applied later, on a separate instruction:** 23.
- **Deferred by the user:** 29.
- **Declined for now:** 27, 28, 30.

---

## The two findings that were more than cosmetic

### 1. Item 16 — the paragraph that only existed to explain a plumbing workaround

`\textit{Reuse of the persisted models.}` in `sec:hpo` ran five sentences explaining that
the hyperparameter search fitted on "a plain numeric array", so stored models "identify
their inputs by column position only, never by name", while the stored splits are "tables
with named columns", which is why a refit is needed to get "the same model with feature
names attached".

This is a NumPy-array-vs-DataFrame detail. It is in the manuscript because a reader *of the
code* would otherwise ask why a refit happens — but a journal reader only needs the
invariant: the refit is exact, under the same fold-specific hyperparameters and seed, and
predictions are identical. Cut to three clauses.

Note this is the same invariant `methods_edit_review.md` §6 ("Which stages refit, and why —
the name-blindness problem") worked hard to establish. **The fact was right; the register
was wrong.** Worth flagging for future sessions: correctly re-deriving something from the
code does not license explaining it *as* code.

### 2. Item 23 — removing the comparison exposed a false claim

Two sites compared results against a superseded run of the pipeline ("unlike the earlier
run this Discussion previously reported", "than the previous run of this pipeline
suggested"). A revision artefact: the reader has no access to that run.

The complication only appeared while drafting the replacement. The caption said HbA1c "is
**no longer the overwhelming majority driver**" while reporting it in **86.7%** of
counterfactuals. 86.7% *is* an overwhelming majority — the sentence contradicted its own
number. It read as true only relative to the old run; strip the comparison and it is simply
false.

The absolute claim that *is* supportable, from data already in the paper:

```
changes  = 144 + 81 + 59 + 49 + 32 + 22 = 387
per CF   = 387 / 166                    = 2.33 features
cross-check, CF-count-weighted per-model sparsity means:
           (2.52*81 + 2.40*25 + 2.05*60) / 166 = 2.33   ✓
```

So HbA1c is near-universal but almost never alone: the typical reversal is HbA1c *plus one
other lever*, and which lever varies by patient. Stronger than the original claim, true in
absolute terms, and needs no prior run. Both sites now cite
`Table~\ref{tab:localcf-model-level}` for the sparsity range.

**Lesson:** a comparative claim can hide a false absolute one. When deleting the baseline
of a comparison, re-check that what remains still holds.

---

## Verification

The previous session's open item — "Nothing was compiled with LaTeX at any point" — is now
closed. Full `pdflatex → bibtex → pdflatex → pdflatex` after each batch:

- 61 pages, **0 undefined references, 0 undefined citations**.
- Removing the dangling `\ref{sec:sufficiencynecessity}` (item 25) is what cleared the last
  undefined reference; it had been warning at `main.log:790` before this session.
- Overfull hboxes: **9 before, 9 after** — verified by compiling the pre-edit backup in the
  same directory, since the count only means something against a baseline.

Two table-label rewrites initially *widened* pre-existing overfulls (`l2_leaf_reg` →
"$L_2$ leaf regularisation" cost 38pt in an already-overfull table). Tightened to
"$L_2$ regularisation" / "Class weight" / "Features to vary". The DiCE table's overfull
actually improved, 189pt → 153pt.

---

## Method notes, for next time

- **`grep` for `\texttt{}` is not sufficient.** One `scale\_pos\_weight` at main.tex:543 sat
  in plain prose and was missed by the structured scan; it surfaced only in a post-apply
  sweep for bare identifier names. Scan for the *identifiers*, not the markup.
- **Exact-string replacement with a `count(old) != 1` assertion** is the right tool for 40
  scattered prose edits — a miss fails loudly instead of silently doing nothing, and the
  script aborts before writing. One tag (`22 TODO comment`) failed on the first run because
  a Python raw string cannot carry an escaped `"`; nothing was written, so a re-run was
  clean.
- **The file changed under me mid-session.** Between the audit and the apply, the user fixed
  item 26 in the IDE, shifting most line numbers by +1. Re-derive line numbers immediately
  before editing, never trust ones captured a few turns earlier. String-based edits were
  immune; line-based ones would not have been.

---

## Open items

- **Item 23's sibling defect is still live.** The abstract reports the *old* run: HbA1c in
  "91 of 97 (93.8%)", dyslipidemia 19.6%, CKD 23.7%; the Discussion now reports 144 of 166
  (86.7%), 35.5%, 13.3%. Also AUPRC 0.885 (0.868–0.902) vs Table 5's 0.886 (0.871–0.902),
  and n = 190 / 131 vs 59 against the analysed 187 / 130 vs 57. User is handling.
- **Figure 5's artwork may over-promise.** `counterfactual_generation.pdf` probably still
  depicts the sufficiency/necessity checks whose caption text was removed with the dangling
  ref. The figure itself was not inspected.
- **Item 29 deferred:** Figure S4 (`\label{fig:s4}`) loads `s5_actionable_features.pdf`.
- **Items 27/28/30 declined for now:** zero-indexed "Model 0–3" naming; `\includegraphics`
  paths carrying `split0..split3` / `postreport/` (publishers generally require `Fig1.pdf`
  style, so this returns at submission); and the empty Code availability declaration.
- **The "among the first" claim at main.tex:704 is still unverified.** Its `TODO-CITE`
  reminder was deleted as an artefact — the claim it guarded was not checked.
