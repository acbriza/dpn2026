# Final Manuscript Review — Prompt Pack

Copy-paste prompts for the pre-submission review of `manuscript/main.tex`.
Each prompt is self-contained: run it in a **fresh Claude Code session** so that no
prior reasoning contaminates the check, and so that a long check does not crowd out
context needed by the next one.

---

## 0. How to use this pack

**Authoritative sources.** The manuscript is `manuscript/main.tex` (single file, Springer
Nature `sn-jnl` class, `sn-mathphys-num` numbered reference style). Ignore
`manuscript/main-claude.tex` and `manuscript/sn-article.tex` — the first is a superseded
copy, the second is the unmodified Springer template. Bibliography is
`manuscript/bibfile.bib` (`manuscript/sn-bibliography.bib` is template sample data).

**Two rules that apply to every prompt below:**

1. **Read-only by default.** Report findings; do not edit `main.tex` unless the prompt
   says to, or I ask in a follow-up. A review that silently rewrites is a review I cannot audit.
2. **No unverified assertions.** If a check cannot be completed from files actually
   present in the repo, say so explicitly and mark the item `UNVERIFIED` rather than
   inferring. This applies with particular force to citation content and to numbers.

**Severity scale** (use these exact labels):

| Label | Meaning |
| --- | --- |
| `BLOCKER` | Would plausibly cause desk rejection, or is factually wrong in a way that misleads a reader. |
| `MAJOR` | A reviewer would raise it as a substantive point requiring revision. |
| `MINOR` | Should be fixed before submission but would not sink the paper. |
| `NIT` | Style or polish; optional. |

**Output convention.** Every prompt should end by writing its findings to
`manuscript/references/prompts/findings/<check-name>.md`, one row per finding:

```
### F<n> — <severity> — <one-line summary>
- **Location:** main.tex:<line> (Section: <name>) [and the label, e.g. tab:localcf-listing]
- **Text as written:** "<verbatim quote, trimmed>"
- **Problem:** <what is wrong and why it matters>
- **Suggested fix:** <concrete replacement text where possible>
- **Confidence:** high | medium | low — <what would raise it>
```

Findings must be ordered most-severe first, and the file must open with a one-paragraph
summary plus counts per severity. If a check finds nothing, say so plainly — do not
manufacture findings to look productive.

**Manuscript map** (for orientation; re-derive if the file has moved on):

- Front matter: title, 11 authors, 4 affiliations, abstract, keywords — lines ~106–175
- `\section{Introduction}` ~182
- `\section{Data Collection}` ~211
- `\section{Methods}` ~395 (`Machine Learning Modeling`, `Explainability Analysis`, `Counterfactual Generation`)
- `\section{Discussion}` ~680 (`Feature Set and Model Selection`, `CatBoost Model Training and Optimization`, `Feature Importance and Model Interpretability`, `Counterfactual Analysis`, `A Proposed Clinical Usage Workflow`, `Limitations`)
- `\section{Conclusion}` ~1138
- `\section*{Declarations}` ~1172
- Appendix `Supplementary Figures and Tables` ~1214
- ~16 figure environments, ~13 table environments, ~29 `\cite` commands, 57 bib entries

**Where the numbers come from.** Every quantitative claim should trace to a generated
artifact under `manuscript/references/`, which mirrors the pipeline outputs in
`module/experiments/binary/<stage>/<model_code>/<tag>/`:

- `references/eda/` — cohort table, participant flow, univariable discrimination, supplementary EDA
- `references/postreport/selection/selection_metrics_summary.{csv,latex}` — Table of 14 algorithms
- `references/selection/summaries/`, `references/selection/violins/` — AUPRC heatmap, violins
- `references/hyperparameter_optimization/{optimization_metrics_ci.csv, catboost_first_repeat_optimization_metrics.csv, optimization_results.json}`
- `references/explainability/` — pooled PR/ROC, calibration, DCA, feature importance, SHAP (each `.png` has a sibling `.csv` where applicable)
- `references/postreport/counterfactuals/{ioi_summary_per_model, cf_fulltable, cf_changed_features}.{csv,latex}` plus case-study and global plots
- `references/counterfactuals/<patient-code>/` — per-patient counterfactual outputs
- `references/illustrations/` — hand-built pipeline and workflow diagrams

---

# Part A — The checks you asked for

---

## P1 — No coding-specific language

**Purpose:** the manuscript must read as clinical/scientific writing, not as documentation
of a codebase. A reviewer who sees `n_estimators` or `sklearn` in the prose concludes the
methods were written by transcribing a script.

```text
You are copy-editing a clinical machine-learning manuscript for a medical journal
audience. The manuscript is manuscript/main.tex. Read it in full (it is ~1400 lines).
Do not edit it — report only.

Find every place where the prose leaks software-implementation vocabulary that a
clinician-reader would not use, and that carries no scientific content a plain-language
phrase could not carry. Specifically flag:

1. Library, framework, and API names used as if they were methods: scikit-learn, sklearn,
   pandas, numpy, Optuna, joblib, DiCE/dice_ml class names, matplotlib, xgboost,
   lightgbm, shap. Naming the software once in Methods with a version is correct and
   expected — keep those. Flag repeat mentions and mentions inside Results/Discussion prose.
2. Code identifiers rendered as text: snake_case or camelCase names, function or method
   calls with parentheses, keyword-argument syntax (foo=bar), file names with extensions
   (.py, .yml, .csv, .joblib), config-file or directory paths, class names
   (dice_ml.Data, dice_ml.Model, dice_ml.Dice), and hyperparameter names given in
   library spelling rather than statistical English (n_estimators, learning_rate,
   l2_leaf_reg, scale_pos_weight, random_state, n_jobs).
3. Engineering jargon standing in for a scientific term: "run the script", "the pipeline
   was executed", "config", "flag", "boolean", "array", "dataframe", "loop", "iterate",
   "hard-coded", "overwrite mode", "instantiated", "object", "parsed", "the module",
   "seed" used without explanation, "training loop", "callback", "wrapper", "the codebase".
4. Variable codes from the dataset used without a plain-language gloss at first use in
   the main text (DEC_AR, HAND_MEAN_ESC, FEET_MEAN_ESC, NS, GBS, MNSI item codes, and
   similar). A table of codes exists in the manuscript; using a raw code in prose is only
   acceptable after the reader has been given its meaning in that sentence or in the
   immediately preceding one.

For each hit give: line number, the verbatim phrase, why it reads as code rather than
science, and a concrete replacement sentence in clinical register that preserves the
technical meaning exactly. Do not weaken a methodological statement in the name of
readability — if the implementation detail is load-bearing for reproducibility, say so
and propose a rephrasing that keeps it (e.g. move it to a Methods parenthetical or a
supplementary note) rather than deleting it.

Explicitly exclude from your report: figure/table captions' references to file paths that
are LaTeX \includegraphics arguments, LaTeX commands and comments, and anything inside
listings or verbatim environments — those are not prose.

Separately, list any place where the OPPOSITE problem occurs: a modeling decision is
described so loosely that a reader could not tell what was actually done.

Write findings to manuscript/references/prompts/findings/p1-coding-language.md using the
pack's severity scale and finding format.
```

---

## P2 — Integrity of the figures reported

**Purpose:** a figure that no longer matches the run that produced it is the single most
damaging error class in this manuscript, because the pipeline has been re-run and outputs
were copied by hand into `manuscript/references/`.

```text
You are auditing the integrity of every figure in the manuscript manuscript/main.tex
against the artifacts it claims to display. Do not edit the manuscript — report only.

For every figure environment in main.tex (there are ~16 in the main text plus
supplementary ones in the appendix), establish the following and report each as
PASS / FAIL / UNVERIFIED:

1. FILE EXISTS. The \includegraphics path resolves to a real file relative to
   manuscript/. Report any missing file as BLOCKER.
2. PROVENANCE. Identify which pipeline stage produced it and locate the corresponding
   canonical output under module/experiments/binary/<stage>/... . Compare the copy in
   manuscript/references/ with the canonical output: same file (byte-identical or same
   size/mtime lineage)? If the manuscript copy is older than the canonical output, or the
   canonical output does not exist, that figure may be from a superseded run — report it
   as BLOCKER with both timestamps. Use file modification times and, where a sibling .csv
   exists, compare the plotted values.
3. CAPTION NUMBERS. Every number stated in the caption must be checkable against the
   figure's own source data. Where a sibling .csv exists (e.g.
   references/explainability/catboost_pooled_auprc.csv,
   references/postreport/counterfactuals/*.csv), recompute the caption's claims from it
   and report any mismatch with both values. Examples of caption claims that must be
   verified this way: the pooled AUPRC 0.858 (95% CI 0.795-0.916), the mean-of-folds
   0.893 +/- 0.043, the no-skill baseline 0.70 = 130/187, the model AUROC 0.799 and the
   strongest single feature DEC_AR at 0.748, HbA1c changed in 144 of 158 counterfactuals
   (91.1%), insulin 81 (51.3%), dyslipidemia 59 (37.3%), hypertension 49 (31.0%), PAOD
   32 (20.3%), CKD 22 (13.9%), the per-model counts (Model 0: 69 HbA1c / 43 insulin / 41
   dyslipidemia; Model 3: 51 / 31; Model 1: 24 counterfactuals with hypertension 15 and
   PAOD 11), and the sparsity range 2.16-2.65. Verify all of them and any others you find.
4. CAPTION DESCRIBES THE IMAGE. Open each image file and confirm the caption's structural
   claims are true of the rendered figure: the panels it says exist (a)/(b)/(c) are
   present and in the stated order; the marker/color encoding it describes (e.g. "diamonds
   denote the 6 features the counterfactual engine may vary", "filled marker = BH FDR <
   0.05", "red = decrease, blue = increase", node colors in the workflow diagram) matches
   what is drawn; axis directions and reference lines (dashed no-discrimination line at
   0.5, dotted model line, prevalence baseline) exist as described.
5. INTERNAL CONSISTENCY ACROSS FIGURES. Where two figures show the same quantity, the
   values must agree — e.g. the per-model counterfactual counts in the global counts
   figure must sum to the per-patient totals in the changed-features figure, and both must
   agree with the counterfactual tables. Report any disagreement, quoting all sources.
6. STALE-RUN SWEEP. List every file under manuscript/references/ that is referenced by
   main.tex, with its mtime, sorted oldest first, and flag any that predate the newest
   file from the same stage — those are the likely stale ones.

If you cannot open an image to verify a visual claim, mark that specific sub-check
UNVERIFIED and say why; do not assume the caption is right.

Write findings to manuscript/references/prompts/findings/p2-figure-integrity.md, plus a
full PASS/FAIL matrix table (one row per figure).
```

---

## P3 — Coherence among figures, tables, and prose

**Purpose:** the numbers in the running text, the numbers in the captions, and the numbers
in the tables must be one story. This is the check reviewers actually perform.

```text
You are checking numerical and narrative coherence across manuscript/main.tex. Do not
edit — report only.

Build, first, a complete inventory as a table: every figure and table, its label, its
caption's one-line subject, its source artifact under manuscript/references/, and every
line number in the main text where it is referenced by \ref or discussed by name.

Then check all of the following:

1. EVERY FLOAT IS CITED. Every figure and table must be referred to at least once in the
   running text with \ref. List any orphan float (there appear to be ~54 \label commands
   and ~44 distinct \ref targets, so orphans likely exist). Also list any \ref that points
   at a label that does not exist.
2. ORDER OF FIRST MENTION. Floats should be first mentioned in numerical order. Report
   any float that is discussed out of sequence.
3. NUMBER-BY-NUMBER AGREEMENT. Extract every quantitative claim in the running prose that
   restates something from a table or figure, and check it against BOTH the caption and
   the underlying source artifact (.csv/.latex under manuscript/references/). Report a
   finding whenever the three disagree, quoting all three values with their locations.
   Pay particular attention to: cohort counts (201 screened / 190 enrolled / 187 analyzed
   / 130 Confirmed / 57 Unconfirmed, and the 3 dropped codes 36, 46, 173); selection-stage
   metrics (Random Forest 0.892, CatBoost 0.889, Extra Trees 0.885, naive baseline 0.695/
   0.70); tuned-model metrics (AUPRC 0.886 with CI 0.871-0.902, sensitivity 0.886 with CI
   0.861-0.912, specificity 0.527 with CI 0.483-0.571); pooled vs per-fold AUPRC (0.858
   vs 0.893 +/- 0.043); counterfactual yield (8 of 61 candidates, 42 misclassified, 9 also
   borderline, 158 counterfactuals, Model 2 = 0); the six actionable features and the
   count "6"; the 22 candidate predictors and the 2.6 events-per-variable figure; the
   borderline margin 0.08 and the reported margins <= 0.07; the 4 folds x 10 repeats = 40
   runs and the 100-trial / 3-fold inner search.
4. DEFINITIONS MATCH USAGE. A quantity must mean the same thing everywhere it appears.
   Check specifically that: "AUPRC" is always the same estimator (per-fold mean vs pooled)
   and that the manuscript is explicit about which one each number is; "candidates",
   "instances", "counterfactuals", and "patients" are used consistently and their counts
   nest correctly; "model" (one of four fold models) is never confused with "experiment"
   or "study"; threshold values are always attributed to the specific fold they came from.
5. PROSE CLAIMS THE FIGURE SUPPORTS. For each interpretive sentence that points at a
   figure ("X shows that...", "as seen in Fig. Y"), confirm the figure can actually
   support that claim. Flag cases where the prose asserts a comparison, trend, or ranking
   the figure does not display, or asserts a difference between values whose stated
   uncertainty intervals overlap heavily.
6. TABLE-TO-TABLE CONSISTENCY. Rows that appear in more than one table (e.g. the
   per-fold CatBoost models across the optimization and counterfactual tables) must carry
   identical values, thresholds, and fold indices.
7. ABBREVIATION AND CODE CONSISTENCY between captions and prose (e.g. HbA1c vs glycated
   hemoglobin vs A1c; PAOD; CKD; NCS; DEC_AR).

Write findings to manuscript/references/prompts/findings/p3-coherence.md, leading with
the inventory table, then findings most-severe first. Any prose/caption/source
disagreement in a headline number is a BLOCKER.
```

---

## P4 — Fidelity of citations to their sources

**Purpose:** a citation that does not say what the sentence claims it says is a research
integrity problem, not a typo.

```text
You are verifying that every citation in manuscript/main.tex faithfully supports the claim
it is attached to. Do not edit — report only.

Available source material:
- manuscript/rrl/articles/ holds full-text PDFs for 14 cited works. Read the actual PDFs.
- manuscript/references/quotables/quotables.md, .../citables.md, and
  manuscript/rrl/quotables_review.md hold previously extracted quotes; treat these as
  leads, not as evidence — verify against the PDF itself where the PDF exists.
- manuscript/bibfile.bib holds all 57 entries.

Method, applied to every \cite in the main text (~29 citation commands, often
multi-key):
1. Quote the citing sentence verbatim with its line number, and list the keys cited.
2. Resolve each key in bibfile.bib. Check the metadata itself: authors, year, title,
   journal, volume/pages, DOI. Flag entries with missing DOI, wrong year, mangled title
   capitalization, "et al." baked into an author field, or duplicate entries for the same
   work.
3. Classify the claim being supported: (a) a specific numeric or factual claim
   ("underdiagnosed in up to 81%", "42-58% of Filipinos", "AUCs ranging from 0.64 to
   0.93"); (b) a methodological attribution ("DiCE formulates counterfactual generation as
   a constrained optimization problem", "the Youden Index is defined as..."); (c) a broad
   topical gesture ("XAI has gained attention [5,6,7]").
4. For every (a) and (b) claim where the source PDF is present in manuscript/rrl/articles/:
   open the PDF, locate the supporting passage, and quote it with its page or section.
   Then judge: SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED / CONTRADICTED. For
   PARTIALLY, say precisely what the source qualifies that the manuscript drops (population,
   setting, sample size, effect direction, whether it was a primary finding or a cited
   claim in that paper's own introduction).
5. For claims whose source is NOT in manuscript/rrl/articles/, mark them UNVERIFIED and
   list them separately with the exact passage that would need checking. Do not guess at
   the content of a paper you have not read, and do not rely on your background knowledge
   of a well-known paper as if it were verification — say "not verifiable from repo".
6. Flag citation-laundering: a claim attributed to a source that itself cites another
   source for it. The 81% underdiagnosis figure and the 42-58% prevalence figure are the
   likeliest instances; check whether the cited works are the origin of those numbers or
   are relaying them, and report the primary source if the PDF reveals it.
7. Flag citation stuffing (a string of keys where only one is on-point) and thin support
   (a strong claim leaning on a single non-systematic source).
8. Check the DiCE description in Methods against Mothilal et al. 2020 specifically: the
   manuscript says DiCE is used with a GENETIC algorithm, while it also describes DiCE as
   an optimization-based framework. Confirm the description of the method actually used
   matches both the cited paper and what the code does (module/utils2/counterfactuals.py).

Write findings to manuscript/references/prompts/findings/p4-citation-fidelity.md with
three sections: Verified-against-source, Unverifiable-from-repo, and Metadata problems.
Any NOT SUPPORTED or CONTRADICTED finding is a BLOCKER.
```

---

## P5 — Important, strong, and possibly over-reaching claims

**Purpose:** this is a 187-patient single-center study with a 8/61 counterfactual yield.
The gap between what was shown and what is claimed is where reviewers will concentrate.

```text
You are stress-testing the claims in manuscript/main.tex for over-reach. Do not edit —
report only.

Extract every claim that asserts something beyond a bare description of a result, and
build a claim ledger. For each entry give: the verbatim sentence, its line, the section,
the evidence the manuscript itself offers for it, and a verdict:

- SUPPORTED — the study's own design and results establish it.
- OVERSTATED — true in kind but stronger than the evidence in degree, scope, or certainty.
- UNSUPPORTED — no evidence in this manuscript establishes it.
- CAUSAL OVERREACH — an associational result stated in causal or interventional language.
- GENERALIZATION OVERREACH — a single-center Filipino cohort of 187 used to speak about
  populations, settings, or clinical practice generally.

Apply these specific tests, which this study's design makes decisive:

1. CAUSAL LANGUAGE AROUND COUNTERFACTUALS. A DiCE counterfactual is a statement about a
   model's decision boundary, not about patient physiology. Flag every sentence that
   slides from "changing this feature flips the model's prediction" to "changing this
   feature would reduce the patient's risk / improve the outcome / prevent neuropathy".
   Verbs to scrutinize: would reduce, leads to, results in, prevents, improves, achieves,
   drives, causes, enables, ensures.
2. CLINICAL DEPLOYMENT LANGUAGE. The proposed workflow is explicitly a proposal. Flag any
   sentence anywhere else in the manuscript (abstract, introduction, conclusion) that
   presents it as validated, ready, deployable, or as a tool clinicians can use now, and
   any use of "clinical decision support" without the proposal qualifier.
3. PERFORMANCE FRAMING. Check every superlative and comparative: "high performance",
   "robust", "strong discrimination", "outperforms", "state of the art", "excellent". Test
   each against the actual numbers: specificity 0.527 (CI 0.483-0.571), pooled AUPRC 0.858
   against a no-skill baseline of 0.695, and the manuscript's own observation that the best
   single feature reaches AUROC 0.748 against the model's 0.799. Any claim of a meaningful
   gain over the single-feature or the naive baseline must be checked against overlapping
   intervals; flag ranking claims among models whose means differ by less than one SD
   (Random Forest 0.892 / CatBoost 0.889 / Extra Trees 0.885).
4. YIELD AND NEGATIVE RESULTS. Counterfactuals were obtained for 8 of 61 candidates and
   one entire fold model produced none. Flag any generalization about "the counterfactual
   analysis shows..." or "HbA1c is the most actionable feature" that rests on 8 patients
   and 158 counterfactuals without stating that base. The 91.1% HbA1c figure is a
   percentage of counterfactuals, not of patients — flag any place it reads as the latter.
5. MODEL-SELECTION JUSTIFICATION. The manuscript says CatBoost was carried forward "for
   its fit to a small cohort and its smoother probability surface for counterfactual
   search" despite not being top-ranked. Check whether the manuscript actually demonstrates
   the smoother-surface claim or merely asserts it; if asserted, it must be labeled as a
   rationale, not a finding.
6. NOVELTY CLAIMS. Any "first to", "novel", "unlike previous studies" must be checked
   against the cited literature, including the two 2025 systematic reviews in
   manuscript/rrl/articles/ (Ma; Sun). Flag unverifiable priority claims.
7. HEDGING CALIBRATION IN BOTH DIRECTIONS. Also flag under-claiming: places where a
   genuine, defensible finding is buried in hedges. Over-hedging reads as low confidence
   and invites reviewers to doubt the whole.
8. ABSTRACT AND CONCLUSION SPECIFICALLY. These are read hardest. Check each sentence
   against what the body actually establishes, and flag any claim that appears in the
   abstract or conclusion in a stronger form than in the Discussion.

For every OVERSTATED / UNSUPPORTED / OVERREACH finding, supply a rewritten sentence that
keeps the finding but sizes the claim to the evidence.

Write findings to manuscript/references/prompts/findings/p5-claims.md, leading with the
full claim ledger table, then the detailed findings.
```

---

## P6 — American English spelling and construction

```text
You are enforcing US English throughout manuscript/main.tex. Do not edit — report only,
as a line-by-line list I can apply.

Report every instance of:
1. British spellings: -ise/-isation for -ize/-ization (analyse, optimisation, normalise,
   randomised, characterise, recognise, summarise, utilise), -our (behaviour, colour,
   favour, labour, tumour), -re (centre, fibre, metre, litre), -ll- doubling (modelled,
   labelled, travelled, cancelled, signalling, fuelled), -ae-/-oe- medical digraphs
   (haemoglobin, anaemia, oedema, foetal, paediatric, aetiology), -ogue (analogue,
   catalogue, dialogue), defence/offence/licence/practise as verb-noun pairs, programme,
   grey, sulphur, ageing, judgement, enquire, whilst, amongst, towards (prefer toward),
   learnt/spelt/burnt, storey, kerb, manoeuvre.
   NOTE: -yse/-yze — "analyse/analyze" — and "modelling/modeling" are the two most likely
   to appear in an ML manuscript; check every occurrence.
2. British punctuation and construction: logical quotation (period/comma outside the
   closing quote) where US style puts it inside; single quotation marks as primary;
   "different to"; "in hospital" without an article; collective nouns taking a plural verb
   ("the team have", "the cohort were"); dates in "12 March 2023" form where the journal
   or US convention wants "March 12, 2023"; missing serial/Oxford comma (US scientific
   style generally uses it — flag inconsistency either way and recommend one).
3. Mixed usage: if the manuscript uses both spellings of the same word anywhere, that is a
   MAJOR finding regardless of which one dominates.

Do NOT change: proper nouns, institution names, journal titles, or anything inside
bibfile.bib entries or \cite keys (bibliography style handles those), and do not "correct"
a quoted passage from a source.

Report as a table: line | current | replacement | note. Then give me a single sed-ready
list of exact substitutions at the end, and state the total count. Also state explicitly
if the manuscript is already consistently US English — that is a valid and useful result.

Write to manuscript/references/prompts/findings/p6-american-english.md.
```

---

## P7 — Language and grammar

```text
You are line-editing manuscript/main.tex for language and grammar at the standard of a
Springer clinical-informatics journal. Do not edit the file — produce a correction list.

Work section by section (Abstract, Introduction, Data Collection, Methods, Discussion and
each of its six subsections, Conclusion, all captions, appendix). For each, report:

1. OUTRIGHT ERRORS: subject-verb disagreement, tense inconsistency, dangling and
   misplaced modifiers, faulty parallelism in lists, pronoun reference with no clear
   antecedent (especially "this" or "it" opening a sentence), comma splices, run-ons,
   sentence fragments, article errors (a/an/the — including missing articles before
   singular count nouns), preposition errors, and subject-verb number errors around
   collective and technical nouns ("data" as plural, "criteria/criterion").
   Known typo to confirm and fix: "Hurdles in the early diagnosis of DPN includes"
   (Introduction) — verify and check for others of the same kind.
2. TENSE DISCIPLINE: what was done in this study = past tense; what the data or a figure
   shows = present; established knowledge = present. Flag every violation, especially
   Methods sentences drifting into present tense and Discussion sentences drifting into
   future.
3. VOICE AND PERSON: check that first-person plural ("we") is used consistently or not at
   all, and flag mixed usage. Flag agentless passives that hide who did what where it
   matters methodologically ("the threshold was chosen" — by what procedure?).
4. SENTENCE-LEVEL READABILITY: flag sentences over ~45 words, sentences with three or more
   subordinate clauses, and stacked-noun phrases of four or more nouns
   ("fold model decision threshold calibration procedure"). Give a split or rewrite for
   each. This manuscript makes heavy use of em-dashes and long appositive chains,
   particularly in captions — flag captions where the syntax has outrun the reader.
5. PUNCTUATION AND TYPOGRAPHY IN LATEX: en-dash vs hyphen vs em-dash usage (numeric ranges
   take --, "0.795--0.916"); \% escaping; non-breaking spaces before \ref and \cite; "Fig."
   vs "Figure" consistency; spacing after abbreviations that end a sentence; italics for
   Latin abbreviations handled consistently (e.g., i.e., et al., vs., cf.) and each
   followed by the correct punctuation; consistent decimal precision (do not mix 0.89 and
   0.886 for the same quantity); consistent use of the multiplication sign (the manuscript
   currently mixes "x" and $\times$ — check "10-repeat × 4-fold" against "4 folds $\times$
   10 repeats").
6. WORDINESS: report the twenty worst instances of padding ("it is important to note
   that", "in order to", "due to the fact that", "a total of", "in the present study") with
   tightened replacements.

For every finding give line number, verbatim original, corrected version, and a
one-clause reason. Do not rewrite for style where the original is correct — I want
errors and genuine clarity failures, not preference edits.

Write to manuscript/references/prompts/findings/p7-language-grammar.md, organized by
section, with a per-section error count table at the top.
```

---

## P8 — Repetition within a section

```text
You are checking manuscript/main.tex for redundancy WITHIN each main section. Do not edit
— report only.

Ground rule you must respect: repetition ACROSS the Abstract, Introduction, Discussion,
and Conclusion is expected and correct — a point may legitimately appear in all four. Do
NOT report those as redundancy. What you are hunting is a point made twice inside the
SAME main section, where the second statement adds nothing the first did not already give
the reader.

Procedure:
1. For each main section, and for each subsection of the Discussion, build a list of the
   distinct substantive points made, in order, each with its line number and a one-line
   paraphrase. Treat the six Discussion subsections as separate units for the intra-section
   test, but ALSO run the test across the Discussion as a whole, since a point restated in
   two different Discussion subsections is a real redundancy a reviewer will notice.
2. Report every pair of points within the same unit that are substantively the same claim.
   For each pair, quote both, and say which one should be kept and why (usually: keep the
   one where the supporting evidence is presented, cut or cross-reference the other).
3. Flag the specific recurring motifs of this manuscript and count where each is asserted:
   - that the top ensemble models are statistically indistinguishable;
   - that specificity is the weak point / drives unnecessary referrals;
   - that the important features are markers of established neuropathy rather than
     treatable targets;
   - that only a small number of features are actionable, which explains the large
     feature displacements;
   - that Model 2 produced no counterfactuals / DiCE's genetic search failed to converge;
   - that HbA1c dominates but rarely acts alone;
   - that the workflow is a proposal, not a validated protocol;
   - that the cohort is small and single-center.
   For each motif, list every occurrence with line numbers, and recommend which
   occurrences to keep given the across-section allowance.
4. Flag caption/prose duplication: captions in this manuscript are unusually long and
   several appear to restate the interpretive argument given in the body text. A caption
   should be self-contained but should not duplicate a paragraph of the Discussion
   verbatim in substance. Report each instance with both texts side by side and recommend
   what belongs where.
5. Flag near-verbatim sentence reuse anywhere in the file (the same sentence or clause
   appearing twice), which is always a defect even across sections.

Write to manuscript/references/prompts/findings/p8-repetition.md with the per-section
point inventory first, then the redundancy findings, then the motif occurrence table.
```

---

## P9 — Desk-rejection sweep

**Purpose:** editors reject before review for completeness and compliance failures, not for
science. Run this one **last** and re-run it after every batch of edits.

```text
You are the handling editor of a Springer Nature clinical-informatics journal doing the
initial technical check on manuscript/main.tex. Your job is to decide whether this
submission gets sent for review or returned to the authors. Be unsparing. Do not edit —
report only.

Check every item below and return a verdict per item: READY / MUST FIX BEFORE SUBMISSION /
CANNOT ASSESS.

A. TEMPLATE AND PLACEHOLDER RESIDUE — this is the fastest desk rejection there is.
   1. Search the entire file for leftover Springer template text and placeholders. There
      are two \abstract commands in this file (one is the template's "The abstract serves
      both as a general introduction..." boilerplate, the other is the real structured
      abstract). Confirm this, determine which one LaTeX actually uses, and report it as
      a BLOCKER regardless.
   2. Eight co-author \email fields contain the literal string "email address or ORCID".
      Confirm and report. Check corresponding-author designation and that every author has
      an affiliation.
   3. Any remaining lorem-ipsum, "sample", "e.g. this is an example", instructional
      comments left uncommented, or template figure/table examples.
B. DECLARATIONS. The Declarations section is currently a bare itemize list of headings
   with no content. Every one of these must be present with real text: Funding; Conflict
   of interest / Competing interests; Ethics approval and consent to participate (with the
   IRB/ERB name and approval number — this is a human-subjects study at East Avenue
   Medical Center); Consent for publication; Data availability; Materials availability;
   Code availability; Author contributions (CRediT-style). Report each missing one
   separately as a BLOCKER. Acknowledgements is currently empty — confirm that is intended.
C. HUMAN-SUBJECTS AND PRIVACY. This manuscript discusses individual patients by their
   study code (Patients 20, 40, 76, 123, 172 and others) alongside clinical detail.
   Assess whether the ethics statement covers publication of individual-level data, whether
   consent for publication is documented, and whether any combination of reported
   attributes could identify a patient in a single-center cohort of 187. Report any risk.
D. STRUCTURE. The manuscript presents its results inside the Discussion section and has no
   separate Results section. Assess whether this will survive an editorial structure check
   for a research article at a clinical journal (IMRaD is near-universally expected), and
   state plainly what the risk is and what restructuring would cost.
E. REPORTING GUIDELINE. A clinical prediction-model paper is normally expected to follow
   TRIPOD+AI (or CLAIM). Report whether the manuscript names a guideline, whether a
   completed checklist is referenced as supplementary material, and list the TRIPOD+AI
   items that appear to be missing outright.
F. TRIAL/STUDY REGISTRATION, and whether the journal would expect one for this design.
G. FORMAT COMPLIANCE. Word counts (abstract and whole text) against typical limits; figure
   count (~16 main-text figures is high — most journals cap at 6-8 and push the rest to
   supplementary); table count (~13, same issue); reference count (57); whether figures
   are supplied as separate files as the template requires rather than embedded; whether
   the document class option matches the target journal's reference style.
H. BUILD HEALTH. Run the LaTeX build and report: undefined references, undefined
   citations, multiply-defined labels, missing graphics, overfull/underfull boxes,
   and any font or package warning. Quote the log lines. A PDF that does not compile
   cleanly from the submitted single .tex is a rejection risk.
I. FIGURE FILE HEALTH. Any figure supplied at low resolution, any raster PNG where a
   vector PDF exists in the pipeline outputs, any figure whose text would be illegible at
   print width.
J. AI-USE DISCLOSURE. Most publishers now require a statement if generative AI was used in
   preparation. Report whether one is present and flag it as needed if not.
K. TITLE, KEYWORDS, ORCIDs, and whether the corresponding author's contact details are
   complete.
L. ANYTHING ELSE that would make you return this without review. Say it directly.

Finish with an explicit editorial verdict in one paragraph: would you desk-reject this
submission as it stands, and what is the minimum set of fixes that changes that answer.

Write to manuscript/references/prompts/findings/p9-desk-rejection.md, with the BLOCKER
list as a numbered pre-submission checklist at the very top.
```

---

# Part B — Further checks

P10-P25 are checks I proposed; P26 and P27 were requested. The acronym first-use
audit is folded into P17 rather than given its own prompt.

---

## P10 — Abstract-to-body numerical consistency

**Why:** the abstract is the most-read and most-checked text in the paper, and it currently
carries a dozen numbers that must each match the body exactly.

```text
Check every factual and numerical claim in the abstract of manuscript/main.tex (the
structured Background/Methods/Results/Conclusions abstract, not the template placeholder)
against the body of the manuscript and against the source artifacts under
manuscript/references/. Do not edit — report only.

For each abstract claim produce a row: claim | value in abstract | value in body (with
line) | value in source artifact (with file) | verdict.

Verify at minimum: the 81% underdiagnosis figure; 190 recruited and 187 with complete
data; 130 Confirmed / 57 Unconfirmed; Toronto consensus criteria; the exclusion of NCS
from every predictor set; 14 algorithms and 12 feature subsets; ten repeats of stratified
four-fold CV; AUPRC as the selection metric; "best 0.892 versus a no-skill baseline of
0.695"; mean AUPRC 0.886 (95% CI 0.871-0.902); sensitivity 0.886 (0.861-0.912);
specificity 0.527 (0.483-0.571); "Sudoscan plus examination alone came within 0.01 of the
full feature set"; the four named top features; six actionable features; "8 of 61
candidate patients"; HbA1c 91.1%, insulin 51.3%, dyslipidemia 37.3%, hypertension 31.0%;
the 2022-2023 recruitment window and the Quezon City tertiary hospital setting.

Also check: does the abstract claim anything the body does not establish, and does the
body contain a headline result the abstract omits? Is the abstract within a 200-250 word
budget? Does it contain citations, equations, or undefined abbreviations (journals
routinely forbid all three in abstracts)?

Write to manuscript/references/prompts/findings/p10-abstract-consistency.md.
```

---

## P11 — Internal arithmetic and unit audit

**Why:** percentages, sums, and derived quantities are cheap to check mechanically and
embarrassing to get wrong in print.

```text
Recompute, from first principles, every derived number in manuscript/main.tex. Do not
edit — report only. Where a source .csv exists under manuscript/references/, compute from
that; otherwise compute from the numbers the manuscript itself gives, and say which.

Check:
1. Every percentage against its stated numerator and denominator (e.g. 130/187 = 69.5%,
   144/158 = 91.1%, 81/158 = 51.3%, 59/158 = 37.3%, 49/158 = 31.0%, 32/158 = 20.3%,
   22/158 = 13.9%). Report any that do not round correctly, and flag any percentage whose
   denominator is not stated or is ambiguous.
2. Every count that must sum: 201 screened - 11 excluded = 190 enrolled; 190 - 3 = 187;
   130 + 57 = 187; misclassified + borderline-correct = candidates (42 + ? = 61); per-model
   counterfactual counts summing to 158; per-patient counts summing to per-model counts;
   4 folds x 10 repeats = 40 runs.
3. Every stated ratio and derived index: the 2.6 minority-class events per candidate
   predictor across 22 features (check: 57/22 = 2.59); the Youden index wherever reported
   (sensitivity + specificity - 1) against the sensitivity and specificity given in the
   same table.
4. Confidence intervals: check that every reported CI brackets its own point estimate,
   that widths are plausible for n=187 and for 40 runs, and that the manuscript states
   what each CI is over (runs? bootstrap resamples? patients?) — an unlabeled CI is a
   finding.
5. Precision and units: consistent decimal places for the same quantity throughout; HbA1c
   changes stated in the correct unit (percentage points, not percent — flag every
   instance where "%" is used for a change in HbA1c); ESC values in microsiemens; ages in
   years; any quantity reported without units.
6. Rounding consistency between a table's stored value and the prose's rounded restatement.

Write to manuscript/references/prompts/findings/p11-arithmetic.md as a table of every
checked quantity with computed vs stated values, then findings for the mismatches.
```

---

## P12 — TRIPOD+AI / reporting-guideline compliance

**Why:** clinical prediction-model journals increasingly require it, and working through
the checklist surfaces genuine methodological omissions rather than cosmetic ones.

```text
Assess manuscript/main.tex against the TRIPOD+AI reporting guideline for prediction-model
studies (fall back to TRIPOD 2015 plus CLAIM where TRIPOD+AI items are ambiguous). Do not
edit — report only.

Produce the full checklist as a table: item number | item text (abbreviated) | present? |
where in the manuscript (line/section) | what is missing.

Give particular attention to items this study's design makes load-bearing:
- Source of data, eligibility criteria, setting, and dates of recruitment.
- Outcome definition, how and by whom it was assessed, and blinding of outcome assessment
  to predictors (here: the Toronto criteria and the NCS that defines the label).
- Predictor definitions, timing of measurement relative to outcome, and blinding.
- Sample size justification and events-per-variable.
- Missing data: how much, in which variables, and how handled (the pipeline drops rows;
  is the amount and the mechanism reported?).
- Model-building: all candidate predictors, selection procedure, and where in the CV
  structure selection happened.
- Model performance measures: discrimination AND calibration. The repo contains
  calibration and decision-curve outputs
  (references/explainability/catboost_pooled_calibration.png/.csv and
  catboost_pooled_dca.png/.csv) that the manuscript does not appear to present — flag
  this as a MAJOR omission, since TRIPOD requires calibration reporting and the material
  already exists.
- Internal validation procedure and whether optimism was addressed.
- Model presentation: can a reader apply the model? (weights, code, or a deployable
  artifact)
- Interpretability/AI-specific items: fairness or subgroup assessment, the explanation
  methods used and their limitations, human oversight.
- Availability of data, code, and the trained model.

Finish with a prioritized list of the smallest set of additions that would bring the
manuscript to substantial compliance, and note which of them can be satisfied entirely
from artifacts already present in manuscript/references/.

Write to manuscript/references/prompts/findings/p12-tripod-ai.md.
```

---

## P13 — Statistical reporting rigor

```text
Audit the statistical reporting in manuscript/main.tex. Do not edit — report only.

Check:
1. UNCERTAINTY EVERYWHERE. Every point estimate presented as a result should carry an
   interval or an SD, and the manuscript must say what the spread is over (40 CV runs, 4
   folds, bootstrap resamples, patients). List every naked point estimate.
2. NO INFERENCE FROM OVERLAPPING INTERVALS. Flag every comparative or ranking claim
   between quantities whose intervals overlap substantially — in particular the top three
   selection-stage models and any feature-set comparison. State the overlap explicitly in
   each finding.
3. CV VARIANCE IS NOT SAMPLING VARIANCE. Repeated-CV spread across 40 runs on the same 187
   patients understates uncertainty about a new patient population, because the runs are
   not independent. Check whether the manuscript states this; if it presents CV-derived
   CIs as if they were external-validity intervals, that is a MAJOR finding.
4. MULTIPLICITY. The univariable discrimination figure uses Benjamini-Hochberg FDR — check
   that the correction's family is defined (over how many tests) and applied consistently,
   and that nothing elsewhere reports uncorrected significance.
5. CLASS IMBALANCE AND METRIC CHOICE. Confirm the manuscript justifies AUPRC over AUROC
   for a 69.5% prevalence problem, reports the no-skill baseline alongside every AUPRC,
   and does not report accuracy as if it were informative at this prevalence.
6. THRESHOLD REPORTING. Thresholds are tuned per fold. Check that every sensitivity/
   specificity is attributed to a specific threshold and that the manuscript never
   compares threshold-dependent metrics across folds without noting the thresholds differ.
7. POOLED VS MEAN-OF-FOLDS. The manuscript reports both (0.858 pooled, 0.893 mean-of-folds)
   and explains they are different statistics. Verify the explanation is correct and that
   every other metric in the manuscript makes clear which of the two it is.
8. CALIBRATION. Discrimination without calibration is incomplete for a clinical risk
   score. Note whether calibration is reported at all (see the unused artifacts in
   references/explainability/).
9. SAMPLE SIZE AND OVERFITTING. 57 minority events, 22 predictors, 100-trial
   hyperparameter search per fold. Assess whether the manuscript acknowledges the
   optimism this induces, and whether "2.6 events per predictor" is stated in the main
   text and not only in a caption.
10. Any p-value present: check exact reporting (not "p<0.05"), the test named, assumptions
    stated, and that significance is never equated with clinical importance.

Write to manuscript/references/prompts/findings/p13-statistics.md.
```

---

## P14 — Methodological validity and leakage audit (manuscript vs code)

**Why:** the manuscript's Methods must describe what the code actually does. This is the
one check that must read the pipeline, not just the prose.

```text
Verify that the Methods section of manuscript/main.tex faithfully and completely describes
what the pipeline actually does. Do not edit the manuscript — report discrepancies.

Read the manuscript's Methods (Machine Learning Modeling, Explainability Analysis,
Counterfactual Generation) alongside the code: module/dataload.py, module/utils2/
selection.py, optimization.py, explainability.py, counterfactuals.py, the drivers
selreport.py / optreport.py / expreport.py / cfreports.py, and the configs actually used
in module/experiments/*.yml (bin_sel_final_202608.yml, bin_opt_final_202608.yml,
bin_exp_final_202608.yml, bin_cf_final_202608.yml, bin_eda_final_202608.yml). Also read
the *_refactor.md notes next to each stage — they document non-obvious invariants and
known-unfixed bugs that may bear on what the manuscript can claim.

Report discrepancies in both directions: described-but-not-done, and done-but-not-described.

Check specifically:
1. DATA LEAKAGE. Does any preprocessing (imputation, scaling, encoding, feature selection,
   threshold selection) touch data outside the training fold? Confirm the nested structure
   is as described: outer 4x10 repeated stratified CV, inner 3-fold Optuna TPE search with
   100 trials, refit on the full outer-training set, threshold chosen from out-of-fold
   predictions, single scoring on the held-out test set.
2. THE LABEL IS NOT IN THE PREDICTORS. The outcome is defined by NCS. Confirm in code that
   no NCS column reaches any model or the counterfactual features_to_vary (the repo
   asserts this in cfreports.py — verify the assertion covers every path), and confirm the
   manuscript states it.
3. THRESHOLD PROVENANCE. Each fold model's threshold must come from that fold's own stored
   model. Verify in code and confirm the manuscript's description matches.
4. WHICH MODELS ARE REPORTED. The explainability and counterfactual stages reuse the FIRST
   REPEAT's four fold models. Confirm the manuscript says so wherever those results are
   presented, and that no reader could mistake "4 models" for "4 experiments" or for the
   40-run aggregate.
5. CLASS IMBALANCE HANDLING. Confirm the manuscript's account (CatBoost's internal
   handling; no SMOTE) matches the code, including whether scale_pos_weight was tuned and
   what that means for the reported thresholds.
6. COUNTERFACTUAL SETUP. Verify against code: the DiCE backend and method (genetic), the
   background data used per fold, the six features permitted to vary and their permitted
   ranges, the number of counterfactuals requested (20), diversity weight, the candidate
   selection rule (misclassified OR within 0.08 of the fold threshold), and the
   sufficiency/necessity checks. Confirm each is described in the manuscript.
7. RANDOMNESS AND REPRODUCIBILITY. Seeds, library versions, and whether re-running the
   configs reproduces the reported numbers. Note what the manuscript would need to state
   for a reader to reproduce.
8. THE MISSING-DATA STORY. Three records were dropped (codes 36, 46, 173). Verify in code
   which strategy the final configs used (drop vs impute_mean) and that the manuscript
   describes the one actually used.

Write to manuscript/references/prompts/findings/p14-methods-vs-code.md, with a
two-column discrepancy table (manuscript says / code does) at the top. Anything in
category 1 or 2 is a BLOCKER.
```

---

## P15 — Traceability of every reported number to a pipeline artifact

```text
Build a complete traceability matrix for manuscript/main.tex. Do not edit — report only.

Extract EVERY number that appears in the manuscript's main text, tables, and captions
(excluding citation years, line/section numbers, and hyperparameter search settings), and
for each one identify the artifact under manuscript/references/ or
module/experiments/binary/ that produced it, with the exact file and the row/column or
cell.

Output a matrix: number | where it appears in main.tex (line) | claimed meaning | source
file | source location | matches? (yes/no/not-found).

Then report three lists:
1. ORPHAN NUMBERS — numbers with no locatable source artifact. These are the dangerous
   ones: they may be survivors from an earlier pipeline run. Treat each as MAJOR until
   sourced.
2. STALE NUMBERS — numbers that exist in a source artifact but with a different value.
   BLOCKER, each one.
3. UNUSED ARTIFACTS — outputs present under manuscript/references/ that the manuscript
   never uses, especially where they would strengthen it (the calibration and decision-
   curve outputs, the ROC/AUROC plots, the per-split Brier plots, the remaining
   counterfactual plots).

Write to manuscript/references/prompts/findings/p15-traceability.md.
```

---

## P16 — Patient privacy and de-identification

```text
Review manuscript/main.tex for patient-privacy risk. Do not edit — report only.

This is a single-center study of 187 patients at a named hospital in a named city, and the
manuscript presents individual patients by study code (Patients 20, 40, 76, 123, 172 and
others) together with clinical attributes: HbA1c values, comorbidity status (hypertension,
dyslipidemia, CKD, PAOD), insulin use, model probability, and outcome.

Assess:
1. Whether any individual described could be re-identified by someone with access to the
   clinic's records, given the combination of attributes disclosed — and whether that
   matters for the journal's standards even if re-identification requires insider access.
2. Whether the patient codes are the original spreadsheet CODEs (traceable back to the
   source dataset) or study-internal pseudonyms, and whether the manuscript says which.
   Check module/dataload.py and index_to_patient_code() to establish the truth.
3. Whether the supplementary/appendix material, the figures, or any table discloses
   row-level data for the full cohort.
4. Whether the ethics and consent statements (currently absent — see P9) cover
   publication of individual-level case descriptions.
5. Whether the repository itself, if code/data are to be released, would expose the raw
   dataset (dataset/EAMC_DPN_Dataset.xlsx) or any derived file containing identifiers, and
   what a data-availability statement can honestly promise.

Recommend concrete mitigations (e.g. re-lettering case patients as Case A/B/C, banding
continuous values, moving row-level tables to controlled access) and state the cost of
each to the manuscript's argument.

Write to manuscript/references/prompts/findings/p16-privacy.md.
```

---

## P17 — Terminology, acronym first-use, and abbreviation consistency

```text
Audit terminology, abbreviations, and naming consistency across all of
manuscript/main.tex including captions and the appendix. Do not edit — report only.

Produce:
1. AN ACRONYM AND ABBREVIATION REGISTER — this is the manuscript's first-use audit, so
   do it exhaustively rather than by sampling. Build one table covering EVERY acronym,
   initialism, and abbreviation in the file, with a column for each of the checks below,
   and list at the end every acronym failing any of them.
   a. DEFINED AT FIRST USE. The first appearance in the main text must spell the term out
      in full with the acronym in parentheses immediately after — "diabetic peripheral
      neuropathy (DPN)" — and every later appearance must use the acronym alone. Report
      the line of first use and whether the definition is there.
   b. USED BEFORE DEFINED. Flag any acronym whose first appearance is bare and whose
      definition comes later, which is the most common form of this error after text has
      been reordered. Check the title, abstract, keywords, section headings, figure and
      table captions, and the appendix in reading order, not just the body paragraphs.
   c. INDEPENDENT SCOPES. The abstract, each figure/table caption, and (for most
      publishers) the appendix are read independently of the body. An acronym must
      therefore be defined at its first use IN THE ABSTRACT, again at its first use in the
      body, and again in the first caption that uses it. Report each scope separately —
      a term correctly defined in the body but bare in a caption is still a finding.
   d. REDEFINED. Flag any acronym spelled out again after it has already been defined in
      the same scope, and any acronym defined twice in the body.
   e. USED ONCE. Any acronym that appears only once or twice in the whole manuscript
      should be spelled out and the acronym dropped entirely — introducing an acronym the
      reader never needs again costs them memory for nothing. List every such case.
   f. EXPANSION WORDING IS STABLE. The full form must be worded identically at each
      defining occurrence (e.g. not "area under the precision-recall curve" in one place
      and "area under the precision recall curve" or "precision-recall AUC" in another).
   g. TITLE AND KEYWORDS. Acronyms in the title or keywords should be avoided unless
      universally recognized in the field; flag any that appear there, and confirm each is
      still defined at first use in the body regardless.
   h. GRAMMAR AROUND ACRONYMS. Article agreement by pronunciation ("an MNSI score", "a
      SHAP value"), plurals without apostrophes (CFs, not CF's), and consistent
      capitalization of the expanded form (lowercase unless a proper noun — "machine
      learning (ML)", not "Machine Learning (ML)").
   i. NON-STANDARD OR AMBIGUOUS. Flag acronyms that collide with a different common
      meaning in a clinical readership (PR = precision-recall vs pulse rate/per rectum;
      CI = confidence interval vs cardiac index/confidence; DM = diabetes mellitus vs
      diabetic macular; NS = the dataset's variable code vs "not significant"; CV =
      cross-validation vs cardiovascular/coefficient of variation). Each of these needs
      either an unambiguous definition at first use or a different abbreviation.
   Cover at minimum, and add every one you find beyond this list: DPN, NCS, ML, XAI, SHAP,
   DiCE, AUPRC, AUROC/ROC-AUC, PR, CV, CI, MNSI, ESC, PAOD, CKD, HbA1c, TPE, FDR, DCA,
   EPV, DM, IQR, SD/STD, TP/FP/TN/FN, CF, EAMC, IRB/ERB, and the dataset variable codes.
2. A synonym report: places where one concept is named more than one way. Check at least:
   AUPRC vs "area under the precision-recall curve" vs "precision-recall area"; ROC-AUC vs
   AUROC vs "area under the receiver operating characteristic curve"; "fold model" vs
   "split model" vs "model 0-3"; "candidate" vs "instance of interest" vs "selected
   patient"; "Unconfirmed" vs "non-confirmed" vs "negative"; "actionable" vs "modifiable"
   vs "suitable"; "counterfactual" vs "counterfactual explanation" vs "CF"; "feature" vs
   "predictor" vs "variable"; "Sudoscan" capitalization; "glycated hemoglobin" vs "HbA1c"
   vs "A1c". For each, recommend one term and list every line to change.
3. Variable-code consistency between the codes table, the figures, and the prose
   (DEC_AR, HAND_MEAN_ESC, FEET_MEAN_ESC, NS, GBS, and the rest), including underscore
   escaping in LaTeX and consistent typographic treatment.
4. Consistency of the outcome label's name and definition everywhere it appears.
5. "Model" disambiguation: the word is used for the algorithm class, the tuned pipeline,
   and the four per-fold fitted models. Flag every sentence where which sense is meant is
   not immediately clear.

Write to manuscript/references/prompts/findings/p17-terminology.md, leading with the
acronym register table.
```

---

## P18 — Cross-reference, numbering, and float-placement integrity

```text
Check the reference machinery of manuscript/main.tex. Do not edit — report only.

1. Build the label/ref graph. Report: labels never referenced (~10 expected, since there
   are ~54 labels and ~44 distinct \ref targets); \ref to non-existent labels; duplicate
   labels; labels whose prefix does not match their type (tab: on a figure, fig: on a
   section, sec: on a table).
2. Check every section cross-reference in prose ("Section~\ref{sec:hpo}",
   "Section~\ref{sec:cf-plausibility}", "Section~\ref{sec:instance-level-cf}",
   "Section~\ref{sec:limitations}", "Appendix~\ref{secA1}") resolves to the section the
   sentence means, not merely to a valid label.
3. Verify float numbering order matches first-mention order after compilation; read
   main.aux or the compiled PDF to get actual numbers rather than guessing.
4. Check float placement: figures/tables that land pages away from their discussion,
   floats using [p] or [!ht] in ways that will break under the journal's own class,
   supplementary floats interleaved with main-text ones.
5. Check citation numbering and ordering under the sn-mathphys-num style: first-appearance
   order, no duplicate entries for one work, no cited-but-absent or present-but-uncited
   bib entries (compare \cite keys against bibfile.bib's 57 entries and against main.bbl).
6. Check the appendix: every supplementary figure/table is cited from the main text, and
   the Supplementary Information statement matches what is actually supplied.
7. Report any \label placed before its \caption (which silently misnumbers references).

Write to manuscript/references/prompts/findings/p18-crossrefs.md, with the orphan-label
and dangling-ref lists first.
```

---

## P19 — LaTeX build and Springer template compliance

```text
Compile manuscript/main.tex and audit both the build and the template compliance. You may
run latexmk/pdflatex/bibtex; do not edit main.tex — report only.

1. Do a clean build from scratch (remove aux artifacts into a scratch copy, do not delete
   the user's files) and report every warning and error, quoted from the log: undefined
   references and citations, multiply-defined labels, missing graphics, package conflicts,
   font substitution, and the full list of Overfull/Underfull boxes with their line
   numbers and overflow amounts (there is a known history of overfull tables in this
   manuscript).
2. Verify the document class options match the intended target journal
   (currently sn-jnl with sn-mathphys-num). If the target is a clinical journal, the
   reference style is very likely wrong — sn-vancouver-num is the usual choice for
   medical titles. Flag it and say what changing it entails.
3. Check the template's own submission constraints, which are stated in the file's header
   comments: the manuscript must be ONE .tex file with no \input/\include, and figures
   must be attached separately rather than embedded. Confirm compliance.
4. Report any package loaded but unused, any package that conflicts with sn-jnl, and any
   manual formatting that fights the class (hard-coded \vspace, \hphantom, manual
   line breaks in captions, \let\cline\cmidrule redefinitions).
5. Check that tables use booktabs correctly, have no vertical rules, and that every table
   fits within \textwidth at the class's font size; list every table that does not.
6. Report the compiled page count, figure count, table count, and the abstract word count.
7. Confirm the PDF has no broken hyperlinks, no missing fonts, and that all images are
   embedded at usable resolution; list any image below ~300 dpi at its printed size.

Write to manuscript/references/prompts/findings/p19-latex-build.md with the full warning
inventory as an appendix to the findings.
```

---

## P20 — Figure and table presentation quality

```text
Review the visual quality and accessibility of every figure and table in
manuscript/main.tex, viewing the actual image files under manuscript/references/. Do not
edit — report only.

For each figure:
1. LEGIBILITY AT PRINT SIZE. Given the \includegraphics width and the journal column
   width, will axis labels, tick labels, legends, and annotations be readable? Flag every
   figure whose smallest text would fall below roughly 6-7 pt when printed.
2. COLOR. Is the encoding colorblind-safe (deuteranopia in particular)? The counterfactual
   case-study figure encodes direction as red/blue — check whether it is distinguishable
   without color and whether a redundant encoding (shape, position, sign, hatch) exists.
   Flag any figure where color alone carries meaning.
3. GRAYSCALE SURVIVAL. Would the figure remain interpretable printed in black and white?
4. SELF-CONTAINED CAPTIONS. Every caption should let a reader understand the figure
   without the body text: what is plotted, on what data, what the units are, what error
   bars/bands represent, what n is, and what every visual encoding means. Flag captions
   that omit any of these. Separately flag captions that have gone the other way and now
   contain argumentation belonging in the Discussion (several here run 150+ words).
5. AXIS INTEGRITY. Truncated axes, inconsistent scales between panels that invite
   comparison, missing units, missing baselines (a PR curve without its prevalence line, a
   discrimination plot without 0.5).
6. REDUNDANCY. Any figure whose content is fully duplicated by a table, or vice versa —
   candidates for supplementary demotion given the high float count.
7. FIGURE FORMAT. Flag raster PNGs used where the pipeline can emit vector PDF, since
   several figures are currently .png while others are .pdf.

For each table: check units in headers, consistent decimal places per column, defined
footnote symbols, no vertical rules, no cells whose meaning depends on the body text,
sensible row ordering, and that the caption states n and what the +/- values are.

Write to manuscript/references/prompts/findings/p20-figure-quality.md, one subsection per
float.
```

---

## P21 — Limitations completeness and negative-result honesty

```text
Audit the Limitations subsection of manuscript/main.tex (and any limitation stated
elsewhere) for completeness and honesty. Do not edit — report only.

First, list every limitation the manuscript states, with its line.

Then, independently derive the limitations this study's design and results imply, and
report which are MISSING or UNDERSTATED. Consider at least:
- Single center, single country, 187 patients, 57 minority events, 2.6 events per
  predictor; no external validation; no temporal validation.
- Specificity of 0.527 and what that implies for referral burden in deployment.
- Class prevalence of 69.5% is a referral-clinic artifact and will not hold in a screening
  population, which changes every predictive value the model would have in use.
- No calibration reported in the main text (though computed), so the probabilities are
  used as scores without evidence they are calibrated.
- Hyperparameters tuned by 100-trial search on small folds: optimism.
- The counterfactual yield: 8 of 61 candidates, one fold producing none, and what that
  says about the reliability of the counterfactual component as presented.
- Counterfactuals are statements about the model, not about physiology; no intervention
  was performed and no causal identification strategy was used.
- Binary comorbidity encoding makes counterfactual magnitudes clinically uninterpretable
  ("resolve CKD") — the manuscript notes this as future work; check it is also stated as a
  present limitation.
- Actionable features carry the least univariable signal (0.514-0.585), which is why the
  required displacements are large — check this is stated as a limitation and not only as
  an explanation.
- The model's gain over the single best bedside feature (0.799 vs 0.748) is modest.
- Label definition depends on NCS availability and Toronto criteria interpretation;
  outcome assessors' blinding to predictors is unclear.
- Missing data handling by row deletion (3 records) and any resulting selection.
- Generalizability of Sudoscan findings across devices and operators.
- No fairness/subgroup analysis (sex, age, diabetes duration).

Then check the reverse failure: places where the manuscript states a limitation but then
proceeds as if it did not apply — especially in the Conclusion and abstract.

Finally, assess whether the negative results are reported with the prominence they deserve
rather than buried: Model 2's zero counterfactuals, the 13% counterfactual success rate,
and the unstable specificity.

Write to manuscript/references/prompts/findings/p21-limitations.md, with the missing
limitations as drafted sentences ready to insert.
```

---

## P22 — Adversarial reviewer simulation

**Why:** the cheapest way to find what Reviewer 2 will say is to be Reviewer 2 first.

```text
You are Reviewer 2 for a clinical machine-learning journal, assigned manuscript/main.tex.
You are fair but skeptical, you have reviewed many small-cohort prediction-model papers,
and you are unimpressed by methodology described rather than demonstrated. Read the whole
manuscript. Do not edit it.

Write a full referee report in the standard format:

1. SUMMARY OF THE MANUSCRIPT (your own words, 150 words) — this doubles as a test of
   whether the manuscript communicates its contribution.
2. ASSESSMENT: recommendation (reject / major revision / minor revision / accept) with
   the reasoning stated in one paragraph.
3. MAJOR COMMENTS — numbered, each with the specific line or section, what the problem is,
   and what the authors must do to resolve it. Aim for the 6-10 most substantive. Push
   hardest on: whether the counterfactual component is supported by 8 patients; whether
   the model beats the simplest bedside alternative by enough to matter; whether
   specificity of 0.53 is compatible with the claimed screening use case; whether
   CatBoost's selection over higher-scoring models is justified by evidence or by
   preference; whether the absence of external validation and calibration is fatal for the
   clinical claims; and whether the Discussion-as-Results structure hides the actual
   findings.
4. MINOR COMMENTS — numbered.
5. QUESTIONS THE AUTHORS MUST ANSWER — the things you would not accept a revision without.

Then switch roles and write a second, separate report as a CLINICIAN reviewer (an
endocrinologist or neurologist, not a methodologist) who cares only about whether this
would change practice: is the workflow realistic in a Philippine tertiary clinic, are the
actionable features actually actionable, would you refer a patient based on this score,
and is anything in the counterfactual recommendations clinically unsafe or absurd.

Finish with a single prioritized list: the five changes that would most improve this
manuscript's chance of acceptance.

Write to manuscript/references/prompts/findings/p22-reviewer-simulation.md.
```

---

## P23 — Novelty and literature positioning

```text
Assess how manuscript/main.tex positions itself against the literature it cites. Do not
edit — report only.

Source material: manuscript/bibfile.bib (57 entries) and the 14 full-text PDFs in
manuscript/rrl/articles/, which include two 2025 systematic reviews of ML for DPN (Ma;
Sun) and several directly comparable DPN prediction papers (Baskozos 2022, Lian 2023,
Wu Y 2024, Sheikh 2025, Tian 2025) plus counterfactual/XAI-in-medicine work (Qin 2026,
Rotbei 2026, Wu H 2024, Amann 2020, Sirocchi 2024).

Report:
1. THE CONTRIBUTION AS STATED vs AS SUPPORTED. What does the manuscript claim is new
   (counterfactuals for DPN; Sudoscan inclusion; NCS exclusion by design; Philippine
   cohort)? For each, check the cited literature — particularly the two systematic reviews
   — for prior work that does the same thing, and report whether the novelty claim
   survives.
2. MISSING COMPARISON. The manuscript reports "AUCs ranging from 0.64 to 0.93" for prior
   work but does not appear to place its own 0.799/0.886 in a structured comparison. Draft
   the comparison table that a reviewer will ask for: study, cohort size, setting,
   predictors used, whether NCS/electrodiagnostic input was included, metric, value. Note
   that comparing across different label definitions and prevalences is itself a caveat
   the manuscript must state.
3. RELATED WORK GAPS. Identify topics the introduction should engage but does not — e.g.
   prior counterfactual-explanation work in clinical prediction and its known critiques
   (plausibility, actionability, robustness of counterfactuals), and the
   explanation-vs-decision-support distinction the conclusion raises but the introduction
   does not set up.
4. UNCITED-BUT-OBVIOUS. From the PDFs present, list any that are in the repo but never
   cited in main.tex, and say where each would strengthen the argument.
5. RECENCY. Note the distribution of publication years and whether the DPN-ML literature
   from 2024-2026 is adequately represented given how fast it is moving.

Write to manuscript/references/prompts/findings/p23-positioning.md, with the comparison
table as a ready-to-insert LaTeX table.
```

---

## P24 — Title, abstract, and keywords fit

```text
Evaluate the front matter of manuscript/main.tex for discoverability and fit. Do not edit
— report only, with concrete alternatives.

1. TITLE: "An Application of Machine Learning and Counterfactuals for the Pre-screening
   and Explanation of Diabetic Peripheral Neuropathy", short title "ML and Counterfactuals
   for DPN Pre-screening". Assess: does it state the finding or only the activity? ("An
   Application of..." is an activity title; the strongest clinical titles state the
   design, the population, and the claim.) Does it name the study design and setting?
   Is it within typical length limits? Propose three alternatives — one descriptive, one
   finding-forward, one design-forward — each with a matching short title.
2. ABSTRACT: word count against a 250-word budget; structural fit with the target journal's
   required headings; whether any abbreviation is used undefined; whether any claim exceeds
   the body (cross-check with P5 and P10); whether the Conclusions sentence is a conclusion
   or a restatement of results.
3. KEYWORDS: the current ten include both broad and highly specific terms. Assess overlap
   with the title (keywords that duplicate title words are wasted, since title words are
   already indexed), MeSH alignment for a biomedical index, and whether the set would
   surface this paper for the searches its intended readers actually run. Propose a revised
   set with reasoning per term.
4. Check the title/abstract/keywords triple tells one consistent story about what the paper
   claims.

Write to manuscript/references/prompts/findings/p24-front-matter.md.
```

---

## P25 — Voice uniformity and machine-writing tells

**Why:** the manuscript was assembled over many sessions and several sections were
rewritten wholesale; a reviewer noticing a register shift mid-paper reads it as carelessness,
and many journals now ask directly about AI assistance.

```text
Read manuscript/main.tex for voice consistency and for stylistic tells that mark a text as
machine-generated or machine-heavy. Do not edit — report only.

1. REGISTER SHIFTS. Identify passages whose voice differs noticeably from the manuscript's
   baseline: sentence length distribution, em-dash density, use of triadic constructions
   ("X, Y, and Z" three times in a paragraph), the "not merely A but B" and "it is not X;
   it is Y" frames, paragraph-final summary sentences that restate the paragraph, and
   captions that argue rather than describe. Quote the passage, quote a baseline passage
   for contrast, and name the difference.
2. TELLS TO COUNT AND LOCATE: em-dashes (count per section and flag the outliers);
   "delve", "leverage", "underscore", "pivotal", "crucial", "robust" as a filler
   adjective, "it is worth noting", "furthermore/moreover" chains, "landscape", "realm",
   "tapestry", "navigate", "showcase", "seamless", "holistic", "comprehensive" as filler,
   and the "This is not X. It is Y." rhythm. Report counts and the worst instances.
3. HEDGE DENSITY. Count hedges per section; flag paragraphs where the hedging has become
   the content.
4. FLAT ASSERTION CHAINS. Passages of three or more consecutive sentences with the same
   structure (subject-verb-object of similar length) — these read as generated and are
   easy to fix by varying one.
5. Recommend a target voice for the whole manuscript in one sentence, and list the five
   passages most in need of being brought into it.
6. Separately: report whether the manuscript contains an AI-use disclosure statement, and
   draft one appropriate to how this manuscript was prepared if it does not.

Write to manuscript/references/prompts/findings/p25-voice.md.
```

---

## P26 — Adherence to the journal's submission guidelines

**Why:** P9 asks whether an editor would return the paper; this one checks it line by line
against the journal's actual Instructions for Authors. **It needs the target journal named.**
Nothing in the repo names one, and the current class option (`sn-mathphys-num`, a
mathematical/physical-sciences reference style) is very likely wrong for a clinical title —
so decide the target before running this, or run it in shortlist mode as described below.

```text
You are checking manuscript/main.tex for compliance with a specific journal's submission
guidelines. Do not edit the manuscript — produce a compliance matrix and a submission-day
checklist.

TARGET JOURNAL: <<<fill in the journal name and the URL of its Instructions for
Authors / Submission Guidelines page>>>

If no journal is named above, do not invent one. Instead run in SHORTLIST MODE: propose
four candidate journals that fit this paper (a 187-patient single-center diabetic
peripheral neuropathy prediction-model study with counterfactual explanations, from the
Philippines), each with scope fit, typical article type, whether it is indexed in
MEDLINE/PubMed, open-access status and APC, and the two or three guideline requirements
that would most affect this manuscript. Then run the full check below against Springer
Nature's general research-article requirements as a placeholder and mark every
journal-specific item as PENDING JOURNAL SELECTION.

Fetch the journal's live Instructions for Authors rather than working from memory —
guidelines change and a stale requirement is worse than no requirement. Quote the
guideline text for each item you check. If you cannot fetch it, say so and mark the whole
check UNVERIFIED rather than guessing.

Produce a compliance matrix with one row per requirement: requirement (quoted) | what the
manuscript currently does (with line number) | COMPLIES / VIOLATES / NOT APPLICABLE /
AUTHOR DECISION NEEDED | what to change.

Cover at least:

A. SCOPE AND ARTICLE TYPE. Does the journal accept this article type (original research /
   original article)? Does the study fall within its stated scope? Is there an article
   type that fits better (e.g. "brief report" given the cohort size)? Flag scope mismatch
   as a BLOCKER — it is the most common desk rejection reason after incompleteness.
B. LENGTH LIMITS. Whole-text word count, abstract word count, number of figures, number of
   tables, number of references, number of supplementary items. This manuscript currently
   has roughly 16 main-text figures and 13 tables, which exceeds the limit at most clinical
   journals — report the exact overage and propose which specific floats move to
   supplementary, ranked by how little the main argument loses.
C. STRUCTURE. Required section headings and their order (most clinical journals require
   IMRaD with a distinct Results section; this manuscript reports results inside the
   Discussion). Required abstract structure and headings — check the journal's exact
   heading words (Background/Methods/Results/Conclusions vs Objective/Design/Setting/...)
   against what the manuscript uses. Whether abstracts may contain citations,
   abbreviations, or statistics.
D. TITLE PAGE AND FRONT MATTER. Title length limit; running/short title limit; whether the
   design must appear in the title; author name format; maximum author count and whether a
   contribution statement is required for that number (there are 11 authors); ORCID
   requirement (mandatory for corresponding author at many publishers, and every \email
   field for co-authors currently contains placeholder text); affiliation format;
   corresponding-author designation and required contact details; keyword count and whether
   MeSH terms are required.
E. MANDATORY STATEMENTS. List every statement the journal requires, with its required
   heading wording: funding, competing interests, ethics approval and consent to
   participate (with committee name and reference number), consent for publication,
   data availability (and whether a specific repository or access statement is mandatory),
   materials availability, code availability, author contributions (CRediT taxonomy?),
   acknowledgements, and any AI-use statement (see P27). For each, quote the journal's
   required wording and report whether the manuscript has it. All of these are currently
   empty headings in the Declarations section.
F. REPORTING GUIDELINE AND CHECKLIST. Whether the journal mandates TRIPOD+AI, TRIPOD,
   CLAIM, or STARD for this study type; whether a completed checklist must be uploaded as a
   separate file; whether the checklist must cite page/line numbers; whether the guideline
   must be named in the Methods.
G. ETHICS AND REGISTRATION. Declaration-of-Helsinki statement wording; whether the ethics
   committee's approval number must appear in Methods as well as Declarations; whether
   study registration is required for this design and, if so, where the number goes.
H. FIGURES AND TABLES. Accepted file formats and whether figures must be uploaded as
   separate files rather than embedded (the Springer template header in main.tex says they
   must); minimum resolution for line art, halftone, and combination art; maximum
   dimensions; whether color is free online and charged in print; caption placement and
   length limits; whether captions may contain citations; table formatting rules (no
   vertical rules, footnote symbol order); whether figures must be cited in order.
I. REFERENCES. Required style and whether the manuscript's document-class option produces
   it (currently sn-mathphys-num; sn-vancouver-num is the usual clinical choice); maximum
   reference count; DOI requirement; whether preprints, websites, or personal
   communications are permitted and how they are formatted; "et al." threshold; whether
   references must be in first-citation order.
J. SUBMISSION PACKAGE. Enumerate every file the journal requires at submission and what
   this repo currently has or lacks: cover letter (and what it must state), title page as a
   separate blinded/unblinded file, main manuscript file, figure files, table files,
   supplementary material, reporting checklist, ethics documentation, consent forms,
   author agreement or copyright form, highlights, graphical abstract, suggested and
   opposed reviewers, and any conflict-of-interest form per author.
K. PEER-REVIEW MODEL. Single-blind, double-blind, or open. If double-blind, list every
   identifying element that must be stripped: author names and affiliations, the named
   hospital and city, funding acknowledgements, self-citations phrased in the first person,
   dataset and repository names, and file metadata.
L. PRODUCTION FORMALITIES. Line numbering and double spacing for review; page numbering;
   font and margin requirements; whether LaTeX is accepted at all and which template
   version; whether a PDF or source files are submitted; permissions and credit lines for
   any reproduced or adapted material.
M. POLICY ITEMS. Preprint policy and whether posting affects eligibility; data-sharing
   policy; plagiarism/similarity screening threshold; duplicate and overlapping publication;
   authorship criteria (ICMJE) and whether all 11 authors meet them; APC and waiver
   eligibility for the corresponding author's country.

Finish with two deliverables: (1) a numbered pre-submission checklist ordered so it can be
worked through on submission day, with each item marked as "Claude can draft this" or
"author decision required"; and (2) the exact list of changes needed in main.tex, grouped
by whether they are LaTeX/class changes, content additions, or structural rework.

Write to manuscript/references/prompts/findings/p26-submission-guidelines.md.
```

---

## P27 — Adherence to generative-AI / LLM use policy

**Why:** this manuscript and its pipeline were developed with substantial AI assistance —
the repo carries a `CLAUDE.md`, five `*_refactor.md` session records, a prompts directory,
and 29 of 458 commits attributed to Claude. Publisher policies require that use to be
disclosed accurately, and if the code is released the git history will show it. An
inaccurate or missing disclosure is a research-integrity finding, not a formatting one.

```text
You are checking manuscript/main.tex against publisher policy on the use of large language
models and generative AI. Do not edit the manuscript — report findings and draft the
disclosure text.

TARGET JOURNAL / PUBLISHER: <<<fill in; if blank, check against Springer Nature's AI
policy plus the ICMJE recommendations and COPE's position statement, and say that is what
you did>>>

Fetch the publisher's current policy rather than relying on memory, and quote it. Policies
in this area changed materially in 2023-2025 and are still moving.

STEP 1 — ESTABLISH THE FACTS BEFORE JUDGING THEM. Reconstruct, from repository evidence,
what AI actually did in this project. Do not speculate beyond the evidence, and label
anything you cannot establish as UNKNOWN — ASK AUTHOR. Sources: CLAUDE.md; the five
module/*_refactor.md session records; manuscript/references/prompts/ (including
mss_edits.md and this file); git log (458 commits, ~29 mentioning Claude — list them with
dates and what they touched); any Co-Authored-By trailers; and the working-tree history of
main.tex. Classify each use into one of three categories, because publishers treat them
differently:
  (a) AI AS A RESEARCH TOOL — writing or refactoring the analysis pipeline, generating
      figures, computing results. This is normally described in Methods like any other
      software, with the tool and version named, and it interacts with the
      code-availability statement.
  (b) AI AS A WRITING AID — drafting, rewriting, editing, or translating manuscript prose.
      This normally requires a disclosure statement but is NOT described in Methods.
  (c) AI IN THE STUDY DESIGN OR INTERPRETATION — choosing analyses, interpreting results,
      forming conclusions. If any of this occurred, it raises questions about author
      responsibility that the disclosure must address honestly.

STEP 2 — CHECK COMPLIANCE. For each item report COMPLIES / VIOLATES / MISSING / AUTHOR
DECISION NEEDED, quoting the policy:
1. NO AI AUTHORSHIP. Confirm no LLM or tool appears in the author list, in an affiliation,
   in the acknowledgements as an author-like contributor, or in the corresponding-author
   details. Confirm no author is credited to a tool.
2. DISCLOSURE PRESENT. Is there any AI-use statement in the manuscript at all? (There is
   currently none — the Declarations section is an empty template list and
   Acknowledgements is blank.) Determine where the publisher requires it: a dedicated
   declaration heading, the Acknowledgements, the Methods, or the cover letter — some
   require it in more than one place.
3. DISCLOSURE IS ACCURATE AND COMPLETE. Compare the drafted disclosure against your Step 1
   inventory. Under-disclosure is the risk that matters: a statement covering only language
   polishing when AI also wrote analysis code and drafted sections is inaccurate. Flag any
   gap explicitly.
4. TOOL IDENTIFICATION. Policies generally require the tool name, version/model, the
   vendor, the dates of use, and what it was used for. Establish what can be stated
   accurately from the repo (Claude Code; model where recorded in commit trailers; date
   range from git history) and mark the rest as author-supplied.
5. HUMAN RESPONSIBILITY. Policies require authors to take full responsibility for the
   content, including its accuracy and the absence of plagiarism and fabrication. Check
   that the disclosure asserts human review, and cross-reference P4 (citation fidelity):
   fabricated or mis-attributed references are the characteristic failure mode of
   AI-assisted writing, so P4 must have been run and passed before this statement is
   truthful. State plainly whether that condition is currently met.
6. AI-GENERATED IMAGES. Most publishers prohibit AI-generated or AI-modified images
   without prior permission, with a carve-out for images that are the output of a research
   method. Determine, for each figure, whether it is (i) a plot produced by the analysis
   pipeline, (ii) a hand-authored diagram (references/illustrations/detailed_pipeline.pdf,
   clinician_workflow_flow_compact.pdf — establish how these were produced), or (iii)
   generative-model output. Report any figure in category (iii) as a BLOCKER, and flag
   category (ii) if the drawing code was AI-written, noting whether the policy's carve-out
   covers it.
7. AI IS NOT A CITABLE SOURCE. Confirm no bibfile.bib entry and no in-text citation
   attributes a claim to a chatbot, an LLM output, or a personal communication with a
   model.
8. CONFIDENTIALITY. If any author used an LLM on confidential material — unpublished
   patient-level data or another manuscript under review — check whether that is compatible
   with the policy and with the ethics approval. This study's raw dataset is patient data
   from a named hospital: assess whether any disclosed workflow implies patient-level data
   was sent to a third-party model, and report it as a BLOCKER if it did without a stated
   data-protection basis. Establish this from the repo (what the tooling had access to),
   and if you cannot establish it, mark it ASK AUTHOR — do not assume either way.
9. CONSISTENCY WITH WHAT WILL BE PUBLIC. If code is released to satisfy a code-availability
   statement, the git history, CLAUDE.md, the refactor notes, and the prompts directory
   become visible. Check that the disclosure would not be contradicted by the repository a
   reader can open, and list every repo artifact that evidences AI involvement.

STEP 3 — DRAFT THE DISCLOSURE. Produce three versions, each accurate to the Step 1
inventory and none of them overstating human authorship:
  (i) a minimal statement for a publisher that wants one sentence in the Acknowledgements;
  (ii) a full declaration under a "Declaration of generative AI in the writing process" or
       equivalent heading, matching the target publisher's requested wording;
  (iii) the Methods sentence(s) needed for any category-(a) research-tool use, phrased as
       software attribution with tool, version, and role, alongside the existing software
       stack.
Mark clearly which factual slots the authors must fill or confirm rather than you.

Write to manuscript/references/prompts/findings/p27-ai-policy.md, leading with the Step 1
inventory table, then the compliance matrix, then the three drafted statements.
```

---

# Part C — Running the pack

**Suggested order.** Content correctness before polish; polish before compliance; the
editor's eye last.

| Wave | Prompts | Rationale |
| --- | --- | --- |
| 1. Ground truth | P14, P2, P15, P11 | Establish that the numbers and figures are real and current. Everything downstream is worthless if they are not. |
| 2. Internal consistency | P3, P10, P18 | Make the manuscript agree with itself. |
| 3. Evidence and sources | P4, P13, P12, P21 | Make the manuscript agree with its sources and its own limits. |
| 4. Argument | P5, P23, P22 | Size the claims; position them; then attack them. |
| 5. Language | P7, P6, P8, P17, P1, P25 | Polish only after the content stops moving. |
| 6. Presentation | P20, P19, P24 | Floats, build, front matter. |
| 7. Compliance | P26, P27 | Journal guidelines and AI-use policy. Run P26 early too, if the target journal is already chosen — its length and structure limits change what the other checks are aiming at. |
| 8. Gate | P16, P9 | Privacy and the desk-rejection sweep, re-run after all edits are applied. |

**Notes on execution:**

- Run one prompt per fresh session. P2, P14, and P15 are the expensive ones (they read the
  pipeline and open images); give them room.
- P9 must be re-run after edits are applied — it is a gate, not a one-time check.
- Apply fixes in batches by check, and re-run P19 (build) after each batch so a LaTeX
  breakage is caught next to the edit that caused it.
- Keep every findings file. When a finding is dismissed rather than fixed, record why in
  the same file — that record is what you will want when a reviewer raises the same point.
- P26 and P27 both need an input you must supply: the target journal. P26 is close to
  useless without it (it degrades to shortlist mode); P27 falls back to Springer Nature
  plus ICMJE and COPE, which is a reasonable floor.
- P27 is only truthful once P4 has passed — you cannot attest that citations were
  human-verified before verifying them. Run P4 first.
- The acronym first-use audit lives inside P17, item 1; there is no separate prompt for
  it. Run P17 late, after the text has stopped being reordered, since reordering is what
  creates used-before-defined errors in the first place.
- If any check disagrees with an earlier one, the pipeline artifact wins over the
  manuscript, and the manuscript wins over a previous findings file.
