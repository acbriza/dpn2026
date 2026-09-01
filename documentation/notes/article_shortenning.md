# Target article length for BMC Medical Informatics and Decision Making

**Question asked:** what should the target number of pages be for publication in BMC,
based on the usual article length in that journal?

**Answer: 16–20 published pages**, reached by cutting to roughly 9,000–11,000 body words,
6–8 main-text figures, and 4–6 main-text tables. The manuscript as measured on 2026-09-01
projects to about **32 published pages**, longer than any of the 18 recent articles
sampled from the journal.

---

## Why "pages" needs restating before it can be answered

BMC does not paginate. Articles receive an article number, not a page range — in the
Europe PMC records for this journal the `pageInfo` field holds values like `227` and `220`,
which are article numbers. So there is no page budget allocated by the journal, and the
only meaningful quantity is the length of the article's own published PDF, each of which
is numbered 1..N independently.

Two consequences:

1. The 62-page `manuscript/main.pdf` is **not** the number to steer by. That is A4
   single-column `sn-jnl` output, a much looser layout than BMC's published one, and it
   includes the LaTeX appendix that would become Additional files.
2. Length is governed by norms and reviewer patience, not by a stated limit. BMC Research
   articles are understood to have no cap on words, figures, tables, or references — see
   the caveat at the bottom, this was not verifiable on the day.

---

## What the journal actually publishes

Measured from 18 Research articles published in *BMC Medical Informatics and Decision
Making* between June 2025 and September 2026, full text pulled from Europe PMC. Body word
counts exclude figures, tables, and captions, so they are directly comparable to the
manuscript measurement below.

| Metric | Median | IQR | Range |
| --- | --- | --- | --- |
| Body words | 5,584 | 4,500–8,530 | 2,671–11,639 |
| Figures | 6 | 3–7 | 2–15 |
| Tables | 3 | 1–5 | 0–15 |
| References | 42 | 30–68 | 28–79 |
| Abstract words | 288 | 264–351 | 165–411 |

### The 18 articles

| PMCID | Body words | Fig | Tab | Refs | PDF pages | Subject |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PMC13321472 | 2,671 | 2 | 3 | 28 | 9 | AKI outcomes with/without CKD |
| PMC13330188 | 4,372 | 7 | 3 | 31 | — | Superlearner in-hospital mortality |
| PMC13147576 | 4,396 | 8 | 3 | 41 | — | Early osteoporosis screening |
| PMC13277143 | 4,484 | 3 | 3 | 79 | — | ED decision support, heart failure |
| PMC13156892 | 4,506 | 7 | 1 | 46 | — | Forecasting hospital bed occupancy |
| PMC13325577 | 4,789 | 3 | 2 | 41 | — | Nomogram, negative outcome |
| PMC13317301 | 4,808 | 6 | 1 | 40 | — | LLMs in clinical decision-making |
| PMC13292520 | 5,166 | 4 | 1 | 30 | 11 | COVID risk perception/knowledge |
| PMC13312725 | 5,565 | 5 | 7 | 45 | — | ML ventilator therapeutic pressure |
| PMC13274082 | 5,602 | 5 | 15 | 29 | 17 | Data balancing, ED mortality |
| PMC13195871 | 7,182 | 7 | 3 | 30 | 15 | TrainTracks federated learning |
| PMC13227849 | 7,413 | 8 | 10 | 70 | — | SwinUNETR-v2 multi-sequence MRI |
| PMC13330300 | 8,171 | 8 | 0 | 30 | — | Clinical-ShiftEval |
| PMC13326401 | 8,231 | 3 | 1 | 44 | 14 | Care quality dashboard for GPs |
| PMC13270667 | 9,427 | 2 | 0 | 75 | — | Distributed pharmacovigilance |
| PMC13330239 | 9,880 | 5 | 10 | 42 | 19 | Explainable retinal deep learning |
| PMC13188617 | 11,392 | 7 | 4 | 67 | 19 | Depression diagnosis, multi-scale |
| PMC13285456 | 11,639 | 15 | 4 | 72 | 26 | Optimized XAI + digital twin |

Published PDF length for the eight that were downloadable: 9, 11, 14, 15, 17, 19, 19, 26 —
median ~15.5 pages.

### Words-to-pages conversion

Least squares over those eight articles:

```
pages ≈ 3.73
      + 0.686 per 1,000 body words      (≈ 1 page per 1,459 words of prose)
      + 0.673 per figure
      + 0.343 per table
      + 0.334 per 10 references
```

Residuals: −0.12, +0.31, +0.05, +0.40, −0.79, −0.29, +0.86, −0.42 pages. Every article
predicted to under one page.

Caveat on the fit: n=8 with 5 parameters, so it is descriptive rather than estimated with
any real confidence, and the coefficients are not causal. It is good enough to convert a
word-and-float budget into a page expectation, and nothing more.

---

## Where the manuscript stands

Measured from `manuscript/main.tex` on 2026-09-01 (1,498 lines, compiled `main.pdf` = 62
A4 pages under `sn-jnl`):

| Quantity | Value |
| --- | ---: |
| Body words, excluding float environments | 16,101 |
| Caption words (26 main-text floats) | 3,275 |
| Body words including floats and captions | 21,363 |
| Appendix words | 1,127 |
| Figure environments | 16 |
| Table environments | 13 |
| References in `bibfile.bib` | 57 |
| Abstract words | ~349 |

At 16,101 body words the manuscript is **2.9× the journal's median** and longer than every
article in the sample. Applying the conversion:

| Scenario | Body words | Fig | Tab | Projected pages |
| --- | ---: | ---: | ---: | ---: |
| As-is | 16,101 | 16 | 13 | **32** |
| Floats halved, prose untouched | 16,101 | 8 | 6 | 24 |
| Trim to 11k words | 11,000 | 8 | 6 | 21 |
| Trim to 10k words | 10,000 | 7 | 5 | 19 |
| Trim to 9k words | 9,000 | 6 | 4 | 17 |
| Journal median | 5,584 | 6 | 3 | 14 |

Aim at the **top** of the 16–20 band rather than the journal median. A four-stage pipeline
(selection → optimization → explainability → counterfactuals) with a counterfactual case
analysis genuinely needs more room than a single-model nomogram paper, and 20 pages still
sits inside the observed range.

---

## Where the cut comes from

Per-section prose, excluding float environments:

| Section | Words | Share | Comment |
| --- | ---: | ---: | --- |
| Introduction | 832 | 5% | Right already |
| Data Collection | 708 | 4% | Folds into Methods for BMC's structure |
| Methods | 3,552 | 22% | Right already |
| Results | 3,246 | 20% | Right already |
| **Discussion** | **6,568** | **41%** | 2× the Results it discusses |
| **Conclusion** | **1,195** | 7% | BMC Conclusions typically run 150–250 words |

Discussion and Conclusion together are 7,763 words — 48% of the manuscript. The Discussion
alone is longer than a whole median article in this journal.

Suggested budget for a ~10,000-word target:

| Section | Current | Target | Change |
| --- | ---: | ---: | --- |
| Background | 832 | 800–1,000 | keep |
| Methods (incl. Data Collection) | 4,260 | 3,000–3,500 | −20% |
| Results | 3,246 | 3,000–3,500 | keep, may absorb material from Discussion |
| Discussion | 6,568 | 2,500–3,000 | **−55%** |
| Conclusions | 1,195 | 200–300 | **−80%** |

Cutting only Discussion and Conclusions lands ~10,300 words without touching Methods or
Results. Combined with moving 8 figures and 7 tables to Additional files, that projects to
roughly 19–20 pages.

Captions deserve their own pass: 3,275 words across 26 floats averages 126 words each, and
several exceed 150. They add page area and BMC sets a legend length limit.

---

## Caveat: the guidelines were not verifiable

The journal has migrated onto SpringerLink. Both `bmcmedinformdecismak.biomedcentral.com`
and `www.biomedcentral.com` now return 301s to `link.springer.com` (journal 12911), and
an unauthenticated automated fetch of `link.springer.com/journal/12911/submission-guidelines`
is bounced to an `idp.springer.com` authorize endpoint. So:

- "BMC has no length limit" remains a **strong prior, not a verified fact** as of
  2026-09-01. Confirm it in a browser.
- The distribution and page counts above are measured from published output and do not
  depend on that claim.
- P26 in `manuscript/references/prompts/final_check_actual.md` has been updated with the
  new SpringerLink URLs, the fetch-blocking behavior, and this length distribution.

---

## How to reproduce

Manuscript measurements: word counts come from stripping LaTeX comments, taking the region
between `\maketitle` and `\bmhead{Supplementary`, removing `figure`/`table` environments
wholesale, then stripping macros and counting alphanumeric tokens. Page count from
`pdfinfo manuscript/main.pdf`.

Journal sample: Europe PMC REST API.

```
# article list
curl 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=JOURNAL%3A%22BMC%20medical%20informatics%20and%20decision%20making%22%20AND%20SRC%3AMED%20AND%20FIRST_PDATE%3A%5B2025-06-01%20TO%202026-09-01%5D&format=json&pageSize=25&resultType=core'

# full text per article (word/figure/table/reference counts from the JATS XML,
# with <fig> and <table-wrap> elements removed before counting words)
curl "https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML"

# published PDF, for page counts
curl -L "https://europepmc.org/articles/<PMCID>?pdf=render" -o <PMCID>.pdf
pdfinfo <PMCID>.pdf
```
