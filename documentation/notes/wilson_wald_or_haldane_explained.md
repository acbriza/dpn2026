# Wilson intervals, odds ratios, Wald intervals, and the Haldane-Anscombe correction (Figure S2)

This note explains four statistical ideas that appear together in
**Figure S2** (`module/experiments/binary/eda/s2_categorical_features.png`,
caption in
[`s2_categorical_features_caption.txt`](../../module/experiments/binary/eda/s2_categorical_features_caption.txt)):
Wilson 95% confidence intervals, odds ratios, Wald confidence intervals, and
the Haldane-Anscombe correction. The figure has two panels built from the
same 12 binary features (6 comorbidities + 4 neurological exam signs + SEX +
INSULIN... see the feature table below):

- **Panel a** — prevalence (%) of each feature in the Unconfirmed vs.
  Confirmed groups, with **Wilson 95% CIs** on each bar.
- **Panel b** — the **odds ratio** for the same 12 features on a log scale,
  with **Wald CIs**, and the **Haldane-Anscombe correction** applied when a
  2×2 table has an empty cell.

Both panels are two different views of the exact same underlying 2×2 tables
— panel a shows each group's rate on its own, panel b shows how the two
rates compare to each other as a single ratio. Each concept below is
explained four times, at increasing depth, ending with the actual numbers
from this cohort ($n=187$: 130 Confirmed, 57 Unconfirmed).

The implementation lives in
[`edareport.py`](../../module/edareport.py): `wilson_ci` (line 192),
`odds_ratio_ci` (line 159), and `binary_test` (line 177).

---

## 1. Wilson 95% confidence intervals — "how sure are we about this percentage?"

### Grade-schooler
Say you flip a coin 10 times and get 7 heads. Is the coin actually a
"70%-heads" coin, or did you just get a bit lucky? You can't know for sure
from only 10 flips — but you can draw a *range* of percentages that are all
plausible given what you saw, like "somewhere between 40% and 90%." A
confidence interval is exactly that plausible range. Wilson's way of drawing
that range is just a smarter formula that still works fairly even when you
only flipped the coin a couple of times, or when it almost always (or almost
never) landed heads.

### High-schooler
Every bar in Figure S2a is a percentage — e.g., "5.4% of Confirmed patients
have PAOD" — estimated from a small sample. The 95% confidence interval
around that percentage tells you: if we repeated this study over and over,
95% of the intervals we'd compute this way would contain the *true*
population percentage. The obvious ("normal-approximation") way to build
that interval assumes the percentage behaves like a bell curve, but that
assumption breaks down badly when the percentage is small (like 1.8% for
GBS) or the sample is small (57 Unconfirmed patients) — the naive interval
can even dip below 0% or above 100%. The **Wilson score interval** fixes
this by inverting the actual hypothesis test for a proportion instead of
approximating it, so it stays valid — and stays inside $[0,1]$ — even at the
extremes.

### College student
For $k$ successes out of $n$ trials, the naive ("Wald") interval is
$\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$, which is a linear approximation
that performs poorly when $\hat p$ is near 0 or 1 or $n$ is small — exactly
this study's regime for the rarer comorbidities. The **Wilson interval**
instead finds the interval of population proportions $p_0$ for which the
observed $\hat p$ would *not* be rejected by a one-sample z-test, i.e. it
inverts

$$\left|\frac{\hat p - p_0}{\sqrt{p_0(1-p_0)/n}}\right| \le z$$

Solving this quadratic in $p_0$ gives a closed form, centred not at $\hat p$
but pulled slightly toward 0.5:

$$\tilde p = \frac{\hat p + \dfrac{z^2}{2n}}{1 + \dfrac{z^2}{n}}, \qquad
\text{half-width} = \frac{z}{1+\dfrac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}$$

which is exactly `wilson_ci` at
[`edareport.py:192`](../../module/edareport.py#L192) with $z = 1.96$ for
95%. Because the interval is derived from the test-inversion rather than a
local linear approximation, it is bounded in $[0,1]$ by construction and
remains well-calibrated at small $n$ or extreme $\hat p$ — which is why the
caption specifically says it "remain[s] valid for the rare comorbidities
where normal-approximation intervals do not."

### Research-specific: what this means for our comorbidity bars
`GBS` is the extreme case in this cohort: 1/57 Unconfirmed (1.8%) and 1/130
Confirmed (0.8%) have it — 2 patients total out of 187. A normal-approximation
interval around 1.8% with $n=57$ would be numerically unstable (it can go
negative). The Wilson interval instead gives a wide-but-valid range that
correctly communicates "we saw almost no GBS cases, so we genuinely don't
know its true prevalence to any precision" — rather than falsely implying
precision the data doesn't support. `PAOD` (1/57 = 1.8% Unconfirmed, 7/130 =
5.4% Confirmed) is the second most extreme case for the same reason. This is
the direct, visual explanation for why the GBS and PAOD bars in panel a carry
the widest error bars of the 12 features — it's a direct consequence of how
few patients in this cohort have either condition, not an artefact of the
statistical method.

---

## 2. Odds ratios — "how much more (or less) common is this in Confirmed patients?"

### Grade-schooler
Imagine two groups of kids: kids who like broccoli, and kids who don't.
Now ask: in the "likes vegetables" group, how many like broccoli compared to
how many don't? Do the same for the "doesn't like vegetables" group. If the
first ratio is much bigger than the second, liking vegetables in general
seems to go along with liking broccoli specifically. An odds ratio is just a
single number that captures "how much bigger" — 1 means no difference, above
1 means it's more common in the first group, below 1 means less common.

### High-schooler
For each of the 12 features in Figure S2b, we build a 2×2 table: how many
Confirmed and how many Unconfirmed patients have the feature vs. don't.
The **odds** of having the feature in each group is (number with it) ÷
(number without it). The **odds ratio (OR)** is the Confirmed group's odds
divided by the Unconfirmed group's odds. OR = 1 means the feature is equally
common in both groups (no association with DPN status). OR > 1 means the
feature is more common among Confirmed patients; OR < 1 means less common.
Panel b plots these on a *log scale* specifically because ORs are naturally
lopsided — an OR of 4 and an OR of 0.25 represent equally strong effects in
opposite directions, and only a log scale makes that symmetric visually
(hence the dashed reference line sits at OR = 1, not at 0).

### College student
For a 2×2 table with rows = group (Unconfirmed, Confirmed) and columns =
feature (No, Yes) — $a,b$ = Unconfirmed-No/Yes counts, $c,d$ =
Confirmed-No/Yes counts —

$$OR = \frac{d/c}{\,b/a\,} = \frac{ad}{bc}$$

i.e., the odds of feature-positivity in Confirmed patients relative to the
odds in Unconfirmed patients — matching `or_hat = (d * a) / (c * b)` at
[`edareport.py:171`](../../module/edareport.py#L171). Unlike a simple
risk-ratio of prevalences, the OR is symmetric under swapping which category
is "positive" (this matters because it makes ORs from a retrospective,
already-stratified sample like ours interpretable the same way as they would
be from a prospective study) and it's exactly what falls out of the
coefficients of a logistic regression fit to the same 2×2 table, which is
why it's the standard effect size for a binary predictor against a binary
outcome — the same role AUC plays for continuous predictors elsewhere in
this study's EDA (see
[`auc_bh_q_explained.md`](auc_bh_q_explained.md)).

### Research-specific: the actual ORs in this cohort
From `table1_cohort_characteristics.tex`, sorted by strength:

| Feature | Group | OR (95% CI) | q |
|---|---|---|---|
| DEC_AR | Neuro exam | **10.07** (4.53–22.38) | <0.001 |
| DEC_VS | Neuro exam | **7.90** (3.73–16.73) | <0.001 |
| DEC_PPS | Neuro exam | **4.07** (2.05–8.07) | <0.001 |
| SUBJ | Profile | 4.22 (1.81–9.84) | 0.001 |
| DEC_LTS | Neuro exam | **3.01** (1.55–5.86) | 0.002 |
| PAOD | Comorbidity | 3.19 (0.38–26.52) | 0.482 |
| CKD | Comorbidity | 2.14 (0.88–5.22) | 0.114 |
| INSULIN | Profile | 2.08 (1.07–4.04) | 0.046 |
| DSLPDMIA | Comorbidity | 1.76 (0.94–3.31) | 0.106 |
| SEX (male) | Profile | 1.59 (0.80–3.16) | 0.230 |
| HPN | Comorbidity | 1.13 (0.60–2.12) | 0.712 |
| GBS | Comorbidity | 0.43 (0.03–7.06) | 0.543 |

The four neurological examination signs (`DEC_LTS`, `DEC_PPS`, `DEC_VS`,
`DEC_AR`) have both the largest ORs *and* the tightest CIs — they're common
enough (~50% prevalence) that the estimates are precise, and clinically
they're the most directly downstream of the nerve damage DPN represents. The
comorbidities (`HPN`, `PAOD`, `DSLPDMIA`, `CKD`, `GBS`) all straddle OR = 1
with CIs wide enough to be compatible with either no effect or a fairly
large one — this is the same low-prevalence problem discussed in Section 1,
now visible on the ratio scale instead of the percentage scale.

---

## 3. Wald confidence intervals — "the CI method used for the odds ratio"

### Grade-schooler
Once you have your "how many times bigger" number, you still want to say
how sure you are about it — the same "plausible range" idea as before, but
now for a ratio instead of a percentage. The Wald method is one particular
recipe for drawing that range: start with your best-guess ratio and add or
subtract a margin.

### High-schooler
The Wald interval is the "textbook" confidence interval: take the estimate,
add and subtract (roughly) 2 standard errors. It's simple and works well
when the estimate's uncertainty is reasonably bell-curve shaped. The catch
is that an odds ratio itself is *not* bell-curve shaped — it's bounded below
by 0 and stretches out to infinity, and it's lopsided (as noted above, OR=4
and OR=0.25 are equal-and-opposite effects but 4 and 0.25 aren't symmetric
around 1). The trick used here is to apply the Wald method to $\log(OR)$
instead of $OR$ itself, because $\log(OR)$ *is* well-approximated by a bell
curve (it can be positive or negative, symmetric around 0), and then convert
the endpoints back with $\exp(\cdot)$.

### College student
The sampling distribution of $\log(OR)$ is asymptotically normal with
standard error

$$SE(\log OR) = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}$$

so the Wald interval on the log scale is $\log(OR) \pm z \cdot SE(\log OR)$,
and exponentiating gives

$$CI_{OR} = OR \cdot \exp\!\big(\mp z \cdot SE(\log OR)\big)$$

exactly the last two return values of `odds_ratio_ci` at
[`edareport.py:172-174`](../../module/edareport.py#L172-L174). This is why
panel b's error bars look symmetric on the log-scaled x-axis but asymmetric
if you mentally convert back to a linear OR scale — the symmetry lives in
log-space, where the normal approximation is actually reasonable, rather
than in OR-space, where it wouldn't be. Note this is a *different* interval
construction from Wilson (Section 1): Wilson avoids the normal
approximation entirely by inverting a test; Wald embraces the normal
approximation but applies it on a scale (log-odds) where it's much better
behaved.

### Research-specific: reading panel b's error bars correctly
`DEC_AR`'s OR of 10.07 has a CI of (4.53, 22.38) — on the log axis this
looks like a roughly symmetric bar either side of $\log(10.07)$, but 22.38
is nearly 12 units above 10.07 while 4.53 is only about 5.5 below it; that
asymmetry-on-linear-scale/symmetry-on-log-scale is the direct visual
fingerprint of a Wald interval built on $\log(OR)$. Compare to `PAOD`,
OR 3.19 (0.38, 26.52): the interval spans nearly two orders of magnitude
because $a,b,c,d$ (56, 1, 123, 7 — see Section 4) are small enough that
$SE(\log OR) = \sqrt{1/56+1/1+1/123+1/7} \approx 1.09$ is large — one small
cell count ($b=1$) dominates the sum and inflates the whole interval,
exactly mirroring why PAOD's Wilson interval in panel a is also wide.

---

## 4. Haldane-Anscombe correction — "what to do when a cell count is zero"

### Grade-schooler
Now imagine one of your four boxes in the 2×2 table is completely empty —
say, zero patients without the condition and without the outcome. Dividing
by zero breaks the odds-ratio math. The Haldane-Anscombe trick is a small,
standard patch: pretend every box has an extra half a patient in it before
you do the division. It's not a real patient, just a tiny nudge that keeps
the math from breaking while barely changing the answer if the real counts
are reasonably large.

### High-schooler
$OR = ad/(bc)$ is undefined (division by zero) or infinite whenever any one
of the four table cells is exactly 0 — which can easily happen for a rare
comorbidity in a modest-sized study. The **Haldane-Anscombe correction**
adds $0.5$ to every one of the four cells ($a,b,c,d$) whenever *any* of them
is zero, before computing the OR and its standard error. This keeps the
estimate finite and the confidence interval computable, at the cost of a
small, well-understood bias toward OR = 1 (the "no effect" value) —
generally considered an acceptable trade for having a usable number at all.

### College student
Without the correction, both $OR=ad/bc$ and
$SE(\log OR)=\sqrt{1/a+1/b+1/c+1/d}$ are undefined whenever $b=0$ or $c=0$
(the OR itself), or produce $SE \to \infty$ whenever *any* cell is 0. The
correction replaces $(a,b,c,d)$ with $(a+0.5, b+0.5, c+0.5, d+0.5)$ — see
`odds_ratio_ci`, [`edareport.py:167-169`](../../module/edareport.py#L167-L169):

```python
corrected = (t == 0).any()
if corrected:
    t = t + 0.5
```

This is applied only when at least one cell is exactly 0 (not as a blanket
continuity correction on every table), so features with adequate cell counts
get the uncorrected, slightly more precise estimate, and only the tables
that would otherwise be mathematically undefined get the adjustment. In
Figure S2b, features that received the correction are flagged so a reader
can tell the estimate involved this adjustment rather than assume every OR
was computed identically.

### Research-specific: did this actually trigger in our cohort?
Perhaps counter-intuitively, **no feature in this dataset's Figure S2/Table 1
run needed the correction** (`or_corrected = False` for every binary feature
in `feature_statistics.csv`, including GBS and PAOD). Even GBS — the rarest
feature at 2/187 patients — has one case in each outcome group (1 Unconfirmed,
1 Confirmed), so all four cells of its table are $\ge 1$:

| | No GBS | GBS |
|---|---|---|
| Unconfirmed (n=57) | 56 | 1 |
| Confirmed (n=130) | 129 | 1 |

and PAOD's table (56, 1, 123, 7) is similarly cell-complete. The
`odds_ratio_ci` docstring even notes this design intent directly: *"The 0.5
correction keeps GBS (2 positives) and PAOD (8 positives) estimable instead
of returning infinities"* — the correction exists in the code as a safety
net for exactly this kind of rare-comorbidity study, and the caption
documents it as part of the method for full reproducibility/transparency,
even though this particular cohort's counts happened to stay just above the
threshold that would trigger it. A resample, a slightly different cohort
cut, or a rarer feature would very plausibly hit a zero cell, and the method
needs to be specified in advance either way — you can't decide *whether* to
correct after seeing whether a table happens to need it, without letting the
data quietly bias your methodology.

---

## 5. How this connects to the machine learning and counterfactual analysis

### 5.1 Figure S2 is univariate EDA — the same role as Figure S1 and Table 1
Just like the AUC/p/q story for continuous features (see
[`auc_bh_q_explained.md`](auc_bh_q_explained.md)), Figure S2's OR/Wilson-CI
story for binary features is a *screening* step, not the final answer. It
characterizes each of the 12 binary features **in isolation** against the
outcome, before any of them are combined by the multivariate model
(CatBoost, ROC-AUC 0.799). Filled vs. hollow markers in panel b encode
`q < 0.05` (Benjamini-Hochberg corrected across the same 22-feature family
described in Section 4 of the AUC note) — the same false-discovery-rate
logic applies here as for the continuous features, since binary and
continuous predictors are tested together and corrected together.

### 5.2 Diamonds mark the counterfactual-actionable features — and they are *not* the strongest ORs
The caption notes diamonds denote "the features available to the
counterfactual engine": `INSULIN`, `HBA1C`, `HPN`, `PAOD`, `DSLPDMIA`, `CKD`
(`config.dice.cf_features.actionable`, enforced in
[`cfreports.py:190`](../../module/cfreports.py#L190)). Of the six, `HBA1C`
is continuous (appears in Figure S1, not S2); the other five are exactly the
comorbidities that cluster around OR = 1 with wide CIs in panel b — none of
them reach $q<0.05$ except `INSULIN` (q = 0.046, barely). This mirrors the
point made in Section 5.5 of the AUC note for the same feature set: these
five were chosen because a clinician can plausibly act on them (start
insulin, manage lipids, control blood pressure, treat CKD, manage PAOD), not
because they show the strongest population-level association with DPN
status. A patient-specific counterfactual ("controlling DSLPDMIA and CKD
would flip this prediction") is a statement about *this model's* local
decision surface for *this patient*, not a claim that DSLPDMIA or CKD
strongly predicts DPN in general — panel b's own numbers (OR 1.76 and 2.14,
both non-significant) are the evidence that such a claim would be
unsupported at the population level.

### 5.3 This is the direct explanation for the comorbidity-ablation result
The caption's closing sentence states the payoff explicitly: *"This pattern
explains why the comorbidity ablation had little effect on model performance
despite comorbidities forming the majority of the modifiable feature set."*
Concretely: 5 of the 6 counterfactual-actionable features are comorbidities,
and Figure S2b shows all 5 sit close to OR = 1 with CIs comfortably
overlapping the null. If a model ablation experiment removes a set of
features that individually show almost no marginal association with the
outcome, it is expected — not surprising — that the model's overall
performance barely moves when those features are dropped. Figure S2 is what
lets that ablation *result* (a number: performance change under
comorbidity-ablation) be read as *evidence consistent with* rather than
*contradicting* the univariate picture, rather than a mysterious model
artefact. The neurological-exam signs (`DEC_LTS`, `DEC_PPS`, `DEC_VS`,
`DEC_AR`) are the opposite case: strong, significant ORs, but they are
*not* actionable (a clinician cannot "treat" a decreased ankle reflex the
way they can treat hypertension), so they were never eligible for the
counterfactual engine in the first place — they matter for prediction, the
comorbidities matter for actionability, and Figure S2 is the figure that
shows those are two different, only weakly overlapping, sets of features.

---

## 6. Cheat-sheet

- **Wilson CI** (panel a) — confidence interval for a single group's
  prevalence (%). Valid even for rare events / small groups, unlike the
  naive normal-approximation interval. Widest for GBS and PAOD because so
  few patients have them.
- **Odds ratio** (panel b) — ratio of a binary feature's odds in Confirmed
  vs. Unconfirmed patients. OR = 1 is the null; plotted on a log scale so
  effects of equal strength in either direction look symmetric.
- **Wald CI** (panel b's error bars) — the OR's confidence interval, built
  by applying the standard "estimate ± z·SE" recipe to $\log(OR)$ (where the
  normal approximation holds much better) and exponentiating back.
- **Haldane-Anscombe correction** — adds 0.5 to all four 2×2 cells only when
  at least one cell is exactly 0, to keep the OR and its CI finite. Present
  in the code as a safety net; not actually triggered by any feature in this
  particular cohort, since even GBS and PAOD have $\ge 1$ patient in every
  cell.
- Both panels describe the same 12 binary features from two angles: panel a
  is "how common is this in each group," panel b is "how much more common in
  one group than the other" — the same 2×2 tables underlie both.
- The comorbidities that dominate the counterfactual-actionable feature set
  are exactly the features with the weakest, least precise univariate
  associations in Figure S2 — which is *why* they were chosen (they're
  changeable, not necessarily predictive) and *why* ablating them barely
  moved model performance.
