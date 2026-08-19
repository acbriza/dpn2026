# Understanding AUC, the Mann-Whitney U test, and BH q-values in this study

This note explains three things that show up constantly in the EDA outputs
(`module/experiments/binary/eda/`): the **AUC** column, the **p** column, and the
**q** column. It explains each one four times, at increasing depth, and then ties
them directly to the tables/figures this project produces and to why the
machine-learning and counterfactual stages downstream are built the way they are.

Read top to bottom if you want the full picture, or jump straight to
[Section 5](#5-research-specific-what-this-means-for-this-study) if you just want
the version anchored to our actual numbers.

---

## 1. The one-paragraph version

For every candidate predictor (AGE, HBA1C, DM_DUR, ...), we ask: *do Confirmed and
Unconfirmed patients differ on this feature?* **AUC** answers "by how much" (an
effect size, 0.5 = no difference). **p** (from the Mann-Whitney U test) answers "how
surprising would this much difference be if there were actually no real difference
in the population?" **q** is p after being adjusted for the fact that we asked the
same question 22 times at once (Table 1) or 18 times at once (the nerve-conduction
audit in Table 2) — without that adjustment, testing many features guarantees some
will look "significant" by chance alone. All three numbers come from a **single**
underlying calculation (the rank-sum of the two groups), which is a point worth
sitting with: AUC and q are not two independent lines of evidence, they are the
same evidence reported twice, once as magnitude and once as calibrated surprise.

---

## 2. AUC — "how separated are the two groups?"

### Grade-schooler
Imagine every patient's measurement written on a card, and you lay all the cards in
a line from smallest to largest. If you can guess pretty well just by looking at
where a card sits in that line — "cards near the front are usually the healthy
kids, cards near the back are usually the kids with the condition" — then that
measurement is *good at separating* the two groups. AUC is a number from 0 to 1
that says how good that guessing game is. 1 means perfect guessing, 0.5 means
you're just as good as flipping a coin, and 0 means it's perfectly backwards
(the healthy kids are always at the back instead).

### High-schooler
AUC (**A**rea **U**nder the ROC **C**urve) measures how well one number can tell two
groups apart, without picking any particular cutoff. Concretely: pick one random
person from each group. AUC is the probability that the "Confirmed" person's value
is higher than the "Unconfirmed" person's value. AUC = 0.5 means the feature is
useless for telling the groups apart (a coin flip). AUC = 1.0 means it's a perfect
separator (every Confirmed patient's value is higher than every Unconfirmed
patient's value). AUC = 0.0 means it separates perfectly, just backwards. The
farther AUC sits from 0.5 in either direction, the stronger the feature.

### College student
AUC here is the **directional, rank-based concordance statistic**:

$$\text{AUC} = P(X_{\text{confirmed}} > X_{\text{unconfirmed}})$$

estimated non-parametrically from ranks rather than from a fitted curve — see
`auc_from_ranks` in [`edareport.py:124`](../../module/edareport.py#L124). It is
computed as

$$\text{AUC} = \frac{R_1 - \dfrac{n_1(n_1+1)}{2}}{n_1 n_0}$$

where $R_1$ is the sum of ranks belonging to the Confirmed group across the pooled,
jointly-ranked sample. This is numerically identical to the classic ROC-AUC (as
`sklearn.roc_auc_score` would compute it, ties handled via mid-ranks) but is
computed directly from ranks so a stratified bootstrap CI (2000 resamples,
[`edareport.py:140`](../../module/edareport.py#L140)) can be run cheaply for every
feature. Crucially, this same numerator ($R_1 - n_1(n_1+1)/2$) **is** the
Mann-Whitney U statistic — which is why AUC, p, and q described below are not
independent of one another.

### Where it appears in our figures
- **Table 1** (cohort characteristics): AUC per feature as the effect-size column.
- **Table 2** (feature eligibility / leakage audit): AUC for *all 40* recorded
  features, including the 18 nerve-conduction variables that are excluded from
  modelling.
- **Figure 2** (`fig2_univariable_discrimination`): a forest plot of AUC with
  bootstrap CIs for all 22 modelled features, ordered by strength.
- **Figure S1** (`s1_continuous_rainclouds`): AUC printed in each panel title for
  the 10 continuous predictors.
- **Figure S2** (`s2_categorical_features`): the categorical counterpart.

---

## 3. The Mann-Whitney U test and its p-value — "could this be chance?"

### Grade-schooler
Even if two teams are actually exactly the same at running, one team might still
happen to finish a little ahead just by luck on any given day. The p-value asks:
*if the teams truly were the same, how often would we see a gap this big or bigger,
just from luck?* A small p-value (like 0.001) means "this would almost never happen
by luck alone" — so we start to believe the teams really are different, not just
unlucky.

### High-schooler
The Mann-Whitney U test is the significance test that goes with the AUC idea above.
It doesn't compare averages — it compares whole *rank orders* between two groups,
which makes it robust to skewed or lumpy data (useful here: several of our features,
like `DM_DUR` and `FEET_PCT_ASYM`, are zero-inflated and nothing like a bell curve —
see [Figure S1](../../module/experiments/binary/eda/s1_continuous_rainclouds.png)).
It produces a p-value: the probability of seeing a rank-order difference at least
this extreme, *if* the two groups were really drawn from the same distribution. Small
p ⇒ the observed separation is unlikely to be a fluke.

### College student
Formally, `mannwhitneyu(x_confirmed, x_unconfirmed, alternative="two-sided")` tests

$$H_0: P(X_{\text{confirmed}} > X_{\text{unconfirmed}}) = 0.5$$

against a two-sided alternative, using the same rank-sum statistic $U$ as above.
Under $H_0$ its sampling distribution is known
(normal-approximated here given the sample sizes), yielding a p-value without any
assumption that the underlying feature is normally distributed — only that
observations are independent. This is why it's used uniformly for the 10 continuous
predictors instead of a t-test: distributional shape is not assumed, only ranks
matter. Binary features get the analogous **Fisher's exact test** or **chi-square**
test instead (`binary_test`, [`edareport.py:177`](../../module/edareport.py#L177)),
switching to Fisher's whenever any expected cell count falls below 5 (true for PAOD
and GBS in this cohort) since chi-square's asymptotic approximation breaks down
there.

### Worked example — how the rank-sum is actually computed
Take a toy case with 3 Unconfirmed and 4 Confirmed patients:

```
Unconfirmed: 2, 6, 9
Confirmed:   3, 5, 7, 10

Pool everyone and rank 1..7 (ties would get the average of their span — see below):
value:  2   3   5   6   7   9   10
group:  U   C   C   U   C   U   C
rank:   1   2   3   4   5   6   7
```

$$R_1 = 2 + 3 + 5 + 7 = 17$$

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2} = 17 - \frac{4 \cdot 5}{2} = 17 - 10 = 7$$

$$\text{AUC} = \frac{U_1}{n_1 n_0} = \frac{7}{12} \approx 0.583$$

The subtracted term $\frac{n_1(n_1+1)}{2}$ is just $1+2+\dots+n_1$ — the smallest
possible rank-sum Confirmed could have gotten, i.e. if every Confirmed value were
the lowest in the pool. $U_1$ is what's left after removing that baseline, and it
has an exact, checkable meaning: it equals the number of (Confirmed, Unconfirmed)
pairs where the Confirmed value wins. Confirmed by brute force here — 3 beats {2}
(1 win), 5 beats {2} (1), 7 beats {2, 6} (2), 10 beats {2, 6, 9} (3), total =
$1+1+2+3=$ **7**, matching $U_1$ exactly. This is why `auc_from_ranks`
([`edareport.py:124`](../../module/edareport.py#L124)) can get AUC from one
`rankdata()` call ($O(n \log n)$) instead of comparing every pair directly
($O(n_1 n_0)$) — the rank-sum is a shortcut for the pairwise-win count.

**Ties**: when values repeat (e.g. `DM_DUR` has exact zeros for ~13% of patients),
all tied values get the *average* of the ranks they jointly span, via
`scipy.stats.rankdata`'s default `"average"` method — e.g. three-way tied values
that would occupy ranks 4, 5, 6 all get rank 5 instead. This is also why the null
variance of $U$ used for the p-value is tie-corrected downward: a dataset with many
ties has fewer distinguishable orderings, so by chance alone $U$ is less variable,
and the test accounts for that rather than treating tied data as if it were
continuous.

**From rank-sum to p-value**: $U_1$ (equivalently $\text{AUC} \times n_1 n_0$) is
exactly what `mannwhitneyu` standardizes against its null distribution — under
$H_0$, $U_1$ has mean and variance

$$\mathbb{E}[U_1] = \frac{n_1 n_0}{2}, \qquad \text{Var}(U_1) = \frac{n_1 n_0 (n_1+n_0+1)}{12} \ \text{(reduced for ties)}$$

so

$$z = \frac{U_1 - \mathbb{E}[U_1]}{\sqrt{\text{Var}(U_1)}}$$

gives a p-value from the normal approximation. One rank-sum, computed once, becomes
the effect size (AUC), the test statistic ($U$), and — after standardizing — the
p-value.

---

## 4. BH q-values — "correcting for asking many questions at once"

### Grade-schooler
If you ask one coin "are you a magic coin?" and flip it 5 times and it lands heads
every time, that's pretty surprising. But if you ask *1,000 different coins* the
same question, just by luck a few of them will land heads 5 times in a row even if
none of them are magic. If you don't account for how many coins you asked, you'll
end up calling some perfectly normal coins "magic" by mistake. The BH correction is
a rule for adjusting your surprise-meter when you've asked the same question many
times, so you don't fool yourself.

### High-schooler
When you test 22 features at once, even if *none* of them truly matter, you'd
expect roughly 1 in 20 of them (5%) to show p < 0.05 purely by chance — that's what
"5% significance level" means per test. Test 22 features and you'd expect about 1
false alarm just from volume. The Benjamini-Hochberg (BH) procedure adjusts the raw
p-values into **q-values** that control the expected proportion of false alarms
among everything you call "significant" (the *false discovery rate*, FDR). A
feature with q < 0.05 means: among all the features across the whole study that
clear this same bar, we expect fewer than 5% of them to be false alarms — a much
safer standard than trusting 22 individual, uncorrected p-values.

### College student
`benjamini_hochberg` ([`edareport.py:206`](../../module/edareport.py#L206)) wraps
`statsmodels`' `multipletests(..., method="fdr_bh")`. Given p-values sorted
ascending $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, BH finds the largest $k$ such
that

$$p_{(k)} \le \frac{k}{m}\alpha$$

and rejects $H_0$ for all tests up to that rank; the q-value for each test is the
minimum FDR at which it would be called significant. This controls

$$\mathbb{E}\!\left[\frac{\text{false positives}}{\text{total rejections}}\right] \le \alpha$$

which is a weaker and more appropriate guarantee for exploratory screening than the
family-wise error rate control of a Bonferroni correction (which would be
needlessly conservative here, at the cost of missing real effects).

**One design choice worth calling out explicitly**: q-values in this study are
computed *separately* within two families —

```python
for modelled in (True, False):
    mask = stats.modelled == modelled
    stats.loc[mask, "q"] = benjamini_hochberg(stats.loc[mask, "p"].values)
```
([`edareport.py:428`](../../module/edareport.py#L428))

— the 22 modelled predictors as one family, and the 18 excluded nerve-conduction
variables as a separate family. Table 2's caption explains why: "*Pooling would
deflate the predictors' q-values, since the excluded set is significant by
construction and pushes every predictor to a later rank.*" The nerve-conduction
variables are part of the *definition* of the outcome (2009 Toronto consensus
criteria require an abnormal nerve conduction study for a Confirmed diagnosis), so
their p-values are enormous outliers by design — mixing them in would make every
real predictor's rank-based q artificially small look artificially large (worse),
distorting the correction for the family we actually care about.

### Where it appears in our figures
- **Table 1**: q-value per feature, computed across all 22 modelled predictors.
- **Table 2**: q-value per feature, computed within each of the two families above.
- **Figure 2**: marker fill (solid vs hollow) encodes q < 0.05.
- **Figure S1 / S2**: q printed in each panel title, alongside AUC.

---

## 5. Research-specific: what this means for *this* study

### 5.1 AUC and q are the same test, read twice — don't double-count them
Because AUC is built from the same rank-sum as the Mann-Whitney U statistic, a
panel like `NS  AUC 0.28, q < 0.001` in
[Figure S1](../../module/experiments/binary/eda/s1_continuous_rainclouds.png) is
not "two methods agreeing" — it's one calculation shown as a magnitude (AUC) and as
a calibrated surprise level (q). The features with the most extreme AUCs will
mechanically have the smallest q's, given similar sample sizes. What's actually
informative is the *pairing*: a feature can have a small q purely because n=187 is
enough to detect even a modest, unremarkable shift — small q does **not** imply the
groups are cleanly separable. Every panel in Figure S1 shows heavy visual overlap
between Confirmed and Unconfirmed, even for the q < 0.001 features. Statistical
significance here answers "is there *some* real shift", not "is this feature useful
for classifying an individual patient."

### 5.2 The headline numbers from this run

| Quantity | Value | Where |
|---|---|---|
| Modelled candidate predictors | 22 | Table 1, Figure 2 |
| ...of which reach q < 0.05 | 14 | Table 1 |
| Best single predictor | `DEC_AR`, AUC 0.748 | Table 2, Figure 2 |
| Nerve-conduction (excluded) AUC range | 0.63 – 0.93 (absolute) | Table 2 |
| Multivariable model (CatBoost) ROC-AUC | 0.799 | Figure 2 reference line |
| Actionable (counterfactual) features' AUC range | 0.514 – 0.585 | Table 2, §5.3 below |

### 5.3 Why the nerve-conduction features have huge AUC — and why that's *disqualifying*, not exciting
Table 2 exists specifically to show that the 18 nerve-conduction study (NCS)
variables reach AUC up to 0.93 with q-values as small as `1e-19` — far stronger
than any permitted predictor. This is presented as **evidence for exclusion**, not
evidence of value: a 2009 Toronto-consensus "Confirmed DPN" diagnosis *requires* an
abnormal nerve conduction study, so an NCS variable predicting the outcome is
circular — it's partially *reading the label off the outcome definition itself*.
Using it as a model feature would be target leakage. This is a case where the
statistical story (extremely strong, extremely significant) is exactly the signal
that the feature must be thrown away, which is the opposite intuition from every
other use of AUC/q in this report — worth remembering when skimming Table 2.

### 5.4 The gap between univariate AUC (~0.5-0.75) and the fitted model's AUC (0.799)
No single permitted feature gets close to the CatBoost model's 0.799 ROC-AUC — the
best univariate feature (`DEC_AR`) tops out at 0.748, and most modelled features sit
much lower. This is the concrete, numeric argument for why the pipeline moves past
univariate EDA screening into multivariate machine learning: no single measurement
substitutes for the pattern across all 22 features combined, and the EDA stage's
job is to characterize and audit each predictor individually before they're handed
to a model that can combine them.

### 5.5 The counterfactual-actionable features are *not* chosen by q-value — and that's intentional
This is the connection most worth internalizing. The six features permitted to vary
during counterfactual generation
(`config.dice.cf_features.actionable` = `INSULIN, HBA1C, HPN, PAOD, DSLPDMIA, CKD`,
enforced in [`cfreports.py:190`](../../module/cfreports.py#L190)) are picked by
**clinical actionability** — things a patient or clinician could plausibly change —
not by statistical strength. Looking at their actual univariate numbers from this
run:

| Feature | AUC | q | Significant at q<0.05? |
|---|---|---|---|
| INSULIN | 0.585 | 0.046 | barely yes |
| DSLPDMIA | 0.570 | 0.106 | no |
| CKD | 0.554 | 0.114 | no |
| HBA1C | 0.545 | 0.379 | no |
| PAOD | 0.518 | 0.482 | no |
| HPN | 0.514 | 0.712 | no |

Five of the six actionable features do **not** clear the q < 0.05 bar on their own,
and HBA1C — arguably the most clinically central marker of glycaemic control in
diabetes — is nowhere close (q = 0.38). At first glance that can look like a
contradiction: why let the counterfactual engine vary features that "don't matter"?

It isn't a contradiction, because the two questions are different in kind:

- **Table 1/2's q-value** asks a *population-level, marginal* question: across the
  whole cohort, does this feature's distribution differ between Confirmed and
  Unconfirmed on average, ignoring every other feature?
- **Counterfactual generation** asks a *model-conditional, individual-level*
  question: for *this one borderline patient*, holding the fitted multivariate
  model fixed, is there a small actionable change to their INSULIN/HBA1C/HPN/etc.
  that flips *their* prediction?

A feature can be a weak, noisy discriminator across 187 people on average and still
be exactly the lever that moves one specific patient near the decision boundary —
that's a property of the fitted model's local decision surface for that patient,
which a population-level rank test has no way to see. So it is expected, not a
red flag, that the actionable set's univariate AUC/q values are unremarkable: they
were never meant to be the strongest predictors, only the *changeable* ones. The
EDA's job here is to make sure we go into counterfactual generation with an honest
picture of how weak these features are individually, so a resulting counterfactual
("increase INSULIN and DSLPDMIA control to flip this prediction") is read as a
model-specific, patient-specific recommendation — not overinterpreted as "INSULIN
strongly predicts DPN in general," which Table 2's own numbers show is not true.

### 5.6 Quick map: every artefact that carries AUC / p / q

| Artefact | AUC | p / q | Family the q is computed within |
|---|---|---|---|
| `table1_cohort_characteristics` | ✓ | ✓ | 22 modelled features |
| `table2_feature_eligibility` | ✓ (all 40 features) | ✓ | 22 modelled / 18 excluded NCS (separately) |
| `fig2_univariable_discrimination` | ✓ (forest, with bootstrap CI) | q via marker fill | 22 modelled features |
| `s1_continuous_rainclouds` | ✓ (panel titles) | q (panel titles) | 22 modelled features |
| `s2_categorical_features` | ✓ | q | 22 modelled features |
| `s5_actionable_features` | ✓ | q | 22 modelled features (subset: the 6 actionable) |
| `s6_ncs_descriptives` | — | — | descriptive only, no test |
| `fig1a_participant_flow`, `fig1bh`/`fig1bv_classification_composition` | — | — | cohort composition only, no statistical test |

---

## 6. Cheat-sheet

- **AUC** — effect size. 0.5 = no separation. Distance from 0.5 = strength, in
  either direction.
- **p** — from the Mann-Whitney U test (continuous features) or Fisher's/chi-square
  (binary features). Probability of seeing a rank-order imbalance this large (i.e.
  an AUC this far from 0.5) if the two groups truly had no tendency to differ.
- **q** — p, corrected for testing many features at once (Benjamini-Hochberg FDR).
  Use q, not raw p, whenever comparing across the feature table.
- AUC and q are **not independent evidence** for continuous features — they come
  from the same rank statistic.
- Small q ⇏ clinically separable; it only means the shift is unlikely to be pure
  chance at this sample size.
- High AUC/tiny q for the nerve-conduction features is a **disqualifying** signal
  (leakage), not a useful one.
- The gap between best univariate AUC (0.748) and the fitted model's AUC (0.799)
  is the quantitative case for multivariate ML over univariate screening.
- Counterfactual-actionable features are chosen for **clinical changeability**, not
  statistical strength — most have q > 0.05, and that is expected, not a defect.
