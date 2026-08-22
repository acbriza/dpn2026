# CatBoost vs. Random Forest: rationale for small-n and counterfactual generation

## The numeric caveat first

Worth flagging: in the feature-subset comparison in
[selection_metrics_summary.csv](../../module/experiments/binary/postreport/final_202608/selection/selection_metrics_summary.csv),
Random Forest edges CatBoost on nearly every raw metric (AUPRC 0.892 vs 0.889, Youden 0.446
vs 0.425, specificity 0.529 vs 0.512), and the gap is well inside one standard deviation. So
the argument for CatBoost can't rest on raw discriminative performance in this table — it has
to rest on properties that matter *downstream*, for small-n stability and for the
counterfactual pipeline specifically. That's actually the stronger, more defensible argument
anyway.

## Small-dataset handling (n=190)

- **Ordered boosting vs. bootstrap aggregation.** CatBoost's ordered boosting computes each
  tree's residuals using a permutation-restricted subset that excludes the point being
  scored, specifically to prevent the target leakage that classic gradient boosting suffers
  from when the same rows repeatedly inform both the split statistics and the gradient. RF
  sidesteps this differently (bagging + averaging), but with only ~142 training rows per
  fold, individual RF trees are grown to full depth and can produce many single- or
  few-sample leaves — high-variance, low-bias estimators that bagging only partially tames.
  CatBoost's symmetric (oblivious) tree structure is a much stronger structural regularizer:
  every node at a given depth uses the *same* split condition, which drastically shrinks the
  hypothesis space relative to RF's unconstrained CART trees — exactly the kind of
  regularization that matters more as n shrinks.
- **Native categorical handling.** The dataset mixes continuous labs, ordinal exam scores,
  and categorical comorbidity flags. CatBoost encodes categoricals natively via ordered
  target statistics; RF requires one-hot or ordinal pre-encoding. One-hot on n=190
  fragments already-scarce rows across sparse dummy columns and inflates effective
  dimensionality relative to sample size — a real overfitting risk at this scale.
- **Built-in imbalance handling instead of SMOTE.**
  [mss_edits.md:13](../../manuscript/references/prompts/mss_edits.md#L13) explicitly notes
  SMOTE was deliberately avoided in favor of CatBoost's native class-weighting. That's a
  small-n-specific decision: synthetic interpolation in a 190-patient clinical space risks
  generating implausible synthetic patients near the decision boundary, and — critically for
  this pipeline — if such points ever leaked into the counterfactual-generation frame, they'd
  contaminate the clinical plausibility of the explanations. Loss-level class weighting
  avoids that entirely.

## Counterfactual generation (DiCE)

This is the sharper argument, and it's mechanical, not just aesthetic. The wrapper in
[utils2/counterfactuals.py](../../module/utils2/counterfactuals.py) runs DiCE's **sklearn
backend with the genetic algorithm** — meaning the counterfactual search's fitness function
is driven directly by `predict_proba`, treating the model as a black box. The *shape* of that
probability surface determines how well the search performs:

- **RF's `predict_proba` is a vote-share average** — a coarse, quantized statistic assembled
  from ~100 trees' leaf-class proportions. On a small, high-purity dataset, many leaves
  become pure or near-pure, so nearby candidate patients in DiCE's mutation/crossover steps
  often land on identical or near-identical probability values — flat regions with no
  gradient signal for the genetic algorithm to climb.
- **CatBoost's output is an additive sum of ~1000 shallow trees at a small learning rate**,
  passed through a sigmoid — a far finer-grained, closer-to-continuous score. That directly
  benefits a fitness-driven search: smaller perturbations produce smoother, more monotonic
  probability movement, which is what "minimal sufficient change" claims (the
  sufficiency/necessity checks in `cfreports.py`) actually depend on for clinical
  plausibility.
- The pipeline already found DiCE's genetic search to be a fragile, plateau-sensitive process
  even under CatBoost —
  [cfreports_refactor.md:158-177](../../module/cfreports_refactor.md#L158-L177) documents it
  silently violating `features_to_vary` constraints in low-signal regions (worse and less
  recoverable under LogisticRegression's linear surface, per that table). A coarser RF
  probability surface would only expand the flat regions where that kind of search
  instability happens.
- Native categorical support again matters here: DiCE's `permitted_range`/`features_to_vary`
  operate per-column in the model's own encoding. Native categorical handling means the
  search — and the resulting counterfactual "recipes" reported to clinicians — stays in the
  original clinical variables (e.g., a comorbidity flag) rather than requiring one-hot
  dummies that then need error-prone post-hoc collapsing back into a single categorical
  decision.

## Synthesis

Given RF and CatBoost are statistically indistinguishable on raw AUPRC in this table, the
deciding factors are properties that don't show up in a single metrics column but matter
directly for this study's two hard requirements:

1. **Regularization suited to n=190.** CatBoost's ordered boosting and symmetric-tree
   structure impose a tighter, more principled hypothesis-space constraint than RF's
   bagged-but-unconstrained CART trees, and its native categorical/imbalance handling avoids
   the dimensionality inflation (one-hot) and synthetic-data risk (SMOTE) that a small
   clinical cohort is especially vulnerable to.
2. **Compatibility with a `predict_proba`-driven genetic counterfactual search.** DiCE's
   sklearn-backend genetic algorithm needs a smooth, high-resolution probability surface to
   reliably find minimal, valid, diverse counterfactuals. CatBoost's additive low-learning-rate
   ensemble supplies that; RF's vote-share averaging is comparatively quantized and would
   widen the flat regions where the search already struggles (as documented empirically in
   [cfreports_refactor.md:158-177](../../module/cfreports_refactor.md#L158-L177)).

In other words, the choice isn't "CatBoost predicts better" — on this table it doesn't,
meaningfully. It's "CatBoost is the more defensible engine for a pipeline whose second half
depends on interrogating a smooth, well-behaved probability function to generate clinically
actionable counterfactuals from 190 patients." That's a methodological argument, not a
leaderboard one, and it's the version worth putting in the manuscript if reviewers push on
the RF/CatBoost near-tie — it also dovetails with the existing justification at
[main.tex:690](../../manuscript/main.tex#L690), which already leans on "native categorical
handling" but doesn't yet make the counterfactual-search-compatibility case explicit.
