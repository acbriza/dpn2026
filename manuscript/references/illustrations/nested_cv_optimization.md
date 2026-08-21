# Nested CV Optimization Infographic — Build Notes

Companion document for `documentation/nested_cv_optimization.svg`.
Generated 2026-08-13 on branch `codereviewed`.

---

## 1. Originating request

The figure was produced from the following prompt (verbatim):

> Generate a infographic  for the nested_cv_optimization. It will be used for a research paper publication.
> Put special attention to how data was split (show and illustrate size per split) for the optimization steps. Include which metrics were used as the basis for optimization. It would be good to mention also what parameters were searched and what parameters were kept constant.
>
> Use SVG format so that the graphics can be edited. Ensure all shapes (e.g. rectangles for layers, circles for nodes) are distinct vector elements, and use basic fonts like Arial so it imports cleanly into Canva as editable elements. Wrap it in a single .svg code block.  Save the resulting file in the documentation folder

---

## 2. Sources of truth

Every number on the figure traces to one of these:

| What | Source |
|---|---|
| Algorithm / control flow | `module/utils2/optimization.py` → `nested_cv_optimization()` |
| Caller, param space construction | `module/optreport.py` (lines 66–104) |
| All hyperparameters and CV settings | `module/experiments/bin_opt_final_202608.yml` |
| Cohort, features, target | `dataset/EAMC_DPN_Dataset.xlsx` via `module/dataload.py` → `DPN_data` |

`module/optreport.py` resolves its config directory unconditionally at
`optreport.py:37`:

```python
config_path = Path(script_dir / 'experiments')
```

so the config is always read from `module/experiments/`. Worth knowing when
picking the right file to cite, because `module/configs/` holds ~15 similarly
named YAMLs with different values — e.g. `l2_leaf_reg` min is `0.02` in
`configs/bin_opt_final_auprc_f1.yml` but `1.0` in the executed
`experiments/bin_opt_final_202608.yml`. Only the latter is what the figure
documents.

One stale pointer to be aware of: the sample filename in the comment at
`optreport.py:30` is `bin_opt_final.yml`, which exists only in `configs/`.
`experiments/` contains just `bin_opt_final_202608.yml` and
`bin_sel_final_202608.yml`, so that sample would not resolve if used verbatim.

---

## 3. How the numbers were derived

Nothing on the figure was estimated. Two probes were run.

### 3.1 Cohort shape

The dataset was loaded through the project's own loader so that the same column
exclusions apply (non-data columns dropped, NCS-derived columns removed):

```python
from dataload import DPN_data
import ymlconfig

config = ymlconfig.dict_to_namespace(
    ymlconfig.load_config(Path('experiments/bin_opt_final_202608.yml'))
)
D = DPN_data(str(Path('.').resolve() / config.data.dataset_path))
D.load(classification=config.experiment.classification_type)

dfdpn = D.df
data_cols = dfdpn.drop(D.non_data_cols, axis=1, errors="ignore").columns
no_ncs = [c for c in data_cols if c not in D.ncs_cols]
X, y = dfdpn[no_ncs], dfdpn[D.get_target_column()]
```

Result:

- `X.shape == (187, 22)`
- target column: `Confirmed_Binary_DPN`
- class counts: `{1: 130, 0: 57}` → 69.5 % / 30.5 %
- features: `SEX, AGE, SUBJ, DM_DUR, INSULIN, HBA1C, HPN, PAOD, DSLPDMIA, CKD, GBS, DEC_VS, DEC_PPS, DEC_LTS, DEC_AR, MNSI, FEET_MEAN_ESC, FEET_PCT_ASYM, HAND_MEAN_ESC, HAND_PCT_ASYM, NS, CAS`

Run with the `dpncf` conda environment (`~/.conda/envs/dpncf/bin/python`) — the
system Python 3.9 has an `openpyxl` too old for this pandas build.

### 3.2 Exact split sizes

Rather than dividing 187 by 4 and rounding, the real splitter objects were
enumerated with the production seed so that stratification rounding is exact:

```python
y = np.array([1]*130 + [0]*57)
outer = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=42)
for i, (tr, te) in enumerate(outer.split(X, y)):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + i)
    ...
```

Observed ranges across all 40 outer iterations and their inner folds:

| Split | n | DPN+ | DPN− |
|---|---|---|---|
| Full cohort | 187 | 130 | 57 |
| Outer train | 140–141 | 97–98 | 42–43 |
| Outer test (held out) | 46–47 | 32–33 | 14–15 |
| Inner train | 93–94 | — | — |
| Inner validation (held out) | 46–47 | 32–33 | 14–15 |

Inner-CV seed is `random_state + fold_idx` (`optimization.py:103`), i.e. 42…81
across the 40 outer iterations.

### 3.3 Fit count

```
40 outer iterations x (100 trials x 3 inner fits + 1 full refit + 3 OOF refits)
= 40 x 304
= 12,160 CatBoost fits
```

---

## 4. What the figure asserts, panel by panel

| Panel | Content |
|---|---|
| **A** | Cohort: 187 patients, 22 predictors, class balance bar |
| **B** | Outer loop: `RepeatedStratifiedKFold(k=4, n_repeats=10)` = 40 iterations, `random_state=42`; one repeat drawn as 4 rows; train/test sizes |
| **C** | Inner loop: `StratifiedKFold(k=3)` on outer-train only, splits materialised once and reused; Optuna TPE, 100 trials, objective = mean AUPRC over 3 inner folds |
| **D** | Three-step refit + threshold: full-outer-train refit → 3 OOF refits → PR-curve threshold maximising F-score at β = 1 |
| **E** | Outer-test evaluation, the 13 recorded metrics, pooling as mean ± 95 % CI (z = 1.96, NaN-aware) |
| **F** | Searched hyperparameters and their ranges/priors |
| **G** | Held-constant settings |
| **H** | Which criterion governs which stage |
| **I** | Fit accounting and the full-coverage assertion |

### Metrics, by role

The figure deliberately separates three different metrics that are easy to
conflate in a methods section:

1. **Inner-loop objective** — `auprc` (`average_precision_score`), meaned over the
   3 inner validation folds. This is what Optuna maximises.
2. **Threshold criterion** — `fscore` with `fscore_beta: 1`, maximised on the
   precision–recall curve of the inner OOF probabilities.
3. **Outer-loop reporting** — accuracy, precision, sensitivity, specificity, F1,
   F1.25, F1.5, F1.75, F2, Youden J, ROC-AUC, AUPRC, plus the selected threshold.
   No optimisation happens at this level.

Both (1) and (2) are switchable in config (`optimization_metric` accepts
`roc-auc`/`auprc`; `threshold_selection_metric` accepts `roc-auc`/`fscore`). The
figure documents the values actually set in `bin_opt_final_202608.yml`.

### Searched vs. constant

Searched (Optuna TPE, 100 trials/iteration, sampler seed `42 + iteration`):

| Parameter | Range | Sampling |
|---|---|---|
| `depth` | 4 – 10 | integer, uniform |
| `learning_rate` | 0.003 – 0.3 | float, log-uniform |
| `l2_leaf_reg` | 1.0 – 10.0 | float, uniform |
| `scale_pos_weight` | 0.3 – 1.0 | float, uniform |

Constant: `CatBoostClassifier`, `loss_function=Logloss`, `eval_metric=AUC`,
`iterations=500`, `early_stopping_rounds=50`, `random_state=42`, `verbose=0`.

The fixed entries are not returned by `study.best_params` (which only carries
`trial.suggest_*` keys), so the code stashes the full dict in a trial user-attr
`full_params` and reads it back for the refits — see `optimization.py:113-123`
and `optimization.py:166-167`. Without that, the refits would silently fall back
to CatBoost's own defaults.

---

## 5. Design decisions

### Canva / editability constraints

Driven by the request that shapes import as distinct editable elements:

- Every shape is a discrete `<rect>`, `<circle>`, `<line>`, or `<polygon>`.
- **No** `<style>` block, `<defs>`, `<use>`, `<symbol>`, `<marker>`, `<pattern>`,
  gradient, or CSS class. Arrowheads are real `<polygon>` triangles, not markers.
- All styling is inline presentation attributes.
- Every `<text>` is a standalone element with `font-family="Arial"` — no
  `<tspan>`, no multi-line text, so each label moves independently.
- Canvas is `1660 x 1260` with a matching `viewBox`.

### Colour

Hue encodes **split role only**; class composition uses grayscale so the two
encodings never compete.

| Role | Hex |
|---|---|
| Used to fit (outer train, inner train) | `#2a78d6` |
| Outer test — held out | `#eb6834` |
| Inner validation — held out | `#1baf7a` |
| Hyperparameter search accent | `#4a3aa7` |
| DPN-positive / DPN-negative | `#52514e` / `#c3c2b7` |

Inner-train reuses the outer-train blue rather than taking a fourth hue. That is
deliberate: inner train *is* training data, and reusing the colour makes the
nesting legible. An earlier draft used a light blue tint (`#86b6ef`) for inner
train; it was rejected because a same-hue tint fails as a distinct categorical
slot (chroma floor, and normal-vision ΔE ~10–15 against the base blue).

The four-colour set was validated against a white surface with all pairs in
play:

```
[PASS] Lightness band       all 4 inside L 0.43-0.77
[PASS] Chroma floor         all 4 >= 0.1
[PASS] CVD separation       worst #1baf7a<->#eb6834 dE 9.2 (deutan)
[PASS] Normal-vision floor  worst #4a3aa7<->#2a78d6 dE 16.3
[WARN] Contrast vs surface  #1baf7a at 2.82:1 - relief required
```

The contrast warning is discharged by direct labelling: every coloured block
carries an in-place `train` / `val` / `test` label, so nothing depends on colour
alone. This also keeps the figure readable in grayscale print.

### Verification

The SVG was rasterised (`convert -density 96`) and inspected before delivery, to
check for label collisions and overflow that a colour validator cannot catch. One
issue was found and fixed (two lines in panel I were 12 px apart and collided);
the DPN-negative percentage was also restored to match the positive-class label.

---

## 6. Caveats worth stating in the manuscript

Both are visible in the code but not evident from the figure alone.

1. **`early_stopping_rounds` is inactive in the panel-D step-1 refit.** That refit
   intentionally consumes all of outer-train, so no `eval_set` exists and CatBoost
   trains the full 500 iterations regardless. It *is* active during the inner-loop
   trials and the OOF refits, where the held-out inner fold serves as `eval_set`.
   Documented in-code at `optimization.py:169-173` and in `optreport_refactor.md`.

2. **The 40 outer iterations are not independent.** They are 10 repeats over the
   same 187 patients. The confidence interval in `mean_confidence_interval()` uses
   `n = 40` as though they were, which understates the interval width. This is
   conventional for repeated CV, but reviewers of clinical ML work do raise it, so
   it likely belongs in the limitations section.

Additional detail worth carrying into the methods text: the threshold is selected
on inner-CV OOF predictions of the outer-training set and never on the outer test
fold, which is what keeps the outer estimate from being optimistically biased.
The code also asserts at runtime that, within each repeat, the four test folds
tile all 187 indices (`optimization.py:289-290`).

---

## 7. Regenerating

The SVG is hand-authored, not script-generated. To update it after a config
change, edit `documentation/nested_cv_optimization.svg` directly and re-check the
numbers against §3. To preview:

```bash
convert -density 96 -background white \
  documentation/nested_cv_optimization.svg /tmp/ncv.png
```
