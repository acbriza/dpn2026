# `cfreports.py` & `counterfactuals.py` Review & Refactor Summary

_2026-08-16_

Session summary of a correctness review and documentation/refactoring pass on
[cfreports.py](cfreports.py) and [utils2/counterfactuals.py](utils2/counterfactuals.py)
(the counterfactual reporting pipeline), intended to run as:

```
python module/cfreports.py bin_cf_final_202608.yml
```

Findings marked **reproduced** were confirmed by executing the code, not by inspection
alone. Several of the most serious ones were only visible at runtime.

## Applied changes

### Correctness bugs

- **`cfreports.py` could not run at all against the current trained-models file.** The
  per-split refit called `CatBoostClassifier(**best_params, verbose=0)`, but `best_params`
  from the optimization stage already contains `'verbose': 0` on all four splits, so Python
  raised `TypeError: got multiple values for keyword argument 'verbose'` before CatBoost was
  reached. The script died at the first split, before any counterfactual work. This is the
  same defect already fixed in `expreport.py` during its review — the two scripts shared the
  pattern. **Reproduced** end to end.

- **The refit produced a different model from the one whose threshold is reported.** Rather
  than repair the refit, it was removed: the pipeline now uses `split_results[midx]['model']`,
  the model the optimization stage trained and selected. Measured against the stored models
  on every split's test set:

  | refit variant | max abs prob. difference | label flips at tuned threshold |
  |---|---|---|
  | no `cat_features`, `random_seed=42` | **0.0000** | **0 / 47** |
  | no `cat_features`, no seed | up to 0.086 | up to 3 |
  | `cat_features`, no seed (what the code did) | up to **0.113** | up to **3** |

  This matters beyond reproducibility: the refitted model fed `get_instances_of_interest`,
  so the *study population* — which patients count as borderline or misclassified — was
  selected by a model that differs from the one whose threshold and metrics are published.
  The refit existed to set `cat_features`, because DiCE hands the model `category` dtype
  columns that a CatBoost model trained without `cat_features` rejects; `CatBoostWrapper`
  now casts those columns back instead, which removes the need for the refit entirely.

- **DiCE returns categorical features as strings, corrupting every downstream analysis.**
  `'0' != 0`, so `get_most_changed_feature` reported **all 12 categorical features as changed
  in 26 of 26 counterfactuals** — including `SEX` and `SUBJ`, which are not in
  `features_to_vary` and cannot change — and `get_local_cf_distances` raised
  `TypeError: unsupported operand type(s) for -: 'str' and 'float'`. The failure was
  path-dependent and so had gone unnoticed: a *resumed* run read the counterfactuals back
  through `pd.read_csv`, which converts the types, while a *fresh* run did not. That is why
  instance `096` in the existing outputs has no distance files while resumed instances do.
  Fixed at the source in `generate_diverse_cfs`, before `drop_duplicates` so a string row and
  its numeric twin now deduplicate. **Reproduced**.

- **Reports were labelled with the wrong patient.** Both heatmaps derived the patient ID from
  the dataframe row — one as `qidx`, the other as `qidx + 1` — so the same patient carried two
  different numbers across the outputs, and neither was the patient's code. Cleaning drops 3
  rows (35, 45, 172 per `cleaning_report.txt`) and resets the index, so `qidx + 1` is correct
  only up to row 34 and drifts by 1, 2 and then 3 thereafter. Verified against
  `DPN_data.index_to_patient_code`: row 96 is patient 99, not 97. See _Patient identity_ below.

- **`get_most_changed_feature` never wrote its intended header.** Summing the boolean mask
  gives a `Series`; the `reset_index()` result was discarded and `.columns = [...]` on a
  Series is silently ignored by pandas, so the file carried the default `,0` header. It also
  ranked `Confirmed_Binary_DPN` first in every report, because the outcome column is present
  in the counterfactuals but not in the query instance, making the comparison `NaN`.

- **`get_local_cf_distances` mutated its caller's frame, miscounted sparsity, and ignored
  `sort_by`.** It wrote `sparsity`, `L1_dist` and `L2_dist` into the caller's `df_dcf`, so
  those columns leaked into the progressive filtering and re-plotting that run afterwards.
  The same phantom outcome column inflated sparsity by one for every row (a counterfactual
  identical to the instance scored 1). Both `sort_values(...)` results were unassigned, so
  `sort_by` did nothing. Now works on a copy, compares shared feature columns only, and
  applies the sort while preserving row labels so a row still traces to its heatmap entry.

- **`--no-replot` was dead.** The flag was parsed, stored and passed all the way into
  `generate_local_cf_reports`, which never read it. Figures were always redrawn.

- **Global importance: `n_cpus=-1` did the opposite of what it documented.** The docstring
  said "-1 to use all", but `if n_cpus < 2` routed `-1` to the serial path — and had it
  reached the parallel branch, `np.array_split(X_test, -1)` raises `ValueError`. The parallel
  branch was also mis-shaped: it returns a features×1 frame while the plotting code expects
  the serial branch's 1×features layout, so `s = df_imp.iloc[0]` would have yielded a single
  feature. Fixed, and the chunk count is now capped at `len(X_test) // 10` because DiCE
  rejects a global importance call with fewer than 10 query instances — with ~38-row test
  splits, 3 chunks is the ceiling regardless of core count. `-1` now resolves to 80% of
  available cores. **Reproduced**, including the aggregation's validity:

  | feature | serial (whole set) | parallel (3 chunks) | abs diff |
  |---|---|---|---|
  | HBA1C | 0.9600 | 0.9633 | 0.0033 |
  | HPN | 0.7317 | 0.7267 | 0.0050 |
  | INSULIN | 0.0133 | 0.0100 | 0.0033 |
  | CKD | 0.0100 | 0.0083 | 0.0017 |

  Identical ranking, differences within the genetic search's own run-to-run noise, as
  expected: `summary_importance` is a per-instance mean, so a size-weighted average of chunk
  means equals the whole-set mean.

- **`timeout.py` could report finished work as a timeout.** The parent called
  `p.join(timeout=seconds)` *before* draining the result queue, so a result larger than the
  OS pipe buffer left the child blocked on flush and apparently alive. **Reproduced**: an 8 MB
  frame that computes in 0.1 s raised `TimeoutError` after the full budget. Now the queue is
  drained first; the same payload returns in 0.1 s.

- **A "no counterfactuals found" run cached an instance-only csv that later runs accepted as
  a valid result**, producing empty reports that looked real. This bit the verification run
  itself. A single-row file is now treated as a cache miss.

- Smaller fixes: `get_global_permitted_range` raised `UnboundLocalError` when saving with
  `verbosity=0` (latent — the caller hardcodes `verbosity=1`); `check_sufficiency` built its
  result frame inside the loop and returned `None`; the per-batch figure height shrank
  cumulatively across batches, so every short batch permanently halved later ones;
  `SPSC_L` appeared twice in the report's metadata table so `SPSC_R` was never displayed;
  an empty `nofiltering/` tree was created on every progressive run; `dataset_path` was
  resolved by slicing `[3:]` off `'../dataset/...'`, which only works when the caller's
  working directory is the repo root (the sibling report scripts already resolve against
  `script_dir`); `/n` instead of `\n` in two console messages.

### Patient identity

Everything the pipeline writes is now keyed by the patient's code in the source spreadsheet
rather than the cleaned dataframe row:

| | before | after |
|---|---|---|
| folder | `nofiltering/038/` | `nofiltering/040/` |
| csv files | `catboost_split0_local_cf.csv` | `catboost_split0_patient040_local_cf.csv` |
| figures | `..._local_cf_39.png` (wrong patient) | `..._local_cf_patient040.png` |
| `instances_of_interest.csv` | indexed by `qidx` | indexed by `patient_code` |
| CLI | `--instances 38` | `--patient_codes 40` |

`instances_of_interest.csv` keeps `qidx` as a column, and the in-memory frames stay indexed by
dataframe row, because that is what the pipeline looks up internally. No `patient_code` column
was added to `local_cf.csv`: that file is read back as the counterfactual set, so an extra
column would become a phantom feature in every diff — the same class of bug as the outcome
column above.

### Counterfactuals that violate `features_to_vary`

1–2 of ~19 counterfactuals for instance 38 came back with `SUBJ` changed, although `SUBJ` is
on the unactionable list and DiCE was told to hold it fixed. Such a counterfactual is not
actionable advice, so `drop_cfs_outside_features_to_vary` now removes them before the csv is
written — meaning every figure and file describes the same set — printing the count and
saving the dropped rows to `*_local_cf_unactionable.csv`.

The cause was investigated and is **not** the way DiCE is called here. DiCE's genetic search
pins non-varied features to the query instance in all three places that build candidates
(`do_random_init`, `do_KD_init`, and the mutation branch of `mate`), and a clean sklearn
reproduction stayed at **zero violations across 300+ counterfactuals** while varying each
structural difference in turn:

| variant | violations |
|---|---|
| baseline, plain sklearn model | none (60 CFs) |
| query instance from a frame other than the `dice_ml.Data` frame | none (60 CFs) |
| partial `permitted_range` on `generate_counterfactuals` | none (60 CFs) |
| partial `permitted_range` on `dice_ml.Data` *and* on generate | none (60 CFs) |
| dataframe column order differing from DiCE's continuous+categorical order | none (60 CFs) |

It also reproduces with DiCE's own default parameters on the real explainer, which rules out
the config's `proximity_weight` / `diversity_weight` / `categorical_penalty` / `algorithm`
settings. The remaining suspects are the CatBoost model behind `CatBoostWrapper` and the real
data/schema; a RandomForest-on-real-data comparison was started but had not finished within
this session. Worth reporting upstream once isolated.

### Documentation & cleanup

- Module docstring for `counterfactuals.py` describing the per-patient pipeline and the two
  conventions that run through it (patient codes, and normalising DiCE's output).
- Docstrings added for `test_wrapped_model`, `get_global_permitted_range`,
  `get_local_permitted_range`, `generate_diverse_cfs`, `generate_local_cf_reports`, both
  heatmap functions, `get_most_changed_feature` and `get_local_cf_distances`. The
  `get_global_importance` docstring documented a `total_CFs` parameter it does not take and
  misdescribed `n_cpus`.
- `TIMEOUT_PRESETS` replaces four module-level wrapper functions and a four-branch `if/elif`
  chain; `cfreports.py` takes `--gen-timeout`'s choices and its help text from the same dict,
  so the CLI cannot drift from the presets (this also retired the `3houra` typo).
- `_finish_figure()` replaces the `plt.close(fig) if backend in ["Agg"] else plt.show()`
  idiom repeated at three call sites.
- `instance_artifact_filename()` builds every per-patient filename in one place.
- Unused imports removed (`time`, `roc_curve`, `roc_auc_score`); the two mid-file imports
  (`joblib.Parallel`, `LinearSegmentedColormap`) moved to the top.
- Mutable default arguments (`highlight_features=[]`, `permitted_range={}`) replaced with
  `None`; leftover debug `print(instance.columns)` removed; `hightlight_cells` typo fixed.

## Verification

- Full pipeline run on the real config against a scratch tag, split 0, patient 40 (row 38):
  counterfactuals generate in ~5.5 min, and **every** artifact is written — including the two
  distance csvs that a fresh run had never produced. Sparsity of the instance row is now 0,
  the most-changed counts list only actionable features, and the guard reported
  `Removed 2 counterfactuals that changed features outside features_to_vary: ['SUBJ']`.
- `--patient_codes 40` resolves to row 38 and resumes from the patient-coded cache;
  `--patient_codes 999` exits 2 with the valid range.
- Global importance verified at `n_cpus=-1` (32 cores available, clamped to 3 chunks),
  `n_cpus=1`, and on a 12-instance set that must clamp to a single chunk.
- Timeout wrapper checked for normal return, error propagation, timeout, and an 8 MB result.
- The 1-hour preset was exercised for real: patient 74 (row 71) exhausted its budget, wrote
  `error.txt`, and the run continued cleanly to the next patient.

## Found but left unchanged

- **Permitted ranges are computed from the whole of `dfXy`, train and test together**, in both
  `get_global_permitted_range` and `get_local_permitted_range`. These constrain plausibility
  rather than model fitting, so this is defensible, but it is worth stating explicitly in the
  write-up since the per-split reports otherwise read as test-set-only. Noted in both
  docstrings.
- **`config.experiment.random_seed` is not used by this pipeline.** Reproducibility rests on
  `np.random.seed(s)` / `random.seed(s)` per repeat inside `generate_diverse_cfs`, which now
  run inside the timeout child process. (The value 42 *is* the seed that reproduces the stored
  models exactly, which is how the refit question above was settled.)
- **`--patient_codes` selects patients, but the mode names still say `skip_instances` /
  `redo_instances`.** Renaming them would break existing command lines and scripts.
- **Difficulty is not predicted by margin.** Patient 74 sits 0.0141 from the threshold and
  found nothing in an hour; patient 40, at 0.0143, finished in 5.5 minutes. Relevant when
  choosing `--gen-timeout` for a batch.
- **The metadata table in `plot_local_cf_heatmap2` hardcodes the full feature list** including
  NCS and Sudoscan columns. It renders `Xfull`, so it must be updated by hand if the dataset
  schema changes.
- **The notebooks (`cfreports.ipynb`, `counterfactuals.ipynb`) call these functions with the
  old signatures** and would need the added `patient_code` argument to run.

## Files changed

- [cfreports.py](cfreports.py)
- [utils2/counterfactuals.py](utils2/counterfactuals.py)
- [utils2/timeout.py](utils2/timeout.py)
