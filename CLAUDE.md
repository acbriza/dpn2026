# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A clinical ML research pipeline predicting diabetic peripheral neuropathy (DPN) from the
EAMC dataset (`dataset/EAMC_DPN_Dataset.xlsx`, 190 patients), with counterfactual
explanations (DiCE) for individual predictions. Built for a manuscript submission
(`module/legacy/202608/overleaf/main.tex` — note: often stale relative to the code;
re-derive study facts from the pipeline and its output reports, not from the manuscript).

## Environment setup

```
conda env create -f installation/dpncf.yml --name dpncf
conda activate dpncf
pip install numpy==1.26.4 scipy==1.13.0 pandas==2.2.2 scikit-learn==1.4.2 dice-ml==0.9 \
  xgboost matplotlib seaborn openpyxl lightgbm catboost shap
```
See `installation/environment_manual_install.txt` for exact pinned versions.

## Commands

Run tests from the repo root (they insert the repo root onto `sys.path` themselves):
```
pytest tests/
pytest tests/test_dataload.py::test_load_binary_default_drop_strategy   # single test
```
Tests load the real dataset (`dataset/EAMC_DPN_Dataset.xlsx`), not mocks/fixtures.
Some tests are intentionally `xfail(strict=True)` — they document a known, currently-unfixed
bug rather than a broken test; don't "fix" them without addressing the underlying bug.

Pipeline stages are run as scripts from inside `module/` (paths are resolved relative to
the script's own directory, so `cwd` doesn't matter, but conventionally run from `module/`).
Each reads a YAML config from `module/experiments/`:
```
python selreport.py bin_sel_final_202608.yml 16 overwrite     # <config> <n_cpus> [overwrite]
python optreport.py bin_opt_final_202608.yml overwrite        # <config> [overwrite]
python expreport.py bin_exp_final_202608.yml overwrite        # <config> [overwrite]
python cfreports.py bin_cf_final_202608.yml [skip_instances|redo_instances|global_only] --model-idx N --patient-codes 53,67
python edareport.py bin_eda_final_202608.yml
```
`module/configs/*.yml` holds older/dev configs (referenced by `.vscode/launch.json`);
`module/experiments/*.yml` holds the current ones actually read by the scripts above.

## Architecture

**Pipeline stages** (each stage's output feeds the next; each has a `<stage>report.py`
CLI driver plus a `utils2/<stage>.py` module with the actual implementation):

1. **selection** (`selreport.py` + `utils2/selection.py`) — benchmark candidate
   feature sets/models via repeated k-fold, pick a model.
2. **optimization** (`optreport.py` + `utils2/optimization.py`) — nested-CV
   hyperparameter search (Optuna) for the selected model, per fold, with
   AUPRC-based optimization and f-beta thresholding.
3. **explainability** (`expreport.py` + `utils2/explainability.py`) — retrain final
   per-fold models, produce feature importance / SHAP / ROC-AUC reports.
4. **counterfactuals** (`cfreports.py` + `utils2/counterfactuals.py`) — generate DiCE
   counterfactuals per patient per fold model, with sufficiency/necessity checks and
   per-patient/global report plots.
5. **eda** (`edareport.py`) — exploratory data analysis report; supersedes the older
   `eda.py`, which is stale and left in place only for reference.

Each report script sets its output directory as
`module/experiments/<classification_type>/<stage>/<model_code>/<tag>/` (from the config's
`experiment.classification_type/stage/tag` and `model.code`) and copies the config file
into it, so every run's config and outputs travel together.

**Data loading** (`module/dataload.py`, class `DPN_data`): the single source of truth for
reading and cleaning the raw Excel sheet — column groups (profile/comorbidity/neuro
exam/MNSI/nerve conduction studies/sudoscan), value normalization, NaN handling
(`drop` or `impute_mean`), and building the binary (`Confirmed_Binary_DPN`) or multiclass
(`DPN_Status`, ordinal 0-3) target from the raw Confirmed/Probable/Possible/Any_DPN
columns. `patient_codes` tracks the original 1-based spreadsheet CODE for each surviving
row after cleaning drops rows — never assume row index + 1 == patient code; use
`index_to_patient_code()`. **Nerve conduction study (NCS) columns are loaded but excluded
from every modeling/counterfactual stage** — a deliberate scope decision, enforced by an
assertion in `cfreports.py` that no NCS feature can ever be in `features_to_vary`.

**Config loading** (`module/ymlconfig.py`): YAML configs are loaded into nested
`SimpleNamespace` objects (`config.experiment.classification_type`, `config.model.code`,
etc.) rather than dicts.

**`module/legacy/`**: older superseded notebooks/scripts kept for reference only. Don't
build on or extend these. **`module/legacy/202608/`**: the original per-stage notebooks
(`selection.ipynb`, `hparam_opt.ipynb`, `explainability.ipynb`, `counterfactuals.ipynb`,
`cfreports.ipynb`) that the current `*report.py` scripts were derived from/replaced, plus
the manuscript's Overleaf project (`module/legacy/202608/overleaf/`). Also reference-only.

**`*_refactor.md` files next to each stage's source** (`dataload_refactor.md`,
`selreport_refactor.md`, `optreport_refactor.md`, `expreport_refactor.md`): session
records of correctness bugs found and fixed in that stage, plus bugs found but
*intentionally* left unfixed (with the reason). Read the relevant one before making
non-trivial changes to a stage — it documents non-obvious invariants (e.g. why a model is
refit a certain way, why a threshold must come from a specific stored model) that aren't
otherwise written down anywhere.
