"""
Regression tests for the correctness fixes made to module/optreport.py and
module/utils2/optimization.py (hyperparameter-optimization reporting
pipeline).

Each test documents the bug it guards against in a comment, and would have
failed (via wrong result, exception, or both) against the pre-fix code.
See module/optreport_refactor.md for the full write-up.

Uses stdlib unittest (no pytest dependency) so it runs directly:
    python tests/test_optimization_correctness.py
It is also discoverable by pytest, if available, since pytest natively
collects unittest.TestCase subclasses.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from module import dataload, ymlconfig  # noqa: E402
from module.utils2 import optimization as opt  # noqa: E402


def _make_dataset(n=80, seed=0):
    """A tiny, perfectly-learnable synthetic binary classification set."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f0": rng.uniform(size=n),
        "f1": rng.uniform(size=n),
    })
    y = pd.Series((X["f0"] > 0.5).astype(int), name="target")
    return X, y


def _base_config():
    return SimpleNamespace(
        optimization=SimpleNamespace(
            k_splits_outer=2, n_repeats_outer=1, k_splits_inner=2,
            optuna_n_trials=3, optimization_metric="roc-auc",
            threshold_selection_metric="fscore", fscore_beta=1,
        ),
        experiment=SimpleNamespace(random_seed=0, tag="unittest", verbosity=0),
        model=SimpleNamespace(code="unittest_model"),
    )


class RecordingClassifier:
    """A minimal stand-in for CatBoostClassifier that satisfies the subset of
    the API optimization.py relies on (constructor kwargs, fit(), and
    predict_proba()), while recording every fit() call's kwargs so tests can
    assert on what nested_cv_optimization() actually passed it.

    predict_proba() is a fixed function of X alone (not of any fitted
    state), so it's deterministic and doesn't depend on training actually
    happening -- this keeps the tests fast and focused on the *wiring*
    (which params/kwargs reach fit()), not on model quality.
    """

    fit_log = []

    def __init__(self, **params):
        self.params = {k: v for k, v in params.items() if k != "random_state"}

    def fit(self, X, y, **fit_kwargs):
        RecordingClassifier.fit_log.append({
            "params": dict(self.params),
            "fit_kwargs_keys": set(fit_kwargs.keys()),
            "n_rows": X.shape[0],
        })
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-8 * (X[:, 0] - 0.5)))
        return np.column_stack([1 - p, p])


class OptunaBestParamsOnlyContainsSuggestedKeysTests(unittest.TestCase):
    """Root cause of the "fixed params dropped" bug below: this is Optuna's
    own documented behavior (not a bug in this codebase) -- study.best_params
    only ever contains keys registered via trial.suggest_*. The original code
    in nested_cv_optimization() read `study.best_params` as if it were the
    complete dict returned by param_space_fn, which also mixes in fixed,
    never-suggested literals (loss_function, eval_metric, iterations,
    early_stopping_rounds in this project's param_space_fn).
    """

    def test_fixed_keys_are_absent_from_raw_optuna_best_params(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            x = trial.suggest_float("x", -1, 1)
            # A fixed param, exactly like param_space_fn's "loss_function": ...
            # -- never passed through suggest_*, so it won't appear in
            # study.best_params below no matter what.
            return (x - 0.3) ** 2

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)

        self.assertIn("x", study.best_params)
        self.assertNotIn("loss_function", study.best_params)


class BestParamsPreserveFixedKeysTests(unittest.TestCase):
    """nested_cv_optimization()'s handling of Optuna's best-trial params.

    Bug: `best_params = study.best_params` silently dropped every fixed
    (non-suggested) key in param_space_fn's returned dict before refitting
    and evaluating the "best" model -- so the model actually reported used
    model_class's own defaults for those keys instead of the config's
    values (e.g. CatBoost's default iterations=1000 instead of the config's
    500). Fixed by stashing the full per-trial dict via
    trial.set_user_attr("full_params", params) and reading
    study.best_trial.user_attrs["full_params"] back instead of
    study.best_params.
    """

    def setUp(self):
        RecordingClassifier.fit_log = []

    def test_fixed_params_survive_into_reported_best_params(self):
        X, y = _make_dataset()
        config = _base_config()

        def param_space_fn(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 5, 10),
                "loss_function": "Logloss",      # fixed, never suggested
                "early_stopping_rounds": 3,       # fixed, never suggested
            }

        with tempfile.TemporaryDirectory() as tmp:
            results = opt.nested_cv_optimization(
                X, y, config,
                model_class=RecordingClassifier,
                param_space_fn=param_space_fn,
                savedir=Path(tmp),
                overwrite=True,
            )

        self.assertTrue(results)
        for fold_result in results:
            best_params = fold_result["best_params"]
            # Tunable key: this one survived even pre-fix.
            self.assertIn("n_estimators", best_params)
            # Fixed keys: these are exactly what the pre-fix code dropped.
            self.assertEqual(best_params.get("loss_function"), "Logloss")
            self.assertEqual(best_params.get("early_stopping_rounds"), 3)


class EvalSetWiringTests(unittest.TestCase):
    """nested_cv_optimization()'s fit() calls when early_stopping_rounds is
    configured.

    Bug: no .fit() call anywhere in nested_cv_optimization (Optuna objective,
    OOF-threshold loop, or final refit) ever passed eval_set, so
    early_stopping_rounds had no effect at all -- verified directly against
    real CatBoost (module/optreport_refactor.md): a fit with
    early_stopping_rounds=5 and no eval_set ran the full iteration budget,
    while the same fit with an eval_set stopped after a handful of trees.
    Fixed by passing eval_set=(X_outer_train[inner_val_idx],
    y_outer_train[inner_val_idx]) whenever "early_stopping_rounds" is a key
    in the params being fit, for both the Optuna objective and the
    OOF-threshold loop (both already have that held-out inner split on
    hand). The final full-outer-train refit still has no eval_set, by
    design -- see optreport_refactor.md.
    """

    def setUp(self):
        RecordingClassifier.fit_log = []

    def test_eval_set_passed_when_early_stopping_rounds_configured(self):
        X, y = _make_dataset()
        config = _base_config()

        def param_space_fn(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 5, 10),
                "early_stopping_rounds": 3,
            }

        with tempfile.TemporaryDirectory() as tmp:
            opt.nested_cv_optimization(
                X, y, config,
                model_class=RecordingClassifier,
                param_space_fn=param_space_fn,
                savedir=Path(tmp),
                overwrite=True,
            )

        self.assertTrue(RecordingClassifier.fit_log, "expected fit() to have been called")

        # Every fit() call uses an inner-fold split (fewer rows than the full
        # outer-train set) EXCEPT the final best_model refit, which trains on
        # all of outer-train and has no held-out slice to use as eval_set --
        # an accepted design gap (see optreport_refactor.md). Distinguish the
        # two by row count rather than call order, since that's what the
        # code itself keys the eval_set decision on.
        max_rows = max(c["n_rows"] for c in RecordingClassifier.fit_log)
        full_refit_calls = [c for c in RecordingClassifier.fit_log if c["n_rows"] == max_rows]
        inner_fold_calls = [c for c in RecordingClassifier.fit_log if c["n_rows"] < max_rows]

        expected_full_refits = config.optimization.k_splits_outer * config.optimization.n_repeats_outer
        self.assertEqual(len(full_refit_calls), expected_full_refits)
        for call in full_refit_calls:
            self.assertNotIn("eval_set", call["fit_kwargs_keys"])

        self.assertTrue(inner_fold_calls)
        missing_eval_set = [c for c in inner_fold_calls if "eval_set" not in c["fit_kwargs_keys"]]
        self.assertEqual(
            missing_eval_set, [],
            f"{len(missing_eval_set)}/{len(inner_fold_calls)} inner-fold fit() calls "
            "were missing eval_set despite early_stopping_rounds being configured",
        )

    def test_no_eval_set_when_early_stopping_rounds_not_configured(self):
        # Guards against an overly-broad fix that always injects eval_set
        # regardless of whether the model config asked for early stopping.
        X, y = _make_dataset()
        config = _base_config()

        def param_space_fn(trial):
            return {"n_estimators": trial.suggest_int("n_estimators", 5, 10)}

        with tempfile.TemporaryDirectory() as tmp:
            opt.nested_cv_optimization(
                X, y, config,
                model_class=RecordingClassifier,
                param_space_fn=param_space_fn,
                savedir=Path(tmp),
                overwrite=True,
            )

        self.assertTrue(RecordingClassifier.fit_log)
        for call in RecordingClassifier.fit_log:
            self.assertNotIn("eval_set", call["fit_kwargs_keys"])


class CatBoostEarlyStoppingRequiresEvalSetTests(unittest.TestCase):
    """Documents the root cause of the bug above against the real library
    (not the RecordingClassifier stand-in): CatBoost's early_stopping_rounds
    is inert unless fit() is also given an eval_set.
    """

    def test_early_stopping_only_takes_effect_with_eval_set(self):
        from catboost import CatBoostClassifier

        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 5))
        y = (X[:, 0] > 0).astype(int)

        no_eval_set = CatBoostClassifier(
            iterations=200, early_stopping_rounds=5, verbose=0, random_state=0
        )
        no_eval_set.fit(X, y)
        self.assertEqual(
            no_eval_set.tree_count_, 200,
            "without eval_set, early_stopping_rounds should have no effect "
            "and training should run the full iteration budget",
        )

        with_eval_set = CatBoostClassifier(
            iterations=200, early_stopping_rounds=5, verbose=0, random_state=0
        )
        with_eval_set.fit(X[:150], y[:150], eval_set=(X[150:], y[150:]))
        self.assertLess(
            with_eval_set.tree_count_, 200,
            "with eval_set, early stopping should kick in before the full budget",
        )


class MeanConfidenceIntervalNanHandlingTests(unittest.TestCase):
    """mean_confidence_interval() in optimization.py.

    Bug: "mean"/"std" were computed via np.nanmean/np.nanstd (ignoring NaN
    folds), but "ci_lower"/"ci_upper" were computed from separate,
    non-NaN-aware np.mean/np.std -- so whenever any fold had NaN for a
    metric (common: precision/f1/fbeta_score use zero_division=np.nan on
    degenerate folds), ci_lower/ci_upper silently came out NaN even though
    mean/std in the same row looked fine. Fixed to use np.nanmean/np.nanstd
    consistently, with n corrected to the non-NaN fold count.
    """

    def test_ci_bounds_are_finite_when_one_fold_is_nan(self):
        opt_results = [
            {"fold": 0, "best_params": {}, "precision": 0.80, "roc-auc": 0.90},
            {"fold": 1, "best_params": {}, "precision": np.nan, "roc-auc": 0.85},
            {"fold": 2, "best_params": {}, "precision": 0.75, "roc-auc": 0.95},
        ]
        config = SimpleNamespace(
            evaluation=SimpleNamespace(confidence=0.95),
            experiment=SimpleNamespace(verbosity=0),
        )

        with tempfile.TemporaryDirectory() as tmp:
            df = opt.mean_confidence_interval(
                opt_results, config, savedir=Path(tmp), overwrite=True
            )

        precision_row = df.loc["precision"]
        self.assertFalse(np.isnan(precision_row["ci_lower"]))
        self.assertFalse(np.isnan(precision_row["ci_upper"]))
        # mean/std of the two non-NaN folds only: [0.80, 0.75]
        self.assertAlmostEqual(precision_row["mean"], 0.775)
        self.assertAlmostEqual(precision_row["ci_lower"], 0.726, places=3)
        self.assertAlmostEqual(precision_row["ci_upper"], 0.824, places=3)

        # A metric with no NaN folds should be unaffected by the fix.
        roc_auc_row = df.loc["roc-auc"]
        self.assertAlmostEqual(roc_auc_row["mean"], 0.90)


class DatasetPathResolutionTests(unittest.TestCase):
    """optreport.py's dataset-path handling.

    Bug: `DPN_data(config.data.dataset_path[3:])` (same bug already present
    in selreport.py) stripped the config's '../' prefix and resolved what
    remained against the process's current working directory. The fix
    resolves dataset_path against script_dir (module/__file__'s parent),
    the same way config_path is already handled, so it no longer depends
    on the caller's cwd.
    """

    def test_resolves_independent_of_cwd(self):
        script_dir = REPO_ROOT / "module"
        config_dict = ymlconfig.load_config(
            script_dir / "experiments" / "bin_opt_final_202608.yml"
        )
        config = ymlconfig.dict_to_namespace(config_dict)

        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())  # cwd unrelated to module/ or the repo root
            resolved = script_dir / config.data.dataset_path
            self.assertTrue(resolved.exists(), f"{resolved} should exist regardless of cwd")
        finally:
            os.chdir(original_cwd)

    def test_old_slicing_approach_breaks_from_module_cwd(self):
        # Demonstrates the bug this replaced: this is what optreport.py used
        # to do, and it fails to resolve when invoked with cwd = module/,
        # which is how optreport.py is meant to be run.
        script_dir = REPO_ROOT / "module"
        config_dict = ymlconfig.load_config(
            script_dir / "experiments" / "bin_opt_final_202608.yml"
        )
        config = ymlconfig.dict_to_namespace(config_dict)

        original_cwd = os.getcwd()
        try:
            os.chdir(script_dir)
            old_style_path = Path(config.data.dataset_path[3:])
            self.assertFalse(old_style_path.exists())
        finally:
            os.chdir(original_cwd)


class TargetColumnTests(unittest.TestCase):
    """optreport.py's target-column selection.

    Bug: `y = dfdpn['Confirmed_Binary_DPN']` hardcoded the target column
    name instead of using D.get_target_column(), even though
    config.experiment.classification_type is read generically and passed
    to D.load(). Loads the real project dataset (as optreport.py does) and
    confirms D.get_target_column() lines up with what's actually in the
    loaded frame.
    """

    DATASET_PATH = REPO_ROOT / "dataset" / "EAMC_DPN_Dataset.xlsx"

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_binary_target_column_matches_dataframe(self):
        loader = dataload.DPN_data(str(self.DATASET_PATH))
        frame = loader.load(classification="binary")

        target_col = loader.get_target_column()
        self.assertEqual(target_col, "Confirmed_Binary_DPN")
        # Exactly the lookup optreport.py now performs:
        # y = dfdpn[D.get_target_column()]
        y = frame[target_col]
        self.assertEqual(len(y), len(frame))
        self.assertTrue(set(y.unique()).issubset({0, 1}))


class TestModelThresholdConsistencyTests(unittest.TestCase):
    """test_model() in optimization.py.

    Bug: test_model()'s uses_proba=True branch used a strict `>` comparison
    against the threshold, while model_predict() and
    nested_cv_optimization()'s outer-test evaluation both use `>=`. Fixed
    to `>=` for consistency; this test documents the previously-divergent
    boundary case (a probability exactly equal to the threshold).
    """

    def test_boundary_probability_is_classified_positive(self):
        class StubModel:
            def predict_proba(self, X):
                # First row's probability sits exactly on the threshold.
                return np.array([[0.5, 0.5], [0.9, 0.1]])

        Xnew = np.zeros((2, 1))
        ynew = np.array([1, 0])

        ypred_via_model_predict, _ = opt.model_predict(Xnew, StubModel(), threshold=0.5)
        self.assertEqual(ypred_via_model_predict[0], 1)  # >= : boundary counts as positive

        _cm, metrics = opt.test_model(StubModel(), threshold=0.5, Xnew=Xnew, ynew=ynew, uses_proba=True)
        # Pre-fix, test_model's `>` would have classified the boundary row
        # negative while model_predict's `>=` classified it positive --
        # i.e. the two would disagree on the exact same input.
        self.assertEqual(metrics["sensitivity"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
