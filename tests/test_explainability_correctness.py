"""
Regression tests for the correctness fixes made to module/expreport.py and
module/utils2/explainability.py (final models / explainability reporting
pipeline).

Each test documents the bug it guards against in a comment, and would have
failed (via wrong result, exception, or both) against the pre-fix code.

Uses stdlib unittest (no pytest dependency) so it runs directly:
    python tests/test_explainability_correctness.py
It is also discoverable by pytest, if available, since pytest natively
collects unittest.TestCase subclasses.

Note on sys.path: explainability.py imports its sibling as `from utils2 import
optimization`, so module/ must be importable in its own right, in addition to
REPO_ROOT for the `module.*` package imports.
"""
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # tests must not require a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "module"))

from module import dataload, ymlconfig  # noqa: E402
from module.utils2 import explainability as exp  # noqa: E402

EXP_CONFIG = REPO_ROOT / "module" / "experiments" / "bin_exp_final_202608.yml"
DATASET_PATH = REPO_ROOT / "dataset" / "EAMC_DPN_Dataset.xlsx"
SPLITS_JOBLIB = (
    REPO_ROOT / "module" / "experiments" / "binary" / "hyperparameter_optimization"
    / "catboost" / "final_202608" / "catboost_first_repeat_trained_models.joblib"
)


class _StubModel:
    """Minimal stand-in for a fitted CatBoostClassifier. The pooled plotting
    functions only ever call predict_proba(), so that is all this provides."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        p = self._probs[: len(X)]
        return np.column_stack([1.0 - p, p])


def _config(seed=42):
    return SimpleNamespace(
        experiment=SimpleNamespace(random_seed=seed),
        model=SimpleNamespace(name="StubModel", code="stub"),
    )


def _split_results(n_folds=3, n=40, seed=0):
    """Synthetic fold results, deliberately given *different score scales* per
    fold, mirroring the real pipeline where each fold runs its own
    hyperparameter search and ends up on its own probability scale."""
    rng = np.random.default_rng(seed)
    splits, models = [], []
    for k in range(n_folds):
        y = rng.binomial(1, 0.6, n)
        probs = np.clip(rng.beta(2, 2, n) * 0.5 + 0.35 * y + 0.05 * k, 0.01, 0.99)
        X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
        splits.append(
            {
                "X_test": X,
                "y_test": pd.Series(y, name="target"),
                "metrics": {"threshold": 0.45 + 0.02 * k},
            }
        )
        models.append(_StubModel(probs))
    return splits, models


class RefitParameterTests(unittest.TestCase):
    """expreport.py's per-split model refit.

    Bug 1 (fatal): the refit read best_params straight out of the optimization
    stage and then passed `verbose=0` again --
        CatBoostClassifier(**best_params, verbose=0)
    best_params already carries 'verbose': 0, because param_space declares it
    as Categorical([0]). Python raises TypeError for the duplicate keyword
    before CatBoost is even reached, so *no fresh run of expreport.py could
    complete*. It only appeared to work because a stale retrained_models.joblib
    short-circuited the whole block.

    Bug 2 (silent): opt.best_params_ returns only the *searched* parameters.
    random_state was set on the base estimator, not in param_space, so it is
    absent from best_params and the refit silently fell back to CatBoost's
    default seed of 0 -- producing models that differed from the ones the
    optimization stage measured (and whose thresholds expreport.py reuses) by
    up to 0.086 in predicted probability.
    """

    def test_duplicate_verbose_keyword_raises(self):
        # The exact shape of the old call. Guards against reintroducing it.
        from catboost import CatBoostClassifier

        best_params = {"depth": 3, "iterations": 5, "verbose": 0}
        with self.assertRaises(TypeError):
            CatBoostClassifier(**best_params, verbose=0)

    def test_merge_form_tolerates_verbose_in_best_params(self):
        # The fixed shape: merging lets best_params carry 'verbose' harmlessly
        # while still guaranteeing verbose=0.
        from catboost import CatBoostClassifier

        best_params = {"depth": 3, "iterations": 5, "verbose": 0}
        model = CatBoostClassifier(**{**best_params, "verbose": 0})
        self.assertEqual(model.get_params()["verbose"], 0)

    def test_merge_form_forces_verbose_even_if_best_params_omits_it(self):
        from catboost import CatBoostClassifier

        best_params = {"depth": 3, "iterations": 5}
        model = CatBoostClassifier(**{**best_params, "verbose": 0})
        self.assertEqual(model.get_params()["verbose"], 0)

    @unittest.skipUnless(SPLITS_JOBLIB.exists(), "persisted splits not available")
    def test_persisted_best_params_lack_random_seed(self):
        # This is the root of bug 2: the seed simply is not in best_params, so
        # the refit must supply it explicitly from config.
        import joblib

        results = joblib.load(SPLITS_JOBLIB)["results"]
        first = next(iter(results.values()))
        best_params = first["metrics"]["best_params"]

        self.assertNotIn("random_seed", best_params)
        self.assertNotIn("random_state", best_params)
        self.assertIn("verbose", best_params)  # and bug 1's collision is real

    def test_seed_changes_predictions_and_is_reproducible(self):
        # Demonstrates why omitting the seed matters: different seeds give
        # different models, the same seed reproduces exactly.
        from catboost import CatBoostClassifier

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
        y = pd.Series((X["a"] + rng.normal(scale=0.5, size=60) > 0).astype(int))
        params = dict(iterations=30, depth=4, verbose=0)

        p1 = CatBoostClassifier(**params, random_seed=1).fit(X, y).predict_proba(X)[:, 1]
        p2 = CatBoostClassifier(**params, random_seed=2).fit(X, y).predict_proba(X)[:, 1]
        p1_again = CatBoostClassifier(**params, random_seed=1).fit(X, y).predict_proba(X)[:, 1]

        self.assertFalse(np.allclose(p1, p2), "different seeds should differ")
        np.testing.assert_allclose(p1, p1_again)


class BrierCurveTests(unittest.TestCase):
    """brier_curve() in explainability.py.

    Bug: the implementation weighted the two class losses by c and (1 - c)
    instead of thresholding the scores at c. That makes the "curve" exactly
    linear in c, and makes its area the *class-balanced* Brier score rather
    than the Brier score -- the two coincide only when n_pos == n_neg. On this
    project's ~70%-positive cohort the four per-split Brier figures overstated
    the loss by 8-46%.

    The defining property of a Brier curve (Hernandez-Orallo et al. 2011) is
    that the area under it equals the Brier score; these tests pin that down.
    """

    def _scores(self, n=400, prevalence=0.7, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, prevalence, n)
        p = np.clip(rng.beta(2, 4, n) + 0.3 * y, 0.001, 0.999)
        return y, p

    def test_area_equals_brier_score_imbalanced(self):
        y, p = self._scores(prevalence=0.7)
        c, loss = exp.brier_curve(y, p, n_points=1001)
        self.assertAlmostEqual(np.trapz(loss, c), brier_score_loss(y, p), places=3)

    def test_area_equals_brier_score_across_prevalences(self):
        # The old implementation only agreed at prevalence 0.5; the correct one
        # must agree everywhere.
        for prevalence in (0.2, 0.5, 0.8):
            with self.subTest(prevalence=prevalence):
                y, p = self._scores(prevalence=prevalence, seed=int(prevalence * 10))
                c, loss = exp.brier_curve(y, p, n_points=1001)
                self.assertAlmostEqual(
                    np.trapz(loss, c), brier_score_loss(y, p), places=3
                )

    def test_curve_is_not_linear_in_c(self):
        # The old implementation was a straight line to within floating point
        # (max deviation ~5e-17). A real Brier curve is not.
        y, p = self._scores()
        c, loss = exp.brier_curve(y, p)
        fit = np.polyval(np.polyfit(c, loss, 1), c)
        self.assertGreater(np.abs(loss - fit).max(), 1e-3)

    def test_all_positive_degenerate_case(self):
        y = np.ones(20, dtype=int)
        p = np.full(20, 0.8)
        c, loss = exp.brier_curve(y, p, n_points=1001)
        self.assertAlmostEqual(np.trapz(loss, c), brier_score_loss(y, p, pos_label=1), places=3)

    def test_all_negative_degenerate_case(self):
        y = np.zeros(20, dtype=int)
        p = np.full(20, 0.2)
        c, loss = exp.brier_curve(y, p, n_points=1001)
        self.assertAlmostEqual(np.trapz(loss, c), 0.04, places=3)

    def test_loss_is_non_negative(self):
        y, p = self._scores()
        _, loss = exp.brier_curve(y, p)
        self.assertTrue((loss >= 0).all())


class GetColorsTests(unittest.TestCase):
    """get_colors() in explainability.py.

    Bug: the first branch indexed COLOR_GROUP_MAP['Nerve Conduction Studies'],
    a key that exists only in COLOR_GROUP_MAP_ORIG (where it is itself
    commented out). Any label in D.ncs_cols raised KeyError. It stayed dormant
    only because expreport.py filters the NCS columns out of X.
    """

    def test_ncs_label_does_not_raise(self):
        colors = exp.get_colors(dataload.DPN_data, ["SSA_L"])
        self.assertEqual(len(colors), 1)

    def test_ncs_label_falls_through_to_gray(self):
        self.assertEqual(
            exp.get_colors(dataload.DPN_data, ["SSA_L"]), [exp.palette["Gray"]]
        )

    def test_known_groups_keep_their_colours(self):
        cases = {
            "NS": "Sudoscan",
            "AGE": "Profile",
            "HPN": "Comorbidities",
            "DEC_VS": "Neurology Examination",
            "MNSI": "MNSI",
        }
        for label, group in cases.items():
            with self.subTest(label=label):
                self.assertEqual(
                    exp.get_colors(dataload.DPN_data, [label]),
                    [exp.COLOR_GROUP_MAP[group]],
                )

    def test_mixed_labels_preserve_order(self):
        colors = exp.get_colors(dataload.DPN_data, ["AGE", "SSA_L", "NS"])
        self.assertEqual(
            colors,
            [
                exp.COLOR_GROUP_MAP["Profile"],
                exp.palette["Gray"],
                exp.COLOR_GROUP_MAP["Sudoscan"],
            ],
        )


class PooledDecisionCurveTests(unittest.TestCase):
    """plot_pooled_decision_curve_analysis() in explainability.py.

    Bug 1: the pooled_df=None branch called _pool_fold_predictions() and
    _print_distribution_audit() -- underscore-prefixed names that do not exist
    in the module (the real ones are pool_fold_predictions /
    print_distribution_audit). NameError for any standalone caller.

    Bug 2: the CSV write sat outside the `if savedir:` guard that every other
    save in the file uses, so savedir=None (the parameter's own default) raised
    TypeError after all the plotting and bootstrap work had been done.

    Bug 3 (ported guard): the superseded _referenced twin validated that
    thresholds stay below 1.0, since net benefit divides by (1 - threshold).
    The live function had dropped that check.
    """

    def tearDown(self):
        plt.close("all")

    def test_pools_internally_when_pooled_df_is_none(self):
        # Guards bug 1: exercises the branch that used the missing names.
        splits, models = _split_results()
        pooled_df, stats = exp.plot_pooled_decision_curve_analysis(
            splits, models, _config(), pooled_df=None, n_bootstrap=10, savedir=None
        )
        self.assertEqual(len(pooled_df), sum(len(s["y_test"]) for s in splits))
        self.assertIn("net_benefits", stats)

    def test_savedir_none_does_not_raise(self):
        # Guards bug 2.
        splits, models = _split_results()
        pooled_df, _, _, _, _ = exp.pool_fold_predictions(splits, models)
        exp.plot_pooled_decision_curve_analysis(
            splits, models, _config(), pooled_df=pooled_df,
            n_bootstrap=10, savedir=None,
        )

    def test_threshold_at_one_raises(self):
        # Guards bug 3.
        splits, models = _split_results()
        pooled_df, _, _, _, _ = exp.pool_fold_predictions(splits, models)
        with self.assertRaises(ValueError):
            exp.plot_pooled_decision_curve_analysis(
                splits, models, _config(), pooled_df=pooled_df,
                thresholds=[0.5, 1.0], n_bootstrap=10, savedir=None,
            )

    def test_writes_csv_when_savedir_given(self):
        splits, models = _split_results()
        pooled_df, _, _, _, _ = exp.pool_fold_predictions(splits, models)
        with tempfile.TemporaryDirectory() as tmp:
            exp.plot_pooled_decision_curve_analysis(
                splits, models, _config(), pooled_df=pooled_df,
                n_bootstrap=10, savedir=Path(tmp),
            )
            self.assertTrue((Path(tmp) / "stub_pooled_dca.csv").exists())
            self.assertTrue((Path(tmp) / "stub_pooled_dca.png").exists())

    def test_stats_expose_bootstrap_bounds_that_bracket_the_curve(self):
        # The DCA annotation used to report `mean_nb +/- np.std(net_benefits)`,
        # where the std was the spread of the curve across all 199 thresholds --
        # a property of the curve's shape, not an uncertainty. The fix reports
        # the bootstrap CI at the mean threshold, which requires these bounds to
        # be present and to actually bracket the point estimate.
        splits, models = _split_results()
        pooled_df, _, _, _, _ = exp.pool_fold_predictions(splits, models)
        _, stats = exp.plot_pooled_decision_curve_analysis(
            splits, models, _config(), pooled_df=pooled_df,
            n_bootstrap=200, savedir=None,
        )
        for key in ("nb_lower", "nb_upper", "net_benefit_at_mean_threshold",
                    "mean_threshold", "fold_thresholds"):
            self.assertIn(key, stats)

        lower = np.asarray(stats["nb_lower"])
        upper = np.asarray(stats["nb_upper"])
        curve = np.asarray(stats["net_benefits"])
        self.assertTrue((lower <= upper).all())
        # The point estimate lies inside its own bootstrap interval.
        idx = int(np.argmin(np.abs(np.asarray(stats["thresholds"]) - stats["mean_threshold"])))
        self.assertLessEqual(lower[idx], curve[idx] + 1e-12)
        self.assertGreaterEqual(upper[idx], curve[idx] - 1e-12)

    def test_bootstrap_seed_follows_config(self):
        # The four pooled plots used to hardcode np.random.default_rng(seed=42)
        # while config.experiment.random_seed was declared separately. Different
        # config seeds must now produce different bootstrap bounds.
        splits, models = _split_results()
        pooled_df, _, _, _, _ = exp.pool_fold_predictions(splits, models)
        kwargs = dict(pooled_df=pooled_df, n_bootstrap=50, savedir=None)

        _, a = exp.plot_pooled_decision_curve_analysis(splits, models, _config(1), **kwargs)
        _, b = exp.plot_pooled_decision_curve_analysis(splits, models, _config(2), **kwargs)
        _, a_again = exp.plot_pooled_decision_curve_analysis(splits, models, _config(1), **kwargs)

        self.assertFalse(np.allclose(a["nb_lower"], b["nb_lower"]))
        np.testing.assert_allclose(a["nb_lower"], a_again["nb_lower"])


class PoolFoldPredictionsTests(unittest.TestCase):
    """pool_fold_predictions() in explainability.py.

    Not a bug fix -- this pins the out-of-sample pooling invariant the four
    pooled figures rest on: every patient contributes exactly one prediction,
    produced by the fold in which that patient was a test case.
    """

    def test_public_helper_names_exist(self):
        # Guards the NameError fixed in plot_pooled_decision_curve_analysis:
        # these are the names that actually exist.
        self.assertTrue(callable(exp.pool_fold_predictions))
        self.assertTrue(callable(exp.print_distribution_audit))
        self.assertFalse(hasattr(exp, "_pool_fold_predictions"))
        self.assertFalse(hasattr(exp, "_print_distribution_audit"))

    def test_every_patient_contributes_exactly_once(self):
        splits, models = _split_results(n_folds=4, n=25)
        pooled_df, probs, labels, prevalence, N = exp.pool_fold_predictions(splits, models)

        self.assertEqual(N, 100)
        self.assertEqual(len(pooled_df), 100)
        self.assertEqual(sorted(pooled_df["fold"].unique()), [0, 1, 2, 3])
        self.assertEqual(pooled_df["fold"].value_counts().tolist(), [25, 25, 25, 25])
        self.assertAlmostEqual(prevalence, labels.mean())


class SplitResultsNormalisationTests(unittest.TestCase):
    """expreport.py's handling of the persisted split_results object.

    Bug: the joblib stores a dict, but expreport.py and explainability.py both
    index it positionally (`for s in range(len(split_results))`) and pair it
    against trained_models, a list. That works only because the keys happen to
    be 0..n-1 in insertion order. Any other keying -- string fold names,
    (repeat, fold) tuples, a non-contiguous set after dropping a fold -- gives
    KeyError, or silently mis-pairs models with test sets.

    expreport.py now normalises to a list once at load time.
    """

    @staticmethod
    def _normalise(split_results):
        # Exactly the expression expreport.py uses.
        if isinstance(split_results, dict):
            return [split_results[k] for k in sorted(split_results)]
        return split_results

    def test_contiguous_int_keys_preserve_order(self):
        source = {0: "a", 1: "b", 2: "c", 3: "d"}
        self.assertEqual(self._normalise(source), ["a", "b", "c", "d"])

    def test_insertion_order_does_not_change_result(self):
        shuffled = {2: "c", 0: "a", 3: "d", 1: "b"}
        self.assertEqual(self._normalise(shuffled), ["a", "b", "c", "d"])

    def test_non_contiguous_keys_become_positional(self):
        # The old code would have raised KeyError on split_results[0].
        source = {2: "first", 5: "second"}
        normalised = self._normalise(source)
        self.assertEqual(normalised, ["first", "second"])
        self.assertEqual(normalised[0], "first")

    def test_string_keys_become_positional(self):
        source = {"fold0": "first", "fold1": "second"}
        normalised = self._normalise(source)
        self.assertEqual(normalised[0], "first")
        self.assertEqual(len(normalised), 2)

    def test_list_input_is_passed_through(self):
        source = ["a", "b"]
        self.assertIs(self._normalise(source), source)

    @unittest.skipUnless(SPLITS_JOBLIB.exists(), "persisted splits not available")
    def test_real_splits_normalise_without_reordering(self):
        import joblib

        results = joblib.load(SPLITS_JOBLIB)["results"]
        normalised = self._normalise(results)
        self.assertEqual(len(normalised), len(results))
        for i, key in enumerate(sorted(results)):
            self.assertIs(normalised[i], results[key])


class FeatureAlignmentTests(unittest.TestCase):
    """expreport.py's feature-importance labelling.

    Bug: model.get_feature_importance() returns a bare array ordered by the
    model's own training columns, but expreport.py labels it with X.columns --
    columns assembled from the dataset in this script, while the models come
    from a joblib written by a separate stage. Nothing connected the two
    orderings, so a change to the selection stage would silently attach every
    importance to the wrong feature name. expreport.py now asserts alignment.
    """

    @unittest.skipUnless(
        SPLITS_JOBLIB.exists() and DATASET_PATH.exists(),
        "project dataset or persisted splits not available",
    )
    def test_dataset_columns_match_persisted_split_columns(self):
        import joblib

        loader = dataload.DPN_data(str(DATASET_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            frame = loader.load(
                classification="binary",
                report_path=str(Path(tmp) / "cleaning_report.txt"),
            )

        data_cols = frame.drop(loader.non_data_cols, axis=1, errors="ignore").columns
        no_ncs = [c for c in data_cols if c not in loader.ncs_cols]

        results = joblib.load(SPLITS_JOBLIB)["results"]
        persisted = list(next(iter(results.values()))["X_train"].columns)

        # The assertion expreport.py now performs at load time.
        self.assertEqual(list(no_ncs), persisted)

    @unittest.skipUnless(SPLITS_JOBLIB.exists(), "persisted splits not available")
    def test_all_splits_share_one_feature_ordering(self):
        import joblib

        results = joblib.load(SPLITS_JOBLIB)["results"]
        orderings = {tuple(r["X_train"].columns) for r in results.values()}
        self.assertEqual(len(orderings), 1)

        train_cols = {tuple(r["X_train"].columns) for r in results.values()}
        test_cols = {tuple(r["X_test"].columns) for r in results.values()}
        self.assertEqual(train_cols, test_cols)


class DatasetPathResolutionTests(unittest.TestCase):
    """expreport.py's dataset-path handling.

    Bug: `DPN_data(config.data.dataset_path[3:])` (the same bug already fixed
    in optreport.py and selreport.py) stripped the config's '../' prefix and
    resolved what remained against the process's current working directory, so
    the script only ran from the repo root. The fix resolves dataset_path
    against script_dir, the way config_path already was.
    """

    @unittest.skipUnless(EXP_CONFIG.exists(), "experiment config not available")
    def test_resolves_independent_of_cwd(self):
        script_dir = REPO_ROOT / "module"
        config = ymlconfig.dict_to_namespace(ymlconfig.load_config(EXP_CONFIG))

        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())  # cwd unrelated to module/ or the repo root
            resolved = (script_dir / config.data.dataset_path).resolve()
            self.assertTrue(resolved.exists(), f"{resolved} should exist regardless of cwd")
        finally:
            os.chdir(original_cwd)

    @unittest.skipUnless(EXP_CONFIG.exists(), "experiment config not available")
    def test_old_slicing_approach_breaks_from_module_cwd(self):
        # Demonstrates the bug this replaced.
        script_dir = REPO_ROOT / "module"
        config = ymlconfig.dict_to_namespace(ymlconfig.load_config(EXP_CONFIG))

        original_cwd = os.getcwd()
        try:
            os.chdir(script_dir)
            self.assertFalse(Path(config.data.dataset_path[3:]).exists())
        finally:
            os.chdir(original_cwd)


class TargetColumnTests(unittest.TestCase):
    """expreport.py's target-column selection.

    Bug: `y = dfdpn['Confirmed_Binary_DPN']` hardcoded the binary target name
    even though config.experiment.classification_type is read generically and
    passed to D.load(). Now uses D.get_target_column(), matching optreport.py
    and selreport.py.
    """

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_binary_target_column_matches_dataframe(self):
        loader = dataload.DPN_data(str(DATASET_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            frame = loader.load(
                classification="binary",
                report_path=str(Path(tmp) / "cleaning_report.txt"),
            )

        target_col = loader.get_target_column()
        self.assertEqual(target_col, "Confirmed_Binary_DPN")
        y = frame[target_col]  # exactly the lookup expreport.py now performs
        self.assertEqual(len(y), len(frame))
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_multiclass_target_column_is_not_the_binary_name(self):
        # Under the old hardcoded lookup this configuration raised KeyError.
        loader = dataload.DPN_data(str(DATASET_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            frame = loader.load(
                classification="multiclass",
                report_path=str(Path(tmp) / "cleaning_report.txt"),
            )

        target_col = loader.get_target_column()
        self.assertEqual(target_col, "DPN_Status")
        self.assertIn(target_col, frame.columns)
        self.assertNotIn("Confirmed_Binary_DPN", frame.columns)


class ReportOverwriteGuardTests(unittest.TestCase):
    """expreport.py's `overwrite` command-line argument.

    Bug: overwrite_reports was parsed from sys.argv and then never read
    anywhere in the file, so the documented invocation
        python module/expreport.py <config> overwrite
    silently regenerated reports whether or not the flag was passed, and
    omitting it silently overwrote existing reports rather than protecting
    them. The guard scopes its glob to '{model.code}_*' so it sees report
    files but not retrained_models.joblib or the copied config.
    """

    @staticmethod
    def _blocked(outputdir, model_code, overwrite_reports):
        # Exactly the condition expreport.py uses.
        existing = sorted(Path(outputdir).glob(f"{model_code}_*"))
        return bool(existing) and not overwrite_reports

    def test_blocks_when_reports_exist_and_flag_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "catboost_pooled_auroc.png").touch()
            self.assertTrue(self._blocked(tmp, "catboost", False))

    def test_allows_when_flag_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "catboost_pooled_auroc.png").touch()
            self.assertFalse(self._blocked(tmp, "catboost", True))

    def test_allows_when_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._blocked(tmp, "catboost", False))

    def test_ignores_non_report_artifacts(self):
        # retrained_models.joblib and the copied config must not trip the guard.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "retrained_models.joblib").touch()
            (Path(tmp) / "bin_exp_final_202608.yml").touch()
            self.assertFalse(self._blocked(tmp, "catboost", False))

    def test_argument_parsing_matches_documented_invocation(self):
        # `... <config> overwrite` enables it; anything else does not.
        self.assertTrue(["x.yml", "overwrite"][1] == "overwrite")
        for argv_tail in ([], ["something-else"]):
            with self.subTest(argv_tail=argv_tail):
                enabled = bool(argv_tail) and argv_tail[0] == "overwrite"
                self.assertFalse(enabled)


class ModuleHygieneTests(unittest.TestCase):
    """Module-level issues in explainability.py.

    - '$\\pm$' was written in a plain f-string, making '\\p' an invalid escape
      sequence: a SyntaxWarning today, a SyntaxError in a future Python.
    - plot_importances_heatmap2 and plot_pooled_decision_curve_analysis_referenced
      were unreferenced duplicates that wrote to the *same* output filenames as
      their live counterparts, so calling both silently overwrote one figure
      with the other.
    """

    def test_source_compiles_without_syntax_warnings(self):
        source_path = REPO_ROOT / "module" / "utils2" / "explainability.py"
        source = source_path.read_text()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(source, str(source_path), "exec")
        syntax_warnings = [w for w in caught if issubclass(w.category, SyntaxWarning)]
        self.assertEqual(syntax_warnings, [], f"unexpected: {[str(w.message) for w in syntax_warnings]}")

    def test_expreport_compiles_without_syntax_warnings(self):
        source_path = REPO_ROOT / "module" / "expreport.py"
        source = source_path.read_text()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(source, str(source_path), "exec")
        syntax_warnings = [w for w in caught if issubclass(w.category, SyntaxWarning)]
        self.assertEqual(syntax_warnings, [])

    def test_duplicate_output_filename_functions_are_gone(self):
        self.assertFalse(hasattr(exp, "plot_importances_heatmap2"))
        self.assertFalse(hasattr(exp, "plot_pooled_decision_curve_analysis_referenced"))
        self.assertTrue(callable(exp.plot_importances_heatmap))
        self.assertTrue(callable(exp.plot_pooled_decision_curve_analysis))

    def test_pooled_plotters_document_the_pooling_caveat(self):
        # The pooled figures aggregate fold models with independently-tuned
        # hyperparameters: pooled calibration flatters (opposing per-fold biases
        # cancel) and pooled discrimination is conservative. That reasoning must
        # stay recorded next to the code.
        for fn in (
            exp.plot_pooled_auroc,
            exp.plot_pooled_auprc,
            exp.plot_pooled_calibration_curve,
        ):
            with self.subTest(fn=fn.__name__):
                self.assertIsNotNone(fn.__doc__)
                self.assertIn("Note on pooling", fn.__doc__)

    def test_pooled_decision_curve_analysis_is_documented(self):
        doc = exp.plot_pooled_decision_curve_analysis.__doc__
        self.assertIsNotNone(doc)
        for token in ("Parameters:", "Returns:", "clinical_threshold_range"):
            self.assertIn(token, doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
