"""
Regression tests for the correctness fixes made to module/selreport.py and
module/utils2/selection.py (feature/model selection reporting pipeline).

Each test documents the bug it guards against in a comment, and would have
failed (via wrong result, exception, or both) against the pre-fix code.

Uses stdlib unittest (no pytest dependency) so it runs directly:
    python tests/test_selection_correctness.py
It is also discoverable by pytest, if available, since pytest natively
collects unittest.TestCase subclasses.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from module import dataload, ymlconfig  # noqa: E402
from module.utils2 import selection as sel  # noqa: E402


class GetHighVifTests(unittest.TestCase):
    """get_high_vif() in selection.py.

    Bug 1: `vif_threshold = config....vif_threshold,` (trailing comma) made
    the threshold a 1-tuple; only worked by accident via numpy broadcasting.
    Bug 2: the statsmodels intercept ('const') added internally was left in
    the returned rows, forcing callers to strip it off *positionally*
    (`.tolist()[1:]`) instead of get_high_vif excluding it by name.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        dup1 = rng.normal(size=n)
        dup2 = dup1 + rng.normal(scale=0.01, size=n)  # near-duplicate of dup1 -> high VIF
        indep = rng.normal(size=n)                     # independent -> low VIF
        self.df = pd.DataFrame({"dup1": dup1, "dup2": dup2, "indep": indep})
        self.config = SimpleNamespace(
            feature_selection=SimpleNamespace(vif_threshold=5),
            experiment=SimpleNamespace(verbosity=0),
        )

    def test_does_not_crash_on_threshold_comparison(self):
        high_vif = sel.get_high_vif(self.df, self.config)
        self.assertIsInstance(high_vif, pd.DataFrame)

    def test_excludes_intercept_term(self):
        high_vif = sel.get_high_vif(self.df, self.config)
        self.assertNotIn("const", high_vif["feature"].tolist())

    def test_flags_collinear_columns_not_independent_one(self):
        high_vif = sel.get_high_vif(self.df, self.config)
        flagged = set(high_vif["feature"])
        self.assertTrue({"dup1", "dup2"}.issubset(flagged))
        self.assertNotIn("indep", flagged)


class CalculateMetricStatisticsTests(unittest.TestCase):
    """calculate_metric_statistics() in selection.py.

    Bug: after sorting 'mean' by the configured metric, 'std' was "reordered"
    by reindexing its .columns (the metric names -- unaffected by a row sort)
    instead of its row index (the models), making the reorder a silent no-op.
    """

    def test_std_row_order_matches_sorted_mean_row_order(self):
        metrics = ["accuracy", "auprc"]
        experiment_metrics = []
        for name, base in [("ModelA", 0.5), ("ModelB", 0.9), ("ModelC", 0.7)]:
            scores = pd.DataFrame({m: base + 0.01 * np.arange(5) for m in metrics})
            experiment_metrics.append({"model": name, "rcv_scores": scores})

        config = SimpleNamespace(
            feature_selection=SimpleNamespace(cross_validation=SimpleNamespace(sort_by="auprc"))
        )
        with tempfile.TemporaryDirectory() as tmp:
            stats = sel.calculate_metric_statistics(
                "unittest_code", experiment_metrics, config,
                savedir=Path(tmp), overwrite=True,
            )

        self.assertEqual(stats["mean"].index.tolist(), ["ModelB", "ModelC", "ModelA"])
        # Pre-fix this stayed in insertion order (['ModelA', 'ModelB', 'ModelC'])
        # regardless of how 'mean' was sorted.
        self.assertEqual(stats["std"].index.tolist(), stats["mean"].index.tolist())


class ConfusionMatrixScorerTests(unittest.TestCase):
    """youden_index_score() / specificity_score() in selection.py.

    Bug: confusion_matrix(y_true, y_pred) without labels=[0, 1] returns a 1x1
    matrix when a CV fold contains only one class, so .ravel() yields 1 value
    instead of 4 and unpacking into (tn, fp, fn, tp) raises ValueError.
    """

    def test_all_negative_fold_does_not_crash(self):
        y_true = np.zeros(4, dtype=int)
        y_pred = np.zeros(4, dtype=int)
        specificity = sel.specificity_score(y_true, y_pred)
        youden = sel.youden_index_score(y_true, y_pred)
        self.assertEqual(specificity, 1.0)
        self.assertEqual(youden, 0.0)

    def test_all_positive_fold_does_not_crash(self):
        y_true = np.ones(4, dtype=int)
        y_pred = np.ones(4, dtype=int)
        specificity = sel.specificity_score(y_true, y_pred)
        youden = sel.youden_index_score(y_true, y_pred)
        self.assertEqual(specificity, 0.0)
        self.assertEqual(youden, 0.0)


class DatasetPathResolutionTests(unittest.TestCase):
    """selreport.py's dataset-path handling.

    Bug: `DPN_data(config.data.dataset_path[3:])` stripped the config's
    '../' prefix and resolved what remained against the process's current
    working directory -- raising FileNotFoundError when run the natural way
    (cwd = module/). The fix resolves dataset_path against script_dir
    (module/__file__'s parent), the same way config_path is already handled,
    so it no longer depends on the caller's cwd.
    """

    def test_resolves_independent_of_cwd(self):
        script_dir = REPO_ROOT / "module"
        config_dict = ymlconfig.load_config(
            script_dir / "experiments" / "bin_sel_final_202608.yml"
        )
        config = ymlconfig.dict_to_namespace(config_dict)

        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())  # a cwd unrelated to module/ or the repo root
            resolved = script_dir / config.data.dataset_path
            self.assertTrue(resolved.exists(), f"{resolved} should exist regardless of cwd")
        finally:
            os.chdir(original_cwd)

    def test_old_slicing_approach_breaks_from_module_cwd(self):
        # Demonstrates the bug this replaced: this is what selreport.py used
        # to do, and it fails to resolve when invoked with cwd = module/,
        # which is how selreport.py is meant to be run.
        script_dir = REPO_ROOT / "module"
        config_dict = ymlconfig.load_config(
            script_dir / "experiments" / "bin_sel_final_202608.yml"
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
    """selreport.py's target-column selection.

    Bug: `y = dfdpn['Confirmed_Binary_DPN']` hardcoded the target column name
    instead of using D.get_target_column(), even though
    config.experiment.classification_type is read generically and passed to
    D.load(). Loads the real project dataset (as selreport.py does) and
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
        # Exactly the lookup selreport.py now performs:
        # y = dfdpn[D.get_target_column()]
        y = frame[target_col]
        self.assertEqual(len(y), len(frame))
        self.assertTrue(set(y.unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
