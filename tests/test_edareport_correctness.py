"""
Regression tests for module/edareport.py (the EDA reporting pipeline).

The statistics in this report go straight into a manuscript, so the tests
concentrate on the things that would be wrong silently: the discrimination
measure and its direction, the small-sample choices (Fisher's exact test,
Haldane-Anscombe correction, Wilson intervals), the modelling/excluded feature
split, and the provenance parser that turns the loader's audit trail into a
table. Layout is not tested; wrong numbers are.

Uses stdlib unittest (no pytest dependency) so it runs directly:
    python tests/test_edareport_correctness.py
It is also discoverable by pytest, which collects unittest.TestCase subclasses.

Note on sys.path: edareport.py imports its siblings as `from dataload import
DPN_data`, so module/ must be importable in its own right, in addition to
REPO_ROOT for the `module.*` package imports.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # tests must not require a display

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "module"))

from module import dataload  # noqa: E402
from module import edareport as eda  # noqa: E402

DATASET_PATH = REPO_ROOT / "dataset" / "EAMC_DPN_Dataset.xlsx"


class TestDiscrimination(unittest.TestCase):
    """The AUC is the effect size in Table 1, Table 2 and Figure 2."""

    def test_matches_sklearn_including_ties(self):
        # Binary predictors are all ties, so a rank implementation that mishandles
        # them would disagree with roc_auc_score exactly where it matters most.
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        for x in (rng.normal(size=200), rng.integers(0, 2, 200).astype(float),
                  np.round(rng.normal(size=200))):
            self.assertAlmostEqual(eda.auc_from_ranks(y, x), roc_auc_score(y, x), places=12)

    def test_direction_is_preserved(self):
        # Figure 2 reads values below 0.5 as "lower in the Confirmed group", so the
        # measure must not be folded to >= 0.5 anywhere in the pipeline.
        y = np.array([0] * 10 + [1] * 10)
        higher_in_cases = np.concatenate([np.zeros(10), np.ones(10)])
        self.assertEqual(eda.auc_from_ranks(y, higher_in_cases), 1.0)
        self.assertEqual(eda.auc_from_ranks(y, -higher_in_cases), 0.0)

    def test_bootstrap_ci_brackets_estimate_and_is_seeded(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        y = np.array([0] * 40 + [1] * 60)
        x = np.concatenate([np.random.default_rng(1).normal(0, 1, 40),
                            np.random.default_rng(2).normal(1, 1, 60)])
        lo_a, hi_a = eda.auc_bootstrap_ci(y, x, 300, rng_a)
        lo_b, hi_b = eda.auc_bootstrap_ci(y, x, 300, rng_b)
        self.assertEqual((lo_a, hi_a), (lo_b, hi_b))       # reproducible
        self.assertLess(lo_a, eda.auc_from_ranks(y, x))
        self.assertGreater(hi_a, eda.auc_from_ranks(y, x))


class TestSmallSampleHandling(unittest.TestCase):
    """GBS has 2 positives and PAOD has 8, so the sparse-table paths are live."""

    def test_fisher_is_used_when_an_expected_count_is_small(self):
        sparse = [[56, 1], [129, 1]]        # PAOD/GBS-like
        dense = [[25, 32], [40, 90]]
        self.assertEqual(eda.binary_test(sparse, 5)[1], "Fisher")
        self.assertEqual(eda.binary_test(dense, 5)[1], "chi-square")

    def test_odds_ratio_stays_finite_with_an_empty_cell(self):
        # Without the Haldane-Anscombe correction this is a division by zero and
        # the table cell becomes inf or nan.
        or_hat, lo, hi, corrected = eda.odds_ratio_ci([[57, 0], [120, 10]])
        self.assertTrue(corrected)
        for value in (or_hat, lo, hi):
            self.assertTrue(np.isfinite(value))
        self.assertLess(lo, or_hat)
        self.assertLess(or_hat, hi)

    def test_odds_ratio_is_uncorrected_when_all_cells_are_populated(self):
        or_hat, _, _, corrected = eda.odds_ratio_ci([[40, 17], [60, 70]])
        self.assertFalse(corrected)
        self.assertAlmostEqual(or_hat, (70 * 40) / (60 * 17))

    def test_wilson_interval_stays_within_bounds(self):
        for k, n in [(0, 57), (57, 57), (1, 57), (30, 130)]:
            lo, hi = eda.wilson_ci(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, k / n)
            self.assertGreaterEqual(hi, k / n)


class TestFeatureInventory(unittest.TestCase):
    """Which features may be modelled, and which may be varied, drives everything."""

    def setUp(self):
        self.D = dataload.DPN_data(str(DATASET_PATH))
        self.config = SimpleNamespace(
            analysis=SimpleNamespace(excluded_groups=["Nerve conduction study"]),
            cf_features=SimpleNamespace(actionable="INSULIN,HBA1C,HPN,PAOD,DSLPDMIA,CKD"))

    def test_ncs_is_excluded_and_nothing_else_is(self):
        inventory = eda.build_inventory(self.D, self.config)
        excluded = set(inventory.loc[~inventory.modelled, "feature"])
        self.assertEqual(excluded, set(self.D.ncs_cols))
        # The modelling set must match what cfreports.py actually trains on.
        self.assertEqual(int(inventory.modelled.sum()),
                         len(self.D.data_cols) - len(self.D.ncs_cols))

    def test_actionable_features_are_a_subset_of_the_modelled_features(self):
        # An actionable feature that is not modelled would mean the counterfactual
        # config and the EDA disagree about the feature space.
        inventory = eda.build_inventory(self.D, self.config)
        actionable = set(inventory.loc[inventory.actionable, "feature"])
        self.assertEqual(actionable, {"INSULIN", "HBA1C", "HPN", "PAOD", "DSLPDMIA", "CKD"})
        self.assertTrue(actionable.issubset(set(inventory.loc[inventory.modelled, "feature"])))

    def test_an_actionable_feature_outside_the_modelled_set_is_rejected(self):
        bad = SimpleNamespace(
            analysis=SimpleNamespace(excluded_groups=["Nerve conduction study"]),
            cf_features=SimpleNamespace(actionable="SSA_L"))
        with self.assertRaises(AssertionError):
            eda.build_inventory(self.D, bad)

    def test_every_feature_is_classified_exactly_once(self):
        inventory = eda.build_inventory(self.D, self.config)
        self.assertEqual(len(inventory), len(self.D.data_cols))
        self.assertEqual(len(set(inventory.feature)), len(inventory))


class TestFalseDiscoveryRateFamilies(unittest.TestCase):
    def test_q_is_computed_within_each_family(self):
        # Pooling the 22 candidate predictors with the 18 excluded nerve conduction
        # variables would deflate the q-values of the predictors, because the
        # excluded set is significant by construction and pushes every predictor
        # to a later rank. Hence Table 1 and Table 2 correct within each family.
        p = np.array([0.001, 0.02, 0.30, 0.60])
        q_alone = eda.benjamini_hochberg(p)
        q_pooled = eda.benjamini_hochberg(np.concatenate([p, np.full(18, 1e-12)]))[:4]
        self.assertTrue((q_alone >= q_pooled).all())
        self.assertTrue((q_alone > q_pooled).any())


class TestProvenanceParser(unittest.TestCase):
    """Table S1 is generated from the loader's audit trail, not typed by hand."""

    def _parse(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleaning_report.txt"
            path.write_text(text)
            return eda.parse_cleaning_report(path)

    def test_replacements_and_dropped_rows_are_separated(self):
        replacements, dropped = self._parse(
            "Value replacement report - 3 cells changed\n"
            "row  131  DM_DUR        '<1' -> '1'\n"
            "row  120  DM_DUR        '>10' -> '11'\n"
            "row    0  SSA_L         'NR' -> 0\n"
            "row   35  NaN in: NS, CAS\n")
        self.assertEqual(replacements[("DM_DUR", "'<1'", "'1'")], 1)
        self.assertEqual(replacements[("SSA_L", "'NR'", "0")], 1)
        # A dropped row must not also be counted as a replacement: the 'NaN in:'
        # line has no arrow, so a looser regex would silently mis-file it.
        self.assertNotIn(("NaN", "in:", "NS, CAS"), replacements)
        self.assertEqual(dropped, [(36, "NS, CAS")])

    def test_dropped_rows_are_reported_as_patient_codes(self):
        # dataload documents patient CODE as dataframe row + 1; the flow diagram
        # in Figure 1 names these patients, so an off-by-one is visible in print.
        _, dropped = self._parse("row   35  NaN in: NS\nrow  172  NaN in: NS\n")
        self.assertEqual([code for code, _ in dropped], [36, 173])

    def test_parser_agrees_with_a_real_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "cleaning_report.txt"
            loader = dataload.DPN_data(str(DATASET_PATH))
            df = loader.load(classification="binary", report_path=str(report))
            replacements, dropped = eda.parse_cleaning_report(report)
        self.assertEqual(len(df) + len(dropped), 190)
        dropped_codes = {code for code, _ in dropped}
        self.assertTrue(dropped_codes.isdisjoint(set(loader.patient_codes.tolist())))
        self.assertEqual(sum(replacements.values()), 392)


class TestFormatting(unittest.TestCase):
    def test_p_values_use_the_conventional_floor(self):
        self.assertEqual(eda.fmt_p(0.0004), r"$<$0.001")
        self.assertEqual(eda.fmt_p(0.0296), "0.030")
        self.assertEqual(eda.fmt_p(float("nan")), "--")

    def test_feature_codes_are_escaped_for_latex(self):
        # Unescaped underscores in codes such as DEC_AR are a LaTeX math-mode error.
        self.assertEqual(eda.latex_escape("DEC_AR"), r"DEC\_AR")
        self.assertEqual(eda.latex_escape("95% CI"), r"95\% CI")

    def test_integer_valued_features_are_not_given_false_precision(self):
        self.assertEqual(eda.decimals_for([1, 2, 3]), 0)
        self.assertEqual(eda.decimals_for([1.5, 2.0]), 1)
        self.assertEqual(eda.fmt_median_iqr(np.array([1, 2, 3, 4, 5]), 0), "3 [2--4]")


class TestStatisticsOnTheRealCohort(unittest.TestCase):
    """End-to-end check of the numbers the manuscript quotes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        loader = dataload.DPN_data(str(DATASET_PATH))
        cls.df = loader.load(classification="binary",
                             report_path=str(Path(cls.tmp.name) / "report.txt"))
        cls.target = loader.get_target_column()
        config = SimpleNamespace(
            analysis=SimpleNamespace(excluded_groups=["Nerve conduction study"],
                                     bootstrap_n=200, fisher_expected_min=5,
                                     vif_threshold=5),
            cf_features=SimpleNamespace(actionable="INSULIN,HBA1C,HPN,PAOD,DSLPDMIA,CKD"))
        inventory = eda.build_inventory(loader, config)
        cls.stats = eda.feature_statistics(cls.df, inventory, cls.target, config,
                                           np.random.default_rng(42))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cohort_shape(self):
        self.assertEqual(len(self.df), 187)
        self.assertEqual(int((self.df[self.target] == 1).sum()), 130)
        self.assertEqual(int((self.df[self.target] == 0).sum()), 57)

    def test_every_feature_has_a_finite_estimate(self):
        for column in ["auc", "auc_lo", "auc_hi", "p", "q"]:
            self.assertTrue(np.isfinite(self.stats[column]).all(),
                            f"non-finite value in {column}")

    def test_excluded_nerve_conduction_features_discriminate_far_better(self):
        # The premise of the leakage audit in Table 2: the excluded variables
        # separate the groups better than anything the models are allowed to use.
        modelled = self.stats[self.stats.modelled].strength.max()
        excluded = self.stats[~self.stats.modelled].strength.max()
        self.assertGreater(excluded, modelled)

    def test_non_recordable_counts_are_only_set_for_excluded_nerve_features(self):
        self.assertTrue(self.stats.loc[self.stats.modelled, "n_nonrecordable"].isna().all())
        self.assertTrue(self.stats.loc[~self.stats.modelled, "n_nonrecordable"].notna().all())

    def test_non_recordable_counts_match_the_stored_zeros(self):
        for row in self.stats[~self.stats.modelled].itertuples():
            self.assertEqual(int(row.n_nonrecordable), int((self.df[row.feature] == 0).sum()))
            self.assertEqual(int(row.n_nonrecordable),
                             int(row.n_nonrecordable_unconf) + int(row.n_nonrecordable_conf))


if __name__ == "__main__":
    unittest.main(verbosity=2)
