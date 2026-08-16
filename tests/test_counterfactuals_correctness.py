"""
Regression tests for the correctness fixes made to module/cfreports.py and
module/utils2/counterfactuals.py (the counterfactual reporting pipeline).

Each test documents the bug it guards against, and would have failed (via wrong
result, exception, or both) against the pre-fix code.

The heaviest emphasis is on patient identity. `qidx` is a row of the *cleaned*
dataframe; `patient_code` is the patient's ID in the source spreadsheet. Cleaning
drops rows with NaN numeric values and resets the index, so the two numbers
diverge, and `qidx + 1` -- which the reports used to print -- is only correct up
to the first dropped row. Every folder, csv and figure is now keyed by the code,
while the in-memory frames stay indexed by the row, so the conversion between
them has to be exactly right in both directions.

Uses stdlib unittest (no pytest dependency) so it runs directly:
    python tests/test_counterfactuals_correctness.py
It is also discoverable by pytest, which collects unittest.TestCase subclasses.

Note on sys.path: counterfactuals.py imports its sibling as `from utils2 import
explainability`, so module/ must be importable in its own right, in addition to
REPO_ROOT for the `module.*` package imports.
"""
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # tests must not require a display

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "module"))

from module import dataload  # noqa: E402
from module.utils2 import counterfactuals as cf  # noqa: E402
from module.utils2 import timeout as timeout_mod  # noqa: E402

DATASET_PATH = REPO_ROOT / "dataset" / "EAMC_DPN_Dataset.xlsx"
CF_CONFIG = REPO_ROOT / "module" / "experiments" / "bin_cf_final_202608.yml"
CFREPORTS = REPO_ROOT / "module" / "cfreports.py"

# Rows dropped by _clean_raw_values for this dataset, from dataset/cleaning_report.txt.
# They are what makes patient_code != qidx + 1.
DROPPED_RAW_ROWS = (35, 45, 172)

warnings.filterwarnings("ignore")


def _config(nzfill=3, verbosity=0, model_code="catboost"):
    """Minimal stand-in for the parsed yml config, with the fields these functions read."""
    return SimpleNamespace(
        model=SimpleNamespace(code=model_code, name="CatBoost"),
        experiment=SimpleNamespace(verbosity=verbosity),
        reporting=SimpleNamespace(
            nzfill=nzfill,
            cf_heatmap=SimpleNamespace(save_every=15,
                                       figsize=SimpleNamespace(x=15, y=6.5)),
        ),
        dice=SimpleNamespace(
            cf_features=SimpleNamespace(
                actionable="INSULIN,HBA1C", unactionable="SEX,AGE", progressive="none"),
            local_cf=SimpleNamespace(
                nrepeats=1, total_CFs=3, posthoc_sparsity_algorithm="binary",
                posthoc_sparsity_param=0.05, proximity_weight=0.5, diversity_weight=1.0,
                categorical_penalty=0.1, algorithm="DiverseCF"),
            global_cf=SimpleNamespace(total_CFs=20, posthoc_sparsity_algorithm="binary"),
        ),
    )


class _StubExplainer:
    """Stand-in for a dice_ml.Dice explainer.

    Returns counterfactuals shaped the way DiCE really returns them: categorical
    features as strings, and the outcome column present even though the query
    instance does not carry it.
    """

    def __init__(self, cfs_df):
        self._cfs = cfs_df

    def generate_counterfactuals(self, *args, **kwargs):
        example = SimpleNamespace(final_cfs_df=self._cfs.copy())
        return SimpleNamespace(cf_examples_list=[example])


# ---------------------------------------------------------------------------
# qidx <-> patient_code
# ---------------------------------------------------------------------------

class QidxPatientCodeTests(unittest.TestCase):
    """The reports used to label patients `qidx + 1`, which is wrong for every row
    after the first dropped one, and the two heatmaps disagreed (one used qidx, the
    other qidx + 1) so a single patient appeared under two different numbers."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            return
        cls.D = dataload.DPN_data(str(DATASET_PATH))
        cls.D.load(classification="binary")

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_patient_code_is_not_row_plus_one_after_a_dropped_row(self):
        # The exact bug: qidx + 1 held only until the first dropped row.
        first_drop = DROPPED_RAW_ROWS[0]
        self.assertEqual(self.D.index_to_patient_code(first_drop - 1), first_drop)
        # from here on, qidx + 1 understates the code
        for row in (first_drop, first_drop + 5, len(self.D.df) - 1):
            self.assertNotEqual(
                self.D.index_to_patient_code(row), row + 1,
                f"row {row}: qidx + 1 must not be treated as the patient code")

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_code_is_the_row_th_surviving_raw_row(self):
        # Derived independently of the loader: the nth cleaned row is the nth raw row
        # that was not dropped, and codes are 1-based.
        surviving = [r for r in range(190) if r not in DROPPED_RAW_ROWS]
        self.assertEqual(len(surviving), len(self.D.df))
        for row in range(len(self.D.df)):
            self.assertEqual(int(self.D.index_to_patient_code(row)), surviving[row] + 1,
                             f"row {row} maps to the wrong patient code")

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_offset_grows_at_each_dropped_row(self):
        # The gap between code and row widens by one as each dropped row is passed,
        # which is why a single fixed offset (qidx + 1) cannot be right for all rows.
        offsets = {row: int(self.D.index_to_patient_code(row)) - row
                   for row in range(len(self.D.df))}
        self.assertEqual(offsets[0], 1)
        self.assertEqual(offsets[DROPPED_RAW_ROWS[0]], 2)
        self.assertEqual(offsets[len(self.D.df) - 1], 1 + len(DROPPED_RAW_ROWS))
        self.assertTrue(all(offsets[r] <= offsets[r + 1] for r in range(len(offsets) - 1)),
                        "the offset must never shrink as rows advance")

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_round_trip_code_to_row_to_code_is_identity(self):
        # cfreports.py builds this mapping to translate --patient_codes into rows.
        row_of_code = {int(code): row for row, code in enumerate(self.D.patient_codes)}
        self.assertEqual(len(row_of_code), len(self.D.patient_codes),
                         "patient codes must be unique, or the CLI mapping loses rows")
        for row in range(len(self.D.df)):
            code = int(self.D.index_to_patient_code(row))
            self.assertEqual(row_of_code[code], row,
                             f"code {code} must map back to row {row}")

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_dropped_rows_have_no_patient_code(self):
        # A dropped patient must not be reachable: their code is absent from the mapping.
        codes = {int(c) for c in self.D.patient_codes}
        for raw_row in DROPPED_RAW_ROWS:
            self.assertNotIn(raw_row + 1, codes,
                             f"patient {raw_row + 1} was dropped and must not resolve")

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_cleaned_index_is_contiguous(self):
        # Positional and label lookups agree only while this holds; generate_local_cf_reports
        # selects the query instance with .loc, and the report table reads Xfull with .loc.
        self.assertTrue(self.D.df.index.equals(pd.RangeIndex(len(self.D.df))))

    @unittest.skipUnless(DATASET_PATH.exists(), "project dataset not available")
    def test_known_mapping_examples(self):
        # Spot values used throughout the review, so a change in cleaning is noticed here.
        for row, code in [(0, 1), (34, 35), (35, 37), (38, 40), (71, 74), (96, 99)]:
            self.assertEqual(int(self.D.index_to_patient_code(row)), code,
                             f"row {row} should be patient {code}")


class InstanceArtifactFilenameTests(unittest.TestCase):
    """Per-patient artifacts are named for the patient code, zero-padded with
    config.reporting.nzfill, so csvs and figures agree with their folder."""

    def test_filename_carries_padded_patient_code(self):
        name = cf.instance_artifact_filename(_config(nzfill=3), 0, 40, "local_cf.csv")
        self.assertEqual(name, "catboost_split0_patient040_local_cf.csv")

    def test_padding_follows_config(self):
        name = cf.instance_artifact_filename(_config(nzfill=5), 2, 40, "local_cf.csv")
        self.assertEqual(name, "catboost_split2_patient00040_local_cf.csv")

    def test_filename_never_embeds_the_dataframe_row(self):
        # Guards the old naming, which used qidx (or qidx + 1) and so pointed at
        # the wrong patient once rows had been dropped.
        qidx, patient_code = 38, 40
        name = cf.instance_artifact_filename(_config(), 0, patient_code, "local_cf.csv")
        self.assertIn("040", name)
        self.assertNotIn(f"patient{qidx:03d}", name)
        self.assertNotIn(f"patient{qidx + 1:03d}", name)

    def test_omitting_the_code_falls_back_to_the_unlabelled_form(self):
        name = cf.instance_artifact_filename(_config(), 1, None, "local_cf.csv")
        self.assertEqual(name, "catboost_split1_local_cf.csv")


class ReportFolderNamingTests(unittest.TestCase):
    """generate_local_cf_reports writes into a folder named for the patient code.
    It used to use str(qidx).zfill(3)."""

    def _run(self, tmp, qidx, patient_code):
        config = _config()
        features = ["INSULIN", "HBA1C"]
        instance = pd.DataFrame({"SEX": [0], "INSULIN": [1.0], "HBA1C": [7.5]}, index=[qidx])
        dfXy = pd.concat([instance, pd.Series([0], index=[qidx],
                                              name="Confirmed_Binary_DPN")], axis=1)
        cfs = pd.DataFrame({"SEX": ["0", "0"], "INSULIN": ["1.0", "0.0"],
                            "HBA1C": ["6.5", "7.0"], "Confirmed_Binary_DPN": ["0", "0"]})
        ioi_df = pd.DataFrame({"pred": [1], "actual": [1], "pred_proba": [0.51],
                               "margin": [0.01]}, index=[qidx])
        savedir = Path(tmp)
        cf.generate_local_cf_reports(
            dfXy, _StubExplainer(cfs), ioi_df, qidx, instance,
            patient_code=patient_code,
            features_to_vary=features, config=config, split_index=0, threshold=0.5,
            categorical_cols=["SEX"], continuous_cols=["INSULIN", "HBA1C"],
            replot=False,           # skip figures: they need the full DPN feature table
            savedir=savedir)
        return savedir

    def test_folder_is_named_for_the_patient_not_the_row(self):
        qidx, patient_code = 38, 40
        with tempfile.TemporaryDirectory() as tmp:
            savedir = self._run(tmp, qidx, patient_code)
            self.assertTrue((savedir / "nofiltering" / "040").is_dir(),
                            "outputs must land in a folder named for the patient code")
            self.assertFalse((savedir / "nofiltering" / "038").exists(),
                             "the dataframe row must not name a folder")
            self.assertFalse((savedir / "nofiltering" / "039").exists(),
                             "qidx + 1 must not name a folder either")

    def test_every_artifact_in_the_folder_carries_the_patient_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            savedir = self._run(tmp, 38, 40)
            written = sorted(p.name for p in (savedir / "nofiltering" / "040").iterdir())
            self.assertTrue(written, "expected per-patient artifacts to be written")
            for name in written:
                self.assertIn("patient040", name, f"{name} is not keyed by patient code")

    def test_no_stray_nofiltering_folder_when_progressive_configured(self):
        # The directory used to be created and then reassigned, leaving an empty tree.
        config = _config()
        config.dice.cf_features.progressive = "DEC_VS"
        qidx, patient_code = 38, 40
        instance = pd.DataFrame({"SEX": [0], "INSULIN": [1.0], "HBA1C": [7.5],
                                 "DEC_VS": [1]}, index=[qidx])
        dfXy = pd.concat([instance, pd.Series([0], index=[qidx],
                                              name="Confirmed_Binary_DPN")], axis=1)
        cfs = pd.DataFrame({"SEX": ["0"], "INSULIN": ["0.0"], "HBA1C": ["6.5"],
                            "DEC_VS": ["1"], "Confirmed_Binary_DPN": ["0"]})
        ioi_df = pd.DataFrame({"pred": [1], "actual": [1], "pred_proba": [0.51],
                               "margin": [0.01]}, index=[qidx])
        with tempfile.TemporaryDirectory() as tmp:
            savedir = Path(tmp)
            cf.generate_local_cf_reports(
                dfXy, _StubExplainer(cfs), ioi_df, qidx, instance,
                patient_code=patient_code, features_to_vary=["INSULIN", "HBA1C"],
                config=config, split_index=0, threshold=0.5,
                categorical_cols=["SEX", "DEC_VS"], continuous_cols=["INSULIN", "HBA1C"],
                replot=False, savedir=savedir)
            self.assertTrue((savedir / "unfiltered" / "040").is_dir())
            self.assertFalse((savedir / "nofiltering").exists(),
                             "'nofiltering' must not be created on a progressive run")


class InstancesOfInterestTests(unittest.TestCase):
    """The saved csv is keyed by patient code, while the returned frame stays indexed
    by dataframe row because that is what the rest of the pipeline looks up."""

    def _ioi(self, savedir=None, patient_codes=None):
        X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=[0, 1, 2, 3])
        y_test = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])

        class _M:
            def predict(self, X):
                return np.array([0, 1, 0, 1])

            def predict_proba(self, X):
                p = np.array([0.10, 0.52, 0.49, 0.95])
                return np.column_stack([1 - p, p])

        return cf.get_instances_of_interest(
            _M(), X_test, y_test, _config(), 0, threshold=0.5, delta=0.05,
            patient_codes=patient_codes, savedir=savedir)

    def test_returned_frame_is_indexed_by_dataframe_row(self):
        ioi_df, _ = self._ioi(patient_codes=np.array([1, 2, 4, 5]))
        for label in ioi_df.index:
            self.assertIn(label, [0, 1, 2, 3])

    def test_saved_csv_is_keyed_by_patient_code_and_keeps_the_row(self):
        codes = np.array([1, 2, 4, 5])   # a dropped row between 2 and 4
        with tempfile.TemporaryDirectory() as tmp:
            ioi_df, _ = self._ioi(savedir=Path(tmp), patient_codes=codes)
            written = pd.read_csv(Path(tmp) / "catboost_split0_instances_of_interest.csv",
                                  index_col=0)
            self.assertEqual(written.index.name, "patient_code")
            self.assertIn("qidx", written.columns)
            for code, row in zip(written.index, written["qidx"]):
                self.assertEqual(int(code), int(codes[row]),
                                 "each saved row must pair the code with its own row")
            self.assertListEqual(list(written["qidx"]), list(ioi_df.index))

    def test_without_patient_codes_the_csv_keeps_the_old_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._ioi(savedir=Path(tmp))
            written = pd.read_csv(Path(tmp) / "catboost_split0_instances_of_interest.csv",
                                  index_col=0)
            self.assertNotIn("qidx", written.columns)


class CliPatientCodeTests(unittest.TestCase):
    """--patient_codes takes spreadsheet codes and translates them to dataframe rows."""

    @unittest.skipUnless(CF_CONFIG.exists() and DATASET_PATH.exists() and CFREPORTS.exists(),
                         "cfreports.py, its config or the dataset not available")
    def test_unknown_patient_code_is_rejected(self):
        # An unknown code used to silently match no instance and do nothing.
        proc = subprocess.run(
            [sys.executable, str(CFREPORTS), CF_CONFIG.name,
             "redo_instances", "--model-idx", "0", "--patient_codes", "9999"],
            cwd=str(REPO_ROOT / "module"), capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown patient code", proc.stderr.lower())

    @unittest.skipUnless(CFREPORTS.exists(), "cfreports.py not available")
    def test_instances_flag_is_gone(self):
        # Renamed to --patient_codes; the old flag took dataframe rows.
        proc = subprocess.run([sys.executable, str(CFREPORTS), "--help"],
                              cwd=str(REPO_ROOT / "module"),
                              capture_output=True, text=True, timeout=300)
        self.assertIn("--patient_codes", proc.stdout)
        self.assertNotIn("--instances ", proc.stdout)


# ---------------------------------------------------------------------------
# DiCE output normalisation
# ---------------------------------------------------------------------------

class NumericConversionTests(unittest.TestCase):
    """DiCE returns categorical features as strings. Left mixed, '0' != 0 marked every
    categorical as changed and the distance subtraction raised TypeError."""

    def test_numeric_text_columns_become_numeric(self):
        df = pd.DataFrame({"SEX": [0, "0", "1"], "HBA1C": [7.5, "7.0", "6.1"]})
        out = cf._to_numeric_where_possible(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(out["SEX"]))
        self.assertEqual(out["SEX"].iloc[0], out["SEX"].iloc[1])

    def test_existing_nans_are_preserved_and_do_not_block_conversion(self):
        df = pd.DataFrame({"outcome": [np.nan, "0", "1"]})
        out = cf._to_numeric_where_possible(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(out["outcome"]))
        self.assertTrue(np.isnan(out["outcome"].iloc[0]))

    def test_genuinely_non_numeric_columns_are_left_alone(self):
        df = pd.DataFrame({"note": ["a", "b", "c"]})
        out = cf._to_numeric_where_possible(df)
        self.assertListEqual(list(out["note"]), ["a", "b", "c"])

    def test_subtraction_works_after_conversion(self):
        df = cf._to_numeric_where_possible(
            pd.DataFrame({"HBA1C": [7.5, "7.0"]}))
        diff = df["HBA1C"].iloc[1] - df["HBA1C"].iloc[0]
        self.assertAlmostEqual(diff, -0.5)


class FeaturesToVaryGuardTests(unittest.TestCase):
    """DiCE occasionally returns a counterfactual that moved a feature it was told to
    hold fixed. Such a counterfactual is not actionable and is dropped."""

    def _frame(self):
        # row 0 is the query instance; rows 2 and 4 move SUBJ, which is not varied
        return pd.DataFrame({
            "SUBJ":    [1, 1, 0, 1, 0],
            "HBA1C":   [7.5, 7.0, 6.5, 6.1, 5.9],
            "Confirmed_Binary_DPN": [np.nan, 0, 0, 0, 0],
        })

    def test_violating_counterfactuals_are_removed(self):
        df = self._frame()
        instance = df.iloc[[0]].drop(columns=["Confirmed_Binary_DPN"])
        kept = cf.drop_cfs_outside_features_to_vary(df, instance, ["HBA1C"])
        self.assertEqual(len(kept), 3)                     # instance + 2 valid CFs
        self.assertTrue((kept["SUBJ"] == 1).all())

    def test_query_instance_row_is_always_kept_first(self):
        df = self._frame()
        instance = df.iloc[[0]].drop(columns=["Confirmed_Binary_DPN"])
        kept = cf.drop_cfs_outside_features_to_vary(df, instance, ["HBA1C"])
        self.assertEqual(kept.iloc[0]["HBA1C"], df.iloc[0]["HBA1C"])

    def test_dropped_rows_are_saved_for_inspection(self):
        df = self._frame()
        instance = df.iloc[[0]].drop(columns=["Confirmed_Binary_DPN"])
        with tempfile.TemporaryDirectory() as tmp:
            cf.drop_cfs_outside_features_to_vary(
                df, instance, ["HBA1C"], config=_config(), split_index=0,
                patient_code=40, savedir=Path(tmp))
            expected = Path(tmp) / "catboost_split0_patient040_local_cf_unactionable.csv"
            self.assertTrue(expected.exists())
            self.assertEqual(len(pd.read_csv(expected)), 2)

    def test_is_a_no_op_when_every_counterfactual_is_valid(self):
        df = self._frame()
        df = df[df["SUBJ"] == 1].reset_index(drop=True)
        instance = df.iloc[[0]].drop(columns=["Confirmed_Binary_DPN"])
        kept = cf.drop_cfs_outside_features_to_vary(df, instance, ["HBA1C"])
        self.assertEqual(len(kept), len(df))

    def test_is_idempotent(self):
        df = self._frame()
        instance = df.iloc[[0]].drop(columns=["Confirmed_Binary_DPN"])
        once = cf.drop_cfs_outside_features_to_vary(df, instance, ["HBA1C"])
        twice = cf.drop_cfs_outside_features_to_vary(once, instance, ["HBA1C"])
        self.assertEqual(len(once), len(twice))

    def test_outcome_column_is_not_treated_as_a_feature(self):
        # It is absent from the instance, so comparing it would flag every row.
        df = self._frame()
        instance = df.iloc[[0]].drop(columns=["Confirmed_Binary_DPN"])
        kept = cf.drop_cfs_outside_features_to_vary(df, instance, ["HBA1C", "SUBJ"])
        self.assertEqual(len(kept), len(df))


# ---------------------------------------------------------------------------
# per-counterfactual analyses
# ---------------------------------------------------------------------------

class MostChangedFeatureTests(unittest.TestCase):
    """The csv used to carry a ",0" header, and ranked the outcome column first."""

    def _frames(self):
        instance = pd.DataFrame({"A": [1.0], "B": [2.0]})
        df_cf = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 2.0],
                              "Confirmed_Binary_DPN": [0, 1]})
        return instance, df_cf

    def test_result_is_a_two_column_frame(self):
        instance, df_cf = self._frames()
        out = cf.get_most_changed_feature(df_cf, instance, _config(), 0, savedir=None)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertListEqual(list(out.columns), ["feature", "change count"])

    def test_outcome_column_is_excluded(self):
        instance, df_cf = self._frames()
        out = cf.get_most_changed_feature(df_cf, instance, _config(), 0, savedir=None)
        self.assertNotIn("Confirmed_Binary_DPN", list(out["feature"]))

    def test_saved_csv_has_the_named_header(self):
        instance, df_cf = self._frames()
        with tempfile.TemporaryDirectory() as tmp:
            cf.get_most_changed_feature(df_cf, instance, _config(), 0,
                                        savedir=Path(tmp), patient_code=40)
            path = Path(tmp) / "catboost_split0_patient040_local_cf_most_changed.csv"
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text().splitlines()[0], "feature,change count")

    def test_unvaried_categorical_is_not_reported_as_changed(self):
        # With DiCE's strings left unconverted, '0' != 0 made every categorical
        # look changed in every counterfactual.
        instance = pd.DataFrame({"SEX": [0], "HBA1C": [7.5]})
        df_cf = cf._to_numeric_where_possible(
            pd.DataFrame({"SEX": ["0", "0"], "HBA1C": ["7.0", "6.5"]}))
        out = cf.get_most_changed_feature(df_cf, instance, _config(), 0, savedir=None)
        counts = dict(zip(out["feature"], out["change count"]))
        self.assertEqual(counts["SEX"], 0)
        self.assertEqual(counts["HBA1C"], 2)


class LocalCfDistanceTests(unittest.TestCase):
    """The function mutated its caller's frame, counted the outcome column in
    sparsity, and discarded its sort_by result."""

    def _frames(self):
        instance = pd.DataFrame({"A": [1.0], "B": [2.0]})
        cf_df = pd.DataFrame({"A": [1.0, 3.0, 2.0], "B": [2.0, 2.0, 2.0],
                              "Confirmed_Binary_DPN": [0, 1, 1]})
        return instance, cf_df

    def test_caller_frame_is_not_modified(self):
        instance, cf_df = self._frames()
        before = list(cf_df.columns)
        cf.get_local_cf_distances(instance, cf_df, _config(), 0, savedir=None)
        self.assertListEqual(list(cf_df.columns), before)

    def test_identical_row_has_zero_sparsity(self):
        # The outcome column diffed to NaN and counted as an altered column.
        instance, cf_df = self._frames()
        _diffs, out = cf.get_local_cf_distances(instance, cf_df, _config(), 0, savedir=None)
        self.assertEqual(out.loc[0, "sparsity"], 0)
        self.assertEqual(out.loc[0, "L1_dist"], 0.0)

    def test_sort_by_is_applied_and_row_labels_survive(self):
        instance, cf_df = self._frames()
        _diffs, out = cf.get_local_cf_distances(instance, cf_df, _config(), 0,
                                                sort_by="L2_dist", savedir=None)
        self.assertTrue(out["L2_dist"].is_monotonic_increasing)
        self.assertSetEqual(set(out.index), {0, 1, 2})

    def test_empty_input_returns_empty_frames(self):
        instance, _ = self._frames()
        diffs, out = cf.get_local_cf_distances(instance, pd.DataFrame(), _config(), 0)
        self.assertTrue(diffs.empty and out.empty)


class CatBoostWrapperTests(unittest.TestCase):
    """DiCE hands the model 'category' dtype columns; a CatBoost model trained without
    cat_features rejects them outright."""

    class _Model:
        def predict_proba(self, X):
            for col in X.columns:
                assert not isinstance(X[col].dtype, pd.CategoricalDtype), \
                    f"column {col} reached the model as category dtype"
            p = np.asarray(X["HBA1C"], dtype=float) / 10.0
            return np.column_stack([1 - p, p])

    def test_category_columns_are_cast_before_reaching_the_model(self):
        wrapped = cf.CatBoostWrapper(self._Model(), threshold=0.5)
        X = pd.DataFrame({"SEX": pd.Categorical([0, 1]), "HBA1C": [4.0, 8.0]})
        proba = wrapped.predict_proba(X)
        np.testing.assert_allclose(proba[:, 1], [0.4, 0.8])

    def test_threshold_is_applied_in_predict(self):
        wrapped = cf.CatBoostWrapper(self._Model(), threshold=0.75)
        X = pd.DataFrame({"SEX": pd.Categorical([0, 1]), "HBA1C": [4.0, 8.0]})
        np.testing.assert_array_equal(wrapped.predict(X), [0, 1])

    def test_frames_without_categoricals_are_untouched(self):
        wrapped = cf.CatBoostWrapper(self._Model(), threshold=0.5)
        X = pd.DataFrame({"SEX": [0, 1], "HBA1C": [4.0, 8.0]})
        self.assertIs(cf.CatBoostWrapper._decategorize(X), X)


class PermittedRangeTests(unittest.TestCase):
    """get_global_permitted_range built its dataframe only under verbosity > 0, then
    used it in the savedir branch."""

    def test_saving_works_at_zero_verbosity(self):
        dfXy = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [0.0, 1.0, 2.0]})
        with tempfile.TemporaryDirectory() as tmp:
            out = cf.get_global_permitted_range(dfXy, ["A", "B"], _config(), 0,
                                                verbosity=0, savedir=Path(tmp))
            self.assertTrue((Path(tmp) / "catboost_split0_global_permitted_range.csv").exists())
            self.assertSetEqual(set(out), {"A", "B"})

    def test_a_zero_minimum_stays_at_zero(self):
        dfXy = pd.DataFrame({"B": [0.0, 1.0, 2.0]})
        out = cf.get_global_permitted_range(dfXy, ["B"], _config(), 0)
        self.assertEqual(out["B"][0], 0)


class GenerateDiverseCfsCacheTests(unittest.TestCase):
    """A run that found nothing still wrote a csv holding only the query instance, and
    resuming treated it as a valid result, producing empty reports that looked real."""

    def _args(self, tmp, cfs):
        instance = pd.DataFrame({"SEX": [0], "HBA1C": [7.5]})
        return dict(dice_exp=_StubExplainer(cfs), instance=instance, config=_config(),
                    split_index=0, threshold=0.5, features_to_vary=["HBA1C"],
                    patient_code=40, savedir=Path(tmp))

    def test_single_row_cache_is_regenerated(self):
        cfs = pd.DataFrame({"SEX": ["0"], "HBA1C": ["6.5"],
                            "Confirmed_Binary_DPN": ["0"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catboost_split0_patient040_local_cf.csv"
            pd.DataFrame({"SEX": [0], "HBA1C": [7.5]}).to_csv(path, index=False)
            out = cf.generate_diverse_cfs(**self._args(tmp, cfs))
            self.assertGreater(len(out), 1, "an instance-only cache must not be reused")

    def test_populated_cache_is_reused(self):
        cached = pd.DataFrame({"SEX": [0, 0], "HBA1C": [7.5, 6.5]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catboost_split0_patient040_local_cf.csv"
            cached.to_csv(path, index=False)
            out = cf.generate_diverse_cfs(**self._args(tmp, pd.DataFrame()))
            self.assertEqual(len(out), 2)

    def test_saved_csv_has_no_index_column(self):
        # Writing the index made it reappear as 'Unnamed: 0', a phantom feature in
        # every diff computed downstream.
        cfs = pd.DataFrame({"SEX": ["0"], "HBA1C": ["6.5"],
                            "Confirmed_Binary_DPN": ["0"]})
        with tempfile.TemporaryDirectory() as tmp:
            cf.generate_diverse_cfs(**self._args(tmp, cfs))
            written = pd.read_csv(Path(tmp) / "catboost_split0_patient040_local_cf.csv")
            for col in written.columns:
                self.assertFalse(col.startswith("Unnamed"))


# ---------------------------------------------------------------------------
# global importance and the timeout wrapper
# ---------------------------------------------------------------------------

class GlobalImportanceTests(unittest.TestCase):
    """n_cpus=-1 was documented as 'use all' but took the serial path, and would have
    crashed in np.array_split; chunking also ignored DiCE's 10-instance floor."""

    class _Exp:
        def __init__(self):
            self.chunk_sizes = []

        def global_feature_importance(self, X_chunk, **kwargs):
            self.chunk_sizes.append(len(X_chunk))
            return SimpleNamespace(summary_importance={"HBA1C": 0.9, "HPN": 0.5})

    def test_chunks_never_fall_below_dices_ten_instance_floor(self):
        exp = self._Exp()
        X = pd.DataFrame({"a": np.arange(38.0)})
        cf.parallel_global_feature_importance(
            exp, X, _config(), ["HBA1C"], {}, 0.5, "binary", n_jobs=8)
        self.assertTrue(all(n >= 10 for n in exp.chunk_sizes),
                        f"chunk sizes {exp.chunk_sizes} must all be >= 10")

    def test_minus_one_resolves_instead_of_crashing_array_split(self):
        exp = self._Exp()
        X = pd.DataFrame({"a": np.arange(38.0)})
        out, _ = cf.parallel_global_feature_importance(
            exp, X, _config(), ["HBA1C"], {}, 0.5, "binary", n_jobs=-1)
        self.assertFalse(out.empty)

    def test_small_test_set_clamps_to_a_single_chunk(self):
        exp = self._Exp()
        X = pd.DataFrame({"a": np.arange(12.0)})
        cf.parallel_global_feature_importance(
            exp, X, _config(), ["HBA1C"], {}, 0.5, "binary", n_jobs=8)
        self.assertEqual(exp.chunk_sizes, [12])

    def test_aggregation_tolerates_a_feature_missing_from_a_chunk(self):
        # DiCE omits a feature whose importance is zero in that chunk.
        out = cf._aggregate_importance([{"A": 1.0, "B": 0.5}, {"A": 0.5}], [10, 10])
        self.assertAlmostEqual(out.loc["A", "importance"], 0.75)
        self.assertAlmostEqual(out.loc["B", "importance"], 0.25)


class TimeoutWrapperTests(unittest.TestCase):
    """The parent joined the child before draining the result queue, so a result larger
    than the pipe buffer looked like a timeout."""

    def test_large_result_returns_promptly(self):
        @timeout_mod.timeout(30)
        def big():
            return pd.DataFrame(np.zeros((20000, 50)))

        started = time.time()
        out = big()
        self.assertEqual(out.shape, (20000, 50))
        self.assertLess(time.time() - started, 25,
                        "a large result must not stall until the timeout")

    def test_timeout_still_raises(self):
        @timeout_mod.timeout(2)
        def slow():
            time.sleep(60)

        with self.assertRaises(TimeoutError):
            slow()

    def test_child_exception_propagates(self):
        @timeout_mod.timeout(30)
        def boom():
            raise ValueError("from the child")

        with self.assertRaises(ValueError):
            boom()


class TimeoutPresetTests(unittest.TestCase):
    """The CLI choices and the presets are taken from one dict so they cannot drift."""

    def test_presets_cover_the_documented_names(self):
        self.assertSetEqual(set(cf.TIMEOUT_PRESETS), {"fast", "normal", "long", "extended"})

    def test_known_preset_wraps_the_function(self):
        self.assertIsNot(cf.timed_generate_diverse_cfs("fast"), cf.generate_diverse_cfs)

    def test_unknown_preset_runs_without_a_limit(self):
        self.assertIs(cf.timed_generate_diverse_cfs(None), cf.generate_diverse_cfs)
        self.assertIs(cf.timed_generate_diverse_cfs("nonsense"), cf.generate_diverse_cfs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
