from pathlib import Path

import pytest

from module import dataload

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "EAMC_DPN_Dataset.xlsx"


@pytest.fixture
def loader():
    return dataload.DPN_data(str(DATASET_PATH))


def _load(loader, tmp_path, **kwargs):
    kwargs.setdefault("report_path", str(tmp_path / "cleaning_report.txt"))
    return loader.load(**kwargs)


def test_read_raw_has_expected_shape_and_values(loader):
    raw = loader.read_raw()

    assert raw.shape == (190, len(dataload.DPN_data.col_names))
    assert list(raw.columns) == dataload.DPN_data.col_names
    # Spot-check the first data row against the source spreadsheet (patient CODE=1).
    assert raw.loc[0, "SEX"] == "M"
    assert raw.loc[0, "AGE"] == 64
    assert raw.loc[0, "Confirmed"] == 1


def test_load_binary_default_drop_strategy(loader, tmp_path):
    df = _load(loader, tmp_path, classification="binary")

    assert loader.get_target_column() == "Confirmed_Binary_DPN"
    assert loader.current_labels == ["Negative", "Positive"]
    assert df.shape[0] == 187  # 3 of 190 rows dropped for missing numeric values
    assert not df.isnull().any().any()
    assert df["Confirmed_Binary_DPN"].sum() == 130


def test_load_multiclass_default_drop_strategy(loader, tmp_path):
    df = _load(loader, tmp_path, classification="multiclass")

    assert loader.get_target_column() == "DPN_Status"
    assert loader.current_labels == ["Negative", "Possible", "Probable", "Confirmed"]
    assert df.shape[0] == 187
    assert not df.isnull().any().any()
    assert df["DPN_Status"].value_counts().to_dict() == {3: 130, 2: 27, 1: 22, 0: 8}


def test_drop_strategy_removes_rows_with_missing_numeric_values(loader, tmp_path):
    _load(loader, tmp_path, classification="binary", nan_strategy="drop")

    dropped_codes = sorted(set(range(1, 191)) - set(loader.patient_codes))
    assert dropped_codes == [36, 46, 173]


def test_patient_code_round_trip(loader, tmp_path):
    _load(loader, tmp_path, classification="binary", nan_strategy="drop")

    assert loader.index_to_patient_code(0) == 1
    with pytest.raises(IndexError):
        loader.index_to_patient_code(len(loader.patient_codes))


def test_classification_counts_are_internally_consistent(loader, tmp_path):
    _load(loader, tmp_path, classification="binary", nan_strategy="impute_mean")

    counts = loader.classification_df.iloc[0]
    assert counts["Negative"] + counts["Any_DPN"] == 190
    assert counts["Confirmed"] + counts["Probable"] + counts["Possible"] == counts["Any_DPN"]


def test_one_hot_encode_joins_without_row_misalignment(loader, tmp_path):
    df = _load(loader, tmp_path, classification="binary", one_hot_encode=True)

    for col in dataload.DPN_data.categorical_cols:
        assert col not in df.columns
    assert "SEX_1" in df.columns


@pytest.mark.xfail(
    reason="known bug: NaNs in categorical_cols (e.g. INSULIN, patient CODE=46) are "
    "invisible to nan_strategy, which only inspects current_numeric_cols. Under "
    "'impute_mean' they are never filled, and they only happen to be caught by "
    "'drop' when the same row also has a missing numeric value.",
    strict=True,
)
def test_impute_mean_strategy_leaves_no_nan_anywhere(loader, tmp_path):
    df = _load(loader, tmp_path, classification="binary", nan_strategy="impute_mean")

    assert not df.isnull().any().any()


def test_invalid_classification_raises(loader):
    with pytest.raises(ValueError):
        loader.load(classification="not-a-real-mode")


def test_invalid_nan_strategy_raises(loader):
    with pytest.raises(ValueError):
        loader.load(nan_strategy="not-a-real-strategy")
