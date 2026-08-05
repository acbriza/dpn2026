import pandas as pd

from module import dataload


def _make_raw_frame():
    columns = dataload.DPN_data.col_names + ["Confirmed", "Probable", "Possible", "Any_DPN"]
    rows = []
    for i in range(191):
        row = {
            "SEX": "F" if i % 2 == 0 else "M",
            "AGE": 50 + i % 20,
            "SUBJ": "A" if i % 3 != 0 else "B",
            "DM_DUR": 5 + (i % 10),
            "INSULIN": "Y" if i % 4 == 0 else "N",
            "HBA1C": 6.0 + (i % 5) * 0.2,
            "HPN": "N",
            "PAOD": "N",
            "DSLPDMIA": "N",
            "CKD": "N",
            "GBS": "N",
            "DEC_VS": "N",
            "DEC_PPS": "N",
            "DEC_LTS": "N",
            "DEC_AR": "N",
            "MNSI": 1 + (i % 3),
            "SSA_L": 1.0,
            "SSC_L": 1.0,
            "SPSA_L": 1.0,
            "SPSC_L": 1.0,
            "MCV_L": 1.0,
            "DL_L": 1.0,
            "CMAPANK_L": 1.0,
            "CMAPKNE_L": 1.0,
            "FWAVE_L": 1.0,
            "SSA_R": 1.0,
            "SSC_R": 1.0,
            "SPSA_R": 1.0,
            "SPSC_R": 1.0,
            "MCV_R": 1.0,
            "DL_R": 1.0,
            "CMAPANK_R": 1.0,
            "CMAPKNE_L": 1.0,
            "FWAVE_R": 1.0,
            "FEET_MEAN_ESC": 1.0,
            "FEET_PCT_ASYM": 1.0,
            "HAND_MEAN_ESC": 1.0,
            "HAND_PCT_ASYM": 1.0,
            "NS": 1.0,
            "CAS": 1.0,
            "Confirmed": 1 if i == 0 else 0,
            "Probable": 0,
            "Possible": 0,
            "Any_DPN": 1 if i == 0 else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    return df


def test_binary_loader_builds_expected_target(monkeypatch):
    monkeypatch.setattr(dataload.pd, "read_excel", lambda *args, **kwargs: _make_raw_frame())

    loader = dataload.DPN_data("dummy.xlsx")
    frame = loader.load(classification="binary")

    assert loader.current_labels == ["Negative", "Positive"]
    assert loader.get_target_column() == "Confirmed_Binary_DPN"
    assert "Confirmed_Binary_DPN" in frame.columns
    assert frame["Confirmed_Binary_DPN"].sum() == 1


def test_multiclass_loader_builds_expected_target(monkeypatch):
    monkeypatch.setattr(dataload.pd, "read_excel", lambda *args, **kwargs: _make_raw_frame())

    loader = dataload.DPN_data("dummy.xlsx")
    frame = loader.load(classification="multiclass")

    assert loader.current_labels == ["Negative", "Possible", "Probable", "Confirmed"]
    assert loader.get_target_column() == "DPN_Status"
    assert "DPN_Status" in frame.columns
    assert frame["DPN_Status"].iloc[0] == 3
