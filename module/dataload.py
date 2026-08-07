from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

pd.set_option('future.no_silent_downcasting', True)

class DPN_data:
    # define columns
    code_col = ['CODE']
    profile_cols = ['SEX', 'AGE', 'SUBJ', 'DM_DUR', 'INSULIN', 'HBA1C']
    comorbidity_cols = ['HPN', 'PAOD', 'DSLPDMIA', 'CKD', 'GBS']
    neuro_cols = ['DEC_VS', 'DEC_PPS', 'DEC_LTS', 'DEC_AR']
    mnsi_col = ['MNSI']
    ncs_cols = ['SSA_L', 'SSC_L', 'SPSA_L', 'SPSC_L', 'MCV_L', 'DL_L', 'CMAPANK_L', 'CMAPKNE_L', 'FWAVE_L',
                'SSA_R', 'SSC_R', 'SPSA_R', 'SPSC_R', 'MCV_R', 'DL_R', 'CMAPANK_R', 'CMAPKNE_R', 'FWAVE_R']
    sudo_cols = ['FEET_MEAN_ESC', 'FEET_PCT_ASYM', 'HAND_MEAN_ESC', 'HAND_PCT_ASYM', 'NS', 'CAS']
    column_classes = ['Confirmed', 'Probable', 'Possible', 'Any_DPN']
    binary_cols = ['SEX', 'SUBJ', 'INSULIN'] + comorbidity_cols

    # These are initial definitions; numeric_cols will be adjusted in load()
    # (column_classes is deliberately excluded here -- it holds the raw
    # classification columns, which are consolidated into DPN_Status in load()
    # rather than treated as a numeric feature)
    initial_numeric_cols = ['AGE', 'DM_DUR', 'HBA1C'] + [
        'MNSI'] + ncs_cols + sudo_cols
    categorical_cols = ['SEX', 'SUBJ', 'INSULIN'] + comorbidity_cols + neuro_cols
    col_names = profile_cols + comorbidity_cols + neuro_cols + mnsi_col + ncs_cols + sudo_cols + column_classes

    # define list for training purposes
    multi_classes_labels = ['Negative', 'Possible', 'Probable', 'Confirmed']  # Labels for multiclass DPN_Status
    binary_classes_labels = ['Negative', 'Positive']  # Labels for multiclass DPN_Status
    # binary_class_label will now represent the specific 'Confirmed_Binary' column
    binary_class_column = 'Confirmed_Binary_DPN'
    multi_class_column = 'DPN_Status'
    non_data_cols = multi_classes_labels + [binary_class_column] + [multi_class_column]  # Adjusting this based on what's added/removed later
    data_cols = profile_cols + comorbidity_cols + neuro_cols + mnsi_col + ncs_cols + sudo_cols

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None  # Initialize df to None
        self.current_numeric_cols: list[str] = []  # To store numeric columns after classification choice
        self.current_target_column: Optional[str] = None  # To store the name of the active target column
        self.patient_codes: Optional[np.ndarray] = None  # To store the original patient codes after dropping rows

    def read_raw(self) -> pd.DataFrame:
        """
        Read the source spreadsheet. Expose this method to allow checking the raw data 
        externally before cleaning and classification."""
        df = pd.read_excel(self.filepath, skiprows=3, usecols="B:G, I:AT", names=self.col_names, na_values=['-'],
                           decimal=',')
        return df

    def load(self, one_hot_encode: bool = False, one_hot_drop: str = "first",
              classification: str = "binary", report_path: Optional[str] = None,
              nan_strategy: str = "drop") -> pd.DataFrame:
        """
        Load data and return a dataframe, possibly with one-hot-encoded categorical data.
        Allows for 'binary' or 'multiclass' DPN classification.

        Args:
            one_hot_encode: If True, one-hot encode categorical_cols and join the
                result back onto the returned dataframe in place of the originals.
            one_hot_drop: Passed through to sklearn's OneHotEncoder(drop=...).
            classification: 'binary' or 'multiclass'; selects which DPN target
                column is built and exposed via get_target_column().
            report_path: Where to write the text report of value-replacement cleaning
                (see _clean_raw_values). Defaults to "cleaning_report.txt" next to filepath.
            nan_strategy: 'drop' to remove rows with a NaN in any numeric column (default),
                or 'impute_mean' to fill NaNs with the column mean instead. See _clean_raw_values.

        Returns:
            The loaded, cleaned dataframe (also stored on self.df).

        Side effects:
            Sets self.df, self.current_numeric_cols, self.current_target_column,
            and self.current_labels and self.patient_codes.
        """
        if classification not in ["binary", "multiclass"]:
            raise ValueError("Invalid value for 'classification'. Must be 'binary' or 'multiclass'.")

        df = self.read_raw()
        df = self._verify_rows(df)
        df = self._clean_raw_values(df, report_path=report_path, nan_strategy=nan_strategy)
        df = self._build_dpn_status(df)
        df = self._apply_classification_target(df, classification)

        # Now, self.df will store the DataFrame with the chosen classification target and features
        self.df = df

        if one_hot_encode:
            self.df = self._one_hot_encode(self.df, one_hot_drop)
        # If no one-hot encoding, self.df is already assigned above

        return self.df

    def _verify_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim dataframe to the 190 subject rows."""
        df = df.dropna(how='all') # drop blank row (2nd to the last row)
        df = df[:-1]  # the sheet's last row is a totals row, not a subject record, so drop it
        if df.shape != (190, len(self.col_names)):
            raise ValueError(f"Expected 190 rows x {len(self.col_names)} columns after cleanup, got {df.shape}")
        return df

    def _clean_raw_values(self, df: pd.DataFrame, report_path: Optional[str] = None,
                           nan_strategy: str = "drop") -> pd.DataFrame:
        """Normalize raw cell values (units, response codes, Y/N flags) and handle missing numerics.

        Args:
            nan_strategy: 'drop' to remove rows with a NaN in any numeric column (default),
                or 'impute_mean' to fill NaNs with the column mean instead.
        """
        if nan_strategy not in ("drop", "impute_mean"):
            raise ValueError("Invalid value for 'nan_strategy'. Must be 'drop' or 'impute_mean'.")

        replaced_cells = []  # (row index, column, old value, new value) for every value replaced below

        self._record_replacements(df, {"<1": "1", ">10": "11"}, replaced_cells, column="DM_DUR")
        df["DM_DUR"] = df["DM_DUR"].replace({"<1": "1", ">10": "11"}).astype('float')

        self._record_replacements(df, {'NR': 0}, replaced_cells)
        df.replace('NR', 0, inplace=True)

        self._record_replacements(df, {'NO F WAVE': 0}, replaced_cells)
        df.replace('NO F WAVE', 0, inplace=True)

        # no need to record replacements for Y/M/N/F since they are being replaced with 1/0, which is a standard mapping
        df.replace({'Y': 1, 'M': 1, 'N': 0, 'F': 0}, inplace=True)

        # --- Explicitly convert all intended numeric columns to float ---
        # This is the crucial step to resolve the 'object' dtype error
        for col in self.initial_numeric_cols:
            if col in df.columns:  # Check if column exists in the dataframe
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN

        # Start with a copy of initial_numeric_cols
        self.current_numeric_cols = list(self.initial_numeric_cols)

        # Handle NaNs in numeric columns AFTER conversion to float: either impute with the
        # column mean, or drop the whole row (default), depending on nan_strategy.
        imputed_cells = []  # (row index, column, NaN, column mean) for every value imputed below
        dropped_rows = []  # (row index, [columns with NaN]) for every row dropped below
        if nan_strategy == "impute_mean":
            for col in self.current_numeric_cols:
                if col in df.columns:
                    if df[col].isnull().any():
                        col_mean = df[col].mean()
                        for idx in df.index[df[col].isnull()]:
                            imputed_cells.append((idx, col, np.nan, col_mean))
                        df[col] = df[col].fillna(col_mean)  # Fill with mean for numeric NaNs
        else:
            present_numeric_cols = [col for col in self.current_numeric_cols if col in df.columns]
            nan_mask = df[present_numeric_cols].isnull()
            rows_with_nan = nan_mask.any(axis=1)
            for idx in df.index[rows_with_nan]:
                nan_cols = nan_mask.columns[nan_mask.loc[idx]].tolist()
                dropped_rows.append((idx, nan_cols))
            df = df[~rows_with_nan]

            self.patient_codes = df.index.to_numpy() + 1 # Get original Patient Code in the Excel sheet (1-based)
            df = df.reset_index() # Reset index after dropping rows to maintain a clean 0..n-1 range            

        self._write_cleaning_report(replaced_cells, imputed_cells, dropped_rows, report_path)

        return df

    @staticmethod
    def _record_replacements(df: pd.DataFrame, mapping: dict, sink: list, column: Optional[str] = None) -> None:
        """Record which (row, column) cells currently hold each key in mapping, before it gets replaced."""
        cols = [column] if column else list(df.columns)
        for old_value, new_value in mapping.items():
            mask = df[cols] == old_value
            for col in cols:
                for idx in df.index[mask[col]]:
                    sink.append((idx, col, old_value, new_value))

    def _write_cleaning_report(self, replaced_cells: list, imputed_cells: list, dropped_rows: list,
                                report_path: Optional[str] = None) -> None:
        """Write a text file documenting every cell/row _clean_raw_values changed: value
        replacements, NaN imputations, and rows dropped for having a NaN numeric value."""
        def fmt(value):
            return f"{value:.4f}" if isinstance(value, float) else repr(value)

        path = Path(report_path) if report_path else Path(self.filepath).with_name("cleaning_report.txt")
        with open(path, "w") as f:
            f.write(f"Value replacement report - {len(replaced_cells)} cells changed\n")
            f.write("(row index = dataframe row position after header/blank/totals rows are dropped)\n")
            f.write("Thus, the patient's Code is row+1\n\n")
            for idx, col, old_value, new_value in replaced_cells:
                f.write(f"row {idx:>4}  {col:<12}  {fmt(old_value)} -> {fmt(new_value)}\n")

            f.write(f"\nMean-imputation report - {len(imputed_cells)} cells filled\n\n")
            for idx, col, old_value, new_value in imputed_cells:
                f.write(f"row {idx:>4}  {col:<12}  {fmt(old_value)} -> {fmt(new_value)}\n")

            f.write(f"\nDropped-rows report - {len(dropped_rows)} rows dropped for NaN numeric values\n\n")
            for idx, nan_cols in dropped_rows:
                f.write(f"row {idx:>4}  NaN in: {', '.join(nan_cols)}\n")

        print(f"Data laoding cleaning report written to {path}.")

    def _build_dpn_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate the raw Confirmed/Probable/Possible/Any_DPN columns into an ordinal DPN_Status column."""
        # --- First, create the DPN_Status (ordinal) column regardless of final classification ---
        # This acts as an intermediate step to derive both binary and multiclass targets
        def combine_dpn_status_ordinal(row):
            if row['Confirmed'] == 1:
                return 3  # Confirmed
            elif row['Probable'] == 1:
                return 2  # Probable
            elif row['Possible'] == 1:
                return 1  # Possible
            else:
                return 0  # Negative (no DPN classification)

        df['DPN_Status'] = df.apply(combine_dpn_status_ordinal, axis=1)

        # Drop the original classification columns as they are now consolidated into DPN_Status
        # and will not be used as direct features or targets
        df = df.drop(columns=['Confirmed', 'Probable', 'Possible', 'Any_DPN'])

        # Update current_numeric_cols to remove the original classification columns
        for col in self.column_classes:
            if col in self.current_numeric_cols:
                self.current_numeric_cols.remove(col)

        return df

    def _apply_classification_target(self, df: pd.DataFrame, classification: str) -> pd.DataFrame:
        """Derive the active target column (and its labels) from DPN_Status for the chosen classification mode."""
        # --- Now, handle the final classification target based on user choice ---
        if classification == "binary":
            # The binary target is 1 if DPN_Status is 'Confirmed' (value 3), else 0
            self.current_target_column = self.binary_class_column
            df[self.current_target_column] = (df['DPN_Status'] == 3).astype(int)
            self.current_labels = self.binary_classes_labels

            # Drop the intermediate 'DPN_Status' column if only binary 'Confirmed_Binary_DPN' is desired as target
            df = df.drop(columns=['DPN_Status'])
            # Remove 'DPN_Status' from numeric_cols as it's no longer a feature, but the source of the new target
            if 'DPN_Status' in self.current_numeric_cols:
                self.current_numeric_cols.remove('DPN_Status')

        elif classification == "multiclass":
            # For multiclass, 'DPN_Status' (0, 1, 2, 3) is the target
            self.current_target_column = 'DPN_Status'
            self.current_labels = self.multi_classes_labels
            # No need to drop 'DPN_Status' here, as it is the target

        # The binary target is now 'Confirmed_Binary_DPN' and the multiclass target is 'DPN_Status'.
        return df

    def _one_hot_encode(self, df: pd.DataFrame, one_hot_drop: str) -> pd.DataFrame:
        """One-hot encode categorical_cols and join the result back onto df in their place."""
        encoder = OneHotEncoder(sparse_output=False, drop=one_hot_drop)
        encoded_data = encoder.fit_transform(df[self.categorical_cols])

        # index=df.index keeps the join below aligned by row even if df's index
        # isn't a clean 0..n-1 range (e.g. if dropna(how='all') in _read_raw dropped
        # rows out of the middle of the sheet); df_encoded would otherwise get its
        # own default RangeIndex and rows could silently mismatch on join.
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.categorical_cols),
                                   index=df.index)
        # Drop original categorical columns from df before joining encoded ones
        return df.drop(columns=self.categorical_cols).join(df_encoded)

    def index_to_patient_code(self, index: int) -> int:
        """Returns the original patient code (1-based) for a given row index in the cleaned dataframe."""
        if self.patient_codes is None:
            raise ValueError("Patient codes are not available. Ensure that 'load' has been called.")
        if index < 0 or index >= len(self.patient_codes):
            raise IndexError(f"Index {index} is out of bounds for patient codes of length {len(self.patient_codes)}.")
        return self.patient_codes[index]

    def get_numeric_cols(self) -> list[str]:
        """Returns the list of numeric columns after loading and classification choice."""
        return self.current_numeric_cols

    def get_categorical_cols(self) -> list[str]:
        """Returns the list of categorical columns."""
        # These remain consistent regardless of encoding choice in terms of list of original columns
        return self.categorical_cols

    def get_target_column(self) -> Optional[str]:
        """Returns the name of the active target column ('Confirmed_Binary_DPN' or 'DPN_Status')."""
        return self.current_target_column
