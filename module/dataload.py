import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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
    binary_cols = ['SEX', 'SUBJ', 'DM_DUR', 'INSULIN'] + comorbidity_cols  
     
    # These are initial definitions; numeric_cols will be adjusted in load()
    initial_numeric_cols = ['AGE', 'DM_DUR', 'HBA1C'] + [
        'MNSI'] + ncs_cols + sudo_cols  # Keep column_classes here for initial load
    categorical_cols = ['SEX', 'SUBJ', 'INSULIN'] + comorbidity_cols + neuro_cols
    col_types = [profile_cols, comorbidity_cols, neuro_cols, mnsi_col, ncs_cols, sudo_cols, column_classes]
    col_names = profile_cols + comorbidity_cols + neuro_cols + mnsi_col + ncs_cols + sudo_cols + column_classes

    # define list for training purposes
    multi_classes_labels = ['Negative', 'Possible', 'Probable', 'Confirmed']  # Labels for multiclass DPN_Status
    binary_classes_labels = ['Negative', 'Positive']  # Labels for multiclass DPN_Status
    # binary_class_label will now represent the specific 'Confirmed_Binary' column
    binary_class_column = 'Confirmed_Binary_DPN'
    multi_class_column = 'DPN_Status'
    non_data_cols = multi_classes_labels + [binary_class_column] + [multi_class_column]  # Adjusting this based on what's added/removed later
    data_cols = profile_cols + comorbidity_cols + neuro_cols + mnsi_col + ncs_cols + sudo_cols

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None  # Initialize df to None
        self.current_numeric_cols = []  # To store numeric columns after classification choice
        self.current_target_column = None  # To store the name of the active target column

    def load(self, one_hot_encode=False, one_hot_drop="first", classification="binary"):
        """
        Load data and return a dataframe, possibly with one-hot-encoded categorical data.
        Allows for 'binary' or 'multiclass' DPN classification.
        """
        if classification not in ["binary", "multiclass"]:
            raise ValueError("Invalid value for 'classification'. Must be 'binary' or 'multiclass'.")

        df = pd.read_excel(self.filepath, skiprows=3, usecols="B:G, I:AT", names=self.col_names, na_values=['-'],
                           decimal=',')
        df = df.dropna(how='all')
        df = df[:-1]
        assert df.shape == (190, len(self.col_names))
        # pd.set_option('future.no_silent_downcasting', True)
        df["DM_DUR"] = df["DM_DUR"].replace({"<1": "1", ">10": "11"}).astype('float')
        df.replace('NR', 0, inplace=True)
        df.replace('NO F WAVE', 0, inplace=True)
        df.replace({'Y': 1, 'M': 1, 'N': 0, 'F': 0, np.nan: 0}, inplace=True)


        # --- Explicitly convert all intended numeric columns to float ---
        # This is the crucial step to resolve the 'object' dtype error
        for col in self.initial_numeric_cols:
            if col in df.columns:  # Check if column exists in the dataframe
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN

        # Start with a copy of initial_numeric_cols
        self.current_numeric_cols = list(self.initial_numeric_cols)

        # Impute NaNs in numeric columns AFTER conversion to float
        for col in self.current_numeric_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())  # Fill with mean for numeric NaNs

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
        # Add the newly created 'DPN_Status' to numeric columns as it's an ordinal numerical value

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

        # Now, self.df will store the DataFrame with the chosen classification target and features
        self.df = df

        if one_hot_encode:
            encoder = OneHotEncoder(sparse_output=False, drop=one_hot_drop)
            encoded_data = encoder.fit_transform(self.df[self.categorical_cols])

            df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.categorical_cols))
            # Drop original categorical columns from self.df before joining encoded ones
            self.df = self.df.drop(columns=self.categorical_cols).join(df_encoded)
        # If no one-hot encoding, self.df is already assigned from the classification block

        # The 'Negative' column from before is no longer relevant as the binary target is now 'Confirmed_Binary_DPN'
        # and the multiclass target is 'DPN_Status' directly.
        # So, no need to create df['Negative'] = 1 - df['Any_DPN']

        return self.df

    def get_numeric_cols(self):
        """Returns the list of numeric columns after loading and classification choice."""
        return self.current_numeric_cols

    def get_categorical_cols(self):
        """Returns the list of categorical columns."""
        # These remain consistent regardless of encoding choice in terms of list of original columns
        return self.categorical_cols

    def get_target_column(self):
        """Returns the name of the active target column ('Confirmed_Binary_DPN' or 'DPN_Status')."""
        return self.current_target_column
