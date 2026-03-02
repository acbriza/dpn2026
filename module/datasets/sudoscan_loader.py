import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SudoscanDataset:
    def __init__(self,
                 filepath="C:\\Users\\Admin\\Desktop\\projects\\OJT\\materials\\Sudoscan Working File with Stats - Copy.xlsx",
                 sheet_name="New RAw",
                 random_state=42):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.numeric_cols = None
        self.categorical_cols = None
        self.dataset = self.load()
        self.features_to_vary = self.dataset.drop(['SEX', 'AGE', 'CKD', 'GBS', 'DM DUR', 'NS', 'CAS (%)', 'DPN_Status', ],
                                                  axis=1).columns.tolist()

    def load(self):
        """Load and preprocess the dataset."""
        # Load Excel
        dataset = pd.read_excel(self.filepath, sheet_name=self.sheet_name)
        dataset = dataset.iloc[0:190]

        # Rename columns
        dataset = dataset.rename(columns={
            "Unnamed: 1": "SEX",
            "Unnamed: 2": "AGE",
            "Unnamed: 3": "SUBJ",
            "Unnamed: 4": "DM DUR",
            "Unnamed: 5": "INSULIN",
            "Unnamed: 6": "HBA1C",
            "Unnamed: 7": "DATE",
            "Unnamed: 8": "HPN",
            "Unnamed: 9": "PAOD",
            "Unnamed: 10": "DSLPDMIA",
            "Unnamed: 11": "CKD",
            "Unnamed: 12": "GBS",
            "Unnamed: 13": "DEC VS",
            "Unnamed: 14": "DEC PPS",
            "Unnamed: 15": "DEC PPS",
            "Unnamed: 16": "DEC LTS",
            "Unnamed: 17": "DEC AR",
            "Unnamed: 18": "MNSI",
            "SUDOSCAN": "FEET MEAN ESC",
            "Unnamed: 37": "FEET %ASSYM",
            "Unnamed: 38": "HANDS MEAN ESC",
            "Unnamed: 39": "HANDS %ASSYM",
            "Unnamed: 40": "NS",
            "Unnamed: 41": "CAS (%)",
            "Unnamed: 42": "Confirmed",
            "Unnamed: 43": "Probable",
            "Unnamed: 44": "Possible",
            "Unnamed: 45": "Any DPN"
        })

        # Drop DATE and SUBJ (optional ID)
        dataset = dataset.drop(['index', 'DATE', 'SUBJ'], axis=1)

        # Parse DM DUR
        def parse_dm_duration(val):
            val = str(val).strip()
            if val in ['', 'NR', 'nan']:
                return np.nan
            elif val == '<1':
                return np.random.uniform(0.1, 1.0)
            elif val == '>10':
                return np.random.uniform(10.1, 15.0)
            else:
                try:
                    return float(val)
                except ValueError:
                    return np.nan

        dataset['DM DUR'] = dataset['DM DUR'].apply(parse_dm_duration)

        # Combine DPN Status
        def combine_dpn(row):
            if row['Confirmed'] == 1:
                # return float(3)
                return 3
            elif row['Probable'] == 1:
                # return float(2)
                return 2
            elif row['Possible'] == 1:
                # return float(1)
                return 1
            else:
                # return float(0)
                return 0

        dataset['DPN_Status'] = dataset.apply(combine_dpn, axis=1)
        dataset = dataset.drop(['Confirmed', 'Probable', 'Possible', 'Any DPN'], axis=1)

        # Define columns
        self.categorical_cols = ['SEX', 'INSULIN', 'HPN', 'PAOD', 'DSLPDMIA', 'CKD', 'GBS',
                                 'DEC VS', 'DEC PPS', 'DEC LTS', 'DEC AR']
        self.numeric_cols = (dataset.columns.difference(self.categorical_cols)
                             .drop('DPN_Status')
                             .drop('NS')
                             .drop('CAS (%)'))

        numeric_cols = self.numeric_cols
        categorical_cols = self.categorical_cols

        # ✅ GENERALIZED: Clean numeric columns
        for col in numeric_cols:
            dataset[col] = (
                dataset[col]
                .astype(str).str.strip()
                .str.replace(',', '.', regex=False)
                .replace(['', 'NR', 'NO F WAVE', '-', 'nan', 'NaN'], np.nan)
            )
            dataset[col] = dataset[col].astype(float)

        # Fill numeric NaNs with column mean
        dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

        # ✅ GENERALIZED: Clean categorical columns
        for col in categorical_cols:
            dataset[col] = dataset[col].astype(str).str.strip().str.lower()

            # Replace blanks or weird 'nan' strings with mode()
            dataset[col] = dataset[col].replace(['', 'nan', 'none', 'NaN'], np.nan)
            if dataset[col].isnull().sum() > 0:
                most_common = dataset[col].mode()[0]
                dataset[col] = dataset[col].fillna(most_common)

            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])

        # Save references
        self.dataset = dataset
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        return dataset

    def split(self, dataset=None, train_percentage=80, random_state=None):
        """Split into train/test sets with stratification."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        if dataset is None:
            dataset = self.dataset

        dataset = dataset.drop(['NS', 'CAS (%)'], axis=1)

        X = dataset.drop('DPN_Status', axis=1)
        y = dataset['DPN_Status']

        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            test_size= 1 - (train_percentage / 100),
            stratify=y,
            random_state=random_state,
        )

        return x_train, x_test, y_train, y_test

    def get_features_to_vary(self):
        return self.features_to_vary

    def get_cols(self):
        return self.numeric_cols, self.categorical_cols

    def get_x(self):
        return self.dataset.drop('DPN_Status', axis=1)

    def get_y(self):
        return self.dataset['DPN_Status']

    def get_dataset(self):
        return self.dataset

    def validate_dataset_types(dataset, numeric_cols, categorical_cols, outcome_name):
        print("\n=== VALIDATING DATASET TYPES ===")

        # Check numeric columns
        for col in numeric_cols:
            dtype = dataset[col].dtype
            if dtype != 'float64':
                print(f"❗ WARNING: {col} is {dtype}, expected float64.")

        # Check categorical columns
        for col in categorical_cols:
            dtype = dataset[col].dtype
            if not (np.issubdtype(dtype, np.integer) or dtype == 'category'):
                print(f"❗ WARNING: {col} is {dtype}, expected int32/int64 or category.")

        # Check outcome column
        outcome_dtype = dataset[outcome_name].dtype
        print(f"\nOutcome '{outcome_name}' dtype: {outcome_dtype}")
        if not np.issubdtype(outcome_dtype, np.integer):
            print(f"❗ WARNING: Outcome column is {outcome_dtype}, expected integer.")
        else:
            print("✅ Outcome dtype is good.")

        print("\nSample unique outcome values:", dataset[outcome_name].unique())

        print("Validation complete.\n")
