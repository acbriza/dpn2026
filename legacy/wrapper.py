from module.dataload import DPN_data
from module.eda import EDA

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mord import LogisticAT
import xgboost as xgb



class Main:
    def __init__(self, estimators, run_type, dataset="default"):
        '''
        :param estimators: Dictionary of estimators
        :param run_type: 'binary' or 'ordinal'
        :param dataset: pandas Dataframe
        '''
        self.estimators = estimators
        assert run_type in ['binary', 'multiclass']
        assert dataset is not None

        if dataset == 'default':
            self.D = DPN_data("../dataset/Sudoscan Working File with Stats.xlsx")
            self.D.load(classification=run_type)

            self.df = self.D.df
            self.data_cols = self.df.drop(self.D.non_data_cols, axis=1, errors="ignore").columns

    def display_stats(self):
        df = self.df
        D = self.D
        # --- 1. Tabular Statistics ---
        print("--- Step 1: Display Tabular Statistics ---")
        print(df.columns)
        EDA.display_tabular_stats(df, include_object=False)

        # --- 2. Tabular Statistics as Graphs ---
        print("\n--- Step 2: Display Tabular Statistics as Graphs ---")

        # Visualize Missing Values Count
        EDA.plot_missing_values(df)

        # Visualize Unique Values Count for all numeric columns(excluding binary columns)
        EDA.plot_unique_values_count(df[D.initial_numeric_cols])

        # --- 3. Feature Distributions ---
        print("\n--- Step 3: Plot Feature Distributions ---")

        # Plot Numerical Distributions (histograms and box plots)
        EDA.plot_numerical_distributions(df, D.initial_numeric_cols)

        # Plot Categorical Distributions (count plots)
        EDA.plot_categorical_distributions(df, D.categorical_cols)


logistic_regression = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression())
])

mord = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticAT())
])

estimators = {
    "dummy": DummyClassifier(),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "xgb": xgb.XGBClassifier(),
    "logistic_regression": logistic_regression,
    "mord_logisticat": mord
}

obj = Main(estimators, 'multiclass')
obj.display_stats()
