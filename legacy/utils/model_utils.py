import inspect

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

import mord


# Assuming the following are defined for a complete example
# mord is installed: pip install mord

def get_estimator_category(estimator):
    """
    Detects the category of a scikit-learn (or scikit-learn-like) estimator,
    including those wrapped in a Pipeline.
    """
    # If the estimator is a Pipeline, get the final classifier/regressor
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]

    categories = {
        "explainable": False,
        "accurate" : False,
        "ordinal": False,
        "requires_scaling": False,
        "sensitive_to_collinearity": False
    }

    # --- Categorization Logic ---
    # Linear Models (including generalized linear models like LogisticAT)
    linear_models = (
        LogisticRegression,
        LinearRegression,
        RidgeClassifier,
        mord.LogisticAT
    )
    if isinstance(estimator, linear_models):
        categories["explainable"] = True
        categories["sensitive_to_collinearity"] = True
        categories["requires_scaling"] = True

    # Ordinal-specific Models
    if isinstance(estimator, mord.LogisticAT):
        categories["ordinal"] = True

    # Distance-Based Models
    if isinstance(estimator, KNeighborsClassifier):
        categories["explainable"] = True
        categories["requires_scaling"] = True

    # Tree-Based Models
    if isinstance(estimator, DecisionTreeClassifier):
        categories["explainable"] = True

    # Neural Networks
    if isinstance(estimator, xgb.XGBClassifier):
        categories["accurate"] = True

    # Neural Networks
    if isinstance(estimator, MLPClassifier):
        categories["requires_scaling"] = True

    # Support Vector Machines (SVC, SVR)
    if isinstance(estimator, SVC):
        categories["requires_scaling"] = True
        # Linear SVMs are sensitive to collinearity
        if estimator.kernel == 'linear':
            categories["sensitive_to_collinearity"] = True

    # Tree-based Models (generally do not require scaling/collinearity checks)
    if isinstance(estimator, RandomForestClassifier):
        categories["accurate"] = True
        categories["sensitive_to_collinearity"] = False
        categories["requires_scaling"] = False

    return categories

# --- Your Models from the previous prompt ---
# Correctly instantiate the mord model

logistic_regression = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression())
])

mord_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', mord.LogisticAT())
])

initial_models = {
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "xgb": xgb.XGBClassifier(),
    "logistic_regression": logistic_regression,
    "mord_logisticat": mord_pipeline
}

for model_name, model in initial_models.items():
    print(f"Checking {model_name} estimator")
    print(get_estimator_category(model))