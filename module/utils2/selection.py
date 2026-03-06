""" Utility Functions For Feature and Model Selection
"""


import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Classical ML models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Gradient boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import sys 
sys.path.append('..')  

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, make_scorer
)

# Define Models
models = {
    "Naive": DummyClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3),

    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),

    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),

    "Linear SVM": SVC(kernel="linear", probability=True),
    "RBF SVM": SVC(kernel="rbf", probability=True),
}

def get_column_types(X):
    """
    Returns list of binary and continuous column names/indices.
    Works for both Pandas DataFrames and Numpy Arrays.
    """
    # Convert to DataFrame if it's a numpy array for easier handling
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X)
    else:
        df = X

    binary_cols = []
    continuous_cols = []

    for col in df.columns:
        # Check if column has only 0s and 1s (ignoring NaNs)
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
            
    return binary_cols, continuous_cols


def build_smart_pipeline(model_name, model_instance, X_train, verbosity):
    """
    Builds a pipeline that:
    1. Leaves binary columns alone ('passthrough').
    2. Scales continuous columns based on what the model needs.
    """

    # Detect columns based on the training data provided
    binary_cols, continuous_cols = get_column_types(X_train)

    # Define Model Groups
    NEEDS_STANDARD = ["RBF SVM", "Linear SVM", "Naive Bayes", "Logistic Regression", "LDA", "QDA"]
    NEEDS_MINMAX = ['kNN', ]
    
    # Select the Scaler
    scaler = None
    if model_name in NEEDS_STANDARD:
        scaler = StandardScaler()
        scale_type = "StandardScaler"
    elif model_name in NEEDS_MINMAX:
        scaler = MinMaxScaler()
        scale_type = "MinMaxScaler"
    else:
        scaler = None # e.g. Trees don't need scaling
        scale_type = "None"

    # Build the Preprocessor
    if scaler:
        # If scaling is needed, apply it ONLY to continuous cols
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale_continuous', scaler, continuous_cols),
                ('keep_binary', 'passthrough', binary_cols)
            ],
            # remainder='drop' # safety: drop anything else not accounted for
        )
        steps = [('preprocessor', preprocessor), ('model', model_instance)]
        print(f"⚙️ {model_name}: Scaling continuous cols with {scale_type}")
    else:
        # If no scaling needed (e.g. Random Forest), skip preprocessor entirely
        steps = [('model', model_instance)]
        if (verbosity>0):
            print(f"⏩ {model_name}: No scaling applied.")

    return Pipeline(steps)

def youden_index_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def get_youden_scorer():
    return make_scorer(youden_index_score, greater_is_better=True)

def get_specificity_scorer():
    return make_scorer(specificity_score)


def benchmark_models(X, y, cv_splits, n_repeats, random_state, verbosity):
    """
    Model benchmarking
    Uses repeated stratified k-fold 
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rcv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "youden": get_youden_scorer(),
        "specificity": get_specificity_scorer()
    }

    results = []
    for name, model in tqdm(models.items()):
        pipe = build_smart_pipeline(name, model, X, verbosity)
        #scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        scores = cross_validate(pipe, X, y, cv=rcv, scoring=scoring, n_jobs=-1, error_score="raise")
        algo_results = {
            "Accuracy": scores["test_accuracy"],
            "Precision": scores["test_precision"],
            "Sensitivity": scores["test_recall"],
            "Specificity": scores["test_specificity"],
            "Youden Index": scores["test_youden"],
            "F1": scores["test_f1"],
            "ROC-AUC": scores["test_roc_auc"],
        }
        results.append({
            "model": name,
            "rcv_scores" : pd.DataFrame(algo_results)
        })

    # sort by Youden Index instead of ROC-AUC
    return results