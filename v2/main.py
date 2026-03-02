import copy

from mord import LogisticAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from v2.model_runner import ModelRunner

# Define the binary classification models
binary_models = {
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "xgb": xgb.XGBClassifier(),
    "logistic_regression": Pipeline(steps=[('scaler', MinMaxScaler()), ('classifier', LogisticRegression())]),
}

# Copy all the classifiers from the binary models dict, then add MORD.LogisticAT model
multiclass_models = copy.deepcopy(binary_models)
multiclass_models["mord_logisticat"] = Pipeline(steps=[('scaler', MinMaxScaler()), ('classifier', LogisticAT())])

# Training and running multiclass classification models
multiclass_runner = ModelRunner(multiclass_models, classification="multiclass")
multiclass_runner.fit()
multiclass_metrics = multiclass_runner.get_metrics()

# Training and running binary classification models
binary_runner = ModelRunner(binary_models, classification="binary")
binary_runner.fit()
binary_metrics = binary_runner.get_metrics()