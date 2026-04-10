import json
import warnings
from pprint import pprint

import dice_ml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from module.backends.backend_adapter import get_dice_components
from module.models.optimizers import grid_search_cv_binary

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from module.dataload import DPN_data
from module.eda import EDA

import xgboost as xgb

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)  # decimal places for outputs from numpy
pd.set_option("display.precision", 3)  # decimal places for outputs from pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# MAIN METHOD
def main():
    # =============================================================================== #
    # =============================================================================== #
    # DATA LOADING
    D = DPN_data("../dataset/Sudoscan Working File with Stats.xlsx")
    D.load(classification="binary")

    df = D.df

    # DATA PREPARATION
    # SETTING X TO BE THE DATA COLUMNS AND Y TO BE THE CONFIRMED_BINARY_DPN COLUMN
    data_cols = df.drop(D.non_data_cols, axis=1, errors="ignore").columns
    X = df[data_cols]
    y = df[D.current_target_column]

    # SPLIT DATA INTO X AND Y FOR BOTH TRAINING AND TESTING
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    # =============================================================================== #
    # =============================================================================== #
    # DICTIONARY OF THE INITIAL MODELS THAT USE THE DEFAULT PARAMETERS
    initial_models = {
        "dummy": DummyClassifier().fit(X_train, y_train),
        "random_forest": RandomForestClassifier().fit(X_train, y_train),
        "decision_tree": DecisionTreeClassifier().fit(X_train, y_train),
        "xgb": xgb.XGBClassifier().fit(X_train, y_train),
        "logistic_regression": LogisticRegression().fit(X_train, y_train)
    }

    # RUNNING INITIAL MODELS, CALCULATING, AND STORING METRICS
    verbosity = 0

    initial_model_runs = {}
    for model_name, model in initial_models.items():
        stats = EDA.binary_classification_metrics(
            confusion_matrix(
                y_val,
                model.predict(X_val)
            ),
            labels=D.binary_class_label, verbosity=verbosity)

        pprint(stats)

        initial_model_runs[model_name] = stats

    # =============================================================================== #
    # =============================================================================== #
    # RUNNING MODEL OPTIMIZATION
    # MODEL ESTIMATORS
    estimators = {
        "dummy": DummyClassifier(),
        "random_forest": RandomForestClassifier(),
        "decision_tree": DecisionTreeClassifier(),
        "xgb": xgb.XGBClassifier(),
        "logistic_regression": LogisticRegression()
    }

    # LOADING JSON FILE STORING PARAMETER CONFIGURATIONS
    with open('model_configs/param_grids/binary_param_grids.json', 'r') as file:
        param_grids = json.load(file)

    optimized_params = {}

    # CONDUCTING GRID_SEARCH_CV ON THE BINARY CLASSIFICATION MODELS
    for estimator_name, estimator in estimators.items():
        params = grid_search_cv_binary(
            estimator,
            param_grids.get(estimator_name, {}),
            (X_train, y_train),
            scoring='youden_index',
            verbosity=verbosity,
            cv_splits=2
        )

        print("Optimized parameters for {}: {}".format(estimator_name, params))

        # TODO DUMMY, RANDOM FOREST, DEICION TREE, AND LOGISTIC REGRESSION PARAMETER SETS
        optimized_params[estimator_name] = params.best_params_

    pprint(optimized_params)

    # =============================================================================== #
    # =============================================================================== #
    # RUNNING DiCE COUNTERFACTUAL EXAMPLES

    optimized_models = {
        "xgb": xgb.XGBClassifier(**optimized_params["xgb"]).fit(X_train, y_train),
        "random_forest": RandomForestClassifier(**optimized_params["random_forest"]).fit(X_train, y_train),
        "decision_tree": DecisionTreeClassifier(**optimized_params["decision_tree"]).fit(X_train, y_train),
        "logistic_regression": LogisticRegression(**optimized_params["logistic_regression"]).fit(X_train, y_train)
    }

    for model_name, model in optimized_models.items():

        stats = EDA.binary_classification_metrics(
            confusion_matrix(
                y_val,
                model.predict(X_val)
            ),
            labels=D.binary_class_label, verbosity=verbosity)

        print(f"Statistics for optimized {model_name} model")
        pprint(stats)

    for model_name, model in optimized_models.items():
        d = dice_ml.Data(dataframe=df, continuous_features=data_cols.tolist(), outcome_name='Confirmed_Binary_DPN')
        m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")

        exp = dice_ml.Dice(d, m, method="genetic")

        print(f"generating counterfactuals for the {model_name} model")

        e1 = exp.generate_counterfactuals(
            X_val, total_CFs=10, desired_class="opposite")

        print(e1.cf_examples_list)

    # dice_model, dice_data = get_dice_components(
    #     model=model,
    #     backend=config["model_1"]["backend"],
    #     x_train=x_train,
    #     y_train=y_train,
    #     input_features=config["dataset"]["feature_set_1"],
    #     continuous_features=loader.numeric_cols.tolist(),
    #     target=config["dataset"]["target"]
    # )
    #
    # # Generate counterfactuals
    # dice = DiCEWrapper(dice_model, dice_data)
    #
    # # After you have trained your model...
    # y_preds = model.predict(x_test)
    #
    # # Generate batches of CFs for the whole test set
    # batched_cfs = dice.generate_batched(
    #     query_instances=x_test,
    #     predictions=y_preds,
    #     features_to_vary=loader.get_features_to_vary(),
    #     total_CFs=1
    # )
    #
    # # Example: loop through the outputs
    # for desired_class, cf_obj in batched_cfs.items():
    #     print(f"\n=== Counterfactuals for desired class {desired_class} ===")
    #     print(cf_obj.cf_examples_list[0].final_cfs_df)


if __name__ == "__main__":
    main()
