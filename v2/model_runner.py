from pprint import pprint
from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from module.eda import EDA
from module.dataload import DPN_data

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
)
from sklearn.svm import LinearSVC, LinearSVR
from mord import LogisticAT


class ModelRunner:

    X_reduced = None

    # default train test split parameters
    train_test_split_params = {
        'test_size' : 0.25,
        'random_state' : 0
    }

    def __init__(self, models: dict, model_params: dict | None = None, classification: Literal["binary", "multiclass"] = "binary"):

        self.classification = classification

        self.D = DPN_data("../dataset/Sudoscan Working File with Stats.xlsx")
        self.df = self.D.load(classification=classification)

        self.X = self.df[self.D.data_cols]
        self.y = self.df[self.D.current_target_column]

        self.models = models

        if model_params:
            # TODO create the _set_params(models, model_params) method
            self._set_params(models, model_params)

        # If any of the models needs colinear feature elimination, create an X_pruned dataframe
        if any(self._needs_colinear_elimination(model) for model in models.values()):
            self.drop_colinear_features()

    def set_train_test_split_params(self, *kwargs):
        """
        Update train_test_split_params safely.
        Example: set_split_params(test_size=0.3, random_state=42)
        """
        valid_keys = {'test_size', 'train_size', 'random_state', 'shuffle'}

        for key, value in kwargs.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter: '{key}'. Allowed keys: {valid_keys}")
            self.train_test_split_params[key] = value

    def get_train_test_split_params(self):
        return self.train_test_split_params

    @staticmethod
    def set_model_params(models, model_params):
        for model_name, model in models.items():
            if isinstance(model, Pipeline):
                model.named_steps["classifier"].set_params(**model_params[model_name])
            else:
                model.set_params(**model_params[model])

    @staticmethod
    def _needs_colinear_elimination(model) -> bool:
        assert isinstance(model, BaseEstimator)

        if isinstance(model, Pipeline):
            # If the classifier is wrapped in a pipeline, set estimator to the inner classifier
            estimator = model.named_steps["classifier"]
        else:
            estimator = model

        if isinstance(estimator, type):
            raise ValueError("Expected an estimator instance, but got a class instead.")

        # Tuple of the classifiers/estimators that would need colinear feature elimination
        linear_models = (
            LinearRegression,
            LogisticRegression,
            Ridge,
            Lasso,
            ElasticNet,
            SGDClassifier,
            SGDRegressor,
            LinearSVC,
            LinearSVR,
            LogisticAT
        )
        return isinstance(estimator, linear_models)

    def drop_colinear_features(self):
        # Step 1: Get the list of features to drop
        features_to_drop = EDA.get_features_to_drop(self.X, self.y, threshold=0.8)

        # Step 2: Print the results and your next steps
        if features_to_drop:
            print('\n', "=" * 100, '\n')
            print("Features recommended for dropping due to high correlation:")
            pprint(features_to_drop)

            # Step 3: Create a new DataFrame with the features dropped
            self.X_reduced = self.X.drop(columns=features_to_drop)

            print("\nShape of original X:", self.X.shape)
            print("Shape of reduced X:", self.X_reduced.shape)
            print('\n', "=" * 100, '\n')
        else:
            print('\n', "=" * 100, '\n')
            print("No features were identified for dropping with a correlation threshold of 0.8.")
            print('\n', "=" * 100, '\n')

    def _get_xy(self, model):
        if self._needs_colinear_elimination(model):
            X = self.X_reduced
        else:
            X = self.X
        y = self.y

        return X, y

    def fit(self):
        for model in self.models.values():

            # Setting the X, y and splits for the specific model
            X, y = self._get_xy(model)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

            model.fit(X_train, y_train)

    def get_metrics(self, verbosity=0):
        initial_model_runs = {}

        # Get the metrics for each model
        for model_name, model in self.models.items():

            # Setting the X, y and splits for the specific model
            X, y = self._get_xy(model)
            X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, **self.train_test_split_params)

            # Choosing which metrics computation method to use
            assert self.classification in ["binary", "multiclass"]
            metrics_fn = (
                EDA.binary_classification_metrics
                if self.classification == 'binary'
                else EDA.multiclass_metrics
            )
            cm = confusion_matrix(y_val, model.predict(X_val))
            stats = metrics_fn(cm, labels=self.D.current_labels, verbosity=verbosity)

            initial_model_runs[model_name] = stats

        pprint(initial_model_runs)
        return initial_model_runs
