import json
import time
from pprint import pprint
from typing import Dict

from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from module.eda import EDA


def optimize(estimators: Dict, verbosity: int):

    with open('model_configs/binary_param_grids.json', 'r') as file:
        param_grids = json.load(file)

    optimized_params = {}

    start_time = time.time()

    # CONDUCTING GRID_SEARCH_CV ON THE BINARY CLASSIFICATION MODELS
    for estimator_name, model in estimators.items():

        estimator = model[0]
        X_train_temp, y_train_temp, X_val_temp, y_val_temp = model[1]

        params = grid_search_cv_binary(
            estimator,
            param_grids.get(estimator_name, {}),
            (X_train_temp, y_train_temp),
            scoring='youden_index',
            verbosity=verbosity,
            cv_splits=2
        )

        print("Optimized parameters for {}: {}".format(estimator_name, params))

        optimized_params[estimator_name] = params.best_params_

    end_time = time.time()

    pprint(optimized_params)

    print(f"Grid Search finished in {end_time - start_time} seconds")


def grid_search_cv_binary(estimator, param_grid, train_data,
                          verbosity, labels="", scoring='youden_index', cv_splits=5, n_iter=100):
    """
    Performs a randomized hyperparameter search for a binary classification model.

    This function uses RandomizedSearchCV for efficient hyperparameter tuning,
    which is more practical than a full grid search on large parameter spaces.
    It uses a custom scoring function based on a confusion matrix.

    Args:
        estimator: The machine learning model to be tuned (e.g., XGBClassifier()).
        param_grid: A dictionary of parameters and their distributions to sample from.
        X_train (array-like): The training data features.
        y_train (array-like): The training data labels.
        verbosity (int): Controls the amount of output. 0 for silent, 1 for summary,
                         2 for detailed per-run metrics.
        labels (list): A list of class labels, used for clearer output in the
                       confusion matrix. Defaults to an empty string, but a list
                       like ["Negative", "Positive"] is recommended.
        scoring (str): The name of the custom metric to optimize. Defaults to 'youden_index'.
        cv_splits (int): The number of cross-validation folds. Defaults to 5.
        n_iter (int): Number of parameter settings that are sampled. Defaults to 100.
                      A higher value means a more exhaustive search.

    Returns:
        A fitted RandomizedSearchCV object.
    """

    X_train, y_train = train_data

    def get_binary_score_cv(y_true, y_pred, labels=labels, verbosity=verbosity):
        """
        Custom scoring function to be used by RandomizedSearchCV.
        It calculates a confusion matrix and returns a specific metric.
        """
        # Note: 'labels' from the outer function is used here.
        # This is okay because nested functions can access outer function's scope.
        cm = confusion_matrix(y_true, y_pred)

        # Assume EDA is a module with a function `binary_classification_metrics`.
        # This function should return a dictionary of metrics.
        metrics = EDA.binary_classification_metrics(cm, labels=labels, verbosity=verbosity)

        # Only print metrics if verbosity is high enough
        if verbosity > 1:
            pprint(metrics)

        # Return the score for the specified metric
        return metrics[scoring]

    # Wrap scorer for RandomizedSearchCV
    # 'greater_is_better=True' tells the search to maximize this score.
    scorer = make_scorer(get_binary_score_cv, greater_is_better=True)

    # Use RandomizedSearchCV for a more efficient search
    grid = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring=scorer,
        cv=cv_splits,
        n_iter=n_iter,
        random_state=42, # Set a random state for reproducibility
        verbose=verbosity # Let RandomizedSearchCV handle its own verbosity
    )

    # Fit the model
    grid.fit(X_train, y_train)

    if verbosity > 0:
        print("\n--- RandomizedSearchCV Results ---")
        print("Best parameters found:", grid.best_params_)
        # Use a more descriptive term than "score_aggregation"
        print(f"Best cross-validated {scoring}: {grid.best_score_:.4f} (average score)")

    return grid


def grid_search_cv_multiclass(estimator, param_grid, train_data,
                              verbosity, labels="", scoring='accuracy', cv_splits=5, n_iter=100,
                              **scoring_kwargs):
    """
    Performs a randomized hyperparameter search for a multiclass classification model.
    This function is designed to handle ordinal classification by leveraging
    custom scoring functions.

    This function uses RandomizedSearchCV for efficient hyperparameter tuning,
    which is more practical than a full grid search on large parameter spaces.
    It uses a custom scoring function that calculates a confusion matrix
    and a specific metric.

    Args:
        estimator: The machine learning model to be tuned (e.g., XGBClassifier()).
        param_grid: A dictionary of parameters and their distributions to sample from.
        train_data (tuple): A tuple containing (X_train, y_train).
        verbosity (int): Controls the amount of output. 0 for silent, 1 for summary,
                         2 for detailed per-run metrics.
        labels (list): A list of class labels, used for clearer output in the
                       confusion matrix. Defaults to an empty string, but a list
                       like ["Poor", "Fair", "Good"] is recommended.
        scoring (str): The name of the custom metric to optimize. Defaults to 'accuracy'.
        cv_splits (int): The number of cross-validation folds. Defaults to 5.
        n_iter (int): Number of parameter settings that are sampled. Defaults to 100.
                      A higher value means a more exhaustive search.
        **scoring_kwargs: Additional keyword arguments to pass to the scoring function.

    Returns:
        A fitted RandomizedSearchCV object.
    """

    X_train, y_train = train_data

    def get_multiclass_score_cv(y_true, y_pred, labels=labels, verbosity=verbosity, **kwargs):
        """
        Custom scoring function to be used by RandomizedSearchCV.
        It calculates a confusion matrix and returns a specific metric.
        This function handles ordinal classification metrics by assuming
        the 'EDA.multiclass_classification_metrics' function can calculate them.
        """
        # Note: 'labels' and 'verbosity' from the outer function are used here.
        cm = confusion_matrix(y_true, y_pred)

        # Assume EDA is a module with a function `multiclass_classification_metrics`.
        # This function should return a dictionary of metrics, including ordinal ones.
        metrics = EDA.multiclass_classification_metrics(cm, labels=labels, verbosity=verbosity, **kwargs)

        # Only print metrics if verbosity is high enough
        if verbosity > 1:
            pprint(metrics)

        # Return the score for the specified metric
        return metrics[scoring]

    # Wrap scorer for RandomizedSearchCV
    # 'greater_is_better=True' tells the search to maximize this score.
    scorer = make_scorer(get_multiclass_score_cv, greater_is_better=True, **scoring_kwargs)

    # Use RandomizedSearchCV for a more efficient search
    grid = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring=scorer,
        cv=cv_splits,
        n_iter=n_iter,
        random_state=42, # Set a random state for reproducibility
        verbose=2 # Let RandomizedSearchCV handle its own verbosity
    )

    # Fit the model
    grid.fit(X_train, y_train)

    if verbosity > 0:
        print("\n--- RandomizedSearchCV Results ---")
        print("Best parameters found:", grid.best_params_)
        # Use a more descriptive term than "score_aggregation"
        print(f"Best cross-validated {scoring}: {grid.best_score_:.4f} (average score)")

    return grid
