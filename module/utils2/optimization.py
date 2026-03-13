import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import optuna
import optuna.visualization as vis
optuna.logging.set_verbosity(optuna.logging.WARNING)
from skopt import BayesSearchCV

from scipy import stats

import sys 
sys.path.append('..')  

def nested_cv_youden_optuna(
    X,
    y,
    model_class,
    param_space_fn,
    n_splits_outer=3,
    n_repeats_outer=3,
    n_splits_inner=3,
    n_iter=30,
    random_state=42,
):
    """
    Nested cross-validation with:
    - Outer: RepeatedStratifiedKFold
    - Inner: Optuna TPE study optimizing ROC-AUC
    - Threshold selected on inner-CV OOF predictions using Youden index,
      then applied to the outer test fold (avoids optimistic bias)

    Args:
        X:               Feature matrix (numpy array)
        y:               Target vector (numpy array)
        model_class:     Uninstantiated estimator class (e.g. CatBoostClassifier)
        param_space_fn:  Callable (trial) -> dict of hyperparameters.
                         Receives an optuna.trial.Trial and returns a param dict
                         compatible with model_class(**params).
                         Example:
                             def param_space_fn(trial):
                                 return {
                                     "depth": trial.suggest_int("depth", 4, 10),
                                     "learning_rate": trial.suggest_float(
                                         "learning_rate", 1e-3, 0.3, log=True
                                     ),
                                 }
        n_splits_outer:  Outer CV folds
        n_repeats_outer: Outer CV repeats
        n_splits_inner:  Inner CV folds (used inside each Optuna trial)
        n_iter:          Number of Optuna trials per outer fold
        random_state:    Base random seed

    Returns:
        dict with mean/std summary metrics and per-fold results
    """

    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits_outer,
        n_repeats=n_repeats_outer,
        random_state=random_state,
    )

    fold_results = []

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        # ── Inner CV: materialize splits once ────────────────────────────────
        # Reused by both Optuna objective and the OOF threshold loop so that
        # the threshold is never chosen on data the model was trained on.
        current_seed = random_state + fold_idx

        inner_cv = StratifiedKFold(
            n_splits=n_splits_inner,
            shuffle=True,
            random_state=current_seed,
        )
        inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))

        # ── Optuna objective ─────────────────────────────────────────────────
        def objective(trial):
            params = param_space_fn(trial)

            fold_aucs = []
            for inner_train_idx, inner_val_idx in inner_splits:
                m = model_class(**params, random_state=random_state)
                m.fit(
                    X_outer_train[inner_train_idx],
                    y_outer_train[inner_train_idx],
                    verbose=0,
                )
                val_proba = m.predict_proba(X_outer_train[inner_val_idx])[:, 1]

                if len(np.unique(y_outer_train[inner_val_idx])) > 1:
                    fold_aucs.append(
                        roc_auc_score(y_outer_train[inner_val_idx], val_proba)
                    )

            return float(np.mean(fold_aucs)) if fold_aucs else 0.0

        sampler = optuna.samplers.TPESampler(seed=current_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

        best_params = study.best_params

        # ── Refit best model on full outer training set ───────────────────────
        best_model = model_class(**best_params, random_state=random_state)
        best_model.fit(X_outer_train, y_outer_train, verbose=0)

        # ── Threshold selection on inner-CV OOF predictions ──────────────────
        oof_proba = np.zeros(len(y_outer_train))
        for inner_train_idx, inner_val_idx in inner_splits:
            fold_model = model_class(**best_params, random_state=random_state)
            fold_model.fit(
                X_outer_train[inner_train_idx],
                y_outer_train[inner_train_idx],
                verbose=0,
            )
            oof_proba[inner_val_idx] = fold_model.predict_proba(
                X_outer_train[inner_val_idx]
            )[:, 1]

        fpr, tpr, threshold_candidates = roc_curve(y_outer_train, oof_proba)
        youden_index = tpr - fpr
        best_threshold = float(threshold_candidates[np.argmax(youden_index)])

        # ── Evaluation on outer test fold ─────────────────────────────────────
        y_test_proba = best_model.predict_proba(X_outer_test)[:, 1]
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

        cm = confusion_matrix(y_outer_test, y_test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        youden_test = (
            sensitivity + specificity - 1
            if not (np.isnan(sensitivity) or np.isnan(specificity))
            else np.nan
        )
        roc_auc = (
            roc_auc_score(y_outer_test, y_test_proba)
            if len(np.unique(y_outer_test)) > 1
            else np.nan
        )

        fold_results.append(
            {
                "fold": fold_idx,
                "roc_auc": roc_auc,
                "youden": youden_test,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "threshold": best_threshold,
                "best_params": best_params,
            }
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _nanstats(key):
        vals = [f[key] for f in fold_results]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    roc_mean, roc_std = _nanstats("roc_auc")
    you_mean, you_std = _nanstats("youden")
    sen_mean, _ = _nanstats("sensitivity")
    spe_mean, _ = _nanstats("specificity")
    thr_mean, thr_std = _nanstats("threshold")

    return {
        # ── Summary stats ──
        "roc_auc_mean": roc_mean,
        "roc_auc_std": roc_std,
        "youden_mean": you_mean,
        "youden_std": you_std,
        "sensitivity_mean": sen_mean,
        "specificity_mean": spe_mean,
        "threshold_mean": thr_mean,
        "threshold_std": thr_std,
        # ── Per-fold detail ──
        "folds": fold_results,
    }

model_class = {
    'catboost': CatBoostClassifier,
    'random_forest' : RandomForestClassifier,
}

#### OPTUNA OPTIMIZATION #######

def mean_confidence_interval(results, config):
    confidence = config.evaluation.confidence
    verbosity = config.experiment.verbosity

    opt_results_ci = {}
    for metric in ["youden", "roc_auc"]:
        scores = [fold[metric] for fold in results['folds']]
        
        scores = np.array(scores)
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        stderr = std / np.sqrt(n)

        # z = 1.96 confidence 0.95
        z = stats.norm.ppf((1 + confidence) / 2.)  
        margin = z * stderr

        opt_results_ci[metric] =  {
            "mean": mean,
            "std": std,
            "ci_lower": mean - margin,
            "ci_upper": mean + margin,
            "n_folds": n
        } 

        if verbosity > 0:
            print(f"{metric} {confidence*100}% CI: {opt_results_ci[metric]}")

    return opt_results_ci

def train_final_model_with_threshold_recalculation(X, y, model, param_space, n_splits_inner=3, n_iter=30, random_state=42, n_jobs=-1):
    """
    Train the final deployable model on ALL data:
    1. BayesSearchCV to find best hyperparameters (inner CV on full dataset)
    2. Refit on full dataset with best params
    3. Threshold via Youden index on OOF predictions
    """

    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)

    # Step 1: Hyperparameter search on full data
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        scoring="roc_auc",
        cv=inner_cv,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,  # fits final model on all data with best params
        verbose=0,
    )
    opt.fit(X, y)
    final_model = opt.best_estimator_

    # Step 2: Youden threshold via OOF probabilities on full dataset
    oof_proba = np.zeros(len(y))
    for inner_train_idx, inner_val_idx in inner_cv.split(X, y):
        fold_model = opt.best_estimator_.__class__(**opt.best_params_)
        fold_model.fit(X[inner_train_idx], y[inner_train_idx])
        oof_proba[inner_val_idx] = fold_model.predict_proba(X[inner_val_idx])[:, 1]

    fpr, tpr, thresholds = roc_curve(y, oof_proba)
    best_threshold = float(thresholds[np.argmax(tpr - fpr)])

    return final_model, best_threshold, opt.best_params_

def train_final_model(X, y, model, param_space, n_splits_inner=5, n_iter=50, random_state=42, n_jobs=-1):
    """
    Train the final deployable model on ALL data:
    1. BayesSearchCV to find best hyperparameters (inner CV on full dataset)
    2. Refit on full dataset with best params
    3. No best threshold recalculation; we will use the one from the cross validated folds
    """

    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)

    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        scoring="roc_auc",
        cv=inner_cv,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    opt.fit(X, y, callback=None)
    return opt.best_estimator_, opt.best_params_

def model_predict(X_new, model, threshold):
    proba = model.predict_proba(X_new)[:, 1]
    return (proba >= threshold).astype(int), proba


def test_model(model, threshold, Xnew, ynew, uses_proba=False):
    if uses_proba:
        ypredproba = model.predict(Xnew)
        ypred = (ypredproba > threshold).astype(int)
    else:
        ypred, ypredproba = model_predict(Xnew, model, threshold)

    cm = confusion_matrix(ynew, ypred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    youden_test = (
        sensitivity + specificity - 1
        if not (np.isnan(sensitivity) or np.isnan(specificity))
        else np.nan
    )
    roc_auc = (
        roc_auc_score(ynew, ypredproba)
        if len(np.unique(ynew)) > 1
        else np.nan
    )

    print()
    print(cm)
    print('youden: ', youden_test)
    print('roc_auc: ', roc_auc)
    return youden_test, roc_auc