""" Utility Functions For Feature and Model Selection
"""


import numpy as np
import pandas as pd
import seaborn as sns
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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sys 
sys.path.append('..')  

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, make_scorer
)

# variance inflation factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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
        if (verbosity>0):
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


def benchmark_models(X, y, cv_splits, n_repeats, random_state, verbosity, experiment_tag):
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
    if verbosity>0:
        model_items=models.items()
    else: # show progress bar
        model_items=tqdm(models.items())
    for i, (name, model) in enumerate(model_items):
        pipe = build_smart_pipeline(name, model, X, verbosity)
        #scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        scores = cross_validate(pipe, X, y, cv=rcv, scoring=scoring, n_jobs=-1, error_score="raise")
        algo_results = {
            "accuracy": scores["test_accuracy"],
            "precision": scores["test_precision"],
            "sensitivity": scores["test_recall"],
            "specificity": scores["test_specificity"],
            "youden": scores["test_youden"],
            "f1": scores["test_f1"],
            "roc-auc": scores["test_roc_auc"],
        }
        results.append({
            "model": name,
            "rcv_scores" : pd.DataFrame(algo_results)
        })
        if experiment_tag in ['development', 'debug'] and i>1:
            # stop after 3rd algorithm
            break

    # sort by Youden Index instead of ROC-AUC
    return results

metric_fullname = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc-auc": "ROC-AUC",
    "youden": "Youden Index",
    "specificity": "Specificity"
}

def calculate_metric_statistics(experiment_metrics, sorting_metric=None):
    """
    scoring_list: list of metrics to calculate statistics (e.g. youden, roc-auc)
    Calculate statistics (e.g. mean and standard deviation) of repeated cross validation runs for an experiment
    Return: dictionary {
                'mean' : mean/std of all metrics, sorted by  youden 
                'std' : mean/std of all metrics, sorted by  youden
            }
    """
    metric_stats = {}
    metric_stats['mean'] = pd.concat(
        [pd.DataFrame({algo['model']: algo['rcv_scores'].mean()}) for algo in experiment_metrics],
        axis=1
        ).T   
    
    if sorting_metric:
        metric_stats['mean'] = metric_stats['mean'].sort_values(by=sorting_metric, ascending=False)
        column_order =  metric_stats['mean'].columns.to_list()
    
    metric_stats['std'] = pd.concat(
        [pd.DataFrame({algo['model']: algo['rcv_scores'].std()}) for algo in experiment_metrics],
        axis=1
        ).T   

    # reorder according to sorting
    if sorting_metric:
        metric_stats['std'] = metric_stats['std'][column_order]
        
    
    return metric_stats

def get_metric_scores(experiment_metrics, exp_code, metrics_stats, target_metric):
    """
        Get the list of youden scores only
        exp_code : e.g. 'all', 'ncs'
        target_metric : e.g. 'youden', 'roc-auc'
    """    
    dfm = pd.concat(
        [pd.DataFrame({algo['model']: algo['rcv_scores'][target_metric]}) for algo in experiment_metrics[exp_code]],
        axis=1
        )    
    # sort columns by target_metric means
    column_order = metrics_stats[exp_code]['mean'].T.columns.tolist() 
    return dfm[column_order] 


# plot a violin plot
def plot_metric_scores(metric_scores, exp_code, sorted, experiment_tag, target_metric, savedir):
    plt.figure(figsize=(8, 4))
    if sorted:
        metric_scores_sorted = metric_scores[exp_code].mean().to_frame(name="avg").sort_values(by='avg', ascending=False)
        metric_scores_sorted_indices = metric_scores_sorted.index.to_list()
        sns.violinplot(data=metric_scores[exp_code][metric_scores_sorted_indices])
    else:
        sns.violinplot(data=metric_scores[exp_code])
    plt.title(f'Distribution of {metric_fullname[target_metric]} Across Multiple Runs')
    plt.ylabel(metric_fullname[target_metric])
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=75)
    if experiment_tag not in ['debug']:
        plt.savefig(
        savedir / f'{target_metric}_violin.png',
            bbox_inches='tight',
            dpi=300
        )
    plt.show()
    plt.close
    

def get_high_vif(df, vif_threshold, verbosity):
    # Example: df is your DataFrame with predictors
    df2 = df.copy()

    # Add constant term (intercept) for statsmodels
    df2 = add_constant(df2)

    # Compute VIF for each column
    vif_data = pd.DataFrame()
    vif_data["feature"] = df2.columns
    vif_data["VIF"] = [variance_inflation_factor(df2.values, i) 
                    for i in range(df2.shape[1])]
    high_vif = vif_data[vif_data["VIF"]>vif_threshold]
    if verbosity>0:
        print(f'Features with VIF higher than {vif_threshold}')
        print(high_vif)
    return high_vif

def create_model_summary_table(metrics_stats, savedir, target_metric, topk,  
                               exclude_features=[], include_mean=True, show_plot=True, save_fig=True, savename_suffix=""):
    
    metric_table = pd.DataFrame()
    for feature_set_name, df in metrics_stats.items():
        if feature_set_name not in exclude_features:
            metric_table[feature_set_name] = df['mean'][target_metric]
    if topk > 0:
        topk_avg = metric_table.apply(lambda col: col.nlargest(topk).mean())
        metric_table.loc[f"Top {topk} Avg"] = topk_avg
    if include_mean:
        metric_table['mean'] = metric_table.mean(axis=1)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(metric_table, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Average {metric_fullname[target_metric]} Across Models and Feature Sets")
    plt.ylabel("Model")
    plt.xlabel("Feature Set")
    if save_fig:
        savename = f'{target_metric}_summary_table'
        if include_mean:
            savename += f'_mean'
        if topk:
            savename += f'_top{topk}'
        if savename_suffix:
            savename +=   f'_{savename_suffix}'
        savename = savedir / f'{savename}.png'   
        plt.savefig(savename , bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close