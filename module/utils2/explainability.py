import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import precision_recall_curve, average_precision_score

from pathlib import Path
import joblib
from datetime import datetime
from tqdm import tqdm

import shap

from utils2 import optimization as hpo


palette = {
    'Teal': '#8DD3C7', 
    'Yellow': '#FFFFB3', 
    'Lavender': '#BEBADA', 
    'Coral': '#FB8072', 
    'Blue': '#80B1D3', 
    'Orange': '#FDB462', 
    'Green': '#B3DE69', 
    'Pink': '#FCCDE5', 
    'Purple': '#BC80BD', 
    'Gray': '#D9D9D9', 
    'Red': '#E41A1C' 
}

COLOR_GROUP_MAP = {
    # 'Nerve Conduction Studies': palette['Blue'],
    'Sudoscan': palette['Orange'],
    'Profile': palette['Teal'],
    'Comorbidities': palette['Pink'],
    'Neurology Examination': palette['Blue'],
    'MNSI': palette['Green'],
    # 'Others': palette['Gray']
}


def get_colors(DPN_data, labels):
    D = DPN_data
    return [
        COLOR_GROUP_MAP['Nerve Conduction Studies']  if label in D.ncs_cols else 
        COLOR_GROUP_MAP['Sudoscan'] if label in D.sudo_cols else 
        COLOR_GROUP_MAP['Profile'] if label in D.profile_cols else 
        COLOR_GROUP_MAP['Comorbidities'] if label in D.comorbidity_cols else # Assuming 'Red' is the intended color
        COLOR_GROUP_MAP['Neurology Examination'] if label in D.neuro_cols else 
        COLOR_GROUP_MAP['MNSI'] if label in D.mnsi_col else 
        palette['Gray']
        for label in labels
    ]


def get_ksplit_trained_models(
        X, y, config, *,
        savedir: Path,
        overwrite: bool = False, 
    ):

    ksplit_trained_models_filename = savedir / f"{config.model.code}_ksplit_trained_models.joblib"
    if not overwrite and ksplit_trained_models_filename.is_file():
        print(f'{ksplit_trained_models_filename.name} exists. Returning values from contents.')
        ksplit_trained_models = joblib.load(ksplit_trained_models_filename)
        return ksplit_trained_models

    catboost_model = CatBoostClassifier(
        verbose=0,
        loss_function="Logloss",
        eval_metric="F1",
        random_state=config.experiment.random_seed, 
        thread_count=-1
    )    

    param_space = {
        'depth': Integer(
            config.param_space.depth.min, 
            config.param_space.depth.max),
        'learning_rate': Real(
            config.param_space.learning_rate.min, 
            config.param_space.learning_rate.max, 
            prior='log-uniform'),
        'l2_leaf_reg': Real(
            config.param_space.l2_leaf_reg.min, 
            config.param_space.l2_leaf_reg.max, 
            prior='uniform'),
        "scale_pos_weight": Real(
            config.param_space.scale_pos_weight.min, 
            config.param_space.scale_pos_weight.max,
            prior='uniform'),
        'iterations': Categorical([config.param_space.iterations]),
        "early_stopping_rounds": Categorical([config.param_space.early_stopping_rounds]),
        "verbose": Categorical([0]),        
    }    

    skf = StratifiedKFold(
        n_splits=config.optimization.k_splits_outer, 
        shuffle=True, 
        random_state=config.experiment.random_seed)

    split_results = []

    for train_idx, test_idx in tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="K-Fold"):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_model, best_params, best_threshold = hpo.train_final_model(
            X_train.values, 
            y_train.values, 
            config,
            model=catboost_model,
            param_space=param_space,
            n_jobs=1
        )

        # no need to retrain model since refit=true in train_final_model
        cm, metrics= hpo.test_model(best_model, best_threshold, X_test, y_test)

        result = {
            "model" : best_model,
            "threshold" : best_threshold,
            "best_params" : best_params,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "cm": cm,
            "metrics" : metrics,
        }
        split_results.append(result)

    # don't get mean to highlight these are separate models
    ksplit_trained_models_metrics_filename = savedir / f"{config.model.code}_ksplit_trained_models_metrics.csv"
    df_ksm = pd.DataFrame([result['metrics'] for result in split_results])
    df_ksm.to_csv(ksplit_trained_models_metrics_filename)

    # create a dictionary for saving results
    rundate = datetime.now().strftime("%Y-%m-%d")
    ksplit_trained_models = {
        "results": split_results,
        "summary": df_ksm,
        "rundate": rundate,
        "tag" : config.experiment.tag
    }          

    joblib.dump(ksplit_trained_models, ksplit_trained_models_filename)    

    return ksplit_trained_models


def plot_importances(DPN_data, model, split_index, feature_names, config, 
                     minimum=None, limit=None, 
                     savedir=None):

    D = DPN_data
    importances = model.get_feature_importance()
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    if minimum:
        feature_importances = feature_importances[feature_importances>minimum]
    if limit:
        feature_importances = feature_importances[:limit]

    plt.figure(figsize=(10,8))
    ax = feature_importances.plot(
        kind='barh',
        color = get_colors(D, feature_importances.index.to_list())
    )

    title = f"{config.model.name} Feature Importances"
    if limit:
        title = f'Top {limit} {title}'
    if minimum:
        title = f'{title} (min {minimum:.3f})'
    plt.title(title)
    plt.xlabel("Importance")

    # --- Legend Addition ---
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in COLOR_GROUP_MAP.items()
    ]

    # Add the legend to the plot
    ax.legend(
        handles=legend_handles, 
        title="Feature Groups", 
        loc='best' # Adjust location as needed (e.g., 'best', 'outside')
    )

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_features_importances'
        if limit:
            filename = f'{filename}_top-{limit}'
        if minimum:
            filename = f'{filename}_min-{minimum:.3f}'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()
    plt.close()


def plot_roc_auc(y_test, y_proba, split_index, config, savedir=None):

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance level')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="best")
    plt.grid(True)

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_roc_auc'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()
    plt.close()


def plot_decision_curve_analysis(model, split_index, X, y, config, thresholds=None, savedir=None):
    """
    Perform Decision Curve Analysis (DCA) for a trained classifier.

    Parameters:
    -----------
    model : sklearn-like estimator
        Must have a predict_proba method.
    X : array-like
        Feature matrix (test set).
    y : array-like
        True binary labels.
    thresholds : array-like, optional
        List or array of thresholds to evaluate. Defaults to np.linspace(0.01, 0.99, 50).
    label : str, optional
        Label for the model curve.

    Returns:
    --------
    thresholds : np.array
        Threshold probabilities.
    net_benefits : list
        Net benefit values for the model.
    """

    # Default thresholds
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)

    # Get predicted probabilities
    y_pred_prob = model.predict_proba(X)[:, 1]

    N = len(y)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_pred_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        net_benefit = (tp / N) - (fp / N) * (pt / (1 - pt))
        net_benefits.append(net_benefit)

    # Plotting
    plt.plot(thresholds, net_benefits, label=config.model.name, linewidth=2)
    plt.plot(thresholds, [0]*len(thresholds), linestyle="--", label="Treat None")
    plt.plot(thresholds, thresholds, linestyle="--", label="Treat All")  # Simplified version

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(f"DCA for {config.model.name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_dca'
        plt.savefig(savedir / f'{filename}.png')    
    plt.show()
    plt.close()

    return thresholds, net_benefits


def plot_shap(DPN_data, model, split_index, config, X_test, savedir=None):
    
    D = DPN_data

    def current_model_predict(X):
        """
        Define a simple predict function for the current model
        This closure ensures 'model' refers to the correct model in each iteration
        For classifiers, .predict_proba is often preferred for SHAP for better interpretability
        especially for multi-class.
        """
        return model.predict_proba(X)[:, 1] #.

    # Use a masker: your input DataFrame
    #. Don't use a masker so we preserve correlations
    #. masker = shap.maskers.Independent(X_val)

    # Create explainer with custom predict function
    # Specify the masker for consistency
    #. explainer = shap.Explainer(current_model_predict, masker=masker)

    #. use x_test directly
    explainer = shap.Explainer(current_model_predict, X_test) 

    # Compute SHAP values
    shap_values = explainer(X_test)

    plt.figure(figsize=(8, 8))

    # 3.1 Generate the SHAP summary plot (bar type)
    # We set show=False to prevent Matplotlib from displaying it immediately
    shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", plot_size=(10, 8))

    # 3.2 Get the current Axes object (which contains the plot)
    # This is usually the first (and only) Axes created by the shap plot.
    fig, ax = plt.gcf(), plt.gca()

    # 3.3 Identify the features and assign colors
    # The SHAP bar plot automatically orders features by importance (the Y-axis labels)
    feature_names = [label.get_text() for label in ax.get_yticklabels()]
    bar_colors = get_colors(D, feature_names)

    # 3.4 Manually re-color the bars 
    # The bars are the first container of rectangles in the axes.
    # They are typically stored in ax.containers[0]
    for bar, color in zip(ax.containers[0].patches, bar_colors):
        bar.set_color(color)

    # 3.5 Add Custom Legend
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in COLOR_GROUP_MAP.items()
    ]

    # Modify tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Modify the x-axis label specifically
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=14)

    # Invert the y-axis
    ax.invert_yaxis()
              
    # Set the title first (from your original prompt)
    plt.title("SHAP Values Feature Importance", fontsize=16)

    # Add the legend to the plot
    ax.legend(
        handles=legend_handles, 
        title="Feature Group", 
        loc='upper right', # Adjust location as needed
        bbox_to_anchor=(0.9, 0.9), # Place outside the plot area, for example
        borderaxespad=0.
    )

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_shap'
        plt.savefig(savedir / f'{filename}.png')    

    plt.show()
    plt.close()


def plot_cv_auprc(y_reals, y_probas, config, savedir=None):
    """
    y_reals: list of arrays containing true labels for each fold
    y_probas: list of arrays containing predicted probabilities for each fold
    """
    plt.figure(figsize=(8, 7))
    
    # Store metrics for averaging
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    aucs = []

    for i, (y_real, y_proba) in enumerate(zip(y_reals, y_probas)):
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        pr_auc = average_precision_score(y_real, y_proba)
        aucs.append(pr_auc)
        
        # Plot individual fold
        plt.plot(recall, precision, lw=1, alpha=0.7, label=f'Fold {i} (AP = {pr_auc:.2f})')
        
        # Interpolate precision to allow averaging
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)

    # Calculate and plot Mean
    mean_precision = np.mean(precisions, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_recall, mean_precision, color='b',
            label=f'Mean PR (AP = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
            lw=2, alpha=.8)

    # Plot No-Skill Baseline (average across folds)
    total_pos = sum([sum(y) for y in y_reals])
    total_n = sum([len(y) for y in y_reals])
    plt.axhline(y=total_pos/total_n, color='r', linestyle='--', label='No-Skill Baseline')

    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_auprc'
        plt.savefig(savedir / f'{filename}.png')    
    plt.show()    
    plt.close()