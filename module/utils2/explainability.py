import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.colors as mcolors

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

COLOR_GROUP_MAP= {
    'Sudoscan': '#F28E2B',
    'Profile':  '#4E79A7',
    'Comorbidities': '#59A14F',
    'Neurology Examination': '#E15759',
    'MNSI': '#B07AA1',
}

COLOR_GROUP_MAP_ORIG = {
    # 'Nerve Conduction Studies': palette['Blue'],
    'Sudoscan': palette['Orange'],
    'Profile': palette['Teal'],
    'Comorbidities': palette['Pink'],
    'Neurology Examination': palette['Blue'],
    'MNSI': palette['Green'],
    # 'Others': palette['Gray']
}

PLOT_COLORS = ['#009E73', '#0072B2', '#D55E00', '#CC79A7',
              '#000000', '#E69F00',  '#F0E442', '#56B4E9',]

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

    def get_model_paramspace():
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
            'loss_function': Categorical([config.param_space.loss_function]),
            'eval_metric': Categorical([config.param_space.eval_metric]),
            'iterations': Categorical([config.param_space.iterations]),
            "early_stopping_rounds": Categorical([config.param_space.early_stopping_rounds]),
            "verbose": Categorical([0]),        
        }   
        return catboost_model, param_space 

    skf = StratifiedKFold(
        n_splits=config.optimization.k_splits_outer, 
        shuffle=True, 
        random_state=config.experiment.random_seed)

    split_results = []

    for split_idx, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="K-Fold")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        catboost_model, param_space = get_model_paramspace()

        best_model, best_params, best_threshold = hpo.train_final_model(
            X_train.values, 
            y_train.values, 
            config,
            model=catboost_model,
            param_space=param_space,
            split_index=split_idx,
            n_jobs=1
        )

        # no need to retrain model since refit=true in train_final_model
        cm, metrics = hpo.test_model(best_model, best_threshold, X_test, y_test)

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

        # save best parameters to file
        best_params_filename = savedir / f"{config.model.code}_split{split_idx}_best_params.csv"
        df_best_params = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
        df_best_params.to_csv(best_params_filename, index=False)
        


    # don't get mean to highlight these are separate models
    ksplit_trained_models_metrics_filename = savedir / f"{config.model.code}_ksplit_trained_models_metrics.csv"
    df_ksm = pd.DataFrame([result['metrics'] for result in split_results])
    df_ksm.to_csv(ksplit_trained_models_metrics_filename, index_label='split')

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

def plot_importances_heatmap(DPN_data, all_importances, feature_names, config,
                              minimum=None, limit=None,
                              savedir=None):
    D = DPN_data

    # Build DataFrame: rows=features, cols=splits
    df = pd.DataFrame(all_importances, index=feature_names)

    # Filter by mean importance
    df['_mean'] = df.mean(axis=1)
    if minimum:
        df = df[df['_mean'] > minimum]
    df = df.sort_values('_mean', ascending=False)
    if limit:
        df = df.head(limit)
    df = df.drop(columns='_mean')

    # Build title
    title = f"{config.model.name} Feature Importances"
    if limit:
        title = f'Top {limit} {title}'
    if minimum:
        title = f'{title} (min {minimum:.3f})'

    # --- Plot ---
    fig, (ax_legend, ax_heatmap) = plt.subplots(
        1, 2,
        figsize=(10, max(6, len(df) * 0.4)),
        gridspec_kw={'width_ratios': [2, 4]}
    )

    # Heatmap
    sns.heatmap(
        df,
        ax=ax_heatmap,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Importance'},
        yticklabels=True
    )
    ax_heatmap.set_title(title)
    # ax_heatmap.set_xlabel('Split')
    ax_heatmap.set_ylabel('')

    # Color each y-axis tick label by its feature group
    row_colors = get_colors(D, df.index.to_list())
    for tick_label, color in zip(ax_heatmap.get_yticklabels(), row_colors):
        tick_label.set_color(color)
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')
    ax_heatmap.tick_params(axis='y', labelsize=10)

    # Legend panel
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in COLOR_GROUP_MAP.items()
    ]
    ax_legend.legend(handles=legend_handles, title='Feature Groups', loc='center',
                     frameon=False, fontsize=9)
    ax_legend.axis('off')

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_all_splits_feature_importances'
        if limit:
            filename = f'{filename}_top-{limit}'
        if minimum:
            filename = f'{filename}_min-{minimum:.3f}'
        plt.savefig(savedir / f'{filename}.png', bbox_inches='tight')
    plt.show()
    plt.close()    

def plot_importances_heatmap2(DPN_data, all_importances, feature_names, config,
                              minimum=None, limit=None,
                              savedir=None):
    D = DPN_data

    # Build DataFrame: rows=features, cols=splits
    df = pd.DataFrame(all_importances, index=feature_names)

    # Filter by mean importance
    df['_mean'] = df.mean(axis=1)
    if minimum:
        df = df[df['_mean'] > minimum]
    df = df.sort_values('_mean', ascending=False)
    if limit:
        df = df.head(limit)
    df = df.drop(columns='_mean')

    # --- Feature group row colors ---
    row_colors = pd.Series(
        get_colors(D, df.index.to_list()),
        index=df.index
    )

    # Build title
    title = f"{config.model.name} Feature Importances"
    if limit:
        title = f'Top {limit} {title}'
    if minimum:
        title = f'{title} (min {minimum:.3f})'

    # --- Plot ---
    fig, (ax_legend, ax_colors, ax_heatmap) = plt.subplots(
        1, 3,
        figsize=(10, max(6, len(df) * 0.4)),
        gridspec_kw={'width_ratios': [2, 0.2, 4]}
    )

    # Heatmap
    sns.heatmap(
        df,
        ax=ax_heatmap,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Importance'},
        yticklabels=True
    )
    ax_heatmap.set_title(title)
    ax_heatmap.set_xlabel('Split')
    ax_heatmap.set_ylabel('')
    ax_heatmap.tick_params(axis='y', labelsize=9)

    # Feature group color strip (left of heatmap)
    color_matrix = [[mcolors.to_rgba(c)] for c in row_colors]
    ax_colors.imshow(color_matrix, aspect='auto', interpolation='none')
    ax_colors.set_xticks([])
    ax_colors.set_yticks([])
    ax_colors.set_ylabel('')

    # Legend panel
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in COLOR_GROUP_MAP.items()
    ]
    ax_legend.legend(handles=legend_handles, title='Feature Groups', loc='center',
                     frameon=False, fontsize=9)
    ax_legend.axis('off')

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_all_splits_feature_importances'
        if limit:
            filename = f'{filename}_top-{limit}'
        if minimum:
            filename = f'{filename}_min-{minimum:.3f}'
        plt.savefig(savedir / f'{filename}.png', bbox_inches='tight')
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

def plot_roc_auc_overlapping(roc_data, config, savedir=None):
    """
    roc_data: list of (y_test, y_proba) tuples, one per split
    """
    base_fpr = np.linspace(0, 1, 101)  # common x-axis for interpolation
    tprs, aucs = [], []

    plt.figure(figsize=(8, 7))

    for s, (y_test, y_proba) in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPR onto the common FPR grid for averaging
        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, 
                 color=PLOT_COLORS[s % len(PLOT_COLORS)], lw=1.5, alpha=0.6,
                 label=f'Model {s} (AUC = {roc_auc:.2f})')

    # Compute mean and std across splits
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(base_fpr, mean_tpr, color='b', lw=2, alpha=0.8,
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    # plt.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
    #                  color='black', alpha=0.15, label='± 1 std. dev.')

    plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if savedir:
        filename = f'{config.model.code}_all_splits_roc_auc'
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
    X_test_numeric = X_test.astype(float)
    explainer = shap.Explainer(current_model_predict, X_test_numeric) 

    # Compute SHAP values
    shap_values = explainer(X_test_numeric)

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

def collect_shap(DPN_data, model, split_index, config, X_test, all_shap_importances):
    """Computes SHAP values for one split and stores mean absolute values."""

    def current_model_predict(X):
        return model.predict_proba(X)[:, 1]

    X_test_numeric = X_test.astype(float)
    explainer = shap.Explainer(current_model_predict, X_test_numeric)
    shap_values = explainer(X_test_numeric)

    # Mean absolute SHAP value per feature — mirrors feature importance magnitude
    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=X_test.columns
    )
    all_shap_importances[f'Model {split_index}'] = mean_abs_shap


def plot_shap_heatmap(DPN_data, all_shap_importances, config,
                      minimum=None, limit=None,
                      savedir=None):
    D = DPN_data

    # Build DataFrame: rows=features, cols=splits
    df = pd.DataFrame(all_shap_importances)

    # Sort by mean absolute SHAP across splits, apply filters
    df['_mean'] = df.mean(axis=1)
    if minimum:
        df = df[df['_mean'] > minimum]
    df = df.sort_values('_mean', ascending=False)
    if limit:
        df = df.head(limit)
    df = df.drop(columns='_mean')

    # Build title
    title = f"{config.model.name} Mean Absolute SHAP Values"
    if limit:
        title = f'Top {limit} {title}'
    if minimum:
        title = f'{title} (min {minimum:.3f})'

    # --- Plot ---
    fig, (ax_legend, ax_heatmap) = plt.subplots(
        1, 2,
        figsize=(10, max(6, len(df) * 0.4)),
        gridspec_kw={'width_ratios': [2, 4]}
    )

    sns.heatmap(
        df,
        ax=ax_heatmap,
        cmap='viridis',          # we're using absolute value of SHAP values
        annot=True,
        fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'Mean |SHAP value|'},
        yticklabels=True
    )
    ax_heatmap.set_title(title)
    # ax_heatmap.set_xlabel('Split')
    ax_heatmap.set_ylabel('')

    # Color y-axis tick labels by feature group
    row_colors = get_colors(D, df.index.to_list())
    for tick_label, color in zip(ax_heatmap.get_yticklabels(), row_colors):
        tick_label.set_color(color)
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')
    ax_heatmap.tick_params(axis='y', labelsize=10)

    # Legend panel
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in COLOR_GROUP_MAP.items()
    ]
    ax_legend.legend(handles=legend_handles, title='Feature Groups', loc='center',
                     frameon=False, fontsize=9)
    ax_legend.axis('off')

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_all_splits_shap'
        if limit:
            filename = f'{filename}_top-{limit}'
        if minimum:
            filename = f'{filename}_min-{minimum:.3f}'
        plt.savefig(savedir / f'{filename}.png', bbox_inches='tight')
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
        plt.plot(recall, precision, 
                 color=PLOT_COLORS[i % len(PLOT_COLORS)], lw=1.5, alpha=0.6, 
                 label=f'Model {i} (AP = {pr_auc:.2f})')
        
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
        filename = f'{config.model.code}_all_splits_auprc'
        plt.savefig(savedir / f'{filename}.png')    
    plt.show()    
    plt.close()