import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.metrics import roc_curve, auc, auc, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

import matplotlib.gridspec as gridspec
from scipy.stats import beta as beta_dist

from pathlib import Path
import joblib
from datetime import datetime
from tqdm import tqdm
import json

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

# ---------------------------------------------------------------------
# DECISION CURVE ANALYSIS
# ---------------------------------------------------------------------
def plot_decision_curve_analysis(model, model_threshold, split_index, X, y, config, thresholds=None, savedir=None):
    """
    Perform Decision Curve Analysis (DCA) for a trained classifier.

    Parameters:
    -----------
    model : sklearn-like estimator
        Must have a predict_proba method.
    split_index : int or str
        Identifier for the current data split (used in output filename).
    X : array-like
        Feature matrix (test set).
    y : array-like
        True binary labels.
    config : object
        Config object with `model.name` and `model.code` attributes.
    thresholds : array-like, optional
        List or array of thresholds to evaluate. Defaults to np.linspace(0.01, 0.99, 50).
    savedir : Path, optional
        Directory to save the plot to.

    Returns:
    --------
    thresholds : np.array
        Threshold probabilities.
    net_benefits : list
        Net benefit values for the model.
    """

    # Default thresholds
    if thresholds is None:
        thresholds =  np.arange(0.005, 1.0, 0.005) #np.linspace(0.01, 0.99, 50)
    else:
        thresholds = np.asarray(thresholds)
        if np.any(thresholds >= 1.0):
            raise ValueError("Thresholds must be < 1.0 to avoid division by zero in net benefit calculation.")

    # Get predicted probabilities
    y_pred_prob = model.predict_proba(X)[:, 1]

    N = len(y)
    prevalence = np.sum(y) / N

    net_benefits = []
    treat_all_net_benefits = []
    
    def compute_net_benefit(thresh):
        """ compute net benefit at a certain threshold """
        y_pred = (y_pred_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        # Net benefit of the model at this threshold
        net_benefit = (tp / N) - (fp / N) * (thresh / (1 - thresh))
        return net_benefit

    for pt in thresholds:
        net_benefit = compute_net_benefit(pt)
        net_benefits.append(net_benefit)

        # Net benefit of treating everyone at this threshold
        treat_all_nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        treat_all_net_benefits.append(treat_all_nb)
        

    # Plotting
    plt.plot(thresholds, net_benefits, label=config.model.name, linewidth=2)
    plt.plot(thresholds, [0] * len(thresholds), linestyle="--", label="Treat None")
    plt.plot(thresholds, treat_all_net_benefits, linestyle="--", label="Treat All")
    model_nb = compute_net_benefit(model_threshold)
    print(model_nb)
    plt.annotate(
        f'({model_threshold:.3f}, {model_nb:.3f})', 
        xy=(model_threshold, model_nb),  
        xytext=(model_threshold, model_nb+0.15),  
        ha="center",                    # Horizontally center-aligns the text over the point
        va="bottom",                    # Vertically aligns the bottom of the box above the offset
        bbox=dict(
            boxstyle="round,pad=0.4", 
            fc="#F0F0F0", 
            ec="gray", 
            lw=1
        ),
        arrowprops=dict(color="gray", 
                        linewidth=1, 
                        arrowstyle="-|>,head_length=0.2,head_width=0.1",
                        shrinkA=0,# Forces arrow tail to touch the text box
                        shrinkB=0) # Forces arrowhead to perfectly touch the chart point), 
    )
    plt.ylim(-1,1)    

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

    # save values as CSV
    dfnb = pd.DataFrame({
        'threshold' : thresholds,
        'treat_all' : treat_all_net_benefits,
        'net_benefits' : net_benefits
    })
    dfnb.to_csv(savedir / f'{config.model.code}_split{split_index}_nb.csv')

    return thresholds, net_benefits

# ---------------------------------------------------------------------
# CALIBRATION PLOT (Reliability Diagram)
# ---------------------------------------------------------------------
def plot_calibration_curve(
        model, split_index, X, y, config, 
        savedir=None, n_bins=10, strategy="uniform",
        color="#1f77b4"):
    """
    Plots predicted probability (binned) vs. observed frequency of positives.

    strategy: 'uniform' (equal-width bins) or 'quantile' (equal-count bins).
              Quantile bins are often preferred with imbalanced clinical data
              so that no bin is empty/sparse.
    Inputs
    -------
        y_true : array-like of shape (n_samples,), binary labels {0, 1}
        y_prob : array-like of shape (n_samples,), predicted probabilities for class 1
                (e.g., from catboost_model.predict_proba(X_test)[:, 1])
    """
    y_true = y
    y_prob = model.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    plt.figure(figsize=(7, 7))

    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Model calibration curve
    plt.plot(prob_pred, prob_true, marker="o", color=color, label=f"Model {split_index}")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency of positives")
    plt.title(f"Calibration Plot with {n_bins} {strategy}-based bins")  
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left")

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_calibration'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()
    plt.close()


"""
Calibration Plot (Reliability Diagram) with Clopper-Pearson Confidence Intervals
for binary classification, optimized for small samples (e.g., n~200).

Inputs expected:
    y_true : array-like of shape (n_samples,), binary labels {0, 1}
    y_prob : array-like of shape (n_samples,), predicted probabilities for class 1
             (e.g., from catboost_model.predict_proba(X_test)[:, 1])
"""

# ---------------------------------------------------------------------
# clopper_pearson_ci - CALIBRATION PLOT HELPER
# ---------------------------------------------------------------------

def clopper_pearson_ci(n_pos, n_total, confidence=0.95):
    """
    Exact Clopper-Pearson binomial confidence interval.

    Parameters
    ----------
    n_pos   : int or array — number of positive outcomes in the bin
    n_total : int or array — total observations in the bin
    confidence : float — e.g. 0.95 for 95% CI

    Returns
    -------
    lo, hi : lower and upper bounds of the CI
    """
    alpha = 1 - confidence
    # Lower bound: 0 when n_pos == 0
    lo = np.where(
        n_pos == 0,
        0.0,
        beta_dist.ppf(alpha / 2, n_pos, n_total - n_pos + 1)
    )
    # Upper bound: 1 when n_pos == n_total
    hi = np.where(
        n_pos == n_total,
        1.0,
        beta_dist.ppf(1 - alpha / 2, n_pos + 1, n_total - n_pos)
    )
    return lo, hi

# ---------------------------------------------------------------------
# compute_calibration_bins - CALIBRATION PLOT HELPER
# ---------------------------------------------------------------------
def compute_calibration_bins(y_true, y_prob, n_bins=5, strategy="quantile"):
    """
    Computes calibration bin statistics manually so that we can recover
    the exact n_total and n_pos per bin — required for confidence intervals.

    sklearn's calibration_curve returns prob_true and prob_pred but not
    the raw bin counts, so we recompute them here using the same binning logic.

    Returns
    -------
    prob_pred  : mean predicted probability per bin (x-axis)
    prob_true  : observed event rate per bin (y-axis)
    n_total    : number of observations per bin
    n_pos      : number of positive outcomes per bin
    ci_lo      : lower 95% CI bound per bin
    ci_hi      : upper 95% CI bound per bin
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if strategy == "quantile":
        # Equal-count bins based on predicted probability quantiles
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    # Ensure bin edges are unique (can collapse with small n or clustered scores)
    bins = np.unique(bins)

    prob_pred = []
    prob_true = []
    n_total_list = []
    n_pos_list = []

    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            # Include the right edge in the last bin
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])

        n = mask.sum()
        if n == 0:
            continue  # skip empty bins

        n_pos = y_true[mask].sum()
        prob_pred.append(y_prob[mask].mean())
        prob_true.append(n_pos / n)
        n_total_list.append(n)
        n_pos_list.append(n_pos)

    prob_pred  = np.array(prob_pred)
    prob_true  = np.array(prob_true)
    n_total    = np.array(n_total_list)
    n_pos      = np.array(n_pos_list)

    ci_lo, ci_hi = clopper_pearson_ci(n_pos, n_total, confidence=0.95)

    return prob_pred, prob_true, n_total, n_pos, ci_lo, ci_hi


# ---------------------------------------------------------------------
# CALIBRATION PLOT WITH CI
# ---------------------------------------------------------------------

def plot_calibration_with_ci(
        model, split_index, X, y, config, 
        savedir=None, n_bins=5, strategy="quantile",
        color="#1f77b4", show_histogram=True):
    """
    Plots a calibration curve with Clopper-Pearson 95% confidence intervals,
    and an optional histogram of predicted probabilities along the bottom.

    Parameters
    ----------
    y_true         : binary outcome array
    y_prob         : predicted probability array
    n_bins         : number of calibration bins (5 recommended for n~200)
    strategy       : 'quantile' (recommended) or 'uniform'
    label          : model name for the legend
    color          : line/marker color
    ax             : matplotlib Axes (optional; creates figure if None)
    show_histogram : if True, adds a rug/histogram below the plot showing
                     score distribution for positives and negatives separately
    """
    y_true = y
    y_prob = model.predict_proba(X)[:, 1]
    label=f"Model {split_index}"
    prob_pred, prob_true, n_total, n_pos, ci_lo, ci_hi = compute_calibration_bins(
        y_true, y_prob,
        n_bins=n_bins, strategy=strategy
    )

    # Asymmetric error bars: distance from observed rate to CI bounds
    err_lo = prob_true - ci_lo
    err_hi = ci_hi - prob_true

    if show_histogram:
        fig = plt.figure(figsize=(7, 7))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
        ax  = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax_hist = None

    # --- Perfect calibration reference line ---
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            linewidth=1.2, label="Perfect calibration", zorder=1)

    # --- Calibration curve with error bars ---
    ax.errorbar(
        prob_pred, prob_true,
        yerr=[err_lo, err_hi],
        fmt="o-",
        color=color,
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        markersize=7,
        linewidth=1.8,
        label=f"{label} (95% CI)",
        zorder=3
    )

    # --- Annotate each point with bin sample size ---
    for xp, yp, n in zip(prob_pred, prob_true, n_total):
        ax.annotate(
            f"n={n}",
            xy=(xp, yp),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            color="dimgray"
        )

    ax.set_ylabel("Observed frequency of positives", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=13, pad=10)

    # --- Optional: histogram of predicted probabilities ---
    if show_histogram and ax_hist is not None:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        ax_hist.hist(y_prob[y_true == 0], bins=25, range=(0, 1),
                     color="#2196F3", alpha=0.6, label="Negative")
        ax_hist.hist(y_prob[y_true == 1], bins=25, range=(0, 1),
                     color="#F44336", alpha=0.6, label="Positive")

        ax_hist.set_xlabel("Predicted probability", fontsize=12)
        ax_hist.set_ylabel("Count", fontsize=10)
        ax_hist.legend(fontsize=8, loc="upper right")
        ax_hist.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_calibration'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()
    plt.close()

    # return ax




# ---------------------------------------------------------------------
# BRIER CURVE
# ---------------------------------------------------------------------
def brier_curve(y_true, y_prob, n_points=101):
    """
    Computes a Brier Curve following Hernandez-Orallo et al.

    Sweeps an "operating condition" c in [0, 1], representing a hypothetical
    proportion of the positive class (equivalently, a cost-proportion under
    cost-sensitive reasoning). At each c, computes the expected quadratic
    (Brier-type) loss of the classifier's score distribution, reweighted to
    reflect that operating condition.

    Inputs
    -------
        y_true : array-like of shape (n_samples,), binary labels {0, 1}
        y_prob : array-like of shape (n_samples,), predicted probabilities for class 1
                (e.g., from catboost_model.predict_proba(X_test)[:, 1])

    Returns
    -------
    c_values : array of operating conditions (x-axis)
    loss_values : expected loss at each operating condition (y-axis)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    c_values = np.linspace(0, 1, n_points)
    loss_values = np.zeros(n_points)

    for i, c in enumerate(c_values):
        # Reweight the contribution of positive vs negative cases according
        # to the hypothetical operating condition c, instead of using the
        # empirical class balance directly.
        if n_pos > 0:
            loss_pos = np.mean((1 - pos_scores) ** 2) * c
        else:
            loss_pos = 0.0

        if n_neg > 0:
            loss_neg = np.mean(neg_scores ** 2) * (1 - c)
        else:
            loss_neg = 0.0

        loss_values[i] = loss_pos + loss_neg

    return c_values, loss_values


def plot_brier_curve(
        model, split_index, X, y, config, 
        savedir=None, n_points=101, 
        color="#d62728"):
    """
    Plots the Brier Curve: expected loss vs. operating condition.
    Also reports the area under the curve (analogous to AUC, lower is better).
    """
    y_true = y
    y_prob = model.predict_proba(X)[:, 1]

    c_values, loss_values = brier_curve(y_true, y_prob, n_points=n_points)
    auc_brier = np.trapz(loss_values, c_values)

    plt.figure(figsize=(7, 7))

    plt.plot(c_values, loss_values, color=color,
             label=f"Model {split_index} (AUC = {auc_brier:.3f})")
    plt.fill_between(c_values, loss_values, alpha=0.1, color=color)

    plt.xlabel("Operating condition (assumed proportion of positives)")
    plt.ylabel("Expected loss (Brier-type)")
    plt.title("Brier Curve")
    plt.xlim([0, 1])
    plt.ylim(bottom=0)
    plt.legend(loc="upper right")

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_brier'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()
    plt.close()


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


# =============================================================================
# HELPER FUNCTIONS FOR POOLED MODEL PERFORMANCE PLOTS 
# =============================================================================

def pool_fold_predictions(split_results, trained_models):
    """
    Shared utility: pool fold-matched out-of-sample predictions across all CV splits.

    Each patient's probability is generated exclusively by the model fold in which
    that patient served as a test case, preserving out-of-sample integrity.

    Parameters:
    -----------
    split_results : list of dict
        Each dict must contain 'X_test' and 'y_test'.
    trained_models : list
        List of trained sklearn-like estimators with predict_proba.

    Returns:
    --------
    pooled_df : pd.DataFrame
        Columns: y_true, y_pred_prob, fold
    all_probs : np.ndarray
    all_labels : np.ndarray
    prevalence : float
    N : int
    """
    all_probs  = []
    all_labels = []
    all_folds  = []

    for s in range(len(split_results)):
        X_test    = split_results[s]['X_test']
        y_test    = split_results[s]['y_test']
        fold_prob = trained_models[s].predict_proba(X_test)[:, 1]

        all_probs.extend(fold_prob)
        all_labels.extend(y_test)
        all_folds.extend([s] * len(y_test))

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    pooled_df = pd.DataFrame({
        'y_true':      all_labels,
        'y_pred_prob': all_probs,
        'fold':        all_folds
    })

    N          = len(all_labels)
    prevalence = np.sum(all_labels) / N

    return pooled_df, all_probs, all_labels, prevalence, N


def print_distribution_audit(split_results, trained_models, all_probs, all_labels, prevalence, N):
    """
    Shared utility: print fold-level probability distribution audit to console.
    Call once before any of the four plot functions if desired.
    """
    print("=== Fold Probability Distribution Audit ===")
    for s in range(len(split_results)):
        fp = trained_models[s].predict_proba(split_results[s]['X_test'])[:, 1]
        print(f"  Fold {s}: mean={fp.mean():.4f}, std={fp.std():.4f}, "
              f"min={fp.min():.4f}, max={fp.max():.4f}")
    print(f"  Pooled : mean={all_probs.mean():.4f}, std={all_probs.std():.4f}, "
          f"min={all_probs.min():.4f}, max={all_probs.max():.4f}")
    print(f"  Total pooled patients : {N}")
    print(f"  Pooled prevalence     : {prevalence:.4f} "
          f"({int(np.sum(all_labels))}/{N})\n")


# =============================================================================
# POOLED AUROC
# =============================================================================
def plot_pooled_auroc(
    split_results,
    trained_models,
    config,
    pooled_df=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    savedir=None
):
    """
    Plot pooled AUROC curve with bootstrapped confidence interval band.

    Parameters:
    -----------
    split_results : list of dict
        Each dict must contain 'X_test', 'y_test', and 'metrics' (with 'threshold').
    trained_models : list
        Trained sklearn-like estimators (one per fold) with predict_proba.
    config : object
        Config with model.name and model.code.
    pooled_df : pd.DataFrame, optional
        Pre-computed pooled predictions from _pool_fold_predictions().
        Pass this to avoid recomputing probabilities when calling multiple plot functions.
    n_bootstrap : int
        Bootstrap iterations. Default 1000.
    confidence_level : float
        CI level. Default 0.95.
    savedir : Path, optional
        Output directory.

    Returns:
    --------
    pooled_df : pd.DataFrame
    auroc_stats : dict
        auroc, auroc_ci, fold_aurocs, fpr, tpr
    """
    # Pool predictions
    if pooled_df is None:
        pooled_df, all_probs, all_labels, prevalence, N = \
            pool_fold_predictions(split_results, trained_models)
        print_distribution_audit(split_results, trained_models,
                                  all_probs, all_labels, prevalence, N)
    else:
        all_probs  = pooled_df['y_pred_prob'].values
        all_labels = pooled_df['y_true'].values
        N          = len(all_labels)
        prevalence = np.sum(all_labels) / N

    # Pooled ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auroc       = auc(fpr, tpr)

    # Bootstrap CI
    alpha    = 1 - confidence_level
    rng      = np.random.default_rng(seed=42)
    fpr_grid = np.linspace(0, 1, 200)

    boot_auroc      = []
    boot_tpr_curves = []

    for _ in range(n_bootstrap):
        idx      = rng.integers(0, N, size=N)
        b_probs  = all_probs[idx]
        b_labels = all_labels[idx]
        if len(np.unique(b_labels)) < 2:
            continue
        boot_auroc.append(roc_auc_score(b_labels, b_probs))
        b_fpr, b_tpr, _ = roc_curve(b_labels, b_probs)
        boot_tpr_curves.append(np.interp(fpr_grid, b_fpr, b_tpr))

    boot_auroc      = np.array(boot_auroc)
    boot_tpr_curves = np.array(boot_tpr_curves)

    auroc_ci  = (
        np.percentile(boot_auroc, alpha / 2 * 100),
        np.percentile(boot_auroc, (1 - alpha / 2) * 100)
    )
    tpr_lower = np.percentile(boot_tpr_curves, alpha / 2 * 100,       axis=0)
    tpr_upper = np.percentile(boot_tpr_curves, (1 - alpha / 2) * 100, axis=0)

    # Per-fold scalars for annotation
    fold_aurocs = []
    for s in range(len(split_results)):
        y_t = np.array(split_results[s]['y_test'])
        fp  = trained_models[s].predict_proba(split_results[s]['X_test'])[:, 1]
        fold_aurocs.append(roc_auc_score(y_t, fp))

    print(f"=== AUROC ===")
    print(f"  Pooled : {auroc:.4f}  [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
    for s, fa in enumerate(fold_aurocs):
        print(f"  Fold {s}: {fa:.4f}")
    print(f"  Mean   : {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.fill_between(fpr_grid, tpr_lower, tpr_upper,
                    alpha=0.20, color='steelblue',
                    label=f'{confidence_level:.0%} CI '
                          f'(bootstrap')
    ax.plot(fpr, tpr,
            color='steelblue', linewidth=2,
            label=f'{config.model.name} (pooled)')
    ax.plot([0, 1], [0, 1],
            linestyle='--', color='gray', linewidth=1.5,
            label='Random classifier')

    stats_text = (
        # f"Prevalence : {prevalence:.3f} %\n"
        f"AUROC      : {auroc:.3f} [{auroc_ci[0]:.3f} – {auroc_ci[1]:.3f}]\n"
        f"Mean & Std : {np.mean(fold_aurocs):.3f} ± {np.std(fold_aurocs):.3f}"
    )
    ax.text(
            0.47, 0.15, 
            stats_text,transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F0', ec='gray', lw=0),
            family='monospace'
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)',      fontsize=11)
    ax.set_title(
        f'Receiver Operating Characteristic (pooled)',
        fontsize=10
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir / f'{config.model.code}_pooled_auroc.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Save CSV
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    if savedir:
        roc_df.to_csv(savedir / f'{config.model.code}_pooled_auroc.csv', index=False)

    auroc_stats = {
        'auroc':       auroc,
        'auroc_ci':    auroc_ci,
        'fold_aurocs': fold_aurocs,
        'fpr':         fpr,
        'tpr':         tpr
    }
    return pooled_df, auroc_stats


# =============================================================================
# FIGURE 2 — AUPRC
# =============================================================================
def plot_pooled_auprc(
    split_results,
    trained_models,
    config,
    pooled_df=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    savedir=None
):
    """
    Plot pooled Precision-Recall curve with bootstrapped confidence interval band.

    Baseline reference is plotted at the positive class prevalence (not the diagonal),
    which is the correct random-classifier baseline for imbalanced datasets.

    Parameters:
    -----------
    split_results : list of dict
    trained_models : list
    config : object
    pooled_df : pd.DataFrame, optional
    n_bootstrap : int
    confidence_level : float
    savedir : Path, optional

    Returns:
    --------
    pooled_df : pd.DataFrame
    auprc_stats : dict
        auprc, auprc_ci, fold_auprcs, precision, recall, prevalence
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if pooled_df is None:
        pooled_df, all_probs, all_labels, prevalence, N = \
            pool_fold_predictions(split_results, trained_models)
        print_distribution_audit(split_results, trained_models,
                                  all_probs, all_labels, prevalence, N)
    else:
        all_probs  = pooled_df['y_pred_prob'].values
        all_labels = pooled_df['y_true'].values
        N          = len(all_labels)
        prevalence = np.sum(all_labels) / N

    # Pooled PRC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auprc                = average_precision_score(all_labels, all_probs)

    # Bootstrap CI
    alpha        = 1 - confidence_level
    rng          = np.random.default_rng(seed=42)
    recall_grid  = np.linspace(0, 1, 200)

    boot_auprc            = []
    boot_precision_curves = []

    for _ in range(n_bootstrap):
        idx      = rng.integers(0, N, size=N)
        b_probs  = all_probs[idx]
        b_labels = all_labels[idx]
        if len(np.unique(b_labels)) < 2:
            continue
        boot_auprc.append(average_precision_score(b_labels, b_probs))
        b_prec, b_rec, _ = precision_recall_curve(b_labels, b_probs)
        boot_precision_curves.append(
            np.interp(recall_grid, b_rec[::-1], b_prec[::-1])
        )

    boot_auprc            = np.array(boot_auprc)
    boot_precision_curves = np.array(boot_precision_curves)

    auprc_ci = (
        np.percentile(boot_auprc, alpha / 2 * 100),
        np.percentile(boot_auprc, (1 - alpha / 2) * 100)
    )
    precision_lower = np.percentile(boot_precision_curves, alpha / 2 * 100,       axis=0)
    precision_upper = np.percentile(boot_precision_curves, (1 - alpha / 2) * 100, axis=0)

    # Per-fold scalars
    fold_auprcs = []
    for s in range(len(split_results)):
        y_t = np.array(split_results[s]['y_test'])
        fp  = trained_models[s].predict_proba(split_results[s]['X_test'])[:, 1]
        fold_auprcs.append(average_precision_score(y_t, fp))

    print(f"=== AUPRC ===")
    print(f"  Pooled   : {auprc:.4f}  [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]")
    print(f"  Baseline : {prevalence:.4f} (positive class prevalence)")
    for s, fa in enumerate(fold_auprcs):
        print(f"  Fold {s}  : {fa:.4f}")
    print(f"  Mean     : {np.mean(fold_auprcs):.4f} ± {np.std(fold_auprcs):.4f}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.fill_between(recall_grid, precision_lower, precision_upper,
                    alpha=0.20, color='darkorange',
                    label=f'{confidence_level:.0%} CI '
                          f'(bootstrap)')
    ax.plot(recall, precision,
            color='darkorange', linewidth=2,
            label=f'{config.model.name}')
    ax.axhline(y=prevalence,
               linestyle='--', color='gray', linewidth=1.5,
               label=f'Random classifier (prevalence = {prevalence:.2f})')

    stats_text = (
        f"AUPRC      : {auprc:.3f} [{auprc_ci[0]:.3f}–{auprc_ci[1]:.3f}]\n"
        f"Mean & Std : {np.mean(fold_auprcs):.3f} ± {np.std(fold_auprcs):.3f}"
        # f"Baseline : {prevalence:.3f} (prevalence)"
    )
    ax.text(
            0.5, 0.15,  
            stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F0', ec='gray', lw=0),
            family='monospace')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Recall (Sensitivity)',  fontsize=11)
    ax.set_ylabel('Precision (PPV)',  fontsize=11)
    ax.set_title(
        f'Precision-Recall Curve',
        fontsize=10
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir / f'{config.model.code}_pooled_auprc.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Save CSV
    prc_df = pd.DataFrame({'recall': recall[:-1], 'precision': precision[:-1]})
    if savedir:
        prc_df.to_csv(savedir / f'{config.model.code}_pooled_auprc.csv', index=False)

    auprc_stats = {
        'auprc':       auprc,
        'auprc_ci':    auprc_ci,
        'fold_auprcs': fold_auprcs,
        'precision':   precision,
        'recall':      recall,
        'prevalence':  prevalence
    }
    return pooled_df, auprc_stats


# =============================================================================
# FIGURE 3 — CALIBRATION
# =============================================================================
def plot_pooled_calibration_curve(
    split_results,
    trained_models,
    config,
    pooled_df=None,
    n_bins=10,
    n_bootstrap=1000,
    confidence_level=0.95,
    savedir=None
):
    """
    Plot pooled calibration (reliability) curve with bootstrapped CI band,
    ECE, MCE, and Brier score annotations.

    Parameters:
    -----------
    split_results : list of dict
    trained_models : list
    config : object
    pooled_df : pd.DataFrame, optional
    n_bins : int
        Reliability diagram bins. Default 10.
    n_bootstrap : int
    confidence_level : float
    savedir : Path, optional

    Returns:
    --------
    pooled_df : pd.DataFrame
    calibration_stats : dict
        ECE, MCE, brier_score, ece_ci, brier_ci,
        fraction_of_positives, mean_predicted_value
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if pooled_df is None:
        pooled_df, all_probs, all_labels, prevalence, N = \
            pool_fold_predictions(split_results, trained_models)
        print_distribution_audit(split_results, trained_models,
                                  all_probs, all_labels, prevalence, N)
    else:
        all_probs  = pooled_df['y_pred_prob'].values
        all_labels = pooled_df['y_true'].values
        N          = len(all_labels)
        prevalence = np.sum(all_labels) / N

    # Pooled calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        all_labels, all_probs, n_bins=n_bins, strategy='uniform'
    )

    # Scalar metrics
    def compute_ece(probs, labels, n_bins):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(probs)) * abs(labels[mask].mean() - probs[mask].mean())
        return ece

    def compute_mce(probs, labels, n_bins):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            mce = max(mce, abs(labels[mask].mean() - probs[mask].mean()))
        return mce

    ece         = compute_ece(all_probs, all_labels, n_bins)
    mce         = compute_mce(all_probs, all_labels, n_bins)
    brier_score = brier_score_loss(all_labels, all_probs)

    # Bootstrap CI
    alpha       = 1 - confidence_level
    rng         = np.random.default_rng(seed=42)
    boot_ece    = []
    boot_brier  = []
    boot_curves = []

    for _ in range(n_bootstrap):
        idx      = rng.integers(0, N, size=N)
        b_probs  = all_probs[idx]
        b_labels = all_labels[idx]
        boot_ece.append(compute_ece(b_probs, b_labels, n_bins))
        boot_brier.append(brier_score_loss(b_labels, b_probs))
        try:
            fop, mpv = calibration_curve(b_labels, b_probs,
                                         n_bins=n_bins, strategy='uniform')
            boot_curves.append(
                np.interp(mean_predicted_value, mpv, fop,
                          left=np.nan, right=np.nan)
            )
        except Exception:
            pass

    boot_ece    = np.array(boot_ece)
    boot_brier  = np.array(boot_brier)
    boot_curves = np.array(boot_curves)

    ece_ci   = (np.nanpercentile(boot_ece,   alpha / 2 * 100),
                np.nanpercentile(boot_ece,   (1 - alpha / 2) * 100))
    brier_ci = (np.nanpercentile(boot_brier, alpha / 2 * 100),
                np.nanpercentile(boot_brier, (1 - alpha / 2) * 100))

    curve_lower = np.nanpercentile(boot_curves, alpha / 2 * 100,       axis=0)
    curve_upper = np.nanpercentile(boot_curves, (1 - alpha / 2) * 100, axis=0)

    print(f"=== Calibration Metrics ===")
    print(f"  ECE         : {ece:.4f}  [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]")
    print(f"  MCE         : {mce:.4f}")
    print(f"  Brier Score : {brier_score:.4f}  [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]\n")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1],
            linestyle='--', color='gray', linewidth=1.5,
            label='Perfect calibration')

    valid = ~(np.isnan(curve_lower) | np.isnan(curve_upper))
    if valid.any():
        ax.fill_between(mean_predicted_value[valid],
                        curve_lower[valid], curve_upper[valid],
                        alpha=0.20, color='steelblue',
                        label=f'{confidence_level:.0%} CI '
                              f'(bootstrap)')

    ax.plot(mean_predicted_value, fraction_of_positives,
            marker='o', linewidth=2, color='steelblue',
            markersize=6, markerfacecolor='white', markeredgewidth=2,
            label=f'{config.model.name}')

    # Rug plot
    ax.plot(all_probs[all_labels == 1],
            np.full(int(np.sum(all_labels == 1)), -0.03),
            '|', color='steelblue', alpha=0.4, markersize=8,
            label='DPN positive')
    ax.plot(all_probs[all_labels == 0],
            np.full(int(np.sum(all_labels == 0)), -0.06),
            '|', color='firebrick', alpha=0.4, markersize=8,
            label='DPN negative')

    stats_text = (
        f"ECE  : {ece:.3f} [{ece_ci[0]:.3f}–{ece_ci[1]:.3f}]\n"
        f"MCE  : {mce:.3f}\n"
        f"Brier: {brier_score:.3f} [{brier_ci[0]:.3f}–{brier_ci[1]:.3f}]"
    )
    ax.text(0.02, 0.78, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F0', ec='gray', lw=0),
            family='monospace')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.10, 1.05)
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives',      fontsize=11)
    ax.set_title(
        f'Calibration Curve',
        fontsize=10
    )
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir / f'{config.model.code}_pooled_calibration.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Save CSV
    cal_df = pd.DataFrame({
        'mean_predicted_value':  mean_predicted_value,
        'fraction_of_positives': fraction_of_positives,
        'ci_lower':              curve_lower,
        'ci_upper':              curve_upper
    })
    if savedir:
        cal_df.to_csv(savedir / f'{config.model.code}_pooled_calibration.csv',
                      index=False)

    calibration_stats = {
        'ECE':                   ece,
        'MCE':                   mce,
        'brier_score':           brier_score,
        'ece_ci':                ece_ci,
        'brier_ci':              brier_ci,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value':  mean_predicted_value
    }
    return pooled_df, calibration_stats


# =============================================================================
# FIGURE 4 — DECISION CURVE ANALYSIS
# =============================================================================
def plot_pooled_decision_curve_analysis(
    split_results,
    trained_models,
    config,
    pooled_df=None,
    thresholds=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    clinical_threshold_range=(0.30, 0.60),
    savedir=None
):
    from sklearn.metrics import confusion_matrix

    if pooled_df is None:
        pooled_df, all_probs, all_labels, prevalence, N = \
            _pool_fold_predictions(split_results, trained_models)
        _print_distribution_audit(split_results, trained_models,
                                  all_probs, all_labels, prevalence, N)
    else:
        all_probs  = pooled_df['y_pred_prob'].values
        all_labels = pooled_df['y_true'].values
        N          = len(all_labels)
        prevalence = np.mean(all_labels)

    if thresholds is None:
        thresholds = np.arange(0.005, 1.0, 0.005)
    thresholds = np.asarray(thresholds)

    # --- Vectorised net benefit computation ---
    def compute_nb_vectorised(probs, labels, thresh_arr):
        n          = len(labels)
        pred_mat   = (probs[np.newaxis, :] >= thresh_arr[:, np.newaxis])
        tp         = (pred_mat &  labels[np.newaxis, :].astype(bool)).sum(axis=1)
        fp         = (pred_mat & ~labels[np.newaxis, :].astype(bool)).sum(axis=1)
        nb         = (tp / n) - (fp / n) * (thresh_arr / (1 - thresh_arr))
        treat_all  = np.mean(labels) - (1 - np.mean(labels)) * \
                     (thresh_arr / (1 - thresh_arr))
        return nb, treat_all

    net_benefits, treat_all_nb = compute_nb_vectorised(
        all_probs, all_labels, thresholds
    )[:2]

    # --- Bootstrap CI on model net benefit curve ---
    alpha          = 1 - confidence_level
    rng            = np.random.default_rng(seed=42)
    boot_nb_curves = []

    for _ in range(n_bootstrap):
        idx    = rng.integers(0, N, size=N)
        b_nb, _ = compute_nb_vectorised(
            all_probs[idx], all_labels[idx], thresholds
        )
        boot_nb_curves.append(b_nb)

    boot_nb_curves = np.array(boot_nb_curves)
    nb_lower = np.percentile(boot_nb_curves, alpha / 2 * 100,       axis=0)
    nb_upper = np.percentile(boot_nb_curves, (1 - alpha / 2) * 100, axis=0)

    # --- Fold thresholds ---
    fold_thresholds = [split_results[s]['metrics']['threshold']
                       for s in range(len(split_results))]
    mean_threshold  = np.mean(fold_thresholds)
    mean_nb_idx     = np.argmin(np.abs(thresholds - mean_threshold))
    mean_nb         = net_benefits[mean_nb_idx]

    print(f"=== DCA ===")
    print(f"  Mean threshold               : {mean_threshold:.4f}")
    print(f"  Net benefit at mean threshold: {mean_nb:.4f}")
    for s, ft in enumerate(fold_thresholds):
        idx_ft = np.argmin(np.abs(thresholds - ft))
        print(f"  Fold {s}: threshold={ft:.4f}, "
              f"net benefit={net_benefits[idx_ft]:.4f}")
    print()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Clinically relevant threshold shading
    ax.axvspan(*clinical_threshold_range,
               alpha=0.09, color='orange',
               label=f'Clinically relevant range '
                     f'({clinical_threshold_range[0]}–'
                     f'{clinical_threshold_range[1]})')

    # Zero net benefit reference
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

    # CI band
    ax.fill_between(thresholds, nb_lower, nb_upper,
                    alpha=0.12, color='steelblue',
                    label=f'{confidence_level:.0%} CI '
                          f'(bootstrap)')

    # Model curve
    ax.plot(thresholds, net_benefits,
            color='steelblue', linewidth=2,
            label=f'{config.model.name}')

    # Reference lines
    ax.plot(thresholds, [0] * len(thresholds),
            linestyle='--', color='gray', linewidth=1.5,
            label='Treat None')
    ax.plot(thresholds, treat_all_nb,
            linestyle='--', color='firebrick', linewidth=1.5,
            label='Treat All')

    # Mean threshold annotation
    # ax.annotate(
    #     f'Mean threshold\n({mean_threshold:.3f}, {mean_nb:.3f})',
    #     xy=(mean_threshold, mean_nb),
    #     xytext=(mean_threshold, mean_nb + 0.12),
    #     ha='center', va='bottom',
    #     bbox=dict(boxstyle='round,pad=0.4', fc='#F0F0F0', ec='gray', lw=1),
    #     arrowprops=dict(color='gray', linewidth=1,
    #                     arrowstyle='-|>,head_length=0.2,head_width=0.1',
    #                     shrinkA=0, shrinkB=0)
    # )

    # Fold threshold summary (replaces individual tick marks)
    threshold_text = (
        # f"Fold thresholds: "
        # f"{', '.join([f'{ft:.3f}' for ft in fold_thresholds])}\n"
        f"Threshold Mean: {mean_threshold:.3f} ± {np.std(fold_thresholds):.3f}\n"
        f"Net Benefit Mean: {mean_nb:.3f} ± {np.std(net_benefits):.3f}"
    )
    ax.text(0.98, 0.7, threshold_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F0F0F0', ec='gray', lw=0),
            family='monospace')

    ax.set_ylim(-0.1, prevalence + 0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Threshold Probability', fontsize=11)
    ax.set_ylabel('Net Benefit',           fontsize=11)
    ax.set_title(
        f'Decision Curve Analysis',
        fontsize=10
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir / f'{config.model.code}_pooled_dca.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- Save CSV ---
    pd.DataFrame({
        'threshold':   thresholds,
        'net_benefit': net_benefits,
        'treat_all':   treat_all_nb,
        'ci_lower':    nb_lower,
        'ci_upper':    nb_upper
    }).to_csv(savedir / f'{config.model.code}_pooled_dca.csv', index=False)

    return pooled_df, {
        'thresholds':                    thresholds,
        'net_benefits':                  net_benefits,
        'treat_all_net_benefits':        treat_all_nb,
        'nb_lower':                      nb_lower,
        'nb_upper':                      nb_upper,
        'mean_threshold':                mean_threshold,
        'net_benefit_at_mean_threshold': mean_nb,
        'fold_thresholds':               fold_thresholds
    }

def plot_pooled_decision_curve_analysis_referenced(
    split_results,
    trained_models,
    config,
    pooled_df=None,
    thresholds=None,
    savedir=None
):
    """
    Plot pooled Decision Curve Analysis using fold-matched out-of-sample
    predicted probabilities.

    DCA is threshold-agnostic by design — it sweeps across all threshold
    values — making inter-model threshold differences immaterial to pooling.

    Parameters:
    -----------
    split_results : list of dict
        Each dict must contain 'X_test', 'y_test', and
        'metrics' (dict with key 'threshold').
    trained_models : list
    config : object
    pooled_df : pd.DataFrame, optional
    thresholds : array-like, optional
        Defaults to np.arange(0.005, 1.0, 0.005).
    savedir : Path, optional

    Returns:
    --------
    pooled_df : pd.DataFrame
    dca_stats : dict
        thresholds, net_benefits, treat_all_net_benefits,
        mean_threshold, net_benefit_at_mean_threshold, fold_thresholds
    """
    from sklearn.metrics import confusion_matrix

    if pooled_df is None:
        pooled_df, all_probs, all_labels, prevalence, N = \
            _pool_fold_predictions(split_results, trained_models)
        _print_distribution_audit(split_results, trained_models,
                                  all_probs, all_labels, prevalence, N)
    else:
        all_probs  = pooled_df['y_pred_prob'].values
        all_labels = pooled_df['y_true'].values
        N          = len(all_labels)
        prevalence = np.sum(all_labels) / N

    if thresholds is None:
        thresholds = np.arange(0.005, 1.0, 0.005)
    else:
        thresholds = np.asarray(thresholds)
        if np.any(thresholds >= 1.0):
            raise ValueError("Thresholds must be < 1.0.")

    # Net benefit computation
    def compute_net_benefit(probs, labels, n, thresh):
        y_pred = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
        return (tp / n) - (fp / n) * (thresh / (1 - thresh))

    net_benefits          = []
    treat_all_net_benefits = []

    for pt in thresholds:
        net_benefits.append(compute_net_benefit(all_probs, all_labels, N, pt))
        treat_all_net_benefits.append(
            prevalence - (1 - prevalence) * (pt / (1 - pt))
        )

    # Fold thresholds for annotation
    fold_thresholds = [split_results[s]['metrics']['threshold']
                       for s in range(len(split_results))]
    mean_threshold  = np.mean(fold_thresholds)
    mean_nb         = compute_net_benefit(all_probs, all_labels, N, mean_threshold)

    print(f"=== DCA ===")
    print(f"  Mean fold threshold          : {mean_threshold:.4f}")
    print(f"  Net benefit at mean threshold: {mean_nb:.4f}")
    for s, ft in enumerate(fold_thresholds):
        nb = compute_net_benefit(all_probs, all_labels, N, ft)
        print(f"  Fold {s}: threshold={ft:.4f}, net benefit={nb:.4f}")
    print()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, net_benefits,
            color='steelblue', linewidth=2,
            label=f'{config.model.name} (Pooled, n={N})')
    ax.plot(thresholds, [0] * len(thresholds),
            linestyle='--', color='gray', linewidth=1.5,
            label='Treat None')
    ax.plot(thresholds, treat_all_net_benefits,
            linestyle='--', color='firebrick', linewidth=1.5,
            label='Treat All')

    # Mean threshold annotation
    ax.annotate(
        f'Mean threshold\n({mean_threshold:.3f}, {mean_nb:.3f})',
        xy=(mean_threshold, mean_nb),
        xytext=(mean_threshold, mean_nb + 0.15),
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='#F0F0F0', ec='gray', lw=1),
        arrowprops=dict(color='gray', linewidth=1,
                        arrowstyle='-|>,head_length=0.2,head_width=0.1',
                        shrinkA=0, shrinkB=0)
    )

    # Individual fold threshold tick marks
    for s, ft in enumerate(fold_thresholds):
        nb_at_fold = compute_net_benefit(all_probs, all_labels, N, ft)
        ax.plot(ft, nb_at_fold,
                marker='|', color='steelblue',
                markersize=10, markeredgewidth=1.5,
                label=f'Fold {s} threshold ({ft:.3f})')

    ax.set_ylim(-0.1, prevalence + 0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Threshold Probability', fontsize=11)
    ax.set_ylabel('Net Benefit',           fontsize=11)
    ax.set_title(
        f'Decision Curve Analysis — {config.model.name}\n'
        f'Pooled fold-matched predictions  |  n={N}  |  '
        f'Prevalence {prevalence:.1%}',
        fontsize=10
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir / f'{config.model.code}_pooled_dca.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Save CSV
    dfnb = pd.DataFrame({
        'threshold':    thresholds,
        'treat_all':    treat_all_net_benefits,
        'net_benefit':  net_benefits
    })
    if savedir:
        dfnb.to_csv(savedir / f'{config.model.code}_pooled_dca.csv', index=False)

    dca_stats = {
        'thresholds':                   thresholds,
        'net_benefits':                 net_benefits,
        'treat_all_net_benefits':       treat_all_net_benefits,
        'mean_threshold':               mean_threshold,
        'net_benefit_at_mean_threshold': mean_nb,
        'fold_thresholds':              fold_thresholds
    }
    return pooled_df, dca_stats