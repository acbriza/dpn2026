import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, confusion_matrix

import numpy as np
import pandas as pd

from pathlib import Path
import shap

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
    'Nerve Conduction Studies': palette['Blue'],
    'Sudoscan': palette['Orange'],
    'Profile': palette['Teal'],
    'Comorbidities': palette['Red'],
    'Neurology Examination': palette['Pink'],
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


def plot_importances(DPN_data, config, importances, feature_names, 
                     minimum=None, limit=None, 
                     savedir=None):

    D = DPN_data
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
        filename = f'{config.model.code}_features_importances'
        if limit:
            filename = f'{filename}_top-{limit}'
        if minimum:
            filename = f'{filename}_min-{minimum:.3f}'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()


def plot_roc_auc(config, y_test, y_proba, savedir=None):

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
        filename = f'{config.model.code}_roc_auc'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()


def plot_decision_curve_analysis(config, model, X, y, thresholds=None, savedir=None):
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
        filename = f'{config.model.code}_dca'
        plt.savefig(savedir / f'{filename}.png')    
    plt.show()
    return thresholds, net_benefits


def plot_shap(DPN_data, config, model, X_test, savedir=None):
    
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
    shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", plot_size=None)

    # 3.2 Get the current Axes object (which contains the plot)
    # This is usually the first (and only) Axes created by the shap plot.
    ax = plt.gca()

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

    # Set the title first (from your original prompt)
    plt.title("SHAP Values Feature Importance")

    # Add the legend to the plot
    ax.legend(
        handles=legend_handles, 
        title="Feature Group", 
        loc='lower right', # Adjust location as needed
        bbox_to_anchor=(1.05, 0.1), # Place outside the plot area, for example
        borderaxespad=0.
    )

    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_shap'
        plt.savefig(savedir / f'{filename}.png')    

    plt.show()
