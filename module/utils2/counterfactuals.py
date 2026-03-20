
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from IPython.display import display

from utils2 import explainability as exp

actionable_cols = ['HBA1C', 'DSLPDMIA', 'INSULIN']
progressive_cols = ['AGE', 'DM_DUR', 'HPN', 'PAOD', 'CKD', 'GBS', 'DEC_VS', 'DEC_PPS', 'DEC_LTS', 'DEC_AR', 'MNSI']
immutable_cols = ["SEX"]

# Feature list from data loader
# profile_cols = ['SEX', 'AGE', 'SUBJ', 'DM_DUR', 'INSULIN', 'HBA1C']
#     comorbidity_cols = ['HPN', 'PAOD', 'DSLPDMIA', 'CKD', 'GBS']
#     neuro_cols = ['DEC_VS', 'DEC_PPS', 'DEC_LTS', 'DEC_AR']
#     mnsi_col = ['MNSI']
#     ncs_cols = ['SSA_L', 'SSC_L', 'SPSA_L', 'SPSC_L', 'MCV_L', 'DL_L', 'CMAPANK_L', 'CMAPKNE_L', 'FWAVE_L',
#                 'SSA_R', 'SSC_R', 'SPSA_R', 'SPSC_R', 'MCV_R', 'DL_R', 'CMAPANK_R', 'CMAPKNE_R', 'FWAVE_R']
#     sudo_cols = ['FEET_MEAN_ESC', 'FEET_PCT_ASYM', 'HAND_MEAN_ESC', 'HAND_PCT_ASYM', 'NS', 'CAS']

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Thin sklearn-compatible wrapper around CatBoostClassifier that
    applies a configurable probability threshold in predict().
 
    DiCE calls predict_proba() internally to score candidate CFs and
    calls predict() to assign the final class label shown in the output.
    Both methods respect our custom threshold.
    """
 
    def __init__(self, catboost_model: CatBoostClassifier, threshold: float = 0.5):
        self.catboost_model = catboost_model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])   # required by DiCE's sklearn backend
 
    def fit(self, X, y):
        """No-op: the model is already trained."""
        return self
 
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw [P(0), P(1)] probabilities from CatBoost."""
        return self.catboost_model.predict_proba(X)
 
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Apply custom threshold to class-1 probability."""
        proba_1 = self.predict_proba(X)[:, 1]
        return (proba_1 >= self.threshold).astype(int)
    
def test_wrapped_model(model, wrapped_model, X_test, y_test, threshold):
    # Evaluation at default threshold (0.5)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix at default threshold (0.5):")
    print(cm)
    # print(classification_report(y_test, y_pred, target_names=["Confirmed", "non-Confirmed"]))

    # Evaluation at custom threshold
    y_pred_custom = wrapped_model.predict(X_test)
    y_pred_proba_custom = wrapped_model.predict_proba(X_test)
    print(f"Confusion Matrix at custom threshold ({threshold}):")
    cm = confusion_matrix(y_test, y_pred_custom)
    print(cm)
    # print(classification_report(y_test, y_pred_custom, target_names=["Confirmed", "non-Confirmed"]))

    # get dissimilar predictions
    mask = y_pred_custom != y_pred

    print('Rows with different predictions at thresholds: ')
    df = X_test[mask].copy()
    df['pred_0.50'] = y_pred[mask]
    df[f'pred_{threshold:.2f}'] = y_pred_custom[mask]
    df['pred_proba'] = y_pred_proba[mask][:,1] 
    display(df)
    return 

def get_global_permitted_range(dfXy, continuous_cols, verbosity=0):
    global_permitted_range = {}
    for col in continuous_cols: # no need to set range for categorical columns
        stdev = dfXy[col].std()
        minval = dfXy[col].min()
        maxval = dfXy[col].max()
        minval = 0 if minval==0 else max(0, minval-stdev)
        maxval = maxval + stdev
        global_permitted_range[col] = [minval, maxval]

    # create a dataframe for visualization
    if verbosity>0:
        global_permitted_range_df = pd.DataFrame(global_permitted_range).transpose()
        global_permitted_range_df.columns = ['min', 'max']
        display(global_permitted_range_df)
    return global_permitted_range


def plot_global_importance(dice_obj, DPN_data, X_test, split_index, config, 
                           highlight_features=[], total_CFs=10, 
                           title_suffix="", filename_suffix="", savedir=None):
    """
    Parameters:
    dexp: DiCE explainer object
    X_test: test set 
    total_CFs: Number of counterfactuals to generate
    """
    D = DPN_data
    cobj = dice_obj.global_feature_importance(X_test, total_CFs=total_CFs, posthoc_sparsity_param=None)
    df_imp = pd.DataFrame([cobj.summary_importance])

    s = df_imp.iloc[0]
    s_trimmed = s[s>0]

    feature_names = s_trimmed.index.to_list()
    bar_colors = exp.get_colors(D, feature_names) 

    HIGHLIGHT_COLOR = '#0000CD'
    #ACTIONABLE_FEATURES = {'HBA1C', 'DSLPDMIA', 'INSULIN'} 

    fig, ax = plt.subplots(figsize=(10, 8))
    s_trimmed.plot.barh(ax=ax, color=bar_colors)

    # Get all the y-axis tick labels
    y_labels = ax.get_yticklabels()

    # Loop through each label and check if it should be bold
    for label in y_labels:
        feature_name = label.get_text() # Get the text of the label
        
        if feature_name in highlight_features:
            # Set the font properties for bolding
            label.set_fontweight('bold')
            # Optional: Increase size for more emphasis
            # label.set_fontsize(11)
            label.set_color(HIGHLIGHT_COLOR) 

    # Create Custom Legend Handles
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in exp.COLOR_GROUP_MAP.items()
    ]

    # Add the Legend
    ax.legend(
        handles=legend_handles,
        title="Feature Group",
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98), # Place the legend outside the plot area
        borderaxespad=0.1
    )

    ax.set_title(f"Global Importance ({title_suffix})")
    ax.set_ylabel("Category")
    ax.set_xlabel("Value")
    plt.tight_layout()
    if savedir:
        filename = f'{config.model.code}_split{split_index}_global_cf'
        if filename_suffix:
            filename = f'{filename}_{filename_suffix}'
        plt.savefig(savedir / f'{filename}.png')
    plt.show()
