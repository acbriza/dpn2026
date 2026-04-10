
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math
import seaborn as sns
from pathlib import Path
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from IPython.display import display

backend = matplotlib.get_backend()


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
    
def test_wrapped_model(model, wrapped_model, X_test, y_test, threshold, verbosity):
    # Evaluation at default threshold (0.5)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    cm_default = confusion_matrix(y_test, y_pred)

    # Evaluation at custom threshold
    y_pred_custom = wrapped_model.predict(X_test)
    y_pred_proba_custom = wrapped_model.predict_proba(X_test)
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    
    if verbosity>0: 
        # get dissimilar predictions
        mask = y_pred_custom != y_pred

        print(f"Confusion Matrix at default threshold (0.5): /n{cm_default}")
        # print(classification_report(y_test, y_pred, target_names=["Confirmed", "non-Confirmed"]))
        print(f"Confusion Matrix at custom threshold ({threshold:.2f}): /n({cm_custom}):")
        # print(classification_report(y_test, y_pred_custom, target_names=["Confirmed", "non-Confirmed"]))

        print('Rows with different predictions at thresholds: ')
        df = X_test[mask].copy()
        df['pred_0.50'] = y_pred[mask]
        df[f'pred_{threshold:.2f}'] = y_pred_custom[mask]
        df['pred_proba_0.50'] = y_pred_proba[mask][:,1] 
        df[f'pred_proba_{threshold:.2f}'] = y_pred_proba_custom[mask][:,1] 
        display(df)
    return 

# GLOBAL COUNTERFACTUALS 
# ----------------------

def get_global_permitted_range(dfXy, continuous_cols, config, split_index, verbosity=0, savedir=None):
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
    if savedir:
        filename = f'{config.model.code}_split{split_index}_global_permitted_range.csv'
        global_permitted_range_df.to_csv(savedir / filename)        
    return global_permitted_range

