
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import os
import seaborn as sns

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

# GLOBAL COUNTERFACTUALS 
# ----------------------

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


def plot_global_importance(dice_exp, DPN_data, X_test, split_index, config, 
                           highlight_features=[], total_CFs=10, 
                           title_suffix="", filename_suffix="", savedir=None):
    """
    Parameters:
    dice_exp: DiCE explainer object
    X_test: test set 
    total_CFs: Number of counterfactuals to generate
    """
    D = DPN_data
    cobj = dice_exp.global_feature_importance(X_test, total_CFs=total_CFs, posthoc_sparsity_param=None)
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


# LOCAL COUNTERFACTUALS 
# ----------------------

def get_local_permitted_range(dfXy, instance, allfeature_cols, 
                              categorical_cols, continuous_cols, monotonic_cols):
    local_permitted_range = {}
    for col in allfeature_cols:

        if col in categorical_cols:
            # it does not make sense to set a range for categoricals
            continue
        
        instance_val = instance.iloc[0][col]
        col_stdev = dfXy[col].std()
        col_min = dfXy[col].min()
        col_max = dfXy[col].max()

        if col in monotonic_cols: # true for categoricals and continuous
            minval = instance_val 
        elif col in continuous_cols:
            minval = 0 if instance_val==0 else max(0, instance_val-col_stdev)    
        else: # fall back default 
            minval = col_min

        if col in continuous_cols:
            maxval = instance_val + col_stdev
        # elif col in D.categorical_cols:
        #     maxval = max(1, col_max)
        else: # fall back default 
            maxval = col_max

        local_permitted_range[col] = [minval, maxval]

    # view_local_permitted_range:

    # visualize instance_permitted_range
    instance_permitted_range_df = pd.DataFrame(local_permitted_range).transpose()
    instance_permitted_range_df.columns = ['min', 'max']

    # visualize min max with patient data
    df_vis = pd.concat([instance, instance_permitted_range_df.transpose()]).transpose()    
    display(df_vis)

    return local_permitted_range

def generate_sample_local_cf_with_permitted_range(dfXy, dice_exp, instance, permitted_range, config, CFs=5):
    print(f"generating counterfactuals for the {config.model.name} model")    
    e1 = dice_exp.generate_counterfactuals(
        instance, total_CFs=CFs, 
        desired_class="opposite", 
        permitted_range=permitted_range,
        features_to_vary=dfXy.columns.drop(['SEX', 'Confirmed_Binary_DPN']).to_list()
        )    
    if config.experiment.verbosity > 0:
        e1.visualize_as_dataframe(show_only_changes=True)   
    return e1  


def get_instances_of_interest(model, X_test, y_test, config, split_index, threshold=0.5, delta=0.1, savedir=None):
    """Return instances of interest (ioi - misclassified and borderline instances) around the decision threshold."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    misclassified_mask = y_pred != y_test

    margin = np.abs(y_proba - threshold)
    borderline_mask = margin <= delta

    ioi_idx = np.where(misclassified_mask | borderline_mask)[0]
    ioi_df = X_test.iloc[ioi_idx].copy()
    ioi_df["pred_proba"] = y_proba[ioi_idx]
    ioi_df["margin"] = margin[ioi_idx]
    ioi_df["pred"] = (y_proba[ioi_idx] >= threshold).astype(int)
    ioi_df["actual"] = y_test.iloc[ioi_idx].values
    ioi_df["misclassified"] = ioi_df["pred"] != ioi_df["actual"]
    
    print(f"Found {len(ioi_df)} misclassified and borderline cases (|p - {threshold:.4f}| ≤ {delta})")
    
    display_cols = X_test.columns[:4].to_list() + ['margin', 'misclassified', 'pred_proba','pred','actual']
    if config.experiment.verbosity > 0:
        display(ioi_df[display_cols])
    if savedir:
        filename = f'{config.model.code}_split{split_index}_local_cf.csv'
        ioi_df.to_csv(savedir / filename)
    return ioi_df, display_cols


def generate_diverse_cfs(dice_exp, instance, total_CFs=30, features_to_vary='all', permitted_range={}, seeds=[0,1,2,3,4], diversity_weight=1.5):
    """Generate diverse counterfactuals across multiple seeds."""
    all_cfs = []
    for s in seeds:
        # manually set random seed
        np.random.seed(s)
        random.seed(s) 

        cf = dice_exp.generate_counterfactuals(
            instance,
            total_CFs=total_CFs,
            desired_class="opposite",
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
            #random_seed=s,
            diversity_weight=diversity_weight
        )        
        df_cf = cf.cf_examples_list[0].final_cfs_df
        if not df_cf.empty:
            all_cfs.append(df_cf)
    if all_cfs:
        combined = pd.concat(all_cfs).drop_duplicates().reset_index(drop=True)
        return combined
    else:
        return pd.DataFrame()
    

def plot_local_cf_heatmap(dfXy, df_dcf, query_instance, 
                          query_idx, pred, actual,                           
                          config,
                          split_index,
                          savedir=None,
                          ):   
    z = config.dice.nzfill
    save_every = config.dice.heatmap.save_every
    verbosity = config.experiment.verbosity
    figsize = (config.dice.heatmap.figsize.x, config.dice.heatmap.figsize.y)

    # Compute differences (each row in df_large vs. the single row)
    diffs = df_dcf - query_instance.iloc[0]
    if verbosity: print("diffs.shape: ", diffs.shape)
    
    batch_ranges = [(b*save_every, b*save_every+save_every) for b in range(int(np.ceil(df_dcf.shape[0]/save_every))) ]
    if verbosity: print("batch_ranges: ",  batch_ranges)
    
    for idx_start, idx_end in batch_ranges:
        if verbosity: print("idx_start, idx_end: ", idx_start, idx_end)
        diff = diffs.iloc[idx_start: idx_end]
        
        diff = diff[dfXy.drop('Confirmed_Binary_DPN', axis=1).columns]
        if verbosity: print("diff.shape: ",  diff.shape)

        # Create mask where values == 0
        mask = diff == 0

        # Plot heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            diff,
            mask=mask,            # hide zero differences
            cmap="RdBu",        # diverging color map centered at 0
            center=0,
            annot=True,           # show annotations
            fmt=".2f",            # format annotations to 2 decimal places
            annot_kws={"size": 6},# smaller font
            cbar_kws={'label': 'Difference'}
        )

        ax.set_yticks(np.arange(len(diff)) + 0.5)
        ax.set_yticklabels(diff.index, rotation=0, fontsize=8)

        # Make x-tick labels smaller too
        ax.set_xticklabels(diff.columns, rotation=45, ha='right', fontsize=8)

        # plt.title("Differences from Instance", fontsize=12)
        idx_str = str(query_idx).zfill(z)
        plt.title(f"Counterfactuals for Patient {idx_str}: predicted {pred}, actual {actual}", fontsize=12, pad=20)
        plt.xlabel("Features")
        plt.ylabel("Counterfactuals")

        # 1. Get the feature values from the query instance
        desired_order = diff.columns.tolist() 
        query_instance_reordered = query_instance[desired_order]

        query_values = query_instance_reordered.iloc[0].values
        NEW_Y_POSITION = -0.05 

        # 2. Loop through each feature to place the value text
        for i, value in enumerate(query_values):
            # i + 0.5 centers the text in the column cell.
            # y = -0.3: Increased separation from the heatmap for visual clarity/alignment.
            ax.text(
                x=i + 0.5,
                y=NEW_Y_POSITION, #-0.3, # Adjusted from -0.2 to -0.3
                s="0" if value==0 else "1" if value==1 else f"{value:.2f}",  # Display the value, formatted
                ha='center',
                va='bottom',
                fontsize=6,
                # fontweight='bold',
                color='#1a1a1a' # Slightly darker color for visibility
            )

        # 3. Add a row header label for the new values
        ax.text(
            x=-0.5, # Position to the left of the Y-axis labels
            y=NEW_Y_POSITION, #-0.3, # Adjusted from -0.2 to -0.3
            s="Instance Values:",
            ha='right',
            va='bottom',
            fontsize=6,
            # fontweight='bold',
            color='#1a1a1a' # Slightly darker color for visibility
        )

        plt.tight_layout()
        if savedir:
            idx_end = min(idx_end, idx_start+diff.shape[0])
            filename = f'{config.model.code}_split{split_index}_local_cf'
            filename += f'idx{idx_start}-idx{idx_end-1}'
            filename += '.png'
            plt.savefig(savedir / filename)
        return