
import numpy as np
import pandas as pd
import matplotlib
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

backend = matplotlib.get_backend()

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


def get_global_importance(dice_exp, DPN_data, X_test, config, split_index, 
                          features_to_vary, threshold, global_permitted_range, 
                          highlight_features=[], filename_suffix="", savedir=None):
    """
    Parameters:
    dice_exp: DiCE explainer object
    X_test: test set 
    total_CFs: Number of counterfactuals to generate
    """
    if savedir:
        filename = f'{config.model.code}_split{split_index}_global_importance'
        if filename_suffix:
            filename = f'{filename}_{filename_suffix}'
        fullpath_plot = savedir / f'{filename}.png'
        fullpath_values = savedir / f'{filename}.csv'
        if fullpath_plot.is_file():
            print(f'{fullpath_plot} already exists.')
            return

    D = DPN_data
    if config.dice.global_cf.posthoc_sparsity_param=='None':
        posthoc_sparsity_param = None
    else:
        posthoc_sparsity_param = config.dice.global_cf.posthoc_sparsity_param
    cobj = dice_exp.global_feature_importance(
        X_test, 
        total_CFs=config.dice.global_cf.total_CFs, 
        features_to_vary=features_to_vary,
        permitted_range=global_permitted_range,
        stopping_threshold=threshold,
        posthoc_sparsity_param=posthoc_sparsity_param,
        verbose=0)
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

    ax.set_title(f"Global Feature Importance for Model {split_index}")
    ax.set_ylabel("Category")
    ax.set_xlabel("Value")
    plt.tight_layout()
    if savedir:
        df_imp.T.to_csv(fullpath_values)
        fig.savefig(fullpath_plot)
    plt.close(fig) if backend in ["Agg"] else plt.show()

# LOCAL COUNTERFACTUALS 
# ----------------------

def get_local_permitted_range(dfXy, instance, features_to_vary, 
                              categorical_cols, continuous_cols, 
                              monotonic_cols, config, split_index, savedir=None):
    local_permitted_range = {}
    for col in features_to_vary:

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
    df_vis.columns = ['instance', 'min', 'max']
    print("Local permitted range:")
    display(df_vis)
    if savedir:
        filename = f'{config.model.code}_split{split_index}_instance_permitted_range.csv'
        df_vis.to_csv(savedir / filename)        
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
        filename = f'{config.model.code}_split{split_index}_instances_of_interest.csv'
        ioi_df.to_csv(savedir / filename)
    return ioi_df, display_cols


def generate_diverse_cfs(dice_exp, instance, config, split_index, 
                         threshold, features_to_vary, permitted_range={}, 
                         savedir=None):
    """Generate diverse counterfactuals across multiple seeds."""
    
    all_cfs = []
    seeds = list(range(config.dice.local_cf.nrepeats))
    for s in seeds:
        # manually set random seed
        np.random.seed(s)
        random.seed(s) 
        try:
            cf = dice_exp.generate_counterfactuals(
                instance,
                total_CFs=config.dice.local_cf.total_CFs,
                desired_class="opposite",
                permitted_range=permitted_range,
                features_to_vary=features_to_vary,
                stopping_threshold=threshold,
                posthoc_sparsity_algorithm=config.dice.local_cf.posthoc_sparsity_algorithm,
                posthoc_sparsity_param=config.dice.local_cf.posthoc_sparsity_param,
                proximity_weight=config.dice.local_cf.proximity_weight,
                diversity_weight=config.dice.local_cf.diversity_weight,
                categorical_penalty=config.dice.local_cf.categorical_penalty,
                algorithm=config.dice.local_cf.algorithm,
            )        
            df_cf = cf.cf_examples_list[0].final_cfs_df
            if not df_cf.empty:
                all_cfs.append(df_cf)
        except Exception as e:
            print(f"[DiCE] No CFs found — {e}")
    if all_cfs:
        combined_dfs = pd.concat(all_cfs).drop_duplicates().reset_index(drop=True)         
    else:        
        combined_dfs = instance.iloc[0:0].copy() # empty dataframe with only the headers
    if savedir:
        filename = f'{config.model.code}_split{split_index}_local_cf.csv'
        combined_dfs.to_csv(savedir / filename)
    return combined_dfs

    
def plot_local_cf_heatmap(dfXy, df_dcf, query_instance, 
                          query_idx, pred, actual,
                          config,
                          split_index,
                          highlight_invalid=False,
                          categorical_cols=None,
                          savedir=None,
                          ):   
    z = config.reporting.nzfill
    save_every = config.reporting.cf_heatmap.save_every
    verbosity = config.experiment.verbosity
    figsize = (config.reporting.cf_heatmap.figsize.x, config.reporting.cf_heatmap.figsize.y)

    # Compute differences (each row in df_large vs. the single row)
    diffs = df_dcf - query_instance.iloc[0]
    if verbosity: print("diffs.shape: ", diffs.shape)
    
    batch_ranges = [(b*save_every, b*save_every+save_every) for b in range(int(np.ceil(df_dcf.shape[0]/save_every))) ]
    if verbosity: print("batch_ranges: ",  batch_ranges)
    
    for idx_start, idx_end in batch_ranges:
        # limit to this batch
        if verbosity: print("idx_start, idx_end: ", idx_start, idx_end)
        diff = diffs.iloc[idx_start: idx_end]
        
        # reorder to column from original dataframe
        diff = diff[dfXy.drop('Confirmed_Binary_DPN', axis=1).columns]
        if verbosity: print("diff.shape: ",  diff.shape)

        # Create mask where values == 0
        mask = diff == 0

        # Plot heatmap
        fig = plt.figure(figsize=figsize)
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

        if highlight_invalid:
            progressive_categorical_cols = list(set(progressive_cols) & set(categorical_cols))
            hightlight_cells = []
            yticklabels = ax.get_yticklabels()
            for row_idx in range(diff.shape[0]):
                highlight_row = False
                for col in progressive_categorical_cols:
                    col_idx = diff.columns.get_loc(col)
                    delta = df_dcf.iat[row_idx,col_idx] - query_instance.iat[0,col_idx]
                    if delta == -1:
                        hightlight_cells.append((row_idx, col_idx))
                        highlight_row = True
                if highlight_row:
                    yticklabels[row_idx].set_bbox(dict(
                        facecolor='yellow',
                        edgecolor='none',
                        boxstyle='round,pad=0.2'
                    ))                
            
            # row_idx, col_idx = 1, 19   # <-- change this to your desired position
            # # --- Modify annotation text ---
            for text in ax.texts:
                x, y = text.get_position()
                
                # Convert heatmap coords → dataframe indices
                col = int(x - 0.5)
                row = int(y - 0.5)

                if (row, col) in hightlight_cells:
                    text.set_weight("bold")      
                    text.set_backgroundcolor("yellow")

        ax.set_yticks(np.arange(len(diff)) + 0.5)
        ax.set_yticklabels(diff.index, rotation=0, fontsize=8)

        # Make x-tick labels smaller too
        ax.set_xticklabels(diff.columns, rotation=45, ha='right', fontsize=8)

        # plt.title("Differences from Instance", fontsize=12)
        qstr = str(query_idx).zfill(z)
        plt.title(f"Counterfactuals for Patient {qstr}: predicted {pred}, actual {actual}", fontsize=12, pad=20)
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
            idx_start_str = str(idx_start).zfill(3)
            idx_end_str = str(idx_end-1).zfill(3)
            filename += f'_qidx{qstr}_{idx_start_str}-{idx_end_str}'
            filename += '.png'
            plt.savefig(savedir / filename)
            print(f'Counterfactual heatmaps saved to {filename} in {savedir}')
        plt.close(fig) if backend in ["Agg"] else plt.show()
        return


def get_most_changed_feature(df_cf, instance, config, split_index, savedir):
    # Boolean mask: True if feature changed compared to the original instance
    changed_mask = df_cf.ne(instance.iloc[0])

    # Count how many counterfactuals changed each feature
    change_counts_df = changed_mask.sum()
    change_counts_df = change_counts_df.sort_values(ascending=False)
    change_counts_df.reset_index()
    change_counts_df.columns = ['feature', 'change count']
    if savedir:       
        filename = f"{config.model.code}_split{split_index}_local_cf_most_changed"
        change_counts_df.to_csv(savedir / f'{filename}.csv')    
    return change_counts_df


def get_local_cf_distances(
        instance_df, cf_df, config, split_index, 
        feature_costs=None, sort_by=None, savedir=None):
    """
    Compute distances, sparsity, and feasibility per counterfactual.
    feature_costs: optional dict of feature->cost weights
    """
    if cf_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    x0 = instance_df.iloc[0]
    diffs = cf_df.sub(x0)
    sparsity = (diffs != 0).sum(axis=1)   # number of columns altered
    l1 = np.abs(diffs).sum(axis=1)        
    l2 = np.sqrt((diffs**2).sum(axis=1))

    cf_df["sparsity"] = sparsity
    cf_df["L1_dist"] = l1
    cf_df["L2_dist"] = l2

    if feature_costs:
        cf_df["cost"] = sum(np.abs(diffs[f]) * feature_costs.get(f, 1) for f in diffs.columns)

    if sort_by == 'L1_dist':        
        cf_df.sort_values("L1_dist").reset_index(drop=True)
    elif sort_by == 'L2_dist':        
        cf_df.sort_values("L2_dist").reset_index(drop=True)
    
    # generate a dataframe with the diffs and the analysis
    diffs = cf_df.drop(columns=['sparsity', 'L1_dist', 'L2_dist']).sub(x0)     
    diffs = pd.concat([diffs, cf_df[['sparsity', 'L1_dist', 'L2_dist']]], axis=1)
    
    if savedir:
        filename = f'{config.model.code}_split{split_index}_local_cf_distance_diffs'
        diffs.to_csv(savedir / f'{filename}.csv')    

        filename = f'{config.model.code}_split{split_index}_local_cf_distances'
        cf_df.to_csv(savedir / f'{filename}.csv')    
    return diffs,  cf_df

def filter_invalid_progressive_cfs(df_dcf, query_instance, config, split_index, categorical_cols, savedir):
    """
    For progressive features,  if a counterfactual sets to 0 what was originally 1, 
    it is an invalid counterfactual.
        cf  orig
        1   1   no change, ok
        1   0   valid
        0   1   invalid
        0   0   no change, ok    
    """
    progressive_categorical_cols = list(set(progressive_cols) & set(categorical_cols))
    print('Checking for invalid counterfactuals in these columns:\n', progressive_categorical_cols)

    diffs = df_dcf[progressive_categorical_cols] - query_instance.iloc[0][progressive_categorical_cols] 

    # Create a boolean mask where cells equal -1
    mask = (diffs == -1).any(axis=1)
    nfiltered = mask.sum()
    if nfiltered:
        print(f'Removed {nfiltered} invalid progressive counterfactuals')
    else:
        print(f'All counterfactuals are valid. None was filtered.')
    filtered_df = df_dcf.copy()[~mask]
    if savedir:       
        filename = f'{config.model.code}_split{split_index}_local_cf.csv'
        filtered_df.to_csv(savedir / filename)    
    return filtered_df


# NECESSITY AND SUFFICIENCY 
# -------------------------

def check_sufficiency(dice_exp, instance, check_features, permitted_range, 
                      desired_class="opposite", maxiterations=500):
    """
    Determine the sufficiency of each feature for one instance.
    A sufficient feature  change is one that can cause the outcome change by itself.
    """
    results = {}
    for f in check_features:
        results[f] = "insufficient"
        print(f'Checking sufficiency for {f}...')
        # --- Sufficiency: vary only this feature  ---
        try:
            cf_suf = dice_exp.generate_counterfactuals(
                instance, total_CFs=1, desired_class=desired_class, features_to_vary=[f], 
                permitted_range=permitted_range, maxiterations=maxiterations,
            )
            if len(cf_suf.cf_examples_list[0].final_cfs_df) > 0:
                results[f] = "sufficient"
        except Exception as e:
            print(f'Error calculating sufficiency for {f}')
            results[f] = "error"
            pass
        print(f'{f}: {results[f]}')        
        results_df = pd.DataFrame(results, index=['sufficiency']).T.reset_index(names="feature")
    return 

def check_necessity(dice_exp, instance, all_features, permitted_range, desired_class="opposite", 
                    maxiterations=500, total_CFs=3, nrepeats=5, verbose=False):
    """Determine necessity (vary all except this one across multiple seeds) of each feature for one instance."""

    results = {}
    for f in all_features:
        print(f'Checking necessity for {f}:')    
        results[f] = "unnecessary"   

        # --- Necessity: vary all except this one across multiple seeds ---
        features_wo_f = [feat for feat in all_features if feat != f]
        found_cf = False
        seeds = list(range(nrepeats))
        for seed in seeds:
        
            np.random.seed(seed)
            random.seed(seed) 

            cf_nec = dice_exp.generate_counterfactuals(
                instance, total_CFs=total_CFs, desired_class=desired_class,
                features_to_vary=features_wo_f, permitted_range=permitted_range, 
                maxiterations=maxiterations,
                verbose=verbose,                
                #random_seed=seed
            )
            if len(cf_nec.cf_examples_list[0].final_cfs_df) > 0:
                found_cf = True
                break

        if not found_cf:
            results[f] = "necessary"

        results_df = pd.DataFrame(results, index=['necessity']).T.reset_index(names="feature")
    return results_df 


def generate_local_cf_reports(dfXy, dice_exp, ioi_df, qidx, 
                              features_to_vary,                              
                              config,
                              split_index,
                              threshold,
                              categorical_cols,
                              continuous_cols,
                              remove_invalid_progressive_cfs=True,
                              savedir=None):
    
    unfiltered_cfs_savedir = savedir / 'unfiltered' / str(qidx).zfill(3) 
    filtered_cfs_savedir = savedir / 'filtered_progressive' / str(qidx).zfill(3) 
    unfiltered_cfs_savedir.mkdir(parents=True, exist_ok=True) 
    filtered_cfs_savedir.mkdir(parents=True, exist_ok=True) 
        
    print(f'Creating reports for Instance {qidx}...')
    print(f'Outputs will be saved to {savedir}.')

    X = dfXy.drop(['Confirmed_Binary_DPN'], axis=1)
    query_instance = X[qidx:qidx+1]

    print('Calculating instance permitted range...')
    instance_permitted_range = get_local_permitted_range(
        dfXy, query_instance, features_to_vary, categorical_cols, continuous_cols, 
        progressive_cols, config, split_index, savedir=unfiltered_cfs_savedir)

    print('Generating Counterfactuals...')
    df_dcf = generate_diverse_cfs(
        dice_exp,
        query_instance, 
        config,
        split_index,
        threshold,
        features_to_vary=features_to_vary,
        permitted_range=instance_permitted_range,
        savedir=unfiltered_cfs_savedir
        )
    
    print('plotting heatmaps...')
    plot_local_cf_heatmap(dfXy, df_dcf, query_instance, 
                        query_idx=qidx, 
                        pred=ioi_df.loc[qidx].pred, 
                        actual=ioi_df.loc[qidx].actual, 
                        config=config, split_index=split_index,
                        savedir=unfiltered_cfs_savedir)    
    
    print('Getting changed features...')
    get_most_changed_feature(df_dcf, query_instance, config, split_index, savedir=unfiltered_cfs_savedir)

    print('Computing Distances...')
    _diffs, _cf_ana = get_local_cf_distances(
        query_instance, df_dcf, config, split_index, sort_by="L2_dist", savedir=unfiltered_cfs_savedir)

    if remove_invalid_progressive_cfs:
        print('removing invalid progressive counterfactuals...')
        df_dcf = filter_invalid_progressive_cfs(df_dcf, query_instance, config, split_index, categorical_cols, savedir=filtered_cfs_savedir)

        plot_local_cf_heatmap(dfXy, df_dcf, query_instance, 
                            query_idx=qidx, 
                            pred=ioi_df.loc[qidx].pred, 
                            actual=ioi_df.loc[qidx].actual, 
                            config=config, split_index=split_index,
                            savedir=filtered_cfs_savedir)  
        
        print('Getting changed features...')
        get_most_changed_feature(df_dcf, query_instance, config, split_index, savedir=filtered_cfs_savedir)

        print('Computing Distances...')
        _diffs, _cf_ana = get_local_cf_distances(
            query_instance, df_dcf, config, split_index, sort_by="L2_dist", savedir=filtered_cfs_savedir)

