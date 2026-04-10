import seaborn as sns
import warnings
# import time
import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

from sklearn.model_selection import train_test_split

import dice_ml
from dice_ml import Dice

import sys 
sys.path.append('..')  


# from module.backends.backend_adapter import get_dice_components
from dataload import DPN_data
# from module.eda import EDA

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)  # decimal places for outputs from numpy
pd.set_option("display.precision", 3)  # decimal places for outputs from pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

D = DPN_data("../dataset/Sudoscan Working File with Stats.xlsx")
D.load(classification="binary")

df = D.df
data_cols = df.drop(D.non_data_cols, axis=1, errors="ignore").columns

X = df[data_cols]
y = df['Confirmed_Binary_DPN']

rf_optimized_results = joblib.load(rf"outputs\all_features\random_forest.pkl")

model_name = rf_optimized_results["name"]
best_params = rf_optimized_results["best_params"]
best_score = rf_optimized_results["best_score"]  
optimized_model = rf_optimized_results["optimized_model"] 
optimized_model_metrics = rf_optimized_results["optimized_model_metrics"]

dfXy = pd.concat([X, y], axis=1)

allfeature_cols = dfXy.columns.drop('Confirmed_Binary_DPN').to_list()
continuous_cols = dfXy.columns.difference(D.categorical_cols+['Confirmed_Binary_DPN']).to_list()

actionable_cols = ['HBA1C', 'DSLPDMIA', 'INSULIN']
progressive_cols = ['AGE', 'DM_DUR', 'HPN', 'PAOD', 'CKD', 'GBS', 'DEC_VS', 'DEC_PPS', 'DEC_LTS', 'DEC_AR', 'MNSI']
immutable_cols = ["SEX"]

def get_global_permitted_range(continuous_cols):
    global_permitted_range = {}
    for col in continuous_cols: # no need to set range for categorical columns
        stdev = dfXy[col].std()
        minval = dfXy[col].min()
        maxval = dfXy[col].max()
        minval = 0 if minval==0 else max(0, minval-stdev)
        maxval = maxval + stdev
        global_permitted_range[col] = [minval, maxval]
    return global_permitted_range

global_permitted_range = get_global_permitted_range(continuous_cols)

d = dice_ml.Data(dataframe=dfXy, 
                 categorical_features = D.categorical_cols,
                 continuous_features=continuous_cols,                  
                 permitted_range = global_permitted_range,                 
                 outcome_name='Confirmed_Binary_DPN')
m = dice_ml.Model(model=optimized_model, backend="sklearn", model_type="classifier")
exp = dice_ml.Dice(d, m, method="genetic")

def get_local_permitted_range(instance, allfeature_cols, continuous_cols, monotonic_cols):
    local_permitted_range = {}
    for col in allfeature_cols:

        if col in D.categorical_cols:
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
    return local_permitted_range

pidx = 40
query_instance = X[pidx:pidx+1]
instance_permitted_range = get_local_permitted_range(
    query_instance, allfeature_cols, continuous_cols, progressive_cols)

import multiprocessing 

def run_dice_in_process(queue, dice_exp, instance, total_CFs, desired_class,
                        features_to_vary, permitted_range, maxiterations):
    """Run DiCE in a separate process so it can be timed out."""
    try:
        cf_suf = dice_exp.generate_counterfactuals(
                        instance, total_CFs=total_CFs, desired_class=desired_class, features_to_vary=features_to_vary, 
                        permitted_range=permitted_range, maxiterations=maxiterations,
                    )
        queue.put(cf_suf)
    except Exception as e:
        queue.put(e)
    print("Done", cf_suf)
    return

def sample_func(queue, somearg):
    try:
        queue.put(somearg)
    except Exception as e:
        queue.put(e)
    print("Done", somearg)


def generate_cfs_with_timeout(timeout_sec, dice_exp, instance, total_CFs, desired_class,
                          features_to_vary, permitted_range, maxiterations):
    """Run generate_counterfactuals() with timeout control."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        # target=run_dice_in_process,
        # args=(q, dice_exp, instance, total_CFs, desired_class,features_to_vary, permitted_range, maxiterations)
        target=run_dice_in_process(q, dice_exp, instance, total_CFs, 
                                   desired_class, features_to_vary, permitted_range, maxiterations)
        # target=sample_func,
        # args=(q, random.randint(1,10)),
    )
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # Timeout occurred

    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        raise result
    return result

def check_sufficiency_with_timeout(dice_exp, instance, all_features, permitted_range, 
                      desired_class="opposite", 
                      maxiterations=500, timeout_sec=30*60):
    """Determine sufficiency of each feature for one instance."""

    results = {}
    for f in all_features:
        results[f] = {"sufficient": "False"}
        print(f)

        try:
            cf_suf = generate_cfs_with_timeout(
                timeout_sec, dice_exp, instance, total_CFs=1, desired_class=desired_class, features_to_vary=[f], 
                permitted_range=permitted_range, maxiterations=maxiterations,
            )

            # timeout
            if cf_suf is None:
                print(f"⏰ Timeout (>{timeout_sec} sec) for feature '{f}' — skipping to next.")
                results[f]["sufficient"] = "Timeout"
                continue

            if len(cf_suf.cf_examples_list[0].final_cfs_df) > 0:
                results[f]["sufficient"] = "True"

        except Exception as e:
            print(f'Error calculating sufficiency for {f}')
            print(f'{e}')
            results[f]["sufficient"] = "Error"
            
        print('sufficient: ', results[f]["sufficient"])

        results_df = pd.DataFrame(results).T.reset_index(names="feature")
        results_df.to_csv('sufficiency.csv')
    return results_df

def check_sufficiency(dice_exp, instance, all_features, permitted_range, desired_class="opposite", 
                      maxiterations=500):
    """Determine necessity and sufficiency of each feature for one instance."""
    results = {}
    for f in all_features:
        results[f] = {"sufficient": "False"}
        print(f)

        # --- Sufficiency: vary only this feature  ---
        try:
            cf_suf = dice_exp.generate_counterfactuals(
                instance, total_CFs=1, desired_class=desired_class, features_to_vary=[f], 
                permitted_range=permitted_range, maxiterations=maxiterations,
            )
            if len(cf_suf.cf_examples_list[0].final_cfs_df) > 0:
                results[f]["sufficient"] = "True"
            print(f'Successfully calculated sufficiency for {f}')
        except Exception as e:
            print(f'Error calculating sufficiency for {f}')
            print(f'{e}')
            pass
        print('sufficient: ', results[f]["sufficient"])

        results_df = pd.DataFrame(results).T.reset_index(names="feature")
        results_df.to_csv('sufficiency.csv')
    return results_df

if __name__ == '__main__':
    forced_timeout_features = ['HBA1C',  'DEC_AR', 'DEC_VS', 'DEC_LTS', 'DEC_PPS', 'CMAPKNE_L', 'FEET_PCT_ASYM', 'INSULIN', 'HPN', 'HAND_PCT_ASYM',  'DSLPDMIA', 'DL_R']
    sufficient_features = ['SSA_R', 'SSA_L', 'DL_L']
    check_features = sufficient_features+forced_timeout_features
    #check_features = sufficient_features
    
    # df_s = check_sufficiency(
    #     exp,
    #     query_instance,
    #     #all_features=dfXy.columns.drop(['SEX', 'Confirmed_Binary_DPN']).to_list()[:3],
    #     all_features=check_features,
    #     maxiterations=2000,
    #     permitted_range=instance_permitted_range,
    # )

    df_s = check_sufficiency_with_timeout(
        exp,
        query_instance,
        #all_features=dfXy.columns.drop(['SEX', 'Confirmed_Binary_DPN']).to_list()[:3],
        all_features=check_features,
        maxiterations=2000,
        permitted_range=instance_permitted_range,
        timeout_sec=10*60
    )    

    print(df_s)