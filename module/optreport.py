"""
    Produce optimization report for selected selected top model (CatBoost)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import joblib
import json
from pathlib import Path
import shutil
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier
from skopt.space import Integer, Real


import sys 
sys.path.append('..')  
import warnings
warnings.filterwarnings('ignore')

from dataload import DPN_data
import ymlconfig

from utils2 import optimization as hpo

def main():
    if len(sys.argv) < 2:
        print("Usage: python optreports.py <config file> <overwrite>")
        sys.exit(1)

    if len(sys.argv) == 2:
        overwrite_optimization_reports = False
    else:
        overwrite_optimization_reports = sys.argv[2]=='overwrite'
    
    config_path = Path(r'experiments')

    # sample config_filename = bin_opt_final.yml
    config_filename = sys.argv[1]

    # ## Read Config File    
    current_file = Path(__file__).resolve() # Get the absolute path of the current file
    script_dir = current_file.parent # Get the directory containing the file

    config_path = Path(script_dir /'experiments')
    config_dict = ymlconfig.load_config(config_path / config_filename)
    config = ymlconfig.dict_to_namespace(config_dict)
    print(config)

    # #### Set output directory
    outputdir = config_path /  config.experiment.classification_type /  config.experiment.stage / config.model.code / config.experiment.tag 
    outputdir.mkdir(parents=True, exist_ok=True)

    # #### Copy config file to output directory
    source = config_path / config_filename
    destination = outputdir / config_filename
    shutil.copy(source, destination)

    # ## Data Loading
    D = DPN_data(config.data.dataset_path[3:])
    D.load(classification=config.experiment.classification_type)
    dfdpn = D.df
    data_cols = dfdpn.drop(D.non_data_cols, axis=1, errors="ignore").columns
    no_ncs_datacols = [c for c in data_cols if c not in D.ncs_cols]
    X = dfdpn[no_ncs_datacols]
    y = dfdpn['Confirmed_Binary_DPN']
    print(f'X: {X.shape}, y:{y.shape}')
    dfXy = pd.concat([X, y], axis=1)    


    def param_space_fn(trial):
        return  {
            "iterations": trial.suggest_int(
                "iterations", 
                config.param_space.iterations.min, 
                config.param_space.iterations.max),
            "depth": trial.suggest_int(
                "depth", 
                config.param_space.depth.min, 
                config.param_space.depth.max),
            "learning_rate": trial.suggest_float(
                "learning_rate", 
                config.param_space.learning_rate.min, 
                config.param_space.learning_rate.max, 
                log=True),
            "l2_leaf_reg": trial.suggest_int(
                "l2_leaf_reg", 
                config.param_space.l2_leaf_reg.min, 
                config.param_space.l2_leaf_reg.max),
        }    
    
    start_time = datetime.now()    
    print(f'Running repeated crossvalidation, started at: ', start_time.strftime("%H:%M:%S"))
    opt_results = hpo.nested_cv_optimization(
        X.values,
        y.values,
        config,
        model_class=hpo.model_class[config.model.name],   # class, not an instance
        param_space_fn=param_space_fn,
        savedir=outputdir, 
        overwrite=overwrite_optimization_reports,
    )

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(opt_results)
    print(f'Running repeated crossvalidation, took: {elapsed.total_seconds()/60:.2f}, ended at: ',  start_time.strftime("%H:%M:%S"))

    # ### Calculate Confidence Interval 
    opt_ci_df  = hpo.mean_confidence_interval(
        opt_results,
        config, 
        savedir=outputdir, 
        overwrite=overwrite_optimization_reports,
        )
    print()
    print(opt_ci_df)

if __name__ == "__main__":
    main()