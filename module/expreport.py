"""
    Produce final models and explainability reports based on study from explainability.ipynb
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

from utils2 import explainability as exp

def main():
    if len(sys.argv) < 2:
        print("Usage: python expreports.py <config file> <overwrite>")
        sys.exit(1)

    if len(sys.argv) == 2:
        overwrite_reports = False
    else:
        overwrite_reports = sys.argv[2]=='overwrite'
    
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


    start_time = datetime.now()    
    print(f'Training models on data splits, started at: ', start_time.strftime("%H:%M:%S"))
    ksplit_trained_models =  exp.get_ksplit_trained_models(
        X, y, config,
        savedir=outputdir, 
        overwrite=overwrite_reports,        
        )
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nrundate and tag: ", ksplit_trained_models['rundate'], ksplit_trained_models['tag'])
    print(f"summary:\n {ksplit_trained_models['summary']}")
    print(f'Training models on data splits, took: {elapsed.total_seconds()/60:.2f}, ended at: ',  start_time.strftime("%H:%M:%S"))

if __name__ == "__main__":
    main()