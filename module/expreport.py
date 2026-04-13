"""
    Produce final models and explainability reports based on study from explainability.ipynb
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from pathlib import Path
import shutil
from datetime import datetime


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

    split_results = ksplit_trained_models['results']

    # Feature Importances
    for s in range(len(split_results)): 
        model = split_results[s]['model']
        feature_names = X.columns
        exp.plot_importances(D, model, s, feature_names, config, 
                            minimum=None, limit=None, 
                            savedir=outputdir)
    # ROC-AUC
    for s in range(len(split_results)): 
        model = split_results[s]['model']
        X_test = split_results[s]['X_test']
        y_test = split_results[s]['y_test']
        y_proba = model.predict_proba(X_test)[:,1]
        exp.plot_roc_auc(y_test, y_proba, s, config, outputdir);        

    # DCA
    for s in range(len(split_results)): 
        model = split_results[s]['model']
        thresholds, nb = exp.plot_decision_curve_analysis(model, s, X_test, y_test, config, savedir=outputdir)

    # SHAP
    for s in range(len(split_results)): 
        model = split_results[s]['model']
        exp.plot_shap(D, model, s, config, X_test, savedir=outputdir)

    # AUPRC CURVE
    y_test_list = []
    y_proba_list = []
    for s in range(len(split_results)): 
        model = split_results[s]['model']
        X_test = split_results[s]['X_test']
        y_test = split_results[s]['y_test']
        y_proba = model.predict_proba(X_test)[:,1]# y_test: true labels
        y_test_list.append(y_test)
        y_proba_list.append(y_proba)

    exp.plot_cv_auprc(y_test_list, y_proba_list, config, savedir=outputdir)        

if __name__ == "__main__":
    main()