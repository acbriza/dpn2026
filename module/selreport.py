"""
    Produce feature and model selection reports based on study from selection.ipynb
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import joblib
from pathlib import Path
import shutil

import sys 
sys.path.append('..')  
import warnings
warnings.filterwarnings('ignore')

from dataload import DPN_data
import ymlconfig

from utils2 import selection as sel

def main():
    if len(sys.argv) < 2:
        print("Usage: python selreports.py <config file>")
        sys.exit(1)

    config_path = Path(r'experiments')

    # choose between final and development config file
    # config_filename = "bin_sel_final.yml" # final
    config_filename = sys.argv[1]

    # ## Read Config File    
    current_file = Path(__file__).resolve() # Get the absolute path of the current file
    script_dir = current_file.parent # Get the directory containing the file

    config_path = Path(script_dir /'experiments')
    config_dict = ymlconfig.load_config(config_path / config_filename)
    config = ymlconfig.dict_to_namespace(config_dict)

    # #### Set output directory
    outputdir = config_path /  config.experiment.classification_type /  config.experiment.stage / config.experiment.tag 
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
    X = dfdpn[data_cols]
    y = dfdpn['Confirmed_Binary_DPN']
    dfXy = pd.concat([X, y], axis=1)    

    # ## Global Variables   
    model_metrics = {} # key: experiment code, value: {model: <string> (e.g. all, ncs), rcv_cores: <Dataframe> Perfomance metrics of repeated k-fold of algorithms}
    metrics_stats = {} # key: experiment code, value: {stat: <string> (e.g. mean, std), stat (mean/std) of the performance of all algorithms}
    youden_scores = {} # key: experiment code, value: list of youden cv scores algorithms
    rocauc_scores = {} # key: experiment code, value: list of roc-auc cv scores algorithms

    # ## Iterative Group Feature Elimination

    # ### All Features
    benchmark_cols = X.columns.to_list() 
    model_metrics['All'] = sel.benchmark_models(X, y, benchmark_cols, config) 
    metrics_stats['All'] = sel.calculate_metric_statistics(model_metrics['All'], config)

    # #### Get youden and roc-auc scores
    youden_scores['All'] = sel.get_metric_scores(model_metrics, 'All', metrics_stats, 'youden')
    rocauc_scores['All'] = sel.get_metric_scores(model_metrics, 'All', metrics_stats, 'roc-auc')

    sel.plot_metric_scores(rocauc_scores, config, exp_code='All',  target_metric='roc-auc', savedir=outputdir)
    sel.plot_metric_scores(rocauc_scores, config, exp_code='All',  target_metric='youden', savedir=outputdir)
    sel.plot_metric_scores(rocauc_scores, config, exp_code='All',  target_metric='specificity', savedir=outputdir)

if __name__ == "__main__":
    main()