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
    benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config) 
    model_metrics['All'] = benchmark_metrics
    metrics_stats['All'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # ### NCS only
    benchmark_cols = D.ncs_cols
    benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    model_metrics['Ncs'] = benchmark_metrics
    metrics_stats['Ncs'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # ### Sudo only
    benchmark_cols = D.sudo_cols
    benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    model_metrics['Sudo'] = benchmark_metrics
    metrics_stats['Sudo'] = sel.calculate_metric_statistics(benchmark_metrics, config)
    
    # ### No NCS
    Xnoncs = X.drop(columns=D.ncs_cols)
    benchmark_cols = Xnoncs.columns.to_list() 
    benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcs'] = benchmark_metrics
    metrics_stats['NoNcs'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    #### No NCS-derived studies: start using Xnoncs 

    # ### No NCS and Collinear Features
    high_vif = sel.get_high_vif(Xnoncs, config)
    high_vif_features = high_vif.feature.values.tolist()[1:]
    benchmark_cols = [c for c in Xnoncs.columns if c not in high_vif_features]
    benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcsCol'] = benchmark_metrics
    metrics_stats['NoNcsCol'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # ### No NCS and Profile
    benchmark_cols = [c for c in Xnoncs.columns if c not in D.profile_cols]
    benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcsProf'] = benchmark_metrics
    metrics_stats['NoNcsProf'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # ### No NCS and Comorbidities
    benchmark_cols = [c for c in Xnoncs.columns if c not in D.comorbidity_cols]
    benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcsCom'] = benchmark_metrics
    metrics_stats['NoNcsCom'] = sel.calculate_metric_statistics(benchmark_metrics, config)
    
    # ### No NCS and Neurology Exam
    benchmark_cols = [c for c in Xnoncs.columns if c not in D.neuro_cols]
    benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcsNeuro'] = benchmark_metrics
    metrics_stats['NoNcsNeuro'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # ### No NCS and MSI
    benchmark_cols = [c for c in Xnoncs.columns if c not in D.mnsi_col]
    benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcsMsi'] = benchmark_metrics
    metrics_stats['NoNcsMsi'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # ### No NCS and Sudoscan
    benchmark_cols = [c for c in Xnoncs.columns if c not in D.sudo_cols]
    benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    model_metrics['NoNcsSudo'] = benchmark_metrics
    metrics_stats['NoNcsSudo'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    report_metrics = ['accuracy', 'precision', 'specificity', 'f1', 'f2', 'youden', 'roc-auc', 'auprc']
    
    for metric in report_metrics:
        # Summary: including all columns and stats
        sel.create_model_summary_table(metrics_stats, config,
                                target_metric=metric, 
                                exclude_features=[],
                                include_mean=True, 
                                show_plot=True,
                                savedir=outputdir)


        # Summary: clean table without mean, topk
        sel.create_model_summary_table(metrics_stats, config,
                                target_metric=metric, 
                                topk=0, 
                                exclude_features=[],
                                include_mean=False, 
                                show_plot=False,
                                savedir=outputdir);                           

    #### Make violin plots of Best Feature Set
    for feature_group in model_metrics:
        for metric in report_metrics:
            scores= sel.get_metric_scores(model_metrics, feature_group, metrics_stats, metric)
            sel.plot_metric_scores(scores, config, exp_code=feature_group,  target_metric=metric, savedir=outputdir)

if __name__ == "__main__":
    main()