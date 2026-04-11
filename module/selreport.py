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
from datetime import datetime

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

    # ## Iterative Group Feature Elimination
    
    Xnoncs = X.drop(columns=D.ncs_cols)

    def benchmark_featureset(*, feature_set_code, benchmark_cols, verbosity=0):
        start_time = datetime.now()    
        print(f'{feature_set_code} benchmarking models for feature set started at: ', start_time.strftime("%H:%M:%S"))
        benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=verbosity)
        model_metrics[feature_set_code] = benchmark_metrics
        metrics_stats[feature_set_code] = sel.calculate_metric_statistics(benchmark_metrics, config)
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f'{feature_set_code} benchmarking models for feature set took: {elapsed.total_seconds()/60:.2f}, ended at: ',  start_time.strftime("%H:%M:%S"))

        print(f"Elapsed: {elapsed}") 

    high_vif = sel.get_high_vif(Xnoncs, config)
    high_vif_features = high_vif.feature.values.tolist()[1:]

    feature_sets ={
        'All' : None,
        'NoCol' : [c for c in Xnoncs.columns if c not in high_vif_features],
        'NoProf' : [c for c in Xnoncs.columns if c not in D.profile_cols],
        'NoCom': [c for c in Xnoncs.columns if c not in D.comorbidity_cols],
        'NoNeuro' : [c for c in Xnoncs.columns if c not in D.neuro_cols],
        'NoMsi' : [c for c in Xnoncs.columns if c not in D.mnsi_col],
        'NoSudo' : [c for c in Xnoncs.columns if c not in D.sudo_cols],
        'Sudo': D.sudo_cols,
        'SudoProf' : D.sudo_cols + D.profile_cols,
        'SudoCom' : D.sudo_cols + D.comorbidity_cols,
        'SudoNeuro' : D.sudo_cols + D.neuro_cols,
        'SudoMnsi' : D.sudo_cols + D.mnsi_col
    }

    for code, cols in feature_sets.items():
        benchmark_featureset(feature_set_code=code, benchmark_cols=cols)

    # # ### No NCS
    # benchmark_cols = Xnoncs.columns.to_list() 
    # benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    # model_metrics['All'] = benchmark_metrics
    # metrics_stats['All'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # #### No NCS-derived studies: start using Xnoncs 

    # # ### No NCS and Collinear Features

    # benchmark_cols = [c for c in Xnoncs.columns if c not in high_vif_features]
    # benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    # model_metrics['NoCol'] = benchmark_metrics
    # metrics_stats['NoCol'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### No NCS and Profile
    # benchmark_cols = [c for c in Xnoncs.columns if c not in D.profile_cols]
    # benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    # model_metrics['NoProf'] = benchmark_metrics
    # metrics_stats['NoProf'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### No NCS and Comorbidities
    # benchmark_cols = [c for c in Xnoncs.columns if c not in D.comorbidity_cols]
    # benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    # model_metrics['NoCom'] = benchmark_metrics
    # metrics_stats['NoCom'] = sel.calculate_metric_statistics(benchmark_metrics, config)
    
    # # ### No NCS and Neurology Exam
    # benchmark_cols = [c for c in Xnoncs.columns if c not in D.neuro_cols]
    # benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    # model_metrics['NoNeuro'] = benchmark_metrics
    # metrics_stats['NoNeuro'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### No NCS and MSI
    # benchmark_cols = [c for c in Xnoncs.columns if c not in D.mnsi_col]
    # benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    # model_metrics['NoMsi'] = benchmark_metrics
    # metrics_stats['NoMsi'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### No NCS and Sudoscan
    # benchmark_cols = [c for c in Xnoncs.columns if c not in D.sudo_cols]
    # benchmark_metrics = sel.benchmark_models(Xnoncs, y, benchmark_cols, config, verbosity=0)
    # model_metrics['NoSudo'] = benchmark_metrics
    # metrics_stats['NoSudo'] = sel.calculate_metric_statistics(benchmark_metrics, config)
    
    # # ### Ablations for Sudoscan

    # # ### Sudo only
    # benchmark_cols = D.sudo_cols
    # benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    # model_metrics['Sudo'] = benchmark_metrics
    # metrics_stats['Sudo'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### Sudo +  Profile
    # benchmark_cols = D.sudo_cols +  D.profile_cols
    # benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    # model_metrics['SudoProf'] = benchmark_metrics
    # metrics_stats['SudoProf'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### Sudo +  Comorbidity
    # benchmark_cols = D.sudo_cols +  D.comorbidity_cols
    # benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    # model_metrics['SudoCom'] = benchmark_metrics
    # metrics_stats['SudoCom'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### Sudo Profile + Neurology
    # benchmark_cols = D.sudo_cols +  D.neuro_cols
    # benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    # model_metrics['SudoNeuro'] = benchmark_metrics
    # metrics_stats['SudoNeuro'] = sel.calculate_metric_statistics(benchmark_metrics, config)

    # # ### Sudo Profile + MNSI 
    # benchmark_cols = D.sudo_cols +  D.mnsi_col
    # benchmark_metrics = sel.benchmark_models(X, y, benchmark_cols, config, verbosity=0)
    # model_metrics['SudoMnsi'] = benchmark_metrics
    # metrics_stats['SudoMnsi'] = sel.calculate_metric_statistics(benchmark_metrics, config)


    report_metrics = list(sel.metric_fullname.keys()) 
    
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