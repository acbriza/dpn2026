"""
    Produce feature and model selection reports
"""
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import shutil
from datetime import datetime
from tqdm import tqdm

import sys
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
# Repeated k-fold benchmarking across many models/feature sets routinely hits
# non-convergence (e.g. SGDClassifier, LogisticRegression) and undefined-metric
# folds (e.g. precision/recall with no positive predictions); both are already
# handled downstream (NaN'd out or reported as 0 via zero_division). Only these
# two categories are silenced so other warnings (e.g. FutureWarning) still surface.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from dataload import DPN_data
import ymlconfig

from utils2 import selection as sel

def main():
    overwrite_benchmarks = False
    if len(sys.argv) < 2:
        print("Usage: python selreport.py <config file> <n_cores> <overwrite>")
        sys.exit(1)
    if len(sys.argv) == 2:
        n_cpus = -1
    if len(sys.argv) >= 3:
        n_cpus = int(sys.argv[2])
    if len(sys.argv) >= 4:
         if sys.argv[3]=='overwrite':
             overwrite_benchmarks = True
    
    # sample config_filename = bin_sel_final.yml
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
    # dataset_path (e.g. '../dataset/EAMC_DPN_Dataset.xlsx') is written relative to this
    # script's directory, same as config_path above; resolving it against script_dir
    # rather than the process's current working directory keeps this independent of
    # where the script is invoked from.
    D = DPN_data(str(script_dir / config.data.dataset_path))
    D.load(classification=config.experiment.classification_type)
    dfdpn = D.df
    data_cols = dfdpn.drop(D.non_data_cols, axis=1, errors="ignore").columns
    X = dfdpn[data_cols]
    y = dfdpn[D.get_target_column()]

    # ## Global Variables
    # (key was "rcv_cores" in this comment; corrected to "rcv_scores" to match the actual dict key set in selection.py)
    model_metrics = {} # key: experiment code, value: {model: <string> (e.g. all, ncs), rcv_scores: <Dataframe> Performance metrics of repeated k-fold of algorithms}
    metrics_stats = {} # key: experiment code, value: {stat: <string> (e.g. mean, std), stat (mean/std) of the performance of all algorithms}

    # ## Iterative Group Feature Elimination
    
    Xnoncs = X.drop(columns=D.ncs_cols)

    def benchmark_featureset(*, feature_set_code, benchmark_cols, verbosity=0):
        start_time = datetime.now()    
        print(f'{feature_set_code} benchmarking models for feature set started at: ', start_time.strftime("%m-%d %H:%M:%S"))
        benchmark_metrics = sel.benchmark_models_in_parallel(feature_set_code, Xnoncs, y, benchmark_cols, config, 
                                                 savedir=outputdir, overwrite=overwrite_benchmarks, verbosity=verbosity, 
                                                 n_cpus=n_cpus)
        model_metrics[feature_set_code] = benchmark_metrics
        metrics_stats[feature_set_code] = sel.calculate_metric_statistics(feature_set_code, benchmark_metrics, config, 
                                                                          savedir=outputdir, overwrite=overwrite_benchmarks)
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f'{feature_set_code} benchmarking models for feature set took: {elapsed.total_seconds()/60:.2f}, ended at: ',  start_time.strftime("%m-%d %H:%M:%S"))

    high_vif = sel.get_high_vif(Xnoncs, config)
    high_vif_features = high_vif.feature.values.tolist()

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

    report_metrics = list(sel.metric_fullname.keys()) 

    print(f"Generating summaries...") 
    for metric in tqdm(report_metrics):
        # Summary: including all columns and stats
        sel.create_model_summary_table(metrics_stats, config,
                                target_metric=metric, 
                                exclude_features=[],
                                include_mean=False, 
                                include_topk=False,
                                sorting_field='All', 
                                show_plot=True,
                                savedir=outputdir)


        # Summary: clean table without topk
        sel.create_model_summary_table(metrics_stats, config,
                                target_metric=metric, 
                                topk=0, 
                                exclude_features=[],
                                include_mean=True, 
                                include_topk=False,
                                sorting_field='mean',
                                show_plot=False,
                                savedir=outputdir);

        # LaTeX tables of mean, std, median for all models, per feature set
        sel.create_latex_stat_tables(config,
                                target_metric=metric,
                                exclude_features=[],
                                savedir=outputdir)
    print(f"Generating summaries: Done.")

    #### Make violin plots of Best Feature Set
    print(f"Plotting violin plots...") 
    for feature_group in tqdm(model_metrics):
        for metric in report_metrics:
            scores= sel.get_metric_scores(model_metrics, feature_group, metrics_stats, metric)
            sel.plot_metric_scores(scores, config, exp_code=feature_group,  target_metric=metric, savedir=outputdir)
    print(f"Plotting violin plots: Done.") 

if __name__ == "__main__":
    main()