"""
    Produce optimization report for selected model (CatBoost)
"""
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import shutil
from datetime import datetime

import sys
import warnings
warnings.filterwarnings('ignore')

from dataload import DPN_data
import ymlconfig

from utils2 import optimization as hpo

def main():
    if len(sys.argv) < 2:
        print("Usage: python optreport.py <config file> <overwrite>")
        sys.exit(1)

    if len(sys.argv) == 2:
        overwrite_optimization_reports = False
    else:
        overwrite_optimization_reports = sys.argv[2]=='overwrite'
    
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
    # dataset_path (e.g. '../dataset/EAMC_DPN_Dataset.xlsx') is written relative to this
    # script's directory, same as config_path above; resolving it against script_dir
    # rather than the process's current working directory keeps this independent of
    # where the script is invoked from.
    D = DPN_data(str(script_dir / config.data.dataset_path))
    D.load(classification=config.experiment.classification_type)
    dfdpn = D.df
    data_cols = dfdpn.drop(D.non_data_cols, axis=1, errors="ignore").columns
    no_ncs_datacols = [c for c in data_cols if c not in D.ncs_cols]
    X = dfdpn[no_ncs_datacols]
    y = dfdpn[D.get_target_column()]
    print(f'X: {X.shape}, y:{y.shape}')


    def param_space_fn(trial):
        return  {
            "depth": trial.suggest_int(
                "depth", 
                config.param_space.depth.min, 
                config.param_space.depth.max),
            "learning_rate": trial.suggest_float(
                "learning_rate", 
                config.param_space.learning_rate.min, 
                config.param_space.learning_rate.max, 
                log=True),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 
                config.param_space.l2_leaf_reg.min, 
                config.param_space.l2_leaf_reg.max),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 
                config.param_space.scale_pos_weight.min, 
                config.param_space.scale_pos_weight.max,
                ),
            # --- Fixed ---
            "loss_function": config.param_space.loss_function,
            "eval_metric": config.param_space.eval_metric,
            "iterations": config.param_space.iterations,
            "early_stopping_rounds": config.param_space.early_stopping_rounds,
            "verbose": 0,
        }
    
    start_time = datetime.now()    
    print('Running repeated crossvalidation, started at: ', start_time.strftime("%m-%d %H:%M:%S"))
    opt_results = hpo.nested_cv_optimization(
        X,
        y,
        config,
        model_class=hpo.model_class[config.model.code],   # class, not an instance
        param_space_fn=param_space_fn,
        savedir=outputdir, 
        overwrite=overwrite_optimization_reports,
    )

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(opt_results)
    print(f'Running repeated crossvalidation, took: {elapsed.total_seconds()/60:.2f}, ended at: ',  end_time.strftime("%m-%d %H:%M:%S"))

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