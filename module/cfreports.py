"""
    Produce counterfactual reports based on study from counterfactuals.ipynb
"""
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from catboost import CatBoostClassifier

from pathlib import Path
import shutil

import sys 
import time
sys.path.append('..')  
import warnings
warnings.filterwarnings('ignore')

import dice_ml

from dataload import DPN_data
import ymlconfig
from utils2 import counterfactuals as cf
from utils2 import explainability as exp

import argparse

def parse_comma_separated_ints(value):
    """Parse a comma-separated list of integers, e.g. '53,67'"""
    try:
        return [int(i) for i in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer list: '{value}'. Expected format: '53,67'")

parser = argparse.ArgumentParser(
    prog='cfreports.py',
    description='Generate counterfactual reports from a config file.',
    epilog="""
    examples:
    python cfreports.py bin_cf_final.yml
        --> redo all reports

    python cfreports.py bin_cf_final.yml skip_instances --model-idx 2 --instances 53,67
        --> do not overwrite reports of model 2, redo all instances but SKIP 53 & 67

    python cfreports.py bin_cf_final.yml redo_instances --model-idx 2 --instances 53,67
        --> do not overwrite reports of model 2, redo ONLY instances 53 & 67

    python cfreports.py bin_cf_final.yml global_only --model-idx 2
        --> generate Global Importances for Model 2 only
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    'config',
    help='Path to the YAML config file (e.g. bin_cf_final.yml)'
)
parser.add_argument(
    'mode',
    nargs='?',
    choices=['skip_instances', 'redo_instances', 'global_only'],
    default=None,
    help='Run mode: skip_instances, redo_instances, or global_only'
)
parser.add_argument(
    '--model-idx',
    type=int,
    default=None,
    dest='target_model_idx',
    help='Index of the model to target (required when mode is set)'
)
parser.add_argument(
    '--instances',
    type=parse_comma_separated_ints,
    default=[],
    dest='target_instance_indices',
    help='Comma-separated instance indices to skip or redo, e.g. 53,67'
)
parser.add_argument(
    '--speed',
    type=parse_comma_separated_ints,
    default=[],
    dest='target_instance_indices',
    help='Comma-separated instance indices to skip or redo, e.g. 53,67'
)

parser.add_argument(
    '--gen-timeout',
    choices=['quick', 'fast', 'normal'],
    default='fast',
    dest='gen_timeout',
    help='Generation timeout preset: quick, fast (default), normal'
)

def main():
    args = parser.parse_args()

    # Validate: --model-idx is required when mode is set
    if args.mode and args.target_model_idx is None:
        parser.error(f"--model-idx is required when mode is '{args.mode}'")

    # Validate: --instances only makes sense for skip/redo modes
    if args.target_instance_indices and args.mode not in ('skip_instances', 'redo_instances'):
        parser.error("--instances can only be used with 'skip_instances' or 'redo_instances' mode")

    config_filename         = args.config
    target_model_idx        = args.target_model_idx
    target_instance_indices = args.target_instance_indices
    skip_instances          = args.mode == 'skip_instances' or args.mode is None
    redo_instances          = args.mode == 'redo_instances'
    global_only             = args.mode == 'global_only'
    gen_timeout             = args.gen_timeout    
    # ## Read Config File    
    current_file = Path(__file__).resolve() # Get the absolute path of the current file
    script_dir = current_file.parent # Get the directory containing the file

    config_path = Path(script_dir /'experiments')
    config_dict = ymlconfig.load_config(config_path / config_filename)
    config = ymlconfig.dict_to_namespace(config_dict)
    print(config_dict)

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
    Xfull = dfdpn[data_cols]
    no_ncs_datacols = [c for c in data_cols if c not in D.ncs_cols]
    X = dfdpn[no_ncs_datacols]
    y = dfdpn['Confirmed_Binary_DPN']
    print(f'X: {X.shape}, y:{y.shape}')
    dfXy = pd.concat([X, y], axis=1)


    # ## Define custom column lists
    actionable_features = config.dice.cf_features.actionable.split(',')
    unactionable_features = config.dice.cf_features.unactionable.split(',')
    columns_not_to_vary = D.ncs_cols + unactionable_features + ['Confirmed_Binary_DPN']
    features_to_vary = [
        c for c in dfXy.columns 
        if c not in columns_not_to_vary
        ]
    assert set(features_to_vary).isdisjoint(D.ncs_cols)
    assert set(features_to_vary).isdisjoint(unactionable_features)
    assert set(features_to_vary).isdisjoint(columns_not_to_vary)

    continuous_cols = dfXy.columns.difference(D.categorical_cols+['Confirmed_Binary_DPN']).to_list()
    # continuous_cols = [
    #     c for c in dfXy.columns 
    #     if c not in columns_not_to_vary + D.categorical_cols 
    #     ]

    # assert set(continuous_cols).isdisjoint(D.categorical_cols)
    # assert set(continuous_cols).isdisjoint(D.ncs_cols)
    # assert set(continuous_cols).isdisjoint(unactionable_features)
    # assert set(continuous_cols).isdisjoint(columns_not_to_vary)

    print('features to vary columns:\n', len(features_to_vary), features_to_vary)
    print('categorical columns:\n', len(D.categorical_cols), D.categorical_cols)
    print('continuous_columns:\n', len(continuous_cols), continuous_cols)

    # ### Load trained model splits from Explainability Stage
    first_repeat_trained_models = joblib.load(config_path / config.optimization.first_repeat_trained_models_filename)
    assert first_repeat_trained_models['rundate'] == config.optimization.rundate, f"{first_repeat_trained_models['rundate']} != {config.optimization.rundate}"
    assert first_repeat_trained_models['tag'] == config.optimization.tag
    print('rundate:', first_repeat_trained_models['rundate'])
    print('tag:', first_repeat_trained_models['tag'])
    print('split results summary:')
    # print(first_repeat_trained_models['summary'])
    split_results = first_repeat_trained_models['results']

    # ## Loop through model splits
    for midx in split_results.keys():

        if target_model_idx is not None and midx!=target_model_idx:
            # if this model index is not being resumed or reworked
            print(f"Skipping model {midx}...")             
            continue

        print(f"Processing results from model {midx}...")

        # ## Create output directory for this Model split
        split_output_dir = outputdir / f'split{midx}'
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # ## Extract saved variables from split
        best_params = split_results[midx]['metrics']['best_params']
        threshold = split_results[midx]['metrics']['threshold']
        print('best_params:', best_params)        
        print('scale_pos_weight:', best_params["scale_pos_weight"])        
        print('threshold:', threshold)
        
        # ## Extract Test Sets
        X_test = split_results[midx]['X_test']
        y_test = split_results[midx]['y_test']
        dfXy_test = pd.concat([X_test, y_test], axis=1)

        X_train = split_results[midx]['X_train']
        y_train = split_results[midx]['y_train']

        # convert categorical columns in X_train - needed in CatBoost for use in DiCE
        X_train[D.categorical_cols] = X_train[D.categorical_cols].astype(str)
        X_test[D.categorical_cols] = X_test[D.categorical_cols].astype(str)


        # refit model so we can set cat_features (needed in DiCE)
        model=  CatBoostClassifier(**best_params, 
                                cat_features=D.categorical_cols, 
                                verbose=0,
                                ).fit(X_train, y_train)

        # ### Wrap model so we can use a custom threshold
        wrapped_model = cf.CatBoostWrapper(model, threshold)
        cf.test_wrapped_model(model, wrapped_model, X_test, y_test, threshold, config.experiment.verbosity)

        ###  Define Global Permitted Range
        global_permitted_range = cf.get_global_permitted_range(
            dfXy, continuous_cols, config, midx, verbosity=1, savedir=split_output_dir)

        # ### Prepare DiCE Explainer Object
        d = dice_ml.Data(dataframe=dfXy_test, # use only the test set
                        continuous_features=continuous_cols,                  
                        categorical_features=D.categorical_cols,
                        permitted_range = global_permitted_range, 
                        outcome_name='Confirmed_Binary_DPN')
        m = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type="classifier")
        dexp = dice_ml.Dice(d, m, method=config.dice.method)

        # #### Instances of Interest
        print(f"Getting Instances of Interest for model {midx} with delta={config.dice.threshold_delta}...")
        ioi_df, display_cols = cf.get_instances_of_interest(
            wrapped_model, X_test, y_test, config, midx,
            threshold=threshold, 
            delta=config.dice.threshold_delta,
            savedir=split_output_dir)
        qindices = ioi_df.index.to_list()

        if global_only:
            # ### Get Global Importances
            print(f"Getting global importance from model {midx}...")
            print('Start:', time.strftime("%m-%d %H:%M:%S", time.localtime()))
            cf.get_global_importance(dexp, D, X_test, config, midx,
                                    features_to_vary, threshold, global_permitted_range,   
                                    highlight_features=actionable_features, 
                                    filename_suffix="", savedir=split_output_dir, 
                                    n_cpus=-1)
            print('End:', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        else:
            # #### Produce reports for each Instance of Interest
            for qidx in qindices:
                if skip_instances: 
                    # skip target instances (because of error, or reports already exist)
                    if qidx in target_instance_indices:
                        print(f"Skipping instance {qidx}...")        
                        continue

                elif redo_instances:
                    if qidx not in target_instance_indices:
                        # redo only target instances; skip the rest
                        print(f"Skipping and not redoing instance {qidx}...")
                        continue

                print(f"Generating counterfactual analysis for record {qidx}")
                try:
                    cf.generate_local_cf_reports(dfXy, dexp, ioi_df, qidx, Xfull, 
                                            features_to_vary=features_to_vary, 
                                            config=config,
                                            split_index=midx,
                                            threshold=threshold,
                                            categorical_cols=D.categorical_cols,
                                            continuous_cols=continuous_cols,
                                            remove_invalid_progressive_cfs=True,
                                            generation_timeout=gen_timeout,
                                            savedir=split_output_dir
                                            )
                except Exception as e:
                    print(f'Error: Generating counterfactual analysis for record {qidx}')
                    print(f'{e}')
                    continue                
        
if __name__ == "__main__":
    main()