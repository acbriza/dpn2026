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


def main():
    if len(sys.argv) < 2:
        print("Usage: python cfreports.py <config file>")
        print("Usage: python cfreports.py <config file> <skip_instances> <model-split-idx> <instance-indices:>")
        print("Usage: python cfreports.py <config file> <redo_instances> <model-split-idx> <instance-indices:>")
        print("e.g.   python cfreports.py bin_cf_final.yml   --> redo all reports")
        print("e.g.   python cfreports.py bin_cf_final.yml skip_instances 2 53,67--> Do not overwrite reports of model 2, redo all instances but SKIP 53 & 67")
        print("e.g.   python cfreports.py bin_cf_final.yml redo_instances 2 53,67--> Do not overwrite reports of model 2, redo ONLY instances 53 & 67")
        sys.exit(1)

    if len(sys.argv)==2:
        # we only received a config file: run all experiments
        rework = False
        resume = False

    if len(sys.argv)>=3:
        skip_instances = sys.argv[2]=='skip_instances'
        redo_instances = sys.argv[2]=='redo_instances'

    target_model_idx = None
    target_instance_indices = [] 
    if len(sys.argv)>=4:
        # we'll rework this model but not overwrite existing outputs
        target_model_idx = int(sys.argv[3])
    if len(sys.argv)>=5:
        # we'll skip these specific instances
        target_instance_indices = sys.argv[4]
        target_instance_indices = target_instance_indices.split(',')
        target_instance_indices = [int(i) for i in target_instance_indices]

    # config_filename =  "bin_cf_final.yml"
    config_filename = sys.argv[1]

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
    ksplit_trained_models = joblib.load(config_path / config.explainability.ksplit_trained_model_results_file)
    assert ksplit_trained_models['rundate'] == config.explainability.rundate, f"{ksplit_trained_models['rundate']} != {config.explainability.rundate}"
    assert ksplit_trained_models['tag'] == config.explainability.tag
    print('rundate:', ksplit_trained_models['rundate'])
    print('tag:', ksplit_trained_models['tag'])
    print('split results summary:')
    print(ksplit_trained_models['summary'])
    split_results = ksplit_trained_models['results']

    # ## Loop through model splits
    for midx in range(len(split_results)):

        if target_model_idx is not None and midx!=target_model_idx:
            # if this model index is not being resumed or reworked
            print(f"Skipping model {midx}...")             
            continue

        print(f"Processing results from model {midx}...")

        # ## Create output directory for this Model split
        split_output_dir = outputdir / f'split{midx}'
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # ## Extract saved variables from split
        threshold = split_results[midx]['threshold']
        best_params = split_results[midx]['best_params']
        
        # ## Extract Train and Test Sets
        X_test = split_results[midx]['X_test']
        y_test = split_results[midx]['y_test']
        dfXy_test = pd.concat([X_test, y_test], axis=1)

        X_train = split_results[midx]['X_train']
        X_test = split_results[midx]['X_test']

        # convert categorical columns in X_train - needed in CatBoost for use in DiCE
        X_train[D.categorical_cols] = X_train[D.categorical_cols].astype(str)
        X_test[D.categorical_cols] = X_test[D.categorical_cols].astype(str)

        y_train = split_results[midx]['y_train']
        y_test = split_results[midx]['y_test']

        # refit model so we can set cat_features (needed in DiCE)
        model=  CatBoostClassifier(**best_params, 
                                cat_features=D.categorical_cols, 
                                verbose=0
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
        ioi_df, display_cols = cf.get_instances_of_interest(
            wrapped_model, X_test, y_test, config, midx,
            threshold=threshold, delta=0.2, savedir=split_output_dir)
        qindices = ioi_df.index.to_list()

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
            cf.generate_local_cf_reports(dfXy, dexp, ioi_df, qidx, 
                                    features_to_vary=features_to_vary, 
                                    config=config,
                                    split_index=midx,
                                    threshold=threshold,
                                    categorical_cols=D.categorical_cols,
                                    continuous_cols=continuous_cols,
                                    remove_invalid_progressive_cfs=True,
                                    savedir=split_output_dir
                                    )

        
        # ### Get Global Importances
        print(f"Getting global importance from model {midx}...")
        print('Start:', time.strftime("%H:%M:%S", time.localtime()))
        cf.get_global_importance(dexp, D, X_test, config, midx,
                                features_to_vary, threshold, global_permitted_range,   
                                highlight_features=actionable_features, 
                                filename_suffix="", savedir=split_output_dir, 
                                n_cpus=-1)
        print('End:', time.strftime("%H:%M:%S", time.localtime()))

if __name__ == "__main__":
    main()