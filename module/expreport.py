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
import joblib
from catboost import CatBoostClassifier


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


    # ======================================
    # Old code for retraining the model
    # start_time = datetime.now()    
    # print(f'Training models on data splits, started at: ', start_time.strftime("%m-%d %H:%M:%S"))
    # ksplit_trained_models =  exp.get_ksplit_trained_models(
    #     X, y, config,
    #     savedir=outputdir, 
    #     overwrite=overwrite_reports,        
    #     )
    # end_time = datetime.now()
    # elapsed = end_time - start_time
    # print(f"\nrundate and tag: ", ksplit_trained_models['rundate'], ksplit_trained_models['tag'])
    # print(f"summary:\n {ksplit_trained_models['summary']}")
    # print(f'Training models on data splits, took: {elapsed.total_seconds()/60:.2f}, ended at: ',  start_time.strftime("%m-%d %H:%M:%S"))

    # split_results = ksplit_trained_models['results']
    # ======================================

    # ### Load trained model splits from Explainability Stage
    first_repeat_trained_models = joblib.load(config_path / config.optimization.first_repeat_trained_models_filename)
    assert first_repeat_trained_models['rundate'] == config.optimization.rundate, f"{first_repeat_trained_models['rundate']} != {config.optimization.rundate}"
    assert first_repeat_trained_models['tag'] == config.optimization.tag
    print('rundate:', first_repeat_trained_models['rundate'])
    print('tag:', first_repeat_trained_models['tag'])
    print('split results summary:')
    # print(first_repeat_trained_models['summary'])
    split_results = first_repeat_trained_models['results']    

    retrained_models_fullpath = outputdir / 'retrained_models.joblib'
    if retrained_models_fullpath.is_file():
        trained_models = joblib.load(retrained_models_fullpath)
    else:
        # ## Loop through model splits
        trained_models = []
        for midx in split_results.keys():

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
            print(f'Retrainining model {midx}...')
            model =  CatBoostClassifier(**best_params, 
                                    # cat_features=D.categorical_cols, 
                                    verbose=0,
                                    ).fit(X_train, y_train)
            trained_models.append(model)
        joblib.dump(trained_models, retrained_models_fullpath)

    # Feature Importances (Individual plots)
    # for s in range(len(split_results)): 
    #     model = trained_models[s]
    #     feature_names = X.columns
    #     exp.plot_importances(D, model, s, feature_names, config, 
    #                         minimum=None, limit=None, 
    #                         savedir=outputdir)
        
    # Feature Importances (Heat Maps for all splits)
    all_importances = {}
    for s in range(len(split_results)):
        model = trained_models[s]
        feature_names = X.columns
        importances = model.get_feature_importance()
        all_importances[f'Model {s}'] = pd.Series(importances, index=feature_names)

    exp.plot_importances_heatmap(D, all_importances, feature_names, config,
                                minimum=None, limit=None,
                                savedir=outputdir)
                
    # # ROC-AUC (individual plots)
    # for s in range(len(split_results)): 
    #     model = trained_models[s]
    #     X_test = split_results[s]['X_test']
    #     y_test = split_results[s]['y_test']
    #     y_proba = model.predict_proba(X_test)[:,1]
    #     exp.plot_roc_auc(y_test, y_proba, s, config, outputdir);        

    # # ROC-AUC (overlapping plots)
    # Collect ROC data for all splits
    roc_data = []
    for s in range(len(split_results)):
        model = trained_models[s]
        X_test = split_results[s]['X_test']
        y_test = split_results[s]['y_test']
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_data.append((y_test, y_proba))

    exp.plot_roc_auc_overlapping(roc_data, config, outputdir)

    # DCA
    for s in range(len(split_results)): 
        X_test = split_results[s]['X_test']
        y_test = split_results[s]['y_test']
        model = trained_models[s]
        thresholds, nb = exp.plot_decision_curve_analysis(model, s, X_test, y_test, config, savedir=outputdir)

    # SHAP Individual Plots
    # for s in range(len(split_results)): 
    #     model = trained_models[s]
    #     X_test = split_results[s]['X_test']
    #     y_test = split_results[s]['y_test']
    #     exp.plot_shap(D, model, s, config, X_test, savedir=outputdir)

    # SHAP (Heat Maps for all splits)
    all_shap_importances = {}
    for s in range(len(split_results)):
        model = trained_models[s]
        X_test = split_results[s]['X_test']
        exp.collect_shap(D, model, s, config, X_test, all_shap_importances)

    exp.plot_shap_heatmap(D, all_shap_importances, config, savedir=outputdir)        

    # AUPRC CURVE
    y_test_list = []
    y_proba_list = []
    for s in range(len(split_results)): 
        model = trained_models[s]
        X_test = split_results[s]['X_test']
        y_test = split_results[s]['y_test']
        y_proba = model.predict_proba(X_test)[:,1]# y_test: true labels
        y_test_list.append(y_test)
        y_proba_list.append(y_proba)

    exp.plot_cv_auprc(y_test_list, y_proba_list, config, savedir=outputdir)        

if __name__ == "__main__":
    main()