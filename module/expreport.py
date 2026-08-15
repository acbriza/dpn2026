"""
    Produce final models and explainability reports
"""
import pandas as pd
import matplotlib
# must precede the first pyplot import -- here and, more importantly, the one inside
# utils2.explainability, which is imported further down
matplotlib.use('Agg')

from pathlib import Path
import shutil
from datetime import datetime
import joblib
from catboost import CatBoostClassifier


import sys

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

    # Reports are produced as one set from one config, so they are protected as a set.
    # Scoped to '{model.code}_*' so it sees the report files but not retrained_models.joblib
    # or the copied config. Returning here skips the data load, the model refit, the SHAP
    # loop and the four bootstrap passes -- a run either regenerates everything or costs
    # nothing.
    existing_reports = sorted(outputdir.glob(f'{config.model.code}_*'))
    if existing_reports and not overwrite_reports:
        print(f'{len(existing_reports)} report files already exist in {outputdir}.')
        print('Pass "overwrite" on the command line to regenerate them. Nothing to do.')
        return

    # #### Copy config file to output directory
    source = config_path / config_filename
    destination = outputdir / config_filename
    shutil.copy(source, destination)

    # ## Data Loading
    # dataset_path in the yml is written relative to module/ ('../dataset/...'), so
    # resolve it against script_dir rather than slicing off the '../' and relying on
    # the caller's working directory.
    dataset_path = (script_dir / config.data.dataset_path).resolve()
    D = DPN_data(str(dataset_path))
    D.load(classification=config.experiment.classification_type)
    dfdpn = D.df
    data_cols = dfdpn.drop(D.non_data_cols, axis=1, errors="ignore").columns
    no_ncs_datacols = [c for c in data_cols if c not in D.ncs_cols]
    X = dfdpn[no_ncs_datacols]
    # D.load() sets the target column from config.experiment.classification_type
    # ('Confirmed_Binary_DPN' for binary, 'DPN_Status' for multiclass). Hardcoding the
    # binary name here silently ignored the config for any other classification_type.
    # Matches how optreport.py and selreport.py resolve the target.
    y = dfdpn[D.get_target_column()]
    print(f'X: {X.shape}, y:{y.shape}')


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
    assert first_repeat_trained_models['tag'] == config.optimization.tag, \
        f"{first_repeat_trained_models['tag']} != {config.optimization.tag}"
    print('rundate:', first_repeat_trained_models['rundate'])
    print('tag:', first_repeat_trained_models['tag'])
    print('split results summary:')
    # print(first_repeat_trained_models['summary'])
    split_results = first_repeat_trained_models['results']

    # The importance heat map below labels each model's get_feature_importance()
    # output with X.columns. That is only correct if the persisted splits were built
    # from the same feature set in the same order as the X assembled above -- a
    # coupling nothing enforces, since the splits come from a separate stage.
    persisted_cols = list(next(iter(split_results.values()))['X_train'].columns)
    assert list(X.columns) == persisted_cols, (
        f"feature mismatch between dataset and persisted splits:\n"
        f"  dataset : {list(X.columns)}\n"
        f"  splits  : {persisted_cols}"
    )

    # Both this script and explainability.py index splits positionally, via
    # range(len(split_results)), and pair them against the trained_models list.
    # The joblib stores a dict, so normalise once here instead of depending on its
    # keys happening to be 0..n-1 in order. This also matches the "list of dict"
    # contract declared in the explainability.py docstrings.
    if isinstance(split_results, dict):
        split_results = [split_results[k] for k in sorted(split_results)]

    retrained_models_fullpath = outputdir / 'retrained_models.joblib'
    if retrained_models_fullpath.is_file():
        trained_models = joblib.load(retrained_models_fullpath)
    else:
        # ## Loop through model splits
        trained_models = []
        for midx in range(len(split_results)):

            # ## Extract saved variables from split
            best_params = split_results[midx]['metrics']['best_params']
            threshold = split_results[midx]['metrics']['threshold']
            print('best_params:', best_params)        
            print('scale_pos_weight:', best_params["scale_pos_weight"])        
            print('threshold:', threshold)    

            X_train = split_results[midx]['X_train']
            y_train = split_results[midx]['y_train']

            # NOTE: the categorical columns were previously cast to str here "for use in
            # DiCE". Removed: DiCE is not used in this script, cat_features is not set on
            # the fit below, and CatBoost parses the strings back to floats -- verified
            # bit-identical predictions across all 4 splits with and without the cast.

            # refit model to attach feature names (the optimization stage fit on
            # X_train.values, so the stored models only carry positional names)
            print(f'Retrainining model {midx}...')
            # best_params already carries 'verbose': 0 from the optimization stage's
            # param_space, so passing verbose=0 as a separate keyword is a duplicate.
            # Merging keeps verbose=0 guaranteed even if param_space stops setting it.
            # random_state is set on the base estimator during optimization, not in
            # param_space, so it is absent from best_params and the refit would silently
            # use CatBoost's default seed (0). Restoring it reproduces the stored models
            # exactly; without it predictions drift by up to 0.086.
            model =  CatBoostClassifier(**{**best_params,
                                           'verbose': 0,
                                           'random_seed': config.experiment.random_seed},
                                    # cat_features=D.categorical_cols,
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
    feature_names = X.columns
    for s in range(len(split_results)):
        model = trained_models[s]
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

    # DCA, Calibration, Brier
    for s in range(len(split_results)): 
        X_test = split_results[s]['X_test']
        y_test = split_results[s]['y_test']
        model = trained_models[s]
        # model_threshold = split_results[s]['metrics']['threshold']
        # thresholds, nb = exp.plot_decision_curve_analysis(model, model_threshold, s, X_test, y_test, config, savedir=outputdir)

        # exp.plot_calibration_curve(model, s, X_test, y_test, config, savedir=outputdir, n_bins=5, strategy="quantile")
        # exp.plot_calibration_with_ci(model, s, X_test, y_test, config, savedir=outputdir)
        exp.plot_brier_curve(model, s, X_test, y_test, config, savedir=outputdir)

    # Pooled plots
    # thresholds, net_benefits, pooled_df = exp.plot_pooled_decision_curve_analysis(
    #     split_results=split_results,
    #     trained_models=trained_models,
    #     config=config,
    #     savedir=outputdir
    # )    

    # pooled_df, cal_stats = exp.plot_pooled_calibration_curve(
    #     split_results=split_results,
    #     trained_models=trained_models,
    #     config=config,
    #     n_bins=10,
    #     n_bootstrap=1000,
    #     confidence_level=0.95,
    #     savedir=outputdir
    # )

    # Pool once
    pooled_df, all_probs, all_labels, prevalence, N = exp.pool_fold_predictions(
        split_results, trained_models
    )
    exp.print_distribution_audit(
        split_results, trained_models, all_probs, all_labels, prevalence, N
    )

    # Four separate figures, each reusing the same pooled_df
    pooled_df, _auroc_stats = exp.plot_pooled_auroc(
        split_results, trained_models, config,
        pooled_df=pooled_df, savedir=outputdir
    )
    pooled_df, _auprc_stats = exp.plot_pooled_auprc(
        split_results, trained_models, config,
        pooled_df=pooled_df, savedir=outputdir
    )
    pooled_df, _cal_stats = exp.plot_pooled_calibration_curve(
        split_results, trained_models, config,
        pooled_df=pooled_df, savedir=outputdir
    )
    pooled_df, _dca_stats = exp.plot_pooled_decision_curve_analysis(
        split_results, trained_models, config,
        pooled_df=pooled_df, savedir=outputdir
    )

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