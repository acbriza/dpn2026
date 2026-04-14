dependencies setup:

python version 3.12.1
pip version 25.1.1

install these dependencies
"pip install numpy==1.26.4 scipy==1.13.0 pandas==2.2.2 scikit-learn==1.4.2 dice-ml==0.9 xgboost"

Method
=======================================================================================
1. Run multiple models with default parameters, select one model
2. Do repeated kfold with optimization through AUPRC and thresholding through f-beta score for selected model
3. Split data into k-folds (k=3 or 4), apply optimization (AUPRC) and thresholding (fbeta) 
4. Generate CFs through models from the models for each fold

Sequence
=======================================================================================
selection.ipynb - Feature and Model selection 
hparam_opt.ipynb - Repeated kfold of selected model for reporting
explainability.ipynb - Train selected model, Feature Importance, SHAP, ROC-AUC plots
cfreports.py - generate counterfactuals and reports via script
reports.ipynb - experiment for report generation of cfs in batch


COUNTERFACTUAL ANALYSIS (counterfactuls.ipynb and .py)
=======================================================================================
Read Config File
Set output directory for this model
Load data
Define feature lists: all features, continuous cols
Load trained model splits
Do for all splits
    Create output directory for split
    Load model, best params, train-test splits
    Retrain model with best params
    Wrap model for custom threshold
    Get Global Permitted Range *
    Prepare DiCE Explainer Object using global permitted range and wrapped model
    Plot Global Importances *
    Get Instances of Interest for this split *
    For each Instance of Interest
        Get Instance Permitted Range *
        Generate Counterfactuals *
        Plot Local Heatmap *
        Check valid counterfactuals
            Get most changed features * 
            Get L1 & L2 Differences *
        Sufficiency ?
        Necessity ?

Report Summaries
    Perform Summary Report for All Splits
        Average Global Importances
    For All Splits
        Perform Summary Report for All Instances
            Number of cfs
            Number of invalid
            L1 & L2
            Most Changed features
        

