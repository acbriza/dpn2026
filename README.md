dependencies setup:

python version 3.12.1
pip version 25.1.1

install these dependencies
"pip install numpy==1.26.4 scipy==1.13.0 pandas==2.2.2 scikit-learn==1.4.2 dice-ml==0.9 xgboost"


COUNTERFACTUAL ANALYSIS (counterfactuls.ipynb and .py)
======================================================
Read Config File
Set output directory for this model
Load data
Load trained model splits
Do for all splits
    Load model, best params, train-test splits
    Wrap model 
    Define feature lists: all features, continuous cols
    Perform Global Counterfactual Analysis for this split
        Get Global Permitted Range *
        Plot Global Importances *
    Get Instances of Interest for this split *
    For each Instance of Interest
        Get Instance Permitted Range *
        Generate Counterfactuals *
        Plot Local Heatmap *
        Check valid counterfactuals
            Get most change features * 
            Get L1 & L2 Differences *
        Sufficiency ?
        Necessity ?
Get average for 3 splits
    Global Importances
    L1 & L2 Differences

