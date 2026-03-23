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
            Number of invalide
            L1 & L2
            Most Changed features

