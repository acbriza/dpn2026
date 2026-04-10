#%% md
# # DPN
#%%
import json
import seaborn as sns
import warnings
from pprint import pprint

import dice_ml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from module.models.optimizers import grid_search_cv_multiclass, optimize

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mord import LogisticAT

from sklearn.metrics import confusion_matrix

from module.dataload import DPN_data
from module.eda import EDA

import xgboost as xgb

import time
#%%
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)  # decimal places for outputs from numpy
pd.set_option("display.precision", 3)  # decimal places for outputs from pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
def plot_heatmap(corr_matrix, figsize=(6, 4)):
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
#%% md
# ### Data Loading
#%%
D = DPN_data("../dataset/Sudoscan Working File with Stats.xlsx")
D.load(classification="multiclass")
#%%
df = D.df
data_cols = df.drop(D.non_data_cols, axis=1, errors="ignore").columns
#%% md
# ## Multiclass Classification Classes
# ['Negative', 'Possible', 'Probable', and 'Confirmed']
# 
#%% md
# ### Data Inspection
#%%
X = df[data_cols]
y = df['DPN_Status']
#%% md
# ### Train Test Split
#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
#%%
verbosity = 1
#%% md
# ### Colinear Feature Elimination
#%%
# Assuming X and y are already loaded and ready
# X = your features DataFrame, y = your target Series

# Step 1: Get the list of features to drop
features_to_drop = EDA.get_features_to_drop(X, y, threshold=0.8)

# Step 2: Print the results and your next steps
if features_to_drop:
    print("Features recommended for dropping due to high correlation:")
    pprint(features_to_drop)

    # Step 3: Create a new DataFrame with the features dropped
    X_reduced = X.drop(columns=features_to_drop)

    print("\nShape of original X:", X.shape)
    print("Shape of reduced X:", X_reduced.shape)
else:
    print("No features were identified for dropping with a correlation threshold of 0.8.")
#%%
pruned_df = df.drop(columns=features_to_drop)

X_pr = pruned_df.drop(columns=['DPN_Status'])
y_pr = pruned_df['DPN_Status']

X_pr_train, X_pr_val, y_pr_train, y_pr_val = train_test_split(X_pr, y_pr, test_size=0.25, random_state=0, stratify=y)
#%%
def set_splits(model_name):
    if model_name == "logistic_regression":
        return X_pr_train, y_pr_train, X_pr_val, y_pr_val
    else:
        return X_train, y_train, X_val, y_val
#%% md
# ### DICT OF INITIAL MODELS
#%% md
# Logistic Regression Scaling
#%%
logistic_regression = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression())
])

mord = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticAT())
])
#%%
initial_models = {
    "dummy": DummyClassifier(),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "xgb": xgb.XGBClassifier(),
    "logistic_regression": logistic_regression,
    "mord_logisticat": mord
}
#%% md
# ### RUNNING EACH MODEL
#%%
initial_model_runs = {}
for model_name, model in initial_models.items():
    
    X_train_temp, y_train_temp, X_val_temp, y_val_temp = set_splits(model_name)
     
    stats = EDA.multiclass_metrics(
        confusion_matrix(
            y_val_temp,
            model.fit(X_train_temp, y_train_temp).predict(X_val_temp)
        ),
        labels=D.binary_class_label, verbosity=verbosity)

    pprint(stats)

    initial_model_runs[model_name] = stats
    
pprint(initial_model_runs)
#%% md
# ### MULTICLASS GRID SEARCH CV
# 
# DICT OF ESTIMATORS
#%%
estimators = {
    "random_forest": (RandomForestClassifier(), set_splits("random_forest")),
    "decision_tree": (DecisionTreeClassifier(), set_splits("decision_tree")),
    "xgb": (xgb.XGBClassifier(), set_splits("xgb")),
    "logistic_regression": (logistic_regression, set_splits("logistic_regression")),
    "mord_logisticat": (mord, set_splits("mord"))
}
#%% md
# ### RUNNING MULTICLASS GRID SEARCH CV ON EACH ESTIMATOR
#%%
optimized_params = optimize(estimators, verbosity=verbosity)
#%% md
# 
#%% md
# ### DICT OF OMPTIMIZED MODELS
# 
# Generated parameter set is inputted into the classifiers as kwargs
#%%
logistic_regression = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(**optimized_params["logistic_regression"]))
])

mord = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticAT(**optimized_params["mord_logisticat"]))
])
#%%
optimized_models = {
    "xgb": xgb.XGBClassifier(**optimized_params["xgb"]),
    "random_forest": RandomForestClassifier(**optimized_params["random_forest"]),
    "decision_tree": DecisionTreeClassifier(**optimized_params["decision_tree"]),
    "logistic_regression": logistic_regression,
    "mord_logisticat": mord
}
#%%
best_youden = (None, 0)

for model_name, model in optimized_models.items():

    X_train_temp, y_train_temp, X_val_temp, y_val_temp = set_splits(model_name)

    stats = EDA.multiclass_metrics(
        confusion_matrix(
            y_val_temp,
            model.fit(X_train_temp, y_train_temp).predict(X_val_temp)
        ),
        labels="DPN_Status", verbosity=1)

    print(f"Statistics for optimized {model_name} model")
    best_youden = (model_name, stats['weighted_avg']['youden_index']) if stats['weighted_avg']['youden_index'] > best_youden[1] else best_youden
    pprint(stats)
    
print(f"The best model is: {best_youden[0]} with a youden_index of {best_youden[1]}")
#%% md
# ### Generating Counterfactuals for Each Model
#%%
cf_explainers = {}

for model_name, model in optimized_models.items():

    X_train_temp, y_train_temp, X_val_temp, y_val_temp = set_splits(model_name)
    
    d = dice_ml.Data(dataframe=df, continuous_features=df.columns.drop('DPN_Status').tolist(), outcome_name='DPN_Status')
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")

    exp = dice_ml.Dice(d, m, method="genetic")

    print(f"generating counterfactuals for the {model_name} model")

    e1 = exp.generate_counterfactuals(
        X_val_temp, total_CFs=3, desired_class=3)

    cf_explainers[model_name] = e1
#%%
# e1.visualize_as_dataframe()
#%%
# Use the bets multiclass model to plot ROC curve and AUC graphs
best_multiclass_model = optimized_models[best_youden[0]]

X_train_temp, y_train_temp, X_val_temp, y_val_temp = set_splits(best_youden[0])

print(f"\n--- Plotting ROC for the Best Multiclass Model {best_youden[0]} ---")
EDA.plot_roc_multiclass_from_model(
    model=best_multiclass_model, # Replace with your actual multiclass model (e.g., mord.LogisticAT)
    X_val=X_val_temp, # Replace with your actual X_val for multiclass
    y_val=y_val_temp, # Replace with your actual y_val for multiclass
    class_names=['Negative', 'Possible', 'Probable', 'Confirmed'] # Your actual class names
)
#%%
import shap

# --- Automated SHAP Analysis ---
for model_name, model in optimized_models.items():
    print(f"🚀 Analyzing SHAP values for: {model_name}")

    model.fit(X_train, y_train)

    # Define a simple predict function for the current model
    # This closure ensures 'model' refers to the correct model in each iteration
    def current_model_predict(X):
        # For classifiers, .predict_proba is often preferred for SHAP for better interpretability
        # especially for multi-class, but .predict is also valid.
        # We'll use .predict here as per your original request, but it's good to note.
        return model.predict(X)

    # Use a masker: your input DataFrame
    masker = shap.maskers.Independent(X_val)

    # Create explainer with custom predict function
    # Specify the masker for consistency
    explainer = shap.Explainer(current_model_predict, masker=masker)

    # Compute SHAP values
    shap_values = explainer(X_val)

    # Plot the summary plot for the current model
    # show=False prevents immediate display, allowing title to be set
    shap.summary_plot(shap_values, X_val, show=False, plot_type="bar")
    plt.title(f"SHAP Summary Plot for {model_name.replace('_', ' ').title()}")
    plt.tight_layout() # Adjust layout to prevent title overlap
    plt.show() # Display the plot for the current model
    print("-" * 50) # Separator for clarity
