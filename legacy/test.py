import pandas as pd
import json


from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

from pprint import pprint

from tqdm import tqdm

from eda import EDA

from dataload import DPN_data
from module.models.optimizers import grid_search_cv_binary

#import vmodels

print("Hello World!")

score_aggregation='weighted_avg'


D = DPN_data("../dataset/Sudoscan Working File with Stats.xlsx")
D.load()
df = D.df
data_cols = list(set(df.columns) - set(D.non_data_cols))
print(data_cols)

X = df[data_cols]
y = df['Confirmed_Binary_DPN']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
y_val

estimator = xgb.XGBClassifier()

with open('model_configs/param_grids/xgb/xgb_binary_param_grid.json', 'r') as file:
    json_string = json.load(file)

# uncomment parameter grid to use
param_grid = json_string
# param_grid = opt_param_grid

# uncomment scoring to use
scoring='youden_index'

binary_cv_grid = tqdm(grid_search_cv_binary(
    estimator, param_grid,  (X_train, y_train),
    scoring=scoring,
    labels=["Confirmed", "Non-confirmed"],
    verbosity=1,
    cv_splits=5))

print(binary_cv_grid.best_params_)