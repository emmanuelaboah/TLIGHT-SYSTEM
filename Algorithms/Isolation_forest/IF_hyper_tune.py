
import joblib
import csv
import time
import os

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import model_selection
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from ..Utils import utils


# Load preprocessed Dataset
path = ../Dataset/train_data/
train_df = pd.read_csv(path + "train_data.csv", index_col=0)  # train data
test_df = pd.read_csv(path + "test_data_X.csv")  # modify as required - there are 5 different test sets in the dataset directory

# Normalize data
scaler = MinMaxScaler()
trans_pipeline = Pipeline([("scaler", MinMaxScaler())])

#train_data = scaler.fit_transform(train)
train_data = trans_pipeline.fit_transform(train_df)

# Assumption: Train data represents the normal operation of the
# TLIGHT system
# train data class
train_class = np.ones(len(train_df))
train_class = pd.DataFrame(train_class)

def hyper_param_tuning(train, train_df_class, hyper_param, cv=3):

    clf = IsolationForest(random_state=42)
    grid_dt_estimator = model_selection.RandomizedSearchCV(clf,
                                                     hyper_param,
                                                     scoring="accuracy",
                                                     refit=True,
                                                     cv=3,
                                                     return_train_score=True)

    grid_dt_estimator.fit(train, train_df_class)

    return grid_dt_estimator.best_estimator_


# Ranges of Hyper-parameters
param_grid = {'n_estimators': list(range(100, 800)),
              'max_samples': list(range(100, 500)),
              'contamination': ['auto', 0.01, 0.1, 0.2],
              'max_features': [5,10,15],
              'bootstrap': [True, False],
              'n_jobs': [-1]}

# Best Hyper-parameter
best_hyper_param = hyper_param_tuning(train_data, train_class, param_grid)
