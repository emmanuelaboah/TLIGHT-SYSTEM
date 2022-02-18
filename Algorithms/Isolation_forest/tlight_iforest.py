# Isolation Forest

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
from sklearn.metrics import classification_report, confusion_matrix
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
test_data = trans_pipeline.transform(test_df.iloc[:,:-1])


filename = 'iForest_model.sav'
def iForest(train_, test_, fname, contamination=0.05, max_features=10,
          max_samples=180, n_estimators=156, random_state=42):

    clf = IsolationForest(contamination=contamination,
                          max_features=max_features,
                          max_samples=max_samples,
                          n_estimators=n_estimators,
                          random_state=random_state)

    start_train = time.time()
    clf.fit(train_)
    end_train = time.time() - start_train
    print("Training time is {:.2f}s".format(end_train))

    # save the trained model
    joblib.dump(clf, fname)

    # Binary prediction on train data
    train_pred = clf.predict(train_)
    train_pred = pd.DataFrame(train_pred)

    # Decision scores (train data)
    train_dec = clf.decision_function(train_)
    train_dec = pd.DataFrame(train_dec, columns=['decision score'])

    # Binary prediction (test data)
    start_test = time.time()
    test_pred = clf.predict(test_)
    end_test = time.time() - start_test

    print("\nTest time is {:.2f}s".format(end_test))
    test_pred = pd.DataFrame(test_pred)

    # Decision scores (test data)
    test_dec = clf.decision_function(test_)
    test_dec = pd.DataFrame(test_dec)
    print("Done!")

    return train_pred, test_pred, test_dec


def evaluate(pred_df, ground_truth):

    print("====confusion matrix======")
    print(confusion_matrix(ground_truth, pred_df))

    print("====classification report====")
    print(classification_report(ground_truth, pred_df))