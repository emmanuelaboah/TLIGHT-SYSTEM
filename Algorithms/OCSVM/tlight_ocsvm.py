import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import svm

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

def ocSVM(train_, test_, nu=0.1, kernel='poly',
          degree=3, gamma=0.1, coef0=4):

    ocsvm = svm.OneClassSVM(nu=nu,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0)

    start_train = time.time()
    ocsvm.fit(train_)
    end_train = time.time() - start_train

    print("Training time is {:.2f}s".format(end_train))

    train_pred = ocsvm.predict(train_data)
    pd.DataFrame(train_pred)

    # Binary prediction
    start_test = time.time()
    test_pred = ocsvm.predict(test_)
    end_test = time.time() - start_test

    print("\nTest time is {:.2f}s".format(end_test))
    test_pred = pd.DataFrame(test_pred)

    # Decision scores
    test_decision_scores = ocsvm.decision_function(test_data)
    pd.DataFrame(test_decision_scores)
    print("Done!")

    return train_pred, test_pred, test_decision_scores


def evaluate(pred_df, ground_truth):

    print("====confusion matrix======")
    print(confusion_matrix(ground_truth, pred_df))

    print("====classification report====")
    print(classification_report(ground_truth, pred_df))






