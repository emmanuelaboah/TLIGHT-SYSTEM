# ONE-CLASS NEURAL NETWORK

import os
import time
import random
import csv

import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from itertools import zip_longest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


df_ocnn_scores = {}
decision_scorePath = "../Dataset/"

def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):
    newfilePath = path + filename
    print("Writing file to ", path + filename)
    poslist = positiveScores.tolist()
    neglist = negativeScores.tolist()

    d = [poslist, neglist]
    export_data = zip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Training", "Testing"))
        wr.writerows(export_data)
    myfile.close()
    
    return

def tf_OneClass_NN_Relu(data_train, data_test):
    tf.reset_default_graph()

    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes
    h_size = 32  # Number of hidden nodes
    y_size = 1
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K * D + 1)
    rvalue = np.random.normal(0, 1, (len(train_X), y_size))
    nu = 0.04

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h = tf.nn.sigmoid(tf.matmul(X, w_1))
        yhat = tf.matmul(h, w_2)
        return yhat

    g = lambda x: relu(x)

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        #     y[y < 0] = 0
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g, r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)

        term1 = 0.5 * tf.reduce_sum(w ** 2)
        term2 = 0.5 * tf.reduce_sum(V ** 2)
        term3 = 1 / nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
        term4 = -r

        return term1 + term2 + term3 + term4

    # For testing the OCNN algorithm
    test_X = data_test

    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32, shape=(), trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1

    for epoch in range(100):
        # Train with each example
        sess.run(updates, feed_dict={X: train_X, r: rvalue})
        rvalue = nnScore(train_X, w_1, w_2, g)
        with sess.as_default():
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100 * 0.04)
        print("Epoch = %d, r = %f"
              % (epoch + 1, rvalue))

    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()

    rstar = rvalue
    sess.close()
    print("Session Closed!!!")

    pos_decisionScore = arrayTrain - rstar
    neg_decisionScore = arrayTest - rstar
    print(pos_decisionScore)

    write_decisionScores2Csv(decision_scorePath, "OC-NN_Relu.csv", pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore]


# =========== Main ==================
# Load preprocessed Dataset
path = ../Dataset/train_data/
train_df = pd.read_csv(path + "train_data.csv", index_col=0)  # train data
test_df = pd.read_csv(path + "test_data_X.csv")  # modify (X) as required - there are 5 different test sets in the dataset directory

# Normalize data
scaler = MinMaxScaler()
trans_pipeline = Pipeline([("scaler", MinMaxScaler())])

#train_data = scaler.fit_transform(train)
train_data = trans_pipeline.fit_transform(train_df)
test_data = trans_pipeline.transform(test_df.iloc[:,:-1])

ocnn_anomaly_scores = tf_OneClass_NN_Relu(train_data, test_data)
df_ocnn_scores["OCNN_Train"] = ocnn_anomaly_scores[0]
df_ocnn_scores["OCNN_Test"] = ocnn_anomaly_scores[1]

print("Finished!!")
