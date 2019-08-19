#!/usr/bin/env python
# coding: utf-8

#To predict quality of Air Pollution
#Author: Akshay Mattoo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df_train = pd.read_csv('pred_air_quality/Train.csv')
df_test  = pd.read_csv('pred_air_quality/Test/Test.csv')


#Store the training data
train_data = df_train.values

Y_train = train_data [1:,-1]
X_train = train_data [1:,:-1]

X_train = np.mat(X_train)
Y_train = np.mat(Y_train)
Y_train = Y_train.reshape ((1599, 1))


#Obtain the weight matrix
def getWeight (query_point, X_train, tau):
    M = X_train.shape[0]
    W = np.mat(np.eye(M))

    for i in range (M):
        xi = X_train[i]
        x = query_point
        W[i,i] = np.exp((np.sum(xi-x) ** 2)/(-2*tau*tau)) #Compute the weight for each query point and store it in the diagonal Weight matrix

    return W



X_test = df_test.values
X_test = X_test[0:,:]
X_test = np.mat(X_test)
X_test.size


def predict (X_train, Y_train, query_point, tau):
    ones = np.ones ((X_train.shape[0], 1))
    X_ = np.hstack((X_train, ones))
    
    qx = np.hstack((query_point, np.ones((query_point.shape[0], 1))))

    W = getWeight(qx, X_, tau)
   
    theta = np.linalg.pinv(X_.T*(W*X_))*(X_.T*(W*Y_train))
    pred = np.dot(qx, theta)
    return theta, pred


with open ("result.csv",'w') as f:
    f.write("Id,target \n")
    for i in range(0, 400, 1):
        theta, pred = predict (X_train, Y_train, X_test[i], 0.1)
        f.write(str(i))
        f.write(",")
        f.write(str(pred)[2:-2])
        f.write("\n")
