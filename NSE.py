# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:07:36 2019

@author: SHANKAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

from nsepy.history import get_history

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV

infy = get_history(symbol='INFY', start = datetime.date(2018, 1, 1), end = datetime.date(2019, 2, 21))
tcs  = get_history(symbol='TCS' , start = datetime.date(2018, 1, 1,), end = datetime.date(2019, 2, 21))

#infy['index']
#infy.info()

plt.figure(figsize=(16,8))
plt.plot(infy['Close'], label='Close Price History')

plt.figure(figsize=(16,8))
plt.plot(tcs['Close'], label='Close Price History')

# Closing Price is target variable - Infy
X = infy.drop(['Symbol', 'Series', 'Close'], axis=1)
y = infy['Close']

# Closing Price is target variable - TCS
X1 = tcs.drop(['Symbol', 'Series', 'Close'], axis=1)
y1 = tcs['Close']

# Feature Selection Process using RFECV for Infosys
estimator = LinearRegression()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.ranking_

# Feature Selection Process using RFECV for Infosys
estimator = LinearRegression()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X1, y1)
selector.ranking_

# Infosys Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

clf = LinearRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))
print(mean_absolute_error(y_test, pred))

# TCS Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=0)

clf = LinearRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))
print(mean_absolute_error(y_test, pred))


# Hyper Paramaeter Optimization - Tuning the model - Infosys

mod = SVR()
param_grid = {
        'kernel' : ['rbf', 'linear', 'poly', 'sigmoid'],
        'tol' : [0.001, 0.002, 0.003]
        }
clasif = GridSearchCV(mod, param_grid, cv=3)
clasif.fit(X, y)
clasif.best_params_
#clasif.best_score_

model = SVR()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(r2_score(y_test, pred))
