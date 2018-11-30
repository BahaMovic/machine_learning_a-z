# Multiple Linear Regression


# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('50_Startups.csv')
X = Data.iloc[:,:-1].values
Y = Data.iloc[:,4].values


# Categorical Data And Dummy Variables 

from sklearn.preprocessing import OneHotEncoder , LabelEncoder
le = LabelEncoder()
X[: , 3] = le.fit_transform(X[:,3])
ohe = OneHotEncoder( categorical_features= [3])
X = ohe.fit_transform(X).toarray()


# Split Data To Test and Train -- train_test_split

from sklearn.model_selection import train_test_split
X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,train_size=.8 , test_size = .2 , random_state=42)


# Implementation of Multiple Linear Regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)


# Test Predict

y_pred = lr.predict(X_test)








