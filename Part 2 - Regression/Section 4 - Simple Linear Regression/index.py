# Simple Linear Regression

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('Salary_Data.csv')
X = Data.iloc[:,:-1].values
Y = Data.iloc[:,1].values

# Split Data To Test and Train -- train_test_split
from sklearn.model_selection import train_test_split

X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,train_size=.8 , test_size = .2 , random_state=42)


# Implementation of Simple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train , sample_weight=1)

# Test Predict

y_pred = regressor.predict(X_test)


# Draw Result

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.show()
