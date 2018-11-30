# Polynomial Regression

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('Position_Salaries.csv')
X = Data.iloc[:,1:2].values
Y = Data.iloc[:,2].values
'''
# Split Data To Test and Train -- train_test_split
from sklearn.model_selection import train_test_split

X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,train_size=.8 , test_size = .2 , random_state=42)
'''

from sklearn.preprocessing import PolynomialFeatures
pl = PolynomialFeatures(degree=6)
X_p = pl.fit_transform(X)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_p,Y)


plt.scatter(X,Y)
plt.plot(X,lr.predict(X_p),color = 'red')
plt.show()