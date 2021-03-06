# SVR

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('Position_Salaries.csv')
X = Data.iloc[:,1:2].values
Y = Data.iloc[:,2:3].values
'''
# Split Data To Test and Train -- train_test_split
from sklearn.model_selection import train_test_split

X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,train_size=.8 , test_size = .2 , random_state=42)
'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
Y = sc_y.fit_transform(Y)

from sklearn.svm import SVR
pl = SVR(kernel='rbf')
pl.fit(X,Y)




plt.scatter(X,Y)
plt.plot(X,pl.predict(X),color = 'red')
plt.show()