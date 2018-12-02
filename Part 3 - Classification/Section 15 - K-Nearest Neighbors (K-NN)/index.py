# KNN

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('Social_Network_Ads.csv')
X = Data.iloc[:,2:4].values
Y = Data.iloc[:,4:5].values

# Split Data To Test and Train -- train_test_split
from sklearn.model_selection import train_test_split

X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,train_size=.8 , test_size = .2 , random_state=0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
sc_y = StandardScaler()
X_test = sc_y.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
pl = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)
pl.fit(X_train,Y_train)

y_pred = pl.predict(X_test)


# Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = Y_test , y_pred = y_pred)