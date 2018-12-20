# AI

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

## Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])
X[:, 2] = labelencoder2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

## Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6 , activation= 'relu' , input_dim = 11 , init='uniform'))
classifier.add(Dense(output_dim = 6 , activation= 'relu' , input_dim = 11 , init='uniform'))
classifier.add(Dense(output_dim = 1 , activation= 'sigmoid' , input_dim = 11 , init='uniform'))
classifier.compile(optimizer='adam',loss = 'binary_crossentropy' ,metrics=['accuracy'])

classifier.fit(X_train,y_train , batch_size = 10 , nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > .5)

from sklearn.metrics import confusion_matrix
cost = confusion_matrix(y_test,y_pred)


