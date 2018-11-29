# Data Preprocessing Template

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('Data.csv')
X = Data.iloc[:,:-1].values
Y = Data.iloc[:,3].values


# Deal with Missing Data -- Imputer

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN' , strategy='mean',axis=0)

X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Categorical Data -- LabelEncoder

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

X[: , 0] = lb.fit_transform(X[:,0])

lb_y = LabelEncoder()
Y = lb_y.fit_transform(Y)

# Dummy Encoder -- OneHotEncoder 

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()


# Split Data To Test and Train -- train_test_split

from sklearn.model_selection import train_test_split

X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,train_size=.8 , test_size = .2 , random_state=42)

# Scalling Feature 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
















