#  Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
compus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ' ,dataset['Review'][i])
    review = review.lower();
    review = review.split()
    review = [word for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    compus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(compus).toarray()
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size= .8,test_size=.2)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10 , criterion = 'entropy',random_state=42)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)