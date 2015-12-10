#!/usr/bin/env python
#title          :mushroom.py
#description    :A general classification task
#author         :Jakub Klapacz (Based on Henry Lin's implementation)
#version        :0.0.1
#python_version :2.7.6
#================================================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC



data = np.loadtxt("mushroom.data", delimiter=",", dtype=str)
X = data[:, 1:]
y = data[:, 0]

# Have to do some data preprocessing on each column. Each column
# is of categorical type, so we have to remap it into what is
# called a "one hot encoding". Note that if you use the R package
# "rpart", you don't need to do this data preprocessing.
for j, col in enumerate(X.T):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(col)
    X[:, j] = labels
X = X.astype(int)
encoder = OneHotEncoder()
encoded = encoder.fit_transform(X).toarray()

# Allowing the testing set to be 0.25 of the original set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=888)

mushroomforest = RandomForestClassifier()
mushroomforest.fit(X_train, y_train)
mushroom_predictions = mushroomforest.predict(X_test)

score = accuracy_score(y_test, mushroom_predictions)
print "Accuracy of Random Forest Classifier: ", score
matrix = confusion_matrix(y_test, mushroom_predictions)
print "Confusion matrix: \n\tE\tP\nE\t", matrix[0], "\nP\t", matrix[1]


