# -*- coding: utf-8 -*-
"""diabetes-model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/gist/Ritax2003/1f7667cd1cf59942fc0b43d42648b6b9/diabetes-model.ipynb
"""
"""Importing the dependencies"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
#loading dataset to a panda dataframe
diabetes_dataset = pd.read_csv('diabetes-pima-indian-dataset.csv')
X = diabetes_dataset.drop(columns='Outcome',axis = 1)
Y = diabetes_dataset['Outcome']
#data standardization
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
X= standardized_data
Y = diabetes_dataset['Outcome']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.9, stratify=Y, random_state=9)
#training the model
classifier = svm.SVC(kernel='linear',max_iter=120)
#training the support vector machine classifier
classifier.fit(X_train, Y_train)
#joblib dumping for streamlit
joblib.dump(classifier,'diabetes_model.joblib')
#module evaluation
#Accuracy Score on the training data

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy Score of the training data:',training_data_accuracy)
#Accuracy Score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy Score of the test data:',test_data_accuracy)
#making a predictive data
input_data = (0,20,50,20,22,23,0.080,22)
#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the array data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standardize the input data
std_data = scalar.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
#print(prediction)
if(prediction == 0):
  print("Not Diabetic")
else:
  print("Diabetic")