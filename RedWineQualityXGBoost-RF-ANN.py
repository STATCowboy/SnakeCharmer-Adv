#
# Author: Jamey Johnston
# Title: Code Like a Snake Charmer: Advanced Data Modeling in Python!
# Date: 2019/11/08
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/SnakeCharmer-Adv
#

# Train models for Detecting Wine Quality
# Save model to file using pickle
# Load model and make predictions
#

# Import OS and set CWD
import os
from settings import APP_ROOT


import numpy as np
from numpy import loadtxt, vstack, column_stack
import xgboost
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Load the Wine Data
dataset = np.loadtxt(os.path.join(APP_ROOT, "winequality-red.csv"), delimiter=';', skiprows=1)

# Headers of Data
# "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"

# Split the wine data into X (independent variable) and y (dependent variable)
X = dataset[:,0:11].astype(float)
Y = dataset[:,11].astype(int)

# Split wine data into train and validation sets
seed = 7
test_size = 0.3
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit model on Wine Training Data using eXtendeded Gradient Boosting
modelXGB = xgboost.XGBClassifier()
modelXGB.fit(X_train, y_train)

# Make predictions for Validation data
y_predXGB = modelXGB.predict(X_valid)
predictionsXGB = [round(value) for value in y_predXGB]

# Evaluate predictions
accuracyXGB = accuracy_score(y_valid, predictionsXGB)
print("Accuracy of eXtended Gradient Boosting: %.2f%%" % (accuracyXGB * 100.0))

# Create Dataset with Prediction and Inputs
predictionResultXGB = column_stack(([X_valid, vstack(y_valid), vstack(y_predXGB)]))


# Fit model on Wine Training Data using Random Forest save model to Pickle file
modelRF = RandomForestRegressor()
modelRF.fit(X_train, y_train)

# Make predictions for Validation data
y_predRF = modelRF.predict(X_valid)
predictionsRF = [round(value) for value in y_predRF]

# Evaluate predictions
accuracyRF = accuracy_score(y_valid, predictionsRF)
print("Accuracy of Random Forest: %.2f%%" % (accuracyRF * 100.0))

# Create Dataset with Prediction and Inputs
predictionResultRF = column_stack(([X_valid, vstack(y_valid), vstack(y_predRF)]))

# save model to file
pickle.dump(modelRF, open("winequality-red.pickleRF.dat", "wb"))

# Load model from Pickle file
loaded_modelRF = pickle.load(open("winequality-red.pickleRF.dat", "rb"))

# Predict a Wine Quality (Class) from inputs
loaded_modelRF.predict([[6.8, .47, .08, 2.2, .0064, 18.0, 38.0, .999933, 3.2, .64, 9.8, ]])



#
# Try a Simple Decision Tree
# 
from sklearn import tree

# Train Model
wineTree = tree.DecisionTreeClassifier()
wineTree = wineTree.fit(X_train, y_train)

# Make predictions for Validation data
y_predDT = wineTree.predict(X_valid)
predictionsDT = [round(value) for value in y_predDT]

# Evaluate predictions
accuracyDT = accuracy_score(y_valid, predictionsDT)
print("Accuracy of Decision Tree: %.2f%%" % (accuracyDT * 100.0))

# Predict a Wine Quality (Class) from inputs
wineTree.predict([[6.8, .47, .08, 2.2, .0064, 18.0, 38.0, .999933, 3.2, .64, 9.8, ]])


# Display Tree Visually
import graphviz

# Install steps for it to work (on Windows)
# 1. Install windows package from: http://www.graphviz.org/Download_windows.php
# 2. Install python graphviz package (use pip)
# 3. Add C:\Program Files (x86)\Graphviz2.38\bin to User path

dot_data = tree.export_graphviz(wineTree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(os.path.join(APP_ROOT, 'wineTree.gv'), view=False)



# 
# Predict Wine Quality with ANN in Tensorflow/Keras
#
# http://cs231n.github.io/neural-networks-1/
# https://www.tensorflow.org/tutorials/keras/regression


# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Scale the data with `StandardScaler`
X = StandardScaler().fit_transform(X)


# Import `Sequential` from `keras.models`
from keras.models import Sequential
from keras.utils import np_utils

# Import `Dense` from `keras.layers`
from keras.layers import Dense, Conv1D, MaxPooling1D, SimpleRNN, Dropout

import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for train, test in kfold.split(X, Y):
    dummy_y_train = np_utils.to_categorical(Y[train])
    dummy_y_test = np_utils.to_categorical(Y[test])
    model = Sequential()
    model.add(Dense(32, input_dim=11, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(Y.max() + 1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train], dummy_y_train, epochs=150, batch_size=10, verbose=1)


print("input", model.input.name)
print("output", model.output.name)

print("Model's parameters")
print(model.get_weights())

mse_value, mae_value = model.evaluate(X[test], dummy_y_test, verbose=0)

print(mse_value)
print(mae_value)

# evaluate the model
scores = model.evaluate(X[train], dummy_y_train)
print("Training \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred = model.predict(X[test])

# Evaluate predictions
accuracyANN = model.evaluate(X[test], dummy_y_test)
print("Test \n%s: %.2f%%" % (model.metrics_names[1], accuracyANN[1]*100))


from sklearn.metrics import r2_score
r2_score(dummy_y_test, y_pred)
score = model.evaluate(X[test], dummy_y_test, verbose=1)
print(score)
