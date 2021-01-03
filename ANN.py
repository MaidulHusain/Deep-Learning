# Importing Libries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Data Set
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13].values
y = pd.DataFrame(data.iloc[:, 13].values)
# Encoding dependent variable

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [1, 2]),
    remainder="passthrough")

X=A.fit_transform(X)

# Spliting Data Set into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# Second Part let's start ANN
# Importing package and libiries
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initialising ANN
classifier = Sequential()
#Adding the Input Layer and the first hidden layer

classifier.add(Dense(units=6, activation='relu', input_dim=13))
# Adding the second hidden layer
classifier.add(Dense(units=6, activation='relu'))
#Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid'))
#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the ANN to training data
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
#Predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
#Import Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)








