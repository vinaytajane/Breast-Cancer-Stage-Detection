#import libraries 
import numpy as np
import sklearn.datasets

#getting the datasets 
breast_cancer = sklearn.datasets.load_breast_cancer()

print(breast_cancer)

X = breast_cancer.data
Y = breast_cancer.target

print(X)
print(Y)

print(X.shape,Y.shape)

# Import data to the pandas data frames
import pandas as pd

data = pd.DataFrame(breast_cancer.data, columns= breast_cancer.feature_names)

data['class'] = breast_cancer.target

data.head()

data.describe()

print(data['class'].value_counts())

print(breast_cancer.target_names)

data.groupby('class').mean()

# 0 - Malignant
# 1 - Benign

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
print(Y.shape, Y_train.shape, Y_test.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)
#test_size --> to specify the percentage of test data needed

print(Y.shape, Y_train.shape, Y_test.shape)
print(Y.mean(), Y_train.mean(), Y_test.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y)
#stratify --> for correct distribution of data as of the original data

print(Y.mean(), Y_train.mean(), Y_test.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)
#random_state --> specific split of data. each value of random_state split the data differently 

print(X_train.mean(), X_test.mean(), X.mean())

print(X_train)

# Logistic Regression
#import Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression() #loading the Logistic Regression model to the variable "Classifer"

#training the model on the training data
classifier.fit(X_train, Y_train)

# Evaluation of the Model
#import accuracy_score
from sklearn.metrics import accuracy_score 

prediction_on_training_data =classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data :',accuracy_on_training_data)

#predicion on test data
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on Test data :',accuracy_on_test_data)

# Detecting whether the Patient has Breast Cancer in benign or Manlignant State
input_data = (0.45,15.7,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613,19.3345,0.8902,2.217,27.19,0.00751,0.03345,0.03672,0.01137,0.02165,0.005082,15.47,23.75,103.4,741.6,0.1791,0.5249,0.5355,0.1741,0.3985,0.1244)
#change input_data to numpy_array to make predictions
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)

#reshape the array as we are predicting the output for one instance 

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction
prediction = classifier.predict(input_data_reshaped)
print("\n===================================")
print(prediction) #return a list with element [0] if Malignant; returns a list with element [1], if Benign.

if (prediction[0]==0):
  print("The Breast Cancer is Malignant")
else:
  print("The Breast Cancer is Benign")

print("===================================")