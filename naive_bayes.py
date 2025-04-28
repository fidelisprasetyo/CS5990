#-------------------------------------------------------------------------
# AUTHOR: Fidelis Prasetyo
# FILENAME: naive_bayes.py
# SPECIFICATION: compare accuracy of naive bayes with different parameters
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1 day
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

def discretize_class(value):
    closest_class = None
    min_diff = float('inf')
    for class_ in classes:
        diff = abs(value - class_)
        if diff < min_diff:
            min_diff = diff
            closest_class = class_
    return closest_class

def is_close(real_value, predicted_value):
    try:
        diff = 100*(abs(predicted_value - real_value)/real_value)
        diff = abs(diff)
        return (diff < 15)
    except:
        return abs(predicted_value-real_value) < 0.00001

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
df = pd.read_csv("weather_training.csv", sep=',', header=0)
X_train = np.array(df.values)[:,1:6].astype('f')
Y_train = np.array(df.values)[:,-1].astype('f')

#update the training class values according to the discretization (11 values only)
Y_train_discretized = np.array([discretize_class(temp) for temp in Y_train])

#reading the test data
df = pd.read_csv("weather_test.csv", sep=',', header=0)
X_test = np.array(df.values)[:,1:6].astype('f')
Y_test = np.array(df.values)[:,-1].astype('f')

#update the test class values according to the discretization (11 values only)
Y_test_discretized = np.array([discretize_class(temp) for temp in Y_test])

#loop over the hyperparameter value (s)
best_accuracy = 0
best_s = None
for s in s_values:
    correct_count = 0

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_train, Y_train_discretized)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    test = zip(X_test, Y_test_discretized)
    for(x_testSample, y_testSample) in test:
        predicted_value = clf.predict([x_testSample])[0]
        if(y_testSample == predicted_value):
            correct_count = correct_count + 1
    
    accuracy = correct_count/ len(X_test)
    if(accuracy > best_accuracy):
        best_s = s
        print(f"Highest Naive Bayes accuracy so far: {accuracy:.2f}, Parameters: s={s}")

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here



