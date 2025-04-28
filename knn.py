#-------------------------------------------------------------------------
# AUTHOR: Fidelis Prasetyo
# FILENAME: knn.py
# SPECIFICATION: compare accuracy of KNN with different parameters
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1 day
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

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

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
df = pd.read_csv("weather_training.csv", sep=',', header=0)
X_train = np.array(df.values)[:,1:6].astype('f')
Y_train = np.array(df.values)[:,-1].astype('f')
Y_train_discretized = np.array([discretize_class(temp) for temp in Y_train])
#reading the test data
df = pd.read_csv("weather_test.csv", sep=',', header=0)
X_test = np.array(df.values)[:,1:6].astype('f')
Y_test = np.array(df.values)[:,-1].astype('f')
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

best_accuracy = 0
#loop over the hyperparameter values (k, p, and w) ok KNN
for k in k_values:
    for p in p_values:
        for w in w_values:
            correct_count = 0
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_train, Y_train_discretized)

            test = zip(X_test, Y_test)
            for (x_testSample, y_testSample) in test:
                predicted_value = clf.predict([x_testSample])[0]
                if(is_close(y_testSample, predicted_value)):
                    correct_count = correct_count + 1

            accuracy = correct_count/ len(X_test)
            if(accuracy > best_accuracy):
                best_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {accuracy:.2f}, Parameters: k={k}, p={p}, w={w}")
            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here





