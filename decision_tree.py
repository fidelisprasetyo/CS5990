# -------------------------------------------------------------------------
# AUTHOR: Fidelis Prasetyo
# FILENAME: decision_tree.py
# SPECIFICATION: Train and test decision tree
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 1 day
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

def transform_attribute(data):
    attr = []
    if data[0] == 'Yes':
        attr.append(1)
    else:
        attr.append(0)
    if data[1] == 'Single':
        attr.append(1)
        attr.append(0)
        attr.append(0)
    elif data[1] == 'Divorced':
        attr.append(0)
        attr.append(1)
        attr.append(0)
    else:
        attr.append(0)
        attr.append(0)
        attr.append(1)
    income_str = data[2].lower().strip()
    income_float = float(income_str[:-1])
    attr.append(income_float)
    return attr    

def transform_class(data):
    if data[3] == 'Yes':
        return 1
    else:
        return 2

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.

    for data in data_training:
        X.append(transform_attribute(data))

        #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        Y.append(transform_class(data))

    #loop your training and test tasks 10 times here
    accuracy = []

    for i in range (10):

        print("processing " + ds + " loop " + str(i), end="\r")

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        #read the test data and add this data to data_test NumPy
        df = pd.read_csv('cheat_data.csv', sep=',', header=0)
        data_test = np.array(df.values)

        total_count = 0
        true_count = 0

        for data in data_test:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           
            class_predicted = clf.predict([transform_attribute(data)])[0]

            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.

            total_count = total_count+1

            if class_predicted == transform_class(data):
                true_count = true_count+1
        
        accuracy.append(true_count/total_count)

    #find the average accuracy of this model during the 10 runs (training and test set)
    accuracy_avg = sum(accuracy)/len(accuracy)


    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    
    print("final accuracy when training on " + ds + ": " + str(accuracy_avg))



