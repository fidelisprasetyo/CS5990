# -------------------------------------------------------------------------
# AUTHOR: Fidelis Prasetyo
# FILENAME: roc_curve.py
# SPECIFICATION: compute ROC curve
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 1 day
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array

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
        return 0

X = []
y = []

df = pd.read_csv('cheat_data.csv', sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
data_training = np.array(df.values) #creating a training matrix without the id (NumPy library)

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.

for data in data_training:
    X.append(transform_attribute(data))

    # transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
    y.append(transform_class(data))

# split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)

# generate random thresholds for a no-skill prediction (random classifier)
ns_probs = [0] * len(testy)

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()