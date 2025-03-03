# -------------------------------------------------------------------------
# AUTHOR: Fidelis Prasetyo
# FILENAME: pca.py
# SPECIFICATION: Identifying which feature to remove
# FOR: CS 5990 Assignment #2
# TIME SPENT: 1 day
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv('heart_disease_dataset.csv')

# #Create a training matrix without the target variable (Heart Diseas)
# #--> add your Python code here
df_features = df

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here
num_features = df_features.shape[1]

pc1_variance = {}

# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    # Store PC1 variance and the feature removed
    # Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_variance[df_features.columns[i]] = pca.explained_variance_ratio_[0]

# # Find the maximum PC1 variance
# # --> add your Python code here
max_pc1_variance = 0
feature_removed = ""
for key, value in pc1_variance.items():
    if(value > max_pc1_variance):
        max_pc1_variance = value
        feature_removed = key

# # Print results
# # Use the format: Highest PC1 variance found: ? when removing ?

print("Highest PC1 variance found: " + str(max_pc1_variance) + " when removing " + feature_removed)




