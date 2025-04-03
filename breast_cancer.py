import kaggle
import pandas as pd
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Practice PCA principles as outlined in IBM's https://www.ibm.com/think/topics/principal-component-analysis
# Dataset:  https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data 

# C:\Users\carri\.kaggle\kaggle.json  
# Authenticate using kaggle.json (make sure the path is set correctly beforehand)
kaggle.api.authenticate()

# List the files in the dataset
print(kaggle.api.dataset_list_files('uciml/breast-cancer-wisconsin-data').files)

# Download and unzip the dataset to current directory
kaggle.api.dataset_download_files('uciml/breast-cancer-wisconsin-data', path='.', unzip=True)

df = pd.read_csv('data.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1)
features = df.drop('diagnosis', axis=1) #? axis = 1???
labels = df['diagnosis']

# Step 1: Standardize the range of continuous initial variables
scaler = StandardScaler()
# - Subtract the mean and divide by the standard deviation for each variable
scaled_features = scaler.fit_transform(features)

# Step 2: Compute the covariance matrix to identify correlations
# - Covariance matrix captures how variables vary together
# - It is a d x d symmetric matrix, where d is the number of dimensions
cov_matrix = np.cov(scaled_features.T)

# Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix
# - Eigenvectors = principal components (directions of maximum variance)
# - Eigenvalues = amount of variance in each component
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Select the principal components
# - Rank eigenvectors by eigenvalue
explained_variance_ratio = eigenvalues / eigenvalues.sum()

# - Use a scree plot to choose the number of components to retain
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance_ratio), marker='o')
plt.title('Scree Plot: Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Step 5: Transform the data into the new coordinate system
# - Project the standardized data onto the space defined by the selected principal components
# - This results in a dataset with reduced dimensions but preserved information

