# Purpose: Predicting breast cancer. This script focuses on PCA to reduce the dimensionality of the features in the Kaggle dataset.

import kaggle
import pandas as pd
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator

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

# Num PCA comps: Find elbow in plot along the y-axis (eigenvalues or "total variance explained")
# ANSWER: The "elbow" looks to be around the 10th principal component.

# ALTERNATIVE: Find how many components explain at least 95% of the variance
cumulative_variance = np.cumsum(explained_variance_ratio)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Number of components to retain ≥95% variance: {n_components_95}') # And this gives us 10

pca = PCA(n_components=10)


# Step 5: Transform the data into the new coordinate system
# - Project the standardized data onto the space defined by the selected principal components
# - This results in a dataset with reduced dimensions but preserved information

principal_components = pca.fit_transform(scaled_features)

# Generate a PCA plot

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])
pca_df['Diagnosis'] = labels

# Visualize PCA result
plt.figure(figsize=(8, 6))
for diagnosis in ['M', 'B']:
    subset = pca_df[pca_df['Diagnosis'] == diagnosis]
    plt.scatter(subset['PC1'], subset['PC2'], label=diagnosis, alpha=0.6)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Breast Cancer Dataset')
plt.legend()
plt.grid(True)
plt.show()

# Get loadings for PC1 and PC2
loadings = pd.DataFrame(pca.components_[:2].T,  # Transpose first two rows (PC1 and PC2)
                        columns=['PC1', 'PC2'],
                        index=features.columns)

# Print top contributing features
print("Top variables contributing to PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(5))

print("\nTop variables contributing to PC2:")
print(loadings['PC2'].abs().sort_values(ascending=False).head(5))