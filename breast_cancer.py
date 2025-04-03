import kaggle
import pandas as pd
import os

# C:\Users\carri\.kaggle\kaggle.json  
# Authenticate using kaggle.json (make sure the path is set correctly beforehand)
kaggle.api.authenticate()

# List the files in the dataset
print(kaggle.api.dataset_list_files('uciml/breast-cancer-wisconsin-data').files)

# Download and unzip the dataset to current directory
kaggle.api.dataset_download_files('uciml/breast-cancer-wisconsin-data', path='.', unzip=True)

# Load and Prepare the Dataset
training_df = pd.read_csv("StudentsPerformance.csv")  

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
print(training_df.head(200))
print(training_df.describe(include='all'))

# What is the maximum reading score?
max_reading_score = training_df['reading score'].max()
print("What is the maximum math score? \t\t\t\tAnswer: {score:.2f}%".format(score = max_reading_score))

# What is the mean reading score?
mean_reading_score = training_df['reading score'].mean()
print("What is the mean math score? \t\tAnswer: {mean:.4f}%".format(mean = mean_reading_score))

# How many ethnicities are in the dataset?
num_race =  training_df['race/ethnicity'].nunique()
print("How many ethnicities are in the dataset? \t\tAnswer: {number}".format(number = num_race))

# What is the most frequent parental level of education?
most_freq_parental_education = training_df['parental level of education'].value_counts().idxmax()
print("What is the most frequent parental level of education? \t\tAnswer: {type}".format(type = most_freq_parental_education))

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? \t\t\t\tAnswer:", "No" if missing_values == 0 else "Yes")

# correlation matrix
print(training_df.corr(numeric_only = True))

#Dealing w/ categorical data
#Three options: 
# 1. one-hot encoding using its sparse representation to save memeory, 
# 2. embedding (if there lots of categories), 
# 3. hashing
# Choose: embedding (only for practice). In reality, one-hot encoding may actually be more appropriate since we do not have many dimensions.




