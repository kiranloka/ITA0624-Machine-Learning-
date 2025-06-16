# prompt: Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis
# based on a given set of training data samples.

import numpy as np
import pandas as pd

# Define the training data
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

# Initialize the most specific hypothesis
# The number of attributes is the number of columns minus the target column
n_attributes = len(df.columns) - 1
hypothesis = ['0'] * n_attributes

# Implement the FIND-S algorithm
for i in range(len(df)):
    # Consider only positive examples
    if df.iloc[i, -1] == 'Yes':
        for j in range(n_attributes):
            if hypothesis[j] == '0':
                # Initialize with the first positive example
                hypothesis[j] = df.iloc[i, j]
            elif hypothesis[j] != df.iloc[i, j]:
                # Generalize the hypothesis if attributes don't match
                hypothesis[j] = '?'

# Print the most specific hypothesis
print("The most specific hypothesis is:", hypothesis)
