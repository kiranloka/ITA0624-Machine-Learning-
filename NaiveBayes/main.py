# prompt: Write a program to implement Naïve Bayes algorithm in python and to display the results using
# confusion matrix and accuracy. give sample example and output

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Create a sample dataset
data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'target': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['feature1', 'feature2']]
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naïve Bayes classifier (Gaussian Naive Bayes is suitable for continuous features)
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Display the results
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
