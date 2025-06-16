# prompt: Write a program for Implementation of K-Nearest Neighbours (K-NN) in Python
# give sample example to get output

# K-Nearest Neighbors (K-NN) Implementation

# Install scikit-learn if you haven't already
try:
    import sklearn
except ImportError:
    !pip install scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# --- Sample Example ---

# Sample dataset (replace with your actual data)
# X: Features
# y: Labels
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y = np.array([0, 0, 1, 1, 0, 1]) # 0 and 1 represent different classes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-NN classifier object
# n_neighbors is the value of K
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with K={k}: {accuracy}")

# You can now use the trained model to predict new data points
new_data_point = np.array([[6, 7]])
prediction = knn.predict(new_data_point)
print(f"Prediction for {new_data_point}: {prediction}")

# --- General K-NN Implementation Structure ---

# 1. Import necessary libraries: sklearn, numpy, etc.
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# 2. Load or create your dataset:
# X = features
# y = labels

# 3. Split your data into training and testing sets:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=some_number)

# 4. Choose the value of K (number of neighbors):
# k = some_integer

# 5. Create a K-NN classifier instance:
# knn = KNeighborsClassifier(n_neighbors=k)

# 6. Train the model:
# knn.fit(X_train, y_train)

# 7. Make predictions on the test set:
# y_pred = knn.predict(X_test)

# 8. Evaluate the model's performance:
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# 9. Use the model to predict new data points:
# new_data_point = np.array([[feature1, feature2, ...]])
# prediction = knn.predict(new_data_point)
# print("Prediction for new data:", prediction)
