# prompt: Write a program to implement Linear Regression (LR) algorithm in python take sample data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature
y = np.array([2, 4, 5, 4, 5])  # Target

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the sample data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot the results
plt.scatter(X, y, color='blue', label='Sample Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Linear Regression Example")
plt.legend()
plt.grid(True)
plt.show()