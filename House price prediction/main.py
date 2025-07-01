import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Year': [2014, 2013, 2018, 2017, 2015, 2012, 2016, 2011],
    'Present_Price': [9.85, 5.60, 14.20, 13.50, 9.50, 4.50, 10.50, 3.00],
    'Kms_Driven': [6900, 5200, 24000, 19000, 43000, 91000, 36000, 82000],
    'Fuel_Type': ['Petrol', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Diesel', 'Diesel', 'Petrol'],
    'Seller_Type': ['Dealer', 'Individual', 'Dealer', 'Dealer', 'Dealer', 'Dealer', 'Dealer', 'Individual'],
    'Transmission': ['Manual', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Manual', 'Manual'],
    'Owner': [0, 0, 0, 0, 0, 0, 0, 0],
    'Selling_Price': [7.25, 2.85, 9.50, 11.00, 4.75, 2.25, 7.75, 2.00]
}

df = pd.DataFrame(data)

# Preprocessing
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Split into features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n=== Model Evaluation ===")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Coefficients
print("\n=== Feature Coefficients ===")
coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients)

# Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
