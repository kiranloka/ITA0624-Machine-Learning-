# prompt: Compare Linear and Polynomial Regression using Python take sample data and give output

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Evaluate models
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

print("Linear Regression:")
print(f"  Mean Squared Error: {mse_linear:.2f}")
print(f"  R-squared: {r2_linear:.2f}")
print("\nPolynomial Regression (Degree 2):")
print(f"  Mean Squared Error: {mse_poly:.2f}")
print(f"  R-squared: {r2_poly:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Sample Data')
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
# Sort X for plotting the polynomial curve smoothly
X_sorted, y_pred_poly_sorted = zip(*sorted(zip(X, y_pred_poly)))
plt.plot(X_sorted, y_pred_poly_sorted, color='green', label='Polynomial Regression (Degree 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear vs. Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
