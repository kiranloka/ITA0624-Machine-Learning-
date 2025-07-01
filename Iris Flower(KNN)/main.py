from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Output results
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("=== Accuracy Score ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Optional: visualize the first two features
plt.figure(figsize=(8, 6))
for i, label in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=label)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset - Feature Visualization")
plt.legend()
plt.grid(True)
plt.show()
