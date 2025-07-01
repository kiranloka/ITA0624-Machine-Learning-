# prompt: Write a Python Program to Implement Expectation &amp; Maximization Algorithm use sample data and give output

import numpy as np
from sklearn.mixture import GaussianMixture

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
X[:50] += 20  # Add a second cluster

# Initialize and fit the Gaussian Mixture Model
# n_components is the number of clusters (components)
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X)

# Print the results
print("Converged:", gmm.converged_)
print("Number of iterations:", gmm.n_iter_)
print("Log-likelihood lower bound:", gmm.lower_bound_)
print("\nWeights for each component:")
print(gmm.weights_)
print("\nMeans for each component:")
print(gmm.means_)
print("\nCovariances for each component:")
print(gmm.covariances_)

# Predict the cluster assignments for the data
labels = gmm.predict(X)
print("\nSample cluster assignments (first 10 data points):")
print(labels[:10])

# Predict the probability of each data point belonging to each cluster
probabilities = gmm.predict_proba(X)
print("\nSample probabilities for each component (first 10 data points):")
print(probabilities[:10])
