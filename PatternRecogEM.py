#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:24:18 2025

@author: batdora
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import matplotlib.patches as patches
from scipy.stats import multivariate_normal


data = np.load('dataset.npy')

plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)

plt.show()

means = [[0,0],[4,4],[11,11]]

covariances = [np.eye(2) for i in range(0, 3)]

priors = np.ones(3)/3

def expectation_step (data, means, covariances, priors):
    
    points = data.shape[0]
    distributions = len(priors)
    
    
    probabilites = np.zeros([points,distributions])
    
    
    for i in range(0,distributions):
        probabilites[:,i] = priors[i] * mvn.pdf(data, means[i], covariances[i])
        

    probabilites /= probabilites.sum(axis = 1, keepdims = True)

    return probabilites

def maximization_step (data, probabilites):
     points, dims = data.shape
     distributions = probabilites.shape[1]
     
     priors = probabilites.sum(axis=0)/points
     means = np.dot(probabilites.T, data) / probabilites.sum(axis=0)[:, None]
     
     covariances = []
     
     for i in range(distributions):
       diff = data - means[i]
       cov_i = (probabilites[:, i, None] * diff).T @ diff / probabilites[:, i].sum()
       covariances.append(cov_i)
   
     return means, covariances, priors
 
    


#E to M step iterations
    
max_iters = 100
tolerance = 1e-4  # Stop when log-likelihood change is small

for i in range(max_iters):
    old_means = means.copy()

    probabilites = expectation_step(data, means, covariances, priors)
    means, covariances, priors = maximization_step(data, probabilites)

    if np.linalg.norm(means - old_means) < tolerance:
        break  # Stop if means don’t change much


#PLOTTING

# Assign each data point to the Gaussian with the highest responsibility
cluster_assignments = np.argmax(probabilites, axis=1)

# Scatter plot with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='Accent', s=10, alpha=0.7)
plt.title("Cluster Assignments after EM Convergence")
plt.show()


print("Estimated Means:")
print(means)

print("\nEstimated Covariance Matrices:")
for i, cov in enumerate(covariances):
    print(f"Covariance Matrix {i+1}:\n{cov}\n")


# Define the grid for contour plotting
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
pos = np.dstack((x, y))

# Scatter plot of data with cluster assignments
plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='Accent', s=10, alpha=0.7)

# Plot the Gaussian distributions using contour lines
for k in range(len(means)):
    gaussian = multivariate_normal(mean=means[k], cov=covariances[k])
    plt.contour(x, y, gaussian.pdf(pos), levels=10, cmap='Reds', alpha=0.8)

# Plot settings
plt.title("Gaussian Contour Plot")
plt.show()

    