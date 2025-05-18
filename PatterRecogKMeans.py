#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:39:51 2025

@author: batdora
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mahotas
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.special import softmax
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors



train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')


""" Data and Feature Extraction """
# Normalize images to the [0,1] range
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Binarize the normalized images
train_images_binary = (train_images_normalized > 0.3).astype(np.int32)
test_images_binary = (test_images_normalized > 0.3).astype(np.int32)

# Flatten Images
train_data_flat_bin = train_images_binary.reshape(train_images.shape[0], -1)
test_data_flat_bin = test_images_binary.reshape(test_images.shape[0], -1)

# Combine data
data = np.vstack([train_data_flat_bin, test_data_flat_bin])
labels_true = np.hstack([train_labels,test_labels]) # hstack so its (n+m,) not (n+m,1) in the end

# Zernike Moments
def compute_zernike_features(images, max_moments=25, radius=14):
    """
    radius: radius for Zernike computation
    max_moments: how many zernike values to keep
    """
    feature_list = []
    for img in images:
        z = mahotas.features.zernike_moments(img, radius)
        feature_list.append(z[:max_moments])
    return np.array(feature_list)



# Metrics

def euclidean_distance (point,centroids):
    distances = [np.linalg.norm(point - c) for c in centroids]
    return np.argmin(distances)

def manhattan_distance(point, centroids):
    distances = [np.sum(np.abs(point - c)) for c in centroids]
    return np.argmin(distances)

def cosine_distance(point, centroids):
    distances = [1 - np.dot(point, c) / (np.linalg.norm(point) * np.linalg.norm(c)) for c in centroids]
    return np.argmin(distances)

def compute_new_centroids(data, labels, k=5):
    centroids = []
    for i in range(k):
        points_in_cluster = [x for index, x in enumerate(data) if labels[index] == i]
        centroid = np.mean(points_in_cluster, axis = 0)
        centroids.append(centroid)
    return centroids

def k_means(data, k=5, metric="euclidean", max_iter=100, epsilon = 1e-4, seed = 42):
    # Pick initial centroids from datapoints in data in random
    np.random.seed(seed)
    centroids = data[np.random.choice(len(data), size=k, replace=False)]

    # Map string to actual function
    distance_func_map = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_distance
    }
    
    distance_func = distance_func_map[metric]
   
    for iteration in range(max_iter):
        labels = []
        for point in data:
            label = distance_func(point, centroids)
            labels.append(label)
        
        new_centroids = compute_new_centroids(data, labels)

        diff = np.sum([np.linalg.norm(n - o) for n, o in zip(new_centroids, centroids)])
        if diff < epsilon:
            break  # converged

        centroids = new_centroids

    return centroids, labels

def compute_SSE(data, labels, centroids):
    sse = 0
    for i in range(len(data)):
        cluster_idx = labels[i]
        centroid = centroids[cluster_idx]
        distance_squared = np.linalg.norm(data[i] - centroid) ** 2
        sse += distance_squared
    return sse

n_components = [5]
log = []
zernike_amount = [1]

for j in zernike_amount:
    for i in n_components:    
        
        # Zernike
        zernike = compute_zernike_features(train_images_binary,j)
        zernike_test = compute_zernike_features(test_images_binary,j)
    
        zernike_total = np.vstack([zernike,zernike_test])
        
        # PCA
        pca = PCA(n_components= i)
        data_pca = pca.fit_transform(data)
        
        # Append zernike to PCA
        data_pca_zernike = np.hstack([data_pca, zernike_total])
        
        dims = data_pca_zernike.shape[1]
        
        # Pick which data to use
        data_used= data_pca_zernike
        
        # K-Means
        centroids, labels_pred = k_means(data_used, metric= "euclidean")
        
        # Internal metrics
        sse = compute_SSE(data_used, labels_pred, centroids)
        sil_score = silhouette_score(data_used, labels_pred)
        ch_score = calinski_harabasz_score(data_used, labels_pred)
        
        # External metrics (label alignment)
        cm = confusion_matrix(labels_true, labels_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)  # -c maximize total accuracy
            # Allign indices of clusters with true labels
        mapping = {col: row for row, col in zip(row_ind, col_ind)}
        new_predicted_labels = [mapping[cluster] for cluster in labels_pred]
        accuracy = accuracy_score(labels_true, new_predicted_labels)
        nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
        purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
    
        # Log all metrics
        log.append([i, j, sse, sil_score, ch_score, accuracy, nmi_score, purity])
        print(f"Completed PCA with {i} components")


columns = ['PCA Components', 'Zernike Moments', 'SSE', 'Silhouette', 'Calinski-Harabasz', 'Accuracy', 'NMI', 'Purity']
results = pd.DataFrame(log, columns=columns)

### Composite Score Calculation ###

# Compute number of dimensions for normalization
results['Dimensions'] = results['PCA Components'] + results['Zernike Moments']
results['SSE_norm'] = results['SSE'] / results['Dimensions']

# Define which metrics increase with performance
metric_signs = {
    'SSE_norm': -1,
    'Silhouette': 1,
    'Calinski-Harabasz': 1,
    'Accuracy': 1,
    'NMI': 1,
    'Purity': 1
}

# Apply SoftMax to each scaled metric and sum
composite_parts = []
for metric, sign in metric_signs.items():
    values = results[metric].values * sign
    scores = softmax(values)
    composite_parts.append(scores)

# Final composite score
results['Composite Score'] = np.sum(composite_parts, axis=0)
   
### tSNE ###

# Class label mapping
class_names = ['rabbit', 'yoga', 'snowman', 'hand', 'motorbike']

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_tsne = tsne.fit_transform(data_used)

# Set up colormap
cmap_5 = plt.cm.get_cmap('tab10', 5)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 5.5, 1), ncolors=5)

# Plot t-SNE with true labels
plt.figure(figsize=(8, 6))
scatter1 = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_true, cmap=cmap_5, norm=norm, s=5)
cbar1 = plt.colorbar(scatter1, ticks=np.arange(5))
cbar1.ax.set_yticklabels(class_names)
cbar1.set_label("True Class")
plt.title("t-SNE Visualization Colored by True Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()

# Plot t-SNE with predicted cluster labels
plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_pred, cmap=cmap_5, norm=norm, s=5)
cbar2 = plt.colorbar(scatter2, ticks=np.arange(5))
cbar2.set_label("K-Means Cluster")
plt.title("t-SNE Visualization Colored by K-Means Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()



    
    
    