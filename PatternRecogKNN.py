#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:15:47 2025

@author: batdora
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from collections import Counter
from skimage.feature import hog
from skimage import data, exposure


train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')


# Normalize images to the [0,1] range
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Binarize the normalized images
train_images_binary = (train_images_normalized > 0.3).astype(np.int32)
test_images_binary = (test_images_normalized > 0.3).astype(np.int32)


# Flatten Images
X_train_flat_bin = train_images_binary.reshape(train_images.shape[0], -1)
X_test_flat_bin = test_images_binary.reshape(test_images.shape[0], -1)

# Flatten Images
X_train_flat = train_images.reshape(train_images.shape[0], -1)
X_test_flat = test_images.reshape(test_images.shape[0], -1)


"""
# PLOTTING


# Generate a consistent color mapping for class labels
def get_label_color_mapping(labels, palette="tab10"):
    
    #Creates a fixed color mapping for class labels using Seaborn's color palettes.
    
    unique_labels = np.unique(labels)
    color_palette = sns.color_palette(palette, len(unique_labels))  # Get unique colors
    label_to_color = {label: color_palette[i] for i, label in enumerate(unique_labels)}
    return label_to_color

# Apply the color mapping for train labels
label_color_mapping = get_label_color_mapping(train_labels)

# Function to convert labels into colors
def get_colors_from_labels(labels, label_color_mapping):
    return np.array([label_color_mapping[label] for label in labels])


# Scatter plot of first two PCA components (Train)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], 
                hue=train_labels, palette=label_color_mapping, alpha=0.5)
plt.title("PCA Projection (First 2 Principal Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class Labels")
plt.show()

# Scatter plot of first two PCA components (Test)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], 
                hue=test_labels, palette=label_color_mapping, alpha=0.5)
plt.title("PCA Projection for Test (First 2 Principal Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class Labels")
plt.show()

def plot_pca_scatter_grid(X_pca, labels, num_components=20, save_path=None):
    colors = get_colors_from_labels(labels, label_color_mapping)  # Ensure consistent colors

    fig, axes = plt.subplots(num_components, num_components, figsize=(20, 20))

    for i in range(num_components):
        for j in range(num_components):
            ax = axes[i, j]
            
            if i == j:
                ax.axis("off")  # Hide diagonal
            else:
                ax.scatter(X_pca[:, j], X_pca[:, i], color=colors, alpha=0.5, s=1)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_pca_scatter_grid_per_class(X_pca, labels, num_components=20, save_dir="pca_faucet_per_class"):
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    unique_labels = np.unique(labels)

    for class_label in unique_labels:
        # Filter PCA data for this class
        X_class = X_pca[labels == class_label]

        # Create figure
        fig, axes = plt.subplots(num_components, num_components, figsize=(20, 20))
        
        for i in range(num_components):
            for j in range(num_components):
                ax = axes[i, j]
                
                if i == j:
                    ax.axis("off")  # Hide diagonal
                else:
                    ax.scatter(X_class[:, j], X_class[:, i], alpha=0.5, s=1, color=label_color_mapping[class_label])
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Save per-class Faucet Graph
        save_path = os.path.join(save_dir, f"pca_faucet_class_{class_label}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close plot to free memory

        print(f"Saved Faucet Graph for Class {class_label} -> {save_path}")


# Run Faucet Graph Plotting
#plot_pca_scatter_grid(X_train_pca, train_labels, num_components=20, save_path="pca_faucet_graph_fixed.png")

# Run Faucet Graph Generation for Each Class
#plot_pca_scatter_grid_per_class(X_train_pca, train_labels, num_components=20)

"""

def fisher_score(X, y):
    classes = np.unique(y)
    overall_mean = np.mean(X, axis=0)
    
    S_W = np.zeros((X.shape[1], X.shape[1]))
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for cls in classes:
        class_samples = X[y == cls]
        class_mean = np.mean(class_samples, axis=0)
        n_cls = class_samples.shape[0]
        
        # Within-class scatter
        for sample in class_samples:
            diff = (sample - class_mean).reshape(-1, 1)
            S_W += diff @ diff.T
        
        # Between-class scatter
        mean_diff = (class_mean - overall_mean).reshape(-1, 1)
        S_B += n_cls * (mean_diff @ mean_diff.T)
    
    # Avoid division by zero
    if np.trace(S_W) == 0:
        return np.inf
    
    return np.trace(S_B) / np.trace(S_W)

## KNN CLASSIFIER WITH DIFFERENT DISTANCE METRICS

class KNNClassifier:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def distance(self, x1, x2):
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == "cosine":
            return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        else:
            raise ValueError("Unknown distance metric")

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [(self.distance(x, x_train), label) for x_train, label in zip(self.X_train, self.y_train)]
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:self.k]]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = np.sum(y_pred == y_test)
        return correct / len(y_test)
    
    def get_neighbors(self, x):
        distances = [(self.distance(x, x_train), i, label) 
                     for i, (x_train, label) in enumerate(zip(self.X_train, self.y_train))]
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

metrics_list = ["euclidean"]
n_components = [2]
k_list=[3]

for i in range(len(metrics_list)): 
    
    metric = metrics_list[i]
    for l in range(len(n_components)):    
    
        # Apply PCA
        pca = PCA(n_components=n_components[l])
        X_train_pca = pca.fit_transform(X_train_flat_bin)
        X_test_pca = pca.transform(X_test_flat_bin)
        
        # Explained variance ratio for each component
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Calcualte Fisher Score
        
        score = fisher_score(X_train_pca, train_labels)
        print("Fisher Discriminant Score:", score)

        
        #print("Explained Variance Ratio per Component:")
        #print(explained_variance)
 
        #For testing with different k values
        for j in range(len(k_list)):
            
            k = k_list[j]
            
            #For testing without dimension reduciton
            if len(n_components) == 0:
                print(f"\nTesting with flat images directly, {metric} distance and k = {k}:")
                
                # Create and train the KNN classifier
                knn = KNNClassifier(k, distance_metric=metric)
                knn.fit(X_train_flat, train_labels)
                
                # Predict and calculate accuracy
                accuracy = knn.score(X_test_flat, test_labels)
                print(f"Accuracy with {metric} distance: {accuracy:.4f}")
                
                
            #For testing with various dimension reduction parameters
            else:
                print(f"\nTesting with PCA = {n_components[l]}, {metric} distance and k = {k}:")
                
                print("Cumulative Explained Variance:")
                print(cumulative_variance[-1])
                
                # Create and train the KNN classifier
                knn = KNNClassifier(k, distance_metric=metric)
                knn.fit(X_train_pca, train_labels)
        
                
                # Predict and calculate accuracy
                accuracy = knn.score(X_test_pca, test_labels)
                print(f"Accuracy with {metric} distance: {accuracy:.4f}")




