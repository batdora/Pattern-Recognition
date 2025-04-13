#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:40:31 2025

@author: batdora
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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


class BernoulliNBClassifier:
    def __init__(self):
        self.alpha = 1e-10
        self.class_prior_ = {}
        self.feature_prob_ = {}
        self.classes_ = None

    def fit(self, data, labels):

        self.classes_ = np.unique(labels)
        n_samples, n_features = data.shape
        
        # Calculate prior probabilities and feature likelihoods for each class.
        for c in self.classes_:
            X_c = data[labels == c]
            n_c = X_c.shape[0]
            # Prior probability for class c.
            self.class_prior_[c] = n_c / n_samples
            # For each feature, compute probability that feature equals 1 given class c,
            # with smoothing: (count_1 + alpha) / (n_c + 2*alpha)
            self.feature_prob_[c] = (np.sum(X_c, axis=0) + self.alpha) / (n_c + 2 * self.alpha)
        return self

    def predict(self, data):
        predictions = []
        for i in range(data.shape[0]):
            x = data[i]
            log_probs = {}
            for c in self.classes_:
                # Retrieve probability vector for class c
                p = self.feature_prob_[c]
                # Compute log likelihood: sum over features of:
                # x_j * log(p_j) + (1 - x_j) * log(1 - p_j)
                log_likelihood = np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))
                log_prior = np.log(self.class_prior_[c])
                log_probs[c] = log_prior + log_likelihood
            # Predict class with highest log probability
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)

    def score(self, test_data, labels):

        y_pred = self.predict(test_data)
        return np.mean(y_pred == labels)
    

class GaussianNBClassifier:
    def __init__(self):
     
        self.epsilon = 1e-9  # To avoid division by zero in variance computation
        self.classes_ = None    # Unique class labels
        self.means_ = {}        # Mean of features for each class
        self.vars_ = {}         # Variance of features for each class
        self.priors_ = {}       # Prior probability for each class

    def fit(self, data, labels):
        self.classes_ = np.unique(labels)
        n_samples, n_features = data.shape
        
        for c in self.classes_:
            # Select all samples belonging to class c
            X_c = data[labels == c]
            # Compute mean and variance per feature for class c
            self.means_[c] = np.mean(X_c, axis=0)
            self.vars_[c] = np.var(X_c, axis=0) + self.epsilon  # Add epsilon for stability
            # Compute the prior probability for class c
            self.priors_[c] = X_c.shape[0] / n_samples

    def _gaussian_log_probability(self, class_label, sample):
        mean = self.means_[class_label]
        var = self.vars_[class_label]
        # Compute log probability for each feature and sum them up.
        log_prob = -0.5 * np.log(2 * np.pi * var) - ((sample - mean) ** 2) / (2 * var)
        return np.sum(log_prob)

    def predict(self, test_data):
        predictions = []
        for test_sample in test_data:
            # Compute the log posterior for each class
            posteriors = {}
            for c in self.classes_:
                log_prior = np.log(self.priors_[c])
                log_likelihood = self._gaussian_log_probability(c, test_sample)
                posteriors[c] = log_prior + log_likelihood
            # Choose the class with the highest posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

    def score(self, test_data, test_labels):
        y_pred = self.predict(test_data)
        return np.mean(y_pred == test_labels)
    

def plot_gaussian_contours_pc1_pc2(X_full, y, classifier, title, resolution=100):

    X_2d = X_full[:, :2]  # Only first 2 PCA components for plotting
    classes = np.unique(y)
    colors = plt.cm.get_cmap('tab10', len(classes))

    plt.figure(figsize=(8, 6))

    # Scatter actual points in PC1 vs PC2
    for i, c in enumerate(classes):
        plt.scatter(X_2d[y == c, 0], X_2d[y == c, 1], label=f'Class {c}', alpha=0.4, s=15, color=colors(i))

    # Meshgrid for contour plotting
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Plot 2D marginal Gaussians (PC1-PC2)
    for i, c in enumerate(classes):
        mean = classifier.means_[c][:2]
        cov = np.diag(classifier.vars_[c][:2])
        rv = multivariate_normal(mean, cov)
        zz = rv.pdf(grid).reshape(xx.shape)
        plt.contour(xx, yy, zz, levels=3, colors=[colors(i)])

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()



# Bernoulli
bnb = BernoulliNBClassifier()
bnb.fit(X_train_flat_bin, train_labels)
accuracy = bnb.score(X_test_flat_bin, test_labels)
print(f"Bernoulli Naive Bayes Accuracy: {accuracy:.4f}")


n_components = [2]

for i in range(len(n_components)):    
    # PCA for Gaussian
    pca = PCA(n_components= n_components[i])
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    # Gaussian
    gnb = GaussianNBClassifier()
    gnb.fit(X_train_pca, train_labels)
    accuracy_pca = gnb.score(X_test_pca, test_labels)
    print(f"Gaussian Naive Bayes with 50 components PCA Accuracy: {accuracy_pca * 100:.2f}%")

# Plot 2D marginal of Gaussians (PC1 vs PC2)
plot_gaussian_contours_pc1_pc2(X_train_pca, train_labels, gnb, title="Train Set - PC1 vs PC2")
plot_gaussian_contours_pc1_pc2(X_test_pca, test_labels, gnb, title="Test Set - PC1 vs PC2")
