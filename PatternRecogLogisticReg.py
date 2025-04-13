#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:33:01 2025

@author: batdora
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

def categorical_crossentropy_loss(y_true, y_pred):
   
    # Small value to avoid log(0)
    epsilon = 1e-9
    # Compute the loss for each sample and then average
    sample_losses = -np.sum(y_true * np.log(y_pred + epsilon), axis=1)
    return np.mean(sample_losses)


#This is combined derivative for using with SoftMax
def derCrossEntropy(y_true, y_pred):
    return y_pred-y_true

def encoder(labels):
    
    vector = np.zeros([len(labels),5])
    
    for i, k in enumerate(labels):
        vector[i][k]= 1
        
    return np.array(vector)


def xavier_initialize(fan_in, fan_out=5, seed=42):
    np.random.seed(seed)
    stddev = np.sqrt(2 / (fan_in + fan_out))
    weight = np.random.randn(fan_in, fan_out) * stddev
    bias = np.zeros(fan_out)
    return weight, bias


def Softmax(vector):
    
    x_shifted = vector - np.max(vector, axis=-1, keepdims=True)
    
    exps = np.exp(x_shifted)

    softmax_values = exps / np.sum(exps, axis=-1, keepdims=True)
    
    return softmax_values

# Batch Creator

def create_batches(X, X_labels, batch_size, seed=42):
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Take the number of total
    num_samples = X.shape[0]
    
    # Create a random permutation of indices and shuffle X_flat accordingly
    indices = np.random.permutation(num_samples)
    X_shuffled = X[indices]
    X_labels_shuffled = X_labels[indices]
    
    batches = []
    batches_labels = []
    
    for i in range(0, num_samples, batch_size):
        #For data
        batch = X_shuffled[i:i+batch_size]
        batches.append(batch)
        
        #For label
        label = X_labels_shuffled[i:i+batch_size]
        batches_labels.append(label)
    
    return batches, batches_labels


def forward_pass(batch, w, b):
    
    a = batch@w + b
    
    y_pred = Softmax(a)

    return y_pred


def back_propagation (batch, y_pred, y_true, w, b, learning_rate = 0.01):
    
    error = derCrossEntropy(y_true, y_pred)  # shape: (batch_size, num_classes)
    
    db = np.mean(error, axis=0)  # shape: (num_classes,)
    dw = (1 / batch.shape[0]) * (batch.T @ error)

    
    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b
    
    
def predict(X, weights, biases, batch_size=20):
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    predictions = []
    for i in range(0, num_samples, batch_size):
        batch = X_flat[i:i+batch_size]
        outputs = forward_pass(batch, weights, biases)
        preds = np.argmax(outputs, axis=1)
        predictions.extend(preds)
    return np.array(predictions)
    
    
train_labels=encoder(train_labels)
test_labels=encoder(test_labels)


n_components = [2]
batch_num = 20
train_loss_log = []
train_accuracy_log = []
accuracy_log = []
epoch_num = 40


for i in range(len(n_components)):    
    # PCA for Gaussian
    pca = PCA(n_components= n_components[i])
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    batches, batches_labels = create_batches(X_train_pca,train_labels, batch_num)
    
    w, b = xavier_initialize(n_components[i])

    for epoch in range(epoch_num): 
        total_loss = 0
        for j in range(int(X_train_pca.shape[0]/batch_num)):
            
            current_batch = batches[j]
            
            y_true = batches_labels[j]
            
            y_pred = forward_pass(current_batch, w, b)
            
            loss = categorical_crossentropy_loss(y_true, y_pred)
            
            total_loss += loss
            
            w, b = back_propagation(current_batch, y_pred, y_true, w, b)
            
        avg_train_loss = total_loss / len(batches)
        train_loss_log.append(avg_train_loss)
        
        train_preds = predict(X_train_pca, w, b, batch_size=20)
        train_accuracy = np.mean(train_preds == np.argmax(train_labels, axis=1))
        train_accuracy_log.append(train_accuracy)

        
        if epoch % 1 == 0:  # Print every epoch
            test_preds = predict(X_test_pca, w, b, batch_size=20)
            accuracy = np.mean(test_preds == np.argmax(test_labels, axis=1))
            accuracy_log.append(accuracy)
            print(f"Epoch {epoch+1}/{epoch_num}, Avg Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Plotting
    epochs_range = range(1, epoch_num + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_log, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracy_log, label='Train Accuracy')
    plt.plot(epochs_range, accuracy_log, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("loss_accuracy_plot.png")
    plt.show()
        
    
    
    
    
    
    
    
    