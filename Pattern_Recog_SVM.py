#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:17:00 2025

@author: batdora
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mahotas
from cvxopt import matrix, solvers
import pandas as pd
import time
from sklearn.svm import SVC

start = time.time()

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

# Dissect rabbit data
rabbit_train_labels = np.where(train_labels == 0)
rabbit_test_labels = np.where(test_labels == 0)

train_data_rabbit = train_images[rabbit_train_labels]
test_data_rabbit = test_images[rabbit_test_labels]

train_data_rabbit_bin = train_images_binary[rabbit_train_labels]
test_data_rabbit_bin = test_images_binary[rabbit_test_labels]

# Dissect hand data
hand_train_labels = np.where(train_labels == 2)
hand_test_labels = np.where(test_labels ==2)

train_data_hand = train_images[hand_train_labels]
test_data_hand = test_images[hand_test_labels]

train_data_hand_bin = train_images_binary[hand_train_labels]
test_data_hand_bin = test_images_binary[hand_test_labels]

# Combine Data
train_data = np.concatenate([train_data_rabbit, train_data_hand])
train_data_bin = np.concatenate([train_data_rabbit_bin, train_data_hand_bin])

test_data = np.concatenate([test_data_rabbit,test_data_hand])
test_data_bin = np.concatenate([test_data_rabbit_bin,test_data_hand_bin])

# Create labels (rabbit = 1, hand = -1)
train_labels = np.ones(8000)
train_labels[4000:] = -1

test_labels = np.ones(2000)
test_labels[1000:] = -1

# Flatten Images
train_data_flat = train_data.reshape(train_data.shape[0], -1)
test_data_flat = test_data.reshape(test_data.shape[0], -1)

# Flatten Images Binary
train_data_flat_bin = train_data_bin.reshape(train_data.shape[0], -1)
test_data_flat_bin = test_data_bin.reshape(test_data.shape[0], -1)

""" Zernike Features """
def compute_zernike_features(images, max_moments, radius=14):
    """
    radius: radius for Zernike computation
    max_moments: how many zernike values to keep
    """
    feature_list = []
    for img in images:
        z = mahotas.features.zernike_moments(img, radius)
        feature_list.append(z[:max_moments])
    return np.array(feature_list)



def soft_margin_SVM_primal_QP_formatter (data, labels, C):
    
    n, d = data.shape
    
    num_variables = 1+d+n #b + w + slack
    
    """
    slack is per sample
    w is for all samples but is d dimensional
    b is scalar
    """
    
    # Regularization Parameters (only w)
    Q = np.zeros((num_variables,num_variables))
    Q[1:d+1,1:d+1] = np.eye(d)
    
    # Slack variable cost
    p = np.zeros(num_variables)
    p[d+1:] = C
    
    # Inequalities
    
    """
    write as smaller than for KKT conditions (so that QP solver works)
    
    so  the formula becomes
    
    Constraint 1 is
    -y*(w.T@x+b)-slack <= -1
    
    Constraint 2 is
    -slack <= 0
    
    """
    # Constraint 1
    A = np.zeros ((n, num_variables))
    A[:,0] = -1*labels #b
    A[:, 1:d+1] = -1*(labels[:,None]*data) #w
    
    """
    only make the slack of the index 1, leave other slacks per data as 0
    like an identity matrix
    this way you can index each sample's slack with only its slack
    """
    A[:, d+1:] = -np.eye(n)
    
    """
    A*u <= c
    A = [n, 1+d+n]
    u = [1+d+n, 1]
    c = [n,1]
    """
    c = np.ones(n)*-1
    
    #Constraint 2
    A_slack = np.zeros((n, num_variables))
    A_slack[:, d+1:] = -np.eye(n)
    c_slack = np.zeros(n)
    
    
    """
    give to QP solver as this
    G*x <= h
    
    but for that, stack A with A slack and c with c slack
    """
    
    G = np.vstack([A, A_slack])
    h = np.vstack([c,c_slack]).reshape(-1) #from (2n,1) matrix to (2n,) vector
    
    return (
        matrix(Q), 
        matrix(p), 
        matrix(G), 
        matrix(h)
    )
    

def predict(w,b,test_data_pca_zernike):
    
    scores = test_data_pca_zernike@w + b
    
    return np.where(scores >= 0, 1, -1)


def create_batches(X, X_labels, batch_size, seed=42):
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Take the number of total
    num_samples = X.shape[0]
    
    # Create a random permutation of indices and shuffle accordingly
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

n_components = [70]
C = [1]
zernike_hyper = [25]
results = []

# To test make false, for validation make true
val = False

# For home brew SVM switch true
scratch_SVM = False

# If linear SVM switch to false
non_linear = True
# set poly to false for RBF
poly = [3]
coeff = [0.1]
kernel = "poly"
counter = 0

### Current setup is for binarized data ###

for k in range(len(zernike_hyper)):
    
    # Zernike
    zernike = compute_zernike_features(train_data_bin, max_moments=zernike_hyper[k])

    zernike_test = compute_zernike_features(test_data_bin, max_moments=zernike_hyper[k])

    for i in range(len(n_components)):    
        
        # PCA
        pca = PCA(n_components= n_components[i])
        train_data_pca = pca.fit_transform(train_data_flat_bin)
        test_data_pca = pca.transform(test_data_flat_bin)
        
        # Append zernike to PCA
        train_data_pca_zernike = np.hstack([train_data_pca, zernike])
        test_data_pca_zernike = np.hstack([test_data_pca, zernike_test])
        
        dims = test_data_pca_zernike.shape[1]
        
        if val == True:
            # Create validation sets
            batches, batches_labels = create_batches(train_data_pca_zernike, train_labels, batch_size=1600)
            
            for j in range(len(C)): 
                
                cross_fold = []
                
                counter +=1
                
                counter2 = 0
                
                for m in range(5):
                    
                    counter2 +=1
                    validation = batches[m]
                    validation_labels = batches_labels[m]
                    
                    training = np.concatenate([batches[l] for l in range(5) if l != m], axis=0)
                    training_labels = np.concatenate([batches_labels[l] for l in range(5) if l != m], axis=0)
    
                    if scratch_SVM == True:
                    
                        Q,p,G,h = soft_margin_SVM_primal_QP_formatter(training, training_labels, C[j])
                        solution = solvers.qp(Q,p,G,h)
                        
                        #solution comes in dict format with outputs in key x
                        
                        u = np.array(solution["x"]).flatten()
                        
                        b = u[0]
                        w = u[1:dims+1]
                        
                        y_pred = predict(w, b, validation)
                        
                    else:
                        if non_linear == False:
                            model = SVC(C = C[j], kernel = kernel)
                        else:
                            if poly == False:
                                model = SVC(C = C[j], kernel = kernel, gamma="scale")
                            else:
                                model = SVC(C = C[j], kernel = kernel, gamma="scale", degree=poly[0], coef0=coeff[0])
            
                        model.fit(training,training_labels)
                        y_pred = model.predict(validation)
                        
                    print(f"Cross validation number {counter2}")   
                    accuracy_val = np.mean(y_pred == validation_labels)
                    cross_fold.append(accuracy_val)
                    
                print(f"Done with run {counter}, PCA {n_components[i]}, C = {C[j]}")     
                accuracy = np.mean(cross_fold)  
                # add zernike_hyper[k] if you are tuning zerni
                results.append([n_components[i],C[j], zernike_hyper[k], accuracy])
                    
        else:
            
            if scratch_SVM == True:
                Q,p,G,h = soft_margin_SVM_primal_QP_formatter(train_data_pca_zernike, train_labels, C[0])
                solution = solvers.qp(Q,p,G,h)
                
                #solution comes in dict format with outputs in key x
                
                u = np.array(solution["x"]).flatten()
                
                b = u[0]
                w = u[1:dims+1]
                
                y_pred = predict(w, b, test_data_pca_zernike)
            
            else:
                # Linear SVM
                if non_linear == False:
                    model = SVC(C = C[0], kernel = kernel)
                    print("Running test for linear SVM")
                else:
                    # RBF SVM
                    if poly == False:
                        model = SVC(C = C[0], kernel = kernel, gamma="scale")
                        print("Running test for RBF SVM")
                    # Polynomial SVM
                    else:
                        model = SVC(C = C[0], kernel = kernel, gamma="scale", degree=poly[0], coef0=coeff[0])
                        print("Running test for Poly SVM")
                model.fit(train_data_pca_zernike,train_labels)
                y_pred = model.predict(test_data_pca_zernike)
                
            accuracy = np.mean(y_pred == test_labels)
            results.append([n_components[i],C, zernike_hyper[k], accuracy])
            
            print(accuracy)

            
        
end = time.time()
elapsed = end - start
print(f"Took {elapsed:.2f} seconds")                
        
df_results = pd.DataFrame(results, columns=["PC_dims", "C", "Zernike", "accuracy"])
print("Check df_results for results table")

# Plot Support Vectors and Furthest Vectors

# Get signed distances to decision boundary
distances = model.decision_function(train_data_pca_zernike)

# Mask per class
class_0_mask = (train_labels == -1)
class_1_mask = (train_labels == 1)

# Get distances per class
dist_0 = distances[class_0_mask]
dist_1 = distances[class_1_mask]

# Get indices of top 2 furthest (absolute value) per class
furthest_0_idx = np.argsort(np.abs(dist_0))[-2:]  # top 2 for class 0
furthest_1_idx = np.argsort(np.abs(dist_1))[-2:]  # top 2 for class 1

# Map back to original indices in X_train
true_indices_0 = np.where(class_0_mask)[0][furthest_0_idx]
true_indices_1 = np.where(class_1_mask)[0][furthest_1_idx]

# Final indices: 2 from each class
furthest_indices = np.concatenate([true_indices_0, true_indices_1])

furthest_hand_1 = train_data_bin[furthest_indices[0]]
furthest_hand_2 = train_data_bin[furthest_indices[1]]

furthest_rab_1 = train_data_bin[furthest_indices[2]]
furthest_rab_2 = train_data_bin[furthest_indices[3]]

# Get Support Vectors
support_vectors = model.support_

sup_vec_rab_1= train_data_bin[support_vectors[-1]]
sup_vec_rab_2= train_data_bin[support_vectors[-2]]

sup_vec_hand_1= train_data_bin[support_vectors[0]]
sup_vec_hand_2= train_data_bin[support_vectors[1]]

# Collect all images
images = [
    furthest_hand_1,
    furthest_hand_2,
    furthest_rab_1,
    furthest_rab_2,
    sup_vec_hand_1,
    sup_vec_hand_2,
    sup_vec_rab_1,
    sup_vec_rab_2
   
]

titles = [
    "Furthest Hand 1",
    "Furthest Hand 2",
    "Furthest Rabbit 1",
    "Furthest Rabbit 2",
    "Support Hand 1",
    "Support Hand 2",
    "Support Rabbit 1",
    "Support Rabbit 2"
    
]

# Create figure
fig, axes = plt.subplots(1, 8, figsize=(20, 3))

for i, ax in enumerate(axes):
    ax.imshow(images[i].reshape(28, 28), cmap='gray')
    ax.set_title(titles[i], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()




