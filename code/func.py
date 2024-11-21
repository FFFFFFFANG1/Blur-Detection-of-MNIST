import numpy as np
# import tensorflow as tf
import torch
import torch.nn.functional as F

def pca(X, k):
#mean of each feature
  (n_samples, n_features) = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data

# for logistic regression
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def loss(y, y_hat):
   return F.binary_cross_entropy(y_hat, y)
    
# def gradients(X, y, y_hat):
#     loss = loss(y, y_hat)
#     loss.backward()


def update_weights(w, b, X, y, lr):
    y_hat = predict(w, b, X)
    loss = loss(y, y_hat)
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    return w, b

def logistic_regression(X, y, w, b, lr, epochs):
    for i in range(epochs):
        y_hat = predict(w, b, X)
        w, b = update_weights(w, b, X, y, lr)
    return w, b
    

def predict(w, b, X):
    ret =  sigmoid(np.dot(X, w) + b)
    for i in range(len(ret)):
        if ret[i] > 0.5:
            ret[i] = 1
        else:
            ret[i] = -1
    return ret

def accuracy(y, y_hat):
    count = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            count += 1
    return count / len(y)

def simple(val_extracted, val_labels, threshold):
    pred = []
    for i in range(len(val_extracted)):
        if val_extracted[i] > threshold:
            pred.append(-1)
        else:
            pred.append(1)

    acc = 0
    for i in range(len(val_labels)):
        if val_labels[i] == pred[i]:
            acc += 1
    return acc / len(val_labels)
