import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    lr = 0.01
    n_samples, n_features = train_set.shape
    W = np.zeros(n_features)
    b = 0
    for i in range(max_iter):
        for j in range(n_samples):
            y_pred = np.dot(W, train_set[j]) + b
            # gradient
            if y_pred <=0: y_pred = 0
            else: y_pred = 1
            W += lr * (train_labels[j] - y_pred) * train_set[j]
            b += lr * (train_labels[j] - y_pred)
            if j%10000 == 0 and i %10 ==0: print("Iteration: ", i, "Sample: ", j)
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    W, b = trainPerceptron(train_set, train_labels, max_iter)
    n_samples, n_features = dev_set.shape
    y_pred = np.zeros(n_samples)
    ret = []
    y_pred = np.dot(dev_set, W) + b
    for i in range(len(y_pred)):
        if y_pred[i] <= 0: ret.append(0)
        else: ret.append(1)
    return ret
    



