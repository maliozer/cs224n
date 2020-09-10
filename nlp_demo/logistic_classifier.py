#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning and gate with logistic regression

"""
import numpy as np
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

def initizalize_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0
    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def cross_entropy_loss_function(m, A, Y):
    c = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))
    return c


def propagate(w,b,X,Y):
    m = X.shape[1]
    z = (np.dot(w.T, X) + b) 
    
    #compute activation
    A = sigmoid(z)
    
    #compute cross-entropy-loss
    cost = cross_entropy_loss_function(m, A, Y)
    
    #backward propagation
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)* np.sum(A - Y)
    grads = {"dw" : dw, "db" : db}
    
    return grads, cost
    

"""
# Optimization
The goal is to learn ww and bb by minimizing the cost function J. 
For a parameter θ, the update rule is θ=θ−α dθ=θ−α dθ, 
where α is the learning rate.
"""

def optimize(w, b, X, Y, num_iterations, learning_rate):
    #cost cache
    costs = []
    
    for i in trange(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        #update rule
        w = w - (learning_rate * dw) 
        b = b - (learning_rate * db)
        
        if i % 100 == 0:
            costs.append(cost)
    
    params = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0,i] > 0.5 else 0
    
    return Y_prediction
    

if __name__ == '__main__':
    w, b, X, Y = np.random.uniform(size=2), np.random.uniform(size=1), np.array([[0.,1.,0.,1],[0.,0.,1.,1]]), np.array([[0,0,0,1]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations= 10000, learning_rate = 0.01)
    

    X_test = np.array([np.random.uniform(-1,2,2) for x in range(10)])
    preds = predict(params["w"], params["b"], X_test.T)
    print(preds)
    
    model_costs = np.squeeze(costs)
    
    plt.plot(model_costs)
    plt.ylabel('training cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = 0.1")
    plt.show()
