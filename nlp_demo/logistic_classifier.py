#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:08:47 2020

@author: blanc
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def cross_entropy_loss_function(w,b,X,Y):
    m = X.shape[1]
    z = np.dot(w.T,x)+b
    A = sigmoid(z) #compute activation
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A))) #compute cost
    return cost
    

def initizalize_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0
    return w, b


def loss():
     pass














