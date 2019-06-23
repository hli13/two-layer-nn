# -*- coding: utf-8 -*-

import numpy as np
import sys

# element-wise nonliearity
def sigma(z, func):
    if func == 'tanh':
        ZZ = np.tanh(z)
    elif func == 'sigmoidal':
        ZZ = np.exp(z)/(1 + np.exp(z))
    else:
        sys.exit("Unsupported function type!")
    return ZZ

def d_sigma(z, func):
    if func == 'tanh':
        dZZ = 1.0 - np.tanh(z)**2
    elif func == 'sigmoidal':
        dZZ = np.exp(z)/(1 + np.exp(z)) * (1 - np.exp(z)/(1 + np.exp(z)))
    else:
        sys.exit("Unsupported function type!")
    return dZZ

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def forward(x, y, model, func):
    Z = np.dot(model['W'], x) + model['b1']
    H = sigma(Z,func)
    U = np.dot(model['C'], H) + model['b2']
    f = softmax_function(U)
    return (Z, H, f)

def backward(x, y, f, Z, H, model, model_grads, func):
    dU = - 1.0*f
    dU[y] = dU[y] + 1.0
    db2 = dU
    dC = np.outer(dU, np.transpose(H))
    delta = np.dot(np.transpose(model['C']), dU)
    dZZ = d_sigma(Z,func)
    db1 = np.multiply(delta, dZZ)
    dW = np.outer(db1,np.transpose(x))
    model_grads['W'] = dW
    model_grads['C'] = dC
    model_grads['b1'] = db1
    model_grads['b2'] = db2        
    return model_grads