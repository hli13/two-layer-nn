"""
Functions for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
import sys

def sigma(z, func):
    """
    Activation functions
    
    Parameters
    ----------
    z : float
        input
    func : str
        the type of activation adopted
        
    Returns
    -------
    ZZ : float
        output
    """
    if func == 'tanh':
        ZZ = np.tanh(z)
    elif func == 'sigmoid':
        ZZ = np.exp(z)/(1 + np.exp(z))
    else:
        sys.exit("Unsupported function type!")
    return ZZ

def d_sigma(z, func):
    """
    Derivative of activation functions
    
    Parameters
    ----------
    z : float
        input
    func : str
        the type of activation
        
    Returns
    -------
    dZZ : float
        output
    """
    if func == 'tanh':
        dZZ = 1.0 - np.tanh(z)**2
    elif func == 'sigmoid':
        dZZ = np.exp(z)/(1 + np.exp(z)) * (1 - np.exp(z)/(1 + np.exp(z)))
    else:
        sys.exit("Unsupported function type!")
    return dZZ

def softmax_function(z):
    """
    Softmax function
    
    Parameters
    ----------
    z : float
        input
        
    Returns
    -------
    ZZ : float
        output
    """
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def forward(x, model, func):
    """
    Forward propagation of a two-layer neural network
    
    Parameters
    ----------
    x : float
        input
    model : dict
        parameters/weights of the nerual network
    func : str
        the type of activation
        
    Returns
    -------
    Z : float
        output of the linear layer
    H : float
        output after the activation
    f : float
        output of the forward propagation
    """
    Z = np.dot(model['W'], x) + model['b1']
    H = sigma(Z,func)
    U = np.dot(model['C'], H) + model['b2']
    f = softmax_function(U)
    return (Z, H, f)

def backprop(x, y, f, Z, H, model, model_grads, func):
    """
    Backpropagation of a two-layer neural network
    
    Parameters
    ----------
    x : float
        input
    y : int
        ground truth label
    f : float
        output of the forward propagation
    Z : float
        output of the linear layer
    H : float
        output after the activation
    model : dict
        parameters/weights of the nerual network
    model_grads : dict
        Gradients of the parameters/weights of the nerual network
    func : str
        the type of activation
        
    Returns
    -------
    model_grads : dict
        Updated gradients of the parameters/weights of the nerual network
    """
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