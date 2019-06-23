"""
Functions for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
import h5py
import sys

def load_mnist(mnist_dir):
    """
    Load the MNIST dataset
    
    Parameters
    ----------
    func : str
        directory of the MNIST data
        
    Returns
    -------
    mnist : dict
        a dictionary containing the training and test data as well as data 
        sizes and shapes
    """
    MNIST_data = h5py.File(mnist_dir, 'r')
    mnist = {}
    mnist['x_train'] = np.float32( MNIST_data['x_train'][:] )
    mnist['y_train'] = np.int32( np.array( MNIST_data['y_train'][:,0] ) )
    mnist['x_test'] = np.float32( MNIST_data['x_test'][:] )
    mnist['y_test'] = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
    MNIST_data.close()
    
    mnist['n_train'] = mnist['x_train'].shape[0]
    mnist['n_test'] = mnist['x_test'].shape[0]
    mnist['n_input'] = mnist['x_train'].shape[1] # image size 28*28=784
    mnist['n_output'] = len(np.unique(mnist['y_test'])) # num of labels = 10
    
    # print data info
    print("\nMNIST data info")
    print("----------------")
    print("Number of training data: %d" % mnist['n_train'])
    print("Number of test data: %d"  % mnist['n_test'])
    print("Input data shape: %d" % mnist['n_input'])
    print("Output data shape: %d" % mnist['n_output'])
    
    return mnist

def sigma(z, func):
    """
    Activation functions
    
    Parameters
    ----------
    z : ndarray of float
        input
    func : str
        the type of activation adopted
        
    Returns
    -------
    ZZ : ndarray of float
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
    z : ndarray of float
        input
    func : str
        the type of activation
        
    Returns
    -------
    dZZ : ndarray of float
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
    z : ndarray of float
        input
        
    Returns
    -------
    ZZ : ndarray of float
        output
    """
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def forward(x, model, func):
    """
    Forward propagation of a two-layer neural network
    
    Parameters
    ----------
    x : ndarray of float
        input
    model : dict
        parameters/weights of the nerual network
    func : str
        the type of activation
        
    Returns
    -------
    Z : ndarray of float
        output of the linear layer
    H : ndarray of float
        output after the activation
    f : ndarray of float
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
    x : ndarray of float
        input
    y : ndarray of int
        ground truth label
    f : ndarray of float
        output of the forward propagation
    Z : ndarray of float
        output of the linear layer
    H : ndarray of float
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