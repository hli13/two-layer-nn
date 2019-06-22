# -*- coding: utf-8 -*-

import numpy as np
import h5py
import time
import copy
from random import randint
import sys

# load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

####################################################################################
# Implementation of stochastic gradient descent algorithm for nerual networks

# number of inputs
num_inputs = 28*28
# number of outputs
num_outputs = 10
# number of hidden units
num_hidden = 50
# nonlinearity type
func = 'tanh'
#func = 'sigmoidal'

# initialization
model = {}
model['W'] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
model['C'] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model['b1'] = np.random.randn(num_hidden) / np.sqrt(num_hidden)
model['b2'] = np.random.randn(num_outputs) / np.sqrt(num_outputs)
model_grads = copy.deepcopy(model)

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

time1 = time.time()
# initialize learning rate
LR = .01
num_epochs = 20

for epochs in range(num_epochs):
    
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001     
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
        
    total_correct = 0
    
    for n in range( len(x_train)):
        # randomly select a new data sample
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        
        # forward step
        (Z, H, f) = forward(x, y, model, func)
        
        # check the prediction accuracy
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        
        # backward step
        model_grads = backward(x, y, f, Z, H, model, model_grads, func)
        
        # update parameters
        model['W'] = model['W'] + LR*model_grads['W']
        model['C'] = model['C'] + LR*model_grads['C']
        model['b1'] = model['b1'] + LR*model_grads['b1']
        model['b2'] = model['b2'] + LR*model_grads['b2']
        
    print("Epoch # %3d,  Accuracy %8.4f" % ( epochs, total_correct/np.float(len(x_train) ) ) )

time2 = time.time()
print("Training Time: %8.4f (s)" % ( time2-time1 ) )

######################################################
# test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    (_, _, f) = forward(x, y, model, func)
    prediction = np.argmax(f)
    if (prediction == y):
        total_correct += 1
print("Test Accuracy %8.4f" % ( total_correct/np.float(len(x_test) ) ) )