"""
Pipeline for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
import h5py
import time
import copy
from random import randint

import nn_utils

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
#func = 'tanh'
func = 'sigmoid'

# initialization
model = {}
model['W'] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
model['C'] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model['b1'] = np.random.randn(num_hidden) / np.sqrt(num_hidden)
model['b2'] = np.random.randn(num_outputs) / np.sqrt(num_outputs)
model_grads = copy.deepcopy(model)

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
        (Z, H, f) = nn_utils.forward(x, model, func)
        
        # check the prediction accuracy
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        
        # backward step
        model_grads = nn_utils.backprop(x, y, f, Z, H, model, model_grads, func)
        
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
    (_, _, f) = nn_utils.forward(x, model, func)
    prediction = np.argmax(f)
    if (prediction == y):
        total_correct += 1
print("Test Accuracy %8.4f" % ( total_correct/np.float(len(x_test) ) ) )