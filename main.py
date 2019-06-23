"""
Pipeline for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
from random import randint
import time
import copy

import nn_utils

# load MNIST data
mnist_dir = './MNISTdata.hdf5'
mnist = nn_utils.load_mnist(mnist_dir)

####################################################################################
# Implementation of stochastic gradient descent algorithm for nerual networks

# number of inputs
num_inputs = mnist['n_input']
# number of outputs
num_outputs = mnist['n_output']
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

print("\nStart training")
print("---------------")

for epochs in range(num_epochs):
    
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001     
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
        
    total_correct = 0
    
    for n in range( mnist['n_train']):
        # randomly select a new data sample
        n_random = randint(0,mnist['n_train']-1 )
        y = mnist['y_train'][n_random]
        x = mnist['x_train'][n_random][:]
        
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
        
    print("Epoch # %3d,  Accuracy %8.4f" % ( epochs, total_correct/np.float(mnist['n_train'] ) ) )

time2 = time.time()
print("Training Time: %8.4f (s)" % ( time2-time1 ) )

######################################################
# test data
print("\nStart testing")
print("--------------")
total_correct = 0
for n in range( mnist['n_test']):
    y = mnist['y_test'][n]
    x = mnist['x_test'][n][:]
    (_, _, f) = nn_utils.forward(x, model, func)
    prediction = np.argmax(f)
    if (prediction == y):
        total_correct += 1
print("Test Accuracy %8.4f" % ( total_correct/np.float(mnist['n_test']) ) )