"""
Pipeline for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
from random import randint
import time
import copy
import argparse
import nn_utils

# load the MNIST dataset
mnist_dir = './MNISTdata.hdf5'
mnist = nn_utils.load_mnist(mnist_dir)

# parse the arguments
parser = argparse.ArgumentParser()

# hyperparameters
parser.add_argument('--lr', type=float, default=0.1, 
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--decay', type=float, default=0.1, 
                    help='learning rate decay (default: 0.1)')
parser.add_argument('--interval', type=int, default=5, 
                    help='staircase interval for learning rate decay (default: 5')
parser.add_argument('--n_epochs', type=int, default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--n_h', type=int, default=32,
                    help='number of hidden units (default: 32)')
parser.add_argument('--sigma', type=str, default='sigmoid',
                    help='type of activation function (default: sigmoid)')

params = parser.parse_args()

# number of inputs
num_inputs = mnist['n_input']
# number of outputs
num_outputs = mnist['n_output']
# number of hidden units
num_hidden = params.n_h
# activation function type
func = params.sigma
# initial learning rate
LR = params.lr
# length of training
num_epochs = params.n_epochs

# initialization
model = {}
model['W'] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
model['C'] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model['b1'] = np.random.randn(num_hidden) / np.sqrt(num_hidden)
model['b2'] = np.random.randn(num_outputs) / np.sqrt(num_outputs)
model_grads = copy.deepcopy(model)

# print hyperparameters for training
print("\nHyperparameters")
print("-----------------")
print("Initial learning rate : %6.4f" % LR)
print("Learning rate decay : %6.4f" % params.decay)
print("Staircase learning rate decay interval : %d" % params.interval)
print("Number of epochs : %d" % num_epochs)
print("Number of hidden units : %d" % num_hidden)
print("Activation function : %s" % func)

# training the model
print("\nStart training")
print("---------------")

time_start = time.time()

for epochs in range(num_epochs):
    
    # learning rate schedule: staircase decay
    if (epochs > 0 and epochs % params.interval == 0):
        LR *= params.decay
        
    total_correct = 0
    
    for n in range( mnist['n_train']):
        
        # randomly select a new data sample
        n_random = randint(0,mnist['n_train']-1 )
        y = mnist['y_train'][n_random]
        x = mnist['x_train'][n_random][:]
        
        # forward step
        (Z, H, f) = nn_utils.forward(x, model, func)
        
        # check prediction accuracy
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        
        # backpropagation step
        model_grads = nn_utils.backprop(x, y, f, Z, H, model, model_grads, func)
        
        # update model parameters
        model['W'] = model['W'] + LR*model_grads['W']
        model['C'] = model['C'] + LR*model_grads['C']
        model['b1'] = model['b1'] + LR*model_grads['b1']
        model['b2'] = model['b2'] + LR*model_grads['b2']
        
    print("Epoch %3d,  Accuracy %6.4f" % ( epochs, total_correct/np.float(mnist['n_train'] ) ) )

time_end = time.time()
print("Training Time : %8.4f (s)" % ( time_end-time_start ) )

# testing the model
print("\nStart testing")
print("--------------")

total_correct = 0

for n in range( mnist['n_test']):
    
    # load test data sample
    y = mnist['y_test'][n]
    x = mnist['x_test'][n][:]
    
    # forward step and prediction
    (_, _, f) = nn_utils.forward(x, model, func)
    prediction = np.argmax(f)
    
    # check prediction accuracy
    if (prediction == y):
        total_correct += 1
        
print("Test Accuracy : %6.4f" % ( total_correct/np.float(mnist['n_test']) ) )