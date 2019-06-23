"""
Pipeline for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
from random import randint
import time
import nn_utils

# load the MNIST dataset
mnist_dir = './MNISTdata.hdf5'
mnist = nn_utils.load_mnist(mnist_dir)

# parse arguments/hyperparameters
params = nn_utils.parse_params()

# initialization
(model, model_grads) = nn_utils.init_model(mnist,params)

# training the model
print("\nStart training")
print("---------------")

time_start = time.time()

# initial learning rate
LR = params.lr

for epochs in range(params.n_epochs):
    
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
        (Z, H, f) = nn_utils.forward(x, model, params.sigma)
        
        # check prediction accuracy
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        
        # backpropagation step
        model_grads = nn_utils.backprop(x, y, f, Z, H, model, model_grads, params.sigma)
        
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
    (_, _, f) = nn_utils.forward(x, model, params.sigma)
    prediction = np.argmax(f)
    
    # check prediction accuracy
    if (prediction == y):
        total_correct += 1
        
print("Test Accuracy : %6.4f" % ( total_correct/np.float(mnist['n_test']) ) )