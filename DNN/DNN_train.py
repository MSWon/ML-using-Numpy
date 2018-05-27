# -*- coding: utf-8 -*-
"""
Created on Sun May 27 18:37:42 2018

@author: Big Data Guru
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
import DNN_MNIST


output_node = 10
hidden_node = 200
learning_rate = 0.015
training_epoch = 400

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)

np.shape(mnist.train.images[1])  ## (784,1)
np.shape(mnist.train.images)

plt.imshow(mnist.train.images[1].reshape(28,28) , cmap = 'Greys')  ## plotting pic

train_X = mnist.train.images.T
train_Y = mnist.train.labels.T
test_X = mnist.test.images.T
test_Y = mnist.test.labels.T

dnn = DNN_MNIST.DNN(output_node, hidden_node)

## training

init = dnn.initialize(train_X)

W1 = init["W1"] ## input -> hidden
b1 = init["b1"] ## input -> hidden
W2 = init["W2"] ## hidden -> output
b2 = init["b2"] ## hidden -> output

for i in range(training_epoch):
    
    f_p = dnn.foward_propagate(train_X, W1, b1, W2, b2)
    
    b_p = dnn.back_propagate(train_X ,train_Y, f_p, learning_rate)

    W2 = b_p["W2"]
    b2 = b_p["b2"]
    W1 = b_p["W1"]
    b1 = b_p["b1"]

    if(i % 10 == 0):
        cost = dnn.compute_cost(f_p["A2"], train_Y)
        print("Step  {} Cost is {:.4}".format(i+1,cost))

test_data = mnist.test.images.T
test_Y = mnist.test.labels.T  

pred = dnn.predict(test_data,W2,b2,W1,b1)    
target = np.argmax(test_Y , axis = 0)

accuracy_score(pred,target)