# -*- coding: utf-8 -*-
"""
Created on Sun May 27 18:36:53 2018

@author: Big Data Guru
"""

import numpy as np

class DNN():
    
    def __init__(self, output_node, hidden_node):

        self.output_node = output_node
        self.hidden_node = hidden_node        

    def Sigmoid(self, x):
        return(1/(1+np.exp(-x)))
        
    def Softmax(self, x):
        return(np.exp(x) / np.sum(np.exp(x), axis = 0 , keepdims = True))
        
    def compute_cost(self, A2, Y):
    
        m = Y.shape[1] # number of example
        n = Y.shape[0] # number of output node
        # Compute the cross-entropy cost
        logprobs = (1/n)*np.sum(np.log(A2)*Y + np.log(1-A2)*(1-Y),axis=0,keepdims = True)
        cost = (-1/m)*np.sum(logprobs)
        
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))    
        return cost
        
    def initialize(self, x):
        
        number_data = x.shape[1]
        input_node = x.shape[0]
    
        W2 = np.random.randn(self.output_node,self.hidden_node)
        b2 = np.random.randn(self.output_node)
        b2 = b2.reshape(-1,1)
        
        W1 = np.random.randn(self.hidden_node,input_node)
        b1 = np.random.randn(self.hidden_node)
        b1 = b1.reshape(-1,1)
        
        dic = {"W2":W2,"b2":b2,"W1":W1,"b1":b1}
        
        return(dic)
    
    def foward_propagate(self, x, W1, b1, W2, b2):
        
        Z1 = np.dot(W1, x) + b1
        A1 = self.Sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.Softmax(Z2)
        dic = {"W2":W2,"b2":b2,"W1":W1,"b1":b1,
               "Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    
        return(dic)
        
    def back_propagate(self, x ,Y, f_p, learning_rate = 0.001):
               
        m = x.shape[0]
        W2 = f_p["W2"]
        b2 = f_p["b2"]
        W1 = f_p["W1"]
        b1 = f_p["b1"]
    
        Z2 = f_p["Z2"]
        A2 = f_p["A2"]
        Z1 = f_p["Z1"]
        A1 = f_p["A1"]
        
        dZ2 = A2 - Y
        dW2 = (1/m)*np.dot(dZ2,A1.T)
        db2 = (1/m)*np.sum(dZ2, axis = 1 ,keepdims = True)
            
        dZ1 = np.dot(W2.T,dZ2) * self.Sigmoid(Z1)* (1-self.Sigmoid(Z1))
        dW1 = (1/m)*np.dot(dZ1,x.T) 
        db1 = (1/m)*np.sum(dZ1, axis = 1 ,keepdims = True)
    
        ## update    
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
        
        dic = {"W2":W2,"b2":b2,"W1":W1,"b1":b1}
        return(dic)
    
    def predict(self, test_data,W2,b2,W1,b1):
        
        Z1 = np.dot(W1,test_data) + b1
        A1 = self.Sigmoid(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = self.Softmax(Z2)    
        
        return(np.argmax(A2, axis = 0))