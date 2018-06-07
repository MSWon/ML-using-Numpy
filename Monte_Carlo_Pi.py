# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:07:15 2018

@author: jbk48
"""

import math
import random


def In_Circle(x,y):
    return(y - math.sqrt(1-math.pow(x,2)))


def MonteCarlo_pi(n):
    
    in_circle = []
    
    for i in range(n):
        
        sample_x = random.uniform(0,1)
        sample_y = random.uniform(0,1)
        
        if(In_Circle(sample_x,sample_y) <= 0):
            in_circle.append("incircle")
            
    answer = len(in_circle)/n
    pi = 4 * answer
    return(pi)
    
MonteCarlo_pi(1000000)  ## 3.142844
math.pi  ## 3.141592653589793
