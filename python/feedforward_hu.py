# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:57:47 2020

@author: huzhen
"""

from keras.layers import Layer,Dense
from keras import activations
class FeedForward(Layer):
    def __init__(self,units,**kwargs):
        self.units = units
        super(FeedForward,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.dense1 = Dense(self.units,activation=activations.get('relu'))
        self.dense2 = Dense(input_shape[-1])
        self.built = True
        
    def call(self,inputs):
        x = inputs
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def compute_output_shape(self,input_shape):
        return input_shape
