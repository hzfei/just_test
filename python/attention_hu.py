# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:00:42 2020

@author: huzhen
"""

from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


class Attention(Layer):
    def __init__(self,num_head,size_per_head,is_right=False,**kwargs):
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.output_size = self.num_head * self.size_per_head
        self.is_right = is_right
        
        super(Attention,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.WQ = self.add_weight(
                                name = 'WQ',
                                shape = (input_shape[0][-1],self.output_size),
                                initializer = 'glorot_uniform',
                                trainable = True
                                )
        
        self.WQ_bias = self.add_weight(
                                name = 'WQ_bias',
                                shape = (self.output_size,),
                                initializer = 'glorot_uniform',
                                trainable = True
                                )
        self.WK = self.add_weight(
                                name = 'WK',
                                shape = (input_shape[1][-1],self.output_size),
                                initializer = 'glorot_uniform',
                                trainable = True)
        
        self.WK_bias = self.add_weight(
                                name = 'WK_bias',
                                shape = (self.output_size,),
                                initializer = 'glorot_uniform',
                                trainable = True
                                )
        
        self.WV = self.add_weight(
                                name = 'WV',
                                shape = (input_shape[2][-1],self.output_size),
                                initializer = 'glorot_uniform',
                                trainable = True)
        
        self.WV_bias = self.add_weight(
                                name = 'WV_bias',
                                shape = (self.output_size,),
                                initializer = 'glorot_uniform',
                                trainable = True
                                )
        
        self.WO = self.add_weight(
                                name = 'WO',
                                shape = (self.output_size,self.output_size),
                                initializer = 'glorot_uniform',
                                trainable = True
                
                                  )
        
        self.WO_bias = self.add_weight(
                                name = 'WO_bias',
                                shape = (self.output_size,),
                                initializer = 'glorot_uniform',
                                trainable = True
                                )
        
        self.built = True
    
    def Mask(self,inputs,lens,mode='add'):
        if lens is not None:
            return inputs
        mask = K.one_hot(lens[:,0],K.shape(inputs)[1])
        mask = K.cumsum(mask,axis=-1)
        for i in range(K.ndim(inputs) - K.ndim(lens)):
            mask = K.expand_dims(mask,axis=-1)
        #(-1,n|m,1...)
        if mode == 'add':
            inputs = inputs - mask * 1e12
        elif mode == 'mul':
            inputs = inputs * (1 - mask)
        return inputs
    
    def call(self,inputs):
        Q,K_,V = inputs[:3]
        v_len = None
        q_len = None
        if len(inputs) >= 4:
            v_len = inputs[3]
        if len(inputs) >= 5:
            q_len = inputs[4]
        
        
        Q = K.dot(Q,self.WQ) + self.WQ_bias
        Q = K.reshape(Q,(-1,K.shape(Q)[1],self.num_head,self.size_per_head))
        Q = K.permute_dimensions(Q,(0,2,1,3))
        
        K_ = K.dot(K_,self.WK) + self.WK_bias
        K_ = K.reshape(K_,(-1,K.shape(K_)[1],self.num_head,self.size_per_head))
        K_ = K.permute_dimensions(K_,(0,2,1,3))
        
        
        
        V = K.dot(V,self.WV) + self.WV_bias
        V = K.reshape(V,(-1,K.shape(V)[1],self.num_head,self.size_per_head))
        V = K.permute_dimensions(V,(0,2,1,3))
        #A = K.batch_dot(Q,K_,axes=(3,3))#shape = (-1,head,m,n)
        A = tf.einsum('bhmd,bhnd->bhmn',Q,K_)
        if v_len is not None:
            A = K.permute_dimensions(A,[0,3,2,1])#(-1,n,m,nb_head)
            A = self.Mask(A,v_len,'add')
            A = K.permute_dimensions(A,[0,3,2,1])
        
        if self.is_right:
            ones = K.ones_like(A[:1,:1])
            
            mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e10
            A = A - mask
        A = K.softmax(A / self.size_per_head ** 0.5)
        
        #O = K.batch_dot(A,V,axes=(3,2))
        O = tf.einsum('bhmn,bhnd->bhmd',A,V)
        O = K.permute_dimensions(O,(0,2,1,3))
        O = K.reshape(O,(-1,K.shape(O)[1],self.num_head*self.size_per_head))
        
        O = K.dot(O,self.WO) + self.WO_bias
        if q_len is not None:
            O = self.Mask(O,q_len,'mul')
        return O
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0][0],input_shape[0][1],self.output_size)

