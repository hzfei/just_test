from keras.layers import Layer
from keras import backend as K

class LayerNormalization(Layer):

    def __init__(self,**kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epslion = 1e-12
        self.center = True
        self.scala = True
        
    def build(self, input_shape):
        shape = (input_shape[-1],)
        
        
            
        if self.center:
            self.beta = self.add_weight(
                                        name='beta',
                                        shape=shape,
                                        initializer='zeros',
                                        
                                        trainable=True
            )
        if self.scala:
            self.gamma = self.add_weight(
                                        name='gamma',
                                        shape=shape,
                                        initializer='ones',
                
                                        trainable = True
                                        )
            
        
        self.built = True
        
    def call(self,inputs):
        outputs = inputs
        mean = K.mean(outputs,axis=-1,keepdims=True)
        variance = K.mean(K.square(outputs-mean),axis=-1,keepdims=True)
        std = K.sqrt(variance + self.epslion)
        outputs = (outputs - mean) / std
        
        if self.scala:
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta
        return outputs





#class LayerNormalization(Layer):
#    def __init__(self,**kwargs):
#        super(LayerNormalization,self).__init__(**kwargs)
#        self.center = True
#        self.scala = True
#        self.epslion = 1e-12
#    
#    def build(self,input_shape):
#        shape = (input_shape[-1],)
#        if self.center:
#            self.beta = self.add_weight(
#                                        name='beta',
#                                        shape=shape,
#                                        initializer='zeros',
#                                        trainable=True
#                                      )
#        if self.scala:
#            self.gamma = self.add_weight(
#                                         name='gamma',
#                                         shape=shape,
#                                         initializer='ones',
#                                         trainable=True
#                                        )
#        
#        self.built = True
#        
#    
#    def call(self,inputs):
#        outputs = inputs
#        mean = K.mean(outputs,axis=-1,keepdims=True)
#        variance = K.mean(K.square(outputs-mean),axis=-1,keepdims=True)
#        std = K.sqrt(variance+self.epslion)
#        outputs = (outputs-mean)/std
#        if self.scala:
#            outputs = outputs * self.gamma
#        if self.center:
#            outputs = outputs + self.beta
#        return outputs