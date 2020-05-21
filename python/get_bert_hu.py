# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:13:49 2020

@author: huzhen
"""

from keras.layers import Input,Embedding,Dropout,Add,Lambda,Dense
from layernormalization_hu import LayerNormalization
from positionEmbedding_hu import PositionEmbedding
from attention_hu import Attention
from feedforward_hu import FeedForward
from keras.models import Model


def get_bert(weights_path,with_pool=True):
    vocab_token_size = 21128
    vocab_segment_size = 2
    embedding_size = 768
    intermediate_size = 3072
    dropout_rate = 0.1
    num_heads = 12
    hidden_layers = 12
    head_size = embedding_size // num_heads
    num_position_embedding = 512
    input_token = Input(shape=(None,),dtype='int32',name='Input-Token')
    input_segment = Input(shape=(None,),dtype='int32',name='Input-Segment')
    
    embedding_token = Embedding(vocab_token_size,embedding_size,name='Embedding-Token')(input_token)
    embedding_segment = Embedding(vocab_segment_size,embedding_size,name='Embedding-Segment')(input_segment)
    embedding_token_segment = Add(name='Embedding-Token-Segment')([embedding_token,embedding_segment])
    
    embedding_position = PositionEmbedding(num_position_embedding,embedding_size,name='Embedding-Position')(embedding_token_segment)
    
    embedding_norm = LayerNormalization(name='Embedding-Norm')(embedding_position)
    embedding_dropout = Dropout(dropout_rate,name='Embedding-Dropout')(embedding_norm)
    
    """
    transformer
    """
    last = embedding_dropout
    
    for i in range(hidden_layers):
        att = Attention(num_heads,head_size,name='Transformer-{}-MultiHeadSelfAttention'.format(i))([last,last,last])
        att_dropout = Dropout(dropout_rate,name='Transformer-{}-MultiHeadSelfAttention-Dropout'.format(i))(att)
        add = Add(name='Transformer-{}-MultiHeadSelfAttention-Add'.format(i))([last,att_dropout])
        norm = LayerNormalization(name='Transformer-{}-MultiHeadSelfAttention-Norm'.format(i))(add)
        ffnn = FeedForward(intermediate_size,name='Transformer-{}-FeedForward'.format(i))(norm)
        dropout = Dropout(dropout_rate,name='Transformer-{}-FeedForward-Dropout'.format(i))(ffnn)
        add = Add(name='Transformer-{}-FeedForward-Add'.format(i))([norm,dropout])
        norm = LayerNormalization(name='Transformer-{}-FeedForward-Norm'.format(i))(add)
        last = norm
    if with_pool:
        pooler = Lambda(lambda x:x[:,0],name='Pooler')(last)
        pooler = Dense(embedding_size,activation='tanh',name='Pooler-Dense')(pooler)
        model = Model([input_token,input_segment],pooler,name='bert_model')
    else:
        model = Model([input_token,input_segment],last)
    model.load_weights(weights_path)
    return model


from bert4keras.models import build_transformer_model

bert_model = build_transformer_model(config_path = r'chinese_L-12_H-768_A-12/bert_config.json',
                                     checkpoint_path=r'chinese_L-12_H-768_A-12/bert_model.ckpt',
                                     with_pool = True)

bert_model.save_weights(r'bert_pooler_hu.h5')
