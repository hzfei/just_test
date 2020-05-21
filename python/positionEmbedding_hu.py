from keras.layers import Layer
from keras import backend as K

class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer='glorot_uniform'
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embeddings[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)
        if self.merge_mode != 'add':
            pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

