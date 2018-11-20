from tensorflow.random import normal
from keras import backend as K
from keras.initializers import glorot_normal
from keras.layers import Layer


class GaussianLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = None, None, None, None

    def build(self, input_shape):
        n_weight_rows = input_shape[2]
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim, ),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim, ),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        # Extract a random sample from a normal distribution
        return normal(mean=output_mu, stddev=output_sig_pos, shape=(1,))

    def compute_output_shape(self, input_shape):
        """
        The assumption is the output time series is always one-dimensional
        """
        return 1, input_shape[1], 1

