import tensorflow as tf


def gaussian_likelihood(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(0.5*tf.math.log(sigma) + 0.5*tf.math.truediv(tf.math.square(y_true - y_pred), sigma)) + 1e-6 + 6
    return gaussian_loss
