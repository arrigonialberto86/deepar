from keras import backend as K
import tensorflow as tf
import numpy as np


def set_seed_and_reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def clear_keras_session():
    K.clear_session()