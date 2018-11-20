import tensorflow as tf
from deepar.model import NNModel
from deepar.model.layers import GaussianLayer
from keras.layers import Input, Dense, Input
from keras.models import Model
from keras.layers import LSTM
import logging

logger = logging.getLogger('deepar')


class DeepAR(NNModel):
    def __init__(self, ts_obj, steps_per_epoch=50, epochs=100, loss='mse', optimizer='adam', with_nn_structure=None):
        self.ts_obj = ts_obj
        self.inputs, self.z_sample = None, None
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.keras_model = None
        if with_nn_structure:
            self.nn_structure = with_nn_structure
        else:
            self.nn_structure = DeepAR.basic_structure

    @staticmethod
    def basic_structure():
        """
        This is the method that needs to be patched when changing NN structure
        :return: inputs, z_sample ()
        """
        input_shape = (20, 1)
        inputs = Input(shape=input_shape)
        x = LSTM(10, return_sequences=True, dropout=0.2)(inputs)
        x = Dense(10, activation='relu')(x)
        z_sample = GaussianLayer(1, name='main_output')(x)
        return inputs, z_sample, input_shape

    def instantiate_and_fit(self, verbose=False):
        inputs, z_sample, input_shape = self.nn_structure()
        model = Model(inputs, z_sample)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.fit_generator(ts_generator(self.ts_obj,
                                         1,
                                         input_shape[0]),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs)
        if verbose:
            logger.debug('Model was successfully trained')
        self.keras_model = model

    @property
    def model(self):
        return self.keras_model


def ts_generator(ts_obj, batch_shape, n_steps):
    """
    This is a util generator function for Keras
    :param ts_obj: a Dataset child class object that implements the 'next_batch' method
    :param batch_shape:
    :return:
    """
    while 1:
        batch = ts_obj.next_batch(batch_shape, n_steps)
        yield batch[0], batch[1]
