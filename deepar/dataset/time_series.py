from deepar.dataset import Dataset
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('deepar')


class MockTs(Dataset):
    """
    This class generates 'mock' time series data of the form (y = t * np.sin(t/6) / 3 +np.sin(t*2))
    Created mainly for showcase/testing purpose
    """
    def __init__(self, t_min=0, t_max=30, resolution=.1):
        self.t_min = t_min
        self.t_max = t_max
        self.resolution = resolution
        self.data = True

    @staticmethod
    def _time_series(t):
        return t * np.sin(t/6) / 3 + np.sin(t*2)

    def next_batch(self, batch_size, n_steps):
        """
        Generate next batch (x, y), generate y by lagging x (1 step)
        """
        t0 = np.random.rand(batch_size, 1) * (self.t_max - self.t_min - n_steps * self.resolution)
        Ts = t0 + np.arange(0., n_steps + 1) * self.resolution
        ys = self._time_series(Ts)
        return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

    def create_mock_ts(self, n_inputs, n_steps):
        """
        Create a mock time series (whole)
        :return: a Numpy array
        """
        t_instance = np.linspace(12.2, 12.2 + self.resolution * (n_steps + 1), n_steps + 1)
        return self._time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))


class TimeSeries(Dataset):
    def __init__(self, pandas_df, one_hot_root_list=None, grouping_variable='category'):
        super().__init__()
        self.data = pandas_df
        self.one_hot_root_list = one_hot_root_list
        # self.target_only = target_only
        self.grouping_variable = grouping_variable
        if self.data is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')

    def _one_hot_padding(self, pandas_df, padding_df):
        """
        Util padding function
        :param padding_df:
        :param one_hot_root_list:
        :return:
        """
        for one_hot_root in self.one_hot_root_list:
            one_hot_columns = [i for i in pandas_df.columns   # select columns equal to 1
                               if i.startswith(one_hot_root) and pandas_df[i].values[0] == 1]
            for col in one_hot_columns:
                padding_df[col] = 1
        return padding_df

    def _pad_ts(self, pandas_df, desired_len, padding_val=0):
        """
        Add padding int to the time series
        :param pandas_df:
        :param desired_len: (int)
        :param padding_val: (int)
        :return: X (feature_space), y
        """
        pad_length = desired_len - pandas_df.shape[0]
        padding_df = pd.concat([pd.DataFrame({col: padding_val for col in pandas_df.columns},
                                             index=[i for i in range(pad_length)])])

        if self.one_hot_root_list:
            padding_df = self._one_hot_padding(pandas_df, padding_df)

        return pd.concat([padding_df, pandas_df]).reset_index(drop=True)

    @staticmethod
    def _sample_ts(pandas_df, desired_len):
        """

        :param pandas_df: input pandas df with 'target' columns e features
        :param desired_len: desired sample length (number of rows)
        :param padding_val: default is 0
        :param initial_obs: how many observations to skip at the beginning
        :return: a pandas df (sample)
        """
        if pandas_df.shape[0] < desired_len:
            raise ValueError('Desired sample length is greater than df row len')
        if pandas_df.shape[0] == desired_len:
            return pandas_df

        start_index = np.random.choice([i for i in range(0, pandas_df.shape[0] - desired_len + 1)])
        return pandas_df.iloc[start_index: start_index+desired_len, ]

    def next_batch(self, batch_size, n_steps,
                   target_var='target', verbose=False,
                   padding_value=0):
        """
        :param batch_size: how many time series to be sampled in this batch (int)
        :param n_steps: how many RNN cells (int)
        :param target_var: (str)
        :param verbose: (boolean)
        :param padding_value: (float)
        :return: X (feature space), y
        """

        # Select n_batch time series
        groups_list = self.data[self.grouping_variable].unique()
        np.random.shuffle(groups_list)
        selected_groups = groups_list[:batch_size]
        input_data = self.data[self.data[self.grouping_variable].isin(set(selected_groups))]

        # Initial padding for each selected time series to reach n_steps
        sampled = []
        for cat, cat_data in input_data.groupby(self.grouping_variable):
                if cat_data.shape[0] < n_steps:
                    sampled_cat_data = self._pad_ts(pandas_df=cat_data,
                                                    desired_len=n_steps,
                                                    padding_val=padding_value)
                else:
                    sampled_cat_data = self._sample_ts(pandas_df=cat_data,
                                                       desired_len=n_steps)
                sampled.append(sampled_cat_data)
                if verbose:
                    logger.debug('Sampled data for {}'.format(cat))
                    logger.debug(sampled_cat_data)
        rnn_output = pd.concat(sampled).drop(columns=self.grouping_variable).reset_index(drop=True)

        return rnn_output.drop(target_var, 1).as_matrix().reshape(batch_size, n_steps, -1), \
               rnn_output[target_var].as_matrix().reshape(batch_size, n_steps, 1)


