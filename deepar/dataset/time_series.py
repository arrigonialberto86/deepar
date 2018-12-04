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

    @property
    def mock_ts(self):
        """
        Return the data used for training (ranging from self.t_min and self.t_max, with resolution self.resolution)
        :return: a Numpy array
        """
        t_list = [self.t_min]
        results = [self._time_series(t_list[0])]
        while t_list[-1] < self.t_max:
            t_list.append(t_list[-1] + self.resolution)
            results.append(self._time_series(t_list[-1]))
        return results

    def generate_test_data(self, n_steps):
        """
        Generate test data outside of the training set (t > self.t_max)
        :param n_steps:
        :return:
        """
        t_list = [self.t_max + self.resolution]
        results = [self._time_series(t_list[0])]
        for i in range(1, n_steps):
            t_list.append(t_list[-1] + self.resolution)
            results.append(self._time_series(t_list[-1]))
        return results


class TimeSeries(Dataset):
    def __init__(self, pandas_df, one_hot_root_list=None, grouping_variable='category', scaler=None):
        super().__init__()
        self.data = pandas_df
        self.one_hot_root_list = one_hot_root_list
        self.grouping_variable = grouping_variable
        if self.data is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')
        self.scaler = scaler

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

        if self.scaler:
            batch_scaler = self.scaler()
            n_rows = rnn_output.shape[0]
            # Scaling will have to be extended to handle multiple variables!
            rnn_output['feature_1'] = rnn_output.feature_1.astype('float')
            rnn_output[target_var] = rnn_output[target_var].astype('float')

            rnn_output['feature_1'] = batch_scaler.fit_transform(rnn_output.feature_1.values.reshape(n_rows, 1)).reshape(n_rows)
            rnn_output[target_var] = batch_scaler.fit_transform(rnn_output[target_var].values.reshape(n_rows, 1)).reshape(n_rows)

        return rnn_output.drop(target_var, 1).as_matrix().reshape(batch_size, n_steps, -1), \
               rnn_output[target_var].as_matrix().reshape(batch_size, n_steps, 1)


