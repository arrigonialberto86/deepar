import pandas as pd
import unittest
from deepar.dataset.time_series import TimeSeries


class TestRecurrentTs(unittest.TestCase):
    def setUp(self):

        self.data_to_pad = pd.DataFrame({'feature_1': [i for i in range(6)],
                                         'feature_2': [i for i in range(6)],
                                         'target': [i for i in range(6)]})

        self.input_data = pd.DataFrame({'feature_1': [i for i in range(100)],
                                        'feature_2': [i for i in range(100)],
                                        'target': [i for i in range(100)],
                                        'category': [str(int(i//10 + 1)) for i in range(100)]})

        self.data_to_pad_with_categorical = pd.DataFrame({'one_hot_yes': [1, 1, 1, 1, 1, 1],
                                                          'feature_2': [i for i in range(6)],
                                                          'one_hot_no': [0, 0, 0, 0, 0, 0],
                                                          'target': [i for i in range(6)]})
        self.data_to_pad_with_multiple_categorical = pd.DataFrame({'one_hot_yes': [1, 1, 1, 1, 1, 1],
                                                                   'feature_2': [i for i in range(6)],
                                                                   'one_hot_no': [0, 0, 0, 0, 0, 0],
                                                                   'other_no': [0, 0, 0, 0, 0, 0],
                                                                   'other_yes': [1, 1, 1, 1, 1, 1],
                                                                   'target': [i for i in range(6)]})

    def test_len_padding(self):
        rec_instance = TimeSeries(pandas_df=self.data_to_pad)
        results = rec_instance._pad_ts(pandas_df=self.data_to_pad,
                                       desired_len=10)
        self.assertEqual(results.shape[0], 10)

    def test_zero_len_padding(self):
        rec_instance = TimeSeries(pandas_df=self.data_to_pad)
        results = rec_instance._pad_ts(pandas_df=self.data_to_pad,
                                       desired_len=6)  # len is the same as the original time series
        self.assertEqual(results.shape[0], 6)

    def test_next_batch_production(self):
        rec_ts = TimeSeries(self.input_data)
        X_feature_space, y_target = rec_ts.next_batch(batch_size=4, n_steps=10)
        self.assertEqual(len(X_feature_space), 4)
        self.assertEqual(len(X_feature_space[0]), 10)
        self.assertEqual(len(X_feature_space[0][0]), 2)
        self.assertEqual(X_feature_space[3][0][0], y_target[3][0][0])

    def test_padding_with_one_hot(self):
        rec_ts = TimeSeries(pandas_df=self.data_to_pad_with_categorical,
                            one_hot_root_list=['one_hot'])
        results = rec_ts._pad_ts(pandas_df=self.data_to_pad_with_categorical,
                                 desired_len=10)

        self.assertEqual(results.shape[0], 10)
        self.assertEqual(results.one_hot_yes.values[0], 1)
        self.assertEqual(results.one_hot_no.values[0], 0)

    def test_padding_with_one_hot_multiple(self):
        rec_ts = TimeSeries(pandas_df=self.data_to_pad_with_categorical,
                            one_hot_root_list=['one_hot', 'other'])

        results = rec_ts._pad_ts(pandas_df=self.data_to_pad_with_multiple_categorical,
                                 desired_len=10)

        self.assertEqual(results.shape[0], 10)
        self.assertEqual(results.one_hot_yes.values[0], 1)
        self.assertEqual(results.one_hot_no.values[0], 0)
        self.assertEqual(results.other_yes.values[0], 1)
        self.assertEqual(results.other_no.values[0], 0)

    def test_next_batch_covariates(self):
        """
        Feature space is supplied in input if target_only is False (no need to lag y dataset)
        """
        rec_ts = TimeSeries(self.input_data)
        X_feature_space, y_target = rec_ts.next_batch(batch_size=1, n_steps=10)
        self.assertEqual(len(X_feature_space), 1)
        self.assertEqual(len(X_feature_space[0][0]), 2)

    def test_sample_ts(self):
        """
        When the length of the pandas df is longer than required length the function should sample
        from the time series and return that sample
        """
        rec_instance = TimeSeries(pandas_df=self.data_to_pad)
        results = rec_instance._sample_ts(pandas_df=self.data_to_pad,
                                          desired_len=3)
        self.assertEqual(results.shape[0], 3)

if __name__ == '__main__':
    unittest.main()
