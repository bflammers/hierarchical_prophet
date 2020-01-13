
import pandas as pd
import numpy as np

def stack_series(df):
    """Stack multiple time series into single array with corresponding t, ds, and idx arrays
    
    Arguments:
        df {pandas.DataFrame} -- DataFrame with columns [t, y_{0, 1, 2, 3, 4, ...}]
    
    Returns:
        tuple -- tuple of numpy.ndarray --> t, idx, y
    """
    
    # Stack
    df_stacked = df.set_index('t').stack().reset_index()
    
    # Extract relevant columns as numpy arrays
    t = df_stacked['t'].values
    y = df_stacked[0].values

    # Extract series idxs
    col_names = df_stacked['level_1'].astype(str)
    series_num = col_names.str.split('_', expand=True)[1]
    idx = series_num.astype(int).values
    
    return t, idx, y 


def validate_df(df):

    # Validity checks
    col_names = df.columns
    assert 'ds' not in col_names, 'ds should not be included'
    assert 't' in col_names, 't should be included'

    ts_names = [x for x in col_names if x != 't']
    ts_pre = np.array([x[:2] for x in ts_names], dtype=str)
    ts_suf = np.array([x[2:] for x in ts_names], dtype=int)
    assert all(ts_pre == 'y_'), 'time series columns should be pre-fixed with y_'
    assert ts_suf[0] == 0, 'time series columns suffixes should start at 0'
    assert all(np.diff(np.array(ts_suf)) == 1), \
        'series columns should be named y_{0,1,2,3,4,...}'


class MinMaxScaler:

    def __init__(self, centering=True, scaling=True):

        self.centering=centering
        self.scaling=scaling

        self.loc = None
        self.scale = None

    def fit(self, x):

        self.loc = min(x) if self.centering else 0
        self.scale = max(x) - min(x) if self.centering else 1

    def transform(self, x):

        return (x - self.loc) / self.scale

    def fit_transform(self, x):

        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):

        return x * self.scale + self.loc