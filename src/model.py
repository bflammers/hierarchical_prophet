
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt


def stack_series(df):
    """Stack multiple time series into single array with corresponding t, ds, and idx arrays
    
    Arguments:
        df {pandas.DataFrame} -- DataFrame with columns [t, ds, y_{0, 1, 2, 3, 4, ...}]
    
    Returns:
        tuple -- tuple of numpy.ndarray --> ds, t, idx, y
    """

    # Validity checks
    col_names = df.columns
    assert 'ds' in col_names, 'ds not a column'
    assert 't' in col_names, 't not a column'

    ts_names = [x for x in col_names if x not in {'ds', 't'}]
    ts_pre = np.array([x[:2] for x in ts_names], dtype=str)
    ts_suf = np.array([x[2:] for x in ts_names], dtype=int)
    assert all(ts_pre == 'y_'), 'time series columns should be pre-fixed with y_'
    assert ts_suf[0] == 0, 'time series columns suffixes should start at 0'
    assert all(np.diff(np.array(ts_suf)) == 1), \
        'series columns should be named y_{0,1,2,3,4,...}'
    
    # Stack
    df_stacked = df.set_index(['ds', 't']).stack().reset_index()
    
    # Extract relevant columns as numpy arrays
    ds = df_stacked['ds'].values
    t = df_stacked['t'].values
    y = df_stacked[0].values

    # Extract series idxs
    col_names = df_stacked['level_2'].astype(str)
    series_num = col_names.str.split('_', expand=True)[1]
    idx = series_num.astype(int).values
    
    return ds, t, idx, y 


def fourier_series(t, p, n):
    """Make an array of fourier series with a specified period
    
    Arguments:
        t {numpy.array} -- time array
        p {float} -- float specifying period of fourier series
        n {int} -- number of fourier series
    
    Returns:
        numpy.ndarray -- array of fourier series
    """
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


if __name__=='__main__':

    cols = ['ds', 't', 'y_0', 'y_1']

    df = pd.DataFrame(
        {cols[0]: [1,2,3], cols[1]: [0,1,2], cols[2]: [4, 5, 6], cols[3]: [7, 8, 9]}, 
        columns=cols
        )

    print(df)
    ds, t, idx, y = stack_series(df)

    print(y)
