
import numpy as np
import pandas as pd
import datetime
import scipy


def min_max_scaler(x):
    """Min-max scaler function
    
    Arguments:
        x {[list, numpy.ndarray]} -- Array to scale
    
    Returns:
        [list, numpy.ndarray] -- Scaled array
    """
    return (x - min(x)) / (max(x) - min(x))


def random_timeseries(n_series = 5, n_years = 5):
    """Function that generated a dataframe with correlated timeseries
    
    Keyword Arguments:
        n_series {int} -- Number of series (default: {5})
        n_years {int} -- Number of years (default: {5})
    
    Returns:
        pandas.DataFrame -- Dataframe with timeseries
    """
    # Constant
    days_in_year = 365.25
    
    # Generate time array
    t = np.arange(n_years * days_in_year)
    
    # Generate dataframe with datecolumn and t column
    today = datetime.datetime.now().date()
    start = today - datetime.timedelta(days = len(t) - 1)
    
    df = pd.DataFrame({
        'ds': pd.date_range(start=start, end=today, freq='D'), 
        't': t
    })
    
    # Generate base series
    y_1 = np.sin(t / days_in_year * np.pi) * 3 # Yearly period seasonality
    y_1 += np.sin(t / days_in_year * 12 * np.pi) * 2 # Monthly period seasonality
    y_1 += np.sin(t / 7 * np.pi) # Weekly period seasonality
    y_1 += t / days_in_year # Trend
    y_1 += scipy.random.normal(size=len(t))
    
    # Construct df
    df['y_1'] = y_1
    
    # Add random walks to base series
    for i in range(n_series):
        
        rw = scipy.random.normal(scale=5)
        mu = scipy.random.normal(scale=0.01)
        y_i = y_1.copy()
        
        for j in range(len(t)):

            rw += scipy.random.normal(loc=mu, scale=0.2)
            y_i[j] += rw
            
        df['y_{}'.format(i)] = y_i
        
    # Scale numerical series
    df.iloc[:,1:] = df.iloc[:, 1:].apply(min_max_scaler)
    
    return df
    