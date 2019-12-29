
import numpy as np
import pandas as pd
import datetime


days_in_year = 365.25
days_in_month = days_in_year / 12


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


def random_walk(n, start=0, sigma=0.2):

    x = np.ndarray(n)
    x[0] = start
    
    for i in range(1, n):
        x[i] = x[i-1] + np.random.normal(scale=sigma)

    return x


def _random_seasonality_base(time, period):

    x = np.zeros(len(time))

    period_sin = 2 * np.pi * np.random.choice([1, 2, 3, 4])
    period_cos = 2 * np.pi * np.random.choice([1, 2, 3, 4])

    x += np.sin(time / period * period_sin) * \
         np.random.uniform(low=0.5, high=1.5)
    x += np.cos(time / period * period_cos) * \
         np.random.uniform(low=0.5, high=1.5)

    return x


def random_seasonality(time, yearly=True, monthly=False, weekly=True):

    x = np.zeros(len(time))

    # Yearly period seasonality
    if yearly:
        x += _random_seasonality_base(time, days_in_year)

    # Monthly period seasonality
    if monthly:
        x += _random_seasonality_base(time, days_in_month) * 0.6

    # Weekly period seasonality
    if weekly:
        x += _random_seasonality_base(time, 7) * 0.3

    return x


def random_timeseries(n_series = 5, n_years = 5, equal_trend=False,
                      equal_start=False, equal_seasonality=False,
                      equal_error=False):
    """Function that generated a dataframe with correlated timeseries
    
    Keyword Arguments:
        n_series {int} -- Number of series (default: {5})
        n_years {int} -- Number of years (default: {5})
        equal_trend = boolean indicating if trend should be equal for all series
        equal_start = boolean indicating if starting point should be equal for all series
        equal_seasonality = boolean indicating if seasonality should be equal for all series
        equal_error = boolean indicating if the variance of the error terms should be equal
    
    Returns:
        pandas.DataFrame -- Dataframe with timeseries
    """
    
    # Generate time array
    time = np.arange(n_years * days_in_year)
    
    # Generate dataframe with datecolumn and t column
    today = datetime.datetime.now().date()
    start = today - datetime.timedelta(days = len(time) - 1)
    
    df = pd.DataFrame({
        'ds': pd.date_range(start=start, end=today, freq='D'), 
        't': time
    })

    # Construct starting point and trend
    start = 0 
    trend = random_walk(n=len(time))
    seasonality = random_seasonality
    error_sd = 0.5
    
    # Add random walks to base series
    for i in range(n_series):

        if not equal_start:
            start = np.random.uniform(low=-3, high=3)
        
        if not equal_trend:
            trend = random_walk(n=len(time))

        if not equal_seasonality:
            seasonality = random_seasonality(time)

        if not equal_error:
            error_sd = np.random.uniform(high=2)
    
        # Generate base series
        y = start + seasonality + trend
        y += np.random.normal(size=len(time), scale=error_sd) # Error term
        
        # Construct df
        df['y_{}'.format(i)] = y
    
    return df


if __name__=='__main__':

    df = random_timeseries()
    print(df)
