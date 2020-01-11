
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


def validate_df(df):

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


def changepoints(t, n_changepoints, changepoint_range=0.8):
    """Make required arrays for defining chagepoints needed for trend
    
    Arguments:
        t {numpy.ndarray} -- Array of timestamps
        n_changepoints {int} -- Number of changepoints
    
    Keyword Arguments:
        changepoint_range {float} -- Defined proportion of time that 
        changepoints should be included (default: {0.8})
    
    Returns:
        tuple -- tuple of numpy.ndarrays s and A 
    """

    # Array of changepoints in time dimension
    s = np.linspace(0, changepoint_range * np.max(t), n_changepoints + 1)[1:]

    # A is a boolean matrix specifying which observation time stamps (vector t) --> rows
    # have surpasses which changepoint time stamps (vector s) --> columns
    # * 1 casts the boolean to integers
    A = (t[:, None] > s) * 1

    return s, A 


class HierarchicalProphet:

    def __init__(self, scale=True, 
        trend=True, seasonality=True, n_changepoints=20, changepoints_range=0.8, 
        fourier_params=[{'period': 365.25, 'n_fourier': 10}, {'period': 7, 'n_fourier': 4}],
        full_posterior=False, maxeval=5000):

        # Model parameters
        self.scale = scale
        self.trend = trend
        self.seasonality = seasonality
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoints_range
        self.fourier_components = fourier_params
        self.full_posterior = full_posterior
        self.maxeval=maxeval

        # Intialize attributes
        self.scalers = {} # Dictionary of MinMaxScaler objects
        self.model = None # PyMC3 model object
        self.n_series = None # Number of series, determined in fitting step
        self.pe = None # Point estimates

    def fit(self, X):

        # Validate dataframe
        validate_df(X)

        # Fit and transform scalers
        df_scaled = self._fit_transform_scalers(X)

        # Transform dataframe into arrays with required format
        ds, t, idx, y = stack_series(df_scaled)

        # Set number of series
        self.n_series = max(idx) + 1

        # Generate a PyMC3 Model context
        self.model = pm.Model()

        # Construct mu array
        mu_t = np.zeros(shape=t.shape)

        if self.trend:

            # Determine points in time for changepoint
            s, A = changepoints(t, self.n_changepoints, self.changepoint_range)

            # Add trend component
            mu_t += self.add_trend(t, idx, s, A)

        if self.seasonality:

            for fc in self.fourier_components:
                
                # Scale fourier period with time scaler object
                fc['period_scaled'] = self.scalers['t'].transform(fc['period'])

                # Generate array with fourier base components
                fc['F'] = fourier_series(t, fc['period_scaled'], fc['n_fourier'])

                mu_t += self.add_seasonality(idx, fc['F'], str(fc['period']))

        with self.model:

            # Likelihood
            sigma = pm.HalfCauchy('sigma', .5, testval=1, shape=self.n_series)
            Y_obs = pm.Normal('Y_obs', mu=mu_t, sd=sigma[idx], observed=y)

        # Fitting step
        if self.full_posterior:
            raise NotImplementedError('Full posterior not supported yet')
        else:

            with self.model:
                self.pe = pm.find_MAP(maxeval=self.maxeval)


    def _fit_transform_scalers(self, df):

        for col in df:

            # Do not scale date column
            if col != 'ds':
            
                # Construct scaler
                scaler = MinMaxScaler()
                
                # Fit scaler and return scaled column
                df[col] = scaler.fit_transform(df[col])
                
                # Store column scaler in dict
                self.scalers[col] = scaler

        return df

    def add_trend(self, t, idx, s, A):

        with self.model:
            
            # Hyper priors
            k_mu = pm.Normal('k_mu', mu=0., sd=10) # sd=10
            k_sigma = pm.HalfCauchy('k_sigma', testval=1, beta=5) # beta=5
            
            m_mu = pm.Normal('m_mu', mu=0., sd=10) # sd=10
            m_sigma = pm.HalfCauchy('m_sigma', testval=1, beta=5) # beta=5

            delta_b = pm.HalfCauchy('delta_b', testval=0.1, beta=0.1) # beta=0.1
            
            # Priors
            k = pm.Normal('k', k_mu, k_sigma, shape=self.n_series)
            m = pm.Normal('m', m_mu, m_sigma, shape=self.n_series)

            delta = pm.Laplace('delta', 0, delta_b, shape = (self.n_series, self.n_changepoints))
                    
            # Starting point (offset)
            g_t = m[idx]
            
            # Linear trend w/ changepoints
            gamma = -s * delta[idx, :]
            g_t += (k[idx] + (A * delta[idx, :]).sum(axis=1)) * t + (A * gamma).sum(axis=1)

        return g_t

    def add_seasonality(self, idx, F, suffix):

        with self.model:
        
            # Hyper priors
            beta_mu = pm.Normal('beta_mu_{}'.format(suffix), mu=0., sd=3) # Prophet: sd=10 / sd=3
            beta_sigma = pm.HalfCauchy('beta_sigma_{}'.format(suffix), testval=1, beta=2) # Prophet: beta=5 / beta=2
            
            # Priors
            beta = pm.Normal('beta_{}'.format(suffix), beta_mu, beta_sigma, shape = (F.shape[1], self.n_series))
            
            # Seasonality
            s_t = (F * beta[:, idx].T).sum(axis=-1)

        return s_t


if __name__=='__main__':

    import utils

    cols = ['ds', 't', 'y_0', 'y_1']

    df = pd.DataFrame(
        {cols[0]: [1,2,3], cols[1]: [0,1,2], cols[2]: [4, 5, 6], cols[3]: [7, 8, 9]}, 
        columns=cols
        )

    df = utils.random_timeseries()

    hp = HierarchicalProphet()
    hp.fit(df)
