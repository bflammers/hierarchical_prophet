
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import scipy

from src.utils import MinMaxScaler, validate_df, stack_series


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


def changepoints(t, n_changepoints, changepoint_range=0.8):
    """Make required arrays for defining chagepoints needed for trend
    
    Arguments:
        t {numpy.ndarray} -- Array of scaled timestamps
        n_changepoints {int} -- Number of changepoints
    
    Keyword Arguments:
        changepoint_range {float} -- Defined proportion of time that 
        changepoints should be included (default: {0.8})
    
    Returns:
        tuple -- tuple of numpy.ndarrays s and A 
    """

    # Array of changepoints in time dimension
    s = np.linspace(np.min(t), changepoint_range, n_changepoints + 1)[1:]

    # If any points fall into the future (larger than 1), make future changepoints
    if any(t > 1):

        # Calculate number of future changepoints
        n_future_changepoints = int(np.ceil(n_changepoints * (max(t) - 1)))

        # Determine timepoints of future changepoints
        s_future = np.linspace(1, max(t), n_future_changepoints + 1)[1:]

        # Combine with past changepoints
        s = np.append(s, s_future)

    # A is a boolean matrix specifying which observation time stamps (vector t) --> rows
    # have surpasses which changepoint time stamps (vector s) --> columns
    # * 1 casts the boolean to integers
    A = (t[:, None] > s) * 1

    return s, A 


class HierarchicalProphet:

    def __init__(self,
        trend=True, trend_hierarchical=True, seasonality=True, seasonality_hierarchical=False,
        eps_hierarchical=False, n_changepoints=50, changepoints_range=0.8, 
        fourier_params=[{'period': 365.25, 'n_fourier': 8}, {'period': 7, 'n_fourier': 4}],
        full_posterior=False, maxeval=5000):

        # Model parameters
        self.trend = trend
        self.trend_hierarchical = trend_hierarchical
        self.seasonality = seasonality
        self.seasonality_hierarchical = seasonality_hierarchical 
        self.eps_hierarchical = eps_hierarchical
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoints_range
        self.fourier_components = fourier_params
        self.full_posterior = full_posterior
        self.maxeval=maxeval

        # Intialize attributes
        self.df_columns = None
        self.scalers = {} # Dictionary of MinMaxScaler objects
        self.model = None # PyMC3 model object
        self.n_series = None # Number of series, determined in fitting step
        self.pe = None # Point estimates

    def fit(self, X):

        # Validate dataframe, set columns attribute
        validate_df(X)
        self.df_columns = X.columns

        # Fit and transform scalers
        df_scaled = self._fit_transform_scalers(X)

        # Transform dataframe into arrays with required format
        t, idx, y = stack_series(df_scaled)

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

        # Add likelihood function
        self.likelihood(t, y, mu_t, idx)
        
        # Fitting step
        if self.full_posterior:
            raise NotImplementedError('Full posterior not supported yet')
        else:

            with self.model:
                self.pe = pm.find_MAP(maxeval=self.maxeval)

    def _fit_transform_scalers(self, df):

        df_scaled = df.copy()

        for col in df:
            
            # Construct scaler
            scaler = MinMaxScaler()
            
            # Fit scaler and return scaled column
            df_scaled[col] = scaler.fit_transform(df[col])
            
            # Store column scaler in dict
            self.scalers[col] = scaler

        return df_scaled

    def _construct_df(self, t, y, hyper=False):

        if hyper:

            # Construct dataframe with all columns in the right order
            # df = pd.DataFrame({'t': t, 'y': y})
            ys = np.empty((len(t), self.n_series))

            for i, col in enumerate(set(self.df_columns) - {'t'}):
                
                # Load scaler from dict
                scaler = self.scalers[col]
                
                # Rescale column
                ys[:, i] = scaler.inverse_transform(y)[:, 0]

            df = pd.DataFrame({
                't': self.scalers['t'].inverse_transform(t), 
                'y': np.median(ys, axis=1)
                })

        else:

            # Construct dataframe with all columns in the right order
            df = pd.DataFrame(y, columns=[x for x in self.df_columns if x != 't'])
            df['t'] = t
            df = df[self.df_columns]

            for col in df:
                
                # Load scaler from dict
                scaler = self.scalers[col]
                
                # Rescale column
                df[col] = scaler.inverse_transform(df[col])

        return df

    def likelihood(self, t, y, mu_t, idx):

        if self.eps_hierarchical:

            with self.model:

                # Prior
                sigma_beta = pm.HalfCauchy('sigma_beta', .5, testval=1)

                # Likelihood
                sigma = pm.HalfCauchy('sigma', sigma_beta, shape=self.n_series)
                Y_obs = pm.Normal('Y_obs', mu=mu_t, sd=sigma[idx], observed=y)

        else:

            with self.model:

                # Likelihood
                sigma = pm.HalfCauchy('sigma', .5, testval=1, shape=self.n_series)
                Y_obs = pm.Normal('Y_obs', mu=mu_t, sd=sigma[idx], observed=y)


    def add_trend(self, t, idx, s, A):

        if self.trend_hierarchical:

            with self.model:
                
                # Hyper priors, RVs
                k_mu = pm.Normal('k_mu', mu=0., sd=10) # sd=10
                k_sigma = pm.HalfCauchy('k_sigma', testval=1, beta=5) # beta=5
                
                m_mu = pm.Normal('m_mu', mu=0., sd=10) # sd=10
                m_sigma = pm.HalfCauchy('m_sigma', testval=1, beta=5) # beta=5

                delta_b = pm.HalfCauchy('delta_b', testval=0.1, beta=0.1) # beta=0.1

        else:

            # No RVs, fixed parameters
            k_mu = 0
            k_sigma = 5
            
            m_mu = 0
            m_sigma = 10

            delta_b = 0.1

        with self.model:
            
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

    def sample_trend(self, t, hyper=False):

        t = self.scalers['t'].transform(t)

        s, A = changepoints(t, self.n_changepoints, self.changepoint_range)

        if hyper:

            assert self.trend_hierarchical, 'hyper=True but trend not hierarchical'

            m = np.random.normal(self.pe['m_mu'], self.pe['m_sigma'])
            k = np.random.normal(self.pe['k_mu'], self.pe['k_sigma'])
            delta = np.random.laplace(0, self.pe['delta_b'], (A.shape[1], 1))

        else:

            delta = self.pe['delta'].T

            if any(t > 1):

                # MAD is the MLE for beta parameter of laplace distribution
                beta = scipy.stats.median_absolute_deviation(self.pe['delta'], axis=1)

                n_future_changepoints = len(s) - self.n_changepoints
                future_delta = np.random.laplace(0, beta, (n_future_changepoints, self.n_series))
                delta = np.r_[delta, future_delta]

            # Fix dimensions
            m = np.repeat(self.pe['m'][None, :], t.shape[0], axis=0)
            k = np.repeat(self.pe['k'][None, :], t.shape[0], axis=0)

        g_t = m
        g_t += (k + A @ delta) * t[:, None]
        g_t += A @ (-s[:, None] * delta)

        return t, g_t

    def add_seasonality(self, idx, F, suffix):

        if self.seasonality_hierarchical:

            with self.model:
            
                # Hyper priors, RVs
                beta_mu = pm.Normal('beta_mu_{}'.format(suffix), mu=0., sd=3) # Prophet: sd=10
                beta_sigma = pm.HalfCauchy('beta_sigma_{}'.format(suffix), testval=1, beta=2) # Prophet: beta=5

        else:

            # No RVs, fixed parameters
            beta_mu = 0
            beta_sigma = 10

        with self.model:

            # Priors
            beta = pm.Normal('beta_{}'.format(suffix), beta_mu, beta_sigma, shape = (F.shape[1], self.n_series))
            
            # Seasonality
            s_t = (F * beta[:, idx].T).sum(axis=-1)

        return s_t

    def sample_seasonality(self, t, hyper=False):

        if hyper:
            assert self.seasonality_hierarchical, 'hyper=True but seasonality not hierarchical'

        # Scale t
        t = self.scalers['t'].transform(t)

        # Construct s array
        s_t_n_cols = 1 if hyper else self.n_series
        s_t = np.zeros(shape=(t.shape[0], s_t_n_cols))

        for fc in self.fourier_components:

            p = str(fc['period'])
            F = fourier_series(t, fc['period_scaled'], fc['n_fourier'])

            if hyper:
                beta_mu = self.pe['beta_mu_{}'.format(p)]
                beta_sigma = self.pe['beta_sigma_{}'.format(p)]
                beta = np.random.normal(beta_mu, beta_sigma, size=(F.shape[1], 1))
            else:
                beta = self.pe['beta_{}'.format(p)]

            s_t += F @ beta 

        return t, s_t

    def sample_eps(self, t, hyper=False):

        if hyper:

            assert self.eps_hierarchical, 'hyper=True but eps not hierarchical'
            e_t = np.random.normal(0, self.pe['sigma_beta'], size = (len(t), 1))

        else:

            e_t = np.random.normal(0, self.pe['sigma'], size = (len(t), self.n_series))

        return t, e_t

    def sample(self, t, white_noise=True, hyper=False):

        _, g_t = self.sample_trend(t, hyper=hyper)
        t, s_t = self.sample_seasonality(t, hyper=hyper)

        y_t = g_t + s_t 

        if white_noise:
            _, e_t = self.sample_eps(t, hyper=hyper)
            y_t += e_t

        return self._construct_df(t, y_t, hyper=hyper)

    def empirical_quantiles(self, t, quantiles=[0.05, 0.5, 0.95], n_samples=1000, 
        white_noise=True, hyper=False):

        samples = np.ndarray((len(t), self.n_series, n_samples))

        for i in range(n_samples):

            samples[:, :, i] = self.sample(t, white_noise, hyper).drop(columns='t')

        quantiles = np.quantile(samples, quantiles, axis = 2)

        return quantiles


if __name__=='__main__':

    import utils

    cols = ['t', 'y_0', 'y_1']

    df = pd.DataFrame(
        {cols[0]: [1,2,3], cols[1]: [0,1,2], cols[2]: [4, 5, 6]}, 
        columns=cols
        )

    print(stack_series(df))
    exit()

    df = random.random_timeseries()

    hp = HierarchicalProphet()
    hp.fit(df)
