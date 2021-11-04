import numpy as np
from functools import lru_cache
import inspect


import bilby
from scipy.special import gammaln


class GaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function, kwargs=None):
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: float, None
            The standard deviation of the noise
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        self._x = x
        self._y = y

        self.N = len(self.x)
        self.function = function
        self.kwargs = kwargs
        self._noise_log_likelihood = None

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))
        self.sigma = sigma

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        if kwargs is None:
            self._kwargs = dict()
        else:
            self._kwargs = kwargs

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def sigma(self):
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
        if sigma is None:
            self.parameters['sigma'] = None

    @property
    def residual(self):
        return self.y - self.function(self.x, **self.parameters, **self.kwargs)

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._log_l(res=self.y, sigma=self.sigma)
        return self._noise_log_likelihood

    def log_likelihood(self):
        return self._log_l(res=self.residual, sigma=self.sigma)

    @staticmethod
    def _log_l(res, sigma):
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)


class GaussianLikelihoodUniformXErrors(bilby.Likelihood):
    def __init__(self, x, y, sigma, bin_size, function, kwargs=None):
        """
        A general Gaussian likelihood with uniform errors in x- the parameters are inferred from the
        arguments of function. Takes into account the X errors with a Uniform likelihood between the
        bin high and bin low values. Note that the prior for the true x values must be uniform in this range!

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: float
            The standard deviation of the noise.
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.xerr = bin_size
        self.N = len(self.x)
        self.function = function
        self._noise_log_likelihood = 0
        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))

        self.function_keys = self.parameters.keys()
        self.parameters['sigma'] = None

    def noise_log_likelihood(self):
        sigma = self.parameters.get('sigma', self.sigma)
        res = self.y - 0.
        log_a = np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)
        log_b = np.sum(-np.log(self.xerr))
        self._noise_log_likelihood = log_a + log_b
        return self._noise_log_likelihood

    def log_likelihood_a(self):
        sigma = self.parameters.get('sigma', self.sigma)
        res = self.y - self.function(self.x, **self.parameters, **self.kwargs)
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)

    def log_likelihood_b(self):
        log_b = -np.log(self.xerr)
        return np.nan_to_num(np.sum(log_b))

    def log_likelihood(self):
        log_a = self.log_likelihood_a()
        log_b = self.log_likelihood_b()
        return log_a + log_b


class GaussianLikelihoodQuadratureNoise(bilby.Likelihood):
    def __init__(self, x, y, sigma_i, function, kwargs=None):
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma_i: float
            The standard deviation of the noise. This is part of the full noise.
            The sigma used in the likelihood is sigma = sqrt(sigma_i^2 + sigma^2)
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        self.x = x
        self.y = y
        self.sigma_i = sigma_i
        self.N = len(self.x)
        self.function = function
        self._noise_log_likelihood = 0
        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))

        self.function_keys = self.parameters.keys()
        self.parameters['sigma'] = None

    def noise_log_likelihood(self):
        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i ** 2. + sigma_s ** 2.)
        res = self.y - 0.
        self._noise_log_likelihood = np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)
        return self._noise_log_likelihood

    def log_likelihood(self):
        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i ** 2. + sigma_s ** 2.)
        res = self.y - self.function(self.x, **self.parameters, **self.kwargs)
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)


class GaussianLikelihoodQuadratureNoiseNonDetections(bilby.Likelihood):
    def __init__(self, x, y, sigma_i, function, kwargs=None, upperlimit_kwargs=None):
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function. Takes into account non-detections with a Uniform likelihood for those points

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma_i: float
            The standard deviation of the noise. This is part of the full noise.
            The sigma used in the likelihood is sigma = sqrt(sigma_i^2 + sigma^2)
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        self.x = x
        self.y = y
        self.sigma_i = sigma_i
        self.N = len(self.x)
        self.function = function
        self._noise_log_likelihood = 0
        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs
        self.upperlimit_kwargs = upperlimit_kwargs

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))

        self.function_keys = self.parameters.keys()
        self.parameters['sigma'] = None

    def noise_log_likelihood(self):
        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i ** 2. + sigma_s ** 2.)
        res = self.y - 0.
        self._noise_log_likelihood = np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)
        return self._noise_log_likelihood

    def log_likelihood_a(self):
        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i ** 2. + sigma_s ** 2.)
        res = self.y - self.function(self.x, **self.parameters, **self.kwargs)
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)

    def log_likelihood_b(self):
        flux = self.function(**self.parameters, **self.upperlimit_kwargs)
        upperlimits = self.upperlimit_kwargs['flux']
        log_l = np.ones(len(flux))
        mask = flux >= upperlimits
        log_l[~mask] = -np.log(upperlimits[~mask])
        log_l[mask] = np.nan_to_num(-np.inf)
        return np.nan_to_num(np.sum(log_l))

    def log_likelihood(self):
        log_a = self.log_likelihood_a()
        log_b = self.log_likelihood_b()
        return log_a + log_b


class GRBGaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function, kwargs=None):
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: float
            The standard deviation of the noise
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(self.x)
        self.function = function
        self._noise_log_likelihood = 0
        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))

        self.function_keys = self.parameters.keys()
        if self.sigma is None:
            self.parameters['sigma'] = None

    def noise_log_likelihood(self):
        sigma = self.parameters.get('sigma', self.sigma)
        res = self.y - 0.
        self._noise_log_likelihood = np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)
        return self._noise_log_likelihood

    def log_likelihood(self):
        sigma = self.parameters.get('sigma', self.sigma)
        res = self.y - self.function(self.x, **self.parameters, **self.kwargs)
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)


class PoissonLikelihood(bilby.Likelihood):
    def __init__(self, time, counts, function, integrated_rate_function=True, dt=None, kwargs=None):
        """
        Parameters
        ----------
        x, y: array_like
            The data to analyse
        background_rate: array_like
            The background rate
        function:
            The python function to fit to the data
        integrated_rate_function: bool
            Whether the function returns an integrated rate over the `dt` in the bins.
            This should be true if you multiply the rate with `dt` in the function and false if the function
            returns a rate per unit time.
        dt: (array_like, float, None)
            Array of each bin size or single value if all bins are of the same size. If None, assume that
            `dt` is constant and calculate it from the first two elements of `time`.
        """
        self.time = time
        self.counts = counts
        self.function = function
        self._noise_log_likelihood = 0
        if kwargs is None:
            self.kwargs = dict()
        else:
            self.kwargs = kwargs
        self.integrated_rate_function = integrated_rate_function
        self.dt = dt
        parameters = bilby.core.utils.introspection.infer_parameters_from_function(func=function)
        self.parameters = dict.fromkeys(parameters)
        super(PoissonLikelihood, self).__init__(parameters=dict())

    @property
    def dt(self):
        return self.kwargs['dt']

    @dt.setter
    def dt(self, dt):
        if dt is None:
            dt = self.time[1] - self.time[0]
        self.kwargs['dt'] = dt

    def noise_log_likelihood(self):
        background_rate = self.parameters['background_rate'] * self.dt
        return self._poisson_log_likelihood(rate=background_rate)

    def log_likelihood(self):
        rate = self.function(self.time, **self.parameters, **self.kwargs) + \
               self.parameters['background_rate']
        if not self.integrated_rate_function:
            rate *= self.dt
        return self._poisson_log_likelihood(rate=rate)

    def _poisson_log_likelihood(self, rate):
        return np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
