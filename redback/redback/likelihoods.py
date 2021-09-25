import numpy as np
import inspect

import bilby
from scipy.special import gammaln, logsumexp

class GaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function, kwargs):
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
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        if self.kwargs != None:
            model = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            model = self.function(self.x, **self.parameters)

        sigma = self.parameters.get('sigma', self.sigma)

        res = self.y - model
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        return log_l

class GaussianLikelihood_with_uniform_x_errors(bilby.Likelihood):
    def __init__(self, x, y, sigma, bin_size, function, kwargs):
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
        log_a = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        log_b = np.sum(-np.log(self.xerr))
        log_l = log_a + log_b
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood_a(self):
        if self.kwargs != None:
            model = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            model = self.function(self.x, **self.parameters)

        sigma = self.parameters.get('sigma', self.sigma)

        res = self.y - model
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        return log_l

    def log_likelihood_b(self):
        log_b = -np.log(self.xerr)
        return np.nan_to_num(np.sum(log_b))

    def log_likelihood(self):
        log_a = self.log_likelihood_a()
        log_b = self.log_likelihood_b()
        log_l = log_a + log_b
        return log_l

class GaussianLikelihood_quadrature_noise(bilby.Likelihood):
    def __init__(self, x, y, sigma_i, function, kwargs):
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
        self.kwargs = kwargs

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))

        self.function_keys = self.parameters.keys()
        self.parameters['sigma'] = None

    def noise_log_likelihood(self):
        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i**2. + sigma_s**2.)
        res = self.y - 0.
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        if self.kwargs != None:
            model = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            model = self.function(self.x, **self.parameters)

        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i**2. + sigma_s**2.)

        res = self.y - model
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        return log_l

class GaussianLikelihood_quadrature_noise_non_detections(bilby.Likelihood):
    def __init__(self, x, y, sigma_i, function, kwargs, upperlimit_kwargs):
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
        sigma = np.sqrt(self.sigma_i**2. + sigma_s**2.)
        res = self.y - 0.
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood_a(self):
        if self.kwargs != None:
            model = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            model = self.function(self.x, **self.parameters)

        sigma_s = self.parameters['sigma']
        sigma = np.sqrt(self.sigma_i**2. + sigma_s**2.)

        res = self.y - model
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        return log_l

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
        log_l = log_a + log_b
        return log_l

class GRBGaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function, **kwargs):
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
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        model = self.function(self.x, **self.parameters, **self.kwargs)
        sigma = self.parameters.get('sigma', self.sigma)

        res = self.y - model
        log_l = np.sum(- (res / sigma) ** 2 / 2 -
                       np.log(2 * np.pi * sigma ** 2) / 2)
        return log_l


class PoissonLikelihood(bilby.Likelihood):
    def __init__(self, time, counts, function, dt=None, **kwargs):
        """
        Parameters
        ----------
        x, y: array_like
            The data to analyse
        background_rate: array_like
            The background rate
        function:
            The python function to fit to the data
        """
        self.time = time
        self.counts = counts
        self.function = function
        self.kwargs = kwargs
        if dt is None:
            self.dt = self.time[1] - self.time[0]
        else:
            self.dt = dt
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        super(PoissonLikelihood, self).__init__(parameters=dict())

    def noise_log_likelihood(self):
        background_rate = self.parameters['background_rate'] * self.dt
        return self._log_likelihood(rate=background_rate)

    def log_likelihood(self):
        rate = (self.function(self.time, **self.parameters, **self.kwargs)
                + self.parameters['background_rate']) * self.dt
        return self._log_likelihood(rate=rate)

    def _log_likelihood(self, rate):
        return np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
