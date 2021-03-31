import numpy as np
import inspect

import bilby
from scipy.special import gammaln

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

    def noise_log_likelihood(self):
        # model = self.function(self.x, **self.parameters)
        res = self.y - 0.
        log_l = np.sum(- (res / self.sigma) ** 2 / 2 -
                       np.log(2 * np.pi * self.sigma ** 2) / 2)
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        if self.kwargs != None:
            model = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            model = self.function(self.x, **self.parameters)

        res = self.y - model
        log_l = np.sum(- (res / self.sigma) ** 2 / 2 -
                       np.log(2 * np.pi * self.sigma ** 2) / 2)
        return log_l


class GRBGaussianLikelihood(bilby.Likelihood):
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

    def noise_log_likelihood(self):
        # model = self.function(self.x, **self.parameters)
        res = self.y - 0.
        log_l = np.sum(- (res / self.sigma) ** 2 / 2 -
                       np.log(2 * np.pi * self.sigma ** 2) / 2)
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        if self.kwargs != None:
            model = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            model = self.function(self.x, **self.parameters)
        res = self.y - model
        log_l = np.sum(- (res / self.sigma) ** 2 / 2 -
                       np.log(2 * np.pi * self.sigma ** 2) / 2)
        return log_l


class PoissonLikelihood_afterglow(bilby.Likelihood):
    def __init__(self, time, counts, factor, dt, background_rate, function, kwargs):
        """
        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: array_like
            The standard deviation of the noise
        function:
            The python function to fit to the data
        """
        self.time = time
        self.counts = counts
        self.factor = factor
        self.function = function
        self.dt = dt
        self.background_rate = background_rate
        self.kwargs = kwargs
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        super(PoissonLikelihood_afterglow, self).__init__(parameters=dict())

    def noise_log_likelihood(self):
        background_rate = self.background_rate * self.dt
        rate = 0 + background_rate
        log_l = np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        if self.kwargs != None:
            flux = self.function(self.x, **self.parameters, **self.kwargs)
        else:
            flux = self.function(self.x, **self.parameters)

        background_rate = self.background_rate * self.dt
        N = self.factor * flux
        rate = N * self.dt + background_rate
        logl = np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
        return logl
