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


class PoissonLikelihood(bilby.Likelihood):
    def __init__(self, time, counts, function, kwargs):
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
        self.dt = kwargs['dt']
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        super(PoissonLikelihood, self).__init__(parameters=dict())

    def noise_log_likelihood(self):
        background_rate = self.parameters['bkg_rate'] * self.dt
        rate = 0 + background_rate
        log_l = np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
        self._noise_log_likelihood = log_l
        return self._noise_log_likelihood

    def log_likelihood(self):
        if self.kwargs != None:
            rate = self.function(self.time, **self.parameters, **self.kwargs)
        else:
            rate = self.function(self.time, **self.parameters)

        logl = np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
        return logl
