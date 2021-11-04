import numpy as np


import bilby
from scipy.special import gammaln


class _RedbackLikelihood(bilby.Likelihood):

    def __init__(self, x, y, function, kwargs=None):
        self._x = x
        self._y = y
        self.function = function
        self.kwargs = kwargs

        parameters = bilby.core.utils.introspection.infer_parameters_from_function(func=function)
        super().__init__(parameters=dict.fromkeys(parameters))

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
    def N(self):
        return len(self.x)


class GaussianLikelihood(_RedbackLikelihood):
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

        self._noise_log_likelihood = None
        super().__init__(x=x, y=y, function=function, kwargs=kwargs)
        self.sigma = sigma

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
            self._noise_log_likelihood = self._gaussian_log_likelihood(res=self.y, sigma=self.sigma)
        return self._noise_log_likelihood

    def log_likelihood(self):
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.sigma)

    @staticmethod
    def _gaussian_log_likelihood(res, sigma):
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)


class GaussianLikelihoodUniformXErrors(GaussianLikelihood):
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
        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs)
        self.xerr = bin_size * np.ones(self.N)

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            log_x = self.log_likelihood_x()
            log_y = self._gaussian_log_likelihood(res=self.y, sigma=self.sigma)
            self._noise_log_likelihood = log_x + log_y
        return self._noise_log_likelihood

    def log_likelihood_x(self):
        return -np.nan_to_num(np.sum(np.log(self.xerr)))

    def log_likelihood_y(self):
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.sigma)

    def log_likelihood(self):
        return self.log_likelihood_x() + self.log_likelihood_y()


class GaussianLikelihoodQuadratureNoise(GaussianLikelihood):
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
        self.sigma_i = sigma_i
        # These lines of code infer the parameters from the provided function
        super().__init__(x=x, y=y, sigma=None, function=function, kwargs=kwargs)

    @property
    def full_sigma(self):
        return np.sqrt(self.sigma_i ** 2. + self.sigma ** 2.)

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._gaussian_log_likelihood(res=self.y, sigma=self.full_sigma)
        return self._noise_log_likelihood

    def log_likelihood(self):
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma)


class GaussianLikelihoodQuadratureNoiseNonDetections(GaussianLikelihoodQuadratureNoise):
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
        super().__init__(x=x, y=y, sigma_i=sigma_i, function=function, kwargs=kwargs)
        self.upperlimit_kwargs = upperlimit_kwargs

    @property
    def upperlimit_flux(self):
        return self.upperlimit_kwargs['flux']

    def log_likelihood_y(self):
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma)

    def log_likelihood_upper_limit(self):
        flux = self.function(self.x, **self.parameters, **self.upperlimit_kwargs)
        log_l = -np.ones(len(flux)) * np.log(self.upperlimit_flux)
        log_l[flux >= self.upperlimit_flux] = np.nan_to_num(-np.inf)
        return np.nan_to_num(np.sum(log_l))

    def log_likelihood(self):
        return self.log_likelihood_y() + self.log_likelihood_upper_limit()


class GRBGaussianLikelihood(GaussianLikelihood):

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
        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs)


class PoissonLikelihood(_RedbackLikelihood):
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
        dt: array_like, float, None
            Array of each bin size or single value if all bins are of the same size. If None, assume that
            `dt` is constant and calculate it from the first two elements of `time`.
        """
        super(PoissonLikelihood, self).__init__(x=time, y=counts, function=function, kwargs=kwargs)
        self.integrated_rate_function = integrated_rate_function
        self.dt = dt
        self.parameters['background_rate'] = 0

    @property
    def time(self):
        return self.x

    @property
    def counts(self):
        return self.y

    @property
    def dt(self):
        return self.kwargs['dt']

    @dt.setter
    def dt(self, dt):
        if dt is None:
            dt = self.time[1] - self.time[0]
        self.kwargs['dt'] = dt

    @property
    def background_rate(self):
        return self.parameters['background_rate']

    def noise_log_likelihood(self):
        return self._poisson_log_likelihood(rate=self.background_rate * self.dt)

    def log_likelihood(self):
        rate = self.function(self.time, **self.parameters, **self.kwargs) + self.background_rate
        if not self.integrated_rate_function:
            rate *= self.dt
        return self._poisson_log_likelihood(rate=rate)

    def _poisson_log_likelihood(self, rate):
        return np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
