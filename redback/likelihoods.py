import numpy as np
from typing import Any, Union

import bilby
from scipy.special import gammaln


class _RedbackLikelihood(bilby.Likelihood):

    def __init__(self, x: np.ndarray, y: np.ndarray, function: callable, kwargs: dict = None) -> None:
        """

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param function: The model/function that we want to fit.
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: Union[dict, None]
        """
        self.x = x
        self.y = y
        self.function = function
        self.kwargs = kwargs

        parameters = bilby.core.utils.introspection.infer_parameters_from_function(func=function)
        super().__init__(parameters=dict.fromkeys(parameters))

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs: dict) -> None:
        if kwargs is None:
            self._kwargs = dict()
        else:
            self._kwargs = kwargs

    @property
    def n(self) -> int:
        """
        :return: Length of the x/y-values
        :rtype: int
        """
        return len(self.x)


class GaussianLikelihood(_RedbackLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None) -> None:
        """A general Gaussian likelihood - the parameters are inferred from the arguments of function.

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma: The standard deviation of the noise.
        :type sigma: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """

        self._noise_log_likelihood = None
        super().__init__(x=x, y=y, function=function, kwargs=kwargs)
        self.sigma = sigma

    @property
    def sigma(self) -> Union[float, None, np.ndarray]:
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma: Union[float, None, np.ndarray]) -> None:
        self._sigma = sigma
        if sigma is None:
            self.parameters['sigma'] = None

    @property
    def residual(self) -> np.ndarray:
        return self.y - self.function(self.x, **self.parameters, **self.kwargs)

    def noise_log_likelihood(self) -> float:
        """
        :return: The noise log-likelihood, i.e. the log-likelihood assuming the signal is just noise.
        :rtype: float
        """
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._gaussian_log_likelihood(res=self.y, sigma=self.sigma)
        return self._noise_log_likelihood

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.sigma)

    @staticmethod
    def _gaussian_log_likelihood(res: np.ndarray, sigma: Union[float, np.ndarray]) -> Any:
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)


class GaussianLikelihoodUniformXErrors(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, None, np.ndarray],
            bin_size: Union[float, None, np.ndarray], function: callable, kwargs: dict = None) -> None:
        """A general Gaussian likelihood with uniform errors in x- the parameters are inferred from the
        arguments of function. Takes into account the X errors with a Uniform likelihood between the
        bin high and bin low values. Note that the prior for the true x values must be uniform in this range!

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma: The standard deviation of the noise.
        :type sigma: Union[float, None, np.ndarray]
        :param bin_size: The bin sizes.
        :type bin_size: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """

        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs)
        self.xerr = bin_size * np.ones(self.n)

    def noise_log_likelihood(self) -> float:
        """
        :return: The noise log-likelihood, i.e. the log-likelihood assuming the signal is just noise.
        :rtype: float
        """
        if self._noise_log_likelihood is None:
            log_x = self.log_likelihood_x()
            log_y = self._gaussian_log_likelihood(res=self.y, sigma=self.sigma)
            self._noise_log_likelihood = log_x + log_y
        return self._noise_log_likelihood

    def log_likelihood_x(self) -> float:
        """
        :return: The log-likelihood due to x-errors.
        :rtype: float
        """
        return -np.nan_to_num(np.sum(np.log(self.xerr)))

    def log_likelihood_y(self) -> float:
        """
        :return: The log-likelihood due to y-errors.
        :rtype: float
        """
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.sigma)

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        return self.log_likelihood_x() + self.log_likelihood_y()


class GaussianLikelihoodQuadratureNoise(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma_i: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None) -> None:
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function

        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma_i: The standard deviation of the noise. This is part of the full noise.
                        The sigma used in the likelihood is sigma = sqrt(sigma_i^2 + sigma^2)
        :type sigma_i: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """
        self.sigma_i = sigma_i
        # These lines of code infer the parameters from the provided function
        super().__init__(x=x, y=y, sigma=None, function=function, kwargs=kwargs)

    @property
    def full_sigma(self) -> Union[float, np.ndarray]:
        """
        :return: The standard deviation of the full noise
        :rtype: Union[float, np.ndarray]
        """
        return np.sqrt(self.sigma_i ** 2. + self.sigma ** 2.)

    def noise_log_likelihood(self) -> float:
        """
        :return: The noise log-likelihood, i.e. the log-likelihood assuming the signal is just noise.
        :rtype: float
        """
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._gaussian_log_likelihood(res=self.y, sigma=self.full_sigma)
        return self._noise_log_likelihood

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma)


class GaussianLikelihoodQuadratureNoiseNonDetections(GaussianLikelihoodQuadratureNoise):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma_i: Union[float, np.ndarray], function: callable,
            kwargs: dict = None, upperlimit_kwargs: dict = None) -> None:
        """A general Gaussian likelihood - the parameters are inferred from the
        arguments of function. Takes into account non-detections with a Uniform likelihood for those points

        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma_i: The standard deviation of the noise. This is part of the full noise.
                        The sigma used in the likelihood is sigma = sqrt(sigma_i^2 + sigma^2)
        :type sigma_i: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """
        super().__init__(x=x, y=y, sigma_i=sigma_i, function=function, kwargs=kwargs)
        self.upperlimit_kwargs = upperlimit_kwargs

    @property
    def upperlimit_flux(self) -> float:
        """
        :return: The upper limit of the flux.
        :rtype: float
        """
        return self.upperlimit_kwargs['flux']

    def log_likelihood_y(self) -> float:
        """
        :return: The log-likelihood due to y-errors.
        :rtype: float
        """
        return self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma)

    def log_likelihood_upper_limit(self) -> Any:
        """
        :return: The log-likelihood due to the upper limit.
        :rtype: float
        """
        flux = self.function(self.x, **self.parameters, **self.upperlimit_kwargs)
        log_l = -np.ones(len(flux)) * np.log(self.upperlimit_flux)
        log_l[flux >= self.upperlimit_flux] = np.nan_to_num(-np.inf)
        return np.nan_to_num(np.sum(log_l))

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        return self.log_likelihood_y() + self.log_likelihood_upper_limit()


class GRBGaussianLikelihood(GaussianLikelihood):

    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, np.ndarray],
            function: callable, kwargs: dict = None) -> None:
        """A general Gaussian likelihood - the parameters are inferred from the
        arguments of function.

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma: The standard deviation of the noise.
        :type sigma: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """
        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs)


class PoissonLikelihood(_RedbackLikelihood):
    def __init__(
            self, time: np.ndarray, counts: np.ndarray, function: callable, integrated_rate_function: bool = True,
            dt: Union[float, np.ndarray] = None, kwargs: dict = None) -> None:
        """
        :param time: The time values.
        :type time: np.ndarray
        :param counts: The number of counts for the time value.
        :type counts: np.ndarray
        :param function: The python function to fit to the data.
        :type function: callable
        :param integrated_rate_function:
            Whether the function returns an integrated rate over the `dt` in the bins.
            This should be true if you multiply the rate with `dt` in the function and false if the function
            returns a rate per unit time. (Default value = True)
        :type integrated_rate_function: bool
        :param dt:
            Array of each bin size or single value if all bins are of the same size. If None, assume that
            `dt` is constant and calculate it from the first two elements of `time`.
        :type dt: Union[float, None, np.ndarray]
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """
        super(PoissonLikelihood, self).__init__(x=time, y=counts, function=function, kwargs=kwargs)
        self.integrated_rate_function = integrated_rate_function
        self.dt = dt
        self.parameters['background_rate'] = 0

    @property
    def time(self) -> np.ndarray:
        return self.x

    @property
    def counts(self) -> np.ndarray:
        return self.y

    @property
    def dt(self) -> Union[float, np.ndarray]:
        return self.kwargs['dt']

    @dt.setter
    def dt(self, dt: [float, None, np.ndarray]) -> None:
        if dt is None:
            dt = self.time[1] - self.time[0]
        self.kwargs['dt'] = dt

    @property
    def background_rate(self) -> float:
        return self.parameters['background_rate']

    def noise_log_likelihood(self) -> float:
        """
        :return: The noise log-likelihood, i.e. the log-likelihood assuming the signal is just noise.
        :rtype: float
        """
        return self._poisson_log_likelihood(rate=self.background_rate * self.dt)

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        rate = self.function(self.time, **self.parameters, **self.kwargs) + self.background_rate
        if not self.integrated_rate_function:
            rate *= self.dt
        return self._poisson_log_likelihood(rate=rate)

    def _poisson_log_likelihood(self, rate: Union[float, np.ndarray]) -> Any:
        return np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
