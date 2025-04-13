import numpy as np
from typing import Any, Union

import bilby
from scipy.special import gammaln
from redback.utils import logger
from bilby.core.prior import DeltaFunction, Constraint

class _RedbackLikelihood(bilby.Likelihood):

    def __init__(self, x: np.ndarray, y: np.ndarray, function: callable, kwargs: dict = None, priors=None,
                 fiducial_parameters=None) -> None:
        """

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param function: The model/function that we want to fit.
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: Union[dict, None]
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """
        self.x = x
        self.y = y
        self.function = function
        self.kwargs = kwargs
        self.priors = priors
        self.fiducial_parameters = fiducial_parameters

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

    @property
    def parameters_to_be_updated(self):
        if self.priors is None:
            return None
        else:
            parameters_to_be_updated = [key for key in self.priors if not isinstance(
                self.priors[key], (DeltaFunction, Constraint, float, int))]
            return parameters_to_be_updated

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keys = set(self.fiducial_parameters) - set(self.parameters_to_be_updated)
        for key in excluded_parameter_keys:
            parameter_dictionary[key] = self.fiducial_parameters[key]
        return parameter_dictionary

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def get_bounds_from_priors(self, priors):
        bounds = []
        for key in self.parameters_to_be_updated:
            bounds.append([priors[key].minimum, priors[key].maximum])
        return bounds

    def lnlike_scipy_maximize(self, parameter_list):
        self.parameters.update(self.get_parameter_dictionary_from_list(parameter_list))
        return -self.log_likelihood()

    def find_maximum_likelihood_parameters(self, iterations=5, maximization_kwargs=None, method='Nelder-Mead',
                                           break_threshold=1e-3):
        """
        Estimate the maximum likelihood

        :param iterations: Iterations to run the minimizer for before stopping. Default is 5.
        :param maximization_kwargs: Any extra keyword arguments passed to the scipy minimize function
        :param method: Minimize method to use. Default is 'Nelder-Mead'
        :param break_threshold: The threshold for the difference in log likelihood to break the loop. Default is 1e-3.
        :return: Dictionary of maximum likelihood parameters
        """
        from scipy.optimize import minimize
        parameter_bounds = self.get_bounds_from_priors(self.priors)
        if self.priors is None:
            raise ValueError("Priors must be provided to use this functionality")
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        self.parameters.update(self.fiducial_parameters)
        self.parameters["fiducial"] = 0
        updated_parameters_list = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        old_fiducial_ln_likelihood = self.log_likelihood()
        for it in range(iterations):
            logger.info(f"Optimizing fiducial parameters. Iteration : {it + 1}")
            output = minimize(
                self.lnlike_scipy_maximize,
                x0=updated_parameters_list,
                bounds=parameter_bounds,
                method=method,
                **maximization_kwargs,)
            updated_parameters_list = output['x']
            updated_parameters = self.get_parameter_dictionary_from_list(updated_parameters_list)
            self.parameters.update(updated_parameters)
            new_fiducial_ln_likelihood = self.log_likelihood_ratio()
            logger.info(f"Current lnlikelihood: {new_fiducial_ln_likelihood:.2f}")
            logger.info(f"Updated parameters: {updated_parameters}")
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < break_threshold:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood
        return updated_parameters


class GaussianLikelihood(_RedbackLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None, priors=None,
                 fiducial_parameters=None) -> None:
        """A general Gaussian likelihood - the parameters are inferred from the arguments of function.

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma: The standard deviation of the noise.
        :type sigma: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """

        self._noise_log_likelihood = None
        super().__init__(x=x, y=y, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)
        self.sigma = sigma
        if self.sigma is None:
            self.parameters['sigma'] = None


    @property
    def sigma(self) -> Union[float, None, np.ndarray]:
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma: Union[float, None, np.ndarray]) -> None:
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        elif sigma.shape == ((2, len(self.x))):
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like x.')

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
        return np.nan_to_num(self._gaussian_log_likelihood(res=self.residual, sigma=self.sigma))

    @staticmethod
    def _gaussian_log_likelihood(res: np.ndarray, sigma: Union[float, np.ndarray]) -> Any:
        return np.sum(- (res / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)

class MixtureGaussianLikelihood(GaussianLikelihood):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 sigma: Union[float, None, np.ndarray],
                 function: callable, kwargs: dict = None,
                 priors=None, fiducial_parameters=None) -> None:
        """
        Mixture Gaussian likelihood that handles outliers by modeling each data point’s likelihood
        as a weighted sum of two Gaussians. The likelihood for each datum is given by

            L_i = α * N(y_i | f(x_i), σ²) + (1 - α) * N(y_i | f(x_i), σ_out²)

        where:
          - N(y_i | f(x_i), σ²) is the Gaussian probability density evaluated at y_i with mean f(x_i)
            and variance σ².
          - α is the inlier fraction (between 0 and 1).
          - σ_out is the standard deviation for the outlier component.

        In addition, the posterior probability that a data point is an outlier is computed via

            P(outlier | r) = [(1 - α) * p_out(r)] / [α * p_in(r) + (1 - α) * p_out(r)]

        where r is the residual (y - f(x)).

        Parameters
        ----------
        x : np.ndarray
            Independent variable data.
        y : np.ndarray
            Observed dependent variable data.
        sigma : float, None, or np.ndarray
            Standard deviation for the inlier component.
        function : callable
            Model function. It should accept x as the first argument.
        kwargs : dict, optional
            Additional keyword arguments for the model function.
            sigma_out: Standard deviation of outlier data, is set to 10 times the inlier sigma if not provided.
            alpha: Inlier fraction, i.e., fraction of data points from the underlying model. Default is 0.9.
        priors : dict, optional
            Priors for the parameters (not used in this implementation).
        fiducial_parameters : dict, optional
            Starting guesses for the model parameters.
        """
        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs, priors=priors,
                        fiducial_parameters=fiducial_parameters)

        # Set default mixture parameters if not provided.
        if 'alpha' not in self.parameters:
            self.parameters['alpha'] = 0.9
        if 'sigma_out' not in self.parameters:
            if sigma is not None and isinstance(sigma, (int, float)):
                self.parameters['sigma_out'] = sigma * 10
            else:
                self.parameters['sigma_out'] = 10.0

        self._noise_log_likelihood = None
    def _mixture_gaussian_log_likelihood(self, res: np.ndarray,
                                         sigma: Union[float, np.ndarray],
                                         sigma_out: Union[float, np.ndarray],
                                         alpha: float) -> np.ndarray:
        """
        Compute the log-likelihood of the residuals under a mixture of two Gaussians in a stable
        manner using the log-sum-exp trick.

        Parameters
        ----------
        res : np.ndarray
            Array of residuals.
        sigma : float or np.ndarray
            Standard deviation for the inlier Gaussian.
        sigma_out : float or np.ndarray
            Standard deviation for the outlier Gaussian.
        alpha : float
            Inlier fraction (between 0 and 1).

        Returns
        -------
        np.ndarray
            Log-likelihood for each residual under the mixture model.
        """
        # Compute log densities directly for inlier and outlier components:
        logp_in = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (res / sigma) ** 2
        logp_out = -0.5 * np.log(2 * np.pi) - np.log(sigma_out) - 0.5 * (res / sigma_out) ** 2

        # Combine contributions using log-sum-exp:
        # log(sum_i exp(log_a_i)) can be computed as np.logaddexp(log_a, log_b) for two terms.
        term_in = np.log(alpha) + logp_in
        term_out = np.log(1 - alpha) + logp_out

        # np.logaddexp is applied element-wise:
        log_likelihood = np.logaddexp(term_in, term_out)
        return log_likelihood

    def p_in(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the inlier probability density for residuals.

        Parameters
        ----------
        r : np.ndarray
            Residuals.

        Returns
        -------
        np.ndarray
            Inlier probability density evaluated at each residual.
        """
        sigma = self.sigma
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (r / sigma) ** 2)

    def p_out(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the outlier probability density for residuals.

        Parameters
        ----------
        r : np.ndarray
            Residuals.

        Returns
        -------
        np.ndarray
            Outlier probability density evaluated at each residual.
        """
        sigma_out = self.parameters.get('sigma_out')
        return (1 / (np.sqrt(2 * np.pi) * sigma_out)) * np.exp(-0.5 * (r / sigma_out) ** 2)

    def log_likelihood(self) -> float:
        """
        Compute the total log-likelihood for the mixture model.

        For each data point, the log-likelihood is given by

            log(L_i) = log(α * N(0, σ²) + (1 - α) * N(0, σ_out²)).

        Returns
        -------
        float
            The overall log-likelihood (summed over all data points).
        """
        res = self.residual
        alpha = self.parameters.get('alpha')
        sigma_out = self.parameters.get('sigma_out')
        ll = np.sum(self._mixture_gaussian_log_likelihood(res, self.sigma, sigma_out, alpha))
        return ll

    def calculate_outlier_posteriors(self, model_prediction: np.ndarray) -> np.ndarray:
        """
        Calculate the posterior probability that each data point is an outlier.

        Given a model prediction, the residual for each point is computed as:
            r = y - model_prediction.
        Then the posterior is given by

            P(outlier | r) = [(1 - α) * p_out(r)] / [α * p_in(r) + (1 - α) * p_out(r)].

        Parameters
        ----------
        model_prediction : np.ndarray
            Model predictions for each data point.

        Returns
        -------
        np.ndarray
            An array of posterior probabilities (between 0 and 1) for each data point being an outlier.
        """
        r = self.y - model_prediction
        pin = self.p_in(r)
        pout = self.p_out(r)
        alpha = self.parameters.get('alpha')
        numerator = (1 - alpha) * pout
        denominator = alpha * pin + (1 - alpha) * pout
        posteriors = np.where(denominator > 0, numerator / denominator, 0.0)
        return posteriors

class StudentTLikelihood(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
        """
        A Student-t likelihood that handles outliers by assuming that the data are distributed
        according to a Student-t distribution. The probability density function for each residual is
        given by:

            p(r | ν, σ) = Γ((ν+1)/2) / [√(νπ) σ Γ(ν/2)] · [1 + (r/σ)²/ν]^(–(ν+1)/2)

        where ν (nu) is the degrees of freedom and σ is the scale (standard deviation).

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma: The scale (standard deviation) for the model residuals.
        :type sigma: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments will require a prior
            and will be sampled over (unless a fixed value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        :param nu: The degrees of freedom for the Student-t distribution. Default to 3.0.
        :param priors: The priors for the parameters. Default to None if not provided.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to use in the optimization.
        :type fiducial_parameters: Union[dict, None]
        """
        self._noise_log_likelihood = None
        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)

        # Set default degrees of freedom for the Student-t distribution if not provided.
        if 'nu' not in self.parameters:
            self.parameters['nu'] = 3.0  # You can change this default as needed.

    def noise_log_likelihood(self) -> float:
        """
        Compute the log-likelihood assuming the signal is pure noise (i.e. the residuals are just y).
        """
        if self._noise_log_likelihood is None:
            nu = self.parameters.get('nu')
            self._noise_log_likelihood = self._student_t_log_likelihood(res=self.y, sigma=self.sigma, nu=nu)
        return self._noise_log_likelihood

    def log_likelihood(self) -> float:
        """
        Compute the total log-likelihood for the Student-t likelihood model.

        :return: The log-likelihood.
        :rtype: float
        """
        nu = self.parameters.get('nu')
        return np.nan_to_num(self._student_t_log_likelihood(res=self.residual, sigma=self.sigma, nu=nu))

    @staticmethod
    def _student_t_log_likelihood(res: np.ndarray, sigma: Union[float, np.ndarray], nu: float) -> Any:
        """
        Computes the log likelihood of the Student-t distribution for the residuals.

        For each data point, the log probability is given by:

          log[p(r|ν,σ)] = gammaln((ν+1)/2) - gammaln(ν/2)
                           - 0.5*log(νπ) - log(σ)
                           - ((ν+1)/2)*log[1 + (r/σ)²/ν]

        :param res: The residuals.
        :type res: np.ndarray
        :param sigma: The scale parameter.
        :type sigma: Union[float, np.ndarray]
        :param nu: Degrees of freedom.
        :type nu: float
        :return: The total log-likelihood.
        :rtype: float
        """
        term1 = gammaln((nu + 1) / 2) - gammaln(nu / 2)
        term2 = - 0.5 * np.log(nu * np.pi)
        term3 = - np.log(sigma)
        term4 = - ((nu + 1) / 2) * np.log(1 + (res / sigma) ** 2 / nu)
        log_pdf = term1 + term2 + term3 + term4
        return np.sum(log_pdf)


class GaussianLikelihoodUniformXErrors(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, None, np.ndarray],
            bin_size: Union[float, None, np.ndarray], function: callable, kwargs: dict = None, priors=None,
                 fiducial_parameters=None) -> None:
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
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """

        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)
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
        return np.nan_to_num(self.log_likelihood_x() + self.log_likelihood_y())


class GaussianLikelihoodQuadratureNoise(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma_i: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
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
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        """
        self.sigma_i = sigma_i
        # These lines of code infer the parameters from the provided function
        super().__init__(x=x, y=y, sigma=sigma_i, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)

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
        return np.nan_to_num(self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma))

class GaussianLikelihoodWithFractionalNoise(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma_i: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
        """
        A Gaussian likelihood with noise that is proportional to the model.
        The parameters are inferred from the arguments of function

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma_i: The standard deviation of the noise. This is part of the full noise.
                        The sigma used in the likelihood is sigma = sqrt(sigma_i^2*model_y**2)
        :type sigma_i: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """
        self.sigma_i = sigma_i
        # These lines of code infer the parameters from the provided function
        super().__init__(x=x, y=y, sigma=sigma_i, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)

    @property
    def full_sigma(self) -> Union[float, np.ndarray]:
        """
        :return: The standard deviation of the full noise
        :rtype: Union[float, np.ndarray]
        """
        model_y = self.function(self.x, **self.parameters, **self.kwargs)
        return np.sqrt(self.sigma_i**2.*model_y**2)

    def noise_log_likelihood(self) -> float:
        """
        :return: The noise log-likelihood, i.e. the log-likelihood assuming the signal is just noise.
        :rtype: float
        """
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._gaussian_log_likelihood(res=self.y, sigma=self.sigma_i)
        return self._noise_log_likelihood

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        return np.nan_to_num(self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma))

class GaussianLikelihoodWithSystematicNoise(GaussianLikelihood):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma_i: Union[float, None, np.ndarray],
            function: callable, kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
        """
        A Gaussian likelihood with a systematic noise term that is proportional to the model + some additive noise.
        The parameters are inferred from the arguments of function

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma_i: The standard deviation of the noise. This is part of the full noise.
                        The sigma used in the likelihood is sigma = sqrt(sigma_i^2 + model_y**2*sigma^2)
        :type sigma_i: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """
        self.sigma_i = sigma_i
        # These lines of code infer the parameters from the provided function
        super().__init__(x=x, y=y, sigma=sigma_i, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)

    @property
    def full_sigma(self) -> Union[float, np.ndarray]:
        """
        :return: The standard deviation of the full noise
        :rtype: Union[float, np.ndarray]
        """
        model_y = self.function(self.x, **self.parameters, **self.kwargs)
        return np.sqrt(self.sigma_i**2. + model_y**2*self.sigma**2.)

    def noise_log_likelihood(self) -> float:
        """
        :return: The noise log-likelihood, i.e. the log-likelihood assuming the signal is just noise.
        :rtype: float
        """
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._gaussian_log_likelihood(res=self.y, sigma=self.sigma_i)
        return self._noise_log_likelihood

    def log_likelihood(self) -> float:
        """
        :return: The log-likelihood.
        :rtype: float
        """
        return np.nan_to_num(self._gaussian_log_likelihood(res=self.residual, sigma=self.full_sigma))

class GaussianLikelihoodQuadratureNoiseNonDetections(GaussianLikelihoodQuadratureNoise):
    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma_i: Union[float, np.ndarray], function: callable,
            kwargs: dict = None, upperlimit_kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
        """A general Gaussian likelihood - the parameters are inferred from the
        arguments of function. Takes into account non-detections with a Uniform likelihood for those points

        :param x: The x values.
        :type x: np.ndarray
        :param y: The y values.
        :type y: np.ndarray
        :param sigma_i: The standard deviation of the noise. This is part of the full noise.
                        The sigma used in the likelihood is sigma = sqrt(sigma_i^2 + sigma^2)
        :type sigma_i: Union[float, None, np.ndarray]
        :param function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        :type function: callable
        :param kwargs: Any additional keywords for 'function'.
        :type kwargs: dict
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """
        super().__init__(x=x, y=y, sigma_i=sigma_i, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)
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
        return np.nan_to_num(self.log_likelihood_y() + self.log_likelihood_upper_limit())


class GRBGaussianLikelihood(GaussianLikelihood):

    def __init__(
            self, x: np.ndarray, y: np.ndarray, sigma: Union[float, np.ndarray],
            function: callable, kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
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
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """
        super().__init__(x=x, y=y, sigma=sigma, function=function, kwargs=kwargs, priors=priors,
                         fiducial_parameters=fiducial_parameters)


class PoissonLikelihood(_RedbackLikelihood):
    def __init__(
            self, time: np.ndarray, counts: np.ndarray, function: callable, integrated_rate_function: bool = True,
            dt: Union[float, np.ndarray] = None, kwargs: dict = None, priors=None, fiducial_parameters=None) -> None:
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
        :param priors: The priors for the parameters. Default to None if not provided.
        Only necessary if using maximum likelihood estimation functionality.
        :type priors: Union[dict, None]
        :param fiducial_parameters: The starting guesses for model parameters to
        use in the optimization for maximum likelihood estimation. Default to None if not provided.
        :type fiducial_parameters: Union[dict, None]
        """
        super(PoissonLikelihood, self).__init__(x=time, y=counts, function=function, kwargs=kwargs, priors=priors,
                                               fiducial_parameters=fiducial_parameters)
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
        return np.nan_to_num(self._poisson_log_likelihood(rate=rate))

    def _poisson_log_likelihood(self, rate: Union[float, np.ndarray]) -> Any:
        return np.sum(-rate + self.counts * np.log(rate) - gammaln(self.counts + 1))
