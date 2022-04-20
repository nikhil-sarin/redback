import bilby.core.prior
import numpy as np
import os
from typing import Union
import warnings
import matplotlib

import pandas as pd
from bilby.core.result import Result
from bilby.core.result import _determine_file_name # noqa

import redback.transient.transient
from redback import model_library
from redback.transient import TRANSIENT_DICT
from redback.utils import MetaDataAccessor

warnings.simplefilter(action='ignore')


class RedbackResult(Result):
    model = MetaDataAccessor('model')
    transient_type = MetaDataAccessor('transient_type')
    model_kwargs = MetaDataAccessor('model_kwargs')
    name = MetaDataAccessor('name')
    path = MetaDataAccessor('path')

    def __init__(
            self, label: str = 'no_label', outdir: str = '.', sampler: str = None, search_parameter_keys: list = None,
            fixed_parameter_keys: list = None, constraint_parameter_keys: list = None,
            priors: Union[dict, bilby.core.prior.PriorDict] = None, sampler_kwargs: dict = None,
            injection_parameters: dict = None, meta_data: dict = None, posterior: pd.DataFrame = None,
            samples: pd.DataFrame = None, nested_samples: pd.DataFrame = None, log_evidence: float = np.nan,
            log_evidence_err: float = np.nan, information_gain: float = np.nan, log_noise_evidence: float = np.nan,
            log_bayes_factor: float = np.nan, log_likelihood_evaluations: np.ndarray = None,
            log_prior_evaluations: int = None, sampling_time: float = None, nburn: int = None,
            num_likelihood_evaluations: int = None, walkers: int = None, max_autocorrelation_time: float = None,
            use_ratio: bool = None, parameter_labels: list = None, parameter_labels_with_unit: list = None,
            version: str = None) -> None:
        """Constructor for an extension of the regular bilby `Result`. This result adds the capability of utilising the
        plotting methods of the `Transient` such as `plot_lightcurve`. The class does this by reconstructing the
        `Transient` object that was used during the run by saving the required information in `meta_data`.

        :param label: Labels of files produced by this class.
        :type label: str, optional
        :param outdir: Output directory of the result. Default is the current directory.
        :type outdir: str, optional
        :param sampler: The sampler used during the run.
        :type sampler: str, optional
        :param search_parameter_keys: The parameters that were sampled in.
        :type search_parameter_keys: list, optional
        :param fixed_parameter_keys: Parameters that had a `DeltaFunction` prior
        :type fixed_parameter_keys: list, optional
        :param constraint_parameter_keys: Parameters that had a `Constraint` prior
        :type constraint_parameter_keys: list, optional
        :param priors: Dictionary of priors.
        :type priors: Union[dict, bilby.core.prior.PriorDict]
        :param sampler_kwargs: Any keyword arguments passed to the sampling package.
        :type sampler_kwargs: dict, optional
        :param injection_parameters: True parameters if the dataset is simulated.
        :type injection_parameters: dict, optional
        :type meta_data: dict, optional
        :param meta_data: Additional dictionary. Contains the data used during the run and
                          is used to reconstruct the `Transient` object used during the run.
        :param posterior: Posterior samples with log likelihood and log prior values.
        :type posterior: pd.Dataframe, optional
        :param samples: An array of the output posterior samples.
        :type samples: np.ndarray, optional
        :param nested_samples: An array of the unweighted samples
        :type nested_samples: np.ndarray, optional
        :param log_evidence: The log evidence value if provided.
        :type log_evidence: float, optional
        :param log_evidence_err: The log evidence error value if provided
        :type log_evidence_err: float, optional
        :param information_gain: The information gain calculated.
        :type information_gain: float, optional
        :param log_noise_evidence:The log noise evidence.
        :type log_noise_evidence: float, optional
        :param log_bayes_factor:The log Bayes factor if we sampled using the likelihood ratio.
        :type log_bayes_factor: float, optional
        :param log_likelihood_evaluations: The evaluations of the likelihood for each sample point
        :type log_likelihood_evaluations: np.ndarray, optional
        :param log_prior_evaluations: Number of log prior evaluations.
        :type log_prior_evaluations: int, optional
        :param sampling_time: The time taken to complete the sampling in seconds.
        :type sampling_time: float, optional
        :param nburn: The number of burn-in steps discarded for MCMC samplers
        :type nburn: int, optional
        :param num_likelihood_evaluations: Number of total likelihood evaluations.
        :type num_likelihood_evaluations: int, optional
        :param walkers: The samplers taken by an ensemble MCMC samplers.
        :type walkers: array_like, optional
        :param max_autocorrelation_time: The estimated maximum autocorrelation time for MCMC samplers.
        :type max_autocorrelation_time: float, optional
        :param use_ratio:
            A boolean stating whether the likelihood ratio, as opposed to the
            likelihood was used during sampling.
        :type use_ratio: bool, optional
        :param parameter_labels: List of the latex-formatted parameter labels.
        :type parameter_labels: list, optional
        :param parameter_labels_with_unit: List of the latex-formatted parameter labels with units.
        :type parameter_labels_with_unit: list, optional
        :param version: Version information for software used to generate the result. Note,
                        this information is generated when the result object is initialized.
        :type version: str,
        """
        super(RedbackResult, self).__init__(
            label=label, outdir=outdir, sampler=sampler,
            search_parameter_keys=search_parameter_keys, fixed_parameter_keys=fixed_parameter_keys,
            constraint_parameter_keys=constraint_parameter_keys, priors=priors,
            sampler_kwargs=sampler_kwargs, injection_parameters=injection_parameters,
            meta_data=meta_data, posterior=posterior, samples=samples,
            nested_samples=nested_samples, log_evidence=log_evidence,
            log_evidence_err=log_evidence_err, information_gain=information_gain,
            log_noise_evidence=log_noise_evidence, log_bayes_factor=log_bayes_factor,
            log_likelihood_evaluations=log_likelihood_evaluations,
            log_prior_evaluations=log_prior_evaluations, sampling_time=sampling_time, nburn=nburn,
            num_likelihood_evaluations=num_likelihood_evaluations, walkers=walkers,
            max_autocorrelation_time=max_autocorrelation_time, use_ratio=use_ratio,
            parameter_labels=parameter_labels, parameter_labels_with_unit=parameter_labels_with_unit,
            version=version)

    @property
    def transient(self) -> redback.transient.transient.Transient:
        """Reconstruct the transient used during sampling time using the metadata information.

        :return: The reconstructed Transient.
        :rtype: redback.transient.transient.Transient
        """
        return TRANSIENT_DICT[self.transient_type](**self.meta_data)

    def plot_lightcurve(self, model: Union[callable, str] = None, **kwargs: None) -> matplotlib.axes.Axes:
        """ Reconstructs the transient and calls the specific `plot_lightcurve` method.
        Detailed documentation appears below by running `print(plot_lightcurve.__doc__)` """
        if model is None:
            model = model_library.all_models_dict[self.model]
        return self.transient.plot_lightcurve(model=model, posterior=self.posterior,
                                              model_kwargs=self.model_kwargs, **kwargs)

    def plot_residual(self, model: Union[callable, str] = None, **kwargs: None) -> matplotlib.axes.Axes:
        """Reconstructs the transient and calls the specific `plot_residual` method.
        Detailed documentation appears below by running `print(plot_residual.__doc__)` """
        if model is None:
            model = model_library.all_models_dict[self.model]
        return self.transient.plot_residual(model=model, posterior=self.posterior,
                                            model_kwargs=self.model_kwargs, **kwargs)

    def plot_multiband_lightcurve(self, model: Union[callable, str] = None, **kwargs: None) -> matplotlib.axes.Axes:
        """Reconstructs the transient and calls the specific `plot_multiband_lightcurve` method.
        Detailed documentation appears below by running `print(plot_multiband_lightcurve.__doc__)` """
        if model is None:
            model = model_library.all_models_dict[self.model]
        return self.transient.plot_multiband_lightcurve(
            model=model, posterior=self.posterior, model_kwargs=self.model_kwargs, **kwargs)

    def plot_data(self, **kwargs: None) -> matplotlib.axes.Axes:
        """Reconstructs the transient and calls the specific `plot_data` method.
        Detailed documentation appears below by running `print(plot_data.__doc__)` """
        return self.transient.plot_data(**kwargs)

    def plot_multiband(self, **kwargs: None) -> matplotlib.axes.Axes:
        """Reconstructs the transient and calls the specific `plot_multiband` method.
        Detailed documentation appears below by running `print(plot_multiband.__doc__)` """
        return self.transient.plot_multiband(**kwargs)

    plot_data.__doc__ = plot_data.__doc__ + redback.transient.Transient.plot_data.__doc__
    plot_lightcurve.__doc__ = plot_lightcurve.__doc__ + redback.transient.Transient.plot_lightcurve.__doc__
    plot_residual.__doc__ = plot_residual.__doc__ + redback.transient.Transient.plot_residual.__doc__
    plot_multiband.__doc__ = plot_multiband.__doc__ + redback.transient.Transient.plot_multiband.__doc__
    plot_multiband_lightcurve.__doc__ = \
        plot_multiband_lightcurve.__doc__ + redback.transient.Transient.plot_multiband_lightcurve.__doc__


def read_in_result(
        filename: str = None, outdir: str = None, label: str = None,
        extension: str = 'json', gzip: bool = False) -> RedbackResult:
    """
    :param filename: Filename with entire path of result to open.
    :type filename: str, optional
    :param outdir: If filename is not given, directory of the result.
    :type outdir: str, optional
    :param label: If filename is not given, label of the result.
    :type label: str, optional
    :type extension: str, optional
    :param extension: If filename is not given, filename extension.
                      Must be in ('json', 'hdf5', 'h5', 'pkl', 'pickle', 'gz').
                      (Default value = 'json')
    :param gzip: If the file is compressed with gzip. Default is False.
    :type gzip: bool, optional

    :return: The loaded redback result.
    :rtype: RedbackResult
    """
    filename = _determine_file_name(filename, outdir, label, extension, gzip)

    # Get the actual extension (may differ from the default extension if the filename is given)
    extension = os.path.splitext(filename)[1].lstrip('.')
    if extension == 'gz':  # gzipped file
        extension = os.path.splitext(os.path.splitext(filename)[0])[1].lstrip('.')

    if 'json' in extension:
        result = RedbackResult.from_json(filename=filename)
    elif ('hdf5' in extension) or ('h5' in extension):
        result = RedbackResult.from_hdf5(filename=filename)
    elif ("pkl" in extension) or ("pickle" in extension):
        result = RedbackResult.from_pickle(filename=filename)
    elif extension is None:
        raise ValueError("No filetype extension provided")
    else:
        raise ValueError("Filetype {} not understood".format(extension))
    return result
