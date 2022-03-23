import bilby.core.prior
import numpy as np
import os
from typing import Union
import warnings

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
        """
        Constructor for an extension of the regular bilby `Result`. This result adds the capability of utilising the
        plotting methods of the `Transient` such as `plot_lightcurve`. The class does this by reconstructing the
        `Transient` object that was used during the run by saving the required information in `meta_data`.

        Parameters
        ----------
        label: str, optional
            Labels of files produced by this class.
        outdir: str, optional
            Output directory of the result. Default is the current directory.
        sampler: str, optional
            The sampler used during the run.
        search_parameter_keys: list, optional
            The parameters that were sampled in.
        fixed_parameter_keys: list, optional
            Parameters that had a `DeltaFunction` prior
        constraint_parameter_keys: list, optional
            Parameters that had a `Constraint` prior
        priors: Union[dict, bilby.core.prior.PriorDict]
            Dictionary of priors.
        sampler_kwargs: dict, optional
            Any keyword arguments passed to the sampling package.
        injection_parameters: dict, optional
            True parameters if the dataset is simulated.
        meta_data: dict, optional
            Additional dictionary. Contains the data used during the run and is used to reconstruct the `Transient`
            object used during the run.
        posterior: pd.Dataframe, optional
            Posterior samples with log likelihood and log prior values.
        samples: np.ndarray, optional
            An array of the output posterior samples.
        nested_samples: np.ndarray, optional
            An array of the unweighted samples
        log_evidence: float, optional
            The log evidence value if provided.
        log_evidence_err: float, optional
            The log evidence error value if provided
        information_gain: float, optional
            The information gain calculated.
        log_noise_evidence: float, optional
            The log noise evidence.
        log_bayes_factor: float, optional
            The log Bayes factor if we sampled using the likelihood ratio.
        log_likelihood_evaluations: np.ndarray, optional
            The evaluations of the likelihood for each sample point
        log_prior_evaluations: int, optional
            Number of log prior evaluations.
        sampling_time: float, optional
            The time taken to complete the sampling in seconds.
        nburn: int, optional
            The number of burn-in steps discarded for MCMC samplers
        num_likelihood_evaluations: int, optional
            Number of total likelihood evaluations.
        walkers: array_like, optional
            The samplers taken by an ensemble MCMC samplers.
        max_autocorrelation_time: float, optional
            The estimated maximum autocorrelation time for MCMC samplers.
        use_ratio: bool, optional
            A boolean stating whether the likelihood ratio, as opposed to the
            likelihood was used during sampling.
        parameter_labels: list, optional
            List of the latex-formatted parameter labels.
        parameter_labels_with_unit: list, optional
            List of the latex-formatted parameter labels with units.
        version: str,
            Version information for software used to generate the result. Note,
            this information is generated when the result object is initialized
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
        """
        Reconstruct the transient used during sampling time using the metadata information.

        Returns
        -------
        redback.transient.transient.Transient: The reconstructed Transient.
        """
        return TRANSIENT_DICT[self.transient_type](**self.meta_data)

    def plot_lightcurve(self, model: Union[callable, str] = None, **kwargs: None) -> None:
        """
        Reconstructs the transient and calls the specific `plot_lightcurve` method.

        Parameters
        ----------
        model: Union[callable, str], optional
            User specified model.
        kwargs: dict
            Any kwargs to be passed into the `plot_lightcurve` method.
        """
        if model is None:
            model = model_library.all_models_dict[self.model]
        self.transient.plot_lightcurve(model=model, posterior=self.posterior,
                                       model_kwargs=self.model_kwargs, **kwargs)

    def plot_multiband_lightcurve(self, model: Union[callable, str] = None, **kwargs: None) -> None:
        """
        Reconstructs the transient and calls the specific `plot_multiband_lightcurve` method.

        Parameters
        ----------
        model: Union[callable, str], optional
            User specified model.
        kwargs: dict
            Any kwargs to be passed into the `plot_lightcurve` method.
        """
        if model is None:
            model = model_library.all_models_dict[self.model]
        self.transient.plot_multiband_lightcurve(
            model=model, posterior=self.posterior, model_kwargs=self.model_kwargs, **kwargs)

    def plot_data(self, **kwargs: None) -> None:
        """
        Reconstructs the transient and calls the specific `plot_data` method.

        Parameters
        ----------
        kwargs: dict
            Any kwargs to be passed into the `plot_data` method.
        """
        self.transient.plot_data(**kwargs)

    def plot_multiband(self, **kwargs: None) -> None:
        """
        Reconstructs the transient and calls the specific `plot_multiband` method.

        Parameters
        ----------
        kwargs: dict
            Any kwargs to be passed into the `plot_multiband` method.
        """
        self.transient.plot_multiband(**kwargs)


def read_in_result(
        filename: str = None, outdir: str = None, label: str = None,
        extension: str = 'json', gzip: bool = False) -> RedbackResult:
    """

    Parameters
    ----------
    filename: str, optional
        Filename with entire path of result to open.
    outdir: str, optional
        If filename is not given, directory of the result.
    label: str, optional
        If filename is not given, label of the result.
    extension: str, optional
        If filename is not given, filename extension. Must be in ('json', 'hdf5', 'h5', 'pkl', 'pickle', 'gz').
        Default is 'json'.
    gzip: bool, optional
        If the file is compressed with gzip. Default is False.

    Returns
    -------
    RedbackResult: The loaded redback result.

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
