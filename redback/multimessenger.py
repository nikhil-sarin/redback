"""
Multi-messenger analysis framework for joint fitting of transient data across multiple messengers.

This module provides infrastructure for jointly analyzing transients observed through different messengers
(optical, X-ray, radio, gravitational waves, neutrinos, etc.) with shared physical parameters.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import functools
import inspect
import bilby

import redback
from redback.likelihoods import (
    GaussianLikelihood, GaussianLikelihoodQuadratureNoise, GaussianLikelihoodUniformXErrors
)
from redback.model_library import all_models_dict
from redback.result import RedbackResult
from redback.utils import logger
from redback.transient.transient import Transient


def _get_model_function(model: Union[str, callable]) -> callable:
    if isinstance(model, str):
        if model not in all_models_dict:
            raise ValueError(f"Model '{model}' not found in redback model library")
        return all_models_dict[model]
    return model


def _get_model_parameter_names(function: callable) -> List[str]:
    return bilby.core.utils.introspection.infer_parameters_from_function(func=function)


def _get_independent_variable_name(function: callable) -> str:
    try:
        return next(iter(inspect.signature(function).parameters))
    except (StopIteration, TypeError, ValueError):
        return "x"


def _make_parameter_mapped_model(model_func: callable, parameter_mapping: Optional[Dict[str, str]] = None) -> callable:
    """
    Wrap a model so joint-analysis parameter names can differ from the model's native names.

    parameter_mapping maps joint parameter names to native model parameter names, e.g.
    {'viewing_angle': 'thv'} exposes viewing_angle to the sampler and passes it as thv.
    """
    parameter_mapping = parameter_mapping or {}
    model_parameters = _get_model_parameter_names(model_func)
    unknown_native_parameters = sorted(set(parameter_mapping.values()) - set(model_parameters))
    if unknown_native_parameters:
        raise ValueError(
            "Parameter mapping refers to model parameter(s) not present in the model "
            f"signature: {unknown_native_parameters}"
        )
    native_to_joint = {native: joint for joint, native in parameter_mapping.items()}
    exposed_parameters = [native_to_joint.get(parameter, parameter) for parameter in model_parameters]

    duplicates = {parameter for parameter in exposed_parameters if exposed_parameters.count(parameter) > 1}
    if duplicates:
        raise ValueError(f"Parameter mapping creates duplicate sampled parameters: {sorted(duplicates)}")

    @functools.wraps(model_func)
    def mapped_model(x, **parameters):
        native_parameters = parameters.copy()
        for joint_name, native_name in parameter_mapping.items():
            if joint_name in parameters:
                native_parameters[native_name] = parameters[joint_name]
                native_parameters.pop(joint_name, None)
        return model_func(x, **native_parameters)

    independent_variable = inspect.Parameter(
        _get_independent_variable_name(model_func), inspect.Parameter.POSITIONAL_OR_KEYWORD)
    signature_parameters = [independent_variable] + [
        inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in exposed_parameters
    ]
    mapped_model.__signature__ = inspect.Signature(parameters=signature_parameters)
    return mapped_model


def _get_transient_data_for_likelihood(transient: Transient) -> tuple:
    """
    Return transient data in likelihood-ready form.

    If no active band filter is configured, use the full x/y arrays directly.
    This keeps flux-density transients with frequency arrays but no photometric
    band labels from failing inside Transient.get_filtered_data().
    """
    if getattr(transient, "active_bands", None) is None:
        return transient.x, transient.x_err, transient.y, transient.y_err
    return transient.get_filtered_data()


def _has_positive_x_errors(x_err: Optional[np.ndarray]) -> bool:
    return x_err is not None and np.any(np.asarray(x_err) > 0)


def _get_x_error_bin_size(x_err: np.ndarray) -> np.ndarray:
    x_err = np.asarray(x_err)
    if x_err.ndim == 2 and x_err.shape[0] == 2:
        return np.sum(np.abs(x_err), axis=0)
    return x_err


def _validate_shared_parameters(likelihoods: List[bilby.Likelihood], shared_params: Optional[List[str]]) -> None:
    if not shared_params:
        return

    missing_params = []
    singly_present_params = []
    for parameter in shared_params:
        count = sum(parameter in likelihood.parameters for likelihood in likelihoods)
        if count == 0:
            missing_params.append(parameter)
        elif len(likelihoods) > 1 and count == 1:
            singly_present_params.append(parameter)

    if missing_params:
        raise ValueError(
            "Shared parameter(s) are not present in any likelihood after applying "
            f"parameter mappings: {missing_params}"
        )
    if singly_present_params:
        raise ValueError(
            "Shared parameter(s) are present in only one likelihood and therefore are not "
            f"actually shared: {singly_present_params}. Use parameter_mappings or model wrappers "
            "so each relevant likelihood exposes the same sampled parameter name."
        )


def _log_likelihood_accepts_parameters(likelihood: bilby.Likelihood) -> bool:
    try:
        signature = inspect.signature(likelihood.log_likelihood)
    except (TypeError, ValueError):
        return False
    return "parameters" in signature.parameters


class MultiMessengerLikelihood(bilby.Likelihood):
    """A sampler-compatible likelihood product for redback and bilby likelihoods."""

    def __init__(self, *likelihoods: bilby.Likelihood):
        if len(likelihoods) == 0:
            raise ValueError("At least one likelihood is required.")
        self.likelihoods = likelihoods
        self._parameter_names_by_likelihood = [
            (likelihood, set(likelihood.parameters)) for likelihood in likelihoods
        ]
        parameters = {}
        for likelihood in likelihoods:
            parameters.update(likelihood.parameters)
        super().__init__(parameters=parameters)

    def _get_child_parameters(self, likelihood: bilby.Likelihood) -> Dict[str, Any]:
        child_parameter_names = None
        for child_likelihood, parameter_names in self._parameter_names_by_likelihood:
            if child_likelihood is likelihood:
                child_parameter_names = parameter_names
                break
        if child_parameter_names is None:
            raise ValueError("Likelihood is not part of this MultiMessengerLikelihood.")
        return {
            parameter: self.parameters[parameter]
            for parameter in child_parameter_names
            if parameter in self.parameters
        }

    def _update_parameters(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        if parameters is not None:
            self.parameters.update(parameters)

    def log_likelihood(self, parameters: Optional[Dict[str, Any]] = None) -> float:
        self._update_parameters(parameters=parameters)
        log_likelihood = 0.0
        for likelihood in self.likelihoods:
            child_parameters = self._get_child_parameters(likelihood=likelihood)
            likelihood.parameters.update(child_parameters)
            if _log_likelihood_accepts_parameters(likelihood):
                log_likelihood += likelihood.log_likelihood(parameters=child_parameters)
            else:
                log_likelihood += likelihood.log_likelihood()
        return log_likelihood

    def noise_log_likelihood(self) -> float:
        return sum(likelihood.noise_log_likelihood() for likelihood in self.likelihoods)


class MultiMessengerTransient:
    """
    Joint analysis of multiple messengers for transient events.

    This class enables multi-messenger analysis by combining data from different observational
    channels (electromagnetic, gravitational wave, neutrino) and performing joint parameter
    estimation with shared physical parameters.

    Examples
    --------
    Basic usage for a kilonova + GRB afterglow analysis:

    >>> import redback
    >>> mm_transient = MultiMessengerTransient(
    ...     optical_transient=kilonova_transient,
    ...     xray_transient=xray_transient,
    ...     radio_transient=radio_transient
    ... )
    >>> result = mm_transient.fit_joint(
    ...     models={'optical': 'two_component_kilonova_model',
    ...             'xray': 'tophat',
    ...             'radio': 'tophat'},
    ...     shared_params=['viewing_angle', 'luminosity_distance'],
    ...     model_kwargs={'optical': {'output_format': 'magnitude'},
    ...                   'xray': {'output_format': 'flux_density'},
    ...                   'radio': {'output_format': 'flux_density'}},
    ...     priors=priors
    ... )

    Advanced usage with custom likelihoods and GW data:

    >>> mm_transient = MultiMessengerTransient(
    ...     optical_transient=optical_lc,
    ...     gw_likelihood=gw_likelihood  # Pre-constructed bilby GW likelihood
    ... )
    >>> result = mm_transient.fit_joint(
    ...     models={'optical': 'two_component_kilonova_model'},
    ...     shared_params=['viewing_angle', 'luminosity_distance'],
    ...     priors=priors
    ... )
    """

    def __init__(
        self,
        optical_transient: Optional[Transient] = None,
        xray_transient: Optional[Transient] = None,
        radio_transient: Optional[Transient] = None,
        uv_transient: Optional[Transient] = None,
        infrared_transient: Optional[Transient] = None,
        gw_likelihood: Optional[bilby.Likelihood] = None,
        neutrino_likelihood: Optional[bilby.Likelihood] = None,
        custom_likelihoods: Optional[Dict[str, bilby.Likelihood]] = None,
        name: str = 'multimessenger_transient'
    ):
        """
        Initialize a MultiMessengerTransient object.

        Parameters
        ----------
        optical_transient : redback.transient.Transient, optional
            Optical/NIR data as a Redback transient object
        xray_transient : redback.transient.Transient, optional
            X-ray data as a Redback transient object
        radio_transient : redback.transient.Transient, optional
            Radio data as a Redback transient object
        uv_transient : redback.transient.Transient, optional
            UV data as a Redback transient object
        infrared_transient : redback.transient.Transient, optional
            Infrared data as a Redback transient object
        gw_likelihood : bilby.Likelihood, optional
            Pre-constructed gravitational wave likelihood (e.g., from bilby.gw)
        neutrino_likelihood : bilby.Likelihood, optional
            Pre-constructed neutrino likelihood
        custom_likelihoods : dict, optional
            Dictionary of custom likelihood objects with messenger names as keys
        name : str, optional
            Name for this multi-messenger transient (default: 'multimessenger_transient')
        """
        self.name = name

        # Store transient data objects
        self.messengers = {
            'optical': optical_transient,
            'xray': xray_transient,
            'radio': radio_transient,
            'uv': uv_transient,
            'infrared': infrared_transient
        }

        # Remove None entries
        self.messengers = {k: v for k, v in self.messengers.items() if v is not None}

        # Store pre-constructed likelihoods (e.g., for GW or neutrinos)
        self.external_likelihoods = {}
        if gw_likelihood is not None:
            self.external_likelihoods['gw'] = gw_likelihood
        if neutrino_likelihood is not None:
            self.external_likelihoods['neutrino'] = neutrino_likelihood
        if custom_likelihoods is not None:
            self.external_likelihoods.update(custom_likelihoods)

        logger.info(f"Initialized MultiMessengerTransient '{name}' with {len(self.messengers)} "
                   f"transient data objects and {len(self.external_likelihoods)} external likelihoods")

    def _build_likelihood_for_messenger(
        self,
        messenger: str,
        transient: Transient,
        model: Union[str, callable],
        model_kwargs: Optional[Dict] = None,
        likelihood_type: str = 'GaussianLikelihood',
        parameter_mapping: Optional[Dict[str, str]] = None
    ) -> bilby.Likelihood:
        """
        Build a likelihood for a single messenger.

        Parameters
        ----------
        messenger : str
            Name of the messenger (e.g., 'optical', 'xray', 'radio')
        transient : redback.transient.Transient
            Transient data object
        model : str or callable
            Model name (string) or callable function
        model_kwargs : dict, optional
            Additional keyword arguments for the model
        likelihood_type : str, optional
            Type of likelihood to use (default: 'GaussianLikelihood')
            Options: 'GaussianLikelihood', 'GaussianLikelihoodQuadratureNoise'
        parameter_mapping : dict, optional
            Mapping from joint sampled parameter names to native model parameter names
            for this messenger. For example {'viewing_angle': 'thv'} shares the
            sampled viewing_angle parameter with a model that expects thv.

        Returns
        -------
        bilby.Likelihood
            Constructed likelihood object
        """
        if model_kwargs is None:
            model_kwargs = {}

        model_func = _make_parameter_mapped_model(
            _get_model_function(model), parameter_mapping=parameter_mapping)

        # Get data from transient
        x, x_err, y, y_err = _get_transient_data_for_likelihood(transient)

        # Select likelihood class
        if likelihood_type == 'GaussianLikelihood':
            likelihood_class = GaussianLikelihoodUniformXErrors if _has_positive_x_errors(x_err) else GaussianLikelihood
        elif likelihood_type == 'GaussianLikelihoodQuadratureNoise':
            if _has_positive_x_errors(x_err):
                raise ValueError(
                    "GaussianLikelihoodQuadratureNoise does not support x/time errors in "
                    "MultiMessengerTransient. Use GaussianLikelihood or provide a custom likelihood."
                )
            likelihood_class = GaussianLikelihoodQuadratureNoise
        else:
            raise ValueError(f"Unsupported likelihood type: {likelihood_type}")

        # Construct likelihood
        if likelihood_class is GaussianLikelihoodUniformXErrors:
            logger.info(f"Building {likelihood_type} for {messenger} with time errors")
            likelihood = likelihood_class(
                x=x, y=y, sigma=y_err, bin_size=_get_x_error_bin_size(x_err),
                function=model_func, kwargs=model_kwargs
            )
        else:
            if likelihood_type == 'GaussianLikelihoodQuadratureNoise':
                likelihood = likelihood_class(
                    x=x, y=y, sigma_i=y_err, function=model_func, kwargs=model_kwargs
                )
            else:
                likelihood = likelihood_class(
                    x=x, y=y, sigma=y_err, function=model_func, kwargs=model_kwargs
                )

        logger.info(f"Built likelihood for {messenger} messenger with model {model_func.__name__}")
        return likelihood

    def fit_joint(
        self,
        models: Dict[str, Union[str, callable]],
        priors: Union[bilby.core.prior.PriorDict, dict],
        shared_params: Optional[List[str]] = None,
        model_kwargs: Optional[Dict[str, Dict]] = None,
        likelihood_types: Optional[Dict[str, str]] = None,
        parameter_mappings: Optional[Dict[str, Dict[str, str]]] = None,
        sampler: str = 'dynesty',
        nlive: int = 2000,
        walks: int = 200,
        outdir: Optional[str] = None,
        label: Optional[str] = None,
        resume: bool = True,
        plot: bool = True,
        save_format: str = 'json',
        **kwargs
    ) -> bilby.core.result.Result:
        """
        Perform joint multi-messenger analysis.

        This method builds individual likelihoods for each messenger, combines them into a joint
        likelihood, and runs parameter estimation with the specified sampler.

        Parameters
        ----------
        models : dict
            Dictionary mapping messenger names to model names/functions.
            Example: {'optical': 'two_component_kilonova_model', 'xray': 'tophat'}
        priors : bilby.core.prior.PriorDict or dict
            Prior distributions for all parameters. For shared parameters, the same prior
            will be used across all messengers.
        shared_params : list of str, optional
            List of parameter names that are shared across messengers.
            Example: ['viewing_angle', 'luminosity_distance', 'time_of_merger']
            If None, parameters are assumed independent unless they have the same name.
        model_kwargs : dict of dict, optional
            Dictionary mapping messenger names to their model keyword arguments.
            Example: {'optical': {'output_format': 'magnitude'},
                      'xray': {'output_format': 'flux_density', 'frequency': freq_array}}
        likelihood_types : dict of str, optional
            Dictionary mapping messenger names to likelihood types.
            Example: {'optical': 'GaussianLikelihood', 'xray': 'GaussianLikelihoodQuadratureNoise'}
            Default: 'GaussianLikelihood' for all messengers
        parameter_mappings : dict of dict, optional
            Dictionary mapping messenger names to parameter maps. Each map should
            map joint sampled parameter names to that messenger model's native
            parameter names. Example: {'xray': {'viewing_angle': 'thv'}}.
        sampler : str, optional
            Sampler to use (default: 'dynesty'). See bilby documentation for options.
        nlive : int, optional
            Number of live points for nested sampling (default: 2000)
        walks : int, optional
            Number of random walks for dynesty (default: 200)
        outdir : str, optional
            Output directory for results (default: './outdir_multimessenger')
        label : str, optional
            Label for output files (default: self.name)
        resume : bool, optional
            Whether to resume from checkpoint if available (default: True)
        plot : bool, optional
            Whether to create corner plots (default: True)
        save_format : str, optional
            Format for saving results (default: 'json')
        **kwargs
            Additional keyword arguments passed to bilby.run_sampler

        Returns
        -------
        bilby.core.result.Result
            Result object containing posterior samples and evidence

        Notes
        -----
        The joint likelihood is constructed as the product of individual messenger likelihoods:
        L_joint = L_optical × L_xray × L_radio × ...

        For shared parameters, the same parameter value is used across all relevant models,
        allowing the data from different messengers to jointly constrain these parameters.

        Examples
        --------
        >>> result = mm_transient.fit_joint(
        ...     models={'optical': 'two_component_kilonova_model',
        ...             'xray': 'tophat',
        ...             'radio': 'tophat'},
        ...     shared_params=['viewing_angle', 'luminosity_distance'],
        ...     priors=my_priors,
        ...     nlive=2000
        ... )
        """
        if model_kwargs is None:
            model_kwargs = {}

        if likelihood_types is None:
            likelihood_types = {}
        if parameter_mappings is None:
            parameter_mappings = {}

        # Set default output directory and label
        outdir = outdir or './outdir_multimessenger'
        label = label or self.name

        Path(outdir).mkdir(parents=True, exist_ok=True)

        # Build likelihoods for each messenger
        likelihoods = []

        # Build EM likelihoods from transient objects
        for messenger, transient in self.messengers.items():
            if messenger in models:
                model = models[messenger]
                mkwargs = model_kwargs.get(messenger, {})
                ltype = likelihood_types.get(messenger, 'GaussianLikelihood')
                parameter_mapping = parameter_mappings.get(messenger, {})

                likelihood = self._build_likelihood_for_messenger(
                    messenger, transient, model, mkwargs, ltype, parameter_mapping
                )
                likelihoods.append(likelihood)
            else:
                logger.warning(f"No model specified for messenger '{messenger}', skipping")

        # Add external likelihoods (GW, neutrino, etc.)
        for messenger, likelihood in self.external_likelihoods.items():
            logger.info(f"Adding external likelihood for {messenger}")
            likelihoods.append(likelihood)

        if len(likelihoods) == 0:
            raise ValueError("No likelihoods were constructed. Please provide models or external likelihoods.")

        # Construct joint likelihood
        if len(likelihoods) == 1:
            logger.warning("Only one likelihood present. Joint analysis reduces to single-messenger analysis.")
        else:
            logger.info(f"Combining {len(likelihoods)} likelihoods into joint likelihood")
        joint_likelihood = MultiMessengerLikelihood(*likelihoods)

        # Ensure priors is a PriorDict
        if not isinstance(priors, bilby.core.prior.PriorDict):
            priors = bilby.core.prior.PriorDict(priors)

        # Log shared parameters
        if shared_params:
            logger.info(f"Shared parameters across messengers: {', '.join(shared_params)}")
            _validate_shared_parameters(likelihoods=likelihoods, shared_params=shared_params)

        # Prepare metadata
        meta_data = {
            'multimessenger': True,
            'messengers': list(self.messengers.keys()) + list(self.external_likelihoods.keys()),
            'models': {k: v if isinstance(v, str) else v.__name__ for k, v in models.items()},
            'shared_params': shared_params or [],
            'parameter_mappings': parameter_mappings,
            'name': self.name
        }

        # Run sampler
        logger.info(f"Starting joint analysis with {sampler} sampler")
        result = bilby.run_sampler(
            likelihood=joint_likelihood,
            priors=priors,
            sampler=sampler,
            nlive=nlive,
            walks=walks,
            outdir=outdir,
            label=label,
            resume=resume,
            use_ratio=False,
            maxmcmc=10 * walks,
            result_class=RedbackResult,
            meta_data=meta_data,
            save=save_format,
            plot=plot,
            **kwargs
        )

        logger.info("Joint analysis complete")
        return result

    def fit_individual(
        self,
        models: Dict[str, Union[str, callable]],
        priors: Dict[str, Union[bilby.core.prior.PriorDict, dict]],
        model_kwargs: Optional[Dict[str, Dict]] = None,
        parameter_mappings: Optional[Dict[str, Dict[str, str]]] = None,
        sampler: str = 'dynesty',
        nlive: int = 2000,
        walks: int = 200,
        outdir: Optional[str] = None,
        resume: bool = True,
        plot: bool = True,
        **kwargs
    ) -> Dict[str, redback.result.RedbackResult]:
        """
        Fit each messenger independently (for comparison with joint analysis).

        Parameters
        ----------
        models : dict
            Dictionary mapping messenger names to model names/functions
        priors : dict
            Dictionary mapping messenger names to their prior distributions
        model_kwargs : dict of dict, optional
            Dictionary mapping messenger names to their model keyword arguments
        parameter_mappings : dict of dict, optional
            Dictionary mapping messenger names to parameter maps. Each map should
            map sampled parameter names to that messenger model's native parameter
            names, matching :meth:`fit_joint`.
        sampler : str, optional
            Sampler to use (default: 'dynesty')
        nlive : int, optional
            Number of live points (default: 2000)
        walks : int, optional
            Number of random walks (default: 200)
        outdir : str, optional
            Output directory (default: './outdir_individual')
        resume : bool, optional
            Whether to resume from checkpoint (default: True)
        plot : bool, optional
            Whether to create plots (default: True)
        **kwargs
            Additional arguments for bilby.run_sampler

        Returns
        -------
        dict
            Dictionary mapping messenger names to their individual fit results

        Examples
        --------
        >>> individual_results = mm_transient.fit_individual(
        ...     models={'optical': 'two_component_kilonova_model', 'xray': 'tophat'},
        ...     priors={'optical': optical_priors, 'xray': xray_priors}
        ... )
        >>> optical_result = individual_results['optical']
        """
        if model_kwargs is None:
            model_kwargs = {}
        if parameter_mappings is None:
            parameter_mappings = {}

        outdir = outdir or './outdir_individual'
        Path(outdir).mkdir(parents=True, exist_ok=True)

        results = {}

        for messenger, transient in self.messengers.items():
            if messenger not in models:
                logger.warning(f"No model specified for messenger '{messenger}', skipping")
                continue

            if messenger not in priors:
                logger.warning(f"No prior specified for messenger '{messenger}', skipping")
                continue

            model = models[messenger]
            prior = priors[messenger]
            mkwargs = model_kwargs.get(messenger, {})
            parameter_mapping = parameter_mappings.get(messenger, {})
            if parameter_mapping:
                model = _make_parameter_mapped_model(
                    _get_model_function(model), parameter_mapping=parameter_mapping)

            logger.info(f"Fitting {messenger} messenger independently")

            messenger_outdir = f"{outdir}/{messenger}"

            result = redback.fit_model(
                transient=transient,
                model=model,
                prior=prior,
                model_kwargs=mkwargs,
                sampler=sampler,
                nlive=nlive,
                walks=walks,
                outdir=messenger_outdir,
                label=f"{self.name}_{messenger}",
                resume=resume,
                plot=plot,
                **kwargs
            )

            results[messenger] = result
            logger.info(f"Completed fit for {messenger}")

        return results

    def add_messenger(self, messenger_name: str, transient: Optional[Transient] = None,
                     likelihood: Optional[bilby.Likelihood] = None):
        """
        Add a new messenger to the analysis.

        Parameters
        ----------
        messenger_name : str
            Name for the messenger
        transient : redback.transient.Transient, optional
            Transient data object
        likelihood : bilby.Likelihood, optional
            Pre-constructed likelihood object

        Notes
        -----
        Either transient or likelihood must be provided, but not both.
        """
        if transient is not None and likelihood is not None:
            raise ValueError("Provide either transient or likelihood, not both")
        if transient is None and likelihood is None:
            raise ValueError("Must provide either transient or likelihood")

        if transient is not None:
            self.messengers[messenger_name] = transient
            logger.info(f"Added transient data for {messenger_name}")
        else:
            self.external_likelihoods[messenger_name] = likelihood
            logger.info(f"Added external likelihood for {messenger_name}")

    def remove_messenger(self, messenger_name: str):
        """
        Remove a messenger from the analysis.

        Parameters
        ----------
        messenger_name : str
            Name of the messenger to remove
        """
        if messenger_name in self.messengers:
            del self.messengers[messenger_name]
            logger.info(f"Removed {messenger_name} from messengers")
        elif messenger_name in self.external_likelihoods:
            del self.external_likelihoods[messenger_name]
            logger.info(f"Removed {messenger_name} from external likelihoods")
        else:
            logger.warning(f"Messenger '{messenger_name}' not found")

    def __repr__(self):
        transient_messengers = list(self.messengers.keys())
        external_messengers = list(self.external_likelihoods.keys())
        return (f"MultiMessengerTransient(name='{self.name}', "
                f"transients={transient_messengers}, "
                f"external_likelihoods={external_messengers})")


def create_joint_prior(
    individual_priors: Dict[str, bilby.core.prior.PriorDict],
    shared_params: List[str],
    shared_param_priors: Optional[Dict[str, bilby.core.prior.Prior]] = None
) -> bilby.core.prior.PriorDict:
    """
    Create a joint prior dictionary from individual messenger priors.

    This utility function helps construct a prior dictionary for joint multi-messenger
    analysis by combining individual priors and handling shared parameters.

    Parameters
    ----------
    individual_priors : dict
        Dictionary mapping messenger names to their PriorDict objects
    shared_params : list of str
        List of parameter names that are shared across messengers
    shared_param_priors : dict, optional
        Dictionary of prior objects for shared parameters. If not provided,
        the prior from the first messenger will be used.

    Returns
    -------
    bilby.core.prior.PriorDict
        Combined prior dictionary for joint analysis

    Examples
    --------
    >>> optical_priors = bilby.core.prior.PriorDict({
    ...     'viewing_angle': bilby.core.prior.Uniform(0, np.pi/2),
    ...     'kappa': bilby.core.prior.Uniform(0.1, 10)
    ... })
    >>> xray_priors = bilby.core.prior.PriorDict({
    ...     'viewing_angle': bilby.core.prior.Uniform(0, np.pi/2),
    ...     'log_n0': bilby.core.prior.Uniform(-5, 2)
    ... })
    >>> joint_priors = create_joint_prior(
    ...     {'optical': optical_priors, 'xray': xray_priors},
    ...     shared_params=['viewing_angle']
    ... )
    """
    joint_prior = bilby.core.prior.PriorDict()

    # Add priors for shared parameters
    for param in shared_params:
        if shared_param_priors and param in shared_param_priors:
            joint_prior[param] = shared_param_priors[param]
        else:
            # Use the prior from the first messenger that has this parameter
            found_shared_param = False
            for messenger, prior_dict in individual_priors.items():
                if param in prior_dict:
                    joint_prior[param] = prior_dict[param]
                    logger.info(f"Using {messenger} prior for shared parameter '{param}'")
                    found_shared_param = True
                    break
            if not found_shared_param:
                raise ValueError(
                    f"Shared parameter '{param}' is not present in any individual prior. "
                    "Add it to at least one prior dictionary or pass shared_param_priors."
                )

    # Add messenger-specific priors. Parameter names are left unchanged so they
    # match the likelihood parameters built by MultiMessengerTransient.
    for messenger, prior_dict in individual_priors.items():
        for param, prior in prior_dict.items():
            if param not in shared_params:
                if param in joint_prior:
                    raise ValueError(
                        f"Parameter '{param}' appears in multiple priors but is not marked shared. "
                        "Use distinct model parameter names, a custom model wrapper, or include it "
                        "in shared_params if it should be common."
                    )
                joint_prior[param] = prior
            # Shared params are already added, so skip them

    return joint_prior
