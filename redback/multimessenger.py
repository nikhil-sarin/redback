"""
Multi-messenger analysis framework for joint fitting of transient data across multiple messengers.

This module provides infrastructure for jointly analyzing transients observed through different messengers
(optical, X-ray, radio, gravitational waves, neutrinos, etc.) with shared physical parameters.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import bilby

import redback
from redback.likelihoods import GaussianLikelihood, GaussianLikelihoodQuadratureNoise
from redback.model_library import all_models_dict
from redback.result import RedbackResult
from redback.utils import logger
from redback.transient.transient import Transient


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
        likelihood_type: str = 'GaussianLikelihood'
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

        Returns
        -------
        bilby.Likelihood
            Constructed likelihood object
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Convert string model name to function if needed
        if isinstance(model, str):
            if model not in all_models_dict:
                raise ValueError(f"Model '{model}' not found in redback model library")
            model_func = all_models_dict[model]
        else:
            model_func = model

        # Get data from transient
        x, x_err, y, y_err = transient.get_filtered_data()

        # Select likelihood class
        if likelihood_type == 'GaussianLikelihood':
            likelihood_class = GaussianLikelihood
        elif likelihood_type == 'GaussianLikelihoodQuadratureNoise':
            likelihood_class = GaussianLikelihoodQuadratureNoise
        else:
            raise ValueError(f"Unsupported likelihood type: {likelihood_type}")

        # Construct likelihood
        if x_err is not None and np.any(x_err > 0):
            # If time errors are present, use a likelihood that can handle them
            logger.info(f"Building {likelihood_type} for {messenger} with time errors")
            likelihood = likelihood_class(
                x=x, y=y, sigma=y_err, function=model_func, kwargs=model_kwargs
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

                likelihood = self._build_likelihood_for_messenger(
                    messenger, transient, model, mkwargs, ltype
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
            joint_likelihood = likelihoods[0]
        else:
            logger.info(f"Combining {len(likelihoods)} likelihoods into joint likelihood")
            joint_likelihood = bilby.core.likelihood.JointLikelihood(*likelihoods)

        # Ensure priors is a PriorDict
        if not isinstance(priors, bilby.core.prior.PriorDict):
            priors = bilby.core.prior.PriorDict(priors)

        # Log shared parameters
        if shared_params:
            logger.info(f"Shared parameters across messengers: {', '.join(shared_params)}")

        # Prepare metadata
        meta_data = {
            'multimessenger': True,
            'messengers': list(self.messengers.keys()) + list(self.external_likelihoods.keys()),
            'models': {k: v if isinstance(v, str) else v.__name__ for k, v in models.items()},
            'shared_params': shared_params or [],
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
            for messenger, prior_dict in individual_priors.items():
                if param in prior_dict:
                    joint_prior[param] = prior_dict[param]
                    logger.info(f"Using {messenger} prior for shared parameter '{param}'")
                    break

    # Add messenger-specific priors
    for messenger, prior_dict in individual_priors.items():
        for param, prior in prior_dict.items():
            if param not in shared_params:
                # Add messenger prefix to avoid naming conflicts
                param_name = f"{messenger}_{param}"
                joint_prior[param_name] = prior
            # Shared params are already added, so skip them

    return joint_prior
