import numpy as np
import os

import bilby.core.prior
from bilby.core.prior import PriorDict

import redback.model_library
from redback.utils import logger


def get_priors(model, times=None, y=None, yerr=None, dt=None, **kwargs):
    """
    Get the prior for the given model. If the model is a prompt model, the times, y, and yerr must be provided.

    :param model: String referring to a name of a model implemented in Redback.
    :param times: Time array
    :param y: Y values, arbitrary units
    :param yerr: Error on y values, arbitrary units
    :param dt: time interval
    :param kwargs: Extra arguments to be passed to the prior function
    :return: priors: PriorDict object
    """
    prompt_prior_functions = dict(gaussian_prompt=get_gaussian_priors, skew_gaussian=get_skew_gaussian_priors,
                                  skew_exponential=get_skew_exponential_priors, fred=get_fred_priors,
                                  fred_extended=get_fred_extended_priors)

    if model in redback.model_library.modules_dict['prompt_models']:
        if times is None:
            times = np.array([0, 100])
        if y is None:
            y = np.array([1, 1e6])
        if yerr is None:
            yerr = np.array([1, 1e3])
        if dt is None:
            dt = np.ones(len(times))
        rate = y * dt
        priors = prompt_prior_functions[model](times=times, y=rate, yerr=yerr)
        priors['background_rate'] = bilby.core.prior.LogUniform(minimum=np.min(rate), maximum=np.max(rate),
                                                                name='background_rate')
        return priors

    priors = PriorDict()

    if model in redback.model_library.base_models_dict:
        logger.info(f'Setting up prior for base model {model}.')
        logger.info(f'You will need to explicitly set a prior on t0 and or extinction if relevant')

    # Try loading from main priors folder first
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', f'{model}.prior')
        priors.from_file(filename)
        return priors
    except FileNotFoundError:
        pass  # Continue to try the non_default_priors folder

    # Try loading from non_default_priors subfolder
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', 'non_default_priors', f'{model}.prior')
        priors.from_file(filename)
        return priors
    except FileNotFoundError:
        logger.warning(f'No prior file found for model {model} in either priors or non_default_priors folders. '
                       f'Perhaps you also want to set up the prior for the base model? '
                       f'Or you may need to set up your prior explicitly.')
        logger.info('Returning Empty PriorDict.')

    return priors


def get_prompt_priors(model, times, y, yerr, **kwargs):
    if model == 'gaussian':
        get_gaussian_priors(times=times, y=y, yerr=yerr, **kwargs)


def get_gaussian_priors(times, y, yerr, **kwargs):
    dt = np.min(np.diff(times))
    duration = times[-1] - times[0]
    priors = bilby.core.prior.PriorDict()
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=np.min(yerr), maximum=np.max(y),
                                                      name='amplitude', latex_label=r'$A$')
    priors['sigma'] = bilby.core.prior.LogUniform(minimum=3*dt, maximum=duration, name="sigma", latex_label=r"$\sigma$")
    priors['t_0'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[-1], name="t_0", latex_label=r"$t_0$")
    return priors


def get_skew_gaussian_priors(times, y, yerr, **kwargs):
    priors = get_gaussian_priors(times=times, y=y, yerr=yerr, **kwargs)
    for latex_label, part in zip([r"$\sigma_{\mathrm{rise}}$" r"$\sigma_{\mathrm{rise}}$"], ['rise', 'fall']):
        priors[f'sigma_{part}'] = bilby.core.prior.LogUniform(
            minimum=priors['sigma'].minimum, maximum=priors['sigma'].maximum,
            name=f"sigma_{part}", latex_label=latex_label)
    del priors['sigma']
    return priors


def get_skew_exponential_priors(times, y, yerr, **kwargs):
    priors = get_gaussian_priors(times=times, y=y, yerr=yerr, **kwargs)
    for latex_label, part in zip([r"$\tau_{\mathrm{rise}}$" r"$\tau_{\mathrm{rise}}$"], ['rise', 'fall']):
        priors[f'tau_{part}'] = bilby.core.prior.LogUniform(
            minimum=priors['sigma'].minimum, maximum=priors['sigma'].maximum,
            name=f"tau_{part}", latex_label=latex_label)
    del priors['sigma']
    return priors


def get_fred_priors(times, y, yerr, **kwargs):
    priors = bilby.core.prior.PriorDict()
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=np.min(yerr), maximum=np.max(y),
                                                      name='amplitude', latex_label=r'$A$')
    priors['tau'] = bilby.core.prior.Uniform(minimum=1e-3, maximum=1e3, name="t_0", latex_label=r"$t_0$")
    priors['psi'] = bilby.core.prior.Uniform(minimum=1e-3, maximum=1e3, name=r"\psi")
    priors['delta'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[-1], name=r"\delta")
    return priors


def get_fred_extended_priors(times, y, yerr, **kwargs):
    priors = get_fred_priors(times=times, y=y, yerr=yerr, **kwargs)
    priors['gamma'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e3, name=r"$\gamma$")
    priors['nu'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e3, name=r"$\nu")


def get_lensing_priors(nimages=2, dt_min=0.0, dt_max=1000.0, mu_min=0.1, mu_max=100.0, **kwargs):
    """
    Get priors for gravitational lensing parameters.

    :param nimages: Number of lensed images (default: 2)
    :param dt_min: Minimum time delay in days (default: 0.0)
    :param dt_max: Maximum time delay in days (default: 1000.0)
    :param mu_min: Minimum magnification factor (default: 0.1)
    :param mu_max: Maximum magnification factor (default: 100.0)
    :param kwargs: Additional keyword arguments
    :return: PriorDict with lensing parameters

    Example:
        >>> lensing_priors = get_lensing_priors(nimages=3)
        >>> base_priors = get_priors('arnett')
        >>> combined_priors = {**base_priors, **lensing_priors}
    """
    priors = bilby.core.prior.PriorDict()

    for i in range(1, nimages + 1):
        # First image is typically the reference (dt=0)
        if i == 1:
            priors[f'dt_{i}'] = bilby.core.prior.DeltaFunction(
                peak=0.0, name=f'dt_{i}', latex_label=rf'$\Delta t_{i}$ (days)')
        else:
            priors[f'dt_{i}'] = bilby.core.prior.Uniform(
                minimum=dt_min, maximum=dt_max, name=f'dt_{i}',
                latex_label=rf'$\Delta t_{i}$ (days)')

        priors[f'mu_{i}'] = bilby.core.prior.LogUniform(
            minimum=mu_min, maximum=mu_max, name=f'mu_{i}',
            latex_label=rf'$\mu_{i}$')

    return priors
