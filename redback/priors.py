import numpy as np
import os

import bilby.core.prior
from bilby.core.prior import PriorDict

import redback.model_library
from redback.utils import logger


def get_priors(model, times=None, y=None, yerr=None, dt=None, **kwargs):
    prompt_prior_functions = dict(gaussian=get_gaussian_priors, skew_gaussian=get_skew_gaussian_priors,
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
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', f'{model}.prior')
        priors.from_file(filename)
    except FileNotFoundError as e:
        logger.warning(e)
        logger.warning('Returning empty PriorDict.')
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
