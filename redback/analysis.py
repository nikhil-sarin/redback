import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import redback.model_library
from redback.utils import logger
from redback.result import RedbackResult


def _setup_plotting_result(model, model_kwargs, parameters, transient):
    """
    Helper function to setup the plotting result

    :param model: model string or model function
    :param model_kwargs: keyword arguments passed to the model
    :param parameters: parameters to plot
    :param transient: transient object
    :return: a tuple of model, parameters, and result
    """
    if isinstance(parameters, dict):
        parameters = pd.DataFrame.from_dict(parameters)
    parameters["log_likelihood"] = np.arange(len(parameters))
    if isinstance(model, str):
        model = redback.model_library.all_models_dict[model]
    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    meta_data['model_kwargs'] = model_kwargs or dict()
    res = RedbackResult(label="None", outdir="None",
                        search_parameter_keys=None,
                        fixed_parameter_keys=None,
                        constraint_parameter_keys=None, priors=None,
                        sampler_kwargs=dict(), injection_parameters=None,
                        meta_data=meta_data, posterior=parameters, samples=None,
                        nested_samples=None, log_evidence=0,
                        log_evidence_err=0, information_gain=0,
                        log_noise_evidence=0, log_bayes_factor=0,
                        log_likelihood_evaluations=0,
                        log_prior_evaluations=0, sampling_time=0, nburn=0,
                        num_likelihood_evaluations=0, walkers=0,
                        max_autocorrelation_time=0, use_ratio=False,
                        version=None)
    return model, parameters, res


def plot_lightcurve(transient, parameters, model, model_kwargs=None):
    """
    Plot a lightcurve for a given model and parameters

    :param transient: transient object
    :param parameters: parameters to plot
    :param model: model string or model function
    :param model_kwargs: keyword arguments passed to the model
    :return: plot_lightcurve
    """
    model, parameters, res = _setup_plotting_result(model, model_kwargs, parameters, transient)
    return res.plot_lightcurve(model=model, random_models=len(parameters), plot_max_likelihood=False)


def plot_multiband_lightcurve(transient, parameters, model, model_kwargs=None):
    """
    Plot a multiband lightcurve for a given model and parameters

    :param transient: transient object
    :param parameters: parameters to plot
    :param model: model string or model function
    :param model_kwargs: keyword arguments passed to the model
    :return: plot_multiband_lightcurve
    """
    model, parameters, res = _setup_plotting_result(model, model_kwargs, parameters, transient)
    return res.plot_multiband_lightcurve(model=model, random_models=len(parameters), plot_max_likelihood=False)


def plot_evolution_parameters(result, random_models=100):
    """
    Plot evolution parameters for a given evolving_magnetar result

    :param result: redback result
    :param random_models: number of random models to plot
    :return: fig and axes
    """
    logger.warning("This type of plot is only valid for evolving magnetar models")
    tmin = np.log10(np.min(result.metadata['time']))
    tmax = np.log10(np.max(result.metadata['time']))
    time = np.logspace(tmin, tmax, 100)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 10))
    for j in range(random_models):
        s = dict(result.posterior.iloc[np.random.randint(len(result.posterior))])
        s["output"] = "namedtuple"
        model = redback.model_library.all_models_dict["evolving_magnetar_only"]
        output = model(time, **s)
        nn = output.nn
        mu = output.mu
        alpha = output.alpha
        ax[0].plot(time, nn, "--", lw=1, color='red', alpha=2.5, zorder=-1)
        ax[1].plot(time, np.rad2deg(alpha), "--", lw=1, color='red', alpha=2.5, zorder=-1)
        ax[2].plot(time, mu, "--", lw=1, color='red', alpha=2.5, zorder=-1)
        ax[0].set_ylabel('braking index')
        ax[1].set_ylabel('inclination angle')
        ax[2].set_ylabel('magnetic moment')
    for x in range(3):
        ax[x].set_yscale('log')
        ax[x].set_xscale('log')
    fig.supxlabel(r"Time since burst [s]")
    return fig, ax
