import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Union

import bilby

import redback.get_data
from redback.likelihoods import GaussianLikelihood, PoissonLikelihood
from redback.model_library import all_models_dict
from redback.result import RedbackResult
from redback.utils import logger
from redback.transient.afterglow import Afterglow
from redback.transient.prompt import PromptTimeSeries
from redback.transient.transient import OpticalTransient, Transient, Spectrum


dirname = os.path.dirname(__file__)


def fit_model(
        transient: redback.transient.transient.Transient, model: Union[callable, str], outdir: str = None,
        label: str = None, sampler: str = "dynesty", nlive: int = 2000, prior: dict = None, walks: int = 200,
        truncate: bool = True, use_photon_index_prior: bool = False, truncate_method: str = "prompt_time_error",
        resume: bool = True, save_format: str = "json", model_kwargs: dict = None, plot=True, **kwargs)\
        -> redback.result.RedbackResult:
    """
    Main function for fitting models to transient data using Bayesian inference.

    :param transient: The transient to be fitted
    :param model: Name of the model to fit to data or a function.
    :param outdir: Output directory. Will default to a sensible structure if not given.
    :param label: Result file labels. Will use the model name if not given.
    :param sampler: The sampling backend. Nested samplers are encouraged to allow evidence calculation.
                    (Default value = 'dynesty')
    :param nlive: Number of live points.
    :param prior: Priors to use during sampling. If not given, we use the default priors for the given model.
    :param walks: Number of `dynesty` random walks.
    :param truncate: Flag to confirm whether to truncate the prompt emission data
    :param use_photon_index_prior: flag to turn off/on photon index prior and fits according to the curvature effect
    :param truncate_method: method of truncation
    :param resume: Whether to resume the run from a checkpoint if available.
    :param save_format: The format to save the result in. (Default value = 'json'_
    :param model_kwargs: Additional keyword arguments for the model.
    :param clean: If True, rerun the fitting, if false try to use previous results in the output directory.
    :param plot: If True, create corner and lightcurve plot
    :param kwargs: Additional parameters that will be passed to the sampler via bilby
    :return: Redback result object, transient specific data object

    **Example Usage:**

    .. code-block:: python

        import redback

        # Download and load supernova data
        redback.get_data.get_open_access_data(transient='SN2011fe', transient_type='supernova')
        supernova = redback.transient.Supernova.from_open_access_catalogue(
            name='SN2011fe',
            data_mode='magnitude'
        )

        # Fit an Arnett model with default priors
        model = 'arnett'
        result = redback.fit_model(
            transient=supernova,
            model=model,
            nlive=1000,
            plot=True
        )

        # Access posterior samples
        print(result.posterior)

        # Plot the corner plot
        result.plot_corner()

        # Fit with custom priors
        import bilby
        priors = redback.priors.get_priors(model)
        priors['redshift'] = bilby.prior.Uniform(0.0, 0.1, 'redshift')

        result = redback.fit_model(
            transient=supernova,
            model=model,
            prior=priors,
            nlive=2000,
            sampler='dynesty'
        )

        # Fit a GRB afterglow
        redback.get_data.get_bat_xrt_afterglow_data('GRB123456')
        afterglow = redback.transient.Afterglow.from_swift_grb(
            name='GRB123456',
            data_mode='flux'
        )

        result = redback.fit_model(
            transient=afterglow,
            model='tophat',
            model_kwargs={'output_format': 'flux'},
            nlive=1500
        )

        # Use a custom model function
        def my_custom_model(time, amplitude, timescale, **kwargs):
            return amplitude * np.exp(-time / timescale)

        result = redback.fit_model(
            transient=supernova,
            model=my_custom_model,
            prior=my_priors,
            model_kwargs={'output_format': 'magnitude'}
        )
    """
    if isinstance(model, str):
        modelname = model
        model = all_models_dict[model]

    if transient.data_mode in ["flux_density", "magnitude"]:
        if model_kwargs["output_format"] != transient.data_mode:
            raise ValueError(
                f"Transient data mode {transient.data_mode} is inconsistent with "
                f"output format {model_kwargs['output_format']}. These should be the same.")

    if prior is None:
        prior = bilby.prior.PriorDict(filename=f"{dirname}/priors/{modelname}.prior")
        logger.warning(f"No prior given. Using default priors for {modelname}")
    else:
        prior = prior
    outdir = outdir or f"{transient.directory_structure.directory_path}/{model.__name__}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    label = label or transient.name

    if isinstance(transient, Spectrum):
        return _fit_spectrum(transient=transient, model=model, outdir=outdir, label=label, sampler=sampler,
                             nlive=nlive, prior=prior, walks=walks,
                             resume=resume, save_format=save_format, model_kwargs=model_kwargs,
                             plot=plot, **kwargs)

    elif isinstance(transient, Afterglow):
        return _fit_grb(
            transient=transient, model=model, outdir=outdir, label=label, sampler=sampler, nlive=nlive, prior=prior,
            walks=walks, use_photon_index_prior=use_photon_index_prior, resume=resume, save_format=save_format,
            model_kwargs=model_kwargs, truncate=truncate, truncate_method=truncate_method, plot=plot, **kwargs)
    elif isinstance(transient, PromptTimeSeries):
        return _fit_prompt(
            transient=transient, model=model, outdir=outdir, label=label, sampler=sampler, nlive=nlive, prior=prior,
            walks=walks, resume=resume, save_format=save_format, model_kwargs=model_kwargs, plot=plot, **kwargs)
    elif isinstance(transient, OpticalTransient):
        return _fit_optical_transient(
            transient=transient, model=model, outdir=outdir, label=label, sampler=sampler, nlive=nlive, prior=prior,
            walks=walks, truncate=truncate, use_photon_index_prior=use_photon_index_prior,
            truncate_method=truncate_method, resume=resume, save_format=save_format, model_kwargs=model_kwargs,
            plot=plot, **kwargs)
    elif isinstance(transient, Transient):
        return _fit_optical_transient(
            transient=transient, model=model, outdir=outdir, label=label, sampler=sampler, nlive=nlive, prior=prior,
            walks=walks, truncate=truncate, use_photon_index_prior=use_photon_index_prior,
            truncate_method=truncate_method, resume=resume, save_format=save_format, model_kwargs=model_kwargs,
            plot=plot, **kwargs)
    else:
        raise ValueError(f'Source type {transient.__class__.__name__} not known')

def _fit_spectrum(transient, model, outdir, label, likelihood=None, sampler='dynesty', nlive=3000, prior=None, walks=1000,
                  resume=True, save_format='json', model_kwargs=None, plot=True, **kwargs):
    x, y, y_err = transient.angstroms, transient.flux_density, transient.flux_density_err

    if likelihood is None:
        likelihood = GaussianLikelihood(x=x, y=y, sigma=y_err, function=model, kwargs=model_kwargs)
        logger.info("No likelihood provided, using standard GaussianLikelihood")
    else:
        logger.info("Likelihood provided, using custom likelihood {}".format(likelihood.__class__.__name__))
        likelihood = likelihood

    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    model_kwargs = redback.utils.check_kwargs_validity(model_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    result = None
    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = result or bilby.run_sampler(
        likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
        outdir=outdir, plot=plot, use_ratio=False, walks=walks, resume=resume,
        maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
        save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)
    plt.close('all')
    if plot:
        result.plot_spectrum(model=model)
    return result

def _fit_grb(transient, model, outdir, label, likelihood=None, sampler='dynesty', nlive=3000, prior=None, walks=1000,
             use_photon_index_prior=False, resume=True, save_format='json', model_kwargs=None, plot=True, **kwargs):
    if use_photon_index_prior:
        label += '_photon_index'
        if transient.photon_index < 0.:
            logger.info('photon index for GRB', transient.name, 'is negative. Using default prior on alpha_1')
            prior['alpha_1'] = bilby.prior.Uniform(-10, -0.5, 'alpha_1', latex_label=r'$\alpha_{1}$')
        else:
            prior['alpha_1'] = bilby.prior.Gaussian(mu=-(transient.photon_index + 1), sigma=0.1,
                                                    latex_label=r'$\alpha_{1}$')

    if any([transient.flux_data, transient.magnitude_data, transient.flux_density_data]):
        x, x_err, y, y_err = transient.get_filtered_data()
    else:
        x, x_err, y, y_err = transient.x, transient.x_err, transient.y, transient.y_err

    if likelihood is None:
        likelihood = GaussianLikelihood(x=x, y=y, sigma=y_err, function=model, kwargs=model_kwargs)
        logger.info("No likelihood provided, using standard GaussianLikelihood")
    else:
        logger.info("Likelihood provided, using custom likelihood {}".format(likelihood.__class__.__name__))
        likelihood = likelihood

    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    model_kwargs = redback.utils.check_kwargs_validity(model_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    result = None
    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = result or bilby.run_sampler(
        likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
        outdir=outdir, plot=plot, use_ratio=False, walks=walks, resume=resume,
        maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
        save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)
    plt.close('all')
    if plot:
        result.plot_lightcurve(model=model)
    return result


def _fit_optical_transient(transient, model, outdir, label, likelihood=None, sampler='dynesty', nlive=3000, prior=None,
                           walks=1000, resume=True, save_format='json', model_kwargs=None, plot=True, **kwargs):

    if any([transient.flux_data, transient.magnitude_data, transient.flux_density_data]):
        x, x_err, y, y_err = transient.get_filtered_data()
    else:
        x, x_err, y, y_err = transient.x, transient.x_err, transient.y, transient.y_err

    if likelihood is None:
        likelihood = GaussianLikelihood(x=x, y=y, sigma=y_err, function=model, kwargs=model_kwargs)
        logger.info("No likelihood provided, using standard GaussianLikelihood")
    else:
        logger.info("Likelihood provided, using custom likelihood {}".format(likelihood.__class__.__name__))
        likelihood = likelihood

    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    model_kwargs = redback.utils.check_kwargs_validity(model_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    result = None
    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = result or bilby.run_sampler(
        likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
        outdir=outdir, plot=plot, use_ratio=False, walks=walks, resume=resume,
        maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
        save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)
    plt.close('all')
    if plot:
        result.plot_lightcurve(model=model)
    return result


def _fit_prompt(transient, model, outdir, label, likelihood=None, integrated_rate_function=True, sampler='dynesty', nlive=3000,
                prior=None, walks=1000, resume=True, save_format='json',
                model_kwargs=None, plot=True, **kwargs):

    likelihood = likelihood or PoissonLikelihood(
        time=transient.x, counts=transient.y, dt=transient.bin_size, function=model,
        integrated_rate_function=integrated_rate_function, kwargs=model_kwargs)

    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    model_kwargs = redback.utils.check_kwargs_validity(model_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    result = None
    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = result or bilby.run_sampler(
        likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
        outdir=outdir, plot=False, use_ratio=False, walks=walks, resume=resume,
        maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
        save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)
    plt.close('all')
    if plot:
        result.plot_lightcurve(model=model)
    return result
