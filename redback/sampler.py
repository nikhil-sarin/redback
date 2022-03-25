import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Union

import bilby

import redback.get_data
from redback.likelihoods import GaussianLikelihood, GRBGaussianLikelihood, PoissonLikelihood
from redback.model_library import all_models_dict
from redback.result import RedbackResult
from redback.utils import logger
from redback.transient.afterglow import Afterglow
from redback.transient.kilonova import Kilonova
from redback.transient.prompt import PromptTimeSeries
from redback.transient.supernova import Supernova
from redback.transient.tde import TDE

dirname = os.path.dirname(__file__)


def fit_model(transient, model, outdir=None, sampler='dynesty', nlive=2000, prior=None,
              walks=200, truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
              resume=True, save_format='json', model_kwargs=None, **kwargs):
    """
    :param transient: The transient to be fitted
    :type transient: redback.transient.transient.Transient
    :param model: Name of the model to fit to data or a function.
    :type model: Union[callable, str]
    :param sampler: The sampling backend. Nested samplers are encouraged to allow evidence calculation.
                    (Default value = 'dynesty')
    :type sampler: str
    :param nlive: Number of live points.
    :type nlive: int
    :param prior: Priors to use during sampling. If not given, we use the default priors for the given model.
    :type prior: [None, dict, bilby.prior.PriorDict]
    :param walks: Number of `dynesty` random walks.
    :type walks: int
    :param truncate: Flag to confirm whether to truncate the prompt emission data
    :type truncate: bool
    :param use_photon_index_prior: flag to turn off/on photon index prior and fits according to the curvature effect
    :type use_photon_index_prior: bool
    :param truncate_method: method of truncation
    :type truncate_method: str
    :param resume: Whether to resume the run from a checkpoint if available.
    :type resume: bool
    :param save_format: The format to save the result in. (Default value = 'json'_
    :type save_format: str
    :param kwargs: Additional parameters that will be passed to the sampler
    :type kwargs: None
    :return: Redback result object, transient specific data object
    :rtype: redback.result.RedbackResult
    """
    if isinstance(model, str):
        model = all_models_dict[model]

    if transient.data_mode in ["flux_density", "magnitude"]:
        if model_kwargs["output_format"] != transient.data_mode:
            raise ValueError(
                f"Transient data mode {transient.data_mode} is inconsistent with "
                f"output format {model_kwargs['output_format']}. These should be the same.")

    if prior is None:
        prior = bilby.prior.PriorDict(filename=f"{dirname}/Priors/{model}.prior")

    if isinstance(transient, Afterglow):
        return _fit_grb(transient=transient, model=model, outdir=outdir, sampler=sampler, nlive=nlive, prior=prior,
                        walks=walks, use_photon_index_prior=use_photon_index_prior, resume=resume,
                        save_format=save_format, model_kwargs=model_kwargs, truncate=truncate,
                        truncate_method=truncate_method, **kwargs)
    elif isinstance(transient, Kilonova):
        return _fit_kilonova(transient=transient, model=model, outdir=outdir, sampler=sampler, nlive=nlive, prior=prior,
                             walks=walks, resume=resume, save_format=save_format, model_kwargs=model_kwargs,
                             truncate=truncate, use_photon_index_prior=use_photon_index_prior,
                             truncate_method=truncate_method, **kwargs)
    elif isinstance(transient, PromptTimeSeries):
        return _fit_prompt(transient=transient, model=model, outdir=outdir, sampler=sampler, nlive=nlive,
                           prior=prior, walks=walks, use_photon_index_prior=use_photon_index_prior, resume=resume,
                           save_format=save_format, model_kwargs=model_kwargs, **kwargs)
    elif isinstance(transient, Supernova):
        return _fit_kilonova(transient=transient, model=model, outdir=outdir, sampler=sampler, nlive=nlive,
                              prior=prior, walks=walks, truncate=truncate,
                              use_photon_index_prior=use_photon_index_prior, truncate_method=truncate_method,
                              resume=resume, save_format=save_format, model_kwargs=model_kwargs,
                              **kwargs)
    elif isinstance(transient, TDE):
        return _fit_kilonova(transient=transient, model=model, outdir=outdir, sampler=sampler, nlive=nlive,
                        prior=prior, walks=walks, truncate=truncate, use_photon_index_prior=use_photon_index_prior,
                        truncate_method=truncate_method,
                        resume=resume, save_format=save_format, model_kwargs=model_kwargs, **kwargs)
    else:
        raise ValueError(f'Source type {transient.__class__.__name__} not known')


def _fit_grb(transient, model, outdir=None, label=None, sampler='dynesty', nlive=3000, prior=None, walks=1000,
             use_photon_index_prior=False, resume=True, save_format='json', model_kwargs=None, **kwargs):
    if use_photon_index_prior:
        if transient.photon_index < 0.:
            logger.info('photon index for GRB', transient.name, 'is negative. Using default prior on alpha_1')
            prior['alpha_1'] = bilby.prior.Uniform(-10, -0.5, 'alpha_1', latex_label=r'$\alpha_{1}$')
        else:
            prior['alpha_1'] = bilby.prior.Gaussian(mu=-(transient.photon_index + 1), sigma=0.1,
                                                    latex_label=r'$\alpha_{1}$')
    if outdir is None:
        outdir = f"{transient.directory_structure.directory_path}/{model.__name__}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    label = kwargs.get("label", transient.name)
    if use_photon_index_prior:
        label += '_photon_index'

    if transient.flux_density_data or transient.magnitude_data:
        x, x_err, y, y_err = transient.get_filtered_data()
    else:
        x, x_err, y, y_err = transient.x, transient.x_err, transient.y, transient.y_err

    likelihood = kwargs.get('likelihood', GaussianLikelihood(x=x, y=y, sigma=y_err, function=model, kwargs=model_kwargs))

    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = bilby.run_sampler(likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
                               outdir=outdir, plot=True, use_ratio=False, walks=walks, resume=resume,
                               maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
                               nthreads=4, save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)
    plt.close('all')
    return result


def _fit_kilonova(transient, model, outdir=None, sampler='dynesty', nlive=3000, prior=None, walks=1000,
                  resume=True, save_format='json', model_kwargs=None, **kwargs):

    if outdir is None:
        outdir, _, _ = redback.get_data.directory.open_access_directory_structure(transient=transient.name,
                                                                                  transient_type=transient.__class__.__name__)
        outdir = f"{outdir}/{model.__name__}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    label = kwargs.get("label", transient.name)

    if transient.flux_density_data or transient.magnitude_data:
        x, x_err, y, y_err = transient.get_filtered_data()
    else:
        x, x_err, y, y_err = transient.x, transient.x_err, transient.y, transient.y_err

    likelihood = kwargs.get('likelihood', GaussianLikelihood(x=x, y=y, sigma=y_err, function=model, kwargs=model_kwargs))

    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = bilby.run_sampler(likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
                               outdir=outdir, plot=True, use_ratio=False, walks=walks, resume=resume,
                               maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
                               nthreads=4, save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)
    plt.close('all')
    return result


def _fit_prompt(transient, model, outdir, integrated_rate_function=True, sampler='dynesty', nlive=3000,
                prior=None, walks=1000, use_photon_index_prior=False, resume=True, save_format='json',
                model_kwargs=None, **kwargs):


    outdir = f"{outdir}/GRB{name}/{model.__name__}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    label = kwargs.get("label", transient.name)
    if use_photon_index_prior:
        label += '_photon_index'

    likelihood = PoissonLikelihood(time=transient.x, counts=transient.y,
                                   dt=transient.bin_size, function=model,
                                   integrated_rate_function=integrated_rate_function, kwargs=model_kwargs)

    meta_data = dict(model=model.__name__, transient_type="prompt")
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    meta_data['model_kwargs'] = model_kwargs

    if not kwargs.get("clean", False):
        try:
            result = redback.result.read_in_result(
                outdir=outdir, label=label, extension=kwargs.get("extension", "json"), gzip=kwargs.get("gzip", False))
            plt.close('all')
            return result
        except Exception:
            pass

    result = bilby.run_sampler(likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
                               outdir=outdir, plot=False, use_ratio=False, walks=walks, resume=resume,
                               maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
                               nthreads=4, save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)

    plt.close('all')
    return result


def _fit_supernova(**kwargs):
    plt.close('all')
    pass


def _fit_tde(**kwargs):
    plt.close('all')
    pass
