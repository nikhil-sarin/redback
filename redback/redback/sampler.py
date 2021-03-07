import os
import sys
from pathlib import Path
from inspect import getfullargspec

import bilby
import numpy as np
import pandas as pd

from . import grb as tools

from .result import RedbackResult
from .utils import find_path, logger
from .model_library import model_dict

dirname = os.path.dirname(__file__)


class GRBGaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function):
        """

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: array_like
            The standard deviation of the noise
        function:
            The python function to fit to the data
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.function = function
        parameters = getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        super(GRBGaussianLikelihood, self).__init__(parameters=dict())

    def log_likelihood(self):
        res = self.y - self.function(self.x, **self.parameters)
        return -0.5 * (np.sum((res / self.sigma) ** 2
                              + np.log(2 * np.pi * self.sigma ** 2)))


def fit_model(source_type, name, path, model, sampler='dynesty', nlive=3000, prior=None, walks=1000, truncate=True,
              use_photon_index_prior=False, truncate_method='prompt_time_error', data_mode='flux',
              resume=True, save_format='json', **kwargs):
    """

    Parameters
    ----------
    :param source_type: 'GRB', 'Supernova', 'TDE', 'Prompt', 'Kilonova'
    :param name: Telephone number of SGRB, e.g., GRB 140903A
    :param path: Path to the GRB folder which contains GRB data files, if using inbuilt data files then 'GRBData'
    :param model: String to indicate which model to fit to data
    :param sampler: String to indicate which sampler to use, default is dynesty
    and nested samplers are encouraged to allow evidence calculation
    :param nlive: number of live points
    :param prior: if Prior is true user needs to pass a dictionary with priors defined the bilby way
    :param walks: number of walkers
    :param truncate: flag to confirm whether to truncate the prompt emission data
    :param use_photon_index_prior: flag to turn off/on photon index prior and fits according to the curvature effect
    :param truncate_method: method of truncation
    :param data_mode: 'luminosity', 'flux', 'flux_density', depending on which kind of data will be accessed
    :param resume:
    :param save_format:
    :param kwargs: additional parameters that will be passed to the sampler
    :return: bilby result object, GRB data object
    """
    if source_type.upper() in ['GRB', 'SGRB', 'LGRB']:
        return _fit_grb(name, path, model, sampler='dynesty', nlive=3000, prior=None, walks=1000, truncate=True,
                        use_photon_index_prior=False, truncate_method='prompt_time_error', data_mode='flux',
                        resume=True, save_format='json', **kwargs)
    elif source_type.upper() in ['KILONOVA']:
        return _fit_kilonova(name=name, path=path, model=model, sampler='dynesty', nlive=3000, prior=None, walks=1000,
                             truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
                             data_mode='flux', resume=True, save_format='json', **kwargs)
    elif source_type.upper() in ['PROMPT']:
        return _fit_prompt(name=name, path=path, model=model, sampler='dynesty', nlive=3000, prior=None, walks=1000,
                           truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
                           data_mode='flux',
                           resume=True, save_format='json', **kwargs)
    elif source_type.upper() in ['SUPERNOVA']:
        return _fit_supernova(name=name, path=path, model=model, sampler='dynesty', nlive=3000, prior=None, walks=1000,
                              truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
                              data_mode='flux', resume=True, save_format='json', **kwargs)
    elif source_type.upper() in ['TDE']:
        return _fit_tde(name=name, path=path, model=model, sampler='dynesty', nlive=3000, prior=None, walks=1000,
                        truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
                        data_mode='flux',
                        resume=True, save_format='json', **kwargs)


def _fit_grb(name, path, model, sampler='dynesty', nlive=3000, prior=None, walks=1000, truncate=True,
             use_photon_index_prior=False, truncate_method='prompt_time_error', data_mode='flux',
             resume=True, save_format='json', **kwargs):
    data = tools.SGRB(name, path)
    data.load_and_truncate_data(truncate=truncate, truncate_method=truncate_method, data_mode=data_mode)

    if prior is None:
        prior = bilby.prior.PriorDict(filename=f"{dirname}/Priors/{model}.prior")

    if use_photon_index_prior:
        if data.photon_index < 0.:
            logger.info('photon index for GRB', data.name, 'is negative. Using default prior on alpha_1')
            prior['alpha_1'] = bilby.prior.Uniform(-10, -0.5, 'alpha_1', latex_label=r'$\alpha_{1}$')
        else:
            prior['alpha_1'] = bilby.prior.Gaussian(mu=-(data.photon_index + 1), sigma=0.1,
                                                    latex_label=r'$\alpha_{1}$')

    if isinstance(model, str):
        function = model_dict[model]
    else:
        function = model

    outdir = f"{find_path(path)}/GRB{name}/{model}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    label = ''
    if data.luminosity_data:
        df = pd.DataFrame({'time': data.time,
                           'Lum50': data.Lum50,
                           'Lum50_err_positive': data.Lum50_err[1, :],
                           'Lum50_err_negative': data.Lum50_err[0, :],
                           'time_err_negative': data.time_err[0, :],
                           'time_err_positive': data.time_err[1, :]})
        df.to_csv(outdir + "/data.txt", sep=',', index_label=False, index=False)
        likelihood = GRBGaussianLikelihood(x=data.time, y=data.Lum50, sigma=data.Lum50_err, function=function)
    elif data.flux_data:
        df = pd.DataFrame({'time': data.time,
                           'flux': data.flux,
                           'flux_positive': data.flux_err[1, :],
                           'flux_negative': data.flux_err[0, :],
                           'time_err_negative': data.time_err[0, :],
                           'time_err_positive': data.time_err[1, :]})
        df.to_csv(outdir + "/data.txt", sep=',', index_label=False, index=False)
        likelihood = GRBGaussianLikelihood(x=data.time, y=data.flux, sigma=data.flux_err, function=function)
    elif data.fluxdensity_data:
        df = pd.DataFrame({'time': data.time,
                           'flux_density': data.flux_density,
                           'flux_density_positive': data.flux_density_err[1, :],
                           'flux_density_negative': data.flux_density_err[0, :],
                           'time_err_negative': data.time_err[0, :],
                           'time_err_positive': data.time_err[1, :]})
        df.to_csv(outdir + "/data.txt", sep=',', index_label=False, index=False)
        likelihood = GRBGaussianLikelihood(x=data.time, y=data.flux_density, sigma=data.flux_density_err,
                                           function=function)
    else:
        raise ValueError("Not a valid data switch")
    label += data_mode

    if use_photon_index_prior:
        label += '_photon_index'

    meta_data = dict(model=model, transient=data, path=path, use_photon_index_prior=use_photon_index_prior,
                     truncate=truncate, truncate_method=truncate_method, save_format=save_format)

    result = bilby.run_sampler(likelihood=likelihood, priors=prior, label=label, sampler=sampler, nlive=nlive,
                               outdir=outdir, plot=True, use_ratio=False, walks=walks, resume=resume,
                               maxmcmc=10 * walks, result_class=RedbackResult, meta_data=meta_data,
                               nthreads=4, save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)

    return result, data


def _fit_kilonova(**kwargs):
    pass


def _fit_prompt(**kwargs):
    pass


def _fit_supernova(**kwargs):
    pass


def _fit_tde(**kwargs):
    pass


if __name__ == "__main__":
    result, data = fit_model(name=sys.argv[1], path='GRBData', model=sys.argv[2], sampler='pymultinest', nlive=1000,
                             prior=False, walks=100)
