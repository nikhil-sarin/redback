import inspect
import numpy as np
import os
import sys

import bilby
import pandas as pd

from . import grb as tools
from . import models as mm
from .analysis import find_path

dirname = os.path.dirname(__file__)
logger = bilby.core.utils.logger


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
        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        super(GRBGaussianLikelihood, self).__init__(parameters=dict())

    def log_likelihood(self):
        res = self.y - self.function(self.x, **self.parameters)
        return -0.5 * (np.sum((res / self.sigma) ** 2
                              + np.log(2 * np.pi * self.sigma ** 2)))


def fit_model(name, path, model, sampler='dynesty', nlive=3000, prior=False, walks=1000, truncate=True,
              use_photon_index_prior=False, truncate_method='prompt_time_error', luminosity_data=False, resume=True,
              save_format='json', **kwargs):
    """

    Parameters
    ----------
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
    :param luminosity_data:
    :param resume:
    :param save_format:
    :param kwargs: additional parameters that will be passed to the sampler
    :return: bilby result object, GRB data object
    """

    data = tools.SGRB(name, path)
    data.load_and_truncate_data(truncate=truncate, truncate_method=truncate_method, luminosity_data=luminosity_data)

    if not prior:
        priors = bilby.prior.PriorDict(filename=dirname + '/Priors/' + model + '.prior')
    else:
        priors = prior

    if use_photon_index_prior:
        if data.photon_index < 0.:
            logger.info('photon index for GRB', data.name, 'is negative. Using default prior on alpha_1')
            priors['alpha_1'] = bilby.prior.Uniform(-10, -0.5, 'alpha_1', latex_label=r'$\alpha_{1}$')
        else:
            priors['alpha_1'] = bilby.prior.Gaussian(mu=-(data.photon_index + 1), sigma=0.1,
                                                     latex_label=r'$\alpha_{1}$')

    function = None
    if model == 'magnetar_only':
        function = mm.magnetar_only

    if model == 'gw_magnetar':
        function = mm.gw_magnetar

    if model == 'full_magnetar':
        function = mm.full_magnetar

    if model == 'magnetic_dipole_magnetar':
        function = mm.full_magnetar

    if model == 'collapsing_magnetar':
        function = mm.collapsing_magnetar

    if model == 'general_magnetar':
        function = mm.general_magnetar

    if model == 'one_component_fireball':
        function = mm.one_component_fireball_model

    if model == 'two_component_fireball':
        function = mm.two_component_fireball_model

    if model == 'three_component_fireball':
        function = mm.three_component_fireball_model

    if model == 'four_component_fireball':
        function = mm.four_component_fireball_model

    if model == 'five_component_fireball':
        function = mm.five_component_fireball_model

    if model == 'six_component_fireball':
        function = mm.six_component_fireball_model

    if model == 'piecewise_radiative_losses':
        function = mm.piecewise_radiative_losses

    if model == 'radiative_losses':
        function = mm.radiative_losses

    if model == 'radiative_losses_smoothness':
        function = mm.radiative_losses_smoothness

    if model == 'radiative_only':
        function = mm.radiative_only

    if model == 'radiative_losses_mdr':
        function = mm.radiative_losses_mdr

    if model == 'collapsing_radiative_losses':
        function = mm.collapsing_radiative_losses

    outdir = path + '/GRB' + name + '/' + model
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if path == 'default':
        path = find_path(path)
    else:
        path = path
        df = pd.DataFrame({'time': data.time,
                           'Lum50': data.Lum50,
                           'Lum50_err_positive': data.Lum50_err[1, :],
                           'Lum50_err_negative': data.Lum50_err[0, :],
                           'time_err_negative': data.time_err[0, :],
                           'time_err_positive': data.time_err[1, :]})
        df.to_csv(outdir + "/data.txt", sep=',', index_label=False, index=False)

    if data.luminosity_data:
        if use_photon_index_prior:
            label = 'luminosity_photon_index'
        else:
            label = 'luminosity'
    else:
        if use_photon_index_prior:
            label = 'flux_photon_index'
        else:
            label = 'flux'

    likelihood = GRBGaussianLikelihood(x=data.time, y=data.Lum50, sigma=data.Lum50_err, function=function)

    result = bilby.run_sampler(likelihood, priors=priors, label=label, sampler=sampler, nlive=nlive,
                               outdir=outdir, plot=True, use_ratio=False, walks=walks, resume=resume,
                               maxmcmc=10 * walks,
                               nthreads=4, save_bounds=False, nsteps=nlive, nwalkers=walks, save=save_format, **kwargs)

    return result, data


if __name__ == "__main__":
    result, data = fit_model(name=sys.argv[1], path='GRBData', model=sys.argv[2], sampler='pymultinest', nlive=1000,
                             prior=False, walks=100)
