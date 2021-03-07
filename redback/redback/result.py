import warnings
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.result import Result

from . import grb as tools
from . import models as mm
from .model_library import model_dict
from .utils import MetaDataAccessor

warnings.simplefilter(action='ignore')


class RedbackResult(Result):

    model = MetaDataAccessor('model')
    grb = MetaDataAccessor('grb')
    path = MetaDataAccessor('path')
    use_photon_index_prior = MetaDataAccessor('use_photon_index_prior')
    truncate = MetaDataAccessor('truncate')
    truncate_method = MetaDataAccessor('truncate_method')
    luminosity_data = MetaDataAccessor('luminosity_data')
    save_format = MetaDataAccessor('save_format')

    def __init__(self, label='no_label', outdir='.', sampler=None, search_parameter_keys=None,
                 fixed_parameter_keys=None, constraint_parameter_keys=None, priors=None, sampler_kwargs=None,
                 injection_parameters=None, meta_data=None, posterior=None, samples=None, nested_samples=None,
                 log_evidence=np.nan, log_evidence_err=np.nan, log_noise_evidence=np.nan, log_bayes_factor=np.nan,
                 log_likelihood_evaluations=None, log_prior_evaluations=None, sampling_time=None, nburn=None,
                 num_likelihood_evaluations=None, walkers=None, max_autocorrelation_time=None, use_ratio=None,
                 parameter_labels=None, parameter_labels_with_unit=None, gzip=False, version=None):
        super().__init__(label, outdir, sampler, search_parameter_keys, fixed_parameter_keys, constraint_parameter_keys,
                         priors, sampler_kwargs, injection_parameters, meta_data, posterior, samples, nested_samples,
                         log_evidence, log_evidence_err, log_noise_evidence, log_bayes_factor,
                         log_likelihood_evaluations, log_prior_evaluations, sampling_time, nburn,
                         num_likelihood_evaluations, walkers, max_autocorrelation_time, use_ratio, parameter_labels,
                         parameter_labels_with_unit, gzip, version)

    @property
    def plot_directory_structure(self):
        """
        :param data: instantiated GRB class
        :param model: model name
        :return: plot_base_directory
        """
        # set up plotting directory structure
        directory = f"{self.grb.path}/GRB{self.grb.name}"
        plots_base_directory = f"{directory}/{self.model}/plots/"
        Path(plots_base_directory).mkdir(parents=True, exist_ok=True)
        return plots_base_directory

    def save_to_file(self, filename=None, overwrite=False, outdir=None,
                     extension='json', gzip=False):
        pass

    def plot_corner(self, parameters=None, priors=None, titles=True, save=True, dpi=300, **kwargs):
        filename = self.plot_directory_structure + self.model
        if self.use_photon_index_prior:
            filename += '_photon_index_corner.png'
        else:
            filename += '_corner.png'

        super().plot_corner(parameters=parameters, priors=priors, filename=filename, titles=titles,
                            save=save, dpi=dpi, **kwargs)
        if kwargs.get('plot_show', False):
            plt.show()

    def plot_lightcurve(self, model, axes=None, plot_save=True,
                        plot_show=True, random_models=1000, plot_magnetar=False):
        """
        :param model:
        :param axes:
        :param plot_save:
        :param plot_show:
        :param random_models:
        :param plot_magnetar:

        plots the lightcurve
        GRB is the telephone number of the GRB
        model = model to plot
        path = path to GRB folder

        """
        max_l = dict(self.posterior.sort_values(by=['log_likelihood']).iloc[-1])

        for j in range(int(random_models)):
            params = dict(self.posterior.iloc[np.random.randint(len(self.posterior))])
            plot_models(parameters=params, axes=axes, alpha=0.05, lw=2, colour='r', model=model,
                        plot_magnetar=plot_magnetar)

        # plot max likelihood
        plot_models(parameters=max_l, axes=axes, alpha=0.65, lw=2, colour='b', model=model, plot_magnetar=plot_magnetar)

        self.grb.plot_data(axes=axes)

        label = 'lightcurve'
        if self.use_photon_index_prior:
            label = f"_photon_index_{label}"

        if plot_save:
            plt.savefig(f"{self.plot_directory_structure}{model}{label}.png")

        if plot_show:
            plt.show()


def read_in_grb_result(model, grb, path='.', truncate=True, use_photon_index_prior=False,
                       truncate_method='prompt_time_error',
                       luminosity_data=False, save_format='json'):
    """
    :param model: model to analyse
    :param grb: telephone number of GRB
    :param path: path to GRB
    :param truncate: flag to truncate or not
    :param use_photon_index_prior:
    :param truncate_method:
    :param luminosity_data:
    :param save_format:
    :return: bilby result object and data object
    """
    result_path = f"{path}/GRB{grb}/{model}/"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    if luminosity_data:
        label = 'luminosity'
    else:
        label = 'flux'

    if use_photon_index_prior:
        label += '_photon_index'

    result = bilby.result.read_in_result(filename=f"{result_path}{label}_result.{save_format}")
    data = tools.GRB.from_path_and_grb_with_truncation(
        grb=grb, truncate=truncate, path=path, truncate_method=truncate_method,
        luminosity_data=luminosity_data)
    return result, data


def plot_models(parameters, model, plot_magnetar, axes=None, colour='r', alpha=1.0, ls='-', lw=4):
    """
    plot the models
    parameters: dictionary of parameters - 1 set of Parameters
    model: model name
    """
    time = np.logspace(-4, 7, 100)
    ax = axes or plt.gca()

    lightcurve = model_dict[model]
    magnetar_models = ['evolving_magnetar', 'evolving_magnetar_only', 'piecewise_radiative_losses',
                       'radiative_losses', 'radiative_losses_mdr', 'radiative_losses_smoothness', 'radiative_only']
    if model in magnetar_models and plot_magnetar:
        if model == 'radiative_losses_mdr':
            magnetar = mm.magnetar_only(time, nn=3., **parameters)
        else:
            magnetar = mm.magnetar_only(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle='--')
    ax.plot(time, lightcurve, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32)


def calculate_bf(model1, model2, grb, path='.', use_photon_index_prior=False, luminosity_data=False,
                 save_format='json'):
    model1, data = read_in_grb_result(model=model1, grb=grb, path=path, use_photon_index_prior=use_photon_index_prior,
                                      luminosity_data=luminosity_data, save_format=save_format)
    model2, data = read_in_grb_result(model=model2, grb=grb, path=path, use_photon_index_prior=use_photon_index_prior,
                                      luminosity_data=luminosity_data, save_format=save_format)
    log_bf = model1.log_evidence - model2.log_evidence
    return log_bf
