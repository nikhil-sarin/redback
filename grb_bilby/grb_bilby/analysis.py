import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path

import bilby

from . import grb as tools
from . import models as mm
from . import model_dict
from .utils import find_path


warnings.simplefilter(action='ignore')


def read_result(model, grb, path='.', truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
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


def plot_directory_structure(data, model):
    """
    :param data: instantiated GRB class
    :param model: model name
    :return: plot_base_directory
    """
    # set up plotting directory structure
    directory = f"{data.path}/GRB{data.name}"
    plots_base_directory = f"{directory}/{model}/plots/"
    Path(plots_base_directory).mkdir(parents=True, exist_ok=True)
    return plots_base_directory


def plot_lightcurve(grb, model, path='GRBData',
                    axes=None,
                    plot_save=True,
                    plot_show=True, random_models=1000,
                    truncate=True, use_photon_index_prior=False,
                    truncate_method='prompt_time_error',
                    luminosity_data=False, save_format='json',
                    plot_magnetar=False):
    """
    :param grb:
    :param model:
    :param path:
    :param axes:
    :param plot_save:
    :param plot_show:
    :param random_models:
    :param truncate:
    :param use_photon_index_prior:
    :param truncate_method:
    :param luminosity_data:
    :param save_format:
    :param plot_magnetar:

    plots the lightcurve
    GRB is the telephone number of the GRB
    model = model to plot
    path = path to GRB folder

    """
    result, data = read_result(model=model, grb=grb, path=path, truncate=truncate,
                               use_photon_index_prior=use_photon_index_prior, truncate_method=truncate_method,
                               luminosity_data=luminosity_data,
                               save_format=save_format)

    plots_base_directory = plot_directory_structure(data=data, model=model)

    max_l = dict(result.posterior.sort_values(by=['log_likelihood']).iloc[-1])

    for j in range(int(random_models)):
        params = dict(result.posterior.iloc[np.random.randint(len(result.posterior))])
        plot_models(parameters=params, axes=axes, alpha=0.05, lw=2, colour='r', model=model,
                    plot_magnetar=plot_magnetar)

    # plot max likelihood
    plot_models(parameters=max_l, axes=axes, alpha=0.65, lw=2, colour='b', model=model, plot_magnetar=plot_magnetar)

    data.plot_data(axes=axes)

    label = 'lightcurve'
    if use_photon_index_prior:
        label = f"_photon_index_{label}"

    if plot_save:
        plt.savefig(f"{plots_base_directory}{model}{label}.png")

    if plot_show:
        plt.show()


def calculate_bf(model1, model2, grb, path='.', use_photon_index_prior=False, luminosity_data=False,
                 save_format='json'):
    model1, data = read_result(model=model1, grb=grb, path=path, use_photon_index_prior=use_photon_index_prior,
                               luminosity_data=luminosity_data, save_format=save_format)
    model2, data = read_result(model=model2, grb=grb, path=path, use_photon_index_prior=use_photon_index_prior,
                               luminosity_data=luminosity_data, save_format=save_format)
    log_bf = model1.log_evidence - model2.log_evidence
    return log_bf


def plot_corner(grb, model, path='GRBData',
                plot_show=True, truncate=True, use_photon_index_prior=False,
                truncate_method='prompt_time_error',
                luminosity_data=False, save_format='json'):
    result, data = read_result(model=model, grb=grb, path=path, truncate=truncate,
                               use_photon_index_prior=use_photon_index_prior, truncate_method=truncate_method,
                               luminosity_data=luminosity_data,
                               save_format=save_format)
    plots_base_directory = plot_directory_structure(data=data, model=model)

    if use_photon_index_prior:
        filename = plots_base_directory + model + '_photon_index_corner.png'
    else:
        filename = plots_base_directory + model + '_corner.png'

    result.plot_corner(filename=filename)
    if plot_show:
        plt.show()
    return None
