import os
import bilby
import matplotlib.pyplot as plt
import numpy as np
from grb_bilby.processing import GRB as tools
from grb_bilby.models import models as mm

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'paper.mplstyle')
plt.style.use(filename)
import warnings
warnings.simplefilter(action='ignore')


def find_path(path):
    if path == 'default':
        data_dir = os.path.join(dirname, '../data/GRBData')
    else:
        data_dir = path
    return data_dir


def load_data(GRB, path='GRBData', truncate=True, truncate_method='prompt_time_error', luminosity_data=False):
    """
    :param GRB: telephone number
    :param path: default for package data, or path to GRBData folder
    :param truncate: method of truncation/True/False
    :return: data class
    """
    data_dir = find_path(path)
    data = tools.SGRB(name=GRB, path=data_dir)
    data.load_and_truncate_data(truncate=truncate, truncate_method=truncate_method,luminosity_data=luminosity_data)
    return data


def read_result(model, GRB, path='.', truncate=True, use_photon_index_prior=False, truncate_method='prompt_time_error',
                luminosity_data = False, save_format='json'):
    """
    :param model: model to analyse
    :param GRB: telephone number of GRB
    :param path: path to GRB
    :param truncate: flag to truncate or not
    :return: bilby result object and data object
    """
    result_path = path + '/GRB' + GRB + '/' + model + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if save_format == 'hdf5':
        file_format = '.hdf5'
    else:
        file_format = '.json'

    if luminosity_data:
        if use_photon_index_prior:
            result = bilby.result.read_in_result(filename=result_path + 'luminosity_photon_index_result'+file_format)
        if use_photon_index_prior == False:
            result = bilby.result.read_in_result(filename=result_path + 'luminosity_result'+file_format)

    if luminosity_data == False:
        if use_photon_index_prior == True:
            result = bilby.result.read_in_result(filename=result_path + 'flux_photon_index_result'+file_format)
        if use_photon_index_prior == False:
            result = bilby.result.read_in_result(filename=result_path + 'flux_result'+file_format)

    data = load_data(GRB=GRB, truncate=truncate, path=path, truncate_method=truncate_method, luminosity_data=luminosity_data)

    return result, data


def plot_data(GRB, path, truncate, truncate_method='prompt_time_error', axes=None, colour='k',luminosity_data=False):
    """
    plots the data
    GRB is the telephone number of the GRB
    :param GRB:
    :param path:
    :param truncate:
    :param axes:
    :param colour:
    """
    ax = axes or plt.gca()
    data = load_data(GRB=GRB, path=path, truncate=truncate, truncate_method=truncate_method,luminosity_data=luminosity_data)
    tt = data.time
    tt_err = data.time_err
    Lum50 = data.Lum50
    Lum50_err = data.Lum50_err

    ax.errorbar(tt, Lum50,
                xerr=[tt_err[1, :], tt_err[0, :]],
                yerr=[Lum50_err[1, :], Lum50_err[0, :]],
                fmt='x', c=colour, ms=1, elinewidth=2, capsize=0.)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(0.5 * tt[0], 2 * (tt[-1] + tt_err[0, -1]))
    ax.set_ylim(0.5 * min(Lum50), 2. * np.max(Lum50))

    ax.annotate('GRB' + data.name, xy=(0.95, 0.9), xycoords='axes fraction',
                horizontalalignment='right',size=20)

    ax.set_xlabel(r'time since burst [s]')
    if data.luminosity_data == True:
        ax.set_ylabel(r'Luminosity [$10^{50}$ erg s$^{-1}$]')
    else:
        ax.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]')
    ax.tick_params(axis='x', pad=10)

    if axes == None:
        plt.tight_layout()
    plt.grid(b=None)


def plot_models(parameters, model, axes=None, colour='r', alpha=1.0, ls='-', lw=4):
    '''
    plot the models
    parameters: dictionary of parameters - 1 set of Parameters
    model: model name
    '''
    time = np.logspace(-4, 7, 100)
    ax = axes or plt.gca()

    if model == 'magnetar_only':
        lightcurve = mm.magnetar_only(time, **parameters)

    if model == 'full_magnetar':
        lightcurve = mm.full_magnetar(time, **parameters)

    if model == 'general_magnetar':
        lightcurve = mm.general_magnetar(time, **parameters)

    if model == 'collapsing_magnetar':
        lightcurve = mm.collapsing_magnetar(time, **parameters)

    if model == 'one_component_fireball':
        lightcurve = mm.one_component_fireball_model(time, **parameters)

    if model == 'two_component_fireball':
        lightcurve = mm.two_component_fireball_model(time, **parameters)

    if model == 'three_component_fireball':
        lightcurve = mm.three_component_fireball_model(time, **parameters)

    if model == 'four_component_fireball':
        lightcurve = mm.four_component_fireball_model(time, **parameters)

    if model == 'five_component_fireball':
        lightcurve = mm.five_component_fireball_model(time, **parameters)

    if model == 'six_component_fireball':
        lightcurve = mm.six_component_fireball_model(time, **parameters)

    if model == 'piecewise_radiative_losses':
        lightcurve = mm.piecewise_radiative_losses(time, **parameters)
        magnetar = mm.magnetar_only(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle='--')

    if model == 'radiative_losses':
        lightcurve = mm.radiative_losses(time, **parameters)
        magnetar = mm.magnetar_only(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle='--')

    if model == 'radiative_losses_mdr':
        lightcurve = mm.radiative_losses_mdr(time, **parameters)

    if model == 'radiative_losses_smoothness':
        lightcurve = mm.radiative_losses_smoothness(time, **parameters)
        magnetar = mm.magnetar_only(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle='--')

    if model == 'collapsing_radiative_losses':
        lightcurve = mm.collapsing_radiative_losses(time, **parameters)

    ax.plot(time, lightcurve, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32)


def plot_lightcurve(GRB, model, path='GRBData',
                    axes=None,
                    plot_save=True,
                    plot_show=True, random_models=1000,
                    truncate=True, use_photon_index_prior=False,
                    truncate_method='prompt_time_error',
                    luminosity_data=False,save_format='json'):
    """
    plots the lightcurve
    GRB is the telephone number of the GRB
    model = model to plot
    path = path to GRB folder
    """
    ax = axes or plt.gca()

    # read result
    result, data = read_result(model=model, GRB=GRB, path=path, truncate=truncate,
                               use_photon_index_prior=use_photon_index_prior, truncate_method=truncate_method,luminosity_data=luminosity_data,
                               save_format=save_format)

    # set up plotting directory structure
    dir = data.path + '/GRB' + data.name
    plots_base_directory = dir + '/' + model + '/plots/'
    if not os.path.exists(plots_base_directory):
        os.makedirs(plots_base_directory)

    # dictionary of max likelihood parameters
    maxL = dict(result.posterior.sort_values(by=['log_likelihood']).iloc[-1])

    for j in range(int(random_models)):
        params = dict(result.posterior.iloc[np.random.randint(len(result.posterior))])
        plot_models(parameters=params, axes=axes, alpha=0.05, lw=2, colour='r', model=model)

        # plot max likelihood
    plot_models(parameters=maxL, axes=axes, alpha=0.65, lw=2, colour='b', model=model)
    
    plot_data(GRB=GRB, axes=axes, path=path, truncate=truncate, truncate_method=truncate_method, luminosity_data=luminosity_data)

    if plot_save:
        if use_photon_index_prior == True:
            plt.savefig(plots_base_directory + model + '_photon_index_lightcurve.png')
        else:
            plt.savefig(plots_base_directory + model + '_lightcurve.png')

    if plot_show:
        plt.show()

    # plt.close()

def calculate_BF(model1, model2, GRB, path='.', use_photon_index_prior=False, luminosity_data=False,save_format='json'):
    model1, data = read_result(model=model1, GRB=GRB, path=path, use_photon_index_prior=use_photon_index_prior,luminosity_data=luminosity_data,save_format=save_format)
    model2, data = read_result(model=model2, GRB=GRB, path=path, use_photon_index_prior=use_photon_index_prior,luminosity_data=luminosity_data,save_format=save_format)
    logBF = model1.log_evidence - model2.log_evidence
    return logBF
