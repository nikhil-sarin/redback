import bilby
import numpy as np
import os
import matplotlib.pyplot as plt
import grb_bilby.processing.GRB as tools
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

def load_data(GRB, path = 'GRBData', truncate = True):
    """
    :param GRB: telephone number
    :param path: default for package data, or path to GRBData folder
    :param truncate: method of truncation/True/False
    :return: data class
    """
    data_dir = find_path(path)
    data = tools.SGRB(name=GRB, path = data_dir)
    data.load_and_truncate_data(truncate=truncate)
    return data

def read_result(model, GRB, path = '.', truncate = True):
    """
    :param model: model to analyse
    :param GRB: telephone number of GRB
    :param path:
    :param truncate:
    :return: bilby result object and data object
    """
    result_path = path+'/GRB' + GRB +'/'+model + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result = bilby.result.read_in_result(filename = result_path + 'result_result.json')
    data_dir = find_path(path)
    data = tools.SGRB(name=GRB, path = data_dir)
    data.load_and_truncate_data(truncate=truncate)

    return result, data

def plot_data(GRB, path, truncate, axes = None, colour='k'):
    '''
    plots the data
    GRB is the telephone number of the GRB
    '''
    ax = axes or plt.gca()
    data = load_data(GRB = GRB, path = path, truncate = truncate)
    tt =  data.time
    tt_err = data.time_err
    Lum50 = data.Lum50
    Lum50_err = data.Lum50_err

    ax.errorbar(tt, Lum50,
                 xerr=[tt_err[1, :], tt_err[0, :]],
                 yerr=[Lum50_err[1, :], Lum50_err[0, :]],
                 fmt='x', c=colour, ms=1, elinewidth=2, capsize = 0.)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(0.5 * tt[0], 2 * (tt[-1] + tt_err[0, -1]))
    ax.set_ylim(0.5 * min(Lum50), 2. * np.max(Lum50))

    ax.annotate('GRB'+data.name, xy=(0.95, 0.9), xycoords='axes fraction',
                 horizontalalignment='right')

    ax.set_xlabel(r'time since burst [s]')
    ax.set_ylabel(r'Flux erg cm$^{-2}$ s$^{-1}$')
    ax.tick_params(axis='x', pad=10)

    #plt.tight_layout()
    if axes == None:
        plt.tight_layout()

def plot_models(parameters, model, axes = None, colour='r', alpha=1.0, ls='-', lw=4):
    '''
    plot the models
    parameters: dictionary of parameters - 1 set of Parameters
    model: model name
    '''
    time = np.logspace(-4, 7, 100)
    ax = axes or plt.gca()

    if model == 'collapsing_magnetar':
        lightcurve = mm.collapsing_mag(time, **parameters)
    if model == 'full_magnetar':
        lightcurve = mm.full_magnetar(time, **parameters)
    if model == 'collapsing_losses':
        lightcurve = mm.collapsing_losses(time, **parameters)
    if model == 'two_component_fireball':
        lightcurve = mm.two_component_fireball_model(time, **parameters)
    if model == 'radiative_losses':
        lightcurve = mm.radiative_losses(time, **parameters)
        magnetar = mm.magnetar(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle = '--')
    if model == 'radiative_losses_full':
        lightcurve = mm.radiative_losses_full(time, **parameters)
        magnetar = mm.magnetar(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle = '--')
    ax.plot(time, lightcurve, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32)


def plot_lightcurve(GRB, model,path = '.',
                    axes = None,
                    plot_save=True,
                    plot_show=True, random_models = 1000, truncate = True):
    '''
    plots the lightcurve
    GRB is the telephone number of the GRB
    model = model to plot
    path = path to GRB folder
    '''
    ax = axes or plt.gca()

    #read result
    result, data = read_result(model = model, GRB = GRB, path = path, truncate = truncate)

    #set up plotting directory structure
    dir = data.path+'/GRB'+data.name
    plots_base_directory = dir+'/'+model+'/plots/'
    if not os.path.exists(plots_base_directory):
        os.makedirs(plots_base_directory)

    #dictionary of max likelihood parameters
    maxL = dict(result.posterior.iloc[-1])

    #plot max likelihood
    plot_models(parameters = maxL, axes = axes, alpha=0.65, lw=2, colour='b', model = model)

    for j in range(int(random_models)):
        params = dict(result.posterior.iloc[np.random.randint(len(result.posterior))])
        plot_models(parameters = params, axes = axes, alpha=0.05, lw=1, colour='r', model = model)

    plot_data(GRB=GRB, axes = axes, path = path, truncate = truncate)

    if plot_save:
        plt.savefig(plots_base_directory + model + '_lightcurve.png')

    if plot_show:
        plt.show()

def calculate_BF(model1, model2, GRB, path = '.'):
    model1, data = read_result(model = model1, GRB = GRB, path = path)
    model2, data = read_result(model = model2, GRB = GRB, path = path)
    str = GRB
    logBF = model1.log_evidence - model2.log_evidence

    return logBF
