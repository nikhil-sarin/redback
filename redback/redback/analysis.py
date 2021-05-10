from .utils import logger, calc_ABmag_from_fluxdensity, get_functions_dict
from transient_models import afterglow_models
import matplotlib.pyplot as plt

_, modules_dict = get_functions_dict(afterglow_models)

def plot_multiple_multiband_lightcurves():
    pass

def plot_evolution_parameters():
    pass

def plot_multiple_lightcurves():
    pass

def plot_afterglowpy_lightcurves(time, plot = False, **kwargs):
    """
    :param time: Time for the axis
    :param kwargs:
    :return: either the time and flux arrays or plot.
    """
    base_model = kwargs['base_model']
    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    logger.info('Using {} as the base model'.format(base_model))

    flux = function(time, **kwargs)

    if plot:
        plt.loglog(time, flux, lw = 0.1, c='red', alpha = 0.1, zorder = -1)
        return None
    else:
        return time, flux

