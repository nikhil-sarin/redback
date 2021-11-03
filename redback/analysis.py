import matplotlib.pyplot as plt
import numpy as np

from .utils import logger


def plot_multiple_multiband_lightcurves():
    pass


def plot_evolution_parameters():
    pass


def plot_multiple_lightcurves():
    pass


def plot_afterglowpy_lightcurves(time, plot=False, **kwargs):
    """
    :param time: Time for the axis
    :param kwargs:
    :return: either the time and flux arrays or plot.
    """
    from model_library import modules_dict
    base_model = kwargs['base_model']
    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    logger.info('Using {} as the base model'.format(base_model))

    flux = function(time, **kwargs)

    if plot:
        plt.loglog(time, flux, lw=0.1, c='red', alpha=0.1, zorder=-1)
        return None
    else:
        return time, flux


def evaluate_extinction(time, **s):
    nus = np.array([nu_rband, nu_gband, nu_iband])
    nu_1d = nus  # data['Hz']
    t_1d = time
    t, nu = np.meshgrid(t_1d, nu_1d)
    t = t.flatten()
    nu = nu.flatten()
    s['frequency'] = nu
    magnitudes = mm.t0_extinction_models(t, **s)
    magnitudes = magnitudes.reshape(len(nu_1d), len(t_1d))
    return magnitudes, nus


def confidence_interval_lightcurve(result, base_model):
    kwargs = dict()
    frequency_obs = data['frequency'].values
    kwargs['frequency'] = frequency_obs
    kwargs['spread'] = False
    kwargs['latres'] = 2
    kwargs['tres'] = 100
    kwargs['spectype'] = 1
    kwargs['base_model'] = base_model
    kwargs['output_format'] = 'flux_density'
    models = 10
    frequency = 3
    tt_len = 30
    lc_r = np.zeros((models, tt_len))
    lc_i = np.zeros((models, tt_len))
    lc_g = np.zeros((models, tt_len))
    for x in range(models):
        samples = dict(result.posterior.iloc[np.random.randint(len(result.posterior))])
        t0 = samples['t0']
        time = np.linspace(t0 + 1e-5, 58882.55, 30)
        samples.update(kwargs)
        lc, nus = evaluate_extinction(time, **samples)
        lc_r[x] = lc[0]
        lc_g[x] = lc[1]
        lc_i[x] = lc[2]
    lcs = [lc_r, lc_g, lc_i]
    lower_bound = {}
    upper_bound = {}
    median = {}
    for x in range(3):
        lower_bound[x], upper_bound[x], median[x] = redback.utils.calc_confidence_intervals(lcs[x])

    return lower_bound, upper_bound, median
