import numpy as np
from . import extinction_models
from . import integrated_flux_afterglow_models as infam
from . import afterglow_models
from .fireball_models import predeceleration
from ..utils import get_functions_dict, calc_ABmag_from_flux_density, deceleration_timescale, calc_flux_density_from_ABmag
from ..constants import *
from astropy.time import Time
import astropy.units as uu


def t0_extinction_models(time, lognh, factor, **kwargs):
    """
    :param time: time in mjd
    :param lognh: hydrogen column density
    :param factor: prefactor for extinction
    :param kwargs: Must include t0 parameter which is in the same units as the data.
    :return: magnitude or flux_density depending on kwarg 'output_format'
    """
    if kwargs['output_format'] is not 'flux_density' or not 'magnitude':
        raise ValueError('Output format {} not understood. Please use magnitude or flux_density'.format(kwargs['output_format']))
    t0 = kwargs['t0']
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.second).value
    magnitude = extinction_models.extinction_with_afterglow_base_model(time=time, lognh=lognh, factor=factor, **kwargs)
    if kwargs['output_format'] == 'flux_density':
        return calc_flux_density_from_ABmag(magnitude).value
    elif kwargs['output_format'] == 'magnitude':
        return magnitude

def t0_thin_shell_predeceleration(time, **kwargs):
    """
    Assume pre-deceleration behaviour is in thin-shell regime and follows Sari and Piran 1997
    :param time:
    :param kwargs:
    :return: flux or magnitude
    """
    if kwargs['output_format'] is not 'flux_density' or not 'magnitude':
        raise ValueError('Output format {} not understood. Please use magnitude or flux_density'.format(kwargs['output_format']))

    e0 = 10 ** kwargs['loge0']
    nism = 10 ** kwargs['logn0']
    g0 = kwargs['g0']
    frac1 = 3 * e0
    frac2 = 32 * np.pi * g0**8 * nism * proton_mass * speed_of_light**5
    tp = (frac1/frac2)**(1./3.)
    gradient = kwargs['m']
    tt_predec = time[time < tp]
    tt_postdec = time[time >= tp]
    predec_kwargs = kwargs.copy()
    afterglow_kwargs = kwargs.copy()
    f2 = t0_extinction_models(tt_postdec, **afterglow_kwargs)
    f_at_tp = t0_extinction_models(tp, **afterglow_kwargs)
    aa = f_at_tp / (kwargs['tp'] - kwargs['t0']) ** gradient
    predec_kwargs['aa'] = aa
    predec_kwargs['mm'] = gradient

    f1 = extinction_models.extinction_with_predeceleration(tt_predec, **predec_kwargs)
    flux = np.concatenate((f1, f2))

    if kwargs['output_format'] == 'flux_density':
        return flux
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux).value

def t0_exinction_models_with_sampled_t_peak(time, tp, **kwargs):
    """
    Sample in peak time and smoothly connect with afterglowpy output
    :param time: in MJD, times should be after T0.
    :param tp: in MJD
    :param kwargs:
    :return: flux or magnitude depending on kwargs.
    """
    if kwargs['output_format'] is not 'flux_density' or not 'magnitude':
        raise ValueError('Output format {} not understood. Please use magnitude or flux_density'.format(kwargs['output_format']))

    gradient = kwargs['m']
    tt_predec = time[time < tp]
    tt_postdec = time[time >= tp]
    predec_kwargs = kwargs.copy()
    afterglow_kwargs = kwargs.copy()
    f2 = t0_extinction_models(tt_postdec, **afterglow_kwargs)
    f_at_tp = t0_extinction_models(tp, **afterglow_kwargs)
    aa = f_at_tp / (kwargs['tp'] - kwargs['t0']) ** gradient
    predec_kwargs['aa'] = aa
    predec_kwargs['mm'] = gradient
    f1 = extinction_models.extinction_with_predeceleration(tt_predec, **predec_kwargs)
    flux = np.concatenate((f1, f2))
    if kwargs['output_format'] == 'flux_density':
        return flux
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux).value

def t0_afterglowpy_rate_model(time, **kwargs):
    """
    :param time: time in seconds
    :param burst_start: burst start time in seconds
    :param background_rate: background rate. This background rate could be zero.
    :param kwargs: all other keyword arguments needed for the base model
    :return: rate, including the background rate
    """
    dt = kwargs.get('dt', 1)
    burst_start = kwargs.get('burst_start', 0)
    background_rate = kwargs.get('background_rate', 0)
    kwargs['prefactor'] = kwargs['prefactor'][time >= burst_start]
    grb_time = time[time >= burst_start] - burst_start
    rate = np.zeros(len(time))
    rate[time >= burst_start] = infam.integrated_flux_rate_model(grb_time, **kwargs)
    rate[time < burst_start] = (background_rate * dt)
    return rate

def t0_afterglowpy_flux_model(time, burst_start, **kwargs):
    """
    Afterglowpy based integrated flux models with burst_start as a parameter.
    :param time: time in seconds
    :param kwargs: Must include a burst_start and background_rate parameter. This background rate could be zero.
    :return: integrated flux
    """
    grb_time = time[time >= burst_start] - burst_start
    flux = infam.integrated_flux_afterglowpy_base_model(grb_time, **kwargs)
    return flux, grb_time

def t0_afterglowpy_flux_density_model(time, burst_start, **kwargs):
    """
    Afterglowpy based flux density models with burst_start as a parameter.
    :param time: time in seconds
    :param kwargs: Must include a burst_start and background_rate parameter. This background rate could be zero.
    :return: flux density for time > T0 parameter

    """
    from ..model_library import modules_dict
    base_model = kwargs['base_model']
    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    grb_time = time[time >= burst_start] - burst_start
    flux = function(grb_time, **kwargs)
    return flux, grb_time