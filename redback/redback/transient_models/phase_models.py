import numpy as np
import extinction
from . import extinction_models
from . import integrated_flux_afterglow_models as infam
from . import afterglow_models
from .fireball_models import predeceleration
from ..utils import get_functions_dict, calc_ABmag_from_fluxdensity, deceleration_timescale
from ..constants import *

_, modules_dict = get_functions_dict(afterglow_models)

def t0_extinction_models(time, lognh, factor, **kwargs):
    """
    :param time: time in mjd or other unit. Note is not a reference time.
    :param lognh: hydrogen column density
    :param factor: prefactor for extinction
    :param kwargs: Must include t0 parameter which is in the same units as the data.
    :return: magnitude
    """
    time = (time - kwargs['t0']) * 86400
    magnitude = extinction_models.extinction_with_afterglow_base_model(time=time, lognh=lognh, factor=factor, **kwargs)
    return magnitude

def t0_exinction_models_with_predeceleration(time, **kwargs):
    base_model = kwargs['base_model']

    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    gradient = kwargs['mm']
    t0 = kwargs['t0'] * 86400
    t_peak_kwargs = dict.copy(kwargs)
    t_peak_kwargs['frequency'] = kwargs['tpeak_frequency']
    t_peak = deceleration_timescale(**t_peak_kwargs)
    f_at_t_peak = function(t_peak, **t_peak_kwargs)
    a_1 = f_at_t_peak / ((t_peak - t0)**gradient)
    time = (time - kwargs['t0']) * 86400
    predeceleration_time = np.where(time < t_peak)
    afterglow_time = np.where(time >= t_peak)
    f1 = predeceleration(time=time[predeceleration_time], a_1=a_1, mm=gradient, t0=t0)
    f2 = function(time[afterglow_time], **kwargs)
    flux = np.concatenate((f1, f2))
    print(flux)
    factor = kwargs['factor']
    lognh = kwargs['lognh']
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor
    frequency = kwargs['frequency']
    frequency = np.array([frequency])
    # logger.info('Using the fitzpatrick99 extinction law')
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
    # read the base_model dict
    # logger.info('Using {} as the base model for extinction'.format(base_model))
    flux = extinction.apply(mag_extinction, flux)
    magnitude = calc_ABmag_from_fluxdensity(flux).value
    return magnitude

def t0_afterglowpy_rate_model(time, **kwargs):
    """
    :param time: time in seconds
    :param burst_start: burst start time in seconds
    :param bkg_rate: background rate. This background rate could be zero.
    :param kwargs: all other keyword arguments needed for the base model
    :return: rate, including the background rate
    """
    dt = kwargs['dt']
    burst_start = kwargs['burst_start']
    bkg_rate = kwargs['bkg_rate']
    kwargs['prefactor'] = kwargs['prefactor'][time >= burst_start]
    grb_time = time[time >= burst_start] - burst_start
    rate = np.zeros(len(time))
    rate[time >= burst_start] = infam.integrated_flux_rate_model(grb_time, **kwargs)
    rate[time < burst_start] = (bkg_rate * dt)
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

def t0_afterglowpy_fluxdensity_model(time, burst_start, **kwargs):
    """
    Afterglowpy based flux density models with burst_start as a parameter.
    :param time: time in seconds
    :param kwargs: Must include a burst_start and background_rate parameter. This background rate could be zero.
    :return: flux density for time > T0 parameter

    """
    base_model = kwargs['base_model']
    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    grb_time = time[time >= burst_start] - burst_start
    flux = function(grb_time, **kwargs)
    return flux, grb_time