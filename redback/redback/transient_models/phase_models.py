import numpy as np
import extinction
from . import extinction_models
from . import integrated_flux_afterglow_models as infam
from . import afterglow_models
from .fireball_models import predeceleration
from ..utils import get_functions_dict, calc_ABmag_from_fluxdensity, deceleration_timescale, calc_fluxdensity_from_ABmag
from ..constants import *
from astropy.time import Time
import astropy.units as uu

_, modules_dict = get_functions_dict(afterglow_models)

def t0_extinction_models(time, lognh, factor, **kwargs):
    """
    :param time: time in mjd
    :param lognh: hydrogen column density
    :param factor: prefactor for extinction
    :param kwargs: Must include t0 parameter which is in the same units as the data.
    :return: magnitude or fluxdensity depending on kwarg 'output_format'
    """
    if kwargs['output_format'] is not 'flux_density' or not 'magnitude':
        raise ValueError('Output format {} not understood. Please use magnitude or flux_density'.format(kwargs['output_format']))
    t0 = kwargs['t0']
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.second).value
    magnitude = extinction_models.extinction_with_afterglow_base_model(time=time, lognh=lognh, factor=factor, **kwargs)
    if kwargs['output_format'] == 'flux_density':
        return calc_fluxdensity_from_ABmag(magnitude).value
    elif kwargs['output_format'] == 'magnitude':
        return magnitude

def t0_exinction_models_with_sampled_t_peak(time, t0, t_peak, **kwargs):
    base_model = kwargs['base_model']

    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    gradient = kwargs['mm']
    t0_d = t0

    t_peak_kwargs = dict.copy(kwargs)
    t_peak_kwargs['frequency'] = kwargs['tpeak_frequency']
    f_at_t_peak = function(t_peak, **t_peak_kwargs)

    a_1 = f_at_t_peak / ((t_peak - t0) ** gradient) #in day units

    time_d = (time - t0_d)
    time_s = time_d * 86400

    predeceleration_time = np.where(time_s < t_peak)
    afterglow_time = np.where(time_s >= t_peak_d)
    f1 = a_1 * (time_d[predeceleration_time])**gradient
    # f1 = predeceleration(time=time_d[predeceleration_time], a_1=a_1, mm=gradient, t0=t0_d)
    print('f1 is {}'.format(f1))
    f2 = function(time[afterglow_time], **kwargs)
    flux = np.concatenate((f1, f2))
    print(flux)
    print('end')
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

def t0_flux_models_with_predeceleration(time, **kwargs):
    base_model = kwargs['base_model']

    if isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]

    gradient = kwargs['mm']
    t0_d = kwargs['t0']

    t_peak_kwargs = dict.copy(kwargs)
    t_peak_kwargs['frequency'] = kwargs['tpeak_frequency']
    t_peak = deceleration_timescale(**t_peak_kwargs) #in secs
    f_at_t_peak = function(t_peak, **t_peak_kwargs)

    t_peak_ref = (t_peak / 86400) + t0_d
    reference_time_s = (t_peak_ref - t0_d) * 86400
    a_1 = f_at_t_peak / ((reference_time_s) ** gradient)

    predeceleration_time = np.where(time < t_peak_ref)
    afterglow_time = np.where(time >= t_peak_ref)
    f1 = predeceleration(time=time[predeceleration_time], a_1=a_1, mm=gradient, t0=t0_d)
    time_afterglow = (time - t0_d) * 86400

    f2 = function(time_afterglow[afterglow_time], **kwargs)

    flux = np.concatenate((f1, f2))
    return flux

def t0_exinction_models_with_predeceleration(time, **kwargs):
    flux = t0_flux_models_with_predeceleration(time, **kwargs)
    frequency = kwargs['frequency']
    factor = kwargs['factor']
    lognh = kwargs['lognh']
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor

    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
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