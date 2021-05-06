import numpy as np
from . import extinction_models
from . import integrated_flux_afterglow_models as infam

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
    :param time: time in seconds
    :param kwargs: Must include a burst_start and background_rate parameter. This background rate could be zero.
    :return: integrated flux
    """
    grb_time = time[time >= burst_start] - burst_start
    flux = infam.integrated_flux_afterglowpy_base_model(grb_time, **kwargs)
    return flux, grb_time