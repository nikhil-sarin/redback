from inspect import isfunction
import numpy as np

from astropy.time import Time
import astropy.units as uu

from redback.constants import *
from redback.transient_models import extinction_models
from redback.transient_models import integrated_flux_afterglow_models as infam
from redback.utils import calc_ABmag_from_flux_density, calc_flux_density_from_ABmag, citation_wrapper
from collections import namedtuple

extinction_model_functions = {'supernova':extinction_models.extinction_with_supernova_base_model,
                              'kilonova':extinction_models.extinction_with_kilonova_base_model,
                              'afterglow':extinction_models.extinction_with_afterglow_base_model,
                              'tde':extinction_models.extinction_with_tde_base_model,
                              'magnetar_driven':extinction_models.extinction_with_magnetar_driven_base_model}

@citation_wrapper('redback')
def t0_base_model(time, t0, **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
    :return: output of the base_model
    """
    from redback.model_library import all_models_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']
    function = all_models_dict[base_model]
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.day).value
    output = function(time, **kwargs)
    return output


def _t0_with_extinction(time, t0, av, model_type='supernova', **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param av: absolute mag extinction
    :param model_type: what type of transient extinction function to use
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    output = namedtuple('output', ['time', 'observable'])
    function = extinction_model_functions[model_type]
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.day).value
    output.time = time
    output.observable = function(time, av=av, **kwargs)
    return output

@citation_wrapper('redback')
def t0_afterglow_extinction(time, t0, av, **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    summary = _t0_with_extinction(time=time, t0=t0, av=av, model_type='afterglow', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_supernova_extinction(time, t0, av, **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    summary = _t0_with_extinction(time=time, t0=t0, av=av, model_type='supernova', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_kilonova_extinction(time, t0, av, **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    summary = _t0_with_extinction(time=time, t0=t0, av=av, model_type='kilonova', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_tde_extinction(time, t0, av, **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    summary = _t0_with_extinction(time=time, t0=t0, av=av, model_type='tde', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_magnetar_driven_extinction(time, t0, av, **kwargs):
    """
    :param time: time in mjd
    :param t0: start time in mjd
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    summary = _t0_with_extinction(time=time, t0=t0, av=av, model_type='magnetar_driven', **kwargs)
    return summary.observable

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def t0_afterglow_extinction_model_d2g(time, lognh, factor, **kwargs):
    """
    :param time: time in mjd
    :param lognh: hydrogen column density
    :param factor: prefactor for extinction
    :param kwargs: Must include t0 parameter which is in the same units as the data.
                And all the parameters required by the base_model specified using kwargs['base_model']
    :return: magnitude or flux_density depending on kwarg 'output_format'
    """
    t0 = kwargs['t0']
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.day).value
    magnitude = extinction_models.extinction_afterglow_galactic_dust_to_gas_ratio(time=time, lognh=lognh,
                                                                                  factor=factor, **kwargs)
    if kwargs['output_format'] == 'flux_density':
        return calc_flux_density_from_ABmag(magnitude).value
    elif kwargs['output_format'] == 'magnitude':
        return magnitude

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def _t0_thin_shell_predeceleration(time, **kwargs):
    """
    Assume pre-deceleration behaviour is in thin-shell regime and follows Sari and Piran 1997

    :param time:
    :param kwargs:
    :return: flux or magnitude
    """
    e0 = 10 ** kwargs['loge0']
    nism = 10 ** kwargs['logn0']
    g0 = kwargs['g0']
    frac1 = 3 * e0
    frac2 = 32 * np.pi * g0 ** 8 * nism * proton_mass * speed_of_light ** 5
    tp = (frac1 / frac2) ** (1. / 3.)
    gradient = kwargs['m']
    tt_predec = time[time < tp]
    tt_postdec = time[time >= tp]
    predec_kwargs = kwargs.copy()
    afterglow_kwargs = kwargs.copy()
    f2 = t0_afterglow_extinction_model_d2g(tt_postdec, **afterglow_kwargs)
    f_at_tp = t0_afterglow_extinction_model_d2g(tp, **afterglow_kwargs)
    aa = f_at_tp / (kwargs['tp'] - kwargs['t0']) ** gradient
    predec_kwargs['aa'] = aa
    predec_kwargs['mm'] = gradient

    f1 = extinction_models._extinction_with_predeceleration(tt_predec, **predec_kwargs)
    flux = np.concatenate((f1, f2))

    if kwargs['output_format'] == 'flux_density':
        return flux
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def _t0_exinction_models_with_sampled_t_peak(time, tp, **kwargs):
    """
    Sample in peak time and smoothly connect with afterglowpy output

    :param time: in MJD, times should be after T0.
    :param tp: in MJD
    :param kwargs:
    :return: flux or magnitude depending on kwargs.
    """
    gradient = kwargs['m']
    tt_predec = time[time < tp]
    tt_postdec = time[time >= tp]
    predec_kwargs = kwargs.copy()
    afterglow_kwargs = kwargs.copy()
    f2 = t0_afterglow_extinction_model_d2g(tt_postdec, **afterglow_kwargs)
    f_at_tp = t0_afterglow_extinction_model_d2g(tp, **afterglow_kwargs)
    aa = f_at_tp / (kwargs['tp'] - kwargs['t0']) ** gradient
    predec_kwargs['aa'] = aa
    predec_kwargs['mm'] = gradient
    f1 = extinction_models._extinction_with_predeceleration(tt_predec, **predec_kwargs)
    flux = np.concatenate((f1, f2))
    if kwargs['output_format'] == 'flux_density':
        return flux
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210510108S/abstract')
def _t0_afterglowpy_rate_model(time, **kwargs):
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

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210510108S/abstract')
def _t0_afterglowpy_flux_model(time, burst_start, **kwargs):
    """
    Afterglowpy based integrated flux models with burst_start as a parameter.

    :param time: time in seconds
    :param kwargs: Must include a burst_start and background_rate parameter. This background rate could be zero.
    :return: integrated flux
    """
    grb_time = time[time >= burst_start] - burst_start
    flux = infam.integrated_flux_afterglowpy_base_model(grb_time, **kwargs)
    return flux, grb_time

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210510108S/abstract')
def _t0_afterglowpy_flux_density_model(time, burst_start, **kwargs):
    """
    Afterglowpy based flux density models with burst_start as a parameter.

    :param time: time in seconds
    :param kwargs: Must include a burst_start and background_rate parameter. This background rate could be zero.
    :return: flux density for time > T0 parameter

    """
    from ..model_library import modules_dict
    base_model = kwargs['base_model']

    if isfunction(base_model):
        function = base_model
    elif isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")

    grb_time = time[time >= burst_start] - burst_start
    flux = function(grb_time, **kwargs)
    return flux, grb_time
