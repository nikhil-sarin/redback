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
                              'magnetar_driven':extinction_models.extinction_with_magnetar_driven_base_model,
                              'shock_powered':extinction_models.extinction_with_shock_powered_base_model}

@citation_wrapper('redback')
def t0_base_model(time, t0, **kwargs):
    """
    Generic base model with t0 parameter for MJD time handling.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of base model to use.
        - peak_time_mjd : float, optional
            Peak time in MJD (will be converted to peak_time relative to t0).
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'flux', or 'spectra'.
        - frequency : float or np.ndarray, optional
            Frequency if output_format is 'flux_density'.
        - bands : str or np.ndarray, optional
            Bands if output_format is 'magnitude' or 'flux'.

    Returns
    -------
    np.ndarray
        Output of the base model.
    """
    from redback.model_library import all_models_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']
    if 'peak_time_mjd' in kwargs:
        kwargs['peak_time'] = kwargs['peak_time_mjd'] - t0
    function = all_models_dict[base_model]
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.day).value
    transient_time = time[time >= 0.0]
    bad_time = time[time < 0.0]
    if kwargs['base_model'] in ['thin_shell_supernova', 'homologous_expansion_supernova']:
        kwargs['base_model'] = kwargs.get('submodel', 'arnett_bolometric')
    temp_kwargs = kwargs.copy()
    if 'frequency' in temp_kwargs:
        if isinstance(temp_kwargs['frequency'], np.ndarray):
            temp_kwargs['frequency'] = kwargs['frequency'][time >= 0.0]
    if 'bands' in temp_kwargs:
        if isinstance(temp_kwargs['bands'], np.ndarray):
            temp_kwargs['bands'] = kwargs['bands'][time >= 0.0]
    output_real = function(transient_time, **temp_kwargs)
    if kwargs['output_format'] == 'magnitude':
        output_fake = np.zeros(len(bad_time)) + 1000
    else:
        output_fake = np.zeros(len(bad_time))
    output = np.concatenate((output_fake, output_real))
    return output


def _t0_with_extinction(time, t0, av_host, model_type='supernova', **kwargs):
    """
    Generic t0 model with extinction for MJD time handling.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    model_type : str, optional
        Type of model: 'supernova', 'kilonova', 'afterglow', 'tde', 'magnetar_driven', or 'shock_powered'.
        Default is 'supernova'.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of base model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    namedtuple
        Output containing time and observable with extinction applied.
    """
    output = namedtuple('output', ['time', 'observable'])
    function = extinction_model_functions[model_type]
    t0 = Time(t0, format='mjd')
    time = Time(np.asarray(time, dtype=float), format='mjd')
    time = (time - t0).to(uu.day).value
    transient_time = time[time >= 0.0]
    bad_time = time[time < 0.0]
    output_real = function(transient_time, av_host=av_host, **kwargs)
    if kwargs['output_format'] == 'magnitude':
        output_fake = np.zeros(len(bad_time)) + 5000
    else:
        output_fake = np.zeros(len(bad_time))
    output.time = time
    output.observable = np.concatenate((output_fake, output_real))
    return output

@citation_wrapper('redback')
def t0_afterglow_extinction(time, t0, av_host, **kwargs):
    """
    Afterglow model with t0 parameter and host/MW extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of afterglow model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux density or magnitude with extinction applied.
    """
    summary = _t0_with_extinction(time=time, t0=t0, av_host=av_host, model_type='afterglow', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_supernova_extinction(time, t0, av_host, **kwargs):
    """
    Supernova model with t0 parameter and host/MW extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of supernova model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux density or magnitude with extinction applied.
    """
    summary = _t0_with_extinction(time=time, t0=t0, av_host=av_host, model_type='supernova', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_kilonova_extinction(time, t0, av_host, **kwargs):
    """
    Kilonova model with t0 parameter and host/MW extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of kilonova model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux density or magnitude with extinction applied.
    """
    summary = _t0_with_extinction(time=time, t0=t0, av_host=av_host, model_type='kilonova', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_tde_extinction(time, t0, av_host, **kwargs):
    """
    TDE model with t0 parameter and host/MW extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of TDE model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - peak_time_mjd : float, optional
            Peak time in MJD (will be converted to peak_time relative to t0).
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux density or magnitude with extinction applied.
    """
    if 'peak_time_mjd' in kwargs:
        kwargs['peak_time'] = kwargs['peak_time_mjd'] - t0
    summary = _t0_with_extinction(time=time, t0=t0, av_host=av_host, model_type='tde', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_magnetar_driven_extinction(time, t0, av_host, **kwargs):
    """
    Magnetar-driven model with t0 parameter and host/MW extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of magnetar-driven model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux density or magnitude with extinction applied.
    """
    summary = _t0_with_extinction(time=time, t0=t0, av_host=av_host, model_type='magnetar_driven', **kwargs)
    return summary.observable

@citation_wrapper('redback')
def t0_shock_powered_extinction(time, t0, av_host, **kwargs):
    """
    Shock-powered model with t0 parameter and host/MW extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    t0 : float
        Start time in MJD.
    av_host : float
        V-band extinction from host galaxy in magnitudes.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str
            Name of shock-powered model to use.
        - redshift : float
            Source redshift (required).
        - av_mw : float, optional
            MW V-band extinction in magnitudes (default 0.0).
        - rv_host : float, optional
            Host R_V parameter (default 3.1).
        - rv_mw : float, optional
            MW R_V parameter (default 3.1).
        - host_law : str, optional
            Host extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - mw_law : str, optional
            MW extinction law (default 'fitzpatrick99').
            Options: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux density or magnitude with extinction applied.
    """
    summary = _t0_with_extinction(time=time, t0=t0, av_host=av_host, model_type='shock_powered', **kwargs)
    return summary.observable

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def t0_afterglow_extinction_model_d2g(time, lognh, factor, **kwargs):
    """
    Afterglow model with t0 parameter using dust-to-gas ratio for extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD.
    lognh : float
        Log10 hydrogen column density.
    factor : float
        Prefactor for extinction.
    **kwargs : dict
        Additional keyword arguments:

        - t0 : float
            Start time in MJD.
        - base_model : str
            Name of afterglow model to use.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Magnitude or flux density depending on output_format.
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
    Pre-deceleration model assuming thin-shell regime following Sari and Piran 1997.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    **kwargs : dict
        Additional keyword arguments:

        - loge0 : float
            Log10 on-axis isotropic equivalent energy.
        - logn0 : float
            Log10 number density of ISM.
        - g0 : float
            Initial Lorentz factor.
        - m : float
            Pre-deceleration gradient.
        - t0 : float
            Start time.
        - tp : float
            Peak time.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux or magnitude.
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
    Sample peak time and smoothly connect with afterglow model output.

    Parameters
    ----------
    time : np.ndarray
        Time in MJD (times should be after T0).
    tp : float
        Peak time in MJD.
    **kwargs : dict
        Additional keyword arguments:

        - t0 : float
            Start time in MJD.
        - m : float
            Pre-deceleration gradient.
        - output_format : str
            Output format: 'flux_density' or 'magnitude'.

    Returns
    -------
    np.ndarray
        Flux or magnitude depending on output_format.
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
    Afterglow rate model with burst start time.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    **kwargs : dict
        Additional keyword arguments:

        - burst_start : float, optional
            Burst start time in seconds (default 0).
        - background_rate : float, optional
            Background rate (default 0).
        - dt : float, optional
            Time bin width (default 1).
        - prefactor : np.ndarray
            Prefactor array.

    Returns
    -------
    np.ndarray
        Rate including background rate.
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
    Afterglow integrated flux model with burst start time.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    burst_start : float
        Burst start time in seconds.
    **kwargs : dict
        Additional keyword arguments required by the base model.

    Returns
    -------
    tuple
        (flux, grb_time) - integrated flux and GRB time array.
    """
    grb_time = time[time >= burst_start] - burst_start
    flux = infam.integrated_flux_afterglowpy_base_model(grb_time, **kwargs)
    return flux, grb_time

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210510108S/abstract')
def _t0_afterglowpy_flux_density_model(time, burst_start, **kwargs):
    """
    Afterglow flux density model with burst start time.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    burst_start : float
        Burst start time in seconds.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str or callable
            Name or function of afterglow model to use.

    Returns
    -------
    tuple
        (flux_density, grb_time) - flux density and GRB time array for time > T0.
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
