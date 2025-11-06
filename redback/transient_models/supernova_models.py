import numpy as np
import pandas as pd
from redback.transient_models.phenomenological_models import exponential_powerlaw, fallback_lbol
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar
from redback.transient_models.magnetar_driven_ejecta_models import _ejecta_dynamics_and_interaction
from redback.transient_models.shock_powered_models import  _shocked_cocoon, _csm_shock_breakout
import redback.interaction_processes as ip
import redback.sed as sed
from redback.sed import flux_density_to_spectrum, blackbody_to_spectrum
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import (calc_kcorrected_properties, citation_wrapper, logger, get_csm_properties, nu_to_lambda,
                           lambda_to_nu, velocity_from_lorentz_factor, build_spectral_feature_list)
from redback.constants import day_to_s, solar_mass, km_cgs, au_cgs, speed_of_light, sigma_sb
from inspect import isfunction
import astropy.units as uu
from collections import namedtuple
from scipy.interpolate import interp1d, RegularGridInterpolator

homologous_expansion_models = ['exponential_powerlaw_bolometric', 'arnett_bolometric',
                               'basic_magnetar_powered_bolometric','slsn_bolometric',
                               'general_magnetar_slsn_bolometric','csm_interaction_bolometric',
                               'type_1c_bolometric','type_1a_bolometric']

@citation_wrapper('https://zenodo.org/record/6363879#.YkQn3y8RoeY')
def sncosmo_models(time, redshift, model_kwargs=None, **kwargs):
    """
    A wrapper to SNCosmo models.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    model_kwargs : dict, optional
        All model keyword arguments in a dictionary.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz to evaluate model on, must be same shape as time array or a single value.
            Required if output_format is 'flux_density'.
        - sncosmo_model : str
            String of the SNcosmo model to use. Default is 'salt2'.
        - peak_time : float
            SNe peak time in days. Default is 0.
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.
        - mw_extinction : bool
            Whether there is MW extinction. Default is True.
        - host_extinction : bool
            Whether there is host extinction. Default is True.
            If used, adds ebv parameter to kwargs.
        - ebv : float
            Host galaxy E(B-V). Default is 0.1.
        - use_set_peak_magnitude : bool
            Whether to set the peak magnitude. Default is False.
        - peak_abs_mag : float
            SNe peak absolute magnitude. Default is -19.
        - peak_abs_mag_band : str
            Band for peak absolute magnitude. Default is 'standard::b'.
        - magnitude_system : str
            Magnitude system. Default is 'ab'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'flux', or 'sncosmo_source'.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, flux, or sncosmo source object.
    """
    import sncosmo
    peak_time = kwargs.get('peak_time', 0)
    cosmology = kwargs.get('cosmology', cosmo)
    model_name = kwargs.get('sncosmo_model', 'salt2')
    host_extinction = kwargs.get('host_extinction', True)
    mw_extinction = kwargs.get('mw_extinction', True)
    use_set_peak_magnitude = kwargs.get('use_set_peak_magnitude', False)

    model = sncosmo.Model(source=model_name)
    model.set(z=redshift)
    model.set(t0=peak_time)

    if model_kwargs == None:
        _model_kwargs = {}
        for x in kwargs['model_kwarg_names']:
            _model_kwargs[x] = kwargs[x]
    else:
        _model_kwargs = model_kwargs

    model.update(_model_kwargs)

    if host_extinction:
        ebv = kwargs.get('ebv', 0.1)
        model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
        model.set(hostebv=ebv)
    if mw_extinction:
        model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')

    if use_set_peak_magnitude:
        peak_abs_mag = kwargs.get('peak_abs_mag', -19)
        peak_abs_mag_band = kwargs.get('peak_abs_mag_band', 'standard::b')
        magsystem = kwargs.get('magnitude_system', 'ab')
        model.set_source_peakabsmag(peak_abs_mag, band=peak_abs_mag_band, magsys=magsystem, cosmo=cosmology)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']

        if isinstance(frequency, float):
            frequency = np.array([frequency])

        if (len(frequency) != 1 or len(frequency) == len(time)):
            raise ValueError('frequency array must be of length 1 or same size as time array')

        unique_frequency = np.sort(np.unique(frequency))
        angstroms = nu_to_lambda(unique_frequency)

        _flux = model.flux(time, angstroms)

        if len(frequency) > 1:
            _flux = pd.DataFrame(_flux)
            _flux.columns = unique_frequency
            _flux = np.array([_flux[freq].iloc[i] for i, freq in enumerate(frequency)])

        units = uu.erg / uu.s / uu.Hz / uu.cm ** 2.
        _flux = _flux * nu_to_lambda(frequency)
        _flux = _flux / frequency
        _flux = _flux << units

        flux_density = _flux.to(uu.mJy).flatten()
        return flux_density

    if kwargs['output_format'] == 'flux':
        bands = kwargs['bands']
        magnitude = model.bandmag(time=time, band=bands, magsys='ab')
        return np.nan_to_num(sed.bandpass_magnitude_to_flux(magnitude=magnitude, bands=bands))
    elif kwargs['output_format'] == 'magnitude':
        bands = kwargs['bands']
        magnitude = model.bandmag(time=time, band=bands, magsys='ab')
        return np.nan_to_num(magnitude)
    elif kwargs['output_format'] == 'sncosmo_source':
        return model

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2007A%26A...466...11G/abstract, sncosmo')
def salt2(time, redshift, x0, x1, c, peak_time, **kwargs):
    """
    A wrapper to the SALT2 model in SNCosmo.

    Parameters
    ----------
    time : np.ndarray
        Time in days in observer frame (MJD).
    redshift : float
        Source redshift.
    x0 : float
        SALT2 x0 parameter (amplitude).
    x1 : float
        SALT2 x1 parameter (stretch).
    c : float
        SALT2 c parameter (color).
    peak_time : float
        Peak time in MJD.
    **kwargs : dict
        Additional keyword arguments:

        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.
        - mw_extinction : bool
            Whether there is MW extinction. Default is True.
        - host_extinction : bool
            Whether there is host extinction. Default is True.
        - ebv : float
            Host galaxy E(B-V). Default is 0.1.
        - use_set_peak_magnitude : bool
            Whether to set the peak magnitude. Default is False.
        - peak_abs_mag : float
            SNe peak absolute magnitude. Default is -19.
        - peak_abs_mag_band : str
            Band for peak absolute magnitude. Default is 'standard::b'.
        - magnitude_system : str
            Magnitude system. Default is 'ab'.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'flux', or 'sncosmo_source'.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, flux, or sncosmo source object.
    """
    kwargs['sncosmo_model'] = 'salt2'
    kwargs['peak_time'] = peak_time
    model_kwargs = {'x0':x0, 'x1':x1, 'c':c}
    out = sncosmo_models(time=time, redshift=redshift, model_kwargs=model_kwargs, **kwargs)
    return out

@citation_wrapper('https://arxiv.org/abs/1908.05228, SN1998bw papers..., sncosmo, redback')
def sn1998bw_template(time, redshift, amplitude, **kwargs):
    """
    A wrapper to the SN1998bw template. Only valid between 1100-11000 Angstrom and 0.01 to 90 days post explosion in rest frame

    Parameters
    ----------
    time
        time in days in observer frame (post explosion)
    redshift
        redshift
    amplitude
        amplitude scaling factor, where 1.0 is the original brightness of SN1998bw; and f_lambda is scaled by this factor
    kwargs
        Additional keyword arguments required by redback.
    frequency
        Required if output_format is 'flux_density'. frequency to calculate - Must be same length as time array or a single number).
    bands
        Required if output_format is 'magnitude' or 'flux'.
    output_format
        'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    cosmology
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    Returns
    -------
        set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'

    """
    import sncosmo
    model = sncosmo.Model(source='v19-1998bw')
    original_redshift = 0.0085
    cosmology = kwargs.get("cosmology", cosmo)
    original_dl = (43*uu.Mpc).to(uu.cm).value

    # From roughly matching to Galama+ or Clocchiatti+1998bw light curves
    original_peak_time = 15
    model.set(z=original_redshift, t0=original_peak_time)
    model.set_source_peakmag(14.25, band='bessellb', magsys='ab')
    tts = np.geomspace(0.01, 90, 200)
    lls = np.linspace(1620, 11000, 300)
    f_lambda = model.flux(tts, lls) #erg/s/cm^2/Angstrom.
    l_lambda = f_lambda * 4 * np.pi * original_dl**2  # erg/s/Angstrom

    # We consider this the rest frame spectrum of 1998bw. Now we can redshift it and scale it.
    time_obs = tts * (1 + redshift)
    lambda_obs = lls * (1 + redshift)
    dl_new = cosmology.luminosity_distance(redshift).cgs.value
    f_lambda_obs = l_lambda / (4 * np.pi * dl_new**2)
    f_lambda_obs = amplitude * f_lambda_obs * (1 + redshift) # accounting for bandwidth stretching
    f_lambda_obs = f_lambda_obs * uu.erg / uu.s / uu.cm ** 2 / uu.Angstrom
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        # work in obs frame
        ff_array = lambda_to_nu(lambda_obs)

        # Convert flux density to mJy
        fmjy = f_lambda_obs.to(uu.mJy, equivalencies=uu.spectral_density(wav=lambda_obs * uu.Angstrom)).value
        # Create interpolator on obs frame grid
        flux_interpolator = RegularGridInterpolator(
            (time_obs, ff_array),
            fmjy,
            bounds_error=False,
            fill_value=0.0)

        # Prepare points for interpolation
        if isinstance(frequency, (int, float)):
            frequency = np.ones_like(time) * frequency

        # Create points for evaluation
        points = np.column_stack((time, frequency))

        # Return interpolated flux density with (1+z) correction for observer frame
        return flux_interpolator(points)
    elif kwargs['output_format'] == 'spectra':
        return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_obs,
                                                                    lambdas=lambda_obs,
                                                                    spectra=f_lambda_obs)
    else:
        return sed.get_correct_output_format_from_spectra(time=time, time_eval=time_obs,
                                                          spectra=f_lambda_obs, lambda_array=lambda_obs,
                                                          **kwargs)

@citation_wrapper('redback')
def exponential_powerlaw_bolometric(time, lbol_0, alpha_1, alpha_2, tpeak_d, **kwargs):
    """
    Exponential power-law bolometric light curve model.

    Parameters
    ----------
    time : np.ndarray
        Rest frame time in days.
    lbol_0 : float
        Bolometric luminosity scale in erg/s.
    alpha_1 : float
        First power-law exponent.
    alpha_2 : float
        Second power-law exponent.
    tpeak_d : float
        Peak time in days.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can be None for raw engine luminosity.
        - kappa : float
            Opacity (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion).
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
    lbol = exponential_powerlaw(time, a_1=lbol_0, alpha_1=alpha_1, alpha_2=alpha_2,
                                tpeak=tpeak_d, **kwargs)
    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = exponential_powerlaw(dense_times, a_1=lbol_0, alpha_1=alpha_1, alpha_2=alpha_2,
                                tpeak=tpeak_d, **kwargs)
        interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('redback')
def sn_fallback(time, redshift, logl1, tr, **kwargs):
    """
    Supernova fallback accretion model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    logl1 : float
        Bolometric luminosity scale in log10 (erg/s).
    tr : float
        Transition time for luminosity in days.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity (required for Diffusion).
        - mej : float
            Ejecta mass in solar masses.
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature in K.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs["interaction_process"] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs["photosphere"] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs["sed"] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get("cosmology", cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = fallback_lbol(time=time, logl1=logl1, tr=tr)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = fallback_lbol(time=time, logl1=logl1, tr=tr)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('redback')
def sn_nickel_fallback(time, redshift, mej, f_nickel, logl1, tr, **kwargs):
    """
    Supernova model combining nickel decay and fallback accretion.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Total ejecta mass in solar masses.
    f_nickel : float
        Fraction of nickel mass.
    logl1 : float
        Bolometric luminosity scale in log10 (erg/s).
    tr : float
        Transition time for luminosity in days.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature in K.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs["interaction_process"] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs["photosphere"] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs["sed"] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get("cosmology", cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)       
        lbol = fallback_lbol(time=time, logl1=logl1, tr=tr) + _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = fallback_lbol(time=time, logl1=logl1, tr=tr) + _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)                                                              

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def sn_exponential_powerlaw(time, redshift, lbol_0, alpha_1, alpha_2, tpeak_d, **kwargs):
    """
    Supernova model with exponential power-law light curve.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    lbol_0 : float
        Bolometric luminosity scale in erg/s.
    alpha_1 : float
        First power-law exponent.
    alpha_2 : float
        Second power-law exponent.
    tpeak_d : float
        Peak time in days.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity (required for Diffusion).
        - mej : float
            Ejecta mass in solar masses.
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature in K.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = exponential_powerlaw_bolometric(time=time, lbol_0=lbol_0,
                                           alpha_1=alpha_1,alpha_2=alpha_2, tpeak_d=tpeak_d, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
              frequency=frequency, luminosity_distance=dl)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = exponential_powerlaw_bolometric(time=time, lbol_0=lbol_0,
                                           alpha_1=alpha_1,alpha_2=alpha_2, tpeak_d=tpeak_d, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

def _nickelcobalt_engine(time, f_nickel, mej, **kwargs):
    """
    Nickel-56 and Cobalt-56 radioactive decay engine.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.

    Notes
    -----
    Based on Nadyozhin 1994ApJS...92..527N.
    Uses Ni-56 and Co-56 luminosities and lifetimes.
    """
    '1994ApJS...92..527N'
    ni56_lum = 6.45e43
    co56_lum = 1.45e43
    ni56_life = 8.8  # days
    co56_life = 111.3  # days
    nickel_mass = f_nickel * mej
    lbol = nickel_mass * (ni56_lum*np.exp(-time/ni56_life) + co56_lum * np.exp(-time/co56_life))
    return lbol

def _compute_mass_and_nickel(vmin, esn, mej, f_nickel, f_mixing, mass_len,
                            vmax, delta=0.0, n=12.0):
    """
    Compute mass and nickel distributions for broken power-law density profile.

    Following Matzner & McKee (1999) broken power-law density profile.

    Parameters
    ----------
    vmin : float
        Minimum velocity in km/s.
    esn : float
        Supernova explosion energy in foe (10^51 erg).
    mej : float
        Total ejecta mass in solar masses.
    f_nickel : float
        Fraction of nickel mass.
    f_mixing : float
        Fraction of nickel mass that is mixed.
    mass_len : int
        Number of mass shells.
    vmax : float
        Maximum velocity in km/s.
    delta : float, optional
        Inner density profile exponent. Actual mass distribution is 2 - delta. Default is 0.0.
    n : float, optional
        Outer density profile exponent. Actual mass distribution is 2 - n. Default is 12.0.

    Returns
    -------
    vel : np.ndarray
        Velocity grid in km/s.
    v_m : np.ndarray
        Velocity grid in cm/s.
    m_array : np.ndarray
        Mass distribution in solar masses.
    ni_array : np.ndarray
        Nickel distribution in solar masses. Total nickel mass is f_nickel * mej.

    Notes
    -----
    Uses a broken power-law density profile with a break velocity derived from
    Matzner & McKee (1999) shock speed.
    """
    # Create velocity grid in km/s and convert to cm/s.
    vel = np.geomspace(vmin, vmax, mass_len) # km/s
    v_m = vel * km_cgs # cgs

    # Define a break velocity; use shock speed from Matzner & McKee (1999).
    num = 2 * (5 - delta) * (n - 5) * esn * 1e51
    denom = (3 - delta)*(n - 3) * mej * solar_mass
    v_break = np.sqrt(num/denom) / km_cgs

    # For a uniform grid, determine the velocity spacing.
    dv = vel[1] - vel[0]

    # Compute the unnormalized mass distribution using vectorized operations.
    # For the inner part: (v/v_break)^(2 - delta)
    # For the outer part: (v/v_break)^(2 - n)
    m_array = np.where(vel <= v_break,
                       (vel / v_break)**(2.0 - delta),
                       (vel / v_break)**(2.0 - n))
    # Multiply by the bin width.
    m_array = m_array * dv
    # Normalize the mass array so that the summed mass equals mej.
    total_mass = np.sum(m_array)
    m_array = mej * m_array / total_mass

    # --- Compute the nickel distribution ---
    # Total nickel mass.
    ni_mass = f_nickel * mej
    # Only the inner fraction of the shells receives nickel.
    limiting_index = int(mass_len * f_mixing)
    limiting_index = max(limiting_index, 1)
    # Using the same inner profile for the nickel weight.
    _ni_array = np.where(vel[:limiting_index] <= v_break,
                       (vel[:limiting_index] / v_break)**(2.0 - delta),
                       (vel[:limiting_index] / v_break)**(2.0 - n))
    # old, if only considering nickel in the inner power-law shells.
    # _ni_array = (vel[:limiting_index] / vel[0])**(2.0 - delta)

    _ni_array = ni_mass * _ni_array / np.sum(_ni_array)
    ni_array = np.zeros_like(vel)
    ni_array[:limiting_index] = _ni_array

    return vel, v_m, m_array, ni_array


def _nickelmixing(time, mej, esn, kappa, kappa_gamma, f_nickel, f_mixing,
                  temperature_floor, **kwargs):
    """
    Multi-shell nickel mixing model with stratified ejecta.

    Computes the light curve evolution with nickel mixing using a multi-shell
    approach with energy balance and diffusion.

    Parameters
    ----------
    time : np.ndarray
        Time array in source frame in seconds.
    mej : float
        Ejecta mass in solar masses.
    esn : float
        Explosion energy in foe (10^51 erg).
    kappa : float
        Gray opacity at high temperatures (κ_max) in cm²/g.
        If use_gray_opacity is True, this is the constant opacity.
    kappa_gamma : float
        Gamma-ray opacity (assumed constant) in cm²/g.
    f_nickel : float
        Fraction of total ejecta mass that is nickel.
    f_mixing : float
        Fraction of nickel mass that is mixed.
        Low values put all nickel in the first shell.
    temperature_floor : float
        Temperature floor in K, also used as the transition T_crit.
    **kwargs : dict
        Additional keyword arguments:

        - use_broken_powerlaw : bool
            Whether to use a broken power-law for mass and nickel distribution. Default is True.
        - use_gray_opacity : bool
            Whether to use gray opacity. Default is True.
        - delta : float
            Inner density profile exponent. Used if use_broken_powerlaw is True. Default is 1.0.
        - nn : float
            Outer density profile exponent. Used if use_broken_powerlaw is True. Default is 12.0.
        - beta : float
            Velocity power law slope (M ∝ v^-beta). Used if use_broken_powerlaw is False. Default is 3.0.
        - mass_len : int
            Number of mass shells. Default is 1000.
        - vmax : float
            Maximum velocity in km/s. Default is 250000.
        - vmin_frac : float
            Fraction of characteristic velocity that is the minimum velocity. Default is 0.2.
        - kappa_min : float
            Minimum opacity when cool. Default is 0.001 cm²/g.
        - kappa_n : float
            Exponent controlling opacity transition. Default is 10.

    Returns
    -------
    output : namedtuple
        Named tuple with the following fields:

        - time_temp : np.ndarray
            Time in days.
        - lbol : np.ndarray
            Bolometric luminosity in erg/s.
        - t_photosphere : np.ndarray
            Photospheric temperature in K.
        - r_photosphere : np.ndarray
            Photospheric radius in cm.
        - tau : np.ndarray
            Optical depth (2D array: shells × time).
        - v_photosphere : np.ndarray
            Photospheric velocity in cm/s.
        - energy_v : np.ndarray
            Energy per shell (2D array: shells × time).
    """
    # Constants assumed defined elsewhere: day_to_s, solar_mass, km_cgs, speed_of_light, sigma_sb.
    tdays = time / day_to_s
    time_len = len(time)
    mass_len = int(kwargs.get('mass_len', 1000))
    ni_mass = f_nickel * mej
    vmin_frac = kwargs.get('vmin_frac', 0.2)
    vmin = vmin_frac * (2 * (esn * 1e51) / (mej * solar_mass)) ** 0.5 / 1e5
    vmax = kwargs.get('vmax', 250000)
    use_broken_powerlaw = kwargs.get('use_broken_powerlaw', True)
    use_gray_opacity = kwargs.get('use_gray_opacity', True)

    if use_gray_opacity:
        kappa_eff = kappa
    else:
        # Parameters for temperature-dependent opacity:
        # κ_max is the input "kappa" (for hot, fully ionized ejecta)
        # κ_min and the exponent kappa_n control how quickly the opacity falls when T < T_floor.
        kappa_max = kappa
        kappa_min = kwargs.get('kappa_min', 0.001)  # example default in cm²/g
        kappa_n = kwargs.get('kappa_n', 10)  # controls transition steepness

    if use_broken_powerlaw:
        delta = kwargs.get('delta', 1.0)
        nn = kwargs.get('nn', 12.0)
        diffusion_beta = 13.8 / 3 # effective photon diffusion term, extra /3 to cancel the 3 in the td_v formula.
        vel, v_m, m_array, ni_array = _compute_mass_and_nickel(
            vmin=vmin, esn=esn, mej=mej,
            f_nickel=f_nickel, f_mixing=f_mixing,
            mass_len=mass_len, vmax=vmax, delta=delta, n=nn)
    else:
        # Set up velocity array (in km/s)
        beta = kwargs.get('beta', 3.0)  # velocity power-law slope
        diffusion_beta = beta
        vel = np.linspace(vmin, vmax, mass_len)
        # Convert to cgs: cm/s
        v_m = vel * km_cgs
        # Construct a normalized mass distribution (in solar masses)
        m_array = mej * (vel / vmin) ** (-beta)
        total_mass = np.sum(m_array)
        m_array = m_array * (mej / total_mass)
        # Nickel distribution: put the nickel into the first few shells
        limiting_index = int(mass_len * f_mixing)
        limiting_index = max(limiting_index, 1)
        _ni_array = (vel[:limiting_index] / vel[0]) ** (-beta)
        _ni_array = ni_mass * _ni_array / np.sum(_ni_array)
        ni_array = np.zeros_like(vel)
        ni_array[:limiting_index] = _ni_array

    # Radioactive decay luminosities and lifetimes (in erg/s/solar_mass and days, respectively)
    ni56_lum = 6.45e43  # in erg/s/solar mass
    co56_lum = 1.45e43
    ni56_life = 8.8  # days
    co56_life = 111.3  # days

    # Energy deposition rate per shell as a function of time
    edotr = np.zeros((mass_len, time_len))
    edotr[:, :] = (ni56_lum * np.exp(-tdays / ni56_life) +
                   co56_lum * np.exp(-tdays / co56_life))

    # Pre-allocate arrays
    energy_v = np.zeros((mass_len, time_len))
    lum_rad = np.zeros((mass_len, time_len))
    qdot_ni = np.zeros((mass_len, time_len))
    eth_v = np.zeros((mass_len, time_len))
    td_v = np.zeros((mass_len, time_len))
    tlc_v = np.zeros((mass_len, time_len))
    tau = np.zeros((mass_len, time_len))
    v_photosphere = np.zeros(time_len)
    r_photosphere = np.zeros(time_len)

    dt = np.diff(time)

    # Initial conditions: set initial thermal energy from kinetic energy,
    # or zero because stability
    energy_v[:, 0] = 0.  # 0.5 * m_array * solar_mass * v_m ** 2

    # Loop over time steps: update energy and luminosity in each shell.
    for ii in range(time_len - 1):
        if use_gray_opacity:
            kappa_eff = kappa
        else:
            # For the first time step, use κ_max; thereafter update using an effective temperature.
            if ii == 0:
                kappa_eff = kappa_max
            else:
                # Estimate a global effective luminosity and photospheric radius from previous step.
                L_bol_prev = np.sum(lum_rad[:, ii - 1])
                # print(L_bol_prev)
                # For safety, if r_photosphere was not set (or is zero), use temperature_floor.
                if r_photosphere[ii - 1] > 0:
                    T_eff_prev = (L_bol_prev / (4.0 * np.pi * (r_photosphere[ii - 1]) ** 2 * sigma_sb)) ** 0.25
                else:
                    T_eff_prev = temperature_floor
                # Update effective opacity using the temperature-dependent formula:
                # When T_eff ≫ T_floor, κ_eff ~ κ_max; when T_eff ≪ T_floor, κ_eff → κ_min.
                # kappa_eff = kappa_min + (kappa_max - kappa_min) / \
                #             (1.0 + (temperature_floor / T_eff_prev) ** kappa_n)
                kappa_eff = kappa_min + 0.5 * (kappa_max - kappa_min) * \
                            (1.0 + np.tanh((T_eff_prev - temperature_floor) / (50000)))
        # print(kappa_eff)
        td_v[:, ii] = (kappa_eff * m_array * solar_mass * 3) / \
                      (4 * np.pi * v_m * speed_of_light * time[ii] * diffusion_beta)
        # Add minimum diffusion time to prevent instability
        min_diffusion_time = dt[ii] * 1  # Minimum 10x timestep
        td_v[:, ii] = np.maximum(td_v[:, ii], min_diffusion_time)

        tau[:, ii] = (m_array * solar_mass * kappa_eff) / (4 * np.pi * (time[ii] * v_m) ** 2)
        leakage = 3 * kappa_gamma * m_array * solar_mass / (4 * np.pi * v_m ** 2)
        eth_v[:, ii] = 1 - np.exp(-leakage * time[ii] ** (-2))
        qdot_ni[:, ii] = ni_array * edotr[:, ii] * eth_v[:, ii]
        tlc_v[:, ii] = v_m * time[ii] / speed_of_light

        # Prevent division by very small numbers
        t_total = td_v[:, ii] + tlc_v[:, ii]
        lum_rad[:, ii] = energy_v[:, ii] / np.maximum(t_total, dt[ii])
        # Euler integration update for energy in each shell
        energy_change = (qdot_ni[:, ii] - (energy_v[:, ii] / time[ii]) - lum_rad[:, ii]) * dt[ii]

        # Limit energy change to prevent instability (max 50% change per timestep)
        max_change = 0.5 * np.abs(energy_v[:, ii]) + 1e40  # Add small floor
        energy_change = np.clip(energy_change, -max_change, max_change)

        energy_v[:, ii + 1] = energy_v[:, ii] + energy_change
        # Ensure energy stays positive
        energy_v[:, ii + 1] = np.maximum(energy_v[:, ii + 1], 0)

        # Determine the photospheric shell by finding where τ ≃ 1
        photosphere_index = np.argmin(np.abs(tau[:, ii] - 1))
        v_photosphere[ii] = v_m[photosphere_index]
        r_photosphere[ii] = v_photosphere[ii] * time[ii]

    # Final step for luminosity in the last time index
    lum_rad[:, -1] = energy_v[:, -1] / (td_v[:, -1] + tlc_v[:, -1])
    bolometric_luminosity = np.sum(lum_rad, axis=0)

    # # Compute the effective temperature from the global bolometric luminosity and photospheric radius
    temperature = (bolometric_luminosity / (4.0 * np.pi * (r_photosphere) ** 2 * sigma_sb)) ** 0.25
    # Apply a temperature floor
    mask = temperature < temperature_floor
    temperature[mask] = temperature_floor
    r_photosphere[mask] = np.sqrt(bolometric_luminosity[mask] / (4.0 * np.pi * temperature_floor ** 4 * sigma_sb))

    from collections import namedtuple
    outputs = namedtuple('output', ['time_temp', 'lbol', 't_photosphere',
                                    'r_photosphere', 'tau', 'v_photosphere', 'energy_v'])
    outputs.time_temp = time[1:-1] / day_to_s
    outputs.lbol = bolometric_luminosity[1:-1]
    outputs.t_photosphere = temperature[1:-1]
    outputs.r_photosphere = r_photosphere[1:-1]
    outputs.tau = tau[:, 1:-1]
    outputs.energy_v = energy_v[:, 1:-1]
    outputs.v_photosphere = v_photosphere[1:-1]
    return outputs

def nickelmixing_bolometric(time, mej, esn, kappa, kappa_gamma, f_nickel, f_mixing,
                            temperature_floor, **kwargs):
    """
    Bolometric light curve model for supernova with nickel mixing.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in days.
    mej : float
        Ejecta mass in solar masses.
    esn : float
        Explosion energy in foe (10^51 erg).
    kappa : float
        Gray opacity in cm²/g.
    kappa_gamma : float
        Gamma-ray opacity in cm²/g.
    f_nickel : float
        Fraction of nickel mass.
    f_mixing : float
        Fraction of nickel mass that is mixed.
        Low values put all nickel in the first shell.
    temperature_floor : float
        Floor temperature in K.
    **kwargs : dict
        Additional keyword arguments:

        - beta : float
            Power law slope for mass distribution (m = m_0 * (v/v_min)^(-beta)).
        - stop_time : float
            Time to stop calculation. Default is 300 days.
        - mass_len : int
            Number of mass shells. Default is 200.
        - vmax : float
            Maximum velocity in km/s. Default is 100000.
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    dense_resolution = kwargs.get("dense_resolution", 1000)
    stop_time = kwargs.get("stop_time", 300)
    time_temp = np.geomspace(0.01, int(stop_time), int(dense_resolution))
    outputs = _nickelmixing(time_temp * 86400, mej=mej, esn=esn, kappa=kappa,
                               kappa_gamma=kappa_gamma, f_nickel=f_nickel,
                               f_mixing=f_mixing, temperature_floor=temperature_floor, **kwargs)
    lbol = outputs.lbol
    temp_times = outputs.time_temp
    func = interp1d(temp_times, lbol, kind='cubic', fill_value='extrapolate')
    return func(time)

@citation_wrapper('Sarin (in prep)')
def nickelmixing(time, redshift, mej, esn, kappa, kappa_gamma, f_nickel, f_mixing,
                 temperature_floor, **kwargs):
    """
    Supernova model with radioactive decay and nickel mixing.

    Multi-shell model including nickel mixing and stratified ejecta.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    esn : float
        Explosion energy in foe (10^51 erg).
    kappa : float
        Gray opacity in cm²/g.
    kappa_gamma : float
        Gamma-ray opacity in cm²/g.
    f_nickel : float
        Fraction of nickel mass.
    f_mixing : float
        Fraction of nickel mass that is mixed.
        Low values put all nickel in the first shell.
    temperature_floor : float
        Floor temperature in K.
    **kwargs : dict
        Additional keyword arguments:

        - beta : float
            Power law slope for mass distribution (m = m_0 * (v/v_min)^(-beta)).
        - mass_len : int
            Number of mass shells. Default is 200.
        - vmax : float
            Maximum velocity in km/s. Default is 100000.
        - stop_time : float
            Time to stop calculation. Default is 300 days.
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Note: Currently uses FlatLambdaCDM(H0=73, Om0=0.3).

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    # cosmology = kwargs.get('cosmology', cosmo)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=73, Om0=0.3)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    # dl = cosmology.luminosity_distance(redshift).cgs.value
    dense_resolution = kwargs.get("dense_resolution", 1000)
    stop_time = kwargs.get("stop_time", 300)
    time_temp = np.geomspace(0.01, int(stop_time), int(dense_resolution))
    outputs = _nickelmixing(time_temp * 86400, mej=mej, esn=esn, kappa=kappa,
                                                     kappa_gamma=kappa_gamma, f_nickel=f_nickel,
                                                     f_mixing=f_mixing, temperature_floor=temperature_floor, **kwargs)
    time_temp = outputs.time_temp
    temperature = outputs.t_photosphere
    r_photosphere = outputs.r_photosphere
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        temp_func = interp1d(time_temp, y=temperature)
        rad_func = interp1d(time_temp, y=r_photosphere)
        temp = temp_func(time)
        rad = rad_func(time)
        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=rad, frequency=frequency, dl=dl)
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        spectra = blackbody_to_spectrum(
            temperature=temperature,
            r_photosphere=r_photosphere,
            frequency=frequency[:, None],
            dl=dl,
            redshift=redshift,
            lambda_observer_frame=lambda_observer_frame
        )
    if kwargs['output_format'] == 'spectra':
        return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                    lambdas=lambda_observer_frame,
                                                                    spectra=spectra)
    else:
        return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                      spectra=spectra, lambda_array=lambda_observer_frame,
                                                      **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett_bolometric(time, f_nickel, mej, **kwargs):
    """
    Bolometric Arnett model for nickel-powered supernovae.

    Classic Arnett (1982) model for radioactive decay heating.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can be None for raw engine luminosity.
        - kappa : float
            Opacity (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion).
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
    lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
        interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols, mej=mej, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett(time, redshift, f_nickel, mej, **kwargs):
    """
    Arnett model for nickel-powered supernovae.

    Classic Arnett (1982) model for radioactive decay heating with photosphere and SED.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature in K.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.

    Notes
    -----
    Based on Arnett (1982), ApJ 253, 785.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett_with_features(time, redshift, f_nickel, mej, **kwargs):
    """
    A version of the arnett model where SED has time-evolving spectral features.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is BlackbodyWithSpectralFeatures.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion and TemperatureFloor).
        - temperature_floor : float
            Floor temperature in K (required for TemperatureFloor).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.
        - feature_list : list, optional
            Optional list of spectral features. If None, uses default Type Ia features.
        - evolution_mode : str
            'smooth' or 'sharp'. Default is 'smooth'.
        - use_default_features : bool
            If True and no custom features found, use defaults. Default is True.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.

    Notes
    -----
    Feature Parameters (dynamically numbered):
    Features are defined by groups of parameters with pattern: {param}_feature_{N}
    where N starts from 1. All features with the same N are grouped together.

    Required for each feature N:
        - rest_wavelength_feature_N : float
            Central wavelength in Angstroms
        - sigma_feature_N : float
            Gaussian width in Angstroms
        - amplitude_feature_N : float
            Amplitude (negative=absorption, positive=emission), percentage of continuum (e.g., -0.4 = 40% absorption)
        - t_start_feature_N : float
            Start time in source-frame days
        - t_end_feature_N : float
            End time in source-frame days

    Optional for each feature N (smooth mode only):
        - t_rise_feature_N : float
            Rise time in source-frame days (default: 2.0)
        - t_fall_feature_N : float
            Fall time in source-frame days (default: 5.0)

    Examples
    --------
    Single custom feature:
    >>> result = model(time, z, f_ni, mej,
    ...                rest_wavelength_feature_1=6355.0,
    ...                sigma_feature_1=400.0,
    ...                amplitude_feature_1=-0.4,
    ...                t_start_feature_1=0,
    ...                t_end_feature_1=30,
    ...                output_format='magnitude', bands='lsstg')

    Multiple features:
    >>> result = model(time, z, f_ni, mej,
    ...                rest_wavelength_feature_1=6355.0, sigma_feature_1=400.0,
    ...                amplitude_feature_1=-0.4, t_start_feature_1=0, t_end_feature_1=40,
    ...                rest_wavelength_feature_2=3934.0, sigma_feature_2=300.0,
    ...                amplitude_feature_2=-0.5, t_start_feature_2=0, t_end_feature_2=60,
    ...                rest_wavelength_feature_3=8600.0, sigma_feature_3=500.0,
    ...                amplitude_feature_3=-0.3, t_start_feature_3=0, t_end_feature_3=50,
    ...                evolution_mode='smooth',
    ...                output_format='magnitude', bands='lsstg')
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.BlackbodyWithSpectralFeatures)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    # Build feature list from numbered parameters
    feature_list = build_spectral_feature_list(**kwargs)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)

        # Convert time from days to seconds for feature application
        time_seconds = time * 24 * 3600

        sed_1 = kwargs['sed'](
            temperature=photo.photosphere_temperature,
            r_photosphere=photo.r_photosphere,
            frequency=frequency,
            luminosity_distance=dl,
            time=time_seconds,
            feature_list=feature_list,
            evolution_mode=kwargs.get('evolution_mode', 'smooth')
        )
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 3000)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(
            frequency=lambda_to_nu(lambda_observer_frame),
            redshift=redshift,
            time=time_observer_frame
        )
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)

        # Convert time from days to seconds for feature application
        time_seconds = time * 24 * 3600

        sed_1 = kwargs['sed'](
            temperature=photo.photosphere_temperature,
            r_photosphere=photo.r_photosphere,
            frequency=frequency[:, None],
            luminosity_distance=dl,
            time=time_seconds,
            feature_list=feature_list,
            evolution_mode=kwargs.get('evolution_mode', 'smooth')
        )
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)

        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(
                time=time_observer_frame,
                lambdas=lambda_observer_frame,
                spectra=spectra
            )
        else:
            return sed.get_correct_output_format_from_spectra(
                time=time_obs,
                time_eval=time_observer_frame,
                spectra=spectra,
                lambda_array=lambda_observer_frame,
                **kwargs
            )

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract, Piro+2021')
def shock_cooling_and_arnett_bolometric(time, log10_mass, log10_radius, log10_energy,
                             f_nickel, mej, vej, kappa, kappa_gamma, temperature_floor, **kwargs):
    """
    Combined shock cooling and Arnett bolometric luminosity model.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    log10_mass : float
        Log10 mass of extended material in solar masses.
    log10_radius : float
        Log10 radius of extended material in cm.
    log10_energy : float
        Log10 energy of extended material in erg.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    vej : float
        Velocity of ejecta in km/s.
    kappa : float
        Opacity in cm²/g.
    kappa_gamma : float
        Gamma-ray opacity in cm²/g.
    temperature_floor : float
        Temperature floor in K.
    **kwargs : dict
        Additional keyword arguments:

        - nn : float
            Density power law slope.
        - delta : float
            Inner density power law slope.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    from redback.transient_models.shock_powered_models import shock_cooling_bolometric
    lbol_1 = shock_cooling_bolometric(time=time * day_to_s, log10_mass=log10_mass, log10_radius=log10_radius,
                                      log10_energy=log10_energy, **kwargs)
    lbol_2 = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, vej=vej, kappa=kappa,
                               kappa_gamma=kappa_gamma, temperature_floor=temperature_floor, **kwargs)
    return lbol_1 + lbol_2


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract, Piro+2021')
def shock_cooling_and_arnett(time, redshift, log10_mass, log10_radius, log10_energy,
                             f_nickel, mej, vej, kappa, kappa_gamma, temperature_floor, **kwargs):
    """
    Photometric light curve of shock cooling and arnett model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    log10_mass : float
        Log10 mass of extended material in solar masses.
    log10_radius : float
        Log10 radius of extended material in cm.
    log10_energy : float
        Log10 energy of extended material in erg.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    vej : float
        Velocity of ejecta in km/s.
    kappa : float
        Opacity in cm^2/g.
    kappa_gamma : float
        Gamma-ray opacity in cm^2/g.
    temperature_floor : float
        Temperature floor in K.
    **kwargs : dict
        Additional keyword arguments:

        - nn : float
            Density power law slope.
        - delta : float
            Inner density power law slope.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = shock_cooling_and_arnett_bolometric(time, log10_mass=log10_mass, log10_radius=log10_radius,
                                                   log10_energy=log10_energy, f_nickel=f_nickel,
                                                   mej=mej, vej=vej, kappa=kappa, kappa_gamma=kappa_gamma,
                                                   temperature_floor=temperature_floor, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, vej=vej, temperature_floor=temperature_floor, **kwargs)

        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = shock_cooling_and_arnett_bolometric(time, log10_mass=log10_mass, log10_radius=log10_radius,
                                                   log10_energy=log10_energy, f_nickel=f_nickel,
                                                   mej=mej, vej=vej, kappa=kappa, kappa_gamma=kappa_gamma,
                                                   temperature_floor=temperature_floor, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, vej=vej, temperature_floor=temperature_floor, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://academic.oup.com/mnras/article/522/2/2764/7086123#443111844, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def shockcooling_morag_and_arnett_bolometric(time, v_shock, m_env, mej, f_rho, f_nickel, radius, kappa, **kwargs):
    """
    Shock cooling following Morag+ and Arnett model for radioactive decay.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in days.
    v_shock : float
        Shock speed in km/s, also the ejecta velocity in the Arnett calculation.
    m_env : float
        Envelope mass in solar masses.
    mej : float
        Ejecta mass in solar masses.
    f_rho : float
        f_rho parameter. Typically of order unity.
    f_nickel : float
        Fraction of nickel mass.
    radius : float
        Star/envelope radius in units of 10^13 cm.
    kappa : float
        Opacity in cm^2/g.
    **kwargs : dict
        Additional keyword arguments required by the model.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    from redback.transient_models.shock_powered_models import shockcooling_morag_bolometric
    f_rho_m = f_rho * mej
    nickel_lbol = arnett_bolometric(time=time, f_nickel=f_nickel,
                                    mej=mej, interaction_process=ip.Diffusion, kappa=kappa, vej=v_shock, **kwargs)
    sbo_output = shockcooling_morag_bolometric(time=time, v_shock=v_shock, m_env=m_env, f_rho_m=f_rho_m,
                                                     radius=radius, kappa=kappa, **kwargs)
    lbol = nickel_lbol + sbo_output
    return lbol

@citation_wrapper('https://academic.oup.com/mnras/article/522/2/2764/7086123#443111844, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def shockcooling_morag_and_arnett(time, redshift, v_shock, m_env, mej, f_rho, f_nickel, radius, kappa, **kwargs):
    """
    Shock cooling following Morag+ and Arnett model for radioactive decay.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    v_shock : float
        Shock speed in km/s, also the ejecta velocity in the Arnett calculation.
    m_env : float
        Envelope mass in solar masses.
    mej : float
        Ejecta mass in solar masses.
    f_rho : float
        f_rho parameter. Typically of order unity.
    f_nickel : float
        Fraction of nickel mass.
    radius : float
        Star/envelope radius in units of 10^13 cm.
    kappa : float
        Opacity in cm^2/g.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = shockcooling_morag_and_arnett_bolometric(time=time, v_shock=v_shock, m_env=m_env, mej=mej,
                                                        f_rho=f_rho, f_nickel=f_nickel, radius=radius, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_shock, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = shockcooling_morag_and_arnett_bolometric(time=time, v_shock=v_shock, m_env=m_env, mej=mej,
                                                        f_rho=f_rho, f_nickel=f_nickel, radius=radius, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_shock, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:, None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://iopscience.iop.org/article/10.3847/1538-4357/aa64df, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def shockcooling_sapirwaxman_and_arnett_bolometric(time, v_shock, m_env, mej, f_rho, f_nickel, radius, kappa, **kwargs):
    """
    Shock cooling following Sapir and Waxman and Arnett model for radioactive decay.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in days.
    v_shock : float
        Shock speed in km/s, also the ejecta velocity in the Arnett calculation.
    m_env : float
        Envelope mass in solar masses.
    mej : float
        Ejecta mass in solar masses.
    f_rho : float
        f_rho parameter. Typically of order unity.
    f_nickel : float
        Fraction of nickel mass.
    radius : float
        Star/envelope radius in units of 10^13 cm.
    kappa : float
        Opacity in cm^2/g.
    **kwargs : dict
        Additional keyword arguments:

        - n : float
            Index of progenitor density profile, 1.5 (default) or 3.0.
        - RW : bool
            If True, use the simplified Rabinak & Waxman formulation (off by default).

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    from redback.transient_models.shock_powered_models import shockcooling_sapirandwaxman_bolometric
    f_rho_m = f_rho * mej
    nickel_lbol = arnett_bolometric(time=time, f_nickel=f_nickel,
                                    mej=mej, interaction_process=ip.Diffusion, kappa=kappa, vej=v_shock, **kwargs)
    sbo_output = shockcooling_sapirandwaxman_bolometric(time=time, v_shock=v_shock, m_env=m_env, f_rho_m=f_rho_m,
                                                     radius=radius, kappa=kappa, **kwargs)
    lbol = nickel_lbol + sbo_output
    return lbol

@citation_wrapper('https://iopscience.iop.org/article/10.3847/1538-4357/aa64df, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def shockcooling_sapirwaxman_and_arnett(time, redshift, v_shock, m_env, mej, f_rho, f_nickel, radius, kappa, **kwargs):
    """
    Shock cooling following Sapir and Waxman and Arnett model for radioactive decay.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    v_shock : float
        Shock speed in km/s, also the ejecta velocity in the Arnett calculation.
    m_env : float
        Envelope mass in solar masses.
    mej : float
        Ejecta mass in solar masses.
    f_rho : float
        f_rho parameter. Typically of order unity.
    f_nickel : float
        Fraction of nickel mass.
    radius : float
        Star/envelope radius in units of 10^13 cm.
    kappa : float
        Opacity in cm^2/g.
    **kwargs : dict
        Additional keyword arguments:

        - n : float
            Index of progenitor density profile, 1.5 (default) or 3.0.
        - RW : bool
            If True, use the simplified Rabinak & Waxman formulation (off by default).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = shockcooling_morag_and_arnett_bolometric(time=time, v_shock=v_shock, m_env=m_env, mej=mej,
                                                        f_rho=f_rho, f_nickel=f_nickel, radius=radius, kappa=kappa,
                                                        **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_shock, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = shockcooling_morag_and_arnett_bolometric(time=time, v_shock=v_shock, m_env=m_env, mej=mej,
                                                        f_rho=f_rho, f_nickel=f_nickel, radius=radius, kappa=kappa,
                                                        **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_shock, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:, None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('redback')
def basic_magnetar_powered_bolometric(time, p0, bp, mass_ns, theta_pb, **kwargs):
    """
    Basic magnetar-powered bolometric luminosity model.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    p0 : float
        Initial spin period in seconds.
    bp : float
        Polar magnetic field strength in Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion).
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
    lbol = basic_magnetar(time=time * day_to_s, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = basic_magnetar(time=dense_times * day_to_s, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
        interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols,**kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract')
def basic_magnetar_powered(time, redshift, p0, bp, mass_ns, theta_pb,**kwargs):
    """
    Basic magnetar-powered supernova model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    p0 : float
        Initial spin period in seconds.
    bp : float
        Polar magnetic field strength in Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion and TemperatureFloor).
        - temperature_floor : float
            Floor temperature in K (required for TemperatureFloor).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = basic_magnetar_powered_bolometric(time=time, p0=p0,bp=bp, mass_ns=mass_ns, theta_pb=theta_pb,**kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = basic_magnetar_powered_bolometric(time=time, p0=p0,bp=bp, mass_ns=mass_ns, theta_pb=theta_pb,**kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('redback')
def slsn_bolometric(time, p0, bp, mass_ns, theta_pb,**kwargs):
    """
    Superluminous supernova (SLSN) bolometric luminosity model with magnetar constraint.

    Same as basic_magnetar_powered but with constraint on rotational_energy/kinetic_energy and nebula phase.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    p0 : float
        Initial spin period in seconds.
    bp : float
        Polar magnetic field strength in Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion).

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    return basic_magnetar_powered_bolometric(time=time, p0=p0, bp=bp, mass_ns=mass_ns,
                                             theta_pb=theta_pb, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract')
def slsn(time, redshift, p0, bp, mass_ns, theta_pb,**kwargs):
    """
    Superluminous supernova (SLSN) model with magnetar constraint.

    Same as basic_magnetar_powered but with constraint on rotational_energy/kinetic_energy and nebula phase.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    p0 : float
        Initial spin period in seconds.
    bp : float
        Polar magnetic field strength in Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is CutoffBlackbody.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion and TemperatureFloor).
        - temperature_floor : float
            Floor temperature in K (required for TemperatureFloor).
        - cutoff_wavelength : float
            Cutoff wavelength in Angstroms for CutoffBlackbody. Default is 3000.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.CutoffBlackbody)   
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = slsn_bolometric(time=time, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](time=time, luminosity=lbol, temperature=photo.photosphere_temperature,
                    r_photosphere=photo.r_photosphere,frequency=frequency, luminosity_distance=dl,
                    cutoff_wavelength=cutoff_wavelength)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = slsn_bolometric(time=time, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        full_sed = np.zeros((len(time), len(frequency)))
        ss = kwargs['sed'](time=time, temperature=photo.photosphere_temperature,
                        r_photosphere=photo.r_photosphere, frequency=frequency[:, None],
                        luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)
        full_sed = ss.flux_density.to(uu.mJy).value.T
        full_sed = full_sed / (1 + redshift)
        spectra = (full_sed * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                    equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def magnetar_nickel(time, redshift, f_nickel, mej, p0, bp, mass_ns, theta_pb, **kwargs):
    """
    Magnetar-powered supernova with nickel decay.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    p0 : float
        Initial spin period in seconds.
    bp : float
        Polar magnetic field strength in Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion and TemperatureFloor).
        - temperature_floor : float
            Floor temperature in K (required for TemperatureFloor).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol_mag = basic_magnetar(time=time * day_to_s, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
        lbol_arnett = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
        lbol = lbol_mag + lbol_arnett

        if kwargs['interaction_process'] is not None:
            dense_resolution = kwargs.get("dense_resolution", 1000)
            dense_times = np.linspace(0, time[-1]+100, dense_resolution)
            dense_lbols = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
            dense_lbols += basic_magnetar(time=dense_times * day_to_s, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
            interaction_class = kwargs['interaction_process'](time=time, dense_times=dense_times, luminosity=dense_lbols,
                                                              mej=mej, **kwargs)
            lbol = interaction_class.new_luminosity

        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)

        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol_mag = basic_magnetar(time=time * day_to_s, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
        lbol_arnett = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
        lbol = lbol_mag + lbol_arnett
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


@citation_wrapper('redback')
def homologous_expansion_supernova_model_bolometric(time, mej, ek, **kwargs):
    """
    Homologous expansion supernova bolometric luminosity model.

    Assumes homologous expansion to transform kinetic energy to ejecta velocity.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    mej : float
        Ejecta mass in solar masses.
    ek : float
        Kinetic energy in erg.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - base_model : str or function
            Base model from homologous_expansion_models list. Default is 'arnett_bolometric'.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion).

    Returns
    -------
    lbol : np.ndarray or tuple
        Bolometric luminosity in erg/s. If output_format is in ['spectra', 'flux', 'flux_density', 'magnitude'],
        returns (lbol, kwargs) tuple.
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs.get('base_model', 'arnett_bolometric')
    if isfunction(base_model):
        function = base_model
    elif base_model not in homologous_expansion_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['supernova_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")
    v_ejecta = np.sqrt(10.0 * ek / (3.0 * mej * solar_mass)) / km_cgs
    kwargs['vej'] = v_ejecta
    kwargs['mej'] = mej
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    lbol = function(time, **kwargs)
    if kwargs['output_format'] in ['spectra', 'flux', 'flux_density', 'magnitude']:
        return lbol, kwargs
    else:
        return lbol

@citation_wrapper('redback')
def thin_shell_supernova_model_bolometric(time, mej, ek, **kwargs):
    """
    Thin shell ejecta supernova bolometric luminosity model.

    Assumes thin shell ejecta to transform kinetic energy into ejecta velocity.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    mej : float
        Ejecta mass in solar masses.
    ek : float
        Kinetic energy in erg.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - base_model : str or function
            Base model from homologous_expansion_models list. Default is 'arnett_bolometric'.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion).

    Returns
    -------
    lbol : np.ndarray or tuple
        Bolometric luminosity in erg/s. If output_format is in ['spectra', 'flux', 'flux_density', 'magnitude'],
        returns (lbol, kwargs) tuple.
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs.get('base_model', 'arnett_bolometric')
    if isfunction(base_model):
        function = base_model
    elif base_model not in homologous_expansion_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['supernova_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")
    v_ejecta = np.sqrt(2.0 * ek / (mej * solar_mass)) / km_cgs
    kwargs['vej'] = v_ejecta
    kwargs['mej'] = mej

    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    lbol = function(time, **kwargs)
    if kwargs['output_format'] in ['spectra', 'flux', 'flux_density', 'magnitude']:
        return lbol, kwargs
    else:
        return lbol


@citation_wrapper('redback')
def homologous_expansion_supernova(time, redshift, mej, ek, **kwargs):
    """
    Homologous expansion supernova model.

    Assumes homologous expansion to transform kinetic energy to ejecta velocity.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    ek : float
        Kinetic energy in erg.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - base_model : str or function
            Base model from homologous_expansion_models list. Default is 'arnett_bolometric'.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion and TemperatureFloor).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol, kwargs = homologous_expansion_supernova_model_bolometric(time=time, mej=mej, ek=ek, **kwargs)
        photo = kwargs['photosphere'] (time=time, luminosity=lbol, **kwargs)

        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol, kwargs = homologous_expansion_supernova_model_bolometric(time=time, mej=mej, ek=ek, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)

        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('redback')
def thin_shell_supernova(time, redshift, mej, ek, **kwargs):
    """
    Thin shell ejecta supernova model.

    Assumes thin shell ejecta to transform kinetic energy into ejecta velocity.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    ek : float
        Kinetic energy in erg.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - base_model : str or function
            Base model from homologous_expansion_models list. Default is 'arnett_bolometric'.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - temperature_floor : float
            Floor temperature in K (required for Diffusion and TemperatureFloor).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol, kwargs = thin_shell_supernova_model_bolometric(time=time, mej=mej, ek=ek, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol, kwargs = thin_shell_supernova_model_bolometric(time=time, mej=mej, ek=ek, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

def _csm_engine(time, mej, csm_mass, vej, eta, rho, kappa, r0, **kwargs):
    """
    CSM interaction engine calculating bolometric luminosity from CSM shock.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    mej : float
        Ejecta mass in solar masses.
    csm_mass : float
        CSM mass in solar masses.
    vej : float
        Ejecta velocity in km/s.
    eta : float
        CSM density profile exponent.
    rho : float
        CSM density profile amplitude.
    kappa : float
        Opacity in cm^2/g.
    r0 : float
        Radius of CSM shell in AU.
    **kwargs : dict
        Additional keyword arguments:

        - efficiency : float
            Efficiency in converting between kinetic energy and luminosity. Default is 0.5.
        - delta : float
            Inner density power law slope. Default is 1.
        - nn : float
            Ejecta density profile index. Default is 12.

    Returns
    -------
    csm_output : namedtuple
        Named tuple with 'lbol', 'r_photosphere', 'mass_csm_threshold'.
    """
    mej = mej * solar_mass
    csm_mass = csm_mass * solar_mass
    r0 = r0 * au_cgs
    vej = vej * km_cgs
    Esn = 3. * vej ** 2 * mej / 10.
    ti = 1.

    delta = kwargs.get('delta', 1)
    nn = kwargs.get('nn', 12)
    efficiency = kwargs.get('efficiency', 0.5)

    csm_properties = get_csm_properties(nn, eta)
    AA = csm_properties.AA
    Bf = csm_properties.Bf
    Br = csm_properties.Br

    # Derived parameters
    # scaling constant for CSM density profile
    qq = rho * r0 ** eta
    # outer CSM shell radius
    radius_csm = ((3.0 - eta) / (4.0 * np.pi * qq) * csm_mass + r0 ** (3.0 - eta)) ** (1.0 / (3.0 - eta))
    # photosphere radius
    r_photosphere = abs((-2.0 * (1.0 - eta) / (3.0 * kappa * qq) + radius_csm ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))

    # mass of the optically thick CSM (tau > 2/3).
    mass_csm_threshold = np.abs(4.0 * np.pi * qq / (3.0 - eta) * (
            r_photosphere ** (3.0 - eta) - r0 ** (3.0 - eta)))

    # g**n is scaling parameter for ejecta density profile
    g_n = (1.0 / (4.0 * np.pi * (nn - delta)) * (
            2.0 * (5.0 - delta) * (nn - 5.0) * Esn) ** ((nn - 3.) / 2.0) / (
                   (3.0 - delta) * (nn - 3.0) * mej) ** ((nn - 5.0) / 2.0))

    # time at which shock breaks out of optically thick CSM - forward shock
    t_FS = (abs((3.0 - eta) * qq ** ((3.0 - nn) / (nn - eta)) * (
            AA * g_n) ** ((eta - 3.0) / (nn - eta)) /
                (4.0 * np.pi * Bf ** (3.0 - eta))) ** (
                    (nn - eta) / ((nn - 3.0) * (3.0 - eta))) * (mass_csm_threshold) ** (
                    (nn - eta) / ((nn - 3.0) * (3.0 - eta))))

    # time at which reverse shock sweeps up all ejecta - reverse shock
    t_RS = (vej / (Br * (AA * g_n / qq) ** (
            1.0 / (nn - eta))) *
            (1.0 - (3.0 - nn) * mej /
             (4.0 * np.pi * vej **
              (3.0 - nn) * g_n)) ** (1.0 / (3.0 - nn))) ** (
                   (nn - eta) / (eta - 3.0))

    mask_RS = t_RS - time * day_to_s > 0
    mask_FS = t_FS - time * day_to_s > 0

    lbol_FS = 2.0 * np.pi / (nn - eta) ** 3 * g_n ** ((5.0 - eta) / (nn - eta)) * qq ** ((nn - 5.0) / (nn - eta)) \
              * (nn - 3.0) ** 2 * (nn - 5.0) * Bf ** (5.0 - eta) * AA ** ((5.0 - eta) / (nn - eta)) * (time * day_to_s + ti) \
              ** ((2.0 * nn + 6.0 * eta - nn * eta - 15.) / (nn - eta))

    lbol_RS = 2.0 * np.pi * (AA * g_n / qq) ** ((5.0 - nn) / (nn - eta)) * Br ** (5.0 - nn) * g_n * ((3.0 - eta) / (nn - eta)) \
              ** 3 * (time * day_to_s + ti) ** ((2.0 * nn + 6.0 * eta - nn * eta - 15.0) / (nn - eta))
    lbol_FS[~mask_FS] = 0
    lbol_RS[~mask_RS] = 0

    lbol = efficiency * (lbol_FS + lbol_RS)

    csm_output = namedtuple('csm_output', ['lbol', 'r_photosphere', 'mass_csm_threshold'])
    csm_output.lbol = lbol
    csm_output.r_photosphere = r_photosphere
    csm_output.mass_csm_threshold = mass_csm_threshold
    return csm_output


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract, https://ui.adsabs.harvard.edu/abs/2017ApJ...849...70V/abstract, https://ui.adsabs.harvard.edu/abs/2020RNAAS...4...16J/abstract')
def csm_interaction_bolometric(time, mej, csm_mass, vej, eta, rho, kappa, r0, **kwargs):
    """
    CSM interaction bolometric luminosity model.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    mej : float
        Ejecta mass in solar masses.
    csm_mass : float
        CSM mass in solar masses.
    vej : float
        Ejecta velocity in km/s.
    eta : float
        CSM density profile exponent.
    rho : float
        CSM density profile amplitude.
    kappa : float
        Opacity in cm^2/g.
    r0 : float
        Radius of CSM shell in AU.
    **kwargs : dict
        Additional keyword arguments:

        - efficiency : float
            Efficiency in converting between kinetic energy and luminosity. Default is 0.5.
        - delta : float
            Inner density power law slope. Default is 1.
        - nn : float
            Ejecta density profile index. Default is 12.
        - interaction_process : class
            Interaction process class. Default is CSMDiffusion.
            Can also be None for raw engine luminosity.
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    _interaction_process = kwargs.get("interaction_process", ip.CSMDiffusion)

    csm_output = _csm_engine(time=time, mej=mej, csm_mass=csm_mass, vej=vej,
                             eta=eta, rho=rho, kappa=kappa, r0=r0, **kwargs)
    lbol = csm_output.lbol

    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.geomspace(0.1, time[-1]+100, dense_resolution)
        csm_output = _csm_engine(time=dense_times, mej=mej, csm_mass=csm_mass, vej=vej,
                                 eta=eta, rho=rho, kappa=kappa, r0=r0, **kwargs)
        dense_lbols = csm_output.lbol
        r_photosphere = csm_output.r_photosphere
        mass_csm_threshold = csm_output.mass_csm_threshold
        interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols,
                                                kappa=kappa, r_photosphere=r_photosphere,
                                                mass_csm_threshold=mass_csm_threshold, csm_mass=csm_mass, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract, https://ui.adsabs.harvard.edu/abs/2017ApJ...849...70V/abstract, https://ui.adsabs.harvard.edu/abs/2020RNAAS...4...16J/abstract')
def csm_interaction(time, redshift, mej, csm_mass, vej, eta, rho, kappa, r0, **kwargs):
    """
    CSM interaction model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    csm_mass : float
        CSM mass in solar masses.
    vej : float
        Ejecta velocity in km/s.
    eta : float
        CSM density profile exponent.
    rho : float
        CSM density profile amplitude.
    kappa : float
        Opacity in cm^2/g.
    r0 : float
        Radius of CSM shell in AU.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is CSMDiffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g.
        - temperature_floor : float
            Floor temperature in K.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.CSMDiffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        lbol = csm_interaction_bolometric(time=time, mej=mej, csm_mass=csm_mass, vej=vej, eta=eta,
                                          rho=rho, kappa=kappa, r0=r0, **kwargs)

        photo = kwargs['photosphere'](time=time, luminosity=lbol, vej=vej, **kwargs)

        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.linspace(0.1, 500, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = csm_interaction_bolometric(time=time, mej=mej, csm_mass=csm_mass, vej=vej, eta=eta,
                                          rho=rho, kappa=kappa, r0=r0, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, vej=vej, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:, None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def csm_nickel(time, redshift, mej, f_nickel, csm_mass, ek, eta, rho, kappa, r0, **kwargs):
    """
    CSM interaction with nickel decay and homologous expansion.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    f_nickel : float
        Fraction of nickel mass.
    csm_mass : float
        CSM mass in solar masses.
    ek : float
        Kinetic energy in erg.
    eta : float
        CSM density profile exponent.
    rho : float
        CSM density profile amplitude.
    kappa : float
        Opacity in cm^2/g.
    r0 : float
        Radius of CSM shell in AU.
    **kwargs : dict
        Additional keyword arguments:

        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g.
        - temperature_floor : float
            Floor temperature in K.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    vej = np.sqrt(2.0 * ek / (mej * solar_mass)) / km_cgs

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        nickel_lbol = arnett_bolometric(time=time, f_nickel=f_nickel,
                                        mej=mej, interaction_process=ip.Diffusion, kappa=kappa, vej=vej, **kwargs)
        csm_lbol = csm_interaction_bolometric(time=time, mej=mej, csm_mass=csm_mass, eta=eta,
                                          rho=rho, kappa=kappa, r0=r0, vej=vej, interaction_process=ip.CSMDiffusion, **kwargs)
        lbol = nickel_lbol + csm_lbol

        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=vej, **kwargs)

        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency, luminosity_distance=dl)

        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        nickel_lbol = arnett_bolometric(time=time, f_nickel=f_nickel,
                                        mej=mej, interaction_process=ip.Diffusion,kappa=kappa, vej=vej, **kwargs)
        csm_lbol = csm_interaction_bolometric(time=time, mej=mej, csm_mass=csm_mass, eta=eta,
                                          rho=rho, kappa=kappa, r0=r0, vej=vej, interaction_process=ip.CSMDiffusion, **kwargs)
        lbol = nickel_lbol + csm_lbol
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=vej, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def type_1a(time, redshift, f_nickel, mej, **kwargs):
    """
    Type Ia supernova model with line absorption and cutoff blackbody SED.

    A nickel powered explosion with line absorption and cutoff blackbody SED for SNe Ia.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Ejecta mass in solar masses.
    **kwargs : dict
        Additional keyword arguments:

        - kappa : float
            Opacity in cm^2/g.
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g.
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature in K.
        - cutoff_wavelength : float
            Cutoff wavelength in Angstroms. Default is 3000.
        - line_wavelength : float
            Line wavelength in Angstroms. Default is 7500 in observer frame.
        - line_width : float
            Line width in Angstroms. Default is 500.
        - line_time : float
            Line time in days. Default is 50.
        - line_duration : float
            Line duration in days. Default is 25.
        - line_amplitude : float
            Line amplitude. Default is 0.3.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    line_wavelength = kwargs.get('line_wavelength', 7.5e3)
    line_width = kwargs.get('line_width', 500)
    line_time = kwargs.get('line_time', 50)
    line_duration = kwargs.get('line_duration', 25)
    line_amplitude = kwargs.get('line_amplitude', 0.3)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej,
                                 interaction_process=ip.Diffusion, **kwargs)

        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, **kwargs)
        sed_1 = sed.CutoffBlackbody(time=time, luminosity=lbol, temperature=photo.photosphere_temperature,
                                    r_photosphere=photo.r_photosphere, frequency=frequency, luminosity_distance=dl,
                                    cutoff_wavelength=cutoff_wavelength)
        sed_2 = sed.Line(time=time, luminosity=lbol, frequency=frequency, luminosity_distance=dl,
                         sed=sed_1, line_wavelength=line_wavelength,
                         line_width=line_width, line_time=line_time,
                         line_duration=line_duration, line_amplitude=line_amplitude)
        flux_density = sed_2.flux_density.flatten()
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambdas_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambdas_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej,
                                 interaction_process=ip.Diffusion, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, **kwargs)
        # Here we construct the CutoffBlackbody SED with frequency reshaped to (n_freq, 1)
        ss = sed.CutoffBlackbody(time=time, temperature=photo.photosphere_temperature,
                                 r_photosphere=photo.r_photosphere, frequency=frequency[:, None],
                                 luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)
        line_sed = sed.Line(time=time, luminosity=lbol, frequency=frequency[:, None],
                            luminosity_distance=dl, sed=ss,
                            line_wavelength=line_wavelength,
                            line_width=line_width,
                            line_time=line_time,
                            line_duration=line_duration,
                            line_amplitude=line_amplitude)
        full_sed = line_sed.flux_density.to(uu.mJy).value
        full_sed = full_sed / (1 + redshift)
        # The following line converts the full SED (in mJy) to erg/s/cm^2/Angstrom.
        spectra = (full_sed * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(
                                             wav=(lambdas_observer_frame.reshape(-1, 1) * uu.Angstrom))).T
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambdas_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambdas_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def type_1c(time, redshift, f_nickel, mej, pp, **kwargs):
    """
    Type Ic supernova model with synchrotron and blackbody SEDs.

    A nickel powered explosion with synchrotron and blackbody SEDs for SNe Ic.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Ejecta mass in solar masses.
    pp : float
        Power law index for synchrotron.
    **kwargs : dict
        Additional keyword arguments:

        - kappa : float
            Opacity in cm^2/g.
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g.
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature in K.
        - nu_max : float
            Maximum frequency for synchrotron in Hz. Default is 1e9.
        - f0 : float
            Synchrotron normalization. Default is 1e-26.
        - source_radius : float
            Source radius in cm. Default is 1e13.
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    nu_max = kwargs.get('nu_max', 1e9)
    f0 = kwargs.get('f0', 1e-26)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej,
                                 interaction_process=ip.Diffusion, **kwargs)

        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature,
                              r_photosphere=photo.r_photosphere,frequency=frequency, luminosity_distance=dl)
        sed_2 = sed.Synchrotron(frequency=frequency, luminosity_distance=dl, pp=pp, nu_max=nu_max,f0=f0)
        flux_density = sed_1.flux_density + sed_2.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej,
                                 interaction_process=ip.Diffusion, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature,
                              r_photosphere=photo.r_photosphere,frequency=frequency[:,None], luminosity_distance=dl)
        sed_2 = sed.Synchrotron(frequency=frequency[:, None], luminosity_distance=dl, pp=pp, nu_max=nu_max, f0=f0)
        flux_density = sed_1.flux_density + sed_2.flux_density
        fmjy = flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('redback')
def general_magnetar_slsn_bolometric(time, l0, tsd, nn, **kwargs):
    """
    General magnetar-powered SLSN bolometric luminosity model.

    Parameters
    ----------
    time : np.ndarray
        Time in days in source frame.
    l0 : float
        Magnetar energy normalization in erg.
    tsd : float
        Magnetar spin-down damping timescale in source frame days.
    nn : float
        Braking index.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion).
        - dense_resolution : int
            Resolution of dense time array. Default is 1000.

    Returns
    -------
    lbol : np.ndarray
        Bolometric luminosity in erg/s.
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)

    lbol = magnetar_only(time=time * day_to_s, l0=l0, tau=tsd * day_to_s, nn=nn)
    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = magnetar_only(time=dense_times * day_to_s, l0=l0, tau=tsd * day_to_s, nn=nn)
        interaction_class = _interaction_process(time=time, dense_times=dense_times,
                                                 luminosity=dense_lbols,**kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('redback')
def general_magnetar_slsn(time, redshift, l0, tsd, nn, ** kwargs):
    """
    General magnetar-powered SLSN model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    l0 : float
        Magnetar energy normalization in erg.
    tsd : float
        Magnetar spin-down damping timescale in source frame days.
    nn : float
        Braking index.
    **kwargs : dict
        Additional keyword arguments:

        - interaction_process : class
            Interaction process class. Default is Diffusion.
            Can also be None for raw engine luminosity.
        - photosphere : class
            Photosphere model. Default is TemperatureFloor.
        - sed : class
            SED model. Default is Blackbody.
        - kappa : float
            Opacity in cm^2/g (required for Diffusion).
        - kappa_gamma : float
            Gamma-ray opacity in cm^2/g (required for Diffusion).
        - vej : float
            Ejecta velocity in km/s (required for Diffusion and TemperatureFloor).
        - temperature_floor : float
            Floor temperature in K (required for TemperatureFloor).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            'flux_density', 'magnitude', 'spectra', or 'flux'.
        - lambda_array : np.ndarray
            Wavelength array in Angstroms. Default is np.geomspace(100, 60000, 100).
        - cosmology : astropy.cosmology object
            Cosmology to use. Default is Planck18.

    Returns
    -------
    output
        Format set by output_format: flux density (mJy), magnitude, spectra, or flux.
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = general_magnetar_slsn_bolometric(time=time, l0=l0, tsd=tsd, nn=nn, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency = frequency, luminosity_distance = dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = general_magnetar_slsn_bolometric(time=time, l0=l0, tsd=tsd, nn=nn, ** kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                    frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract, https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract')
def general_magnetar_driven_supernova_bolometric(time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, **kwargs):   
    """
    :param time: time in observer frame in days
    :param mej: ejecta mass in solar units
    :param E_sn: supernova explosion energy
    :param kappa: opacity
    :param l0: initial magnetar X-ray luminosity (Is this not the spin-down luminosity?)
    :param tau_sd: magnetar spin down damping timescale (days? seconds?)
    :param nn: braking index
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters - Must be all the kwargs required by the specific interaction_process  used
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s)
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param output_format: whether to output flux density or AB magnitude
    :param f_nickel: Ni^56 mass as a fraction of ejecta mass
    :return: bolometric luminsoity or dynamics output
    """              
    pair_cascade_switch = kwargs.get('pair_cascade_switch', False)
    time_temp = np.geomspace(1e0, 1e8, 2000)
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    beta = np.sqrt(E_sn / (0.5 * mej * solar_mass)) / speed_of_light
    ejecta_radius = 1.0e11
    n_ism = 1.0e-5
    
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, use_r_process=False, **kwargs)
    vej = velocity_from_lorentz_factor(output.lorentz_factor)/km_cgs 
    kwargs['vej'] = vej                                                                             
    lbol_func = interp1d(time_temp, y=output.bolometric_luminosity)
    vej_func = interp1d(time_temp, y=vej)
    time = time * day_to_s    
    lbol = lbol_func(time)
    v_ej = vej_func(time)
    
    dynamics_output = namedtuple('dynamics_output', ['v_ej', 'tau', 'time', 'bolometric_luminosity', 'kinetic_energy', 'erad_total',
                                                     'thermalisation_efficiency', 'magnetar_luminosity', 'erot_total'])

    dynamics_output.v_ej = v_ej
    dynamics_output.tau = output.tau
    dynamics_output.time = output.time
    dynamics_output.bolometric_luminosity = output.bolometric_luminosity
    dynamics_output.kinetic_energy = output.kinetic_energy 
    dynamics_output.erad_total = np.trapz(lbol, x=time)
    dynamics_output.thermalisation_efficiency = output.thermalisation_efficiency
    dynamics_output.magnetar_luminosity = magnetar_luminosity 
    dynamics_output.erot_total = np.trapz(magnetar_luminosity, x=time_temp)

    if kwargs['output_format'] == 'dynamics_output':
        return dynamics_output
    else:
        return lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract, https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract')
def general_magnetar_driven_supernova(time, redshift, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param E_sn: supernova explosion energy
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar spin-down luminosity (in erg/s)
    :param tau_sd: magnetar spin down damping timescale (in seconds)
    :param nn: braking index
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters - Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
             and CutoffBlackbody: cutoff_wavelength, default is 3000 Angstrom
    :param pair_cascade_switch: whether to account for pair cascade losses, default is False
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param output_format: whether to output flux density or AB magnitude
    :param frequency: (frequency to calculate - Must be same length as time array or a single number)
    :param f_nickel: Ni^56 mass as a fraction of ejecta mass
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is CutoffBlackbody.
    :return: flux density or AB magnitude or dynamics output
    """
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.CutoffBlackbody)
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    pair_cascade_switch = kwargs.get('pair_cascade_switch', False)
    ejecta_radius = 1.0e11
    n_ism = 1.0e-5
    
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        time_temp = np.geomspace(1e0, 1e8, 2000)
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
        beta = np.sqrt(E_sn / (0.5 * mej * solar_mass)) / speed_of_light
        output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, use_r_process=False, **kwargs)                                                                                
        vej = velocity_from_lorentz_factor(output.lorentz_factor)/km_cgs 
        kwargs['vej'] = vej                                      
        photo = kwargs['photosphere'](time=time_temp/day_to_s, luminosity=output.bolometric_luminosity, **kwargs)  
        temp_func = interp1d(time_temp/day_to_s, y=photo.photosphere_temperature)
        rad_func = interp1d(time_temp/day_to_s, y=photo.r_photosphere)
        bol_func = interp1d(time_temp/day_to_s, y=output.bolometric_luminosity)
        temp = temp_func(time)
        rad = rad_func(time)  
        lbol = bol_func(time)
        sed_1 = kwargs['sed'](time=time, luminosity=lbol, temperature=temp,
                                              r_photosphere=rad, frequency=frequency, luminosity_distance=dl,
                                              cutoff_wavelength=cutoff_wavelength) 
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(500, 60000, 200))
        time_temp = np.geomspace(1e0, 1e8, 2000)
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                              redshift=redshift, time=time_observer_frame)
        magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
        beta = np.sqrt(E_sn / (0.5 * mej * solar_mass)) / speed_of_light
        output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, use_r_process=False, **kwargs)
        vej = velocity_from_lorentz_factor(output.lorentz_factor)/km_cgs 
        kwargs['vej'] = vej
        photo = kwargs['photosphere'](time=time_temp/day_to_s, luminosity=output.bolometric_luminosity, **kwargs)
        if kwargs['output_format'] == 'dynamics_output':                                      
            erot_total = np.trapz(magnetar_luminosity, x=time_temp)
            erad_total = np.trapz(output.bolometric_luminosity, x=time_temp)          
            dynamics_output = namedtuple('dynamics_output', ['time', 'bolometric_luminosity', 'photosphere_temperature',
                                                     'radius', 'tau', 'kinetic_energy', 'erad_total', 'thermalisation_efficiency', 
                                                     'v_ej', 'magnetar_luminosity', 'erot_total', 'r_photosphere'])
            dynamics_output.time = output.time                                                                    
            dynamics_output.bolometric_luminosity = output.bolometric_luminosity
            dynamics_output.comoving_temperature = photo.photosphere_temperature
            dynamics_output.radius = output.radius
            dynamics_output.tau = output.tau
            dynamics_output.kinetic_energy = output.kinetic_energy
            dynamics_output.erad_total = erad_total
            dynamics_output.thermalisation_efficiency = output.thermalisation_efficiency                                        
            dynamics_output.v_ej = vej
            dynamics_output.magnetar_luminosity = magnetar_luminosity
            dynamics_output.erot_total = erot_total
            dynamics_output.r_photosphere = photo.r_photosphere                   
            return dynamics_output
        else: 
            full_sed = np.zeros((len(time), len(frequency)))
            ss = kwargs['sed'](time=time_temp/day_to_s, temperature=photo.photosphere_temperature,
                            r_photosphere=photo.r_photosphere, frequency=frequency[:, None],
                            luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=output.bolometric_luminosity)
            full_sed = ss.flux_density.to(uu.mJy).value.T
            full_sed = full_sed / (1 + redshift)
            spectra = (full_sed * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                        equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
            if kwargs['output_format'] == 'spectra':
                return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)   
            else: 
                return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                     spectra=spectra, lambda_array=lambda_observer_frame,
                                                     **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def csm_shock_and_arnett_bolometric(time, mej, f_nickel, csm_mass, v_min, beta, shell_radius,
                                    shell_width_ratio, kappa, **kwargs):
    """
    Assumes CSM interaction for a shell-like CSM with a hard outer boundary and arnett model for radioactive decay

    :param time: time in days in source frame
    :param mej: ejecta mass in solar masses
    :param f_nickel: fraction of nickel mass
    :param csm_mass: csm mass in solar masses
    :param v_min: ejecta velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param shell_radius: radius of shell in 10^14 cm
    :param kappa: opacity
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: kappa_gamma, temperature_floor, and any kwarg to
                change any other input physics/parameters from default.
    :return: bolometric luminosity in erg/s
    """
    from redback.transient_models.shock_powered_models import csm_shock_breakout_bolometric
    nickel_lbol = arnett_bolometric(time=time, f_nickel=f_nickel,
                                    mej=mej, interaction_process=ip.Diffusion, kappa=kappa, vej=v_min, **kwargs)
    sbo_output = csm_shock_breakout_bolometric(time=time, v_min=v_min, beta=beta,
                                    kappa=kappa, csm_mass=csm_mass, shell_radius=shell_radius,
                                    shell_width_ratio=shell_width_ratio, **kwargs)
    lbol = nickel_lbol + sbo_output
    return lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def csm_shock_and_arnett(time, redshift, mej, f_nickel, csm_mass, v_min, beta, shell_radius,
               shell_width_ratio, kappa, **kwargs):
    """
    Assumes CSM interaction for a shell-like CSM with a hard outer boundary and arnett model for radioactive decay
    Assumes one single photosphere from the sum of the bolometric luminosities

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param f_nickel: fraction of nickel mass
    :param csm_mass: csm mass in solar masses
    :param v_min: ejecta velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param shell_radius: radius of shell in 10^14 cm
    :param kappa: opacity
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: kappa_gamma, temperature_floor, and any kwarg to
                change any other input physics/parameters from default.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = csm_shock_and_arnett_bolometric(time=time, mej=mej, f_nickel=f_nickel, csm_mass=csm_mass,
                                               v_min=v_min, beta=beta, shell_radius=shell_radius,
                                               shell_width_ratio=shell_width_ratio, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_min, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = csm_shock_and_arnett_bolometric(time=time, mej=mej, f_nickel=f_nickel, csm_mass=csm_mass,
                                               v_min=v_min, beta=beta, shell_radius=shell_radius,
                                               shell_width_ratio=shell_width_ratio, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_min, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency[:, None], luminosity_distance=dl)
        fmjy = sed_1.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def csm_shock_and_arnett_two_rphots(time, redshift, mej, f_nickel, csm_mass, v_min, beta, shell_radius,
               shell_width_ratio, kappa, **kwargs):
    """
    Assumes CSM interaction for a shell-like CSM with a hard outer boundary and arnett model for radioactive decay.
    Assumes the photospheres for the CSM-interaction and the Arnett model are different.

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param f_nickel: fraction of nickel mass
    :param csm_mass: csm mass in solar masses
    :param v_min: ejecta velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param shell_radius: radius of shell in 10^14 cm
    :param kappa: opacity
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: kappa_gamma, temperature_floor, and any kwarg to
                change any other input physics/parameters from default.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        output = _csm_shock_breakout(time=time, csm_mass=csm_mass*solar_mass, v_min=v_min, beta=beta, kappa=kappa,
                                     shell_radius=shell_radius, shell_width_ratio=shell_width_ratio, **kwargs)
        r_phot = output.r_photosphere
        temp = output.temperature
        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=r_phot, dl=dl, frequency=frequency)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, vej=v_min, kappa=kappa,
                                 interaction_process=ip.Diffusion, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_min, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature,
                              r_photosphere=photo.r_photosphere, frequency=frequency, luminosity_distance=dl)
        flux_density += sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 300, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = _csm_shock_breakout(time=time, csm_mass=csm_mass * solar_mass, v_min=v_min, beta=beta, kappa=kappa,
                                     shell_radius=shell_radius, shell_width_ratio=shell_width_ratio, **kwargs)
        fmjy = sed.blackbody_to_flux_density(temperature=output.temperature,
                                             r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, vej=v_min, kappa=kappa,
                                 interaction_process=ip.Diffusion, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_min, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature,
                              r_photosphere=photo.r_photosphere, frequency=frequency[:, None], luminosity_distance=dl)
        fmjy += sed_1.flux_density
        fmjy = fmjy.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

def shocked_cocoon_and_arnett(time, redshift, mej_c, vej_c, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa,
                              mej, f_nickel, vej, **kwargs):
    """
    Emission from a shocked cocoon and arnett model for radioactive decay.
    We assume two different photospheres here.

    :param time: Time in days in observer frame
    :param redshift: redshift
    :param mej_c: cocoon mass (in solar masses)
    :param vej_c: cocoon material velocity (in c)
    :param eta: slope for the cocoon density profile
    :param tshock: shock breakout time (in seconds)
    :param shocked_fraction: fraction of the cocoon shocked
    :param cos_theta_cocoon: cosine of the cocoon opening angle
    :param kappa: opacity
    :param mej: supernova ejecta mass (in solar masses)
    :param f_nickel: fraction of nickel for ejecta mass
    :param vej: supernova ejecta velocity (in km/s)
    :param kwargs: Extra parameters used by model e.g., kappa_gamma, temperature_floor, and any kwarg to
                change any other input physics/parameters from default.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = _shocked_cocoon(time=time, mej=mej_c, vej=vej_c, eta=eta,
                                 tshock=tshock, shocked_fraction=shocked_fraction,
                                 cos_theta_cocoon=cos_theta_cocoon, kappa=kappa)
        flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                                     dl=dl, frequency=frequency)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, vej=vej,
                                 interaction_process=ip.Diffusion, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=vej, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature,
                              r_photosphere=photo.r_photosphere, frequency=frequency, luminosity_distance=dl)
        flux_density += sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        lambda_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 60000, 200))
        time_temp = np.linspace(1e-2, 300, 300)
        time_observer_frame = time_temp
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = _shocked_cocoon(time=time, mej=mej_c, vej=vej_c, eta=eta,
                                 tshock=tshock, shocked_fraction=shocked_fraction,
                                 cos_theta_cocoon=cos_theta_cocoon, kappa=kappa)
        fmjy = sed.blackbody_to_flux_density(temperature=output.temperature,
                                             r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, vej=vej,
                                 interaction_process=ip.Diffusion, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=vej, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature,
                              r_photosphere=photo.r_photosphere, frequency=frequency[:, None], luminosity_distance=dl)
        fmjy += sed_1.flux_density
        fmjy = fmjy.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_bolometric(time, progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Bolometric luminosity for a Type II supernova based on Sarin et al. 2025 surrogate model
    to stella grid in Moriya et al. 2023

    :param time: Time in days in source frame
    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: bolometric luminosity in erg/s
    """
    from redback_surrogates.supernovamodels import typeII_lbol
    tt, lbol = typeII_lbol(time=time, progenitor=progenitor, ni_mass=ni_mass,
                      log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn, **kwargs)
    lbol_func = interp1d(tt, y=lbol, bounds_error=False, fill_value='extrapolate')
    return lbol_func(time)

@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_photosphere_properties(time, progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Photosphere properties for a Type II supernova based on Sarin et al. 2025 surrogate model
    to stella grid in Moriya et al. 2023

    :param time: Time in days in source frame
    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: photosphere properties (temperature in K, radius in cm)
    """
    from redback_surrogates.supernovamodels import typeII_photosphere
    tt, temp, rad = typeII_photosphere(time=time, progenitor=progenitor, ni_mass=ni_mass,
                      log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn, **kwargs)
    temp_func = interp1d(tt, y=temp, bounds_error=False, fill_value='extrapolate')
    rad_func = interp1d(tt, y=rad, bounds_error=False, fill_value='extrapolate')
    return temp_func(time), rad_func(time)

@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_surrogate_sarin25(time, redshift, progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Type II supernova model based on Sarin et al. 2025 surrogate model
    to stella grid in Moriya et al. 2023

    :param time: Time in days in observer frame
    :param redshift: redshift
    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: Additional parameters for the model, such as:
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from redback_surrogates.supernovamodels import typeII_spectra
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs

    # Get the rest-frame spectrum using typeII_spectra
    spectra_output = typeII_spectra(
        progenitor=progenitor,
        ni_mass=ni_mass,
        log10_mdot=log10_mdot,
        beta=beta,
        rcsm=rcsm,
        esn=esn,
        **kwargs
    )

    # Extract components from the output (note: .frequency is actually wavelengths in Angstroms)
    luminosity_density = spectra_output.spectrum  # erg/s/Hz (luminosity density at rest-frame frequencies)
    lambda_rest = spectra_output.frequency.value  # Angstroms (rest-frame wavelengths)
    time_rest = spectra_output.time.value  # days (rest frame)

    # Apply cosmological dimming: L_nu / (4*pi*d_L^2) gives flux that still needs (1+z) correction
    flux_density = luminosity_density / (4 * np.pi * dl ** 2)

    # Handle different output formats
    if kwargs.get('output_format') == 'flux_density':
        # Use redback's K-correction utilities
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, time=time, redshift=redshift)

        # Convert rest-frame wavelengths to rest-frame frequencies for interpolation
        nu_rest = lambda_to_nu(lambda_rest)

        # Convert flux density to mJy
        fmjy = flux_density.to(uu.mJy).value

        # Create interpolator on rest-frame grid
        flux_interpolator = RegularGridInterpolator(
            (time_rest, nu_rest),
            fmjy,
            bounds_error=False,
            fill_value=0.0
        )

        # Prepare points for interpolation
        if isinstance(frequency, (int, float)):
            frequency = np.ones_like(time) * frequency

        # Create points for evaluation
        points = np.column_stack((time, frequency))

        # Return interpolated flux density with (1+z) correction for observer frame
        return flux_interpolator(points) / (1 + redshift)

    else:
        # Create denser grid for output (in rest frame)
        time_rest_dense = np.geomspace(np.min(time_rest), np.max(time_rest), 200)
        lambda_rest_dense = np.geomspace(np.min(lambda_rest), np.max(lambda_rest), 200)

        # Create interpolator for the flux density in rest frame
        flux_interpolator = RegularGridInterpolator(
            (time_rest, lambda_rest),
            flux_density.value,
            bounds_error=False,
            fill_value=0.0
        )

        # Create meshgrid for new grid points
        tt_mesh, ll_mesh = np.meshgrid(time_rest_dense, lambda_rest_dense, indexing='ij')
        points_to_evaluate = np.column_stack((tt_mesh.ravel(), ll_mesh.ravel()))

        # Interpolate flux density onto denser grid
        interpolated_values = flux_interpolator(points_to_evaluate)
        interpolated_flux = interpolated_values.reshape(tt_mesh.shape) * flux_density.unit

        # Convert to observer frame: times and wavelengths
        time_observer_frame = time_rest_dense * (1 + redshift)
        lambda_observer_frame = lambda_rest_dense * (1 + redshift)

        # Apply (1+z) correction to flux and convert units
        # After dividing by (1+z), flux is in observer frame at observer-frame wavelengths
        flux_observer = interpolated_flux / (1 + redshift)
        spectra = flux_observer.to(
            uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
            equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom)
        )

        # Create output structure
        if kwargs.get('output_format') == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(
                time=time_observer_frame,
                lambdas=lambda_observer_frame,
                spectra=spectra
            )
        else:
            # Get correct output format using redback utility
            return sed.get_correct_output_format_from_spectra(
                time=time,  # Original observer frame time for evaluation
                time_eval=time_observer_frame,
                spectra=spectra,
                lambda_array=lambda_observer_frame,
                time_spline_degree=1,
                **kwargs
            )