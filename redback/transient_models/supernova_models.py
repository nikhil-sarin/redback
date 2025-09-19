import numpy as np
import pandas as pd
from redback.transient_models.phenomenological_models import exponential_powerlaw, fallback_lbol
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar
from redback.transient_models.magnetar_driven_ejecta_models import _ejecta_dynamics_and_interaction
from redback.transient_models.shock_powered_models import  _shocked_cocoon, _csm_shock_breakout
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, logger, get_csm_properties, nu_to_lambda, lambda_to_nu, velocity_from_lorentz_factor
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
    A wrapper to SNCosmo models

    :param time: observer frame time in days
    :param redshift: redshift
    :param model_kwargs: all model keyword arguments in a dictionary
    :param kwargs: Additional keyword arguments for redback
    :param frequency: Frequency in Hz to evaluate model on, must be same shape as time array or a single value.
    :param sncosmo_model: String of the SNcosmo model to use.
    :param peak_time: SNe peak time in days
    :param cosmology: astropy cosmology object by default set to Planck18
    :param mw_extinction: Boolean for whether there is MW extinction or not. Default True
    :param host_extinction: Boolean for whether there is host extinction or not. Default True
            if used adds an extra parameter ebv which must also be in kwargs; host galaxy E(B-V). Set to 0.1 by default
    :param use_set_peak_magnitude: Boolean for whether to set the peak magnitude or not. Default False,
        if True the following keyword arguments also apply. Else the brightness is set by the model_kwargs.
    :param peak_abs_mag: SNe peak absolute magnitude default set to -19
    :param peak_abs_mag_band: Band corresponding to the peak abs mag limit, default to standard::b. Must be in SNCosmo
    :param magnitude_system: Mag system; default ab
    :return: set by output format - 'flux_density', 'magnitude', 'flux', 'sncosmo_source'
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
    A wrapper to the salt2 model in sncosmo

    :param time: time in days in observer frame (in mjd days)
    :param redshift: redshift
    :param x0: x0
    :param x1: x1
    :param c: c
    :param peak_time: peak time in mjd
    :param kwargs: Additional keyword arguments
    :param cosmology: astropy cosmology object by default set to Planck18
    :param mw_extinction: Boolean for whether there is MW extinction or not. Default True
    :param host_extinction: Boolean for whether there is host extinction or not. Default True
            if used adds an extra parameter ebv which must also be in kwargs; host galaxy E(B-V). Set to 0.1 by default
    :param use_set_peak_magnitude: Boolean for whether to set the peak magnitude or not. Default False,
        if True the following keyword arguments also apply. Else the brightness is set by the model_kwargs.
    :param peak_abs_mag: SNe peak absolute magnitude default set to -19
    :param peak_abs_mag_band: Band corresponding to the peak abs mag limit, default to standard::b. Must be in SNCosmo
    :param magnitude_system: Mag system; default ab
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :return: set by output format - 'flux_density', 'magnitude', 'flux', 'sncosmo_source'
    """
    kwargs['sncosmo_model'] = 'salt2'
    kwargs['peak_time'] = peak_time
    model_kwargs = {'x0':x0, 'x1':x1, 'c':c}
    out = sncosmo_models(time=time, redshift=redshift, model_kwargs=model_kwargs, **kwargs)
    return out

@citation_wrapper('redback')
def exponential_powerlaw_bolometric(time, lbol_0, alpha_1, alpha_2, tpeak_d, **kwargs):
    """
    :param time: rest frame time in days
    :param lbol_0: bolometric luminosity scale in cgs
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak_d: peak time in days
    :param kwargs: Must be all the kwargs required by the specific interaction_process
            e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
        Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
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

def sn_fallback(time, redshift, logl1, tr, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: source redshift
    :param logl1: bolometric luminosity scale in log10 (cgs)
    :param tr: transition time for luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
        e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, mej (solar masses), vej (km/s), floor temperature
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is ‘flux_density’.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is ‘magnitude’ or ‘flux’.
    :param output_format: ‘flux_density’, ‘magnitude’, ‘spectra’, ‘flux’, ‘sncosmo_source’
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - ‘flux_density’, ‘magnitude’, ‘spectra’, ‘flux’, ‘sncosmo_source’
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)
                                                              
def sn_nickel_fallback(time, redshift, mej, f_nickel, logl1, tr, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: source redshift
    :param mej: total ejecta mass in solar masses
    :param f_nickel: fraction of nickel mass
    :param logl1: bolometric luminosity scale in log10 (cgs)
    :param tr: transition time for luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
        e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, mej (solar masses), vej (km/s), floor temperature
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is ‘flux_density’.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is ‘magnitude’ or ‘flux’.
    :param output_format: ‘flux_density’, ‘magnitude’, ‘spectra’, ‘flux’, ‘sncosmo_source’
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - ‘flux_density’, ‘magnitude’, ‘spectra’, ‘flux’, ‘sncosmo_source’
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
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
def sn_exponential_powerlaw(time, redshift, lbol_0, alpha_1, alpha_2, tpeak_d, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: source redshift
    :param lbol_0: bolometric luminosity scale in cgs
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak_d: peak time in days
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
        e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, mej (solar masses), vej (km/s), floor temperature
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: None
    :return: bolometric_luminosity
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
    Compute the mass and nickel distributions following a broken power-law
    density profile inspired by Matzner & McKee (1999)

    :param vmin: minimum velocity in km/s
    :param esn: supernova explosion energy in foe
    :param mej: total ejecta mass in solar masses
    :param f_nickel: fraction of nickel mass
    :param f_mixing: fraction of nickel mass that is mixed
    :param mass_len: number of mass shells
    :param vmax: maximum velocity in km/s
    :param delta: inner density profile exponent (actual mass dist is 2 - delta)
    :param n: outer density profile exponent (actual mass dist is 2 - n)
    :return: vel in km/s, v_m in cm/s, m_array in solar masses, ni_array in solar masses (total nickel mass is f_nickel*mej)
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
    :param time: time array to evaluate model on in source frame in seconds
    :param mej: ejecta mass in solar masses
    :param esn: explosion energy in foe
    :param beta: velocity power law slope (M ∝ v^-beta)
    :param kappa: gray opacity at high temperatures (κ_max) [cm²/g], if use_gray_opacity is True, this is your kappa.
    :param kappa_gamma: gamma-ray opacity (assumed constant)
    :param f_nickel: fraction of total ejecta mass that is nickel
    :param f_mixing: fraction of nickel mass that is mixed, a low value puts all the nickel in the first shell.
    :param temperature_floor: temperature floor in K, also used as the transition T_crit.
    :param kwargs: Additional keyword arguments:
    :param use_broken_powerlaw (bool): whether to use a broken power-law for the mass and nickel distribution, True by default.
    :param use_gray_opacity (bool): whether to use gray opacity, defaults to True.
    :param delta (float): inner density profile exponent, used if use_broken_powerlaw is True.
    :param nn (float): outer density profile exponent, used if use_broken_powerlaw is True.
    :param beta (float): velocity power law slope, defaults to 3.0. Only used if use_broken_powerlaw is False.
    :param mass_len (int): number of mass shells, defaults to 200.
    :param vmax (float): maximum velocity in km/s, defaults to 100000.
    :param vmin_frac (float): fraction of characteristic velocity that is the minimum velocity, defaults to 1.0.
    :param kappa_min (float): minimum opacity when cool (default: 0.05 cm²/g).
    :param kappa_n   (float): exponent controlling the transition (default: 4.0).
    :return: namedtuple with time_temp (days), lbol, t_photosphere, r_photosphere, tau, and v_photosphere.
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
    A model for the bolometric light curve of a supernova with nickel mixing

    :param time: time in source frame in days
    :param mej: ejecta mass in solar masses
    :param esn: energy of explosion in foe
    :param kappa: gray opacity
    :param kappa_gamma: gamma-ray opacity
    :param f_nickel: fraction of nickel mass
    :param f_mixing: fraction of nickel mass that is mixed, a low value puts all the nickel in the first shell.
    :param kwargs: bolometric luminosity in erg/s
    :param beta: power law slope for mass distribution; m = m_0 * (v/v_min)^(-beta)
    :param stop_time: time to stop ODE at, default is 300 days
    :param mass_len: number of mass shells, defaults to 200
    :param vmax: maximum velocity in km/s, defaults to 100000
    :param dense_resolution: resolution of dense time array, default is 1000
    :return: bolometric luminosity
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

def nickelmixing(time, redshift, mej, esn, kappa, kappa_gamma, f_nickel, f_mixing,
                 temperature_floor, **kwargs):
    """
    A model for the radioactive decay of a supernova with nickel mixing

    :param time: time in observer frame in days
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param esn: energy of explosion in foe
    :param vmax: maximum velocity of ejecta in km/s
    :param kappa: gray opacity
    :param kappa_gamma: gamma-ray opacity
    :param f_nickel: fraction of nickel mass
    :param f_mixing: fraction of nickel mass that is mixed, a low value puts all the nickel in the first shell.
    :param kwargs: additional keyword arguments
    :param beta: power law slope for mass distribution; m = m_0 * (v/v_min)^(-beta)
    :param mass_len: number of mass shells, defaults to 200
    :param vmax: maximum velocity in km/s, defaults to 100000
    :param stop_time: time to stop ODE at, default is 300 days
    :param dense_resolution: resolution of dense time array, default is 1000
    :param frequency: Required if output_format is 'flux_density'.
    frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = sed.blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: Must be all the kwargs required by the specific interaction_process
    :param interaction_process: Default is Diffusion.
        Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
        e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: bolometric_luminosity
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
    :param time: time in days
    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
         e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

def shock_cooling_and_arnett_bolometric(time, log10_mass, log10_radius, log10_energy,
                             f_nickel, mej, vej, kappa, kappa_gamma, temperature_floor, **kwargs):
    """
    Bolometric luminosity of shock cooling and arnett model

    :param time: time in days in source frame
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param vej: velocity of ejecta in km/s
    :param kappa: opacity in cm^2/g
    :param kappa_gamma: gamma-ray opacity in cm^2/g
    :param temperature_floor: temperature floor in K
    :param kwargs: Additional keyword arguments
    :param nn: density power law slope
    :param delta: inner density power law slope
    :return: bolometric luminosity in erg/s
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
    Photometric light curve of shock cooling and arnett model

    :param time: time in days
    :param redshift: source redshift
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param vej: velocity of ejecta in km/s
    :param kappa: opacity in cm^2/g
    :param kappa_gamma: gamma-ray opacity in cm^2/g
    :param temperature_floor: temperature floor in K
    :param kwargs: Additional keyword arguments
    :param nn: density power law slope
    :param delta: inner density power law slope
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    Assumes Shock cooling following Morag+ and arnett model for radioactive decay

    :param time: time in source frame in days
    :param v_shock: shock speed in km/s, also the ejecta velocity in the arnett calculation
    :param m_env: envelope mass in solar masses
    :param mej: ejecta mass in solar masses
    :param f_rho: f_rho. Typically, of order unity
    :param f_nickel: fraction of nickel mass
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :return: bolometric luminosity in erg/s
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
    Assumes Shock cooling following Morag+ and arnett model for radioactive decay

    :param time: time in observer frame in days
    :param redshift: source redshift
    :param v_shock: shock speed in km/s, also the ejecta velocity in the arnett calculation
    :param m_env: envelope mass in solar masses
    :param mej: ejecta mass in solar masses
    :param f_rho: f_rho. Typically, of order unity
    :param f_nickel: fraction of nickel mass
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
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
        lbol = shockcooling_morag_and_arnett_bolometric(time=time, v_shock=v_shock, m_env=m_env, mej=mej,
                                                        f_rho=f_rho, f_nickel=f_nickel, radius=radius, kappa=kappa, **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_shock, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    Assumes Shock cooling following Sapir and Waxman and arnett model for radioactive decay

    :param time: time in source frame in days
    :param v_shock: shock speed in km/s, also the ejecta velocity in the arnett calculation
    :param m_env: envelope mass in solar masses
    :param mej: ejecta mass in solar masses
    :param f_rho: f_rho. Typically, of order unity
    :param f_nickel: fraction of nickel mass
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :param n: index of progenitor density profile, 1.5 (default) or 3.0
    :param RW: If True, use the simplified Rabinak & Waxman formulation (off by default)
    :return: bolometric luminosity in erg/s
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
    Assumes Shock cooling following Sapir and Waxman and arnett model for radioactive decay

    :param time: time in source frame in days
    :param v_shock: shock speed in km/s, also the ejecta velocity in the arnett calculation
    :param m_env: envelope mass in solar masses
    :param mej: ejecta mass in solar masses
    :param f_rho: f_rho. Typically, of order unity
    :param f_nickel: fraction of nickel mass
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :param n: index of progenitor density profile, 1.5 (default) or 3.0
    :param RW: If True, use the simplified Rabinak & Waxman formulation (off by default)
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
        lbol = shockcooling_morag_and_arnett_bolometric(time=time, v_shock=v_shock, m_env=m_env, mej=mej,
                                                        f_rho=f_rho, f_nickel=f_nickel, radius=radius, kappa=kappa,
                                                        **kwargs)
        photo = photosphere.TemperatureFloor(time=time, luminosity=lbol, vej=v_shock, **kwargs)
        sed_1 = sed.Blackbody(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    :param time: time in days in source frame
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param kwargs: Must be all the kwargs required by the specific interaction_process
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
        Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
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
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
         e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number.
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    Same as basic magnetar_powered but with constraint on rotational_energy/kinetic_energy and nebula phase

    :param time: time in days in source frame
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param kwargs: Must be all the kwargs required by the specific interaction_process
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
    """
    return basic_magnetar_powered_bolometric(time=time, p0=p0, bp=bp, mass_ns=mass_ns,
                                             theta_pb=theta_pb, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract')
def slsn(time, redshift, p0, bp, mass_ns, theta_pb,**kwargs):
    """
    Same as basic magnetar_powered but with constraint on rotational_energy/kinetic_energy and nebula phase

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
             and CutoffBlackbody: cutoff_wavelength, default is 3000 Angstrom
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is CutoffBlackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value

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
    :param time: time in days in observer frame
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param redshift: source redshift
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    Assumes homologous expansion to transform kinetic energy to ejecta velocity

    :param time: time in days in source frame
    :param mej: ejecta mass in solar masses
    :param ek: kinetic energy in ergs
    :param kwargs: Must be all the kwargs required by the specific interaction_process
        e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
        'base model' from homologous_expansion_models list
    :param interaction_process: Default is Diffusion.
        Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
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
    Assumes thin shell ejecta to transform kinetic energy into ejecta velocity

    :param time: time in days in source frame
    :param mej: ejecta mass in solar masses
    :param ek: kinetic energy in ergs
    :param kwargs: Must be all the kwargs required by the specific interaction_process
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor,
             'base model' from homologous_expansion_models list
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
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
    Assumes homologous expansion to transform kinetic energy to ejecta velocity

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param ek: kinetic energy in ergs
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
        e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
        'base model' from homologous_expansion_models list
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    Assumes thin shell ejecta to transform kinetic energy into ejecta velocity

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param ek: kinetic energy in ergs
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
             'base model' from homologous_expansion_models list
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value

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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    :param time: time in days in source frame
    :param mej: ejecta mass in solar masses
    :param csm_mass: csm mass in solar masses
    :param vej: ejecta velocity in km/s
    :param eta: csm density profile exponent
    :param rho: csm density profile amplitude
    :param kappa: opacity
    :param r0: radius of csm shell in AU
    :param kwargs:
            efficiency: in converting between kinetic energy and luminosity, default 0.5
            delta: default 1,
            nn: default 12,
    :return: named tuple with 'lbol','r_photosphere' 'mass_csm_threshold'
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
    :param time: time in days in source frame
    :param mej: ejecta mass in solar masses
    :param csm_mass: csm mass in solar masses
    :param vej: ejecta velocity in km/s
    :param eta: csm density profile exponent
    :param rho: csm density profile amplitude
    :param kappa: opacity
    :param r0: radius of csm shell in AU
    :param kwargs:
            efficiency: in converting between kinetic energy and luminosity, default 0.5
            delta: default 1,
            nn: default 12,
            If interaction process is different kwargs must include other keyword arguments that are required.
    :param interaction_process: Default is CSMDiffusion.
        Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
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
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param csm_mass: csm mass in solar masses
    :param vej: ejecta velocity in km/s
    :param eta: csm density profile exponent
    :param rho: csm density profile amplitude
    :param kappa: opacity
    :param r0: radius of csm shell in AU
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa_gamma, temperature_floor
             'base model' from homologous_expansion_models list
    :param interaction_process: Default is CSMDiffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
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
def csm_nickel(time, redshift, mej, f_nickel, csm_mass, ek, eta, rho, kappa, r0, **kwargs):
    """
    Assumes csm and nickel engine with homologous expansion

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param mej: ejecta mass in solar masses
    :param csm_mass: csm mass in solar masses
    :param ek: kinetic energy in ergs
    :param eta: csm density profile exponent
    :param rho: csm density profile amplitude
    :param kappa: opacity
    :param r0: radius of csm shell in AU
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
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
def type_1a(time, redshift, f_nickel, mej, **kwargs):
    """
    A nickel powered explosion with line absorption and cutoff blackbody SED for SNe 1A.

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: ejecta mass in solar masses
    :param kwargs: kappa, kappa_gamma, vej (km/s),
        temperature_floor (K), cutoff_wavelength (default is 3000 Angstrom)
    :param line_wavelength: line wavelength in angstrom, default is 7.5e3 Angstrom in observer frame
    :param line_width: line width in angstrom, default is 500
    :param line_time: line time, default is 50
    :param line_duration: line duration, default is 25
    :param line_amplitude: line amplitude, default is 0.3
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number.
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be an astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        # The following line converts the full SED (in mJy) to erg/s/cm^2/Angstrom.
        spectra = (full_sed * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(
                                             wav=(lambdas_observer_frame.reshape(-1, 1) * uu.Angstrom))).T
        print(spectra.shape)
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
    A nickel powered explosion with synchrotron and blackbody SED's for SNe 1C.

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: ejecta mass in solar masses
    :param pp: power law index for synchrotron
    :param kwargs: kappa, kappa_gamma, vej (km/s), temperature_floor (K), nu_max (default is 1e9 Hz)
        source_radius (default is 1e13), f0: synchrotron normalisation (default is 1e-26).
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    :param time: time in days in source frame
    :param l0: magnetar energy normalisation in ergs
    :param tsd: magnetar spin down damping timescale in source frame days
    :param nn: braking index
    :param kwargs: Must be all the kwargs required by the specific interaction_process,
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s)
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :return: bolometric_luminosity
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
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param l0: magnetar energy normalisation in ergs
    :param tsd: magnetar spin down damping timescale in source frame in days
    :param nn: braking index
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
        return flux_density.to(uu.mJy).value      
    
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
        return flux_density.to(uu.mJy).value
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
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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

    # Extract components from the output
    rest_spectrum = spectra_output.spectrum  # erg/s/Hz in rest frame
    standard_freqs = spectra_output.frequency.value  # Angstrom in rest frame
    standard_times = spectra_output.time.value  # days in rest frame

    # Apply cosmological dimming
    observed_spectrum = rest_spectrum / (4 * np.pi * dl ** 2)

    # Handle different output formats
    if kwargs.get('output_format') == 'flux_density':
        # Use redback's K-correction utilities
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, time=time, redshift=redshift)

        # Convert wavelengths to frequencies for interpolation
        nu_array = lambda_to_nu(standard_freqs)

        # Convert spectrum from erg/s/Hz to erg/s/cm²/Hz (already done above)
        # Convert to wavelength density for astropy conversion
        spectra_lambda = spectra_lambda.to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom)

        # Convert to mJy using astropy
        fmjy = spectra_lambda.to(uu.mJy,
                                 equivalencies=uu.spectral_density(wav=standard_freqs * uu.Angstrom)).value

        # Create interpolator
        flux_interpolator = RegularGridInterpolator(
            (standard_times, nu_array),
            fmjy,
            bounds_error=False,
            fill_value=0.0
        )

        # Prepare points for interpolation
        if isinstance(frequency, (int, float)):
            frequency = np.ones_like(time) * frequency

        # Create points for evaluation
        points = np.column_stack((time, frequency))

        # Return interpolated flux
        return flux_interpolator(points)

    else:
        # Create denser grid for output (in rest frame)
        new_rest_times = np.geomspace(np.min(standard_times), np.max(standard_times), 200)
        new_rest_freqs = np.geomspace(np.min(standard_freqs), np.max(standard_freqs), 200)

        # Create interpolator for the spectrum in rest frame
        spectra_func = RegularGridInterpolator(
            (standard_times, standard_freqs),
            observed_spectrum.value,
            bounds_error=False,
            fill_value=0.0
        )

        # Create meshgrid for new grid points
        tt_mesh, ff_mesh = np.meshgrid(new_rest_times, new_rest_freqs, indexing='ij')
        points_to_evaluate = np.column_stack((tt_mesh.ravel(), ff_mesh.ravel()))

        # Interpolate spectrum onto new grid
        interpolated_values = spectra_func(points_to_evaluate)
        interpolated_spectrum = interpolated_values.reshape(tt_mesh.shape) * observed_spectrum.unit

        # Convert times to observer frame
        time_observer_frame = new_rest_times * (1 + redshift)

        # Convert wavelengths to observer frame
        lambda_observer_frame = new_rest_freqs * (1 + redshift)

        # Convert spectrum units using astropy
        interpolated_spectrum = interpolated_spectrum.to(
            uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
            equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom)
        )

        # Create output structure
        if kwargs.get('output_format') == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(
                time=time_observer_frame,
                lambdas=lambda_observer_frame,
                spectra=interpolated_spectrum
            )
        else:
            # Get correct output format using redback utility
            return sed.get_correct_output_format_from_spectra(
                time=time,  # Original observer frame time for evaluation
                time_eval=time_observer_frame,
                spectra=interpolated_spectrum,
                lambda_array=lambda_observer_frame,
                time_spline_degree=1,
                **kwargs
            )