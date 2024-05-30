import numpy as np
import pandas as pd
from redback.transient_models.phenomenological_models import exponential_powerlaw
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar
from redback.transient_models.magnetar_driven_ejecta_models import _ejecta_dynamics_and_interaction
from redback.transient_models.shock_powered_models import _shock_cooling
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, logger, get_csm_properties, nu_to_lambda, lambda_to_nu, velocity_from_lorentz_factor
from redback.constants import day_to_s, solar_mass, km_cgs, au_cgs, speed_of_light
from inspect import isfunction
import astropy.units as uu
from collections import namedtuple
from scipy.interpolate import interp1d

homologous_expansion_models = ['exponential_powerlaw_bolometric', 'arnett_bolometric',
                               'basic_magnetar_powered_bolometric','slsn_bolometric',
                               'general_magnetar_slsn_bolometric','csm_interaction_bolometric',
                               'type_1c_bolometric','type_1a_bolometric']

@citation_wrapper('https://zenodo.org/record/6363879#.YkQn3y8RoeY')
def sncosmo_models(time, redshift, model_kwargs, **kwargs):
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
    :param peak_abs_mag: SNe peak absolute magnitude default set to -19
    :param peak_abs_mag_band: Band corresponding to the peak abs mag limit, default to standard::b. Must be in SNCosmo
    :param mw_extinction: Boolean for whether there is MW extinction or not. Default True
    :param magnitude_system: Mag system; default ab
    :param host_extinction: Boolean for whether there is host extinction or not. Default True
            if used adds an extra parameter ebv which must also be in kwargs; host galaxy E(B-V). Set to 0.1 by default
    :return: set by output format - 'flux_density', 'magnitude', 'flux', 'sncosmo_source'
    """
    import sncosmo
    cosmology = kwargs.get('cosmology', cosmo)
    peak_time = kwargs.get('peak_time', 0)
    peak_abs_mag = kwargs.get('peak_abs_mag', -19)
    peak_abs_mag_band = kwargs.get('peak_abs_mag_band', 'standard::b')
    model_name = kwargs.get('sncosmo_model', 'salt2')
    host_extinction = kwargs.get('host_extinction', True)
    mw_extinction = kwargs.get('mw_extinction', True)
    magsystem = kwargs.get('magnitude_system', 'ab')

    model = sncosmo.Model(source=model_name)
    model.set(z=redshift)
    model.set(t0=peak_time)
    model.update(model_kwargs)

    if host_extinction:
        ebv = kwargs.get('ebv', 0.1)
        model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
        model.set(hostebv=ebv)
    if mw_extinction:
        model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')

    model.set_source_peakabsmag(peak_abs_mag, band=peak_abs_mag_band, magsys=magsystem, cosmo=cosmology)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']

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
        magnitude = model.bandmag(phase=time, band=bands, magsys='ab')
        return sed.bandpass_magnitude_to_flux(magnitude=magnitude, bands=bands)
    elif kwargs['output_format'] == 'magnitude':
        bands = kwargs['bands']
        magnitude = model.bandmag(phase=time, band=bands, magsys='ab')
        return magnitude
    elif kwargs['output_format'] == 'sncosmo_source':
        return model

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

@citation_wrapper('redback')
def shock_cooling_and_arnett(time, redshift, log10_mass, log10_radius, log10_energy,
                             f_nickel, mej, **kwargs):
    """
    :param time: time in days
    :param redshift: source redshift
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
         e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param nn: density power law slope
    :param delta: inner density power law slope
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
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = _shock_cooling(time * day_to_s, mass=mass, radius=radius, energy=energy, **kwargs)
        lbol_1 = output.lbol
        lbol_2 = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
        lbol = lbol_1 + lbol_2

        if kwargs['interaction_process'] is not None:
            dense_resolution = kwargs.get("dense_resolution", 1000)
            dense_times = np.linspace(0, time[-1]+100, dense_resolution)
            dense_lbols = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
            dense_lbols += _shock_cooling(dense_times * day_to_s, mass=mass, radius=radius, energy=energy, **kwargs).lbol
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
        lbol_1 = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
        lbol_2 = _shock_cooling(time * day_to_s, mass=mass, radius=radius, energy=energy, **kwargs).lbol
        lbol = lbol_1 + lbol_2
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
        for ii in range(len(frequency)):
            ss = kwargs['sed'](time=time, temperature=photo.photosphere_temperature,
                               r_photosphere=photo.r_photosphere, frequency=frequency[ii],
                               luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)
            full_sed[:, ii] = ss.flux_density.to(uu.mJy).value            
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
    radius_csm = ((3.0 - eta) / (4.0 * np.pi * qq) * csm_mass + r0 ** (3.0 - eta)) ** (
            1.0 / (3.0 - eta))
    # photosphere radius
    r_photosphere = abs((-2.0 * (1.0 - eta) / (3.0 * kappa * qq) +
                         radius_csm ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))

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

    mask_1 = t_FS - time * day_to_s > 0
    mask_2 = t_RS - time * day_to_s > 0

    lbol = efficiency * (2.0 * np.pi / (nn - eta) ** 3 * g_n ** ((5.0 - eta) / (nn - eta)) * qq **
                         ((nn - 5.0) / (nn - eta)) * (nn - 3.0) ** 2 * (nn - 5.0) * Bf ** (5.0 - eta) * AA **
                         ((5.0 - eta) / (nn - eta)) * (time * day_to_s + ti) **
                         ((2.0 * nn + 6.0 * eta - nn * eta - 15.) /
                          (nn - eta)) + 2.0 * np.pi * (AA * g_n / qq) **
                         ((5.0 - nn) / (nn - eta)) * Br ** (5.0 - nn) * g_n * ((3.0 - eta) / (nn - eta)) ** 3 * (
                                     time * day_to_s + ti) **
                         ((2.0 * nn + 6.0 * eta - nn * eta - 15.0) / (nn - eta)))

    lbol[~mask_1] = 0
    lbol[~mask_2] = 0

    csm_output = namedtuple('csm_output', ['lbol', 'r_photosphere', 'mass_csm_threshold'])
    csm_output.lbol = lbol
    csm_output.r_photosphere = r_photosphere
    csm_output.mass_csm_threshold = mass_csm_threshold
    return csm_output


@citation_wrapper('redback')
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
    r_photosphere = csm_output.r_photosphere
    mass_csm_threshold = csm_output.mass_csm_threshold

    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = _csm_engine(time=dense_times, mej=mej, csm_mass=csm_mass, vej=vej,
                             eta=eta, rho=rho, kappa=kappa, r0=r0, **kwargs).lbol
        interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols,
                                                kappa=kappa, r_photosphere=r_photosphere,
                                                mass_csm_threshold=mass_csm_threshold, csm_mass=csm_mass, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract')
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
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
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
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    line_wavelength = kwargs.get('line_wavelength',7.5e3)
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
                                    r_photosphere=photo.r_photosphere,frequency=frequency, luminosity_distance=dl,
                                    cutoff_wavelength=cutoff_wavelength)
        sed_2 = sed.Line(time=time, luminosity=lbol, frequency=frequency, luminosity_distance=dl,
                         sed=sed_1, line_wavelength=line_wavelength,
                         line_width=line_width, line_time=line_time,
                         line_duration=line_duration, line_amplitude=line_amplitude)
        flux_density = sed_2.flux_density
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
        full_sed = np.zeros((len(time), len(frequency)))
        for ii in range(len(frequency)):
            ss = sed.CutoffBlackbody(time=time, temperature=photo.photosphere_temperature,
                               r_photosphere=photo.r_photosphere, frequency=frequency[ii],
                               luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)
            sed_2 = sed.Line(time=time, luminosity=lbol, frequency=frequency[ii], luminosity_distance=dl,
                             sed=ss, line_wavelength=line_wavelength,
                             line_width=line_width, line_time=line_time,
                             line_duration=line_duration, line_amplitude=line_amplitude)
            full_sed[:,ii] = sed_2.flux_density.to(uu.mJy).value
        spectra = (full_sed * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambdas_observer_frame * uu.Angstrom))
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

@citation_wrapper('Omand and Sarin (2023)')
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
    erot_total = np.trapz(magnetar_luminosity, x=time_temp)
    erad_total = np.trapz(output.bolometric_luminosity, x=time_temp) 
    
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

@citation_wrapper('Omand and Sarin (2023)')
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
            for ii in range(len(frequency)):
                ss = kwargs['sed'](time=time_temp/day_to_s, temperature=photo.photosphere_temperature,
                               r_photosphere=photo.r_photosphere, frequency=frequency[ii],
                               luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=output.bolometric_luminosity)
                full_sed[:, ii] = ss.flux_density.to(uu.mJy).value
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
