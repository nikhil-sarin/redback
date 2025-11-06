from redback.constants import *
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar, _evolving_gw_and_em_magnetar
import numpy as np
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.interpolate import interp1d
from collections import namedtuple
import astropy.units as uu # noqa
import astropy.constants as cc # noqa
from redback.utils import calc_kcorrected_properties, interpolated_barnes_and_kasen_thermalisation_efficiency, \
    electron_fraction_from_kappa, citation_wrapper, lambda_to_nu, velocity_from_lorentz_factor
from redback.sed import blackbody_to_flux_density, get_correct_output_format_from_spectra, \
    flux_density_to_spectrum, blackbody_to_spectrum

def _ejecta_dynamics_and_interaction(time, mej, beta, ejecta_radius, kappa, n_ism,
                                     magnetar_luminosity, pair_cascade_switch, use_gamma_ray_opacity, **kwargs):
    """
    Calculate ejecta dynamics and interaction with magnetar energy injection.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in seconds.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    magnetar_luminosity : np.ndarray
        Evaluated magnetar luminosity in source frame in erg/s.
    pair_cascade_switch : bool
        Whether to account for pair cascade losses.
    use_gamma_ray_opacity : bool
        Whether to use gamma ray opacity to calculate thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - use_r_process : bool
            Whether to use r-process heating (default: True).
        - kappa_gamma : float
            Gamma-ray opacity for leakage efficiency in cm^2/g, only used if use_gamma_ray_opacity = True.
        - thermalisation_efficiency : float
            Magnetar thermalisation efficiency, only used if use_gamma_ray_opacity = False.
        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - f_nickel : float
            Nickel fraction if not using r_process.

    Returns
    -------
    namedtuple
        Named tuple with fields: 'lorentz_factor', 'bolometric_luminosity', 'comoving_temperature',
        'radius', 'doppler_factor', 'tau', 'time', 'kinetic_energy', 'erad_total',
        'thermalisation_efficiency', 'r_photosphere'.
    """
    mag_lum = magnetar_luminosity
    use_r_process = kwargs.get('use_r_process',True)
    ejecta_albedo = kwargs.get('ejecta_albedo', 0.5)
    pair_cascade_fraction = kwargs.get('pair_cascade_fraction', 0.05)

    mej = mej * solar_mass
    lorentz_factor = []
    radius = []
    doppler_factor = []
    lbol_ejecta = []
    lbol_rest = []
    comoving_temperature = []
    tau = []
    teff = []

    internal_energy = 0.5 * beta ** 2 * mej * speed_of_light ** 2
    comoving_volume = (4 / 3) * np.pi * ejecta_radius ** 3
    gamma = 1 / np.sqrt(1 - beta ** 2)


    t0_comoving = 1.3
    tsigma_comoving = 0.11
    
    ni56_lum = 6.45e43
    co56_lum = 1.45e43
    ni56_life = 8.8*86400  # days
    co56_life = 111.3*86400  # days    

    for i in range(len(time)):
        beta = np.sqrt(1 - 1 / gamma ** 2)
        doppler_factor_temp = 1 / (gamma * (1 - beta))
        if i > 0:
            dt = time[i] - time[i - 1]
            gamma = gamma + dgamma_dt * dt
            ejecta_radius = ejecta_radius + drdt * dt
            comoving_volume = comoving_volume + dcomoving_volume_dt * dt
            internal_energy = internal_energy + dinternal_energy_dt * dt

        swept_mass = (4 / 3) * np.pi * ejecta_radius ** 3 * n_ism * proton_mass
        comoving_pressure = internal_energy / (3 * comoving_volume)
        comoving_time = doppler_factor_temp * time[i]
        comoving_dvdt = 4 * np.pi * ejecta_radius ** 2 * beta * speed_of_light
        rad_denom = (1 / 2) - (1 / 3.141592654) * np.arctan((comoving_time - t0_comoving) / tsigma_comoving)
        if use_r_process:
            comoving_radiative_luminosity = (4 * 10 ** 49 * (mej / (2 * 10 ** 33) * 10 ** 2) * rad_denom ** 1.3)
        else:
            f_nickel = kwargs.get('f_nickel',0)
            nickel_mass = f_nickel * mej / solar_mass
            comoving_radiative_luminosity = nickel_mass * (ni56_lum*np.exp(-comoving_time/ni56_life) + co56_lum * np.exp(-comoving_time/co56_life))
        tau_temp = kappa * (mej / comoving_volume) * (ejecta_radius / gamma)

        if tau_temp <= 1:
            comoving_emitted_luminosity = (internal_energy * speed_of_light) / (ejecta_radius / gamma)
            comoving_temp_temperature = (internal_energy / (radiation_constant * comoving_volume)) ** (1./4.)
        else:
            comoving_emitted_luminosity = (internal_energy * speed_of_light) / (tau_temp * ejecta_radius / gamma)
            comoving_temp_temperature = (internal_energy / (radiation_constant * comoving_volume * tau_temp)) ** (1./4.)

        emitted_luminosity = comoving_emitted_luminosity * doppler_factor_temp ** 2

        vej = velocity_from_lorentz_factor(gamma)

        if use_gamma_ray_opacity:
            kappa_gamma = kwargs["kappa_gamma"]
            prefactor = 3 * kappa_gamma * mej / (4 * np.pi * vej**2)
            thermalisation_efficiency = 1 - np.exp(-prefactor * time[i] ** -2)
        else:
            thermalisation_efficiency = kwargs["thermalisation_efficiency"]

        drdt = (beta * speed_of_light) / (1 - beta)
        dswept_mass_dt = 4 * np.pi * ejecta_radius ** 2 * n_ism * proton_mass * drdt
        dedt = thermalisation_efficiency * mag_lum[
            i] + doppler_factor_temp ** 2 * comoving_radiative_luminosity - doppler_factor_temp ** 2 * comoving_emitted_luminosity
        comoving_dinternal_energydt = thermalisation_efficiency * doppler_factor_temp ** (-2) * mag_lum[
            i] + comoving_radiative_luminosity - comoving_emitted_luminosity - comoving_pressure * comoving_dvdt
        dcomoving_volume_dt = comoving_dvdt * doppler_factor_temp
        dinternal_energy_dt = comoving_dinternal_energydt * doppler_factor_temp
        dgamma_dt = (dedt - gamma * doppler_factor_temp * comoving_dinternal_energydt - (
                    gamma ** 2 - 1) * speed_of_light ** 2 * dswept_mass_dt) / (
                            mej * speed_of_light ** 2 + internal_energy + 2 * gamma * swept_mass * speed_of_light ** 2)
        lorentz_factor.append(gamma)
        lbol_ejecta.append(comoving_emitted_luminosity)
        lbol_rest.append(emitted_luminosity)
        comoving_temperature.append(comoving_temp_temperature)
        radius.append(ejecta_radius)
        tau.append(tau_temp)
        doppler_factor.append(doppler_factor_temp)
        teff.append(thermalisation_efficiency)

    lorentz_factor = np.array(lorentz_factor)
    v0 = ((1/lorentz_factor)**2 + 1)**0.5 * speed_of_light
    bolometric_luminosity = np.array(lbol_rest)
    radius = np.array(radius)

    if pair_cascade_switch:
        tlife_t = (0.6/(1 - ejecta_albedo))*(pair_cascade_fraction/0.1)**0.5 * (mag_lum/1.0e45)**0.5 \
                  * (v0/(0.3*speed_of_light))**(0.5) * (time/86400)**(-0.5)
        bolometric_luminosity = bolometric_luminosity / (1.0 + tlife_t)
        comoving_temperature = (bolometric_luminosity / (4.0 * np.pi * np.array(radius) ** (2.0) * sigma_sb)) ** (0.25)

    dynamics_output = namedtuple('dynamics_output', ['lorentz_factor', 'bolometric_luminosity', 'comoving_temperature',
                                                     'radius', 'doppler_factor', 'tau', 'time', 'kinetic_energy',
                                                     'erad_total', 'thermalisation_efficiency', 'r_photosphere'])

    dynamics_output.lorentz_factor = lorentz_factor
    dynamics_output.bolometric_luminosity = bolometric_luminosity
    dynamics_output.comoving_temperature = np.array(comoving_temperature)
    dynamics_output.radius = radius
    dynamics_output.doppler_factor = np.array(doppler_factor)
    dynamics_output.tau = tau
    dynamics_output.time = time
    dynamics_output.kinetic_energy = (lorentz_factor - 1)*mej*speed_of_light**2
    dynamics_output.erad_total = np.trapz(bolometric_luminosity, x=time)
    dynamics_output.thermalisation_efficiency = teff
    return dynamics_output

def _comoving_blackbody_to_flux_density(dl, frequency, radius, temperature, doppler_factor):
    """
    Calculate flux density from a relativistic blackbody in the comoving frame.

    Parameters
    ----------
    dl : float
        Luminosity distance in cm.
    frequency : np.ndarray or float
        Frequency to calculate in Hz. Must be same length as time array or a single number.
    radius : np.ndarray or float
        Ejecta radius in cm.
    temperature : np.ndarray or float
        Comoving temperature in K.
    doppler_factor : np.ndarray or float
        Doppler factor.

    Returns
    -------
    astropy.units.Quantity
        Flux density.
    """
    ## adding units back in to ensure dimensions are correct
    frequency = frequency * uu.Hz
    radius = radius * uu.cm
    dl = dl * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 2 * np.pi * planck * frequency ** 3 * radius ** 2
    denom = dl ** 2 * speed_of_light ** 2 * doppler_factor ** 2
    frac = 1. / (np.exp((planck * frequency) / (boltzmann_constant * temperature * doppler_factor)) - 1)
    flux_density = num / denom * frac
    return flux_density


def _comoving_blackbody_to_luminosity(frequency, radius, temperature, doppler_factor):
    """
    Calculate luminosity from a relativistic blackbody in the comoving frame.

    Parameters
    ----------
    frequency : np.ndarray or float
        Frequency to calculate in Hz. Must be same length as time array or a single number.
    radius : np.ndarray or float
        Ejecta radius in cm.
    temperature : np.ndarray or float
        Comoving temperature in K.
    doppler_factor : np.ndarray or float
        Doppler factor.

    Returns
    -------
    astropy.units.Quantity
        Luminosity.
    """
    ## adding units back in to ensure dimensions are correct
    frequency = frequency * uu.Hz
    radius = radius * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 8 * np.pi ** 2 * planck * frequency ** 4 * radius ** 2
    denom = speed_of_light ** 2 * doppler_factor ** 2
    frac = 1. / (np.exp((planck * frequency) / (boltzmann_constant * temperature * doppler_factor)) - 1)
    luminosity = num / denom * frac
    return luminosity

def _processing_other_formats(dl, output, redshift, time_obs, time_temp, **kwargs):
    """
    Process the dynamics output into various output formats (spectra, magnitude, flux).

    Parameters
    ----------
    dl : float
        Luminosity distance in cm.
    output : namedtuple
        Dynamics output containing temperature, radius, and other properties.
    redshift : float
        Source redshift.
    time_obs : np.ndarray
        Observed time array in days.
    time_temp : np.ndarray
        Temporary time array in seconds where output is evaluated.
    **kwargs : dict
        Additional keyword arguments:

        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on (default: np.geomspace(100, 60000, 100)).
        - use_relativistic_blackbody : bool
            Whether to use relativistic blackbody formula.
        - output_format : str
            Output format: 'spectra', 'magnitude', 'flux', 'sncosmo_source'.

    Returns
    -------
    namedtuple or array
        Returns the correct output format based on output_format parameter.
    """
    lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
    time_observer_frame = time_temp * (1. + redshift)
    frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                 redshift=redshift, time=time_observer_frame)
    if kwargs['use_relativistic_blackbody']:
        fmjy = _comoving_blackbody_to_flux_density(dl=dl, frequency=frequency[:, None], radius=output.radius,
                                                   temperature=output.comoving_temperature,
                                                   doppler_factor=output.doppler_factor)
        fmjy = fmjy
        fmjy = fmjy.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
    else:
        spectra = blackbody_to_spectrum(
            temperature=output.temperature,
            r_photosphere=output.r_photosphere,
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
        return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                      spectra=spectra, lambda_array=lambda_observer_frame,
                                                      **kwargs)

def _process_flux_density(dl, output, redshift, time, time_temp, **kwargs):
    """
    Process the dynamics output into flux density format.

    Parameters
    ----------
    dl : float
        Luminosity distance in cm.
    output : namedtuple
        Dynamics output containing temperature, radius, and other properties.
    redshift : float
        Source redshift.
    time : np.ndarray
        Observed time array in days.
    time_temp : np.ndarray
        Temporary time array in seconds where output is evaluated.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz to calculate flux density at.
        - use_relativistic_blackbody : bool
            Whether to use relativistic blackbody formula.

    Returns
    -------
    np.ndarray
        Flux density in mJy.
    """
    frequency = kwargs['frequency']
    time = time * day_to_s
    # convert to source frame time and frequency
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    if kwargs['use_relativistic_blackbody']:
        temp_func = interp1d(time_temp, y=output.comoving_temperature)
        rad_func = interp1d(time_temp, y=output.radius)
        d_func = interp1d(time_temp, y=output.doppler_factor)
        temp = temp_func(time)
        rad = rad_func(time)
        df = d_func(time)
        flux_density = _comoving_blackbody_to_flux_density(dl=dl, frequency=frequency, radius=rad, temperature=temp,
                                                       doppler_factor=df)
        flux_density = flux_density / (1 + redshift)
    else:
        temp_func = interp1d(time_temp, y=output.temperature)
        rad_func = interp1d(time_temp, y=output.r_photosphere)
        temp = temp_func(time)
        rad = rad_func(time)
        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=rad, frequency=frequency, dl=dl)
    return flux_density.to(uu.mJy).value / (1 + redshift)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...776L..40Y/abstract')
def basic_mergernova(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, p0, bp,
                     mass_ns, theta_pb, thermalisation_efficiency, **kwargs):
    """
    Basic mergernova model with magnetar energy injection and ejecta-ISM interaction.

    This model combines magnetar spin-down energy injection with ejecta dynamics
    and interaction with the surrounding interstellar medium.

    Parameters
    ----------
    time : np.ndarray
        Time in observer frame in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    p0 : float
        Initial spin period in milliseconds.
    bp : float
        Polar magnetic field strength in units of 10^14 Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: False).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', False)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = np.geomspace(1e-4, 1e8, 500, endpoint=True) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = basic_magnetar(time=time_temp, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              thermalisation_efficiency=thermalisation_efficiency,
                                              pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=False, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_mergernova(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
               thermalisation_efficiency, **kwargs):
    """
    General mergernova model with phenomenological magnetar energy injection.

    This model uses a phenomenological magnetar model with general parameters
    for the spin-down luminosity evolution.

    Parameters
    ----------
    time : np.ndarray
        Time in observer frame in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = np.geomspace(1e-4, 1e8, 500, endpoint=True) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              thermalisation_efficiency=thermalisation_efficiency,
                                              pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=False, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_mergernova_thermalisation(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
               kappa_gamma, **kwargs):
    """
    General mergernova model with gamma-ray opacity-based thermalisation efficiency.

    This variant calculates the thermalisation efficiency from the gamma-ray opacity
    rather than using a fixed value.

    Parameters
    ----------
    time : np.ndarray
        Time in observer frame in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    kappa_gamma : float
        Gamma-ray opacity in cm^2/g used to calculate magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = np.geomspace(1e-4, 1e8, 500, endpoint=True) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_mergernova_evolution(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, logbint,
                                 logbext, p0, chi0, radius, logmoi, kappa_gamma, **kwargs):
    """
    General mergernova model with evolving magnetar including GW and EM emission.

    This model includes both gravitational wave and electromagnetic energy loss
    from the magnetar with evolving spin and magnetic field orientation.

    Parameters
    ----------
    time : np.ndarray
        Time in observer frame in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    logbint : float
        Log10 internal magnetic field in G.
    logbext : float
        Log10 external magnetic field in G.
    p0 : float
        Spin period in seconds.
    chi0 : float
        Initial inclination angle in radians.
    radius : float
        Radius of neutron star in km.
    logmoi : float
        Log10 moment of inertia of neutron star in g cm^2.
    kappa_gamma : float
        Gamma-ray opacity in cm^2/g used to calculate magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = np.geomspace(1e-4, 1e8, 500, endpoint=True) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    bint = 10 ** logbint
    bext = 10 ** logbext
    radius = radius * km_cgs
    moi = 10 ** logmoi
    output = _evolving_gw_and_em_magnetar(time=time_temp, bint=bint, bext=bext, p0=p0, chi0=chi0, radius=radius, moi=moi)
    magnetar_luminosity = output.Edot_d
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

def _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                          **kwargs):
    """
    Calculate luminosity for trapped magnetar emission through ejecta.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in seconds.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : float
            Frequency in Hz to evaluate the mergernova emission (use a typical X-ray frequency).

    Returns
    -------
    np.ndarray
        Luminosity in erg/s.
    """
    time_temp = np.geomspace(1e-4, 1e8, 500, endpoint=True) #in source frame
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              thermalisation_efficiency=thermalisation_efficiency,
                                              pair_cascade_switch=False, use_gamma_ray_opacity=False)
    temp_func = interp1d(time_temp, y=output.comoving_temperature)
    rad_func = interp1d(time_temp, y=output.radius)
    d_func = interp1d(time_temp, y=output.doppler_factor)
    tau_func = interp1d(time_temp, y=output.tau)
    temp = temp_func(time)
    rad = rad_func(time)
    df = d_func(time)
    optical_depth = tau_func(time)
    frequency = kwargs['frequency']
    trapped_ejecta_lum = _comoving_blackbody_to_luminosity(frequency=frequency, radius=rad,
                                                          temperature=temp, doppler_factor=df)
    lsd = magnetar_only(time, l0=l0, tau=tau_sd, nn=nn)
    lum = np.exp(-optical_depth) * lsd + trapped_ejecta_lum
    return lum


def _trapped_magnetar_flux(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                           thermalisation_efficiency, photon_index, **kwargs):
    """
    Calculate flux for trapped magnetar emission through ejecta.

    Parameters
    ----------
    time : np.ndarray
        Time in observer frame in seconds.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    photon_index : float
        Photon index used to calculate k correction and convert from luminosity to flux.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : float
            Frequency in Hz to evaluate the mergernova emission (use a typical X-ray frequency).
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    np.ndarray
        Integrated flux in erg/s/cm^2.
    """
    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    kwargs['frequency'] = frequency
    lum = _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                                **kwargs)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    kcorr = (1. + redshift) ** (photon_index - 2)
    flux = lum / (4 * np.pi * dl ** 2 * kcorr)
    return flux

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...835....7S/abstract')
def trapped_magnetar(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                     **kwargs):
    """
    Trapped magnetar model for X-ray emission escaping through ejecta.

    This model accounts for magnetar X-ray emission that is partially trapped
    by the ejecta and escapes based on the optical depth.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame or observer frame (depending on output format) in seconds.
    redshift : float
        Source redshift (not used if evaluating luminosity).
    mej : float
        Ejecta mass in solar masses.
    beta : float
        Initial ejecta velocity in units of c.
    ejecta_radius : float
        Initial ejecta radius in cm.
    kappa : float
        Opacity in cm^2/g.
    n_ism : float
        ISM number density in cm^-3.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - output_format : str
            Whether to output 'luminosity' or 'flux'.
        - frequency : float
            Frequency in Hz to evaluate the mergernova emission (use a typical X-ray frequency).
        - photon_index : float
            Only used if calculating the flux lightcurve.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    np.ndarray
        Luminosity in erg/s or integrated flux in erg/s/cm^2, depending on output_format.
    """
    if kwargs['output_format'] == 'luminosity':
        return _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                     thermalisation_efficiency, **kwargs)
    elif kwargs['output_format'] == 'flux':
        return _trapped_magnetar_flux(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                      thermalisation_efficiency, **kwargs)

def _general_metzger_magnetar_driven_kilonova_model(time, mej, vej, beta, kappa, magnetar_luminosity,
                                                    use_gamma_ray_opacity, **kwargs):
    """
    General Metzger magnetar-driven kilonova model with stratified ejecta.

    This model treats the ejecta as multiple mass shells with different velocities
    following a power-law distribution. Each shell is heated by radioactive decay
    and optionally by magnetar energy injection.

    Parameters
    ----------
    time : np.ndarray
        Time array to evaluate model on in source frame in seconds.
    mej : float
        Ejecta mass in solar masses.
    vej : float
        Minimum initial velocity in units of c.
    beta : float
        Velocity power law slope (M proportional to v^-beta).
    kappa : float
        Opacity in cm^2/g.
    magnetar_luminosity : np.ndarray
        Evaluated magnetar luminosity in source frame in erg/s.
    use_gamma_ray_opacity : bool
        Whether to use gamma ray opacity to calculate thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.01).
        - kappa_gamma : float
            Gamma-ray opacity for leakage efficiency in cm^2/g, only used if use_gamma_ray_opacity = True.
        - thermalisation_efficiency : float
            Magnetar thermalisation efficiency, only used if use_gamma_ray_opacity = False.
        - neutron_precursor_switch : bool
            Whether to have neutron precursor emission (default: True).
        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - magnetar_heating : str
            Whether magnetar heats 'all_layers' or just 'first_layer' (default: 'first_layer').
        - vmax : float
            Maximum initial velocity of mass layers in units of c (default: 0.7).

    Returns
    -------
    namedtuple
        Named tuple with fields: 'lorentz_factor', 'bolometric_luminosity', 'temperature',
        'r_photosphere', 'kinetic_energy', 'erad_total', 'thermalisation_efficiency'.
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    ejecta_albedo = kwargs.get('ejecta_albedo', 0.5)
    pair_cascade_fraction = kwargs.get('pair_cascade_fraction', 0.01)
    neutron_precursor_switch = kwargs.get('neutron_precursor_switch', True)
    magnetar_heating = kwargs.get('magnetar_heating', 'first_layer')
    vmax = kwargs.get('vmax', 0.7)

    tdays = time/day_to_s
    time_len = len(time)
    mass_len = 200

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    electron_fraction = electron_fraction_from_kappa(kappa)
    t0 = 1.3 #seconds
    sig = 0.11  #seconds
    tau_neutron = 900  #seconds

    # convert to astrophysical units
    m0 = mej * solar_mass
    v0 = vej * speed_of_light
    ek_tot_0 = 0.5 * m0 * v0 ** 2

    # set up mass and velocity layers and normalize them
    vmin = vej
    vel = np.linspace(vmin, vmax, mass_len)
    m_array = mej * (vel/vmin)**(-beta)
    total_mass = np.sum(m_array)
    normalised_mass = m_array * (mej/ total_mass)
    m_array = normalised_mass
    v_m = vel * speed_of_light

    # set up arrays
    time_array = np.tile(time, (mass_len, 1))
    e_th_array = np.tile(e_th, (mass_len, 1))
    edotr = np.zeros((mass_len, time_len))

    time_mask = time > t0
    time_1 = time_array[:, time_mask]
    time_2 = time_array[:, ~time_mask]
    edotr[:,time_mask] = 2.1e10 * e_th_array[:, time_mask] * ((time_1/ (3600. * 24.)) ** (-1.3))
    edotr[:, ~time_mask] = 4.0e18 * (0.5 - (1. / np.pi) * np.arctan((time_2 - t0) / sig)) ** (1.3) * e_th_array[:,~time_mask]

    lsd = magnetar_luminosity

    # set up empty arrays
    energy_v = np.zeros((mass_len, time_len))
    lum_rad = np.zeros((mass_len, time_len))
    qdot_rp = np.zeros((mass_len, time_len))
    td_v = np.zeros((mass_len, time_len))
    tau = np.zeros((mass_len, time_len))
    v_photosphere = np.zeros(time_len)
    v0_array = np.zeros(time_len)
    qdot_magnetar = np.zeros(time_len)
    r_photosphere = np.zeros(time_len)

    if neutron_precursor_switch == True:
        neutron_mass = 1e-8 * solar_mass
        neutron_mass_fraction = 1 - 2*electron_fraction * 2 * np.arctan(neutron_mass / m_array / solar_mass) / np.pi
        rprocess_mass_fraction = 1.0 - neutron_mass_fraction
        initial_neutron_mass_fraction_array = np.tile(neutron_mass_fraction, (time_len, 1)).T
        rprocess_mass_fraction_array = np.tile(rprocess_mass_fraction, (time_len, 1)).T
        neutron_mass_fraction_array = initial_neutron_mass_fraction_array*np.exp(-time_array / tau_neutron)
        edotn = 3.2e14 * neutron_mass_fraction_array
        edotn = edotn * neutron_mass_fraction_array
        edotr = edotn + edotr
        kappa_n = 0.4 * (1.0 - neutron_mass_fraction_array - rprocess_mass_fraction_array)
        kappa = kappa * rprocess_mass_fraction_array
        kappa = kappa_n + kappa

    dt = np.diff(time)
    dm = np.abs(np.diff(m_array))

    #initial conditions
    energy_v[:, 0] = 0.5 * m_array*v_m**2
    lum_rad[:, 0] = 0
    qdot_rp[:, 0] = 0
    kinetic_energy = ek_tot_0

    # solve ODE using euler method for all mass shells v
    for ii in range(time_len - 1):
        # # evolve the velocity due to pdv work of central shell of mass M and thermal energy Ev0
        kinetic_energy = kinetic_energy + (energy_v[0, ii] / time[ii]) * dt[ii]
        # kinetic_energy = kinetic_energy + (np.sum(energy_v[:, ii]) / time[ii]) * dt[ii]
        v0 = (2 * kinetic_energy / m0) ** 0.5
        v0_array[ii] = v0
        v_m = v0 * (m_array / (mej)) ** (-1 / beta)
        v_m[v_m > 3e10] = speed_of_light

        if use_gamma_ray_opacity:
            kappa_gamma = kwargs["kappa_gamma"]
            prefactor = 3 * kappa_gamma * mej / (4 * np.pi * vej**2)
            thermalisation_efficiency = 1 - np.exp(-prefactor * time[ii] ** -2)
        else:
            thermalisation_efficiency = kwargs["thermalisation_efficiency"]
        qdot_magnetar[ii] = thermalisation_efficiency * lsd[ii]

        if magnetar_heating == 'all_layers':
            if neutron_precursor_switch:
                td_v[:-1, ii] = (kappa[:-1, ii] * m_array[:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            else:
                td_v[:-1, ii] = (kappa * m_array[:-1] * solar_mass * 3) / (4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            lum_rad[:-1, ii] = energy_v[:-1, ii] / (td_v[:-1, ii] + time[ii] * (v_m[:-1] / speed_of_light))
            energy_v[:-1, ii + 1] = (qdot_magnetar[ii] + edotr[:-1, ii] * dm * solar_mass - (energy_v[:-1, ii] / time[ii]) - lum_rad[:-1, ii]) * dt[ii] + energy_v[:-1, ii]

        # first mass layer
        # only bottom layer i.e., 0'th mass layer gets magnetar contribution
        if magnetar_heating == 'first_layer':
            if neutron_precursor_switch:
                td_v[0, ii] = (kappa[0, ii] * m_array[0] * solar_mass * 3) / (
                            4 * np.pi * v_m[0] * speed_of_light * time[ii] * beta)
                td_v[1:-1, ii] = (kappa[1:-1, ii] * m_array[1:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[1:-1] * speed_of_light * time[ii] * beta)
            else:
                td_v[0, ii] = (kappa * m_array[0] * solar_mass * 3) / (4 * np.pi * v_m[0] * speed_of_light * time[ii] * beta)
                td_v[1:-1, ii] = (kappa * m_array[1:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[1:-1] * speed_of_light * time[ii] * beta)

            lum_rad[0, ii] = energy_v[0, ii] / (td_v[0, ii] + time[ii] * (v_m[0] / speed_of_light))
            energy_v[0, ii + 1] = (qdot_magnetar[ii] + edotr[0, ii] * dm[0] * solar_mass - (energy_v[0, ii] / time[ii]) - lum_rad[0, ii]) * dt[ii] + energy_v[0, ii]
            # other layers
            lum_rad[1:-1, ii] = energy_v[1:-1, ii] / (td_v[1:-1, ii] + time[ii] * (v_m[1:-1] / speed_of_light))
            energy_v[1:-1, ii + 1] = (edotr[1:-1, ii] * dm[1:] * solar_mass - (energy_v[1:-1, ii] / time[ii]) - lum_rad[1:-1, ii]) * dt[ii] + energy_v[1:-1, ii]

        if neutron_precursor_switch:
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa[:-1, ii] / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))
        else:
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))

        tau[mass_len - 1, ii] = tau[mass_len - 2, ii]
        photosphere_index = np.argmin(np.abs(tau[:, ii] - 1))
        v_photosphere[ii] = v_m[photosphere_index]
        r_photosphere[ii] = v_photosphere[ii] * time[ii]

    bolometric_luminosity = np.sum(lum_rad, axis=0)

    if pair_cascade_switch == True:
        tlife_t = (0.6/(1 - ejecta_albedo))*(pair_cascade_fraction/0.1)**0.5 * (lsd/1.0e45)**0.5 \
                  * (v0/(0.3*speed_of_light))**(0.5) * (time/day_to_s)**(-0.5)
        bolometric_luminosity = bolometric_luminosity / (1.0 + tlife_t)

    temperature = (bolometric_luminosity / (4.0 * np.pi * (r_photosphere) ** (2.0) * sigma_sb)) ** (0.25)

    dynamics_output = namedtuple('dynamics_output', ['lorentz_factor', 'bolometric_luminosity', 'temperature',
                                                     'r_photosphere', 'kinetic_energy','erad_total',
                                                     'thermalisation_efficiency'])
    gamma_beta = v0_array/speed_of_light
    lorentz_factor = 1/(np.sqrt(1 - gamma_beta**2))
    dynamics_output.lorentz_factor = lorentz_factor
    dynamics_output.bolometric_luminosity = bolometric_luminosity
    dynamics_output.temperature = temperature
    dynamics_output.r_photosphere = r_photosphere
    dynamics_output.kinetic_energy = (lorentz_factor - 1)*m0*speed_of_light**2
    dynamics_output.erad_total = np.trapz(bolometric_luminosity, x=time)
    dynamics_output.thermalisation_efficiency = qdot_magnetar/lsd
    return dynamics_output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017LRR....20....3M/abstract')
def metzger_magnetar_driven_kilonova_model(time, redshift, mej, vej, beta, kappa_r, p0, bp,
                                           mass_ns, theta_pb, thermalisation_efficiency, **kwargs):
    """
    Metzger magnetar-driven kilonova model with stratified ejecta and basic magnetar.

    This is a multi-layer kilonova model with magnetar energy injection using the
    basic magnetar spin-down formulation.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    vej : float
        Minimum initial velocity in units of c.
    beta : float
        Velocity power law slope (M proportional to v^-beta).
    kappa_r : float
        Opacity in cm^2/g.
    p0 : float
        Initial spin period in milliseconds.
    bp : float
        Polar magnetic field strength in units of 10^14 Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - neutron_precursor_switch : bool
            Whether to have neutron precursor emission (default: True).
        - magnetar_heating : str
            Whether magnetar heats 'all_layers' or 'first_layer' (default: 'first_layer').
        - vmax : float
            Maximum initial velocity of mass layers in units of c (default: 0.7).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    use_gamma_ray_opacity = False
    kwargs['use_relativistic_blackbody'] = False

    time_temp = np.geomspace(1e-4, 1e7, 300) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = basic_magnetar(time=time_temp, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             thermalisation_efficiency=thermalisation_efficiency,
                                                             **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_metzger_magnetar_driven(time, redshift, mej, vej, beta, kappa_r, l0,
                                    tau_sd, nn, thermalisation_efficiency, **kwargs):
    """
    General Metzger magnetar-driven kilonova with phenomenological magnetar.

    This multi-layer kilonova model uses the general phenomenological magnetar
    formulation for energy injection.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    vej : float
        Minimum initial velocity in units of c.
    beta : float
        Velocity power law slope (M proportional to v^-beta).
    kappa_r : float
        Opacity in cm^2/g.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    thermalisation_efficiency : float
        Magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - neutron_precursor_switch : bool
            Whether to have neutron precursor emission (default: True).
        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - magnetar_heating : str
            Whether magnetar heats 'all_layers' or 'first_layer' (default: 'first_layer').
        - vmax : float
            Maximum initial velocity of mass layers in units of c (default: 0.7).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    use_gamma_ray_opacity = False
    kwargs['use_relativistic_blackbody'] = False

    time_temp = np.geomspace(1e-4, 1e7, 300) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             thermalisation_efficiency=thermalisation_efficiency,
                                                             **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_metzger_magnetar_driven_thermalisation(time, redshift, mej, vej, beta, kappa_r, l0,
                                    tau_sd, nn, kappa_gamma, **kwargs):
    """
    General Metzger magnetar-driven kilonova with gamma-ray opacity thermalisation.

    This variant calculates the thermalisation efficiency from the gamma-ray opacity
    in the multi-layer ejecta model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    vej : float
        Minimum initial velocity in units of c.
    beta : float
        Velocity power law slope (M proportional to v^-beta).
    kappa_r : float
        Opacity in cm^2/g.
    l0 : float
        Initial magnetar X-ray luminosity in erg/s.
    tau_sd : float
        Magnetar spin down damping timescale in seconds.
    nn : float
        Braking index.
    kappa_gamma : float
        Gamma-ray opacity in cm^2/g used to calculate magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - neutron_precursor_switch : bool
            Whether to have neutron precursor emission (default: True).
        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - magnetar_heating : str
            Whether magnetar heats 'all_layers' or 'first_layer' (default: 'first_layer').
        - vmax : float
            Maximum initial velocity of mass layers in units of c (default: 0.7).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    kwargs['use_relativistic_blackbody'] = False
    use_gamma_ray_opacity = True

    time_temp = np.geomspace(1e-4, 1e7, 300) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             kappa_gamma=kappa_gamma, **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_metzger_magnetar_driven_evolution(time, redshift, mej, vej, beta, kappa_r, logbint,
                                 logbext, p0, chi0, radius, logmoi, kappa_gamma, **kwargs):
    """
    General Metzger magnetar-driven kilonova with evolving magnetar and GW losses.

    This model includes the full evolution of the magnetar including both
    electromagnetic and gravitational wave energy losses in a multi-layer ejecta model.

    Parameters
    ----------
    time : np.ndarray
        Observer frame time in days.
    redshift : float
        Source redshift.
    mej : float
        Ejecta mass in solar masses.
    vej : float
        Minimum initial velocity in units of c.
    beta : float
        Velocity power law slope (M proportional to v^-beta).
    kappa_r : float
        Opacity in cm^2/g.
    logbint : float
        Log10 internal magnetic field in G.
    logbext : float
        Log10 external magnetic field in G.
    p0 : float
        Spin period in seconds.
    chi0 : float
        Initial inclination angle in radians.
    radius : float
        Radius of neutron star in km.
    logmoi : float
        Log10 moment of inertia of neutron star in g cm^2.
    kappa_gamma : float
        Gamma-ray opacity in cm^2/g used to calculate magnetar thermalisation efficiency.
    **kwargs : dict
        Additional keyword arguments:

        - ejecta_albedo : float
            Ejecta albedo (default: 0.5).
        - pair_cascade_fraction : float
            Fraction of magnetar luminosity lost to pair cascades (default: 0.05).
        - neutron_precursor_switch : bool
            Whether to have neutron precursor emission (default: True).
        - pair_cascade_switch : bool
            Whether to account for pair cascade losses (default: True).
        - magnetar_heating : str
            Whether magnetar heats 'all_layers' or 'first_layer' (default: 'first_layer').
        - vmax : float
            Maximum initial velocity of mass layers in units of c (default: 0.7).
        - frequency : np.ndarray or float
            Required if output_format is 'flux_density'. Frequency in Hz.
        - bands : str or list
            Required if output_format is 'magnitude' or 'flux'.
        - output_format : str
            Output format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'.
        - lambda_array : np.ndarray
            Optional wavelength array in Angstroms to evaluate the SED on.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation (default: Planck18).

    Returns
    -------
    array_like or namedtuple
        Set by output_format: flux density (mJy), magnitude, spectra, flux, or sncosmo_source.
    """
    use_gamma_ray_opacity = True
    kwargs['use_relativistic_blackbody'] = False

    time_temp = np.geomspace(1e-4, 1e7, 500) #in source frame
    bint = 10 ** logbint
    bext = 10 ** logbext
    radius = radius * km_cgs
    moi = 10 ** logmoi
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    output = _evolving_gw_and_em_magnetar(time=time_temp, bint=bint, bext=bext, p0=p0, chi0=chi0, radius=radius, moi=moi)
    magnetar_luminosity = output.Edot_d
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             kappa_gamma=kappa_gamma, **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)
