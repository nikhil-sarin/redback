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
from redback.sed import blackbody_to_flux_density, get_correct_output_format_from_spectra

def _ejecta_dynamics_and_interaction(time, mej, beta, ejecta_radius, kappa, n_ism,
                                     magnetar_luminosity, pair_cascade_switch, use_gamma_ray_opacity, **kwargs):
    """
    :param time: time in source frame
    :param mej: ejecta mass in solar masses
    :param beta: initial ejecta velocity in c
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param magnetar_luminosity: evaluated magnetar luminosity in source frame
    :param pair_cascade_switch: whether to account for pair cascade losses
    :param use_gamma_ray_opacity: whether to use gamma ray opacity to calculate thermalisation efficiency
    :param kwargs: Additional parameters
    :param use_r_process: whether to use r-process
    :param kappa_gamma: Gamma-ray opacity for leakage efficiency, only used if use_gamma_ray_opacity = True
    :param thermalisation_efficiency: magnetar thermalisation efficiency only used if use_gamma_ray_opacity = False
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param f_nickel: nickel fraction (if not using r_process)
    :return: named tuple with 'lorentz_factor', 'bolometric_luminosity', 'comoving_temperature',
            'radius', 'doppler_factor', 'tau', 'time', 'kinetic_energy',
            'erad_total', 'thermalisation_efficiency'
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
    :param dl: luminosity distance in cm
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number
    :param radius: ejecta radius in cm
    :param temperature: comoving temperature in K
    :param doppler_factor: doppler_factor
    :return: flux_density
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
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number
    :param radius: ejecta radius in cm
    :param temperature: comoving temperature in K
    :param doppler_factor: doppler_factor
    :return: luminosity
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
    Function to process the output of the dynamics function into other formats

    :param dl: luminosity distance in cm
    :param output: dynamics output
    :param redshift: source redshift
    :param time_obs: observed time array in days
    :param time_temp: temporary time array in seconds where output is evaluated
    :param kwargs: extra arguments
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :return: returns the correct output format
    """
    lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
    time_observer_frame = time_temp * (1. + redshift)
    frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                 redshift=redshift, time=time_observer_frame)
    if kwargs['use_relativistic_blackbody']:
        fmjy = _comoving_blackbody_to_flux_density(dl=dl, frequency=frequency[:, None], radius=output.radius,
                                                   temperature=output.comoving_temperature,
                                                   doppler_factor=output.doppler_factor)
    else:
        fmjy = blackbody_to_flux_density(temperature=output.temperature,
                                         r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)

    fmjy = fmjy.T
    spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                 equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
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
    Function to process the output of the dynamics function into flux density

    :param dl: luminosity distance in cm
    :param output: dynamics output
    :param redshift: source redshift
    :param time_obs: observed time array in days
    :param time_temp: temporary time array in seconds where output is evaluated
    :param kwargs: extra arguments
    :return: returns the correct output format
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
    else:
        temp_func = interp1d(time_temp, y=output.temperature)
        rad_func = interp1d(time_temp, y=output.r_photosphere)
        temp = temp_func(time)
        rad = rad_func(time)
        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=rad, frequency=frequency, dl=dl)
    return flux_density.to(uu.mJy).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...776L..40Y/abstract')
def basic_mergernova(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, p0, bp,
                     mass_ns, theta_pb, thermalisation_efficiency, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param p0: initial spin period in milliseconds
    :param bp: polar magnetic field strength in units of 10^14 Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes in radians
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is False
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param logbint: log10 internal magnetic field in G
    :param logbext: log10 external magnetic field in G
    :param p0: spin period in s
    :param chi0: initial inclination angle
    :param radius: radius of NS in KM
    :param logmoi: log10 moment of inertia of NS
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: time in source frame
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :return: luminosity
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
    :param time: time in observer frame in seconds
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' used to calculate k correction and convert from luminosity to flux
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: integrated flux
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
    :param time: time in source frame or observer frame depending on output format in seconds
    :param redshift: redshift - not used if evaluating luminosity
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output luminosity or flux
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' only used if calculating the flux lightcurve
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: luminosity or integrated flux
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
    :param time: time array to evaluate model on in source frame in seconds
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa: opacity
    :param magnetar_luminosity: evaluated magnetar luminosity in source frame
    :param pair_cascade_switch: whether to account for pair cascade losses
    :param use_gamma_ray_opacity: whether to use gamma ray opacity to calculate thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param kappa_gamma: Gamma-ray opacity for leakage efficiency, only used if use_gamma_ray_opacity = True
    :param thermalisation_efficiency: magnetar thermalisation efficiency only used if use_gamma_ray_opacity = False
    :param neutron_precursor_switch: whether to have neutron precursor emission, default True
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer.
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :return: named tuple with 'lorentz_factor', 'bolometric_luminosity', 'temperature',
                'r_photosphere', 'kinetic_energy','erad_total', 'thermalisation_efficiency'
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

    # set up mass and velocity layers
    vmin = vej
    vel = np.linspace(vmin, vmax, mass_len)
    m_array = mej * (vel/vmin)**(-beta)
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
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param p0: initial spin period in milliseconds
    :param bp: polar magnetic field strength in units of 10^14 Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes in radians
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default true
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default true
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param logbint: log10 internal magnetic field in G
    :param logbext: log10 external magnetic field in G
    :param p0: spin period in s
    :param chi0: initial inclination angle
    :param radius: radius of NS in KM
    :param logmoi: log10 moment of inertia of NS
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default true
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
