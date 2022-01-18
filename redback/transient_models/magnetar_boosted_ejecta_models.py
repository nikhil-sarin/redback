from redback.constants import *
from redback.transient_models.magnetar_models import magnetar_only
import numpy as np
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.interpolate import interp1d
import astropy.units as uu # noqa
import astropy.constants as cc # noqa


def metzger_magnetar_boosted_kilonova_model(time, **kwargs):
    pass


def ejecta_dynamics_and_interaction(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                    thermalisation_efficiency, **kwargs):
    """
    :param time: time in source frame
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs:
    :return: lorentz factor, bolometric luminosity, comoving temperature, ejecta radius, doppler factor,
    optical depth (tau)
    """
    mej = mej * solar_mass
    lorentz_factor = []
    radius = []
    doppler_factor = []
    lbol_ejecta = []
    lbol_rest = []
    comoving_temperature = []
    tau = []

    internal_energy = 0.5 * beta ** 2 * mej * speed_of_light ** 2
    comoving_volume = (4 / 3) * np.pi * ejecta_radius ** 3
    gamma = 1 / np.sqrt(1 - beta ** 2)
    mag_lum = magnetar_only(time, l0=l0, tau=tau_sd, nn=nn)

    t0_comoving = 1.3
    tsigma_comoving = 0.11

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
        comoving_radiative_luminosity = (4 * 10 ** 49 * (mej / (2 * 10 ** 33) * 10 ** 2) * rad_denom ** 1.3)
        tau_temp = kappa * (mej / comoving_volume) * (ejecta_radius / gamma)

        if tau_temp <= 1:
            comoving_emitted_luminosity = (internal_energy * speed_of_light) / (ejecta_radius / gamma)
            comoving_temp_temperature = (internal_energy / (radiation_constant * comoving_volume)) ** (1./4.)
        else:
            comoving_emitted_luminosity = (internal_energy * speed_of_light) / (tau_temp * ejecta_radius / gamma)
            comoving_temp_temperature = (internal_energy / (radiation_constant * comoving_volume * tau_temp)) ** (1./4.)

        emitted_luminosity = comoving_emitted_luminosity * doppler_factor_temp ** 2

        thermal_eff = thermalisation_efficiency * np.exp(-1 / tau_temp)

        drdt = (beta * speed_of_light) / (1 - beta)
        dswept_mass_dt = 4 * np.pi * ejecta_radius ** 2 * n_ism * proton_mass * drdt
        dedt = thermalisation_efficiency * mag_lum[
            i] + doppler_factor_temp ** 2 * comoving_radiative_luminosity - doppler_factor_temp ** 2 * comoving_emitted_luminosity
        comoving_dinternal_energydt = thermal_eff * doppler_factor_temp ** (-2) * mag_lum[
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

    return lorentz_factor, lbol_rest, comoving_temperature, radius, doppler_factor, tau


def _comoving_blackbody_to_flux_density(dl, frequencies, radius, temperature, doppler_factor):
    """
    :param dl: luminosity distance in cm
    :param frequencies: frequencies to calculate in Hz - Must be same length as time array or a single number
    :param radius: ejecta radius in cm
    :param temperature: comoving temperature in K
    :param doppler_factor: doppler_factor
    :return: flux_density
    """
    frequencies = frequencies * uu.Hz
    radius = radius * uu.cm
    dl = dl * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 2 * np.pi * planck * frequencies ** 3 * radius ** 2
    denom = dl ** 2 * speed_of_light ** 2 * doppler_factor ** 2
    frac = 1. / (np.exp((planck * frequencies) / (boltzmann_constant * temperature * doppler_factor)) - 1)
    flux_density = num / denom * frac
    return flux_density


def _comoving_blackbody_to_luminosity(frequencies, radius, temperature, doppler_factor):
    """
    :param frequencies: frequencies to calculate in Hz - Must be same length as time array or a single number
    :param radius: ejecta radius in cm
    :param temperature: comoving temperature in K
    :param doppler_factor: doppler_factor
    :return: luminosity
    """
    frequencies = frequencies * uu.Hz
    radius = radius * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 8 * np.pi ** 2 * planck * frequencies ** 4 * radius ** 2
    denom = speed_of_light ** 2 * doppler_factor ** 2
    frac = 1. / (np.exp((planck * frequencies) / (boltzmann_constant * temperature * doppler_factor)) - 1)
    luminosity = num / denom * frac
    return luminosity


def mergernova(time, redshift, frequencies, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
               thermalisation_efficiency, **kwargs):
    """
    :param time: time in observer frame
    :param redshift: redshift
    :param frequencies: frequencies to calculate - Must be same length as time array or a single number
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: output_format - whether to output flux density or AB magnitude
    :return: flux density or AB magnitude
    """
    time_temp = np.logspace(-4, 8, 1000)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    _, bolometric_luminosity, comoving_temperature, radius, doppler_factor, _ = ejecta_dynamics_and_interaction(
        time=time_temp, mej=mej,
        beta=beta, ejecta_radius=ejecta_radius,
        kappa=kappa, n_ism=n_ism, l0=l0,
        tau_sd=tau_sd, nn=nn,
        thermalisation_efficiency=thermalisation_efficiency)
    temp_func = interp1d(time_temp, y=comoving_temperature)
    rad_func = interp1d(time_temp, y=radius)
    d_func = interp1d(time_temp, y=doppler_factor)
    # convert to source frame time
    time = time / (1 + redshift)
    temp = temp_func(time)
    rad = rad_func(time)
    df = d_func(time)
    flux_density = _comoving_blackbody_to_flux_density(dl=dl, frequencies=frequencies, radius=rad, temperature=temp,
                                                      doppler_factor=df)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value


def _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                          **kwargs):
    """
    :param time: time in source frame
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :return: luminosity
    """
    time_temp = np.logspace(-4, 8, 1000)
    _, _, comoving_temperature, radius, doppler_factor, tau = ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                                                                              beta=beta,
                                                                                              ejecta_radius=ejecta_radius,
                                                                                              kappa=kappa, n_ism=n_ism,
                                                                                              l0=l0,
                                                                                              tau_sd=tau_sd, nn=nn,
                                                                                              thermalisation_efficiency=thermalisation_efficiency)
    temp_func = interp1d(time_temp, y=comoving_temperature)
    rad_func = interp1d(time_temp, y=radius)
    d_func = interp1d(time_temp, y=doppler_factor)
    tau_func = interp1d(time_temp, y=tau)
    temp = temp_func(time)
    rad = rad_func(time)
    df = d_func(time)
    optical_depth = tau_func(time)
    frequency = kwargs['frequency']
    trapped_ejecta_lum = _comoving_blackbody_to_luminosity(frequencies=frequency, radius=rad,
                                                          temperature=temp, doppler_factor=df)
    lsd = magnetar_only(time, l0=l0, tau=tau_sd, nn=nn)
    lum = np.exp(-optical_depth) * lsd + trapped_ejecta_lum
    return lum


def _trapped_magnetar_flux(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                           thermalisation_efficiency, photon_index, **kwargs):
    """
    :param time: time in observer frame
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' used to calculate k correction and convert from luminosity to flux
    :return: integrated flux
    """
    lum = _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                                **kwargs)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    kcorr = (1. + redshift) ** (photon_index - 2)
    flux = lum / (4 * np.pi * dl ** 2 * kcorr)
    return flux


def trapped_magnetar(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                     **kwargs):
    """
    :param time: time in source frame or observer frame depending on kwarg
    :param redshift: redshift - not used if evaluating luminosity
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' only used if calculating the flux lightcurve
    :return: luminosity or integrated flux
    """
    if kwargs['output_format'] == 'luminosity':
        return _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                     thermalisation_efficiency, **kwargs)
    elif kwargs['output_format'] == 'flux':
        return _trapped_magnetar_flux(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                      thermalisation_efficiency, **kwargs)
