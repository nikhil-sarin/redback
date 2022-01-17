from redback.constants import *
from redback.transient_models.magnetar_models import magnetar_only
import numpy as np
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.interpolate import interp1d
import astropy.units as uu
import astropy.constants as cc

def metzger_magnetar_boosted_kilonova_model(time, **kwargs):
    pass

def ejecta_dynamics_and_interaction(time, mej, beta,ejecta_radius,kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency, **kwargs):
    mej = mej * solar_mass
    lorentz_factor = []
    radius = []
    doppler_factor = []
    lbol_ejecta = []
    lbol_rest = []
    comoving_temperature = []

    internal_energy = 0.5 * beta ** 2 * mej * speed_of_light ** 2
    comoving_volume = (4 / 3) * np.pi * ejecta_radius ** 3
    gamma = 1 / np.sqrt(1 - beta ** 2)
    mag_lum = magnetar_only(time, l0=l0,tau=tau_sd,nn=nn)

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
        tau = kappa * (mej / comoving_volume) * (ejecta_radius / gamma)

        if tau <= 1:
            comoving_emitted_luminosity = (internal_energy * speed_of_light) / (ejecta_radius / gamma)
            comoving_temp_temperature = (internal_energy / (radiation_constant * comoving_volume))**(1./4.)
        else:
            comoving_emitted_luminosity = (internal_energy * speed_of_light) / (tau * ejecta_radius / gamma)
            comoving_temp_temperature = (internal_energy / (radiation_constant * comoving_volume * tau))**(1./4.)

        emitted_luminosity = comoving_emitted_luminosity * doppler_factor_temp ** 2

        thermal_eff = thermalisation_efficiency * np.exp(-1 / tau)

        drdt = (beta * speed_of_light) / (1 - beta)
        dswept_mass_dt = 4 * np.pi * ejecta_radius ** 2 * n_ism * proton_mass * drdt
        dedt = thermalisation_efficiency * mag_lum[i] + doppler_factor_temp ** 2 * comoving_radiative_luminosity - doppler_factor_temp ** 2 * comoving_emitted_luminosity
        comoving_dinternal_energydt = thermal_eff * doppler_factor_temp ** (-2) * mag_lum[i] + comoving_radiative_luminosity - comoving_emitted_luminosity - comoving_pressure * (comoving_dvdt)
        dcomoving_volume_dt = comoving_dvdt * doppler_factor_temp
        dinternal_energy_dt = comoving_dinternal_energydt * doppler_factor_temp
        dgamma_dt = (dedt - gamma * doppler_factor_temp * comoving_dinternal_energydt - (gamma ** 2 - 1) * speed_of_light ** 2 * dswept_mass_dt) / (
                    mej * speed_of_light ** 2 + internal_energy + 2 * gamma * swept_mass * speed_of_light ** 2)
        lorentz_factor.append(gamma)
        lbol_ejecta.append(comoving_emitted_luminosity)
        lbol_rest.append(emitted_luminosity)
        comoving_temperature.append(comoving_temp_temperature)
        radius.append(ejecta_radius)
        doppler_factor.append(doppler_factor_temp)

    return lorentz_factor, lbol_rest, comoving_temperature, radius, doppler_factor

def mergernova(time, redshift, frequencies, mej, beta,ejecta_radius,kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency, **kwargs):
    time_temp = np.logspace(-4,8,1000)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    _, bolometric_luminosity, comoving_temperature, radius, doppler_factor = ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                                                            beta=beta, ejecta_radius=ejecta_radius,
                                                                            kappa=kappa,n_ism=n_ism, l0=l0,
                                                                            tau_sd=tau_sd, nn=nn,
                                                                            thermalisation_efficiency=thermalisation_efficiency)
    temp_func = interp1d(time_temp, y=comoving_temperature)
    rad_func = interp1d(time_temp, y=radius)
    d_func = interp1d(time_temp, y=doppler_factor)
    temp = temp_func(time)
    rad = rad_func(time)
    df = d_func(time)
    flux_density = comoving_blackbody_to_flux_density(dl=dl, frequencies=frequencies, radius=rad, temperature=temp, doppler_factor=df)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def comoving_blackbody_to_flux_density(dl, frequencies, radius, temperature, doppler_factor):
    frequencies = frequencies * uu.Hz
    radius = radius * uu.cm
    dl = dl * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 2*np.pi*planck*frequencies**3 * radius**2
    denom = dl**2 * speed_of_light**2 * doppler_factor**2
    frac = 1./(np.exp((planck*frequencies)/(boltzmann_constant*temperature*doppler_factor)) - 1)
    flux_density = num/denom * frac
    return flux_density

def trapped_magnetar_lum(time, **kwargs):
    alpha = (1 + nn)/(1 - nn)
    omegat = omega0 * (1. + time/tau)**(alpha)
    lsd = eta * bp**2 * radius**6 * omegat**4
    doppler = 1/(gamma * (1 - beta*np.cos(theta)))
    lnu_x_bb = (8*np.pi**2*doppler**2 *radius**2)/(planck**3*speed_of_light**2)
    tau = kappa * (mej/comoving_volumerime) * (radius/lorentz_factor)
    lum = e**(-tau) * lsd + (lnu_x_bb)
    return lum


def trapped_magnetar_flux(time, **kwargs):
    lum = trapped_magnetar_lum(time, **kwargs)
    kcorr = (1. + redshift)**(photon_index - 2)
    flux = lum/(4*np.pi*dl**2 * kcorr)


