from redback.constants import *
from redback.transient_models.magnetar_models import magnetar_only
import numpy as np
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.interpolate import interp1d
import astropy.units as uu # noqa
import astropy.constants as cc # noqa
from redback.utils import interpolated_barnes_and_kasen_thermalisation_efficiency, blackbody_to_flux_density, electron_fraction_from_kappa

def metzger_magnetar_boosted_kilonova_model(time, redshift, frequencies, mej, vej, beta, kappa_r, l0, tau_sd, nn, thermalisation_efficiency, **kwargs):
    """
    :param time: observer frame time
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: neutron_precursor_switch, pair_cascade_switch, ejecta_albedo, rprocess_mass_fraction, magnetar_heating
    :return: flux_density or magnitude
    """
    time_temp = np.geomspace(1e-4, 1e7, 300)
    bolometric_luminosity, temperature, r_photosphere = _metzger_magnetar_boosted_kilonova_model(time_temp, redshift, mej, vej, beta,
                                                                                               kappa_r, l0, tau_sd, nn,
                                                                                               thermalisation_efficiency, **kwargs)
    #k correction/source frame
    time = time / (1 + redshift)
    frequencies = frequencies / (1 + redshift)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    # interpolate properties onto observation times
    temp_func = interp1d(time_temp, y=temperature)
    rad_func = interp1d(time_temp, y=r_photosphere)
    # convert to source frame time and frequency
    time = time / (1 + redshift)
    frequencies = frequencies / (1 + redshift)

    temp = temp_func(time)
    photosphere = rad_func(time)

    flux_density = blackbody_to_flux_density(temperature, r_photosphere, dl, frequencies)

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def _metzger_magnetar_boosted_kilonova_model(time, mej, vej, beta, kappa_r, l0, tau_sd, nn, thermalisation_efficiency, **kwargs):
    """
    :param time: time array to evaluate model on in source frame
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity 
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: neutron_precursor_switch, pair_cascade_switch, ejecta_albedo, magnetar_heating
    :return: bolometric_luminosity, temperature, photosphere_radius
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    neutron_precursor_switch = kwargs.get('neutron_precursor_switch', True)
    magnetar_heating = kwargs.get('magnetar_heating', 'first_layer')

    if kwargs['pair_cascade_switch'] == True:
        ejecta_albedo = kwargs.get('ejecta_albedo', 0.5)

    time = time
    tdays = time/86400
    time_len = len(time)
    mass_len = 250

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    electron_fraction = electron_fraction_from_kappa(kappa_r)
    t0 = 1.3 #seconds
    sig = 0.11  #seconds
    tau_neutron = 900  #seconds

    # convert to astrophysical units
    m0 = mej * solar_mass
    v0 = vej * speed_of_light
    ek_tot_0 = 0.5 * m0 * v0 ** 2

    # set up mass and velocity layers
    m_array = np.logspace(-8, np.log10(mej), mass_len) #in solar masses
    v_m = v0 * (m_array / (mej)) ** (-1 / beta)
    # don't violate relativity
    v_m[v_m > 3e10] = speed_of_light

    # set up arrays
    time_array = np.tile(time, (mass_len, 1))
    e_th_array = np.tile(e_th, (mass_len, 1))
    edotr = np.zeros((mass_len, time_len))

    time_mask = time > t0
    time_1 = time_array[:, time_mask]
    time_2 = time_array[:, ~time_mask]
    edotr[:,time_mask] = 2.1e10 * e_th_array[:, time_mask] * ((time_1/ (3600. * 24.)) ** (-1.3))
    edotr[:, ~time_mask] = 4.0e18 * (0.5 - (1. / np.pi) * np.arctan((time_2 - t0) / sig)) ** (1.3) * e_th_array[:,~time_mask]
    lsd = magnetar_only(time, l0=l0, tau=tau_sd, nn=nn)
    qdot_magnetar = thermalisation_efficiency * lsd

    # set up empty arrays
    energy_v = np.zeros((mass_len, time_len))
    lum_rad = np.zeros((mass_len, time_len))
    qdot_rp = np.zeros((mass_len, time_len))
    td_v = np.zeros((mass_len, time_len))
    tau = np.zeros((mass_len, time_len))
    v_photosphere = np.zeros(time_len)
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
        kappa_r = kappa_r * rprocess_mass_fraction_array
        kappa_r = kappa_n + kappa_r

    dt = np.diff(time)
    dm = np.diff(m_array)

    #initial conditions
    energy_v[:, 0] = 0.5 * m_array*v_m**2
    lum_rad[:, 0] = 0
    qdot_rp[:, 0] = 0
    kinetic_energy = ek_tot_0

    # solve ODE using euler method for all mass shells v
    for ii in range(time_len - 1):
        # this works for kn only
        # td_v[:-1, ii] = (kappa_r* m_array[:-1] * solar_mass * 3)/ (4*np.pi*v_m[:-1] * speed_of_light * time[ii] * beta)
        # lum_rad[:-1, ii] = energy_v[:-1, ii] / (td_v[:-1, ii] + time[ii] * (v_m[:-1] / speed_of_light))
        # energy_v[:-1, ii + 1] = (edotr[:-1, ii] - (energy_v[:-1, ii] / time[ii]) - lum_rad[:-1, ii]) * dt[ii] + energy_v[:-1, ii]
        # lum_rad[:-1, ii] = lum_rad[:-1, ii] * dm * solar_mass

        # # evolve the velocity due to pdv work of central shell of mass M and thermal energy Ev0
        kinetic_energy = kinetic_energy + (np.sum(energy_v[:, ii]) / time[ii]) * dt[ii]
        v0 = (2 * kinetic_energy / m0) ** 0.5
        v_m = v0 * (m_array / (mej)) ** (-1 / beta)
        v_m[v_m > 3e10] = speed_of_light

        if magnetar_heating == 'all_layers':
            if neutron_precursor_switch:
                td_v[:-1, ii] = (kappa_r[:-1,ii] * m_array[:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            else:
                td_v[:-1, ii] = (kappa_r* m_array[:-1] * solar_mass * 3)/ (4*np.pi*v_m[:-1] * speed_of_light * time[ii] * beta)

            lum_rad[:-1, ii] = energy_v[:-1, ii] / (td_v[:-1, ii] + time[ii] * (v_m[:-1] / speed_of_light))
            energy_v[:-1, ii + 1] = (qdot_magnetar[ii] + edotr[:-1, ii] * dm * solar_mass - (energy_v[:-1, ii] / time[ii]) - lum_rad[:-1, ii]) * dt[ii] + energy_v[:-1, ii]

        # first mass layer
        # only bottom layer i.e., 0'th mass layer gets magnetar contribution
        if magnetar_heating == 'first_layer':
            if neutron_precursor_switch:
                td_v[0, ii] = (kappa_r[0, ii] * m_array[0] * solar_mass * 3) / (
                            4 * np.pi * v_m[0] * speed_of_light * time[ii] * beta)
                td_v[1:-1, ii] = (kappa_r[1:-1, ii] * m_array[1:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[1:-1] * speed_of_light * time[ii] * beta)
            else:
                td_v[0, ii] = (kappa_r* m_array[0] * solar_mass * 3)/ (4*np.pi*v_m[0] * speed_of_light * time[ii] * beta)
                td_v[1:-1, ii] = (kappa_r * m_array[1:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[1:-1] * speed_of_light * time[ii] * beta)

            lum_rad[0, ii] = energy_v[0, ii] / (td_v[0, ii] + time[ii] * (v_m[0] / speed_of_light))
            energy_v[0, ii + 1] = (qdot_magnetar[ii] + edotr[0, ii] * dm[0] * solar_mass - (energy_v[0, ii] / time[ii]) - lum_rad[0, ii]) * dt[ii] + energy_v[0, ii]
            # other layers
            lum_rad[1:-1, ii] = energy_v[1:-1, ii] / (td_v[1:-1, ii] + time[ii] * (v_m[1:-1] / speed_of_light))
            energy_v[1:-1, ii + 1] = (edotr[1:-1, ii] * dm[1:] * solar_mass - (energy_v[1:-1, ii] / time[ii]) - lum_rad[1:-1, ii]) * dt[ii] + energy_v[1:-1, ii]

        if neutron_precursor_switch:
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa_r[:-1, ii] / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))
        else:
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa_r / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))

        tau[mass_len - 1, ii] = tau[mass_len - 2, ii]
        photosphere_index = np.argmin(np.abs(tau[:, ii] - 1))
        v_photosphere[ii] = v_m[photosphere_index]
        r_photosphere[ii] = v_photosphere[ii] * time[ii]

    bolometric_luminosity = np.sum(lum_rad, axis=0)

    if pair_cascade_switch == True:
        tlife_t = (0.6/(1 - ejecta_albedo))*(electron_fraction/0.1)**0.5 * (bolometric_luminosity/1.0e45)**0.5 \
                  * (v0/(0.3*speed_of_light))**(0.5) * (time/86400)**(-0.5)
        bolometric_luminosity = bolometric_luminosity / (1.0 + tlife_t)

    temperature = (bolometric_luminosity / (4.0 * np.pi * (r_photosphere) ** (2.0) * sigma_sb)) ** (0.25)

    return bolometric_luminosity, temperature, r_photosphere

def ejecta_dynamics_and_interaction(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                    thermalisation_efficiency, **kwargs):
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
    ## adding units back in to ensure dimensions are correct
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
    ## adding units back in to ensure dimensions are correct
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
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: output_format - whether to output flux density or AB magnitude
    :return: flux density or AB magnitude
    """
    time_temp = np.geomspace(1e-4, 1e8, 1000, endpoint=True)
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
    # convert to source frame time and frequency
    time = time / (1 + redshift)
    frequencies = frequencies / (1 + redshift)

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
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :return: luminosity
    """
    time_temp = np.geomspace(1e-4, 1e8, 1000, endpoint=True)
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
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' used to calculate k correction and convert from luminosity to flux
    :return: integrated flux
    """
    time = time / (1 + redshift)
    kwargs['frequency'] = kwargs['frequency'] / (1 + redshift)

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
    :param l0: initial magnetar X-ray luminosity
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
