import numpy as np
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, calc_tfb
import redback.constants as cc
import redback.transient_models.phenomenological_models as pm
from collections import namedtuple
import astropy.units as uu
from scipy.interpolate import interp1d

def _analytic_fallback(time, l0, t_0):
    """
    :param time: time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :return: bolometric luminosity
    """
    mask = time - t_0 > 0.
    lbol = np.zeros(len(time))
    lbol[mask] = l0 / (time[mask] * 86400)**(5./3.)
    lbol[~mask] = l0 / (t_0 * 86400)**(5./3.)
    return lbol

def _semianalytical_fallback():
    pass

def _metzger_tde(mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs:
    :return: named tuple with bolometric luminosity, photosphere radius, temperature, and other parameters
    """
    t0 = kwargs.get('t0', 1.0)
    binding_energy_const = kwargs.get('binding_energy_const', 0.8)
    zeta = kwargs.get('zeta',2.0)
    hoverR = kwargs.get('hoverR', 0.3)
    # minimum value of SMBH feedback efficiency
    etamin = 0.01 * (stellar_mass ** (-7. / 15.) * mbh_6 ** (2. / 3.))
    # maximum TDE penetration factor before star is swallowed whole by SMBH
    beta_max = 12. * (stellar_mass ** (7. / 15.)) * (mbh_6 ** (-2. / 3.))

    # gravitational radius
    Rg = cc.graviational_constant * mbh_6 * 1.0e6 * (cc.solar_mass / cc.speed_of_light ** (2.0))
    # stellar mass in cgs
    Mstar = stellar_mass * cc.solar_mass
    # stellar radius in cgs
    Rstar = stellar_mass ** (0.8) * cc.solar_radius
    # tidal radius
    Rt = Rstar * (mbh_6 * 1.0e6 / stellar_mass) ** (1. / 3.)
    # circularization radius
    Rcirc = 2.0 * Rt / beta
    # fall-back time of most tightly bound debris
    tfb = calc_tfb(binding_energy_const, mbh_6, stellar_mass)
    # Eddington luminosity of SMBH in units of 1e40 erg/s
    Ledd40 = 1.4e4 * mbh_6
    time_temp = np.logspace(np.log10(1.0*tfb), np.log10(100*tfb), 500)
    tdays = time_temp/cc.day_to_s

    #set up grids
    # mass of envelope in Msun
    Me = np.empty_like(tdays)
    # thermal energy of envelope in units of 1e40 ergs
    Ee40 = np.empty_like(tdays)
    # virial radius of envelope in cm
    Rv = np.empty_like(tdays)
    # accretion stream radius
    Racc = np.empty_like(tdays)
    # photosphere radius of envelop in cm
    Rph = np.empty_like(tdays)
    # fallback accretion luminosity in units of 1e40 erg/s
    Edotfb40 = np.empty_like(tdays)
    # accretion timescale onto SMBH
    tacc = np.empty_like(tdays)
    # feedback luminosity from SMBH in units of 1e40 erg/s
    Edotbh40 = np.empty_like(tdays)
    # accretion rate onto SMBH in units of g/s
    MdotBH = np.empty_like(tdays)
    # effective temperature of envelope emission in K
    Teff = np.empty_like(tdays)
    # bolometric luminosity of envelope thermal emission
    Lrad = np.empty_like(tdays)
    # nuLnu luminosity of envelope thermal emission at frequency nu
    nuLnu = np.empty_like(tdays)
    # characteristic optical depth through envelope
    Lamb = np.empty_like(tdays)
    # proxy x-ray luminosity (not used directly in optical light curve calculation)
    LX40 = np.empty_like(tdays)

    Mdotfb = (0.8 * Mstar / (3.0 * tfb)) * (time_temp / tfb) ** (-5. / 3.)

    # ** initialize grid quantities at t = t0 [grid point 0] **
    # initial envelope mass at t0
    Me[0] = 0.1 * Mstar + (0.4 * Mstar) * (1.0 - t0 ** (-2. / 3.))
    # initial envelope radius determined by energy of TDE process
    Rv[0] = (2. * Rt ** (2.0) / (5.0 * binding_energy_const * Rstar)) * (Me[0] / Mstar)
    # initial thermal energy of envelope
    Ee40[0] = ((2.0 * cc.graviational_constant * mbh_6 * 1.0e6 * Me[0]) / (5.0 * Rv[0])) * 2.0e-7
    # initial characteristic optical depth
    Lamb[0] = 0.38 * Me[0] / (10.0 *np.pi * Rv[0] ** (2.0))
    # initial photosphere radius
    Rph[0] = Rv[0] * (1.0 + np.log(Lamb[0]))
    # initial fallback stream accretion radius
    Racc[0] = zeta * Rv[0]
    # initial fallback accretion heating rate in 1e40 erg/s
    Edotfb40[0] = (cc.graviational_constant * mbh_6 * 1.0e6 * Mdotfb[0] / Racc[0]) * (2.0e-7)
    # initial luminosity of envelope
    Lrad[0] = Ledd40 + Edotfb40[0]
    # initial SMBH accretion timescale in s
    tacc[0] = 2.2e-17 * (10. / (3. * alpha)) * (Rv[0] ** (2.0)) / (cc.graviational_constant * mbh_6 * 1.0e6 * Rcirc) ** (0.5) * (hoverR) ** (
        -2.0)
    # initial SMBH accretion rate in g/s
    MdotBH[0] = (Me[0] / tacc[0])
    # initial SMBH feedback heating rate in 1e40 erg/s
    Edotbh40[0] = eta * cc.speed_of_light ** (2.0) * (Me[0] / tacc[0]) * (1.0e-40)
    # initial photosphere temperature of envelope in K
    Teff[0] = 1.0e10 * ((Ledd40 + Edotfb40[0]) / (4.0 * np.pi * cc.sigma_sb * Rph[0] ** (2.0))) ** (0.25)

    t = time_temp
    for ii in range(len(time_temp) - 1):
        ii = ii + 1
        Me[ii] = Me[ii - 1] - (MdotBH[ii - 1] - Mdotfb[ii - 1]) * (t[ii] - t[ii - 1])
        # update envelope energy due to SMBH heating + radiative losses
        Ee40[ii] = Ee40[ii - 1] + (Ledd40 - Edotbh40[ii - 1]) * (t[ii] - t[ii - 1])
        # update envelope radius based on its new energy
        Rv[ii] = ((2.0 * cc.graviational_constant * mbh_6 * 1.0e6 * Me[ii]) / (5.0 * Ee40[ii])) * (2.0e-7)
        # update envelope optical depth
        Lamb[ii] = 0.38 * Me[ii] / (10.0 *np.pi * Rv[ii] ** (2.0))
        # update envelope photosphere radius
        Rph[ii] = Rv[ii] * (1.0 + np.log(Lamb[ii]))
        # update accretion radius
        Racc[ii] = zeta * Rv[0] * (t[ii] / tfb) ** (2. / 3.)
        # update fall-back heating rate in 1e40 erg/s
        Edotfb40[ii] = (cc.graviational_constant * mbh_6 * 1.0e6 * Mdotfb[ii] / Racc[ii]) * (2.0e-7)
        # update total radiated luminosity
        Lrad[ii] = Ledd40 + Edotfb40[ii]
        # update photosphere temperature in K
        Teff[ii] = 1.0e10 * ((Ledd40 + Edotfb40[ii]) / (4.0 *np.pi * cc.sigma_sb * Rph[ii] ** (2.0))) ** (0.25)
        # update SMBH accretion timescale in seconds
        tacc[ii] = 2.2e-17 * (10. / (3.0 * alpha)) * (Rv[ii] ** (2.0)) / (cc.graviational_constant * mbh_6 * 1.0e6 * Rcirc) ** (0.5) * (
            hoverR) ** (-2.0)
        # update SMBH accretion rate in g/s
        MdotBH[ii] = (Me[ii] / tacc[ii])
        # update proxy X-ray luminosity
        LX40[ii] = 0.01 * (MdotBH[ii] / 1.0e20) * (cc.speed_of_light ** (2.0) / 1.0e20)
        # update SMBH feedback heating rate
        Edotbh40[ii] = eta * cc.speed_of_light ** (2.0) * (Me[ii] / tacc[ii]) * (1.0e-40)

    output = namedtuple('output', ['bolometric_luminosity', 'photosphere_temperature',
                                   'photosphere_radius', 'lum_xray', 'accretion_radius',
                                   'SMBH_accretion_rate', 'time_temp', 'nulnu',
                                   'time_since_fb','tfb', 'lnu'])
    constraint_1 = np.min(np.where(Rv < Rcirc / 2.))
    constraint_2 = np.min(np.where(Me < 0.0))
    constraint = np.min([constraint_1, constraint_2])
    nu = 6.0e14
    expon = 1. / (np.exp(cc.planck * nu / (cc.boltzmann_constant * Teff)) - 1.0)
    nuLnu40 = (8.0*np.pi ** (2.0) * Rph ** (2.0) / cc.speed_of_light ** (2.0))
    nuLnu40 = nuLnu40 * ((cc.planck * nu) * (nu ** (2.0))) / 1.0e30
    nuLnu40 = nuLnu40 * expon
    nuLnu40 = nuLnu40 * (nu / 1.0e10)

    output.bolometric_luminosity = Lrad[:constraint] * 1e40
    output.photosphere_temperature = Teff[:constraint]
    output.photosphere_radius = Rph[:constraint]
    output.lum_xray = LX40[:constraint]
    output.accretion_radius = Racc[:constraint]
    output.SMBH_accretion_rate = MdotBH[:constraint]
    output.time_temp = time_temp[:constraint]
    output.time_since_fb = output.time_temp - output.time_temp[0]
    output.tfb = tfb
    output.nulnu = nuLnu40[:constraint] * 1e40
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def metzger_tde(time, redshift,  mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    This model is only valid for time after circulation. Use the gaussianrise_metzgertde model for the full lightcurve

    :param time: time in observer frame in days
    :param redshift: redshift
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs: Additional parameters
    :param output_format: whether to output flux density or AB magnitude
    :param frequency: (frequency to calculate - Must be same length as time array or a single number)
    :return: flux density or AB magnitude
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    output = _metzger_tde(mbh_6, stellar_mass, eta, alpha, beta, **kwargs)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    # interpolate properties onto observation times post tfb
    temp_func = interp1d(output.time_since_fb, y=output.photosphere_temperature)
    rad_func = interp1d(output.time_since_fb, y=output.photosphere_radius)

    # convert to source frame time and frequency
    time = time * cc.day_to_s
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

    temp = temp_func(time)
    photosphere = rad_func(time)

    flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                             dl=dl, frequency=frequency)

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('redback')
def gaussianrise_metzger_tde(time, redshift, peak_time, sigma, mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    This model is only valid for time after circulation. Use the gaussianrise_metzgertde model for the full lightcurve

    :param time: time in observer frame in days
    :param redshift: redshift
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs: Additional parameters
    :param output_format: whether to output flux density or AB magnitude
    :param frequency: (frequency to calculate - Must be same length as time array or a single number)
    :return: flux density or AB magnitude
    """
    binding_energy_const = kwargs.get('binding_energy_const', 0.8)
    tfb = calc_tfb(binding_energy_const, mbh_6, stellar_mass)
    frequency = kwargs['frequency']

    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency

    dl = cosmo.luminosity_distance(redshift).cgs.value

    f1 = pm.gaussian_rise(time=tfb, a_1=1, peak_time=peak_time*cc.day_to_s, sigma=sigma*cc.day_to_s)
    output = _metzger_tde(mbh_6, stellar_mass, eta, alpha, beta, **kwargs)

    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    unique_frequency = np.sort(np.unique(frequency))

    f2 = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature[0],
                                       r_photosphere=output.photosphere_radius[0],
                                       dl=dl, frequency=unique_frequency).to(uu.mJy)
    norms = f2.value / f1
    norm_dict = dict(zip(unique_frequency, norms))

    # build flux density function for each frequency
    flux_den_interp_func = {}
    for freq in unique_frequency:
        tt_pre_fb = np.linspace(-100, output.time_temp[0]/cc.day_to_s - 0.01, 100) * cc.day_to_s
        tt_post_fb = output.time_temp
        total_time = np.concatenate([tt_pre_fb, tt_post_fb])
        f1 = pm.gaussian_rise(time=tt_pre_fb, a_1=norm_dict[freq],
                         peak_time=peak_time * cc.day_to_s, sigma=sigma * cc.day_to_s)
        f2 = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature,
                                           r_photosphere=output.photosphere_radius,
                                           dl=dl, frequency=freq).to(uu.mJy)
        flux_den = np.concatenate([f1, f2.value])
        flux_den_interp_func[freq] = interp1d(total_time, flux_den)

    # interpolate onto actual observed frequency and time values
    flux_density = []
    for freq, tt in zip(frequency, time):
        flux_density.append(flux_den_interp_func[freq](tt*cc.day_to_s))

    flux_density = flux_density * uu.mJy
    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('redback')
def tde_analytical_bolometric(time, l0, t_0, **kwargs):
    """
    :param time: rest frame time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
                e.g., for Diffusion: kappa, kappa_gamma, mej (solar masses), vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
    lbol = _analytic_fallback(time=time, l0=l0, t_0=t_0)
    if _interaction_process is not None:
        interaction_class = _interaction_process(time=time, luminosity=lbol, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('redback')
def tde_analytical(time, redshift, l0, t_0, **kwargs):
    """
    :param time: rest frame time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: TemperatureFloor
    :param sed: CutoffBlackbody must have cutoff_wavelength in kwargs or it will default to 3000 Angstrom
    :param kwargs: Must be all the kwargs required by the specific interaction_process
     e.g., for Diffusion TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.CutoffBlackbody)

    frequency = kwargs['frequency']
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    lbol = tde_analytical_bolometric(time=time, l0=l0, t_0=t_0, **kwargs)

    photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
    sed_1 = kwargs['sed'](time=time, temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                 frequency=frequency, luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)

    flux_density = sed_1.flux_density
    flux_density = np.nan_to_num(flux_density)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract')
def tde_semianalytical():
    pass