import numpy as np
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from redback.utils import calc_kcorrected_properties, citation_wrapper, calc_tfb, lambda_to_nu, \
    calc_ABmag_from_flux_density, calc_flux_density_from_ABmag, bands_to_frequency
import redback.constants as cc
import redback.transient_models.phenomenological_models as pm

from collections import namedtuple
from astropy.cosmology import Planck18 as cosmo  # noqa
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

def _cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs:
    :return: named tuple with bolometric luminosity, photosphere radius, temperature, and other parameters
    """
    t_0_init = kwargs.get('t_0_init', 1.0)
    binding_energy_const = kwargs.get('binding_energy_const', 0.8)
    zeta = kwargs.get('zeta',2.0)
    hoverR = kwargs.get('hoverR', 0.3)

    # gravitational radius
    Rg = cc.graviational_constant * mbh_6 * 1.0e6 * (cc.solar_mass / cc.speed_of_light ** (2.0))
    # stellar mass in cgs
    Mstar = stellar_mass * cc.solar_mass
    # stellar radius in cgs
    Rstar = stellar_mass ** (0.8) * cc.solar_radius
    # tidal radius
    Rt = Rstar * (mbh_6*1.0e6 /stellar_mass) ** (1./3.)
    # circularization radius
    Rcirc = 2.0*Rt/beta
    # fall-back time of most tightly bound debris
    tfb = calc_tfb(binding_energy_const, mbh_6, stellar_mass)
    # Eddington luminosity of SMBH in units of 1e40 erg/s
    Ledd40 = 1.4e4 * mbh_6
    time_temp = np.logspace(np.log10(1.0*tfb), np.log10(5000*tfb), 5000)
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

    # ** initialize grid quantities at t = t_0_init [grid point 0] **
    # initial envelope mass at t_0_init
    Me[0] = 0.1 * Mstar + (0.4 * Mstar) * (1.0 - t_0_init**(-2. / 3.))
    # initial envelope radius determined by energy of TDE process
    Rv[0] = (2. * Rt**(2.0)/(5.0 * binding_energy_const * Rstar)) * (Me[0]/Mstar)
    # initial thermal energy of envelope
    Ee40[0] = ((2.0 * cc.graviational_constant * mbh_6 * 1.0e6 * Me[0]) / (5.0 * Rv[0])) * 2.0e-7
    # initial characteristic optical depth
    Lamb[0] = 0.38 * Me[0] / (10.0 *np.pi * Rv[0] ** (2.0))
    # initial photosphere radius
    Rph[0] = Rv[0] * (1.0 + np.log(Lamb[0]))
    # initial fallback stream accretion radius
    Racc[0] = zeta * Rv[0]
    # initial fallback accretion heating rate in 1e40 erg/s
    Edotfb40[0] = (cc.graviational_constant * mbh_6 * 1.0e6 * Mdotfb[0]/Racc[0]) * (2.0e-7)
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
    for ii in range(1, len(time_temp)):
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
                                   'time_since_fb','tfb', 'lnu', 'envelope_radius', 'envelope_mass',
                                   'rtidal', 'rcirc', 'termination_time', 'termination_time_id'])
    try:
        constraint_1 = np.min(np.where(Rv < Rcirc/2.))
        constraint_2 = np.min(np.where(Me < 0.0))
    except ValueError:
        constraint_1 = len(time_temp)
        constraint_2 = len(time_temp)
    constraint = np.min([constraint_1, constraint_2])
    termination_time_id = np.min([constraint_1, constraint_2])
    nu = 6.0e14
    expon = 1. / (np.exp(cc.planck * nu / (cc.boltzmann_constant * Teff)) - 1.0)
    nuLnu40 = (8.0*np.pi ** (2.0) * Rph ** (2.0) / cc.speed_of_light ** (2.0))
    nuLnu40 = nuLnu40 * ((cc.planck * nu) * (nu ** (2.0))) / 1.0e30
    nuLnu40 = nuLnu40 * expon
    nuLnu40 = nuLnu40 * (nu / 1.0e10)

    output.bolometric_luminosity = Lrad[:constraint] * 1e40
    output.photosphere_temperature = Teff[:constraint]
    output.photosphere_radius = Rph[:constraint]
    output.envelope_radius = Rv[:constraint]
    output.envelope_mass = Me[:constraint]
    output.rcirc = Rcirc
    output.rtidal = Rt
    output.lum_xray = LX40[:constraint]
    output.accretion_radius = Racc[:constraint]
    output.SMBH_accretion_rate = MdotBH[:constraint]
    output.time_temp = time_temp[:constraint]
    output.time_since_fb = output.time_temp - output.time_temp[0]
    if constraint == len(time_temp):
        output.termination_time = time_temp[-1] - tfb
    else:
        output.termination_time = time_temp[termination_time_id] - tfb
    output.termination_time_id = termination_time_id
    output.tfb = tfb
    output.nulnu = nuLnu40[:constraint] * 1e40
    return output

@citation_wrapper('https://arxiv.org/abs/2307.15121,https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def cooling_envelope(time, redshift, mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
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
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    output = _cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, **kwargs)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        if isinstance(frequency, float):
            frequency = np.ones(len(time)) * frequency

        # convert to source frame time and frequency
        time = time * cc.day_to_s
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        # interpolate properties onto observation times post tfb
        temp_func = interp1d(output.time_since_fb, y=output.photosphere_temperature)
        rad_func = interp1d(output.time_since_fb, y=output.photosphere_radius)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_observer_frame = output.time_since_fb * (1. + redshift)

        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature,
                                             r_photosphere=output.photosphere_radius,
                                             frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / cc.day_to_s,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://arxiv.org/abs/2307.15121,https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def gaussianrise_cooling_envelope_bolometric(time, peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    Full lightcurve, with gaussian rise till fallback time and then the metzger tde model,
    bolometric version for fitting the bolometric lightcurve

    :param time: time in source frame in days
    :param peak_time: peak time in days
    :param sigma_t: the sharpness of the Gaussian in days
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs: Additional parameters
    :return luminosity in ergs/s
    """
    output = _cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, **kwargs)
    kwargs['binding_energy_const'] = kwargs.get('binding_energy_const', 0.8)
    tfb_sf = calc_tfb(kwargs['binding_energy_const'], mbh_6, stellar_mass)  # source frame
    f1 = pm.gaussian_rise(time=tfb_sf, a_1=1, peak_time=peak_time * cc.day_to_s, sigma_t=sigma_t * cc.day_to_s)

    # get normalisation
    f2 = output.bolometric_luminosity[0]
    norm = f2/f1

    #evaluate giant array of bolometric luminosities
    tt_pre_fb = np.linspace(0, tfb_sf, 100)
    tt_post_fb = output.time_temp
    full_time = np.concatenate([tt_pre_fb, tt_post_fb])
    f1 = pm.gaussian_rise(time=tt_pre_fb, a_1=norm,
                          peak_time=peak_time * cc.day_to_s, sigma_t=sigma_t * cc.day_to_s)
    f2 = output.bolometric_luminosity
    full_lbol = np.concatenate([f1, f2])
    lbol_func = interp1d(full_time, y=full_lbol, fill_value='extrapolate')
    return lbol_func(time*cc.day_to_s)


@citation_wrapper('https://arxiv.org/abs/2307.15121,https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def gaussianrise_cooling_envelope(time, redshift, peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    Full lightcurve, with gaussian rise till fallback time and then the metzger tde model,
    photometric version where each band is fit/joint separately

    :param time: time in observer frame in days
    :param redshift: redshift
    :param peak_time: peak time in days
    :param sigma_t: the sharpness of the Gaussian in days
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs: Additional parameters
    :param xi: Optional argument (default set to one) to change the point where lightcurve switches from Gaussian rise to cooling envelope.
        stitching_point = xi * tfb (where tfb is fallback time). So a xi=1 means the stitching point is at fallback time.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'flux'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'flux'
    """
    binding_energy_const = kwargs.get('binding_energy_const', 0.8)
    tfb_sf = calc_tfb(binding_energy_const, mbh_6, stellar_mass)  # source frame
    tfb_obf = tfb_sf * (1. + redshift)  # observer frame
    xi = kwargs.get('xi', 1.)
    output = _cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, **kwargs)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    stitching_point = xi * tfb_obf

    # normalisation term in observer frame
    f1 = pm.gaussian_rise(time=stitching_point, a_1=1., peak_time=peak_time * cc.day_to_s, sigma_t=sigma_t * cc.day_to_s)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        if isinstance(frequency, float):
            frequency = np.ones(len(time)) * frequency

        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        unique_frequency = np.sort(np.unique(frequency))

        # source frame
        f2 = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature[0],
                                           r_photosphere=output.photosphere_radius[0],
                                           dl=dl, frequency=unique_frequency).to(uu.mJy)
        norms = f2.value / f1
        norm_dict = dict(zip(unique_frequency, norms))

        # build flux density function for each frequency
        flux_den_interp_func = {}
        for freq in unique_frequency:
            tt_pre_fb = np.linspace(0, stitching_point / cc.day_to_s, 200) * cc.day_to_s
            tt_post_fb = xi * (output.time_temp * (1 + redshift))
            total_time = np.concatenate([tt_pre_fb, tt_post_fb])
            f1 = pm.gaussian_rise(time=tt_pre_fb, a_1=norm_dict[freq],
                                  peak_time=peak_time * cc.day_to_s, sigma_t=sigma_t * cc.day_to_s)
            f2 = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature,
                                               r_photosphere=output.photosphere_radius,
                                               dl=dl, frequency=freq).to(uu.mJy)
            flux_den = np.concatenate([f1, f2.value])
            flux_den_interp_func[freq] = interp1d(total_time, flux_den, fill_value='extrapolate')

        # interpolate onto actual observed frequency and time values
        flux_density = []
        for freq, tt in zip(frequency, time):
            flux_density.append(flux_den_interp_func[freq](tt * cc.day_to_s))
        flux_density = flux_density * uu.mJy
        return flux_density.to(uu.mJy).value
    else:
        bands = kwargs['bands']
        if isinstance(bands, str):
            bands = [str(bands) for x in range(len(time))]

        unique_bands = np.unique(bands)
        temp_kwargs = kwargs.copy()
        temp_kwargs['bands'] = unique_bands
        f2 = cooling_envelope(time=0., redshift=redshift,
                              mbh_6=mbh_6, stellar_mass=stellar_mass, eta=eta, alpha=alpha, beta=beta,
                              **temp_kwargs)
        if kwargs['output_format'] == 'magnitude':
            # make the normalisation in fmjy to avoid magnitude normalisation problems
            _f2mjy = calc_flux_density_from_ABmag(f2).value
            norms = _f2mjy / f1
        else:
            norms = f2 / f1

        if isinstance(norms, float):
            norms = np.ones(len(time)) * norms
        norm_dict = dict(zip(unique_bands, norms))

        flux_den_interp_func = {}
        for band in unique_bands:
            tt_pre_fb = np.linspace(0, stitching_point / cc.day_to_s, 100) * cc.day_to_s
            tt_post_fb = output.time_temp * (1 + redshift)
            total_time = np.concatenate([tt_pre_fb, tt_post_fb])
            f1 = pm.gaussian_rise(time=tt_pre_fb, a_1=norm_dict[band],
                                  peak_time=peak_time * cc.day_to_s, sigma_t=sigma_t * cc.day_to_s)
            if kwargs['output_format'] == 'magnitude':
                f1 = calc_ABmag_from_flux_density(f1).value
            temp_kwargs = kwargs.copy()
            temp_kwargs['bands'] = band
            f2 = cooling_envelope(time=output.time_since_fb / cc.day_to_s, redshift=redshift,
                                  mbh_6=mbh_6, stellar_mass=stellar_mass, eta=eta, alpha=alpha, beta=beta,
                                  **temp_kwargs)
            flux_den = np.concatenate([f1, f2])
            flux_den_interp_func[band] = interp1d(total_time, flux_den, fill_value='extrapolate')

        # interpolate onto actual observed band and time values
        output = []
        for freq, tt in zip(bands, time):
            output.append(flux_den_interp_func[freq](tt * cc.day_to_s))
        return np.array(output)

@citation_wrapper('https://arxiv.org/abs/2307.15121,https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def bpl_cooling_envelope(time, redshift, peak_time, alpha_1, alpha_2, mbh_6, stellar_mass, eta, alpha, beta, **kwargs):
    """
    Full lightcurve, with gaussian rise till fallback time and then the metzger tde model,
    photometric version where each band is fit/joint separately

    :param time: time in observer frame in days
    :param redshift: redshift
    :param peak_time: peak time in days
    :param alpha_1: power law index for first power law
    :param alpha_2: power law index for second power law (should be positive)
    :param mbh_6: mass of supermassive black hole in units of 10^6 solar mass
    :param stellar_mass: stellar mass in units of solar masses
    :param eta: SMBH feedback efficiency (typical range: etamin - 0.1)
    :param alpha: disk viscosity
    :param beta: TDE penetration factor (typical range: 1 - beta_max)
    :param kwargs: Additional parameters
    :param xi: Optional argument (default set to one) to change the point where lightcurve switches from Gaussian rise to cooling envelope.
        stitching_point = xi * tfb (where tfb is fallback time). So a xi=1 means the stitching point is at fallback time.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'flux'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'flux'
    """
    binding_energy_const = kwargs.get('binding_energy_const', 0.8)
    tfb_sf = calc_tfb(binding_energy_const, mbh_6, stellar_mass)  # source frame
    tfb_obf = tfb_sf * (1. + redshift)  # observer frame
    xi = kwargs.get('xi', 1.)
    output = _cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, **kwargs)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    stitching_point = xi * tfb_obf

    # normalisation term in observer frame
    f1 = pm.exponential_powerlaw(time=stitching_point, a_1=1., tpeak=peak_time * cc.day_to_s,
                                 alpha_1=alpha_1, alpha_2=alpha_2)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        if isinstance(frequency, float):
            frequency = np.ones(len(time)) * frequency

        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        unique_frequency = np.sort(np.unique(frequency))

        # source frame
        f2 = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature[0],
                                           r_photosphere=output.photosphere_radius[0],
                                           dl=dl, frequency=unique_frequency).to(uu.mJy)
        norms = f2.value / f1
        norm_dict = dict(zip(unique_frequency, norms))

        # build flux density function for each frequency
        flux_den_interp_func = {}
        for freq in unique_frequency:
            tt_pre_fb = np.linspace(0, stitching_point / cc.day_to_s, 200) * cc.day_to_s
            tt_post_fb = xi * (output.time_temp * (1 + redshift))
            total_time = np.concatenate([tt_pre_fb, tt_post_fb])
            f1 = pm.exponential_powerlaw(time=tt_pre_fb, a_1=norm_dict[freq],
                                         tpeak=peak_time * cc.day_to_s, alpha_1=alpha_1, alpha_2=alpha_2)
            f2 = sed.blackbody_to_flux_density(temperature=output.photosphere_temperature,
                                               r_photosphere=output.photosphere_radius,
                                               dl=dl, frequency=freq).to(uu.mJy)
            flux_den = np.concatenate([f1, f2.value])
            flux_den_interp_func[freq] = interp1d(total_time, flux_den, fill_value='extrapolate')

        # interpolate onto actual observed frequency and time values
        flux_density = []
        for freq, tt in zip(frequency, time):
            flux_density.append(flux_den_interp_func[freq](tt * cc.day_to_s))
        flux_density = flux_density * uu.mJy
        return flux_density.to(uu.mJy).value
    else:
        bands = kwargs['bands']
        if isinstance(bands, str):
            bands = [str(bands) for x in range(len(time))]

        unique_bands = np.unique(bands)
        temp_kwargs = kwargs.copy()
        temp_kwargs['bands'] = unique_bands
        f2 = cooling_envelope(time=0., redshift=redshift,
                              mbh_6=mbh_6, stellar_mass=stellar_mass, eta=eta, alpha=alpha, beta=beta,
                              **temp_kwargs)
        if kwargs['output_format'] == 'magnitude':
            # make the normalisation in fmjy to avoid magnitude normalisation problems
            _f2mjy = calc_flux_density_from_ABmag(f2).value
            norms = _f2mjy / f1
        else:
            norms = f2 / f1

        if isinstance(norms, float):
            norms = np.ones(len(time)) * norms
        norm_dict = dict(zip(unique_bands, norms))

        flux_den_interp_func = {}
        for band in unique_bands:
            tt_pre_fb = np.linspace(0, stitching_point / cc.day_to_s, 100) * cc.day_to_s
            tt_post_fb = output.time_temp * (1 + redshift)
            total_time = np.concatenate([tt_pre_fb, tt_post_fb])
            f1 = pm.exponential_powerlaw(time=tt_pre_fb, a_1=norm_dict[band],
                                         tpeak=peak_time * cc.day_to_s, alpha_1=alpha_1, alpha_2=alpha_2)
            if kwargs['output_format'] == 'magnitude':
                f1 = calc_ABmag_from_flux_density(f1).value
            temp_kwargs = kwargs.copy()
            temp_kwargs['bands'] = band
            f2 = cooling_envelope(time=output.time_since_fb / cc.day_to_s, redshift=redshift,
                                  mbh_6=mbh_6, stellar_mass=stellar_mass, eta=eta, alpha=alpha, beta=beta,
                                  **temp_kwargs)
            flux_den = np.concatenate([f1, f2])
            flux_den_interp_func[band] = interp1d(total_time, flux_den, fill_value='extrapolate')

        # interpolate onto actual observed band and time values
        output = []
        for freq, tt in zip(bands, time):
            output.append(flux_den_interp_func[freq](tt * cc.day_to_s))
        return np.array(output)

@citation_wrapper('redback')
def tde_analytical_bolometric(time, l0, t_0_turn, **kwargs):
    """
    :param time: rest frame time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0_turn: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
                e.g., for Diffusion: kappa, kappa_gamma, mej (solar masses), vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
    lbol = _analytic_fallback(time=time, l0=l0, t_0=t_0_turn)
    if _interaction_process is not None:
        dense_resolution = kwargs.get("dense_resolution", 1000)
        dense_times = np.linspace(0, time[-1] + 100, dense_resolution)
        dense_lbols = _analytic_fallback(time=dense_times, l0=l0, t_0=t_0_turn)
        interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('redback')
def tde_analytical(time, redshift, l0, t_0_turn, **kwargs):
    """
    :param time: observer frame time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0_turn: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :param kwargs: Must be all the kwargs required by the specific interaction_process
     e.g., for Diffusion TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: TemperatureFloor
    :param sed: CutoffBlackbody must have cutoff_wavelength in kwargs or it will default to 3000 Angstrom
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
        lbol = tde_analytical_bolometric(time=time, l0=l0, t_0_turn=t_0_turn, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](time=time, temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                     frequency=frequency, luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)

        flux_density = sed_1.flux_density
        flux_density = np.nan_to_num(flux_density)
        return flux_density.to(uu.mJy).value
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 1000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol = tde_analytical_bolometric(time=time, l0=l0, t_0_turn=t_0_turn, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        full_sed = np.zeros((len(time), len(frequency)))
        for ii in range(len(frequency)):
            ss = kwargs['sed'](time=time,temperature=photo.photosphere_temperature,
                                r_photosphere=photo.r_photosphere,frequency=frequency[ii],
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

def _initialize_mosfit_tde_model():
    """
    Initializtion function to load/process data.

    Loads and interpolates tde simulation data. Simulation data is
    from Guillochon 2013 and can be found on astrocrash.net.

    :return: Named tuple with several outputs
    """

    import os
    dirname = os.path.dirname(__file__)
    data_dir = f"{dirname}/../tables/guillochon_tde_data"
    G_cgs = cc.graviational_constant
    Mhbase = 1.0e6 * cc.solar_mass

    gammas = ['4-3', '5-3']

    beta_slope = {gammas[0]: [], gammas[1]: []}
    beta_yinter = {gammas[0]: [], gammas[1]: []}
    sim_beta = {gammas[0]: [], gammas[1]: []}
    mapped_time = {gammas[0]: [], gammas[1]: []}
    premaptime = {gammas[0]: [], gammas[1]: []}
    premapdmdt = {gammas[0]: [], gammas[1]: []}

    for g in gammas:
        dmdedir = os.path.join(data_dir, g)

        sim_beta_files = os.listdir(dmdedir)
        simbeta = [float(b[:-4]) for b in sim_beta_files]
        sortedindices = np.argsort(simbeta)
        simbeta = [simbeta[i] for i in sortedindices]
        sim_beta_files = [sim_beta_files[i] for i in sortedindices]
        sim_beta[g].extend(simbeta)

        time = {}
        dmdt = {}
        ipeak = {}
        _mapped_time = {}

        e, d = np.loadtxt(os.path.join(dmdedir, sim_beta_files[0]))
        ebound = e[e < 0]
        dmdebound = d[e < 0]

        if min(dmdebound) < 0:
            print('beta, gamma, negative dmde bound:', sim_beta[g], g, dmdebound[dmdebound < 0])

        dedt = (1.0 / 3.0) * (-2.0 * ebound) ** (5.0 / 2.0) / (2.0 * np.pi * G_cgs * Mhbase)
        time['lo'] = np.log10((2.0 * np.pi * G_cgs * Mhbase) * (-2.0 * ebound) ** (-3.0 / 2.0))
        dmdt['lo'] = np.log10(dmdebound * dedt)

        ipeak['lo'] = np.argmax(dmdt['lo'])

        time['lo'] = np.array([time['lo'][:ipeak['lo']], time['lo'][ipeak['lo']:]], dtype=object)
        dmdt['lo'] = np.array([dmdt['lo'][:ipeak['lo']], dmdt['lo'][ipeak['lo']:]], dtype=object)

        premaptime[g].append(np.copy(time['lo']))
        premapdmdt[g].append(np.copy(dmdt['lo']))

        for i in range(1, len(sim_beta[g])):
            e, d = np.loadtxt(os.path.join(dmdedir, sim_beta_files[i]))
            ebound = e[e < 0]
            dmdebound = d[e < 0]

            if min(dmdebound) < 0:
                print('beta, gamma, negative dmde bound:', sim_beta[g], g, dmdebound[dmdebound < 0])

            dedt = (1.0 / 3.0) * (-2.0 * ebound) ** (5.0 / 2.0) / (2.0 * np.pi * G_cgs * Mhbase)
            time['hi'] = np.log10((2.0 * np.pi * G_cgs * Mhbase) * (-2.0 * ebound) ** (-3.0 / 2.0))
            dmdt['hi'] = np.log10(dmdebound * dedt)

            ipeak['hi'] = np.argmax(dmdt['hi'])

            time['hi'] = np.array([time['hi'][:ipeak['hi']], time['hi'][ipeak['hi']:]], dtype=object)
            dmdt['hi'] = np.array([dmdt['hi'][:ipeak['hi']], dmdt['hi'][ipeak['hi']:]], dtype=object)

            premapdmdt[g].append(np.copy(dmdt['hi']))
            premaptime[g].append(np.copy(time['hi']))

            _mapped_time['hi'] = []
            _mapped_time['lo'] = []

            beta_slope[g].append([])
            beta_yinter[g].append([])
            mapped_time[g].append([])

            for j in [0, 1]:
                if len(time['lo'][j]) < len(time['hi'][j]):
                    interp = 'lo'
                    nointerp = 'hi'
                else:
                    interp = 'hi'
                    nointerp = 'lo'

                _mapped_time[nointerp].append(
                    1. / (time[nointerp][j][-1] - time[nointerp][j][0]) *
                    (time[nointerp][j] - time[nointerp][j][0]))
                _mapped_time[interp].append(
                    1. / (time[interp][j][-1] - time[interp][j][0]) *
                    (time[interp][j] - time[interp][j][0]))

                _mapped_time[interp][j][0] = 0
                _mapped_time[interp][j][-1] = 1
                _mapped_time[nointerp][j][0] = 0
                _mapped_time[nointerp][j][-1] = 1

                func = interp1d(_mapped_time[interp][j], dmdt[interp][j])
                dmdtinterp = func(_mapped_time[nointerp][j])

                if interp == 'hi':
                    slope = ((dmdtinterp - dmdt['lo'][j]) /
                             (sim_beta[g][i] - sim_beta[g][i - 1]))
                else:
                    slope = ((dmdt['hi'][j] - dmdtinterp) /
                             (sim_beta[g][i] - sim_beta[g][i - 1]))
                beta_slope[g][-1].append(slope)

                yinter1 = (dmdt[nointerp][j] - beta_slope[g][-1][j] *
                           sim_beta[g][i - 1])
                yinter2 = (dmdtinterp - beta_slope[g][-1][j] *
                           sim_beta[g][i])
                beta_yinter[g][-1].append((yinter1 + yinter2) / 2.0)
                mapped_time[g][-1].append(
                    np.array(_mapped_time[nointerp][j]))

            time['lo'] = np.copy(time['hi'])
            dmdt['lo'] = np.copy(dmdt['hi'])

    outs = namedtuple('sim_outputs', ['beta_slope', 'beta_yinter', 'sim_beta', 'mapped_time',
                                      'premaptime', 'premapdmdt'])
    outs = outs(beta_slope=beta_slope, beta_yinter=beta_yinter, sim_beta=sim_beta,
                mapped_time=mapped_time,premaptime=premaptime, premapdmdt=premapdmdt)
    return outs


def _tde_mosfit_engine(times, mbh6, mstar, b, efficiency, leddlimit, **kwargs):
    """
    Produces the processed outputs from simulation data for the TDE model.

    :param times: A dense array of times in rest frame in days
    :param mbh6: black hole mass in units of 10^6 solar masses
    :param mstar: star mass in units of solar masses
    :param b: Relates to beta and gamma values for the star that's disrupted
    :param efficiency: efficiency of the BH
    :param leddlimit: eddington limit for the BH
    :param kwargs: Additional keyword arguments
    :return: Named tuple with several outputs
    """
    beta_interp = True

    outs = _initialize_mosfit_tde_model()
    beta_slope = outs.beta_slope
    beta_yinter = outs.beta_yinter
    sim_beta = outs.sim_beta
    mapped_time = outs.mapped_time
    premaptime = outs.premaptime
    premapdmdt = outs.premapdmdt

    Mhbase = 1.0e6  # in units of Msolar, this is generic Mh used in astrocrash sims
    Mstarbase = 1.0  # in units of Msolar
    Rstarbase = 1.0  # in units of Rsolar
    starmass = mstar

    # Calculate beta values
    if 0 <= b < 1:
        beta43 = 0.6 + 1.25 * b
        beta53 = 0.5 + 0.4 * b
        betas = {'4-3': beta43, '5-3': beta53}
    elif 1 <= b <= 2:
        beta43 = 1.85 + 2.15 * (b - 1)
        beta53 = 0.9 + 1.6 * (b - 1)
        betas = {'4-3': beta43, '5-3': beta53}
    else:
        raise ValueError('b outside range, bmin = 0; bmax = 2')

    # Determine gamma values
    gamma_interp = False
    if starmass <= 0.3 or starmass >= 22:
        gammas = ['5-3']
        beta = betas['5-3']
    elif 1 <= starmass <= 15:
        gammas = ['4-3']
        beta = betas['4-3']
    elif 0.3 < starmass < 1:
        gamma_interp = True
        gammas = ['4-3', '5-3']
        gfrac = (starmass - 1.) / (0.3 - 1.)
        beta = betas['5-3'] + (betas['4-3'] - betas['5-3']) * (1. - gfrac)
    elif 15 < starmass < 22:
        gamma_interp = True
        gammas = ['4-3', '5-3']
        gfrac = (starmass - 15.) / (22. - 15.)
        beta = betas['5-3'] + (betas['4-3'] - betas['5-3']) * (1. - gfrac)

    timedict = {}
    dmdtdict = {}

    sim_beta = outs.sim_beta
    for g in gammas:
        for i in range(len(sim_beta[g])):
            if betas[g] == sim_beta[g][i]:
                beta_interp = False
                interp_index_low = i
                break
            if betas[g] < sim_beta[g][i]:
                interp_index_high = i
                interp_index_low = i - 1
                beta_interp = True
                break

        if beta_interp:
            dmdt = np.array([
                beta_yinter[g][interp_index_low][0] + beta_slope[g][interp_index_low][0] * betas[g],
                beta_yinter[g][interp_index_low][1] + beta_slope[g][interp_index_low][1] * betas[g]
            ], dtype=object)

            time = []
            for i in [0, 1]:
                time_betalo = (mapped_time[g][interp_index_low][i] * (
                            premaptime[g][interp_index_low][i][-1] - premaptime[g][interp_index_low][i][0]) +
                               premaptime[g][interp_index_low][i][0])
                time_betahi = (mapped_time[g][interp_index_low][i] * (
                            premaptime[g][interp_index_high][i][-1] - premaptime[g][interp_index_high][i][0]) +
                               premaptime[g][interp_index_high][i][0])
                time.append(time_betalo + (time_betahi - time_betalo) * (betas[g] - sim_beta[g][interp_index_low]) / (
                            sim_beta[g][interp_index_high] - sim_beta[g][interp_index_low]))
            time = np.array(time, dtype=object)

            timedict[g] = time
            dmdtdict[g] = dmdt
        else:
            timedict[g] = np.copy(premaptime[g][interp_index_low])
            dmdtdict[g] = np.copy(premapdmdt[g][interp_index_low])

    if gamma_interp:
        mapped_time = {'4-3': [], '5-3': []}
        time = []
        dmdt = []
        for j in [0, 1]:
            if len(timedict['4-3'][j]) < len(timedict['5-3'][j]):
                interp = '4-3'
                nointerp = '5-3'
            else:
                interp = '5-3'
                nointerp = '4-3'

            mapped_time[nointerp].append(1. / (timedict[nointerp][j][-1] - timedict[nointerp][j][0]) * (
                        timedict[nointerp][j] - timedict[nointerp][j][0]))
            mapped_time[interp].append(1. / (timedict[interp][j][-1] - timedict[interp][j][0]) * (
                        timedict[interp][j] - timedict[interp][j][0]))
            mapped_time[interp][j][0] = 0
            mapped_time[interp][j][-1] = 1
            mapped_time[nointerp][j][0] = 0
            mapped_time[nointerp][j][-1] = 1

            func = interp1d(mapped_time[interp][j], dmdtdict[interp][j])
            dmdtdict[interp][j] = func(mapped_time[nointerp][j])

            if interp == '5-3':
                time53 = (mapped_time['4-3'][j] * (timedict['5-3'][j][-1] - timedict['5-3'][j][0]) + timedict['5-3'][j][
                    0])
                time.extend(10 ** (timedict['4-3'][j] + (time53 - timedict['4-3'][j]) * gfrac))
            else:
                time43 = (mapped_time['5-3'][j] * (timedict['4-3'][j][-1] - timedict['4-3'][j][0]) + timedict['4-3'][j][
                    0])
                time.extend(10 ** (time43 + (timedict['5-3'][j] - time43) * gfrac))

            dmdt.extend(10 ** (dmdtdict['4-3'][j] + (dmdtdict['5-3'][j] - dmdtdict['4-3'][j]) * gfrac))
    else:
        time = np.concatenate((timedict[g][0], timedict[g][1]))
        time = 10 ** time
        dmdt = np.concatenate((dmdtdict[g][0], dmdtdict[g][1]))
        dmdt = 10 ** dmdt

    time = np.array(time)
    dmdt = np.array(dmdt)

    Mh = mbh6 * 1.0e6

    if starmass < 0.1:
        Mstar_Tout = 0.1
    else:
        Mstar_Tout = starmass

    Z = 0.0134
    log10_Z_02 = np.log10(Z / 0.02)

    Tout_theta = (
                1.71535900 + 0.62246212 * log10_Z_02 - 0.92557761 * log10_Z_02 ** 2 - 1.16996966 * log10_Z_02 ** 3 - 0.30631491 * log10_Z_02 ** 4)
    Tout_l = (
                6.59778800 - 0.42450044 * log10_Z_02 - 12.13339427 * log10_Z_02 ** 2 - 10.73509484 * log10_Z_02 ** 3 - 2.51487077 * log10_Z_02 ** 4)
    Tout_kpa = (
                10.08855000 - 7.11727086 * log10_Z_02 - 31.67119479 * log10_Z_02 ** 2 - 24.24848322 * log10_Z_02 ** 3 - 5.33608972 * log10_Z_02 ** 4)
    Tout_lbda = (
                1.01249500 + 0.32699690 * log10_Z_02 - 0.00923418 * log10_Z_02 ** 2 - 0.03876858 * log10_Z_02 ** 3 - 0.00412750 * log10_Z_02 ** 4)
    Tout_mu = (
                0.07490166 + 0.02410413 * log10_Z_02 + 0.07233664 * log10_Z_02 ** 2 + 0.03040467 * log10_Z_02 ** 3 + 0.00197741 * log10_Z_02 ** 4)
    Tout_nu = 0.01077422
    Tout_eps = (
                3.08223400 + 0.94472050 * log10_Z_02 - 2.15200882 * log10_Z_02 ** 2 - 2.49219496 * log10_Z_02 ** 3 - 0.63848738 * log10_Z_02 ** 4)
    Tout_o = (
                17.84778000 - 7.45345690 * log10_Z_02 - 48.9606685 * log10_Z_02 ** 2 - 40.05386135 * log10_Z_02 ** 3 - 9.09331816 * log10_Z_02 ** 4)
    Tout_pi = (
                0.00022582 - 0.00186899 * log10_Z_02 + 0.00388783 * log10_Z_02 ** 2 + 0.00142402 * log10_Z_02 ** 3 - 0.00007671 * log10_Z_02 ** 4)

    Rstar = ((
                         Tout_theta * Mstar_Tout ** 2.5 + Tout_l * Mstar_Tout ** 6.5 + Tout_kpa * Mstar_Tout ** 11 + Tout_lbda * Mstar_Tout ** 19 + Tout_mu * Mstar_Tout ** 19.5) / (
                         Tout_nu + Tout_eps * Mstar_Tout ** 2 + Tout_o * Mstar_Tout ** 8.5 + Mstar_Tout ** 18.5 + Tout_pi * Mstar_Tout ** 19.5))

    dmdt = (dmdt * np.sqrt(Mhbase / Mh) * (starmass / Mstarbase) ** 2.0 * (Rstarbase / Rstar) ** 1.5)
    time = (time * np.sqrt(Mh / Mhbase) * (Mstarbase / starmass) * (Rstar / Rstarbase) ** 1.5)

    DAY_CGS = 86400
    time = time / DAY_CGS
    tfallback = np.copy(time[0])

    time = time - tfallback
    tpeak = time[np.argmax(dmdt)]

    timeinterpfunc = interp1d(time, dmdt)
    lengthpretimes = len(np.where(times < time[0])[0])
    lengthposttimes = len(np.where(times > time[-1])[0])

    dmdt2 = timeinterpfunc(times[lengthpretimes:(len(times) - lengthposttimes)])
    dmdt1 = np.zeros(lengthpretimes)
    dmdt3 = np.zeros(lengthposttimes)

    dmdtnew = np.append(dmdt1, dmdt2)
    dmdtnew = np.append(dmdtnew, dmdt3)
    dmdtnew[dmdtnew < 0] = 0

    kappa_t = 0.2 * (1 + 0.74)
    Ledd = (4 * np.pi * cc.graviational_constant * Mh * cc.solar_mass * cc.speed_of_light / kappa_t)

    luminosities = (efficiency * dmdtnew * cc.speed_of_light * cc.speed_of_light)
    luminosities = (luminosities * leddlimit * Ledd / (luminosities + leddlimit * Ledd))
    luminosities = [0.0 if np.isnan(x) else x for x in luminosities]

    ProcessedData = namedtuple('ProcessedData', [
        'luminosities', 'Rstar', 'tpeak', 'beta', 'starmass', 'dmdt', 'Ledd', 'tfallback'])
    ProcessedData = ProcessedData(luminosities=luminosities, Rstar=Rstar, tpeak=tpeak, beta=beta, starmass=starmass,
                                  dmdt=dmdtnew, Ledd=Ledd, tfallback=float(tfallback))
    return ProcessedData

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract, https://ui.adsabs.harvard.edu/abs/2013ApJ...767...25G/abstract, https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def _tde_fallback_all_outputs(time, mbh6, mstar, tvisc, bb, eta, leddlimit, **kwargs):
    """
    Identical to the Mosfit model following Guillochon+ 2013 apart from doing the fallback rate fudge in mosfit.

    :param time: A dense array of times in rest frame in days
    :param mbh6: black hole mass in units of 10^6 solar masses
    :param mstar: star mass in units of solar masses
    :param tvisc: viscous timescale in days
    :param bb: Relates to beta and gamma values for the star that's disrupted
    :param eta: efficiency of the BH
    :param leddlimit: eddington limit for the BH
    :param kwargs: Additional keyword arguments
    :return: bolometric luminosity
    """
    _interaction_process = kwargs.get("interaction_process", ip.Viscous)
    dense_resolution = kwargs.get("dense_resolution", 1000)
    dense_times = np.linspace(0, time[-1] + 100, dense_resolution)
    outs = _tde_mosfit_engine(times=dense_times, mbh6=mbh6, mstar=mstar, b=bb, efficiency=eta,
                              leddlimit=leddlimit, **kwargs)
    dense_lbols = outs.luminosities
    interaction_class = _interaction_process(time=time, dense_times=dense_times, luminosity=dense_lbols, t_viscous=tvisc, **kwargs)
    lbol = interaction_class.new_luminosity
    return lbol, outs

def tde_fallback_bolometric(time, mbh6, mstar, tvisc, bb, eta, leddlimit, **kwargs):
    """
    Identical to the Mosfit model following Guillochon+ 2013 apart from doing the fallback rate fudge in mosfit.

    :param time: A dense array of times in rest frame in days
    :param mbh6: black hole mass in units of 10^6 solar masses
    :param mstar: star mass in units of solar masses
    :param tvisc: viscous timescale in days
    :param bb: Relates to beta and gamma values for the star that's disrupted
    :param eta: efficiency of the BH
    :param leddlimit: eddington limit for the BH
    :param kwargs: Additional keyword arguments
    :return: bolometric luminosity
    """
    lbol, _ = _tde_fallback_all_outputs(time=time, mbh6=mbh6, mstar=mstar, tvisc=tvisc, bb=bb, eta=eta,
                                       leddlimit=leddlimit, **kwargs)
    return lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract, https://ui.adsabs.harvard.edu/abs/2013ApJ...767...25G/abstract, https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def tde_fallback(time, redshift, mbh6, mstar, tvisc, bb, eta, leddlimit, rph0, lphoto, **kwargs):
    """
    Identical to the Mosfit model following Guillochon+ 2013 apart from doing the fallback rate fudge in mosfit.

    :param time: Times in observer frame in days
    :param redshift: redshift of the transient
    :param mbh6: black hole mass in units of 10^6 solar masses
    :param mstar: star mass in units of solar masses
    :param tvisc: viscous timescale in days
    :param bb: Relates to beta and gamma values for the star that's disrupted
    :param eta: efficiency of the BH
    :param leddlimit: eddington limit for the BH
    :param rph0: initial photosphere radius
    :param lphoto: photosphere luminosity
    :param kwargs: Additional keyword arguments
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """

    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Viscous)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TDEPhotosphere)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol, outs = _tde_fallback_all_outputs(time=time, mbh6=mbh6, mstar=mstar, tvisc=tvisc, bb=bb, eta=eta,
                                               leddlimit=leddlimit, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, mass_bh=mbh6*1e6,
                                      mass_star=mstar, star_radius=outs.Rstar,
                                      tpeak=outs.tpeak, rph_0=rph0, lphoto=lphoto, beta=outs.beta, **kwargs)
        sed_1 = kwargs['sed'](time=time, temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                     frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        flux_density = np.nan_to_num(flux_density)
        return flux_density.to(uu.mJy).value
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 1000, 300)
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        lbol, outs = _tde_fallback_all_outputs(time=time, mbh6=mbh6, mstar=mstar, tvisc=tvisc, bb=bb, eta=eta,
                                               leddlimit=leddlimit, **kwargs)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, mass_bh=mbh6*1e6, mass_star=mstar,
                                      star_radius=outs.Rstar, tpeak=outs.tpeak, rph_0=rph0, lphoto=lphoto,
                                      beta=outs.beta,**kwargs)
        sed_1 = kwargs['sed'](time=time, temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
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
                                                              
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024arXiv240815048M/abstract')   
def fitted(time, redshift, log_mh, a_bh, m_disc, r0, tvi, t_form, incl, **kwargs):
    """
    An import of FitTeD to model the plateau phase
    
    :param time: observer frame time in days
    :param redshift: redshift
    :param log_mh: log of the black hole mass (solar masses)
    :param a_bh: black hole spin parameter (dimensionless)
    :param m_disc: initial mass of disc ring (solar masses)
    :param r0: initial radius of disc ring (gravitational radii)
    :param tvi: viscous timescale of disc evolution (days)
    :param t_form: time of ring formation prior to t = 0 (days)
    :param incl: disc-observer inclination angle (radians)    
    :param kwargs: Must be all the kwargs required by the specific output_format 
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'  
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    import fitted #user needs to have downloaded and compiled FitTeD in order to run this model
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    ang = 180.0/np.pi*incl
    m = fitted.models.GR_disc()

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        freqs_un = np.unique(frequency)
        nulnus = np.zeros(len(time))
        if len(freqs_un) == 1:
            nulnus = m.model_UV(time, log_mh, a_bh, m_disc, r0, tvi, t_form, ang, frequency)
        else:
            for i in range(0,len(freqs_un)):
                inds = np.where(frequency == freqs_un[i])[0]
                nulnus[inds] = m.model_UV([time[j] for j in inds], log_mh, a_bh, m_disc, r0, tvi, t_form, ang, freqs_un[i])
        flux_density = nulnus/(4.0 * np.pi * dl**2 * frequency)   
        return flux_density/1.0e-26   

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        nulnus = m.model_SEDs(time, log_mh, a_bh, m_disc, r0, tvi, t_form, ang, frequency)
        flux_density = (nulnus/(4.0 * np.pi * dl**2 * frequency[:,np.newaxis] * 1.0e-26)) 
        fmjy = flux_density.T           
        spectra = (fmjy * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))  
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)
                                                              
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024arXiv240815048M/abstract')   
def fitted_pl_decay(time, redshift, log_mh, a_bh, m_disc, r0, tvi, t_form, incl, log_L, t_decay, p, log_T, sigma, t_peak, **kwargs):
    """
    An import of FitTeD to model the plateau phase, with a gaussian rise and power-law decay
    
    :param time: observer frame time in days
    :param redshift: redshift
    :param log_mh: log of the black hole mass (solar masses)
    :param a_bh: black hole spin parameter (dimensionless)
    :param m_disc: initial mass of disc ring (solar masses)
    :param r0: initial radius of disc ring (gravitational radii)
    :param tvi: viscous timescale of disc evolution (days)
    :param t_form: time of ring formation prior to t = 0 (days)
    :param incl: disc-observer inclination angle (radians)    
    :param log_L: single temperature blackbody amplitude for decay model (log_10 erg/s)
    :param t_decay: fallback timescale (days)
    :param p: power-law decay index
    :param log_T: single temperature blackbody temperature for decay model (log_10 Kelvin)
    :param sigma: gaussian rise timescale (days)
    :param t_peak: time of light curve peak (days)
    :param kwargs: Must be all the kwargs required by the specific output_format 
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'  
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    import fitted #user needs to have downloaded and compiled FitTeD in order to run this model
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    ang = 180.0/np.pi*incl
    m = fitted.models.GR_disc(decay_type='pl', rise=True)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        freqs_un = np.unique(frequency)
        
        #initialize arrays
        nulnus_plateau = np.zeros(len(time))
        nulnus_rise = np.zeros(len(time))
        nulnus_decay = np.zeros(len(time))
        
        if len(freqs_un) == 1:
            nulnus_plateau = m.model_UV(time, log_mh, a_bh, m_disc, r0, tvi, t_form, ang, v=freqs_un[0])
            nulnus_decay = m.decay_model(time, log_L, t_decay, p, t_peak, log_T, v=freqs_un[0])
            nulnus_rise = m.rise_model(time, log_L, sigma, t_peak, log_T, v=freqs_un[0])
        else:
            for i in range(0,len(freqs_un)):
                inds = np.where(frequency == freqs_un[i])[0]
                nulnus[inds] = m.model_UV([time[j] for j in inds], log_mh, a_bh, m_disc, r0, tvi, t_form, ang, freqs_un[i])
                nulnus_decay[inds] = m.decay_model([time[j] for j in inds], log_L, t_decay, p, t_peak, log_T, v=freqs_un[i]) 
                nulnus_rise[inds] = m.rise_model([time[j] for j in inds], log_L, sigma, t_peak, log_T, v=freqs_un[i])                                             
        nulnus = nulnus_plateau + nulnus_rise + nulnus_decay    
        flux_density = nulnus/(4.0 * np.pi * dl**2 * frequency)   
        return flux_density/1.0e-26   

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        nulnus_plateau = m.model_SEDs(time, log_mh, a_bh, m_disc, r0, tvi, t_form, ang, frequency)

        freq_0 = 6e14
        l_e_amp = (model.decay_model(time, log_L, t_decay, t_peak, log_T, freq_0) + model.rise_model(time, log_L, sigma, t_peak, log_T, freq_0))
        nulnus_risedecay = ((l_e_amp[:, None] * (frequency/freq_0)**4 * 
                        (np.exp(cc.planck * freq_0/(cc.boltzmann_constant * 10**log_T)) - 1)/(np.exp(cc.planck * frequency/(cc.boltzmann_constant * 10**log_T)) - 1)).T)  
        flux_density = ((nulnus_risedecay + nulnus_plateau)/(4.0 * np.pi * dl**2 * frequency[:,np.newaxis] * 1.0e-26))  
        fmjy = flux_density.T           
        spectra = (fmjy * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))  
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)  
                                                              
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024arXiv240815048M/abstract')   
def fitted_exp_decay(time, redshift, log_mh, a_bh, m_disc, r0, tvi, t_form, incl, log_L, t_decay, log_T, sigma, t_peak, **kwargs):
    """
    An import of FitTeD to model the plateau phase, with a gaussian rise and exponential decay
    
    :param time: observer frame time in days
    :param redshift: redshift
    :param log_mh: log of the black hole mass (solar masses)
    :param a_bh: black hole spin parameter (dimensionless)
    :param m_disc: initial mass of disc ring (solar masses)
    :param r0: initial radius of disc ring (gravitational radii)
    :param tvi: viscous timescale of disc evolution (days)
    :param t_form: time of ring formation prior to t = 0 (days)
    :param incl: disc-observer inclination angle (radians)    
    :param log_L: single temperature blackbody amplitude for decay model (log_10 erg/s)
    :param t_decay: fallback timescale (days)
    :param log_T: single temperature blackbody temperature for decay model (log_10 Kelvin)
    :param sigma: gaussian rise timescale (days)
    :param t_peak: time of light curve peak (days)
    :param kwargs: Must be all the kwargs required by the specific output_format 
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'  
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    import fitted #user needs to have downloaded and compiled FitTeD in order to run this model
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    ang = 180.0/np.pi*incl
    m = fitted.models.GR_disc(decay_type='exp', rise=True)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        freqs_un = np.unique(frequency)
        
        #initialize arrays
        nulnus_plateau = np.zeros(len(time))
        nulnus_rise = np.zeros(len(time))
        nulnus_decay = np.zeros(len(time))
        
        if len(freqs_un) == 1:
            nulnus_plateau = m.model_UV(time, log_mh, a_bh, m_disc, r0, tvi, t_form, ang, v=freqs_un[0])
            nulnus_decay = m.decay_model(time, log_L, t_decay, t_peak, log_T, v=freqs_un[0])
            nulnus_rise = m.rise_model(time, log_L, sigma, t_peak, log_T, v=freqs_un[0])
        else:
            for i in range(0,len(freqs_un)):
                inds = np.where(frequency == freqs_un[i])[0]
                nulnus[inds] = m.model_UV([time[j] for j in inds], log_mh, a_bh, m_disc, r0, tvi, t_form, ang, freqs_un[i])
                nulnus_decay[inds] = m.decay_model([time[j] for j in inds], log_L, t_decay, t_peak, log_T, v=freqs_un[i]) 
                nulnus_rise[inds] = m.rise_model([time[j] for j in inds], log_L, sigma, t_peak, log_T, v=freqs_un[i]) 
        nulnus = nulnus_plateau + nulnus_rise + nulnus_decay        
        flux_density = nulnus/(4.0 * np.pi * dl**2 * frequency)   
        return flux_density/1.0e-26   

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300) # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        nulnus_plateau = m.model_SEDs(time, log_mh, a_bh, m_disc, r0, tvi, t_form, ang, frequency)

        freq_0 = 6e14
        l_e_amp = (m.decay_model(time, log_L, t_decay, t_peak, log_T, freq_0) + m.rise_model(time, log_L, sigma, t_peak, log_T, freq_0))
        nulnus_risedecay = ((l_e_amp[:, None] * (frequency/freq_0)**4 * 
                        (np.exp(cc.planck * freq_0/(cc.boltzmann_constant * 10**log_T)) - 1)/(np.exp(cc.planck * frequency/(cc.boltzmann_constant * 10**log_T)) - 1)).T) 
        flux_density = ((nulnus_risedecay + nulnus_plateau)/(4.0 * np.pi * dl**2 * frequency[:,np.newaxis] * 1.0e-26))  
        fmjy = flux_density.T           
        spectra = (fmjy * uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))  
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)               

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2015ApJ...806..164P/abstract, https://ui.adsabs.harvard.edu/abs/2020ApJ...904...73R/abstract')
def _stream_stream_collision(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega):
    """
    A TDE model based on stream-stream collisions.  Used as input for the bolometric and broadband versions.
    
    :param mbh_6: black hole mass (10^6 solar masses)
    :param mstar: mass of the disrupted star (solar masses)
    :param c1: characteristic distance scale of the emission region in units of the apocenter distance of the most tightly bound debris
    :param f: fraction of the bound mass within the semimajor axis of the most tightly bound debris at peak mass return time
    :param h_r: aspect ratio used to calculate t_cool. This is only used when include_tcool_tdyn_ratio = 1
    :param inc_tcool: if include_tcool_tdyn_ratio = 1, the luminosity is limited by the Eddington luminosity if t_cool / t_dyn < 1.0
    :param del_omega: solid angle (in units of pi) of radiation from the emission region
    :return: physical outputs
    """
    kappa = 0.34
    t_ratio = 0.0
    factor = 1.0
    tcool = 0.0

    rstar = 0.93 * mstar ** (8.0 / 9.0)
    mstar_max = 15.0
    Xi = (1.27 - 0.3 *(mbh_6)**0.242 )*((0.620 + np.exp((min(mstar_max,mstar) - 0.674)/0.212)) 
            / (1.0 + 0.553 *np.exp((min(mstar,mstar_max) - 0.674)/0.212)))
    r_tidal = (mbh_6 * 1e6/ mstar)**(1.0/3.0) * rstar * cc.solar_radius

    epsilon = cc.graviational_constant * (mbh_6 * 1e6 * cc.solar_mass) * (rstar * cc.solar_radius) / r_tidal ** 2.0
    a0 = cc.graviational_constant * (mbh_6 * 1e6 * cc.solar_mass)/ (Xi * epsilon)

    t_dyn = np.pi / np.sqrt(2.0) * a0 ** 1.5 / np.sqrt(cc.graviational_constant * (mbh_6 * 1e6 * cc.solar_mass))
    t_peak = (3.0/2.0)*t_dyn
    mdotmax = mstar * cc.solar_mass / t_dyn / 3.0
    factor_denom = del_omega * cc.sigma_sb * c1**2 * a0**2

    if inc_tcool == 1:
        semi = a0 / 2.0     #semimajor axis of the most bound debris
        area = np.pi * ( c1 * semi ) **2 # emitting area
        tau = kappa * (f * mstar * cc.solar_mass / 2.0) / area / 2.0  # the characteristic vertical optical depth to the midplane of a circular disk with radius ∼ semi. The first "/2.0" comes from the fact that we consider only the bound mass = mstar / 2. The second "/2.0" comes from the fact that the optical depth was integrated to the mid-plane.
        tcool = tau * (h_r) * c1 * semi / cc.speed_of_light
        t_ratio = tcool / t_dyn
        factor = 2.0 / (1.0 + t_ratio)
        factor_denom *= (1.0 + 2.0 * h_r) / 4.0
    
    t_output = np.linspace(t_peak, 1500*cc.day_to_s, 1000)    
    Lmax = mdotmax * (Xi * epsilon) / c1    
    Lobs = Lmax * (t_output / t_peak)**(-5.0/3.0) * factor
    Tobs = (Lobs / factor_denom )**(1.0/4.0)
    
    output = namedtuple('output', ['bolometric_luminosity', 'photosphere_temperature',
                                   'Smbh_6_accretion_rate_max', 'time_temp', 'cooling_time',
                                   'dynamical_time', 'r_tidal','debris_energy'])
    output.bolometric_luminosity = Lobs
    output.photosphere_temperature = Tobs
    output.Smbh_6_accretion_rate_max = mdotmax
    output.time_temp = t_output
    output.cooling_time = tcool
    output.dynamical_time = t_dyn
    output.r_tidal = r_tidal
    output.debris_energy = Xi * epsilon
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2015ApJ...806..164P/abstract, https://ui.adsabs.harvard.edu/abs/2020ApJ...904...73R/abstract')    
def stream_stream_tde_bolometric(time, mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time, **kwargs): 
    """
    A bolometric TDE model based on stream-stream collisions.  The early emission follows a gaussian rise.
    
    :param time: observer frame time in days
    :param mbh_6: black hole mass (10^6 solar masses)
    :param mstar: mass of the disrupted star (solar masses)
    :param c1: characteristic distance scale of the emission region in units of the apocenter distance of the most tightly bound debris
    :param f: fraction of the bound mass within the semimajor axis of the most tightly bound debris at peak mass return time
    :param h_r: aspect ratio used to calculate t_cool. This is only used when include_tcool_tdyn_ratio = 1
    :param inc_tcool: if include_tcool_tdyn_ratio = 1, the luminosity is limited by the Eddington luminosity if t_cool / t_dyn < 1.0
    :param del_omega: solid angle (in units of pi) of radiation from the emission region
    :param peak_time: peak time in days
    :param sigma_t: the sharpness of the Gaussian in days
    :return: bolometric luminosity         
    """
    output = _stream_stream_collision(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega)    
    f1 = pm.gaussian_rise(time=output.time_temp[0] / cc.day_to_s, a_1=1, peak_time=peak_time, sigma_t=sigma_t)
    norm = output.bolometric_luminosity[0] / f1

    #evaluate giant array of bolometric luminosities
    tt_pre_fb = np.linspace(0, output.time_temp[0]-0.001, 100)
    tt_post_fb = output.time_temp
    full_time = np.concatenate([tt_pre_fb, tt_post_fb])
    f1 = norm * np.exp(-(tt_pre_fb - (peak_time * cc.day_to_s))**2.0 / (2 * (sigma_t * cc.day_to_s) **2.0))
    f2 = output.bolometric_luminosity
    full_lbol = np.concatenate([f1, f2])
    lbol_func = interp1d(full_time, y=full_lbol, fill_value='extrapolate')
    return lbol_func(time*cc.day_to_s)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2015ApJ...806..164P/abstract, https://ui.adsabs.harvard.edu/abs/2020ApJ...904...73R/abstract')
def stream_stream_tde(time, redshift, mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time, **kwargs):
    """
    A TDE model based on stream-stream collisions.  The early emission follows a constant temperature gaussian rise.
    
    :param time: observer frame time in days
    :param redshift: redshift
    :param mbh_6: black hole mass (10^6 solar masses)
    :param mstar: mass of the disrupted star (solar masses)
    :param c1: characteristic distance scale of the emission region in units of the apocenter distance of the most tightly bound debris
    :param f: fraction of the bound mass within the semimajor axis of the most tightly bound debris at peak mass return time
    :param h_r: aspect ratio used to calculate t_cool. This is only used when include_tcool_tdyn_ratio = 1
    :param inc_tcool: if include_tcool_tdyn_ratio = 1, the luminosity is limited by the Eddington luminosity if t_cool / t_dyn < 1.0
    :param del_omega: solid angle (in units of pi) of radiation from the emission region
    :param peak_time: peak time in days
    :param sigma_t: the sharpness of the Gaussian in days
    :param kwargs: Must be all the kwargs required by the specific output_format 
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'  
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density' or 'magnitude'     
    """

    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    output = _stream_stream_collision(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega)

    #get bolometric and temperature info
    f1 = pm.gaussian_rise(time=output.time_temp[0] / cc.day_to_s, a_1=1, peak_time=peak_time, sigma_t=sigma_t)
    norm = output.bolometric_luminosity[0] / f1    
    tt_pre_fb = np.linspace(0, output.time_temp[0]-0.001, 100)
    tt_post_fb = output.time_temp
    full_time = np.concatenate([tt_pre_fb, tt_post_fb])
    f1_src = pm.gaussian_rise(time=tt_pre_fb, a_1=norm,
                          peak_time=peak_time * cc.day_to_s, sigma_t=sigma_t * cc.day_to_s)
    f2_src = output.bolometric_luminosity
    full_lbol = np.concatenate([f1_src, f2_src])
    
    temp1 = np.ones(100) * output.photosphere_temperature[0]
    temp2 = output.photosphere_temperature
    full_temp = np.concatenate([temp1, temp2])
    r_eff = np.sqrt(full_lbol / (np.pi * cc.sigma_sb * full_temp**4.0))            
        
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        if isinstance(frequency, float):
            frequency = np.ones(len(time)) * frequency           
    
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        unique_frequency = np.sort(np.unique(frequency))

        # build flux density function for each frequency
        flux_den_interp_func = {}
        total_time = full_time * (1 + redshift)
        for freq in unique_frequency:           
            flux_den = sed.blackbody_to_flux_density(temperature=full_temp,
                                           r_photosphere=r_eff,
                                           dl=dl, frequency=freq).to(uu.mJy)
            flux_den_interp_func[freq] = interp1d(total_time, flux_den, fill_value='extrapolate')

        # interpolate onto actual observed frequency and time values
        flux_density = []
        for freq, tt in zip(frequency, time):
            flux_density.append(flux_den_interp_func[freq](tt * cc.day_to_s))
        flux_density = flux_density * uu.mJy
        return flux_density.to(uu.mJy).value    
        
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_observer_frame = full_time * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        freq_0 = 6e14
        flux_den = sed.blackbody_to_flux_density(temperature=full_temp,
                            r_photosphere=r_eff,
                            dl=dl, frequency=freq_0).to(uu.mJy)           
        fmjy = ((flux_den[:, None] * (frequency/freq_0)**4 * 
                        (np.exp(cc.planck * freq_0/(cc.boltzmann_constant * full_temp[:, None])) - 1) / (np.exp(cc.planck * frequency/(cc.boltzmann_constant * full_temp[:, None])) - 1)).T)                
        spectra = fmjy.T.to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom)) 
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/cc.day_to_s,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs) 