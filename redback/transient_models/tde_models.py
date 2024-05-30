import numpy as np
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from redback.utils import calc_kcorrected_properties, citation_wrapper, calc_tfb, lambda_to_nu, \
    calc_ABmag_from_flux_density, calc_flux_density_from_ABmag
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

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract')
def tde_semianalytical():
    raise NotImplementedError("This model is not yet implemented.")