import numpy as np
from collections import namedtuple
from scipy import special
from scipy.interpolate import interp1d
from redback.constants import *
import redback.sed as sed
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, lambda_to_nu

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract')
def _csm_shock_breakout(time, csm_mass, v_min, beta, kappa, shell_radius, shell_width_ratio, **kwargs):
    """
    Dense CSM shock breakout and cooling model From Margalit 2022

    :param time: time in days
    :param csm_mass: mass of CSM shell in g
    :param v_min: minimum velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param kappa: opacity in cm^2/g
    :param shell_radius: radius of shell in 10^14 cm
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :return: namedtuple with lbol, r_photosphere, and temperature
    """
    v0 = v_min * 1e5
    e0 = 0.5 * csm_mass * v0**2
    shell_radius *= 1e14
    shell_width = shell_width_ratio * shell_radius
    tdyn = shell_radius / v0
    tshell = shell_width / v0
    time = time * day_to_s

    velocity = v0/beta

    tda = (3 * kappa * csm_mass / (4 * np.pi * speed_of_light * velocity)) ** 0.5

    term1 = ((tdyn + tshell + time) ** 3 - (tdyn + beta * time) ** 3) ** (2 / 3)
    term2 = ((tdyn + tshell) ** 3 - tdyn ** 3) ** (1 / 3)
    term3 = (1 + (1 - beta) * time / tshell) ** (
                -3 * (tdyn / tda) ** 2 * ((1 - beta - beta * tshell / tdyn) ** 2) / (1 - beta) ** 3)
    term4 = np.exp(-time * ((1 - beta ** 3) * time + (2 - 4 * beta * (beta + 1)) * tshell + 6 * (1 - beta ** 2) * tdyn) / (
                2 * (1 - beta) ** 2 * tda ** 2))

    lbol = e0 * term1 / (tda ** 2 * (tshell + (1 - beta) * time) ** 2) * term2 * term3 * term4

    volume = 4./3. * np.pi * velocity**3 * ((tdyn + tshell + time)**3 - (tdyn + beta*time)**3)
    radius = velocity * (tdyn + tshell + time)
    rphotosphere = radius - 2*volume/(3 * kappa * csm_mass)
    teff = (lbol / (4 * np.pi * rphotosphere ** 2 * sigma_sb)) ** 0.25
    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature', 'time_temp',
                                   'tdyn', 'tshell', 'e0', 'tda', 'velocity'])
    output.lbol = lbol
    output.r_photosphere = rphotosphere
    output.temperature = teff
    output.tdyn = tdyn
    output.tshell = tshell
    output.e0 = e0
    output.tda = tda
    output.velocity = velocity
    output.time_temp = time
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract')
def csm_shock_breakout_bolometric(time, csm_mass, v_min, beta, kappa, shell_radius, shell_width_ratio, **kwargs):
    """
    Dense CSM shock breakout and cooling model From Margalit 2022

    :param time: time in days in source frame
    :param csm_mass: mass of CSM shell in solar masses
    :param v_min: minimum velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param kappa: opacity in cm^2/g
    :param shell_radius: radius of shell in 10^14 cm
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: Additional parameters required by model
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: bolometric luminosity
    """
    csm_mass = csm_mass * solar_mass
    time_temp = np.geomspace(1e-2, 200, 300)  # days
    outputs = _csm_shock_breakout(time_temp, v_min=v_min, beta=beta,
                                  kappa=kappa, csm_mass=csm_mass, shell_radius=shell_radius,
                                  shell_width_ratio=shell_width_ratio, **kwargs)
    func = interp1d(time_temp, outputs.lbol, fill_value='extrapolate')
    return func(time)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract')
def csm_shock_breakout(time, redshift, csm_mass, v_min, beta, kappa, shell_radius, shell_width_ratio, **kwargs):
    """
    Dense CSM shock breakout and cooling model From Margalit 2022

    :param time: time in days in observer frame
    :param redshift: redshift
    :param csm_mass: mass of CSM shell in solar masses
    :param v_min: minimum velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param kappa: opacity in cm^2/g
    :param shell_radius: radius of shell in 10^14 cm
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: Additional parameters required by model
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :return:
    """
    csm_mass = csm_mass * solar_mass
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_temp = np.geomspace(1e-2, 40, 300) #days
    time_obs = time
    outputs = _csm_shock_breakout(time_temp, v_min=v_min, beta=beta,
                                  kappa=kappa, csm_mass=csm_mass, shell_radius=shell_radius,
                                  shell_width_ratio=shell_width_ratio, **kwargs)
    if kwargs['output_format'] == 'namedtuple':
        return outputs
    elif kwargs['output_format'] == 'flux_density':
        time = time_obs
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=outputs.temperature)
        rad_func = interp1d(time_temp, y=outputs.r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)

        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = sed.blackbody_to_flux_density(temperature=outputs.temperature,
                                         r_photosphere=outputs.r_photosphere, frequency=frequency[:, None], dl=dl)
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


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def _shock_cooling(time, mass, radius, energy, **kwargs):
    """
    :param time: time in source frame in seconds
    :param mass: mass of extended material in solar masses
    :param radius: radius of extended material in cm
    :param energy: energy of extended material in ergs
    :param kwargs: extra parameters to change physics
    :param nn: density power law slope
    :param delta: inner density power law slope
    :return: namedtuple with lbol, r_photosphere, and temperature
    """
    nn = kwargs.get('nn',10)
    delta = kwargs.get('delta',1.1)
    kk_pow = (nn - 3) * (3 - delta) / (4 * np.pi * (nn - delta))
    kappa = 0.2
    mass = mass * solar_mass
    vt = (((nn - 5) * (5 - delta) / ((nn - 3) * (3 - delta))) * (2 * energy / mass))**0.5
    td = ((3 * kappa * kk_pow * mass) / ((nn - 1) * vt * speed_of_light))**0.5

    prefactor = np.pi * (nn - 1) / (3 * (nn - 5)) * speed_of_light * radius * vt**2 / kappa
    lbol_pre_td = prefactor * np.power(td / time, 4 / (nn - 2))
    lbol_post_td = prefactor * np.exp(-0.5 * (time * time / td / td - 1))
    lbol = np.zeros(len(time))
    lbol[time < td] = lbol_pre_td[time < td]
    lbol[time >= td] = lbol_post_td[time >= td]

    tph = np.sqrt(3 * kappa * kk_pow * mass / (2 * (nn - 1) * vt * vt))
    r_photosphere_pre_td = np.power(tph / time, 2 / (nn - 1)) * vt * time
    r_photosphere_post_td = (np.power((delta - 1) / (nn - 1) * ((time / td) ** 2 - 1) + 1, -1 / (delta + 1)) * vt * time)
    r_photosphere = r_photosphere_pre_td + r_photosphere_post_td

    sigmaT4 = lbol / (4 * np.pi * r_photosphere**2)
    temperature = np.power(sigmaT4 / sigma_sb, 0.25)

    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature', 'td'])
    output.lbol = lbol
    output.r_photosphere = r_photosphere
    output.temperature = temperature
    output.td = td
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract')
def _shocked_cocoon_nicholl(time, kappa, mejecta, vejecta, cos_theta_cocoon, shocked_fraction, nn, tshock):
    """
    Shocked cocoon model from Nicholl et al. 2021

    :param time: time in source frame in days
    :param kappa: opacity
    :param mejecta: ejecta in solar masses
    :param vejecta: ejecta velocity in units of c (speed of light)
    :param cos_theta_cocoon: cosine of the cocoon opening angle
    :param shocked_fraction: fraction of the ejecta that is shocked
    :param nn: ejecta power law density profile
    :param tshock: time of shock in source frame in seconds
    :return: luminosity
    """
    ckm = 3e10 / 1e5
    vejecta = vejecta * ckm
    diffusion_constant = solar_mass / (4 * np.pi * speed_of_light * km_cgs)
    num = speed_of_light / km_cgs
    rshock = speed_of_light * tshock
    mshocked = shocked_fraction * mejecta
    theta = np.arccos(cos_theta_cocoon)
    taudiff = np.sqrt(diffusion_constant * kappa * mshocked / vejecta) / day_to_s

    tthin = (num / vejecta) ** 0.5 * taudiff

    l0 = (theta **2 / 2)**(1. / 3.) * (mshocked * solar_mass *
                                       vejecta * km_cgs * rshock / (taudiff * day_to_s)**2)

    lbol = l0 * (time / taudiff)**-(4/(nn+2)) * (1 + np.tanh(tthin-time))/2.
    output = namedtuple('output', ['lbol', 'tthin', 'taudiff','mshocked'])
    output.lbol = lbol
    output.tthin = tthin
    output.taudiff = taudiff
    output.mshocked = mshocked
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def shock_cooling_bolometric(time, log10_mass, log10_radius, log10_energy, **kwargs):
    """
    :param time: time in source frame in seconds
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param kwargs: extra parameters to change physics
    :param nn: density power law slope
    :param delta: inner density power law slope
    :return: bolometric_luminosity
    """
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy
    output = _shock_cooling(time, mass=mass, radius=radius, energy=energy, **kwargs)
    return output.lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def shock_cooling(time, redshift, log10_mass, log10_radius, log10_energy, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param kwargs: extra parameters to change physics and other settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param nn: density power law slope
    :param delta: inner density power law slope
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = _shock_cooling(time*day_to_s, mass=mass, radius=radius, energy=energy, **kwargs)
        flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                             dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        time_temp = np.linspace(1e-2, 10, 100)
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))

        time_observer_frame = time_temp
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = _shock_cooling(time=time * day_to_s, mass=mass, radius=radius, energy=energy, **kwargs)
        fmjy = sed.blackbody_to_flux_density(temperature=output.temperature,
                                             r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
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

def _c_j(p):
    """
    :param p: electron power law slope
    :return: prefactor for emissivity
    """
    term1 = (special.gamma((p+5.0)/4.0)/special.gamma((p+7.0)/4.0))
    term2 = special.gamma((3.0*p+19.0)/12.0)
    term3 = special.gamma((3.0*p-1.0)/12.0)*((p-2.0)/(p+1.0))
    term4 = 3.0**((2.0*p-1.0)/2.0)
    term5 = 2.0**(-(7.0-p)/2.0)*np.pi**(-0.5)
    return term1*term2*term3*term4*term5

def _c_alpha(p):
    """
    :param p: electron power law slope
    :return: prefactor for absorption coefficient
    """
    term1 = (special.gamma((p+6.0)/4.0)/special.gamma((p+8.0)/4.0))
    term2 = special.gamma((3.0*p+2.0)/12.0)
    term3 = special.gamma((3.0*p+22.0)/12.0)*(p-2.0)*3.0**((2.0*p-5.0)/2.0)
    term4 = 2.0**(p/2.0)*np.pi**(3.0/2.0)
    return term1*term2*term3*term4


def _g_theta(theta,p):
    """
    :param theta: dimensionless electron temperature
    :param p: electron power law slope
    :return: correction term for power law electron distribution
    """
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    gtheta = ((p-1.0)*(1e0+aa*theta)/((p-1.0)*gamma_m - p+2.0))*(gamma_m/(3.0*theta))**(p-1.0)
    return gtheta

def _low_freq_jpl_correction(x,theta,p):
    """
    :param x: dimensionless frequency
    :param theta: dimensionless electron temperature
    :param p: electron power law slope
    :return: low-frequency correction to power-law emissivity
    """
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    # synchrotron constant in x<<x_m limit
    Cj_low = -np.pi**1.5*(p-2.0)/( 2.0**(1.0/3.0)*3.0**(1.0/6.0)*(3.0*p-1.0)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Cj_low/_c_j(p))*(gamma_m/(3.0*theta))**(-(3.0*p-1.0)/3.0)*x**((3.0*p-1.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    val = (1e0 + corr**(-s))**(-1.0/s)
    return val

def _low_freq_apl_correction(x,theta,p):
    """
    :param x: dimensionless frequency
    :param theta: dimensionless electron temperature
    :param p: electron power law slope
    :return: low-frequency correction to power-law absorption coefficient
    """
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    # synchrotron constant in x<<x_m limit
    Calpha_low = -2.0**(8.0/3.0)*np.pi**(7.0/2.0)*(p+2.0)*(p-2.0)/( 3.0**(19.0/6.0)*(3.0*p+2)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Calpha_low/_c_alpha(p))*(gamma_m/(3.0*theta))**(-(3.0*p+2.0)/3.0)*x**((3.0*p+2.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    val = ( 1e0 + corr**(-s) )**(-1.0/s)
    return val

def _emissivity_pl(x, nism, bfield, theta, xi, p, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param z_cool: normalised cooling lorentz factor
    :return: synchrotron emissivity of power-law electrons
    """
    val = _c_j(p)*(qe**3/(electron_mass*speed_of_light**2))*xi*nism*bfield*_g_theta(theta=theta,p=p)*x**(-(p-1.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= _low_freq_jpl_correction(x=x,theta=theta,p=p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    emissivity_pl = val
    return emissivity_pl

def _emissivity_thermal(x, nism, bfield, theta, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param z_cool: normalised cooling lorentz factor
    :return: synchrotron emissivity of thermal electrons
    """
    ff = 2.0*theta**2/special.kn(2,1.0/theta)
    ix = 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))
    val = (3.0**0.5/(8.0*np.pi))*(qe**3/(electron_mass*speed_of_light**2))*ff*nism*bfield*x*ix
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum(1e0, (z0/z_cool)**(-1))
    return val

def _alphanu_th(x, nism, bfield, theta, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param z_cool: normalised cooling lorentz factor
    :return: Synchrotron absorption coeff of thermal electrons
    """
    ff = 2.0 * theta ** 2 / special.kn(2, 1.0 / theta)
    ix = 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))
    val = (np.pi*3.0**(-3.0/2.0))*qe*(nism/(theta**5*bfield))*ff*x**(-1.0)*ix
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    return val

def _alphanu_pl(x, nism, bfield, theta, xi, p, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param z_cool: normalised cooling lorentz factor
    :return: Synchrotron absorption coeff of power-law electrons
    """
    val = _c_alpha(p)*qe*(xi*nism/(theta**5*bfield))*_g_theta(theta,p=p)*x**(-(p+4.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= _low_freq_apl_correction(x,theta,p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    return val

def _tau_nu(x, nism, radius, bfield, theta, xi, p, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param radius: characteristic size of the emitting region (in cm)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param z_cool: normalised cooling lorentz factor
    :return: Total (thermal+non-thermal) synchrotron optical depth
    """
    alphanu_pl = _alphanu_pl(x=x,nism=nism,bfield=bfield,theta=theta,xi=xi,p=p,z_cool=z_cool)
    alphanu_thermal = _alphanu_th(x=x, nism=nism, bfield=bfield,theta=theta,z_cool=z_cool)
    val = radius*(alphanu_thermal + alphanu_pl)
    return val

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def _shocked_cocoon(time, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa):
    """
    :param time: source frame time in days
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in c
    :param eta: slope for ejecta density profile
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass shocked
    :param cos_theta_cocoon: cocoon opening angle
    :param kappa: opacity
    :return: namedtuple with lbol, r_photosphere, and temperature
    """
    c_kms = speed_of_light / km_cgs
    vej = vej * c_kms
    diff_const = solar_mass / (4*np.pi * speed_of_light * km_cgs)
    rshock = tshock * speed_of_light
    shocked_mass = mej * shocked_fraction
    theta = np.arccos(cos_theta_cocoon)
    tau_diff = np.sqrt(diff_const * kappa * shocked_mass / vej) / day_to_s

    t_thin = (c_kms / vej) ** 0.5 * tau_diff

    l0 = (theta ** 2 / 2) ** (1 / 3) * (shocked_mass * solar_mass *
                                        vej * km_cgs * rshock / (tau_diff * day_to_s) ** 2)

    lbol = l0 * (time/tau_diff)**(-4/(eta+2)) * (1 + np.tanh(t_thin - time))/2

    v_photosphere = vej * (time / t_thin) ** (-2. / (eta + 3))
    r_photosphere = km_cgs * day_to_s * v_photosphere * time
    temperature = (lbol / (4.0 * np.pi * sigma_sb * r_photosphere**2))**0.25

    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature'])
    output.lbol = lbol
    output.r_photosphere = r_photosphere
    output.temperature = temperature
    return output
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon_bolometric(time, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa, **kwargs):
    """
    :param time: source frame time in days
    :param mej: ejecta mass in solar masses
    :param vej: ejecta mass in km/s
    :param eta: slope for ejecta density profile
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass shocked
    :param cos_theta_cocoon: cocoon opening angle
    :param kappa: opacity
    :param kwargs: None
    :return: bolometric_luminosity
    """
    output = _shocked_cocoon(time, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa)
    return output.lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon(time, redshift, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in c
    :param eta: slope for ejecta density profile
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass shocked
    :param cos_theta_cocoon: cocoon opening angle
    :param kappa: opacity
    :param kwargs: Extra parameters used by function
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
        output = _shocked_cocoon(time=time, mej=mej, vej=vej, eta=eta,
                                 tshock=tshock, shocked_fraction=shocked_fraction,
                                 cos_theta_cocoon=cos_theta_cocoon, kappa=kappa)
        flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                                     dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 60000, 100))
        time_temp = np.linspace(1e-2, 10, 100)
        time_observer_frame = time_temp
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = _shocked_cocoon(time=time, mej=mej, vej=vej, eta=eta,
                                 tshock=tshock, shocked_fraction=shocked_fraction,
                                 cos_theta_cocoon=cos_theta_cocoon, kappa=kappa)
        fmjy = sed.blackbody_to_flux_density(temperature=output.temperature,
                                         r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
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

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...928..122M/abstract')
def csm_truncation_shock():
    raise NotImplementedError("This model is not yet implemented.")
