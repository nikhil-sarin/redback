import numpy as np
from collections import namedtuple
from scipy import special
from redback.constants import *
import redback.sed as sed
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, logger

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
    r_photosphere_post_td = (np.power((delta - 1) / (nn - 1) * ((time / td) ** 2 - 1) + 1, -1 / (delta + 1))* vt * time)
    r_photosphere = np.zeros(len(time))
    r_photosphere[time < td] = r_photosphere_pre_td[time < td]
    r_photosphere[time >= td] = r_photosphere_post_td[time >= td]

    sigmaT4 = lbol / (4 * np.pi * r_photosphere**2)
    temperature = np.power(sigmaT4 / sigma_sb, 0.25)

    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature'])
    output.lbol = lbol
    output.r_photosphere = r_photosphere
    output.temperature = temperature
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
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param kwargs: extra parameters to change physics and other settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param nn: density power law slope
    :param delta: inner density power law slope
    :param output_format: 'flux_density' or 'magnitude'
    :return: flux density or AB magnitude
    """
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy

    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    output = _shock_cooling(time, mass=mass, radius=radius, energy=energy, **kwargs)
    flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                             dl=dl, frequency=frequency)

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def _c_j(p):
    term1 = (special.gamma((p+5.0)/4.0)/special.gamma((p+7.0)/4.0))
    term2 = special.gamma((3.0*p+19.0)/12.0)
    term3 = special.gamma((3.0*p-1.0)/12.0)*((p-2.0)/(p+1.0))
    term4 = 3.0**((2.0*p-1.0)/2.0)
    term5 = 2.0**(-(7.0-p)/2.0)*np.pi**(-0.5)
    return term1*term2*term3*term4*term5

def _c_alpha(p):
    term1 = (special.gamma((p+6.0)/4.0)/special.gamma((p+8.0)/4.0))
    term2 = special.gamma((3.0*p+2.0)/12.0)
    term3 = special.gamma((3.0*p+22.0)/12.0)*(p-2.0)*3.0**((2.0*p-5.0)/2.0)
    term4 = 2.0**(p/2.0)*np.pi**(3.0/2.0)
    return term1*term2*term3*term4


def _g_theta(theta,p):
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    gtheta = ((p-1.0)*(1e0+aa*theta)/((p-1.0)*gamma_m - p+2.0))*(gamma_m/(3.0*theta))**(p-1.0)
    return gtheta

def _low_freq_jpl_correction(x,theta,p):
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
    val = _c_j(p)*(qe**3/(electron_mass*speed_of_light**2))*xi*nism*bfield*_g_theta(theta=theta,p=p)*x**(-(p-1.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= _low_freq_jpl_correction(x=x,theta=theta,p=p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    emissivity_pl = val
    return emissivity_pl

def _emissivity_thermal(x, nism, bfield, theta, z_cool):
    ff = 2.0*theta**2/special.kn(2,1.0/theta)
    ix = 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))
    val = (3.0**0.5/(8.0*np.pi))*(qe**3/(electron_mass*speed_of_light**2))*ff*nism*bfield*x*ix
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum(1e0, (z0/z_cool)**(-1))
    return val

def _alphanu_th(x, nism, bfield, theta, z_cool):
    ff = 2.0 * theta ** 2 / special.kn(2, 1.0 / theta)
    ix = 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))
    val = (np.pi*3.0**(-3.0/2.0))*qe*(nism/(theta**5*bfield))*ff*x**(-1.0)*ix
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    return val

def _alphanu_pl(x, nism, bfield, theta, xi, p, z_cool):
    val = _c_alpha(p)*qe*(xi*nism/(theta**5*bfield))*_g_theta(theta,p=p)*x**(-(p+4.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= _low_freq_apl_correction(x,theta,p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    return val

def _tau_nu(x, nism, radius, bfield, theta, xi, p, z_cool):
    alphanu_pl = _alphanu_pl(x=x,nism=nism,bfield=bfield,theta=theta,xi=xi,p=p,z_cool=z_cool)
    alphanu_thermal = _alphanu_th(x=x, nism=nism, bfield=bfield,theta=theta,z_cool=z_cool)
    val = radius*(alphanu_thermal + alphanu_pl)
    return val

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def _thermal_synchrotron(time, initial_n0, initial_velocity, initial_radius, eta, logepse, logepsb, xi, p, mu, mu_e, **kwargs):
    v0 = initial_velocity * speed_of_light
    t0 = eta*initial_radius/initial_velocity
    radius = initial_radius * (time/t0)**eta
    velocity = v0 * (time/t0)**(eta - 1)
    wind_slope = kwargs.get('wind_slope',2)
    nism = initial_n0 * (radius/initial_radius)**(-wind_slope)

    epsilon_T = 10**logepse
    epsilon_B = 10**logepsb

    frequency = kwargs['frequency']

    ne = 4.0*mu_e*nism
    beta = velocity/speed_of_light

    theta0 = epsilon_T * (9.0 * mu * proton_mass / (32.0 * mu_e * electron_mass)) * beta ** 2
    theta = (5.0*theta0-6.0+(25.0*theta0**2+180.0*theta0+36.0)**0.5)/30.0

    bfield = (9.0*np.pi*epsilon_B*nism*mu*proton_mass)**0.5*velocity
    # mean dynamical time:
    td = radius/velocity

    z_cool = (6.0 * np.pi * electron_mass * speed_of_light / (sigma_T * bfield ** 2 * td)) / theta
    normalised_frequency_denom = 3.0*theta**2*qe*bfield/(4.0*np.pi*electron_mass*speed_of_light)
    x = frequency / normalised_frequency_denom

    emissivity_pl = _emissivity_pl(x=x, nism=ne, bfield=bfield, theta=theta, xi=xi, p=p, z_cool=z_cool)

    emissivity_thermal = _emissivity_thermal(x=x, nism=ne, bfield=bfield, theta=theta, z_cool=z_cool)

    emissivity = emissivity_thermal + emissivity_pl

    tau = _tau_nu(x=x, nism=ne, radius=radius, bfield=bfield, theta=theta, xi=xi, p=p, z_cool=z_cool)

    lnu = 4.0 * np.pi ** 2 * radius ** 3 * emissivity * (1e0 - np.exp(-tau)) / tau
    if np.size(x) > 1:
        lnu[tau < 1e-10] = (4.0 * np.pi ** 2 * radius ** 3 * emissivity)[tau < 1e-10]
    elif tau < 1e-10:
        lnu = 4.0 * np.pi ** 2 * radius ** 3 * emissivity
    return lnu

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def thermal_synchrotron_lnu():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def thermal_synchrotron():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def _shocked_cocoon():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon_bolometric():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...928..122M/abstract')
def csm_truncation_shock():
    pass
