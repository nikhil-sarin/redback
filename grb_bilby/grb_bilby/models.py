import astropy.constants as cc
import numpy as np
import scipy.special as ss
from scipy.integrate import quad
from . constants import *

from astropy.cosmology import Planck15 as cosmo
from scipy.integrate import simps

from .utils import logger

try:
    import afterglowpy as afterglows

    # keep so you can eventually generalise
    jettype_dict = {'tophat': afterglow.jet.TopHat, 'gaussian': afterglow.jet.Gaussian,
                    'powerlaw_w_core': afterglow.jet.PowerLawCore, 'gaussian_w_core': afterglow.jet.GaussianCore,
                    'cocoon': afterglow.Spherical, 'smooth_power_law': afterglow.jet.PowerLaw,
                    'cone': afterglow.jet.Cone}
    spectype_dict = {'no_inverse_compton': 0, 'inverse_compton': 1}

except ModuleNotFoundError as e:
    logger.warning(e)


def cocoon(time, redshift, umax,umin, logEi,k,mej,logn0,p,logepse,logepsb,ksin,g0,**supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False #supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = jettype_dict['cocoon']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    e0 = 10**logEi
    n0 = 10**logn0
    epse = 10**logepse
    epsb = 10**logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'umax': umax, 'E0': e0,
         'umin': umin, 'k':k, 'mej':mej,'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def cone_afterglow(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = jettype_dict['cone']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres, 'thetaWing':thw}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def gaussiancore(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = jettype_dict['gaussian_w_core']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres, 'thetaWing':thw}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def gaussian(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = jettype_dict['gaussian']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres, 'thetaWing':thw}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def smoothpowerlaw(time, redshift, thv, loge0, thw, thc, beta, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = jettype_dict['smooth_power_law']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres, 'thetaWing':thw, 'b':beta}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def powerlawcore(time, redshift, thv, loge0, thw, thc, beta, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = jettype_dict['powerlaw_w_core']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres, 'thetaWing':thw, 'b':beta}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def tophat(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype  = jettype_dict['tophat']
    spectype = supplementary_data['spectype']
    frequency = supplementary_data['frequency']
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres}
    fluxdensity = afterglow.fluxDensity(time, frequency, **Z)
    return fluxdensity

def tophat_integrated(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, **supplementary_data):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = False#supplementary_data['spread']
    latres = supplementary_data['latres']
    tres = supplementary_data['tres']
    jettype = supplementary_data['jettype']
    spectype = supplementary_data['spectype']

    #eventually need this to be smart enough to figure the bounds based on data used
    frequency_bounds = supplementary_data['frequency'] #should be 2 numbers that serve as start and end point
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype,'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0,'p': p,'epsilon_e': epse,'epsilon_B': epsb,
         'xi_N': ksin,'d_L': dl, 'z': redshift, 'L0':0,'q':0,'ts':0, 'g0':g0,
         'spread':spread, 'latRes':latres, 'tRes':tres}

    #nu_1d = np.linspace(7.254e16, 1.693e18, 3)
    nu_1d = np.linspace(frequency_bounds[0], frequency_bounds[1], 3)
    t, nu = np.meshgrid(time, nu_1d)  # meshgrid makes 2D t and n
    t = t.flatten()
    nu = nu.flatten()
    fluxdensity = afterglow.fluxDensity(t, nu, **Z)
    lightcurve_at_nu = fluxdensity.reshape(len(nu_1d), len(time))
    prefactor = 1e-26
    lightcurve_at_nu = prefactor * lightcurve_at_nu
    lightcurve = simps(lightcurve_at_nu, axis=0, x=nu_1d)
    return lightcurve

def mu_function(time, mu0, muinf, tm):
    mu = muinf + (mu0 - muinf)*np.exp(-time/tm)
    return mu

def integrand(time, mu0, muinf, tm):
    mu = muinf + (mu0 - muinf)*np.exp(-time/tm)
    return mu**2

def evolving_magnetar_only(time, mu0, muinf, p0, sinalpha0, tm, II, **kwargs):
    """
    Model from Mus+2019
    :param time: time
    :param kwargs: key word argument for handling plotting
    :return: luminosity (depending on scaling) as a function of time.
    """
    mu0 = mu0 * 1e33 # G cm^3
    muinf = muinf * 1e33 # G cm^3
    tm = tm * 86400 #days
    eta = 0.1
    tau = np.zeros(len(time))
    for ii in range(len(time)):
        tau[ii], _ = quad(integrand, 0, time[ii], args=(mu0, muinf, tm))
    mu = mu_function(time, mu0, muinf, tm)
    omega0 = (2 * np.pi) / p0
    tau = (omega0 ** 2) / (II * speed_of_light ** 3) * tau
    y0 = sinalpha0
    common_frac = (np.log(1 + 2 * tau) - 4 * tau) / (1 + 2 * tau)
    ftau = 2 * (1 - y0 ** 2) ** 2 * tau
    + y0 ** 2 * np.log(1 + 2 * tau)
    + y0 ** 4 * common_frac
    - y0 ** 8 * (np.log(1 + 2 * tau) + common_frac)
    ytau = y0 / ((1 + ftau) ** 0.5)
    omegatau = omega0 * (1 - y0 ** 2) * ((1 + ftau) ** 0.5) / (1 - y0 ** 2 + ftau)
    luminosity = eta * (mu ** 2 * omegatau ** 4) / (speed_of_light ** 3) * (1 + ytau ** 2)
    return luminosity/1e50

def evolving_magnetar(time, A_1, alpha_1, mu0, muinf, p0, sinalpha0, tm, II, **kwargs):
    """
    :param time: time
    :param L0: luminosity parameter
    :param tau: spin-down damping timescale
    :param nn: braking index
    :param kwargs: key word argument for handling plotting
    :return: luminosity or flux (depending on scaling) as a function of time.
    """
    pl = one_component_fireball_model(time=time, a_1=A_1, alpha_1=alpha_1)
    magnetar = evolving_magnetar_only(time=time, mu0=mu0, muinf=muinf,
                                      p0=p0, sinalpha0=sinalpha0, tm=tm, II=II)
    return pl + magnetar


def magnetar_only(time, l0, tau, nn, **kwargs):
    """
    :param time: time
    :param l0: luminosity parameter
    :param tau: spin-down damping timescale
    :param nn: braking index
    :param kwargs: key word argument for handling plotting
    :return: luminosity or flux (depending on scaling) as a function of time.
    """
    lum = l0 * (1. + time / tau) ** ((1. + nn) / (1. - nn))
    return lum


def gw_magnetar(time, a_1, alpha_1, fgw0, tau, nn, log_ii, **kwargs):
    """
    Model from Sarin+2018
    :param time:
    :param a_1:
    :param alpha_1:
    :param fgw0: initial gravitational-wave frequency
    :param tau:
    :param nn:
    :param log_ii: log10 moment of inertia
    :param eta: fixed to 0.1, its a fudge factor for the efficiency
    :param kwargs:
    :return: luminosity
    """
    eta = 0.1
    omega_0 = fgw0 * np.pi  # spin frequency
    ii = 10 ** log_ii
    l0 = ((omega_0 ** 2) * eta * ii) / (2 * tau)
    l0_50 = l0 / 1e50

    magnetar = magnetar_only(time=time, l0=l0_50, tau=tau, nn=nn)
    pl = one_component_fireball_model(time=time, a_1=a_1, alpha_1=alpha_1)

    return pl + magnetar


def full_magnetar(time, a_1, alpha_1, l0, tau, nn, **kwargs):
    """
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param nn:
    :param kwargs:
    :return:
    """
    pl = one_component_fireball_model(time=time, a_1=a_1, alpha_1=alpha_1)
    mag = magnetar_only(time=time, l0=l0, tau=tau, nn=nn)
    return pl + mag


def collapsing_magnetar(time, a_1, alpha_1, l0, tau, nn, tcol, **kwargs):
    """
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param nn:
    :param tcol:
    :param kwargs:
    :return:
    """
    pl = one_component_fireball_model(time, a_1, alpha_1)
    mag = np.heaviside(tcol - time, 1e-50) * magnetar_only(time, l0, tau, nn)

    return pl + mag


def general_magnetar(time, a_1, alpha_1,
                     delta_time_one, alpha_2, delta_time_two, **kwargs):
    """
    Reparameterized millisecond magnetar model from Sarin et al. (2018b) (piecewise)
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: Reparameterized braking index n
    :param delta_time_two: time between end of prompt emission and end of magnetar model plateau phase, (tau)
    """

    time_one = delta_time_one
    tau = delta_time_one + delta_time_two
    nn = (alpha_2 - 1.) / (alpha_2 + 1.)
    gamma = (1. + nn) / (1. - nn)
    num = (a_1 * time_one ** alpha_1)
    denom = ((1. + (time_one / tau)) ** gamma)
    a_2 = num / denom

    w = np.where(time < time_one)
    x = np.where(time > time_one)

    f1 = a_1 * time[w] ** alpha_1
    f2 = a_2 * (1. + (time[x] / tau)) ** (gamma)

    total = np.concatenate((f1, f2))

    return total


def one_component_fireball_model(time, a_1, alpha_1, **kwargs):
    """
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param kwargs:
    :return:
    """
    return a_1 * time ** alpha_1


def two_component_fireball_model(time, a_1, alpha_1,
                                 delta_time_one, alpha_2, **kwargs):
    """
    Two component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    """
    time_one = delta_time_one
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    w = np.where(time < time_one)
    x = np.where(time > time_one)

    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2

    total = np.concatenate((f1, f2))

    return total


def three_component_fireball_model(time, a_1, alpha_1,
                                   delta_time_one, alpha_2,
                                   delta_time_two, alpha_3, **kwargs):
    """
    Three component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where(time > time_two)
    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2
    f3 = amplitude_three * time[y] ** alpha_3

    total = np.concatenate((f1, f2, f3))
    return total


def four_component_fireball_model(time, a_1, alpha_1, delta_time_one,
                                  alpha_2, delta_time_two,
                                  alpha_3, delta_time_three,
                                  alpha_4, **kwargs):
    """
    Four component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where((time_two < time) & (time < time_three))
    z = np.where(time > time_three)
    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2
    f3 = amplitude_three * time[y] ** alpha_3
    f4 = amplitude_four * time[z] ** alpha_4

    total = np.concatenate((f1, f2, f3, f4))

    return total


def five_component_fireball_model(time, a_1, alpha_1,
                                  delta_time_one, alpha_2,
                                  delta_time_two, alpha_3,
                                  delta_time_three, alpha_4,
                                  delta_time_four, alpha_5):
    """
    Five component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four

    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** alpha_4 / (time_four ** alpha_5)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where(time > time_four)

    f1 = a_1 * time[u] ** alpha_1
    f2 = amplitude_two * time[v] ** alpha_2
    f3 = amplitude_three * time[w] ** alpha_3
    f4 = amplitude_four * time[x] ** alpha_4
    f5 = amplitude_five * time[y] ** alpha_5

    total = np.concatenate((f1, f2, f3, f4, f5))

    return total


def six_component_fireball_model(time, a_1, alpha_1,
                                 delta_time_one, alpha_2,
                                 delta_time_two, alpha_3,
                                 delta_time_three, alpha_4,
                                 delta_time_four, alpha_5,
                                 delta_time_five, alpha_6):
    """
    six component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    :param delta_time_five: time between fourth and fifth power laws
    :param alpha_6: power law decay exponent for sixth power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four
    time_five = time_four + delta_time_five

    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** alpha_4 / (time_four ** alpha_5)
    amplitude_six = amplitude_five * time_five ** alpha_5 / (time_five ** alpha_6)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where((time_four < time) & (time < time_five))
    z = np.where(time > time_five)

    f1 = a_1 * time[u] ** alpha_1
    f2 = amplitude_two * time[v] ** alpha_2
    f3 = amplitude_three * time[w] ** alpha_3
    f4 = amplitude_four * time[x] ** alpha_4
    f5 = amplitude_five * time[y] ** alpha_5
    f6 = amplitude_six * time[z] ** alpha_6

    total = np.concatenate((f1, f2, f3, f4, f5, f6))

    return total


def integral_general(time, t0, kappa, tau, nn, **kwargs):
    """
    General integral for radiative losses model
    :param time:
    :param t0:
    :param kappa:
    :param tau:
    :param nn:
    :param kwargs:
    :return:
    """
    alpha = ((1 + nn) / (-1 + nn))
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time / tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -t0 / tau)
    first_term = (time ** (1 + kappa) * pft) / (1 + kappa)
    second_term = (t0 ** (1 + kappa) * pst) / (1 + kappa)
    integral = (first_term - second_term)
    return integral


def integral_general_collapsing(time, t0, kappa, tau, nn, tcol, **kwargs):
    """
    General collapsing integral for radiative losses model
    :param time:
    :param t0:
    :param kappa:
    :param tau:
    :param nn:
    :param tcol:
    :param kwargs:
    :return:
    """
    alpha = ((1 + nn) / (-1 + nn))
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time / tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -t0 / tau)
    first_term = (time ** (1 + kappa) * pft) / (1 + kappa)
    second_term = (t0 ** (1 + kappa) * pst) / (1 + kappa)
    integral = np.heaviside(tcol - time, 1e-50) * (first_term - second_term)
    return integral


def integral_mdr(time, t0, kappa, a, **kwargs):
    z_f = (1 + a * time) ** (-1)
    z_int = (1 + a * t0) ** (-1)
    divisor_i = a ** (1 + kappa) * (kappa - 1) * (1 + a * t0) ** (1 - kappa)
    divisor_f = a ** (1 + kappa) * (kappa - 1) * (1 + a * time) ** (1 - kappa)
    first = ss.hyp2f1(1 - kappa, -kappa, 2 - kappa, z_f) / divisor_f
    second = ss.hyp2f1(1 - kappa, -kappa, 2 - kappa, z_int) / divisor_i
    return first - second


def piecewise_radiative_losses(time, a_1, alpha_1, l0, tau, nn, kappa, t0, **kwargs):
    """
    assumes smoothness and continuity between the prompt and magnetar term by fixing e0 variable
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param nn:
    :param kappa:
    :param t0:
    :param kwargs:
    :return:
    """
    pl_time = np.where(time <= t0)
    magnetar_time = np.where(time > t0)
    e0 = (a_1 * t0 ** alpha_1 * t0) / kappa
    pl = one_component_fireball_model(time[pl_time], a_1, alpha_1)

    loss_term = e0 * (t0 / time[magnetar_time]) ** kappa
    integ = integral_general(time[magnetar_time], t0, kappa, tau, nn)
    energy_loss_total = ((l0 / (time[magnetar_time] ** kappa)) * integ) + loss_term

    lum = (kappa * energy_loss_total / time[magnetar_time])

    total = np.concatenate((pl, lum))

    return total


def radiative_losses(time, a_1, alpha_1, l0, tau, nn, kappa, t0, log_e0, **kwargs):
    """
    radiative losses model with a step function, indicating the magnetar term turns on at T0
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param nn:
    :param kappa:
    :param t0:
    :param log_e0:
    :param kwargs:
    :return:
    """
    e0 = 10 ** log_e0
    pl = one_component_fireball_model(time, a_1, alpha_1)
    loss_term = e0 * (t0 / time) ** kappa
    integ = integral_general(time, t0, kappa, tau, nn)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = pl + np.heaviside(time - t0, 1) * lum

    return total


def radiative_only(time, l0, tau, nn, kappa, t0, log_e0, **kwargs):
    """
    radiative losses model only
    :param time:
    :param l0:
    :param tau:
    :param nn:
    :param kappa:
    :param t0:
    :param log_e0:
    :param kwargs:
    :return:
    """
    e0 = 10 ** log_e0
    loss_term = e0 * (t0 / time) ** kappa
    integ = integral_general(time, t0, kappa, tau, nn)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = np.heaviside(time - t0, 1) * lum

    return total


def radiative_losses_smoothness(time, a_1, alpha_1, l0, tau, nn, kappa, t0, log_e0, **kwargs):
    """
    radiative losses model with a step function, indicating the magnetar term turns on at T0
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param nn:
    :param kappa:
    :param t0:
    :param log_e0:
    :param kwargs:
    :return:
    """
    pl = one_component_fireball_model(time, a_1, alpha_1)
    e0 = 10 ** log_e0
    e0_def = (a_1 * t0 ** alpha_1 * t0) / kappa
    e0_use = np.min([e0, e0_def])
    loss_term = e0_use * (t0 / time) ** kappa
    integ = integral_general(time, t0, kappa, tau, nn)

    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = pl + np.heaviside(time - t0, 1) * lum

    return total


def radiative_losses_mdr(time, a_1, alpha_1, l0, tau, kappa, log_e0, t0, **kwargs):
    """
    radiative losses model for vacuum dipole radiation
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param kappa:
    :param t0:
    :param log_e0:
    :param kwargs:
    :return:
    """
    a = 1. / tau
    e0 = 10 ** log_e0
    pl = one_component_fireball_model(time, a_1, alpha_1)
    loss_term = e0 * (t0 / time) ** kappa
    integ = integral_mdr(time, t0, kappa, a)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term

    lightcurve = (kappa * energy_loss_total / time)

    return np.heaviside(time - t0, 1) * lightcurve + pl


def collapsing_radiative_losses(time, a_1, alpha_1, l0, tau, nn, tcol, kappa, t0, log_e0, **kwargs):
    """
    radiative losses model with collapse time
    :param time:
    :param a_1:
    :param alpha_1:
    :param l0:
    :param tau:
    :param nn:
    :param tcol:
    :param kappa:
    :param t0:
    :param log_e0:
    :param kwargs:
    :return:
    """
    e0 = 10 ** log_e0
    pl = one_component_fireball_model(time, a_1, alpha_1)
    loss_term = e0 * (t0 / time) ** kappa
    integ = integral_general_collapsing(time, t0, kappa, tau, nn, tcol)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = pl + np.heaviside(time - t0, 1) * lum

    return total
