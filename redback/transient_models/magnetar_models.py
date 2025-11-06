import numpy as np
from astropy.cosmology import Planck18 as cosmo  # noqa

import scipy.special as ss
from collections import namedtuple
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumulative_trapezoid as cumtrapz
from inspect import isfunction
from redback.utils import logger, citation_wrapper

from redback.constants import *
from redback.transient_models.fireball_models import one_component_fireball_model

luminosity_models = ['evolving_magnetar', 'evolving_magnetar_only', 'gw_magnetar', 'radiative_losses',
                     'radiative_losses_smoothness', 'radiative_only', 'collapsing_radiative_losses']


def _mu_function(time, mu0, muinf, tm):
    mu = muinf + (mu0 - muinf) * np.exp(-time / tm)
    return mu


def _integrand(time, mu0, muinf, tm):
    mu = muinf + (mu0 - muinf) * np.exp(-time / tm)
    return mu ** 2

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...886....5S/abstract')
def evolving_magnetar_only(time, mu0, muinf, p0, sinalpha0, tm, II, **kwargs):
    """
    Millisecond magnetar model with evolution of inclination angle.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    mu0 : float
        Initial magnetic moment in units of 10^33 G cm^3.
    muinf : float
        Magnetic moment when field relaxes in units of 10^33 G cm^3.
    p0 : float
        Initial spin period.
    sinalpha0 : float
        Initial sin(alpha0) where alpha is the angle between B and P axes.
    tm : float
        Magnetic field decay timescale in days.
    II : float
        Moment of inertia in cgs.
    **kwargs : dict
        Additional keyword arguments:

        - output : str, optional
            Output type: 'luminosity' or full output. Default is 'luminosity'.

    Returns
    -------
    float or namedtuple
        Luminosity (depending on scaling) as a function of time, or full output.
    """
    mu0 = mu0 * 1e33  # G cm^3
    muinf = muinf * 1e33  # G cm^3
    tm = tm * 86400  # days
    eta = 0.1
    tau = np.zeros(len(time))
    for ii in range(len(time)):
        tau[ii], _ = quad(_integrand, 0, time[ii], args=(mu0, muinf, tm)) # noqa
    mu = _mu_function(time, mu0, muinf, tm)
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
    output = kwargs.get('output', 'luminosity')
    if output == 'luminosity':
        return luminosity / 1e50
    else:
        alpha = np.arcsin(ytau)
        nn_frac = (np.sin(alpha) * np.cos(alpha)) / (1 + np.sin(alpha) ** 2)
        omegadot = -(mu ** 2 * omegatau ** 3) / (II * speed_of_light ** 3) * (1 + np.sin(alpha) ** 2)
        mudot = -(mu - muinf) / tm
        nn = 3 + 2 * nn_frac ** 2 + 2 * omegatau / omegadot * mudot / mu
        output = namedtuple('output', ['luminosity', 'omegatau', 'ytau', 'mu', 'nn', 'alpha'])
        output.luminosity = luminosity / 1e50
        output.omegatau = omegatau
        output.ytau = ytau
        output.alpha = alpha
        output.mu = mu
        output.nn = nn
        return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...886....5S/abstract')
def evolving_magnetar(time, a_1, alpha_1, mu0, muinf, p0, sinalpha0, tm, II, **kwargs):
    """
    Millisecond magnetar model with evolution of inclination angle.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    mu0 : float
        Initial magnetic moment in units of 10^33 G cm^3.
    muinf : float
        Magnetic moment when field relaxes in units of 10^33 G cm^3.
    p0 : float
        Initial spin period.
    sinalpha0 : float
        Initial sin(alpha0) where alpha is the angle between B and P axes.
    tm : float
        Magnetic field decay timescale in days.
    II : float
        Moment of inertia in cgs.
    **kwargs : dict
        Additional keyword arguments for handling plotting.

    Returns
    -------
    np.ndarray
        Luminosity (depending on scaling) as a function of time.
    """
    pl = one_component_fireball_model(time=time, a_1=a_1, alpha_1=alpha_1)
    magnetar = evolving_magnetar_only(time=time, mu0=mu0, muinf=muinf,
                                      p0=p0, sinalpha0=sinalpha0, tm=tm, II=II)
    return pl + magnetar

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2001ApJ...552L..35Z/abstract')
def vacuum_dipole_magnetar_only(time, l0, tau, **kwargs):
    """
    Vacuum dipole magnetar model with fixed braking index of 3.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity or flux (depending on scaling of l0) as a function of time.
    """
    nn = 3
    lum = l0 * (1. + time / tau) ** ((1. + nn) / (1. - nn))
    return lum

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.1061R/abstract')
def full_vacuum_dipole_magnetar(time, a_1, alpha_1, l0, tau, **kwargs):
    """
    Generalised millisecond magnetar with curvature effect power law and vacuum dipole radiation.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity or flux (depending on scaling of l0) as a function of time.
    """
    pl = one_component_fireball_model(time=time, a_1=a_1, alpha_1=alpha_1)
    mag = vacuum_dipole_magnetar_only(time=time, l0=l0, tau=tau)
    return pl + mag

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...843L...1L/abstract')
def magnetar_only(time, l0, tau, nn, **kwargs):
    """
    Basic magnetar spin-down model.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity or flux (depending on scaling of l0) as a function of time.
    """
    lum = l0 * (1. + time / tau) ** ((1. + nn) / (1. - nn))
    return lum

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018PhRvD..98d3011S/abstract')
def gw_magnetar(time, a_1, alpha_1, fgw0, tau, nn, log_ii, **kwargs):
    """
    Magnetar model with gravitational wave emission.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    fgw0 : float
        Initial gravitational-wave frequency.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    log_ii : float
        Log10 moment of inertia.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.

    Notes
    -----
    Efficiency parameter eta is fixed to 0.1.
    """
    eta = 0.1
    omega_0 = fgw0 * np.pi  # spin frequency
    ii = 10 ** log_ii
    l0 = ((omega_0 ** 2) * eta * ii) / (2 * tau)
    l0_50 = l0 / 1e50

    magnetar = magnetar_only(time=time, l0=l0_50, tau=tau, nn=nn)
    pl = one_component_fireball_model(time=time, a_1=a_1, alpha_1=alpha_1)

    return pl + magnetar

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2006ApJ...648L..51S/abstract')
def basic_magnetar(time, p0, bp, mass_ns, theta_pb, **kwargs):
    """
    Basic magnetar spin-down model with physical parameters.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds in source frame.
    p0 : float
        Initial spin period in milliseconds.
    bp : float
        Polar magnetic field strength in units of 10^14 Gauss.
    mass_ns : float
        Mass of neutron star in solar masses.
    theta_pb : float
        Angle between spin and magnetic field axes in radians.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    erot = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
    tp = 1.3e5 * bp**(-2) * p0**2 * (mass_ns/1.4)**(3./2.) * (np.sin(theta_pb))**(-2)
    luminosity = erot / tp / (1. + time / tp)**2
    return luminosity

def _evolving_gw_and_em_magnetar(time, bint, bext, p0, chi0, radius, moi, **kwargs):
    """
    Magnetar model with combined GW and EM spin-down with constant spin-magnetic field inclination angle.

    Only EM contributes to observed emission.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in seconds (must be a large array as this function is semianalytic).
    bint : float
        Internal magnetic field in G.
    bext : float
        External magnetic field in G.
    p0 : float
        Spin period in s.
    chi0 : float
        Initial inclination angle.
    radius : float
        Radius of NS in cm.
    moi : float
        Moment of inertia of NS.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    namedtuple
        Output containing energy components and spin evolution.
    """
    epsilon_b = -3e-4 * (bint / bext) ** 2 * (bext / 1e16) ** 2
    omega_0 = 2.0 * np.pi / p0
    erot = 0.5 * moi * omega_0**2

    dt = time[1:] - time[:-1]
    omega = np.zeros_like(time)
    chi = chi0

    omega[0] = omega_0

    for i in range(len(time) - 1):
        omega[i + 1] = omega[i] + dt[i] * (
            -(bext**2*radius**6/(moi*speed_of_light**3))*omega[i]**3*(1. + np.sin(chi)) - (
            2.0*graviational_constant*moi*epsilon_b**2/(5.0*speed_of_light**5)) * omega[i]**5*np.sin(chi)**2*(1.0+15.0*np.sin(chi)**2))

    Edot_d = (bext ** 2 * radius ** 6 / (4*speed_of_light ** 3)) * omega ** 4 * (1.0 + np.sin(chi)**2)
    Edot_gw = (2.0 * graviational_constant * moi ** 2 * epsilon_b ** 2 / (5.0 * speed_of_light ** 5)) * omega ** 6 * np.sin(chi)**2 * (
                1.0 + 15.0 * np.sin(chi)**2)

    Ed = cumtrapz(Edot_d, x=time)
    Egw = cumtrapz(Edot_gw, x=time)

    En_t =  3.5e50*(bint/1e17)**2*(radius/1.5e6)**3
    En_p = 5.5e47 * (bext / 1e14) ** 2 * (radius / 1.5e6) ** 3

    output = namedtuple('output', ['e_gw', 'e_em', 'tsd', 'epsilon_b', 'e_magnetic', 'Edot_d', 'Edot_gw', 'erot'])
    output.e_gw = Egw[-1]
    output.e_em = Ed[-1]
    output.erot = erot
    period = p0
    output.tsd = 2.4 * (period/1e-3)**2 *((bext/1e14)**(2) + 7.2*(bint/1e16)**4*(period/1e-3)**(-2))**(-1) * (60*60)
    output.epsilon_b = epsilon_b
    output.e_magnetic = En_t + En_p
    output.Edot_d = Edot_d
    output.Edot_gw = Edot_gw
    return output

def magnetar_luminosity_evolution(time, logbint, logbext, p0, chi0, radius, logmoi, **kwargs):
    """
    Magnetar luminosity evolution with combined GW and EM spin-down.

    Assumes a constant spin-magnetic field inclination angle. Only EM contributes to observed emission.

    Parameters
    ----------
    time : np.ndarray
        Time in source frame in seconds.
    logbint : float
        Log10 internal magnetic field in G.
    logbext : float
        Log10 external magnetic field in G.
    p0 : float
        Spin period in s.
    chi0 : float
        Initial inclination angle.
    radius : float
        Radius of NS in km.
    logmoi : float
        Log10 moment of inertia of NS.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    time_temp = np.geomspace(1e-4, 1e7, 300)
    bint = 10**logbint
    bext = 10**logbext
    radius = radius * km_cgs
    moi = 10**logmoi
    output = _evolving_gw_and_em_magnetar(time=time_temp, bint=bint, bext=bext, p0=p0, chi0=chi0, radius=radius, moi=moi)
    lum = output.Edot_d
    lum_func = interp1d(time_temp, y=lum)
    return lum_func(time)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...843L...1L/abstract')
def full_magnetar(time, a_1, alpha_1, l0, tau, nn, **kwargs):
    """
    Generalised millisecond magnetar with curvature effect power law.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity or flux (depending on scaling of l0) as a function of time.
    """
    pl = one_component_fireball_model(time=time, a_1=a_1, alpha_1=alpha_1)
    mag = magnetar_only(time=time, l0=l0, tau=tau, nn=nn)
    return pl + mag

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020PhRvD.101f3021S/abstract')
def collapsing_magnetar(time, a_1, alpha_1, l0, tau, nn, tcol, **kwargs):
    """
    Generalised millisecond magnetar with curvature effect power law and a collapse time.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    tcol : float
        Collapse time in seconds.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity or flux (depending on scaling of l0) as a function of time.
    """
    pl = one_component_fireball_model(time, a_1, alpha_1)
    mag = np.heaviside(tcol - time, 1e-50) * magnetar_only(time, l0, tau, nn)

    return pl + mag

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...872..114S/abstract')
def general_magnetar(time, a_1, alpha_1,
                     delta_time_one, alpha_2, delta_time_two, **kwargs):
    """
    Reparameterized millisecond magnetar model (piecewise).

    Parameters
    ----------
    time : np.ndarray
        Time array for power law.
    a_1 : float
        Power law decay amplitude.
    alpha_1 : float
        Power law decay exponent.
    delta_time_one : float
        Time between start and end of prompt emission.
    alpha_2 : float
        Reparameterized braking index n.
    delta_time_two : float
        Time between end of prompt emission and end of magnetar model plateau phase (tau).
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity or flux (depending on scaling of l0) as a function of time.
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
    f2 = a_2 * (1. + (time[x] / tau)) ** gamma
    return np.concatenate((f1, f2))

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def _integral_general(time, t0, kappa, tau, nn, **kwargs):
    """
    General integral for radiative losses model.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    t0 : float
        Time for radiative losses to start in seconds.
    kappa : float
        Radiative efficiency.
    tau : float
        Spin-down damping timescale in seconds.
    nn : float
        Braking index.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Integral result for luminosity calculation.
    """
    first_term, second_term = _get_integral_terms(time=time, t0=t0, kappa=kappa, tau=tau, nn=nn)
    return first_term - second_term

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def _integral_general_collapsing(time, t0, kappa, tau, nn, tcol, **kwargs):
    """
    General integral for radiative losses model with collapse.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    t0 : float
        Time for radiative losses to start in seconds.
    kappa : float
        Radiative efficiency.
    tau : float
        Spin-down damping timescale in seconds.
    nn : float
        Braking index.
    tcol : float
        Collapse time in seconds.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Integral result for luminosity calculation with collapse.
    """
    first_term, second_term = _get_integral_terms(time=time, t0=t0, kappa=kappa, tau=tau, nn=nn)
    return np.heaviside(tcol - time, 1e-50) * (first_term - second_term)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def _get_integral_terms(time, t0, kappa, tau, nn):
    """
    Get integral terms for radiative losses model.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    t0 : float
        Time for radiative losses to start in seconds.
    kappa : float
        Radiative efficiency.
    tau : float
        Spin-down damping timescale in seconds.
    nn : float
        Braking index.

    Returns
    -------
    tuple
        First and second terms of the integral.
    """
    alpha = (1 + nn) / (-1 + nn)
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time / tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -t0 / tau)
    first_term = (time ** (1 + kappa) * pft) / (1 + kappa)
    second_term = (t0 ** (1 + kappa) * pst) / (1 + kappa)
    return first_term, second_term

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def _integral_mdr(time, t0, kappa, a, **kwargs):
    """
    Calculate integral for vacuum dipole radiation.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    t0 : float
        Time for radiative losses to start in seconds.
    kappa : float
        Radiative efficiency.
    a : float
        1/tau (inverse of spin-down damping timescale).
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    np.ndarray
        Integral for vacuum dipole radiation.
    """
    z_f = (1 + a * time) ** (-1)
    z_int = (1 + a * t0) ** (-1)
    divisor_i = a ** (1 + kappa) * (kappa - 1) * (1 + a * t0) ** (1 - kappa)
    divisor_f = a ** (1 + kappa) * (kappa - 1) * (1 + a * time) ** (1 - kappa)
    first = ss.hyp2f1(1 - kappa, -kappa, 2 - kappa, z_f) / divisor_f
    second = ss.hyp2f1(1 - kappa, -kappa, 2 - kappa, z_int) / divisor_i
    return first - second

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def piecewise_radiative_losses(time, a_1, alpha_1, l0, tau, nn, kappa, t0_s, **kwargs):
    """
    Radiative losses model with smoothness and continuity between prompt and magnetar term.

    Assumes smoothness and continuity by fixing e0 variable.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    kappa : float
        Radiative efficiency.
    t0_s : float
        Time for radiative losses to start in seconds.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    pl_time = np.where(time <= t0_s)
    magnetar_time = np.where(time > t0_s)
    e0 = (a_1 * t0_s ** alpha_1 * t0_s) / kappa
    pl = one_component_fireball_model(time[pl_time], a_1, alpha_1)

    loss_term = e0 * (t0_s / time[magnetar_time]) ** kappa
    integ = _integral_general(time[magnetar_time], t0_s, kappa, tau, nn)
    energy_loss_total = ((l0 / (time[magnetar_time] ** kappa)) * integ) + loss_term

    lum = (kappa * energy_loss_total / time[magnetar_time])

    return np.concatenate((pl, lum))

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def radiative_losses(time, a_1, alpha_1, l0, tau, nn, kappa, t0_s, log_e0, **kwargs):
    """
    Radiative losses model with a step function indicating magnetar term turns on at T0.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    kappa : float
        Radiative efficiency.
    t0_s : float
        Time for radiative losses to start in seconds.
    log_e0 : float
        Log10 E0 to connect curvature effect energy with transition point energy, captures flares.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    e0 = 10 ** log_e0
    pl = one_component_fireball_model(time, a_1, alpha_1)
    loss_term = e0 * (t0_s / time) ** kappa
    integ = _integral_general(time, t0_s, kappa, tau, nn)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = pl + np.heaviside(time - t0_s, 1) * lum

    return total

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def radiative_only(time, l0, tau, nn, kappa, t0_s, log_e0, **kwargs):
    """
    Radiative losses model only, without prompt emission.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    kappa : float
        Radiative efficiency.
    t0_s : float
        Time for radiative losses to start in seconds.
    log_e0 : float
        Log10 E0 to connect curvature effect energy smoothly with transition point energy.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    e0 = 10 ** log_e0
    loss_term = e0 * (t0_s / time) ** kappa
    integ = _integral_general(time, t0_s, kappa, tau, nn)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = np.heaviside(time - t0_s, 1) * lum

    return total

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def radiative_losses_smoothness(time, a_1, alpha_1, l0, tau, nn, kappa, t0_s, log_e0, **kwargs):
    """
    Radiative losses model with smoothness constraint on transition energy.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    kappa : float
        Radiative efficiency.
    t0_s : float
        Time for radiative losses to start in seconds.
    log_e0 : float
        Log10 E0 to connect curvature effect energy smoothly with transition point energy.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    pl = one_component_fireball_model(time, a_1, alpha_1)
    e0 = 10 ** log_e0
    e0_def = (a_1 * t0_s ** alpha_1 * t0_s) / kappa
    e0_use = np.min([e0, e0_def])
    loss_term = e0_use * (t0_s / time) ** kappa
    integ = _integral_general(time, t0_s, kappa, tau, nn)

    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = pl + np.heaviside(time - t0_s, 1) * lum

    return total

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2011A%26A...526A.121D/abstract')
def radiative_losses_mdr(time, a_1, alpha_1, l0, tau, kappa, log_e0, t0_s, **kwargs):
    """
    Radiative losses model for vacuum dipole radiation.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    kappa : float
        Radiative efficiency.
    log_e0 : float
        Log10 E0 to connect curvature effect energy smoothly with transition point energy.
    t0_s : float
        Time for radiative losses to start in seconds.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    a = 1. / tau
    e0 = 10 ** log_e0
    pl = one_component_fireball_model(time, a_1, alpha_1)
    loss_term = e0 * (t0_s / time) ** kappa
    integ = _integral_mdr(time, t0_s, kappa, a)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term

    lightcurve = (kappa * energy_loss_total / time)

    return np.heaviside(time - t0_s, 1) * lightcurve + pl

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5986S/abstract')
def collapsing_radiative_losses(time, a_1, alpha_1, l0, tau, nn, tcol, kappa, t0_s, log_e0, **kwargs):
    """
    Radiative losses model with collapse time.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    a_1 : float
        Amplitude of curvature effect power law.
    alpha_1 : float
        Index of curvature effect power law.
    l0 : float
        Initial luminosity parameter.
    tau : float
        Spin-down damping timescale.
    nn : float
        Braking index.
    tcol : float
        Collapse time in seconds.
    kappa : float
        Radiative efficiency.
    t0_s : float
        Time for radiative losses to start in seconds.
    log_e0 : float
        Log10 E0 to connect curvature effect energy smoothly with transition point energy.
    **kwargs : dict
        Additional keyword arguments for handling plotting/other functionality.

    Returns
    -------
    np.ndarray
        Luminosity.
    """
    e0 = 10 ** log_e0
    pl = one_component_fireball_model(time, a_1, alpha_1)
    loss_term = e0 * (t0_s / time) ** kappa
    integ = _integral_general_collapsing(time, t0_s, kappa, tau, nn, tcol)
    energy_loss_total = ((l0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * energy_loss_total / time)
    total = pl + np.heaviside(time - t0_s, 1) * lum

    return total

@citation_wrapper('redback')
def luminosity_based_magnetar_models(time, photon_index, **kwargs):
    """
    Luminosity-based magnetar models for fitting to flux data with redshift prior.

    Parameters
    ----------
    time : np.ndarray
        Time in observer frame.
    photon_index : float
        Photon index.
    **kwargs : dict
        Additional keyword arguments:

        - base_model : str or callable
            Name or function of magnetar model to use.
        - redshift : float
            Source redshift.
        - cosmology : astropy.cosmology object, optional
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    np.ndarray
        Flux density in appropriate units.
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']
    if isfunction(base_model):
        function = base_model
    elif base_model not in luminosity_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['magnetar_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")
    redshift = kwargs['redshift']
    kcorr = (1 + redshift)**(photon_index - 2)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time = time / (1 + redshift)
    lum = function(time, **kwargs) * 1e50
    flux = lum / (4*np.pi*dl**2*kcorr)
    return flux
