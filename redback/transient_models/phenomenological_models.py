import numpy as np
from redback.utils import citation_wrapper
from redback.constants import speed_of_light_si

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract')
def bazin_sne(time, aa, bb, t0, tau_rise, tau_fall, **kwargs):
    """
    Bazin function for CCSN light curves

    :param time: time array in arbitrary units
    :param aa: Normalisation on the Bazin function
    :param bb: Additive constant
    :param t0: start time
    :param tau_rise: exponential rise time
    :param tau_fall: exponential fall time
    :return: flux in units set by AA
    """
    flux = aa * np.exp(-((time - t0) / tau_fall) / (1 + np.exp(-(time - t0) / tau_rise))) + bb
    return flux

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def villar_sne(time, aa, cc, t0, tau_rise, tau_fall, gamma, nu, **kwargs):
    """
    Villar function for SN light curves

    :param time: time array in arbitrary units
    :param aa: normalisation on the Villar function, amplotude
    :param cc: additive constant, baseline flux
    :param t0: start time
    :param tau_rise: exponential rise time
    :param tau_fall: exponential fall time
    :param gamma: plateau duration
    :param nu: related to beta and between 0 an 1; nu = -beta/gamma / A
    :param kwargs:
    :return: flux in units set by AA
    """
    mask1 = time < t0 + gamma
    mask2 = (time >= t0 + gamma)
    flux = np.zeros_like(time)
    norm = cc + (aa / (1 + np.exp(-(time - t0)/tau_rise)))
    flux[mask1] = norm[mask1] * (1 - (nu * ((time[mask1] - t0)/gamma)))
    flux[mask2] = norm[mask2] * ((1 - nu) * np.exp(-((time[mask2] - t0 - gamma)/tau_fall)))
    return np.concatenate((flux[mask1], flux[mask2]))

def fallback_lbol(time, logl1, tr, **kwargs):
    """
    :param time: time in seconds
    :param logl1: luminosity scale in log 10 ergs
    :param tr: transition time for flat luminosity to power-law decay
    :return: lbol
    """
    l1 = 10**logl1
    time = time * 86400
    tr = tr * 86400
    lbol = l1 * time**(-5./3.)
    lbol[time < tr] = l1 * tr**(-5./3.)
    return lbol

def line_spectrum(wavelength, line_amp, cont_amp, x0, **kwargs):
    """
    A gaussian to add or subtract from a continuum spectrum to mimic absorption or emission lines

    :param wavelength: wavelength array in whatever units
    :param line_amp: line amplitude scale
    :param cont_amp: Continuum amplitude scale
    :param x0: Position of emission line
    :return: spectrum in whatever units set by line_amp
    """
    spectrum = line_amp / cont_amp * np.exp(-(wavelength - x0) ** 2. / (2 * cont_amp ** 2) )
    return spectrum

def line_spectrum_with_velocity_dispersion(angstroms, wavelength_center, line_strength, velocity_dispersion):
    """
    A Gaussian line profile with velocity dispersion

    :param angstroms: wavelength array in angstroms or arbitrary units
    :param wavelength_center: center of the line in angstroms
    :param line_strength: line amplitude scale
    :param velocity_dispersion: velocity in m/s
    :return: spectrum in whatever units set by line_strength
    """

    # Calculate the Doppler shift for each wavelength using Gaussian profile
    intensity = line_strength * np.exp(-0.5 * ((angstroms - wavelength_center) / wavelength_center * speed_of_light_si / velocity_dispersion) ** 2)
    return intensity

def gaussian_rise(time, a_1, peak_time, sigma_t, **kwargs):
    """
    :param time: time array in whatver time units
    :param a_1: gaussian rise amplitude scale
    :param peak_time: peak time in whatever units
    :param sigma_t: the sharpness of the Gaussian
    :return: In whatever units set by a_1 
    """
    total = a_1 * np.exp(-(time - peak_time)**2. / (2 * sigma_t ** 2))
    return total

def exponential_powerlaw(time, a_1, alpha_1, alpha_2, tpeak, **kwargs):
    """
    :param time: time array in seconds
    :param a_1: exponential amplitude scale
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak: peak time in seconds
    :param kwargs:
    :return: In whatever units set by a_1
    """
    total = a_1 * (1 - np.exp(-time/tpeak))**alpha_1 * (time/tpeak)**(-alpha_2)
    return total


def two_component_powerlaw(time, a_1, alpha_1,
                           delta_time_one, alpha_2, **kwargs):
    """
    Two component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :return: In whatever units set by a_1
    """
    time_one = delta_time_one
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    w = np.where(time < time_one)
    x = np.where(time > time_one)

    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2

    total = np.concatenate((f1, f2))

    return total


def three_component_powerlaw(time, a_1, alpha_1,
                             delta_time_one, alpha_2,
                             delta_time_two, alpha_3, **kwargs):
    """
    Three component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :return: In whatever units set by a_1
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


def four_component_powerlaw(time, a_1, alpha_1, delta_time_one,
                            alpha_2, delta_time_two,
                            alpha_3, delta_time_three,
                            alpha_4, **kwargs):
    """
    Four component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :return: In whatever units set by a_1
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


def five_component_powerlaw(time, a_1, alpha_1,
                            delta_time_one, alpha_2,
                            delta_time_two, alpha_3,
                            delta_time_three, alpha_4,
                            delta_time_four, alpha_5, **kwargs):
    """
    Five component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    :return: In whatever units set by a_1
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


def six_component_powerlaw(time, a_1, alpha_1,
                           delta_time_one, alpha_2,
                           delta_time_two, alpha_3,
                           delta_time_three, alpha_4,
                           delta_time_four, alpha_5,
                           delta_time_five, alpha_6, **kwargs):
    """
    six component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    :param delta_time_five: time between fourth and fifth power laws
    :param alpha_6: power law decay exponent for sixth power law
    :return: In whatever units set by a_1
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