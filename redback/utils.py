import contextlib
import logging
import os
from collections import namedtuple
from inspect import getmembers, isfunction
from pathlib import Path

from astropy.time import Time
import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import gaussian_kde
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

import redback
from redback.constants import *


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'plot_styles/paper.mplstyle')
plt.style.use(filename)

logger = logging.getLogger('redback')
_bilby_logger = logging.getLogger('bilby')

def sncosmo_bandname_from_band(bands):
    """
    Convert redback data band names to sncosmo compatible band names

    :param bands: List of bands.
    :type bands: list[str]
    :return: An array of sncosmo compatible bandnames associated with the given bands.
    :rtype: np.ndarray
    """
    if bands is None:
        bands = []
    if isinstance(bands, str):
        bands = [bands]
    df = pd.read_csv(f"{dirname}/tables/filters.csv")
    bands_to_flux = {band: wavelength for band, wavelength in zip(df['bands'], df['sncosmo_name'])}
    res = []
    for band in bands:
        try:
            res.append(bands_to_flux[band])
        except KeyError as e:
            logger.info(e)
            raise KeyError(f"Band {band} is not defined in filters.csv!")
    return np.array(res)

def check_kwargs_validity(kwargs):
    if kwargs == None:
        logger.info("No kwargs passed to function")
        return kwargs
    if 'output_format' not in kwargs.keys():
        raise ValueError("output_format must be specified")
    else:
        output_format = kwargs['output_format']
    match = ['frequency', 'bands']
    if any(x in kwargs.keys() for x in match):
        pass
    else:
        raise ValueError("frequency or bands must be specified in model_kwargs")

    if output_format == 'flux_density':
        if 'frequency' not in kwargs.keys():
            kwargs['frequency'] = redback.utils.bands_to_frequency(kwargs['bands'])

    if output_format in ['flux', 'magnitude']:
        if 'bands' not in kwargs.keys():
            kwargs['bands'] = redback.utils.frequency_to_bandname(kwargs['frequency'])

    if output_format == 'spectra':
        kwargs['frequency_array'] = kwargs.get('frequency_array', np.linspace(100, 20000, 100))
    return kwargs

def citation_wrapper(r):
    """
    Wrapper for citation function to allow functions to have a citation attribute
    :param r: proxy argument
    :return: wrapped function
    """
    def wrapper(f):
        f.citation = r
        return f
    return wrapper

def calc_tfb(binding_energy_const, mbh_6, stellar_mass):
    """
    Calculate the fall back timescale for a SMBH disrupting a stellar mass object
    :param binding_energy_const:
    :param mbh_6: SMBH mass in solar masses
    :param stellar_mass: stellar mass in solar masses
    :return: fall back time in seconds
    """
    tfb = 58. * (3600. * 24.) * (mbh_6 ** (0.5)) * (stellar_mass ** (0.2)) * ((binding_energy_const / 0.8) ** (-1.5))
    return tfb

def calculate_normalisation(unique_frequency, model_1, model_2, tref, model_1_dict, model_2_dict):
    """
    Calculate the normalisation for smoothly joining two models together at a reference time.

    :param unique_frequency: An array of unique frequencies. Can be None in which case we assume there is only one normalisation.
    :param model_1: must be redback model with a normalisation parameter
    :param model_2: any redback model
    :param tref: time which transition from model_1 to model_2 takes place
    :param model_1_dict: dictionary of parameters and values for model 1
    :param model_2: dictionary of parameters and values for model 1
    :return: normalisation, namedtuple corresponding to the normalisation for the specific frequency.
    Could be bolometric luminosity, magnitude, or frequency
    """
    from redback.model_library import all_models_dict
    f1 = all_models_dict[model_1](time=tref, a_1=1, **model_1_dict)
    if unique_frequency == None:
        f2 = all_models_dict[model_2](time=tref, **model_2_dict)
        norm = f2/f1
        normalisation = namedtuple('normalisation', ['bolometric_luminosity'])(norm)
    else:
        model_2_dict['frequency'] = unique_frequency
        f2 = all_models_dict[model_2](time=tref, **model_2_dict)
        unique_norms = f2/f1
        dd = dict(zip(unique_frequency, unique_norms))
        normalisation = namedtuple('normalisation', dd.keys())(*dd.values())
    return normalisation

def get_csm_properties(nn, eta):
    """
    Calculate CSM properties for CSM interacting models

    :param nn: csm norm
    :param eta: csm density profile exponent
    :return: csm_properties named tuple
    """
    csm_properties = namedtuple('csm_properties', ['AA', 'Bf', 'Br'])
    filepath = f"{dirname}/tables/csm_table.txt"
    ns, ss, bfs, brs, aas = np.loadtxt(filepath, delimiter=',', unpack=True)
    bfs = np.reshape(bfs, (10, 30)).T
    brs = np.reshape(brs, (10, 30)).T
    aas = np.reshape(aas, (10, 30)).T
    ns = np.unique(ns)
    ss = np.unique(ss)
    bf_func = RegularGridInterpolator((ss, ns), bfs)
    br_func = RegularGridInterpolator((ss, ns), brs)
    aa_func = RegularGridInterpolator((ss, ns), aas)

    Bf = bf_func([nn, eta])[0]
    Br = br_func([nn, eta])[0]
    AA = aa_func([nn, eta])[0]

    csm_properties.AA = AA
    csm_properties.Bf = Bf
    csm_properties.Br = Br
    return csm_properties

def lambda_to_nu(wavelength):
    """
    :param wavelength: wavelength in Angstrom
    :return: frequency in Hertz
    """
    return speed_of_light_si / (wavelength * 1.e-10)


def nu_to_lambda(frequency):
    """
    :param frequency: frequency in Hertz
    :return: wavelength in Angstrom
    """
    return 1.e10 * (speed_of_light_si / frequency)

def calc_kcorrected_properties(frequency, redshift, time):
    """
    Perform k-correction

    :param frequency: observer frame frequency
    :param redshift: source redshift
    :param time: observer frame time
    :return: k-corrected frequency and source frame time
    """
    time = time / (1 + redshift)
    frequency = frequency * (1 + redshift)
    return frequency, time


def mjd_to_jd(mjd):
    """
    Convert MJD to JD

    :param mjd: mjd time
    :return: JD time
    """
    return Time(mjd, format="mjd").jd


def jd_to_mjd(jd):
    """
    Convert JD to MJD

    :param jd: jd time
    :return: MJD time
    """
    return Time(jd, format="jd").mjd


def jd_to_date(jd):
    """
    Convert JD to date

    :param jd: jd time
    :return: date
    """
    year, month, day, _, _, _ = Time(jd, format="jd").to_value("ymdhms")
    return year, month, day


def mjd_to_date(mjd):
    """
    Convert MJD to date

    :param mjd: mjd time
    :return: data
    """
    year, month, day, _, _, _ = Time(mjd, format="mjd").to_value("ymdhms")
    return year, month, day


def date_to_jd(year, month, day):
    """
    Convert date to JD

    :param year:
    :param month:
    :param day:
    :return: JD time
    """
    return Time(dict(year=year, month=month, day=day), format="ymdhms").jd


def date_to_mjd(year, month, day):
    """
    Convert date to MJD

    :param year:
    :param month:
    :param day:
    :return: MJD time
    """
    return Time(dict(year=year, month=month, day=day), format="ymdhms").mjd

def deceleration_timescale(e0, g0, n0):
    """
    Calculate the deceleration timescale for an afterglow

    :param e0: kinetic energy of afterglow
    :param g0: lorentz factor of afterglow
    :param n0: nism number density
    :return: peak time in seconds
    """
    e0 = e0
    gamma0 = g0
    nism = n0
    denom = 32 * np.pi * gamma0 ** 8 * nism * proton_mass * speed_of_light ** 5
    num = 3 * e0
    t_peak = (num / denom) ** (1. / 3.)
    return t_peak

def calc_flux_density_from_ABmag(magnitudes):
    """
    Calculate flux density from AB magnitude assuming monochromatic AB filter

    :param magnitudes:
    :return: flux density
    """
    return (magnitudes * uu.ABmag).to(uu.mJy)

def calc_ABmag_from_flux_density(fluxdensity):
    """
    Calculate AB magnitude from flux density assuming monochromatic AB filter

    :param fluxdensity:
    :return: AB magnitude
    """
    return (fluxdensity * uu.mJy).to(uu.ABmag)

def calc_flux_density_from_vegamag(magnitudes, zeropoint):
    """
    Calculate flux density from Vega magnitude assuming Vega filter

    :param magnitudes:
    :param zeropoint: Vega zeropoint for a given filter in Jy
    :return: flux density in mJy
    """
    zeropoint = zeropoint * 1000
    flux_density = zeropoint * 10 ** (magnitudes/-2.5)
    return flux_density

def calc_vegamag_from_flux_density(fluxdensity, zeropoint):
    """
    Calculate Vega magnitude from flux density assuming Vega filter

    :param fluxdensity: in mJy
    :param zeropoint: Vega zeropoint for a given filter in Jy
    :return: Vega magnitude
    """
    zeropoint = zeropoint * 1000
    magnitude = -2.5 * np.log10(fluxdensity / zeropoint)
    return magnitude

def convert_absolute_mag_to_apparent(magnitude, distance):
    """
    Convert absolute magnitude to apparent

    :param magnitude: AB absolute magnitude
    :param distance: Distance in parsecs
    """
    app_mag = magnitude + 5 * (np.log10(distance) - 1)
    return app_mag


def check_element(driver, id_number):
    """
    checks that an element exists on a website, and provides an exception
    """
    try:
        driver.find_element_by_id(id_number)
    except NoSuchElementException as e:
        print(e)
        return False
    return True

def calc_flux_density_error_from_monochromatic_magnitude(magnitude, magnitude_error, reference_flux, magnitude_system='AB'):
    """
    Calculate flux density error from magnitude error

    :param magnitude: magnitude
    :param magnitude_error: magnitude error
    :param reference_flux: reference flux density
    :param magnitude_system: magnitude system
    :return: Flux density error
    """
    if magnitude_system == 'AB':
        reference_flux = 3631
    prefactor = np.log(10) / (-2.5)
    dfdm = 1000 * prefactor * reference_flux * np.exp(prefactor * magnitude)
    flux_err = ((dfdm * magnitude_error) ** 2) ** 0.5
    return flux_err

def calc_flux_error_from_magnitude(magnitude, magnitude_error, reference_flux):
    """
    Calculate flux error from magnitude error

    :param magnitude: magnitude
    :param magnitude_error: magnitude error
    :param reference_flux: reference flux density
    :return: Flux error
    """
    prefactor = np.log(10) / (-2.5)
    dfdm = prefactor * reference_flux * np.exp(prefactor * magnitude)
    flux_err = ((dfdm * magnitude_error) ** 2) ** 0.5
    return flux_err

def bands_to_zeropoint(bands):
    """
    Bands to zero point

    :param bands: list of bands
    :return: zeropoint for magnitude to flux density calculation
    """
    reference_flux = bands_to_reference_flux(bands)
    zeropoint = 10**(reference_flux/-2.5)
    return zeropoint

def bandpass_magnitude_to_flux(magnitude, bands):
    """
    Convert magnitude to flux

    :param magnitude: magnitude
    :param bands: bandpass
    :return: flux
    """
    reference_flux = bands_to_reference_flux(bands)
    maggi = 10.0**(magnitude / (-2.5))
    flux = maggi * reference_flux
    return flux

def bandpass_flux_to_magnitude(flux, bands):
    """
    Convert flux to magnitude

    :param flux: flux
    :param bands: bandpass
    :return: magnitude
    """
    reference_flux = bands_to_reference_flux(bands)
    maggi = flux / reference_flux
    magnitude = -2.5 * np.log10(maggi)
    return magnitude

def bands_to_reference_flux(bands):
    """
    Looks up the reference flux for a given band from the filters table.

    :param bands: List of bands.
    :type bands: list[str]
    :return: An array of reference flux associated with the given bands.
    :rtype: np.ndarray
    """
    if bands is None:
        bands = []
    if isinstance(bands, str):
        bands = [bands]
    df = pd.read_csv(f"{dirname}/tables/filters.csv")
    bands_to_flux = {band: wavelength for band, wavelength in zip(df['bands'], df['reference_flux'])}
    res = []
    for band in bands:
        try:
            res.append(bands_to_flux[band])
        except KeyError as e:
            logger.info(e)
            raise KeyError(f"Band {band} is not defined in filters.csv!")
    return np.array(res)

def bands_to_frequency(bands):
    """
    Converts a list of bands into an array of frequency in Hz

    :param bands: List of bands.
    :type bands: list[str]
    :return: An array of frequency associated with the given bands.
    :rtype: np.ndarray
    """
    if bands is None:
        bands = []
    df = pd.read_csv(f"{dirname}/tables/filters.csv")
    bands_to_freqs = {band: wavelength for band, wavelength in zip(df['bands'], df['wavelength [Hz]'])}
    res = []
    for band in bands:
        try:
            res.append(bands_to_freqs[band])
        except KeyError as e:
            logger.info(e)
            raise KeyError(f"Band {band} is not defined in filters.csv!")
    return np.array(res)

def frequency_to_bandname(frequency):
    """
    Converts a list of frequencies into an array corresponding band names

    :param frequency: List of bands.
    :type frequency: list[str]
    :return: An array of bandnames associated with the given frequency.
    :rtype: np.ndarray
    """
    if frequency is None:
        frequency = []
    df = pd.read_csv(f"{dirname}/tables/filters.csv")
    freqs_to_bands = {wavelength: band for wavelength, band in zip(df['wavelength [Hz]'], df['bands'])}
    res = []
    for freq in frequency:
        try:
            res.append(freqs_to_bands[freq])
        except KeyError as e:
            logger.info(e)
            raise KeyError(f"Wavelength {freq} is not defined in filters.csv!")
    return np.array(res)

def fetch_driver():
    # open the webdriver
    return webdriver.PhantomJS()


def calc_credible_intervals(samples, interval=0.9):
    """
    Calculate credible intervals from samples

    :param samples: samples array
    :param interval: credible interval to calculate
    :return: lower_bound, upper_bound, median
    """
    if not 0 <= interval <= 1:
        raise ValueError
    lower_bound = np.quantile(samples, 0.5 - interval/2, axis=0)
    upper_bound = np.quantile(samples, 0.5 + interval/2, axis=0)
    median = np.quantile(samples, 0.5, axis=0)
    return lower_bound, upper_bound, median


def calc_one_dimensional_median_and_error_bar(samples, quantiles=(0.16, 0.84), fmt='.2f'):
    """
    Calculate the median and error bar of a one dimensional array of samples

    :param samples: samples array
    :param quantiles: quantiles to calculate
    :param fmt: latex fmt
    :return: summary named tuple
    """
    summary = namedtuple('summary', ['median', 'lower', 'upper', 'string'])

    if len(quantiles) != 2:
        raise ValueError("quantiles must be of length 2")

    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(samples, quants_to_compute * 100)
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]

    fmt = "{{0:{0}}}".format(fmt).format
    string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    summary.string = string_template.format(
        fmt(summary.median), fmt(summary.minus), fmt(summary.plus))
    return summary


def kde_scipy(x, bandwidth=0.05, **kwargs):
    """
    Kernel Density Estimation with Scipy

    :param x: samples
    :param bandwidth: bandwidth of the kernel
    :param kwargs: Any extra kwargs passed to scipy.kde
    :return: gaussian kde object
    """
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde


def cdf(x, plot=True, *args, **kwargs):
    """
    Cumulative distribution function

    :param x: samples
    :param plot: whether to plot the cdf
    :param args: extra args passed to plt.plot
    :param kwargs: extra kwargs passed to plt.plot
    :return: x, y or plot
    """
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def bin_ttes(ttes, bin_size):
    """
    Bin TimeTaggedEvents into bins of size bin_size

    :param ttes: time tagged events
    :param bin_size: bin sizes
    :return: times and counts in bins
    """
    counts, bin_edges = np.histogram(ttes, np.arange(ttes[0], ttes[-1], bin_size))
    times = np.array([bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)])
    return times, counts


def find_path(path):
    """
    Find the path of some data in the package

    :param path:
    :return:
    """
    if path == 'default':
        return os.path.join(dirname, '../data/GRBData')
    else:
        return path


def setup_logger(outdir='.', label=None, log_level='INFO'):
    """
    Setup logging output: call at the start of the script to use

    :param outdir: If supplied, write the logging output to outdir/label.log
    :type outdir: str
    :param label: If supplied, write the logging output to outdir/label.log
    :type label: str
    :param log_level:
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
        (Default value = 'INFO')
    """
    log_file = f'{outdir}/{label}.log'
    with contextlib.suppress(FileNotFoundError):
        os.remove(log_file)  # remove existing log file with the same name instead of appending to it
    bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level=log_level, print_version=True)

    level = _bilby_logger.level
    logger.setLevel(level)

    if not any([type(h) == logging.StreamHandler for h in logger.handlers]):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if not any([type(h) == logging.FileHandler for h in logger.handlers]):
        if label is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info(f'Running redback version: {redback.__version__}')


class MetaDataAccessor(object):
    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. Allows easy access to meta_data dict entries
    """

    def __init__(self, property_name, default=None):
        self.property_name = property_name
        self.container_instance_name = 'meta_data'
        self.default = default

    def __get__(self, instance, owner):
        try:
            return getattr(instance, self.container_instance_name)[self.property_name]
        except KeyError:
            return self.default

    def __set__(self, instance, value):
        getattr(instance, self.container_instance_name)[self.property_name] = value


class DataModeSwitch(object):
    """
    Descriptor class to access boolean data_mode switches.
    """

    def __init__(self, data_mode):
        self.data_mode = data_mode

    def __get__(self, instance, owner):
        return instance.data_mode == self.data_mode

    def __set__(self, instance, value):
        if value:
            instance.data_mode = self.data_mode
        else:
            instance.data_mode = None


class KwargsAccessorWithDefault(object):
    """
    Descriptor class to access a kwarg dictionary with defaults.
    """

    def __init__(self, kwarg, default=None):
        self.kwarg = kwarg
        self.default = default

    def __get__(self, instance, owner):
        return instance.kwargs.get(self.kwarg, self.default)

    def __set__(self, instance, value):
        instance.kwargs[self.kwarg] = value


def get_functions_dict(module):
    models_dict = {}
    _functions_list = [o for o in getmembers(module) if isfunction(o[1])]
    _functions_dict = {f[0]: f[1] for f in _functions_list}
    models_dict[module.__name__.split('.')[-1]] = _functions_dict
    return models_dict


def interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej):
    """
    Uses Barnes+2016 and interpolation to calculate the r-process thermalisation efficiency
    depending on the input mass and velocity
    :param mej: ejecta mass in solar masses
    :param vej: initial ejecta velocity as a fraction of speed of light
    :return: av, bv, dv constants in the thermalisation efficiency equation Eq 25 in Metzger 2017
    """
    v_array = np.array([0.1, 0.2, 0.3])
    mass_array = np.array([1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2])
    a_array = np.asarray([[2.01, 4.52, 8.16], [0.81, 1.9, 3.2],
                     [0.56, 1.31, 2.19], [.27, .55, .95]])
    b_array = np.asarray([[0.28, 0.62, 1.19], [0.19, 0.28, 0.45],
                     [0.17, 0.21, 0.31], [0.10, 0.13, 0.15]])
    d_array = np.asarray([[1.12, 1.39, 1.52], [0.86, 1.21, 1.39],
                     [0.74, 1.13, 1.32], [0.6, 0.9, 1.13]])
    a_func = RegularGridInterpolator((mass_array, v_array), a_array, bounds_error=False, fill_value=None)
    b_func = RegularGridInterpolator((mass_array, v_array), b_array, bounds_error=False, fill_value=None)
    d_func = RegularGridInterpolator((mass_array, v_array), d_array, bounds_error=False, fill_value=None)

    av = a_func([mej, vej])[0]
    bv = b_func([mej, vej])[0]
    dv = d_func([mej, vej])[0]
    return av, bv, dv


def electron_fraction_from_kappa(kappa):
    """
    Uses interpolation from Tanaka+19 to calculate
    the electron fraction based on the temperature independent gray opacity
    :param kappa: temperature independent gray opacity
    :return: electron_fraction
    """

    kappa_array = np.array([1, 3, 5, 20, 30])
    ye_array = np.array([0.4,0.35,0.25,0.2, 0.1])
    kappa_func = interp1d(kappa_array, y=ye_array)
    electron_fraction = kappa_func(kappa)
    return electron_fraction