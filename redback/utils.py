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
from astropy.cosmology import FlatLambdaCDM

import redback
from redback.constants import *

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'plot_styles/paper.mplstyle')
plt.style.use(filename)

logger = logging.getLogger('redback')
_bilby_logger = logging.getLogger('bilby')


def find_nearest(array, value):
    """
    Find the nearest value in an array to a given value.

    :param array: array to search
    :param value: value to search for
    :return: array element closest to value and index of that element
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def download_pointing_tables():
    """
    Download the pointing tables from zenodo.
    """
    return logger.info("Pointing tables downloaded and stored in redback/tables")


def sncosmo_bandname_from_band(bands, warning_style='softest'):
    """
    Convert redback data band names to sncosmo compatible band names

    :param bands: List of bands.
    :type bands: list[str]
    :param warning_style: How to handle warnings. 'soft' will raise a warning, 'hard' will raise an error.
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
            if warning_style == 'hard':
                raise KeyError(f"Band {band} is not defined in filters.csv!")
            elif warning_style == 'soft':
                logger.info(e)
                logger.info(f"Band {band} is not defined in filters.csv!")
                res.append('r')
            else:
                res.append('r')
    return np.array(res)


def check_kwargs_validity(kwargs):
    """
    Check the validity of the kwargs passed to a model

    :param kwargs:
    :return:
    """
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
    if unique_frequency is None:
        f2 = all_models_dict[model_2](time=tref, **model_2_dict)
        norm = f2 / f1
        normalisation = namedtuple('normalisation', ['bolometric_luminosity'])(norm)
    else:
        model_2_dict['frequency'] = unique_frequency
        f2 = all_models_dict[model_2](time=tref, **model_2_dict)
        unique_norms = f2 / f1
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


def abmag_to_flambda(mag, lam_eff):
    """
    Converts an AB magnitude to flux density in erg/s/cm^2/Å
    using the effective wavelength for the band.

    In the AB system, the flux density per unit frequency is:
        f_nu = 10**(-0.4*(mag+48.6))  [erg/s/cm^2/Hz]
    To obtain f_lambda [erg/s/cm^2/Å]:

        f_lambda = f_nu * (c / λ^2) / 1e8,

    where λ is the effective wavelength (in cm) and 1e8 converts from per cm to per Å.
    """
    # effective wavelength from Å to cm:
    lam_eff_cm = lam_eff * 1e-8
    f_nu = 10 ** (-0.4 * (mag + 48.6))
    f_lambda = f_nu * (redback.constants.speed_of_light / lam_eff_cm ** 2) / 1e8
    return f_lambda


def flambda_err_from_mag_err(flux, mag_err):
    """
    Compute the error on the flux given an error on the magnitude.
    Using error propagation for f = A*10^(-0.4*mag):
         df/dmag = -0.4 ln(10)* f  =>  σ_f = 0.4 ln(10)* f * σ_mag.
    """
    return 0.4 * np.log(10) * flux * mag_err

def fnu_to_flambda(f_nu, wavelength_A):
    """
    Convert flux density from erg/s/cm^2/Hz to erg/s/cm^2/Angstrom.

    :param f_nu: flux density in erg/s/cm^2/Hz
    :param wavelength_A: wavelength in Angstrom
    :return: flux density in erg/s/cm^2/Angstrom
    """
    return f_nu * speed_of_light * 1e8 / wavelength_A ** 2


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
    flux_density = zeropoint * 10 ** (magnitudes / -2.5)
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


def bandflux_error_from_limiting_mag(fiveSigmaDepth, bandflux_ref):
    """
    Function to compute the error associated with the flux measurement of the
    transient source, computed based on the observation databse determined
    five-sigma depth.

    Parameters:
    -----------
        fiveSigmaDepth: float
            The magnitude at which an exposure would be recorded as having
            an SNR of 5 for this observation.
        bandflux_ref: float
            The total flux that would be transmitted through the chosen
            bandfilter given the chosen reference system.
    Returns:
    --------
        bandflux_error: float
            The error associated with the computed bandflux.
    """
    # Compute the integrated bandflux error
    # Note this is trivial since the five_sigma_depth incorporates the
    # integrated time of the exposures.
    Flux_five_sigma = bandflux_ref * np.power(10.0, -0.4 * fiveSigmaDepth)
    bandflux_error = Flux_five_sigma / 5.0
    return bandflux_error


def convert_apparent_mag_to_absolute(app_magnitude, redshift, **kwargs):
    """
    Convert apparent magnitude to absolute magnitude assuming planck18 cosmology.

    :param app_magnitude: AB/Vega apparent magnitude
    :param redshift: redshift
    :param kwargs: Additional kwargs
    :param cosmology: Cosmology object if not using default
    :return: absolute magnitude
    """
    from astropy.cosmology import Planck18
    import astropy.units as uu

    # Create a cosmology object
    cosmo = kwargs.get('cosmology', Planck18)
    # Calculate the luminosity distance in megaparsecs (pc)
    luminosity_distance = cosmo.luminosity_distance(redshift).to(uu.pc).value

    # Convert to absolute magnitude using the formula
    absolute_magnitude = app_magnitude - 5 * np.log10(luminosity_distance) + 5
    return absolute_magnitude


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
        driver.find_element('id', id_number)
    except NoSuchElementException as e:
        print(e)
        return False
    return True

def bandpass_flux_to_flux_density(flux, flux_err, delta_nu):
    """
    Convert an integrated flux (and its error) measured over a bandpass
    in erg/s/cm² into a flux density (in erg/s/cm²/Hz) and then into millijanskys (mJy).

    Parameters
    ----------
    flux : float or numpy.ndarray
        Integrated flux in erg/s/cm².
    flux_err : float or numpy.ndarray
        Error in the integrated flux in erg/s/cm².
    delta_nu : float
        Effective bandwidth of the filter in Hz.

    Returns
    -------
    f_nu_mJy : float or numpy.ndarray
        Flux density converted to millijanskys (mJy).
    f_nu_err_mJy : float or numpy.ndarray
        Error in the flux density in mJy.

    Notes
    -----
    The conversion from integrated flux F (erg/s/cm²) to flux density f_nu (erg/s/cm²/Hz)
    assumes a known effective bandwidth Δν in Hz:

        f_nu = F / Δν

    Then, converting to mJy is done using the relation
    1 mJy = 1e-3 Jy = 1e-3 * 1e-23 erg/s/cm²/Hz = 1e-26 erg/s/cm²/Hz.
    Therefore, to convert erg/s/cm²/Hz into mJy, divide by 1e-26.
    """
    # Calculate flux density in erg/s/cm²/Hz
    f_nu = flux / delta_nu
    f_nu_err = flux_err / delta_nu

    # Convert to mJy: 1 mJy = 1e-26 erg/s/cm²/Hz
    conversion_factor = 1e-26  # erg/s/cm²/Hz per mJy
    f_nu_mJy = f_nu / conversion_factor
    f_nu_err_mJy = f_nu_err / conversion_factor

    return f_nu_mJy, f_nu_err_mJy


def abmag_to_flux_density_and_error_inmjy(m_AB, sigma_m):
    """
    Convert an AB magnitude and its uncertainty to a flux density in millijanskys (mJy)
    along with the associated error. In the AB system, the flux density (in erg/s/cm²/Hz) is:

        f_nu = 10^(-0.4*(m_AB + 48.60))

    Since 1 Jansky = 1e-23 erg/s/cm²/Hz and 1 mJy = 1e-3 Jansky,
    1 mJy = 1e-26 erg/s/cm²/Hz. Therefore, to convert f_nu to mJy, we do:

        f_nu(mJy) = f_nu / 1e-26

    The uncertainty in the flux density is propagated as:

        sigma_f(mJy) = 0.9210 * f_nu(mJy) * sigma_m

    Parameters
    ----------
    m_AB : float or array_like
        The AB magnitude value(s).
    sigma_m : float or array_like
        The uncertainty in the AB magnitude.

    Returns
    -------
    f_nu_mjy : float or ndarray
        Flux density in millijanskys (mJy).
    sigma_f_mjy : float or ndarray
        Uncertainty in the flux density (mJy).
    """
    # Compute flux density in erg/s/cm^2/Hz
    f_nu = 10 ** (-0.4 * (m_AB + 48.60))
    # Convert flux density to mJy (1 mJy = 1e-26 erg/s/cm^2/Hz)
    f_nu_mjy = f_nu / 1e-26
    # Propagate the uncertainty (σ_f = 0.9210 * f_nu * σ_m, applied after conversion)
    sigma_f_mjy = 0.9210 * f_nu_mjy * sigma_m
    return f_nu_mjy, sigma_f_mjy



def calc_flux_density_error_from_monochromatic_magnitude(magnitude, magnitude_error, reference_flux,
                                                         magnitude_system='AB'):
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
    zeropoint = 10 ** (reference_flux / -2.5)
    return zeropoint


def bandpass_magnitude_to_flux(magnitude, bands):
    """
    Convert magnitude to flux

    :param magnitude: magnitude
    :param bands: bandpass
    :return: flux
    """
    reference_flux = bands_to_reference_flux(bands)
    maggi = 10.0 ** (magnitude / (-2.5))
    flux = maggi * reference_flux
    return flux


def magnitude_error_from_flux_error(bandflux, bandflux_error):
    """
    Function to propagate the flux error to the mag system.

    Parameters:
    -----------
        bandflux: float
            The total flux transmitted through the bandfilter and recorded by
            the detector.
        bandflux_error: float
            The error on the flux measurement from the measured background
            noise, in this case, the five sigma depth.

    Outputs:
    --------
        magnitude_error: float-scalar
            The flux error propagated into the magnitude system.
    """
    # Compute the per-band magnitude errors
    mask1 = bandflux == np.nan
    mask2 = abs(bandflux) <= 1.0e-20
    magnitude_error = abs((2.5 / np.log(10)) * (bandflux_error / bandflux))
    magnitude_error[mask1 | mask2] = np.nan
    return magnitude_error


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

def bands_to_effective_width(bands):
    """
    Converts a list of bands into an array of effective width in Hz

    :param bands: List of bands.
    :type bands: list[str]
    :return: An array of effective width associated with the given bands.
    :rtype: np.ndarray
    """
    if bands is None:
        bands = []
    df = pd.read_csv(f"{dirname}/tables/filters.csv")
    bands_to_freqs = {band: wavelength for band, wavelength in zip(df['bands'], df['effective_width [Hz]'])}
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
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    return driver


def calc_credible_intervals(samples, interval=0.9):
    """
    Calculate credible intervals from samples

    :param samples: samples array
    :param interval: credible interval to calculate
    :return: lower_bound, upper_bound, median
    """
    if not 0 <= interval <= 1:
        raise ValueError
    lower_bound = np.quantile(samples, 0.5 - interval / 2, axis=0)
    upper_bound = np.quantile(samples, 0.5 + interval / 2, axis=0)
    median = np.quantile(samples, 0.5, axis=0)
    return lower_bound, upper_bound, median

def calc_one_dimensional_median_and_error_bar(samples, quantiles=(0.16, 0.84), fmt='.2f'):
    """
    Calculates the median and error bars for a one-dimensional sample array.

    This function computes the median, lower, and upper error bars based on the
    specified quantiles. It returns these values along with a formatted string
    representation.

    :param samples: An array of numerical values representing the sample data. The
        input must not be empty.
    :type samples: list or numpy.ndarray
    :param quantiles: A tuple specifying the lower and upper quantile values. For
        example, (0.16, 0.84) represents the 16th percentile as the lower quantile
        and the 84th percentile as the upper quantile. Default is (0.16, 0.84).
    :type quantiles: tuple
    :param fmt: A format string for rounding the results in the formatted output.
        Default is '.2f'.
    :type fmt: str
    :return: A namedtuple containing the median, lower error bar, upper error bar,
        and a formatted string representation.
    :rtype: MedianErrorBarResult
    :raises ValueError: If the input `samples` array is empty.
    """
    MedianErrorBarResult = namedtuple('MedianErrorBarResult', ['median', 'lower', 'upper', 'string'])

    if len(samples) == 0:
        raise ValueError("Samples array cannot be empty.")

    median = np.median(samples)
    lower_quantile, upper_quantile = np.quantile(samples, quantiles)

    lower = median - lower_quantile
    upper = upper_quantile - median

    formatted_string = rf"${median:{fmt}}_{{-{lower:{fmt}}}}^{{+{upper:{fmt}}}}$"

    return MedianErrorBarResult(median=median, lower=lower, upper=upper, string=formatted_string)



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
    v_array = np.asarray([0.1, 0.2, 0.3, 0.4])
    mass_array = np.asarray([1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1])
    a_array = np.asarray([[2.01, 4.52, 8.16, 16.3], [0.81, 1.9, 3.2, 5.0],
                          [0.56, 1.31, 2.19, 3.0], [.27, .55, .95, 2.0],
                          [0.20, 0.39, 0.65, 0.9]])
    b_array = np.asarray([[0.28, 0.62, 1.19, 2.4], [0.19, 0.28, 0.45, 0.65],
                          [0.17, 0.21, 0.31, 0.45], [0.10, 0.13, 0.15, 0.17],
                          [0.06, 0.11, 0.12, 0.12]])
    d_array = np.asarray([[1.12, 1.39, 1.52, 1.65], [0.86, 1.21, 1.39, 1.5],
                          [0.74, 1.13, 1.32, 1.4], [0.6, 0.9, 1.13, 1.25],
                          [0.63, 0.79, 1.04, 1.5]])
    a_func = RegularGridInterpolator((mass_array, v_array), a_array, bounds_error=False, fill_value=None)
    b_func = RegularGridInterpolator((mass_array, v_array), b_array, bounds_error=False, fill_value=None)
    d_func = RegularGridInterpolator((mass_array, v_array), d_array, bounds_error=False, fill_value=None)

    av = a_func([mej, vej])[0]
    bv = b_func([mej, vej])[0]
    dv = d_func([mej, vej])[0]
    return av, bv, dv


def heatinggrids():
    # Grid of velocity and Ye
    YE_GRID = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], dtype=np.float64)
    V_GRID = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    # Approximant coefficients on the grid
    E0_GRID = np.array([
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.041, 1.041, 1.041, 1.041,
        1.146, 1.000, 1.041, 1.041, 1.041, 1.041,
        1.146, 1.000, 1.000, 1.000, 1.041, 1.041,
        1.301, 1.398, 1.602, 1.580, 1.763, 1.845,
        0.785, 1.255, 1.673, 1.673, 1.874, 1.874,
        0.863, 0.845, 1.212, 1.365, 1.635, 2.176,
        -2.495, -2.495, -2.097, -2.155, -2.046, -1.824,
        -0.699, -0.699, -0.222, 0.176, 0.176, 0.176,
        -0.398, 0.000, 0.301, 0.477, 0.477, 0.477], dtype=np.float64)

    # Reshape GRIDs to a 2D array
    E0_GRID = E0_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    # ALP_GRID
    ALP_GRID = np.array([
        1.37, 1.38, 1.41, 1.41, 1.41, 1.41,
        1.41, 1.38, 1.37, 1.37, 1.37, 1.37,
        1.41, 1.38, 1.37, 1.37, 1.37, 1.37,
        1.36, 1.25, 1.32, 1.32, 1.34, 1.34,
        1.44, 1.40, 1.46, 1.66, 1.60, 1.60,
        1.36, 1.33, 1.33, 1.33, 1.374, 1.374,
        1.40, 1.358, 1.384, 1.384, 1.384, 1.344,
        1.80, 1.80, 2.10, 2.10, 1.90, 1.90,
        8.00, 8.00, 7.00, 7.00, 7.00, 7.00,
        1.40, 1.40, 1.40, 1.60, 1.60, 1.60
    ], dtype=np.float64)

    ALP_GRID = ALP_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    # T0_GRID
    T0_GRID = np.array([
        1.80, 1.40, 1.20, 1.20, 1.20, 1.20,
        1.40, 1.00, 0.85, 0.85, 0.85, 0.85,
        1.00, 0.80, 0.65, 0.65, 0.61, 0.61,
        0.85, 0.60, 0.45, 0.45, 0.45, 0.45,
        0.65, 0.38, 0.22, 0.18, 0.12, 0.095,
        0.540, 0.31, 0.18, 0.13, 0.095, 0.081,
        0.385, 0.235, 0.1, 0.06, 0.035, 0.025,
        26.0, 26.0, 0.4, 0.4, 0.12, -20.0,
        0.20, 0.12, 0.05, 0.03, 0.025, 0.021,
        0.16, 0.08, 0.04, 0.02, 0.018, 0.016
    ], dtype=np.float64)

    T0_GRID = T0_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    # SIG_GRID
    SIG_GRID = np.array([
        0.08, 0.08, 0.095, 0.095, 0.095, 0.095,
        0.10, 0.08, 0.070, 0.070, 0.070, 0.070,
        0.07, 0.08, 0.070, 0.065, 0.070, 0.070,
        0.040, 0.030, 0.05, 0.05, 0.05, 0.050,
        0.05, 0.030, 0.025, 0.045, 0.05, 0.05,
        0.11, 0.04, 0.021, 0.021, 0.017, 0.017,
        0.10, 0.094, 0.068, 0.05, 0.03, 0.01,
        45.0, 45.0, 45.0, 45.0, 25.0, 40.0,
        0.20, 0.12, 0.05, 0.03, 0.025, 0.021,
        0.03, 0.015, 0.007, 0.01, 0.009, 0.007
    ], dtype=np.float64)

    SIG_GRID = SIG_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    # ALP1_GRID
    ALP1_GRID = np.array([
        7.50, 7.50, 7.50, 7.50, 7.50, 7.50,
        9.00, 9.00, 7.50, 7.50, 7.00, 7.00,
        8.00, 8.00, 7.50, 7.50, 7.00, 7.00,
        8.00, 8.00, 7.50, 7.50, 7.00, 7.00,
        8.00, 8.00, 5.00, 7.50, 7.00, 6.50,
        4.5, 3.8, 4.0, 4.0, 4.0, 4.0,
        2.4, 3.8, 3.8, 3.21, 2.91, 3.61,
        -1.55, -1.55, -0.75, -0.75, -2.50, -5.00,
        -1.55, -1.55, -1.55, -1.55, -1.55, -1.55,
        3.00, 3.00, 3.00, 3.00, 3.00, 3.00
    ], dtype=np.float64)

    ALP1_GRID = ALP1_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    # T1_GRID
    T1_GRID = np.array([
        0.040, 0.025, 0.014, 0.010, 0.008, 0.006,
        0.040, 0.035, 0.020, 0.012, 0.010, 0.008,
        0.080, 0.040, 0.020, 0.012, 0.012, 0.009,
        0.080, 0.040, 0.030, 0.018, 0.012, 0.009,
        0.080, 0.060, 0.065, 0.028, 0.020, 0.015,
        0.14, 0.123, 0.089, 0.060, 0.045, 0.031,
        0.264, 0.1, 0.07, 0.055, 0.042, 0.033,
        1.0, 1.0, 1.0, 1.0, 0.02, 0.01,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.04, 0.02, 0.01, 0.002, 0.002, 0.002
    ], dtype=np.float64)

    T1_GRID = T1_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    SIG1_GRID = np.array([0.250, 0.120, 0.045, 0.028, 0.020, 0.015,
                          0.250, 0.060, 0.035, 0.020, 0.016, 0.012,
                          0.170, 0.090, 0.035, 0.020, 0.012, 0.009,
                          0.170, 0.070, 0.035, 0.015, 0.012, 0.009,
                          0.170, 0.070, 0.050, 0.025, 0.020, 0.020,
                          0.065, 0.067, 0.053, 0.032, 0.032, 0.024,
                          0.075, 0.044, 0.03, 0.02, 0.02, 0.014,
                          10.0, 10.0, 10.0, 10.0, 0.02, 0.01,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                          0.01, 0.005, 0.002, 1e-4, 1e-4, 1e-4])

    SIG1_GRID = SIG1_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    C1_GRID = np.array([27.2, 27.8, 28.2, 28.2, 28.2, 28.2,
                        28.0, 27.8, 27.8, 27.8, 27.8, 27.8,
                        27.5, 27.0, 27.8, 27.8, 27.8, 27.8,
                        28.8, 28.1, 27.8, 27.8, 27.5, 27.5,
                        28.5, 28.0, 27.5, 28.5, 29.2, 29.0,
                        25.0, 27.5, 25.8, 20.9, 29.3, 1.0,
                        28.7, 27.0, 28.0, 28.0, 27.4, 25.3,
                        28.5, 29.1, 29.5, 30.1, 30.4, 29.9,
                        20.4, 20.6, 20.8, 20.9, 20.9, 21.0,
                        29.9, 30.1, 30.1, 30.2, 30.3, 30.3])

    C1_GRID = C1_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    TAU1_GRID = np.array([4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                          4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                          4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                          4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                          4.77, 4.77, 4.77, 4.77, 4.07, 4.07,
                          4.77, 4.77, 28.2, 1.03, 0.613, 1.0,
                          3.4, 14.5, 11.4, 14.3, 13.3, 13.3,
                          2.52, 2.52, 2.52, 2.52, 2.52, 2.52,
                          1.02, 1.02, 1.02, 1.02, 1.02, 1.02,
                          0.22, 0.22, 0.22, 0.22, 0.22, 0.22])

    TAU1_GRID = TAU1_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    C2_GRID = np.array([21.5, 21.5, 22.1, 22.1, 22.1, 22.1,
                        22.3, 21.5, 21.5, 21.8, 21.8, 21.8,
                        22.0, 21.5, 21.5, 22.0, 21.8, 21.8,
                        23.5, 22.5, 22.1, 22.0, 22.2, 22.2,
                        22.0, 22.8, 23.0, 23.0, 23.5, 23.5,
                        10.0, 0.0, 0.0, 19.8, 22.0, 21.0,
                        26.2, 14.1, 18.8, 19.1, 23.8, 19.2,
                        25.4, 25.4, 25.8, 26.0, 26.0, 25.8,
                        18.4, 18.4, 18.6, 18.6, 18.6, 18.6,
                        27.8, 28.0, 28.2, 28.2, 28.3, 28.3])

    C2_GRID = C2_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    TAU2_GRID = np.array([4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                          4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                          4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                          4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                          5.62, 5.62, 5.62, 5.62, 4.62, 4.62,
                          5.62, 5.18, 5.18, 34.7, 8.38, 22.6,
                          0.15, 4.49, 95.0, 95.0, 0.95, 146.,
                          0.12, 0.12, 0.12, 0.12, 0.12, 0.14,
                          0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
                          0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

    TAU2_GRID = TAU2_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    C3_GRID = np.array([19.4, 19.8, 20.1, 20.1, 20.1, 20.1,
                        20.0, 19.8, 19.8, 19.8, 19.8, 19.8,
                        19.9, 19.8, 19.8, 19.8, 19.8, 19.8,
                        5.9, 9.8, 23.5, 23.5, 23.5, 23.5,
                        27.3, 26.9, 26.6, 27.4, 25.8, 25.8,
                        27.8, 26.9, 18.9, 25.4, 24.8, 25.8,
                        22.8, 17.9, 18.9, 25.4, 24.8, 25.5,
                        20.6, 20.2, 19.8, 19.2, 19.5, 18.4,
                        12.6, 13.1, 14.1, 14.5, 14.5, 14.5,
                        24.3, 24.2, 24.0, 24.0, 24.0, 23.9])

    C3_GRID = C3_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    TAU3_GRID = np.array([18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
                          18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
                          18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
                          18.2, 18.2, 0.62, 0.62, 0.62, 0.62,
                          0.18, 0.18, 0.18, 0.18, 0.32, 0.32,
                          0.12, 0.18, 50.8, 0.18, 0.32, 0.32,
                          2.4, 51.8, 50.8, 0.18, 0.32, 0.32,
                          3.0, 2.5, 2.4, 2.4, 2.4, 60.4,
                          200., 200., 200., 200., 200., 200.,
                          8.76, 8.76, 8.76, 8.76, 8.76, 8.76])

    TAU3_GRID = TAU3_GRID.reshape((len(V_GRID), len(YE_GRID)), order='F')

    # make interpolants
    E0_interp = RegularGridInterpolator((V_GRID, YE_GRID), E0_GRID, bounds_error=False, fill_value=None)
    ALP_interp = RegularGridInterpolator((V_GRID, YE_GRID), ALP_GRID, bounds_error=False, fill_value=None)
    T0_interp = RegularGridInterpolator((V_GRID, YE_GRID), T0_GRID, bounds_error=False, fill_value=None)
    SIG_interp = RegularGridInterpolator((V_GRID, YE_GRID), SIG_GRID, bounds_error=False, fill_value=None)
    ALP1_interp = RegularGridInterpolator((V_GRID, YE_GRID), ALP1_GRID, bounds_error=False, fill_value=None)
    T1_interp = RegularGridInterpolator((V_GRID, YE_GRID), T1_GRID, bounds_error=False, fill_value=None)
    SIG1_interp = RegularGridInterpolator((V_GRID, YE_GRID), SIG1_GRID, bounds_error=False, fill_value=None)
    C1_interp = RegularGridInterpolator((V_GRID, YE_GRID), C1_GRID, bounds_error=False, fill_value=None)
    TAU1_interp = RegularGridInterpolator((V_GRID, YE_GRID), TAU1_GRID, bounds_error=False, fill_value=None)
    C2_interp = RegularGridInterpolator((V_GRID, YE_GRID), C2_GRID, bounds_error=False, fill_value=None)
    TAU2_interp = RegularGridInterpolator((V_GRID, YE_GRID), TAU2_GRID, bounds_error=False, fill_value=None)
    C3_interp = RegularGridInterpolator((V_GRID, YE_GRID), C3_GRID, bounds_error=False, fill_value=None)
    TAU3_interp = RegularGridInterpolator((V_GRID, YE_GRID), TAU3_GRID, bounds_error=False, fill_value=None)

    interpolators = namedtuple('interpolators', ['E0', 'ALP', 'T0', 'SIG', 'ALP1', 'T1', 'SIG1',
                                                 'C1', 'TAU1', 'C2', 'TAU2', 'C3', 'TAU3'])
    interpolators.E0 = E0_interp
    interpolators.ALP = ALP_interp
    interpolators.T0 = T0_interp
    interpolators.SIG = SIG_interp
    interpolators.ALP1 = ALP1_interp
    interpolators.T1 = T1_interp
    interpolators.SIG1 = SIG1_interp
    interpolators.C1 = C1_interp
    interpolators.TAU1 = TAU1_interp
    interpolators.C2 = C2_interp
    interpolators.TAU2 = TAU2_interp
    interpolators.C3 = C3_interp
    interpolators.TAU3 = TAU3_interp
    return interpolators


def get_heating_terms(ye, vel, **kwargs):
    ints = heatinggrids()
    e0 = ints.E0([vel, ye])[0]
    alp = ints.ALP([vel, ye])[0]
    t0 = ints.T0([vel, ye])[0]
    sig = ints.SIG([vel, ye])[0]
    alp1 = ints.ALP1([vel, ye])[0]
    t1 = ints.T1([vel, ye])[0]
    sig1 = ints.SIG1([vel, ye])[0]
    c1 = ints.C1([vel, ye])[0]
    tau1 = ints.TAU1([vel, ye])[0]
    c2 = ints.C2([vel, ye])[0]
    tau2 = ints.TAU2([vel, ye])[0]
    c3 = ints.C3([vel, ye])[0]
    tau3 = ints.TAU3([vel, ye])[0]
    heating_terms = namedtuple('heating_terms', ['e0', 'alp', 't0', 'sig', 'alp1', 't1', 'sig1', 'c1',
                                                 'tau1', 'c2', 'tau2', 'c3', 'tau3'])

    heating_rate_fudge = kwargs.get('heating_rate_fudge', 1.0)
    heating_terms.e0 = e0 * heating_rate_fudge
    heating_terms.alp = alp * heating_rate_fudge
    heating_terms.t0 = t0 * heating_rate_fudge
    heating_terms.sig = sig * heating_rate_fudge
    heating_terms.alp1 = alp1 * heating_rate_fudge
    heating_terms.t1 = t1 * heating_rate_fudge
    heating_terms.sig1 = sig1 * heating_rate_fudge
    heating_terms.c1 = c1 * heating_rate_fudge
    heating_terms.tau1 = tau1 * heating_rate_fudge
    heating_terms.c2 = c2 * heating_rate_fudge
    heating_terms.tau2 = tau2 * heating_rate_fudge
    heating_terms.c3 = c3 * heating_rate_fudge
    heating_terms.tau3 = tau3 * heating_rate_fudge
    return heating_terms


def _calculate_rosswogkorobkin24_qdot(time_array, ejecta_velocity, electron_fraction):
    import pickle
    import os
    dirname = os.path.dirname(__file__)
    with open(f"{dirname}/tables/qdot_rosswogkorobkin24.pck", 'rb') as file_handle:
        qdot_object = pickle.load(file_handle)
    steps = len(time_array)
    _ej_velocity = np.repeat(ejecta_velocity, steps)
    _ye = np.repeat(electron_fraction, steps)
    full_array = np.array([_ej_velocity, _ye, time_array]).T
    lum_in = qdot_object(full_array)
    return lum_in


def electron_fraction_from_kappa(kappa):
    """
    Uses interpolation from Tanaka+19 to calculate
    the electron fraction based on the temperature independent gray opacity
    :param kappa: temperature independent gray opacity
    :return: electron_fraction
    """

    kappa_array = np.array([35, 32.2, 22.3, 5.60, 5.36, 3.30, 0.96, 0.5])
    ye_array = np.array([0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.5])
    kappa_func = interp1d(kappa_array, y=ye_array, fill_value='extrapolate')
    electron_fraction = kappa_func(kappa)
    return electron_fraction


def kappa_from_electron_fraction(ye):
    """
    Uses interpolation from Tanaka+19 to calculate
    the opacity based on the electron fraction
    :param ye: electron fraction
    :return: electron_fraction
    """
    kappa_array = np.array([35, 32.2, 22.3, 5.60, 5.36, 3.30, 0.96, 0.5])
    ye_array = np.array([0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.5])
    func = interp1d(ye_array, y=kappa_array, fill_value='extrapolate')
    kappa = func(ye)
    return kappa


def lorentz_factor_from_velocity(velocity):
    """
    Calculates the Lorentz factor for a given velocity
    :param velocity: velocity in cm/s
    :return: Lorentz factor
    """
    return 1 / np.sqrt(1 - (velocity / speed_of_light) ** 2)


def velocity_from_lorentz_factor(lorentz_factor):
    """
    Calculates the velocity for a given Lorentz factor
    :param Lorentz_factor: relativistic Lorentz factor
    :return: velocity in cm/s
    """

    return speed_of_light * np.sqrt(1 - 1 / lorentz_factor ** 2)


class UserCosmology(FlatLambdaCDM):
    """
    Dummy cosmology class that behaves like an Astropy cosmology,
    except that the luminosity distance is fixed to the user‐specified value.

    Parameters
    ----------
    dl : astropy.units.Quantity
        The luminosity distance to return (e.g., 100 * u.Mpc).
    **kwargs
        Additional keyword arguments for FlatLambdaCDM (e.g. H0, Om0) if needed.
    """
    def __init__(self, **kwargs):
        self.dl = kwargs.pop("dl", None)

        if 'H0' not in kwargs:
            kwargs['H0'] = 70 * uu.km / uu.s / uu.Mpc
        if 'Om0' not in kwargs:
            kwargs['Om0'] = 0.3

        # Initialize the parent FlatLambdaCDM class.
        super().__init__(**kwargs)

    def __repr__(self):
        # Optionally, override __repr__ so that when printed the class appears as FlatLambdaCDM.
        base_repr = super().__repr__()
        return base_repr.replace(self.__class__.__name__, "FlatLambdaCDM", 1)

    def set_luminosity_distance(self, dl):
        """
        Set (or update) the user-specified luminosity distance.

        Parameters
        ----------
        dl : astropy.units.Quantity
            The new luminosity distance.
        """
        self.dl = dl

    def luminosity_distance(self, redshift):
        """
        Return the user-specified luminosity distance, ignoring the redshift.

        Parameters
        ----------
        redshift : float or array-like
            Redshift (ignored).

        Returns
        -------
        astropy.units.Quantity
            The user-specified luminosity distance.
        """
        return self.dl
