import contextlib
import logging
import os
from inspect import getmembers, isfunction

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import astropy.units as uu

import bilby

import redback

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'paper.mplstyle')
plt.style.use(filename)

logger = logging.getLogger('redback')
_bilby_logger = logging.getLogger('bilby')

def get_filter_frequencies():
    pass

def calc_ABmag_from_fluxdensity(fluxdensity):
    return (fluxdensity * uu.mJy).to(uu.ABmag)

def calc_fluxdensity_from_ABmag(magnitudes):
    return (magnitudes * uu.ABmag).to(uu.mJy)

def check_element(driver, id_number):
    """
    checks that an element exists on a website, and provides an exception
    """
    try:
        driver.find_element_by_id(id_number)
    except NoSuchElementException:
        return False
    return True

def calc_flux_density_error(magnitude, magnitude_error, reference_flux, magnitude_system = 'AB'):
    if magnitude_system == 'AB':
        reference_flux = 3631
    prefactor = np.log(10)/(-2.5)
    dfdm = 1000 * prefactor * reference_flux * np.exp(prefactor * magnitude)
    flux_err = ((dfdm * magnitude_error)**2)**0.5
    return flux_err

def calc_flux_from_mag(magnitude, reference_flux, magnitude_system = 'AB'):
    if magnitude_system == 'AB':
        reference_flux = 3631
    flux = 10 ** (magnitude/-2.5) * reference_flux #Jansky
    return 1000*flux #return in mJy

def fetch_driver():
    # open the webdriver
    return webdriver.PhantomJS()


def calc_confidence_intervals(samples):
    lower_bound = np.quantile(samples, 0.05, axis=0)
    upper_bound = np.quantile(samples, 0.95, axis=0)
    median = np.quantile(samples, 0.5, axis=0)
    return lower_bound, upper_bound, median


def kde_scipy(x, bandwidth=0.05, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde


def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def find_path(path):
    if path == 'default':
        return os.path.join(dirname, '../data/GRBData')
    else:
        return path


def setup_logger(outdir='.', label=None, log_level='INFO'):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ==========
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
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

all_models_dict = []
modules_dict = {}
def get_functions_dict(module):
    _functions_list = [o for o in getmembers(module) if isfunction(o[1])]
    _functions_dict = {f[0]: f[1] for f in _functions_list}
    all_models_dict.append(_functions_dict)
    modules_dict[module.__name__.split('.')[-1]] = _functions_dict
    return all_models_dict, modules_dict

