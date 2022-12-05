from typing import Union

import numpy as np
from sncosmo import TimeSeriesSource

from redback.constants import *
from redback.utils import nu_to_lambda, bandpass_magnitude_to_flux, citation_wrapper, lambda_to_nu


def blackbody_to_flux_density(temperature, r_photosphere, dl, frequency):
    """
    A general blackbody_to_flux_density formula

    :param temperature: effective temperature in kelvin
    :param r_photosphere: photosphere radius in cm
    :param dl: luminosity_distance in cm
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                      In source frame
    :return: flux_density
    """
    # adding units back in to ensure dimensions are correct
    frequency = frequency * uu.Hz
    radius = r_photosphere * uu.cm
    dl = dl * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 2 * np.pi * planck * frequency ** 3 * radius ** 2
    denom = dl ** 2 * speed_of_light ** 2
    frac = 1. / (np.expm1((planck * frequency) / (boltzmann_constant * temperature)))
    flux_density = num / denom * frac
    return flux_density


class _SED(object):

    # sed units are erg/s/Angstrom - need to turn them into flux density compatible units
    UNITS = uu.erg / uu.s / uu.Hz / uu.cm**2

    def __init__(self, frequency: Union[np.ndarray, float], luminosity_distance: float, length=1) -> None:
        self.sed = None
        if isinstance(frequency, (float, int)):
            self.frequency = np.array([frequency] * length)
        else:
            self.frequency = frequency
        self.luminosity_distance = luminosity_distance

    @property
    def flux_density(self):
        flux_density = self.sed.copy()
        # add distance units
        flux_density /= (4 * np.pi * self.luminosity_distance ** 2)
        # get rid of Angstrom and normalise to frequency
        flux_density *= nu_to_lambda(self.frequency)
        flux_density /= self.frequency

        # add units
        flux_density = flux_density << self.UNITS

        # convert to mJy
        return flux_density.to(uu.mJy)


class CutoffBlackbody(_SED):

    X_CONST = planck * speed_of_light / boltzmann_constant
    FLUX_CONST = 4 * np.pi * 2 * np.pi * planck * speed_of_light ** 2 * angstrom_cgs

    reference = "https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract"

    def __init__(self, time: np.ndarray, temperature: np.ndarray, luminosity: np.ndarray, r_photosphere: np.ndarray,
                 frequency: Union[float, np.ndarray], luminosity_distance: float, cutoff_wavelength: float,
                 **kwargs: None) -> None:
        """
        Blackbody SED with a cutoff

        :param time: time in source frame
        :param temperature: temperature in kelvin
        :param luminosity: luminosity in cgs
        :param r_photosphere: photosphere radius in cm
        :param frequency: frequency in Hz - must be a single number or same length as time array
        :param luminosity_distance: dl in cm
        :param cutoff_wavelength: cutoff wavelength in Angstrom
        :param kwargs: None
        """
        super(CutoffBlackbody, self).__init__(
            frequency=frequency, luminosity_distance=luminosity_distance, length=len(time))
        self.time = time
        self.unique_times = np.unique(self.time)
        tsort = np.argsort(self.time)
        self.uniq_is = np.searchsorted(self.time, self.unique_times, sorter=tsort)

        self.luminosity = luminosity
        self.temperature = temperature
        self.r_photosphere = r_photosphere
        self.cutoff_wavelength = cutoff_wavelength * angstrom_cgs

        self.norms = None

        self.sed = np.zeros(len(self.time))
        self.calculate_flux_density()

    @property
    def wavelength(self):
        if len(self.frequency) == 1:
            self.frequency = np.ones(len(self.time)) * self.frequency
        wavelength = nu_to_lambda(self.frequency) * angstrom_cgs
        return wavelength

    @property
    def mask(self):
        return self.wavelength < self.cutoff_wavelength

    @property
    def nxcs(self):
        return self.X_CONST * np.array(range(1, 11))

    def _set_sed(self):
        self.sed[self.mask] = \
            self.FLUX_CONST * (self.r_photosphere[self.mask]**2 / self.cutoff_wavelength /
                               self.wavelength[self.mask] ** 4) \
            / np.expm1(self.X_CONST / self.wavelength[self.mask] / self.temperature[self.mask])
        self.sed[~self.mask] = \
            self.FLUX_CONST * (self.r_photosphere[~self.mask]**2 / self.wavelength[~self.mask]**5) \
            / np.expm1(self.X_CONST / self.wavelength[~self.mask] / self.temperature[~self.mask])
        # Apply renormalisation
        self.sed *= self.norms[np.searchsorted(self.unique_times, self.time)]

    def _set_norm(self):
        self.norms = self.luminosity[self.uniq_is] / \
                     (self.FLUX_CONST / angstrom_cgs * self.r_photosphere[self.uniq_is] ** 2 * self.temperature[
                         self.uniq_is])

        tp = self.temperature[self.uniq_is].reshape(len(self.unique_times), 1)
        tp2 = tp ** 2
        tp3 = tp ** 3

        c1 = np.exp(-self.nxcs / (self.cutoff_wavelength * tp))

        term_1 = \
             c1 * (self.nxcs ** 2 + 2 * (self.nxcs * self.cutoff_wavelength * tp + self.cutoff_wavelength ** 2 * tp2)) \
            / (self.nxcs ** 3 * self.cutoff_wavelength ** 3)
        term_2 = \
            (6 * tp3 - c1 *
             (self.nxcs ** 3 + 3 * self.nxcs ** 2 * self.cutoff_wavelength * tp + 6 *
              (self.nxcs * self.cutoff_wavelength ** 2 * tp2 + self.cutoff_wavelength ** 3 * tp3))
             / self.cutoff_wavelength ** 3) / self.nxcs ** 4
        f_blue_reds = np.sum(term_1 + term_2, 1)
        self.norms /= f_blue_reds

    def calculate_flux_density(self):
        self._set_norm()
        self._set_sed()
        return self.flux_density


class Blackbody(object):

    reference = "It is a blackbody - Do you really need a reference for this?"

    def __init__(self, temperature: np.ndarray, r_photosphere: np.ndarray, frequency: np.ndarray,
                 luminosity_distance: float, **kwargs: None) -> None:
        """
        Simple Blackbody SED

        :param temperature: effective temperature in kelvin
        :param r_photosphere: photosphere radius in cm
        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                          In source frame
        :param luminosity_distance: luminosity_distance in cm
        :param kwargs: None
        """
        self.temperature = temperature
        self.r_photosphere = r_photosphere
        self.frequency = frequency
        self.luminosity_distance = luminosity_distance

        self.flux_density = self.calculate_flux_density()

    def calculate_flux_density(self):
        self.flux_density = blackbody_to_flux_density(
            temperature=self.temperature, r_photosphere=self.r_photosphere,
            frequency=self.frequency, dl=self.luminosity_distance)
        return self.flux_density


class Synchrotron(_SED):

    reference = "https://ui.adsabs.harvard.edu/abs/2004rvaa.conf...13H/abstract"

    def __init__(self, frequency: Union[np.ndarray, float], luminosity_distance: float, pp: float, nu_max: float,
                 source_radius: float = 1e13, f0: float = 1e-26, **kwargs: None) -> None:
        """
        Synchrotron SED

        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                          In source frame.
        :param luminosity_distance: luminosity_distance in cm
        :param pp: synchrotron power law slope
        :param nu_max: max frequency
        :param source_radius: emitting source radius
        :param f0: frequency normalization
        :param kwargs: None
        """
        super(Synchrotron, self).__init__(frequency=frequency, luminosity_distance=luminosity_distance)
        self.pp = pp
        self.nu_max = nu_max
        self.source_radius = source_radius
        self.f0 = f0
        self.sed = None

        self.calculate_flux_density()

    @property
    def f_max(self):
        return self.f0 * self.source_radius**2 * self.nu_max ** 2.5  # for SSA

    @property
    def mask(self):
        return self.frequency < self.nu_max

    def _set_sed(self):
        self.sed = np.zeros(len(self.frequency))
        self.sed[self.mask] = \
            self.f0 * self.source_radius**2 * (self.frequency[self.mask]/self.nu_max) ** 2.5 \
            * angstrom_cgs / speed_of_light * self.frequency[self.mask] ** 2
        self.sed[~self.mask] = \
            self.f_max * (self.frequency[~self.mask]/self.nu_max)**(-(self.pp - 1.)/2.) \
            * angstrom_cgs / speed_of_light * self.frequency[~self.mask] ** 2

    def calculate_flux_density(self):
        self._set_sed()
        return self.flux_density


class Line(_SED):

    reference = "https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract"

    def __init__(self, time: np.ndarray, luminosity: np.ndarray, frequency: Union[np.ndarray, float],
                 sed: Union[_SED, Blackbody], luminosity_distance: float, line_wavelength: float = 7.5e3,
                 line_width: float = 500, line_time: float = 50, line_duration: float = 25,
                 line_amplitude: float = 0.3, **kwargs: None) -> None:
        """
        Modifies the input SED by accounting for absorption lines

        :param time: time in source frame
        :param luminosity: luminosity in cgs
        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                          In source frame.
        :param sed: instantiated SED class object.
        :param luminosity_distance: luminosity_distance in cm
        :param line_wavelength: line wavelength in angstrom
        :param line_width: line width in angstrom
        :param line_time: line time
        :param line_duration: line duration
        :param line_amplitude: line amplitude
        :param kwargs: None
        """
        super(Line, self).__init__(frequency=frequency, luminosity_distance=luminosity_distance, length=len(time))
        self.time = time
        self.luminosity = luminosity
        self.SED = sed
        self.sed = None
        self.line_wavelength = line_wavelength
        self.line_width = line_width
        self.line_time = line_time
        self.line_duration = line_duration
        self.line_amplitude = line_amplitude

        self.calculate_flux_density()

    @property
    def wavelength(self):
        return nu_to_lambda(self.frequency)

    def calculate_flux_density(self):
        # Mostly from Mosfit/SEDs
        self._set_sed()
        return self.flux_density

    def _set_sed(self):
        amplitude = self.line_amplitude * np.exp(-0.5 * ((self.time - self.line_time) / self.line_duration) ** 2)
        self.sed = self.SED.sed * (1 - amplitude)

        amplitude *= self.luminosity / (self.line_width * (2 * np.pi) ** 0.5)

        amp_new = np.exp(-0.5 * ((self.wavelength - self.line_wavelength) / self.line_width) ** 2)
        self.sed += amplitude * amp_new

def get_correct_output_format_from_spectra(time, time_eval, spectra, frequency_array, **kwargs):
    """
    Use SNcosmo to get the bandpass flux or magnitude in AB from spectra at given times.

    :param time: times in observer frame in days to evaluate the model on
    :param time_eval: times in observer frame where spectra are evaluated. A densely sampled array for accuracy
    :param bands: band array - must be same length as time array or a single band
    :param spectra: spectra in mJy evaluated at all times and frequencies; shape (len(times), len(frequency_array))
    :param frequency_array: frequency array in Angstrom in observer frame
    :param kwargs: Additional parameters
    :param output_format: 'flux', 'magnitude', 'sncosmo_source', 'flux_density'
    :return: flux, magnitude or SNcosmo TimeSeries Source depending on output format kwarg
    """
    source = TimeSeriesSource(phase=time_eval, wave=frequency_array, flux=spectra)
    if kwargs['output_format'] == 'flux':
        bands = kwargs['bands']
        magnitude = source.bandmag(phase=time, band=bands, magsys='ab')
        return bandpass_magnitude_to_flux(magnitude=magnitude, bands=bands)
    elif kwargs['output_format'] == 'magnitude':
        bands = kwargs['bands']
        magnitude = source.bandmag(phase=time, band=bands, magsys='ab')
        return magnitude
    elif kwargs['output_format'] == 'sncosmo_source':
        return source
    else:
        raise ValueError("Output format must be 'flux', 'magnitude', 'sncosmo_source', or 'flux_density'")