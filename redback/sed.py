import numpy as np
from redback.constants import *
from redback.utils import nu_to_lambda, lambda_to_nu

def blackbody_to_flux_density(temperature, r_photosphere, dl, frequency):
    """
    A general blackbody_to_flux_density formula

    :param temperature: effective temperature in kelvin
    :param r_photosphere: photosphere radius in cm
    :param dl: luminosity_distance in cm
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number. In source frame
    :return: flux_density
    """
    ## adding units back in to ensure dimensions are correct
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

class CutoffBlackbody(object):
    def __init__(self, time, temperature, luminosity, r_photosphere,
                 frequency, luminosity_distance, cutoff_wavelength, **kwargs):
        """
        Blackbody SED with a cutoff

        :param time: time in source frame in seconds
        :param luminosity: luminosity in cgs
        :param temperature: temperature in kelvin
        :param r_photosphere: photosphere radius in cm
        :param frequency: frequency in Hz - must be a single number or same length as time array
        :param luminosity_distance: dl in cm
        :param kwargs: None
        """
        self.time = time
        self.luminosity = luminosity
        self.temperature = temperature
        self.r_photosphere = r_photosphere
        self.frequency = frequency
        self.luminosity_distance = luminosity_distance
        self.cutoff_wavelength = cutoff_wavelength
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract'

        self.sed = []
        self.flux_density = self.calculate_flux_density()

    def calculate_flux_density(self):
        # Mostly from Mosfit/SEDs
        cutoff_wavelength = self.cutoff_wavelength * angstrom_cgs
        wavelength = nu_to_lambda(self.frequency)
        x_const = planck * speed_of_light * boltzmann_constant
        flux_const = 4 * np.pi * 2*np.pi * planck * speed_of_light**2 * angstrom_cgs

        mask = wavelength < cutoff_wavelength
        sed = np.zeros(len(self.time))

        sed[mask] = flux_const * (self.r_photosphere[mask]**2 / cutoff_wavelength / wavelength[mask] ** 4) \
                    / np.expm1(x_const / wavelength[mask] / self.temperature[mask])
        sed[~mask] = flux_const * (self.r_photosphere[~mask]**2 / wavelength[~mask]**5) \
                     / np.expm1(x_const / wavelength[~mask] / self.temperature[~mask])

        uniq_times = np.unique(self.time)
        tsort = np.argsort(self.time)
        uniq_is = np.searchsorted(self.time, uniq_times, sorter=tsort)
        lu = len(uniq_times)

        norms = self.luminosity[uniq_is] / \
                (flux_const / angstrom_cgs * self.r_photosphere[uniq_is]**2 * self.temperature[uniq_is])

        rp2 = self.r_photosphere[uniq_is]**2
        rp2 = rp2.reshape(lu, 1)
        tp = self.temperature[uniq_is].reshape(lu, 1)
        tp2 = tp**2
        tp3 = tp**3
        nxcs = x_const * np.array(range(1, 11))
                
        f_blue_reds = \
            np.sum(
                np.exp(-nxcs / (cutoff_wavelength * tp)) * (nxcs ** 2 + 2 * (nxcs * cutoff_wavelength * tp + cutoff_wavelength**2 * tp2)) / (nxcs ** 3 * cutoff_wavelength**3) +
                (
                    (6 * tp3 - np.exp(-nxcs / (cutoff_wavelength * tp)) * (nxcs ** 3 + 3 * nxcs ** 2 * cutoff_wavelength * tp + 6 * (nxcs * cutoff_wavelength**2 * tp2 + cutoff_wavelength**3 *tp3)) / cutoff_wavelength**3) / (nxcs ** 4)
                ), 1
            )
        
        norms /= f_blue_reds

        # Apply renormalisation
        sed *= norms[np.searchsorted(uniq_times, self.time)]

        self.sed = sed

        # sed units are erg/s/Angstrom - need to turn them into flux density compatible units
        units = uu.erg / uu.s / uu.Hz / uu.cm**2.
        sed = sed / (4 * np.pi * self.luminosity_distance ** 2) * lambda_to_nu(1.)

        # add units
        flux_density = sed << units

        # convert to mJy
        flux_density = flux_density.to(uu.mJy)
        return flux_density


class Blackbody(object):
    def __init__(self, temperature, r_photosphere, frequency, luminosity_distance, **kwargs):
        """
        Simple Blackbody SED

        :param temperature: effective temperature in kelvin
        :param r_photosphere: photosphere radius in cm
        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number. In source frame
        :param luminosity_distance: luminosity_distance in cm
        :param kwargs: None
        """
        self.temperature = temperature
        self.r_photosphere = r_photosphere
        self.frequency = frequency
        self.luminosity_distance = luminosity_distance
        self.reference = 'It is a blackbody - Do you really need a reference for this?'

        self.flux_density = self.calculate_flux_density()

    def calculate_flux_density(self):
        flux_density = blackbody_to_flux_density(temperature=self.temperature, r_photosphere=self.r_photosphere,
                                                 frequency=self.frequency, dl=self.luminosity_distance)
        return flux_density


class Synchrotron(object):
    def __init__(self, frequency, luminosity_distance,
                 pp,nu_max, source_radius=1e13, f0=1e-26, **kwargs):
        """
        Synchrotron SED

        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number. In source frame
        :param luminosity_distance: luminosity_distance in cm
        :param pp: synchrotron power law slope
        :param nu_max: max frequency
        :param source_radius: emitting source radius
        :param f0: frequency normalization
        :param kwargs: None
        """
        self.frequency = frequency
        self.luminosity_distance = luminosity_distance
        self.pp = pp
        self.nu_max = nu_max
        self.source_radius = source_radius
        self.f0 = f0
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2004rvaa.conf...13H/abstract'
        self.sed = []

        self.flux_density = self.calculate_flux_density()

    def calculate_flux_density(self):
        fmax = self.f0 * self.source_radius**2 * self.nu_max ** 2.5 # for SSA
        mask = self.frequency < self.nu_max
        sed = np.zeros(len(self.frequency))

        # sed units are erg/s/Angstrom - need to turn them into flux density compatible units
        units = uu.erg / uu.s / uu.hz / uu.cm**2

        sed[mask] = self.f0 * self.source_radius**2 * \
                    (self.frequency/self.nu_max)**2.5 * angstrom_cgs / speed_of_light * self.frequency **2
        sed[~mask] = fmax * (self.frequency/self.nu_max)**(-(self.pp - 1.)/2.) \
                     * angstrom_cgs / speed_of_light * self.frequency **2
        self.sed = sed
        sed = sed / (4*np.pi * self.luminosity_distance**2) * lambda_to_nu(1.)

        # add units
        flux_density = sed << units

        # convert to mJy
        flux_density = flux_density.to(uu.mJy)
        return flux_density

class Line(object):
    def __init__(self, time, luminosity, frequency, sed, luminosity_distance, line_wavelength=7.5e3, line_width=500,
                 line_time=50, line_duration=25, line_amplitude=0.3, **kwargs):
        """
        Modifies the input SED by accounting for absorption lines

        :param time: time in source frame
        :param luminosity: luminosity in cgs
        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number. In source frame
        :param sed: instantiated SED class object.
        :param luminosity_distance: luminosity_distance in cm
        :param line_wavelength: line wavelength in angstrom
        :param line_width: line width in angstrom
        :param line_time: line time
        :param line_duration: line duration
        :param line_amplitude: line amplitude
        :param kwargs: None
        """
        self.time = time
        self.luminosity = luminosity
        self.frequency = frequency
        self.SED = sed
        self.luminosity_distance = luminosity_distance
        self.line_wavelength = line_wavelength
        self.line_width = line_width
        self.line_time = line_time
        self.line_duration = line_duration
        self.line_amplitude = line_amplitude


        self.reference = 'https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract'

        self.flux_density = self.calculate_flux_density()

    def calculate_flux_density(self):
        # Mostly from Mosfit/SEDs
        wavelength = nu_to_lambda(self.frequency)
        amplitude = self.line_amplitude * np.exp(-0.5*((self.time - self.line_time)/self.line_duration)**2)

        seds = self.SED.sed * (1 - amplitude)
        amplitude *= self.luminosity / (self.line_width * (2*np.pi)**0.5)

        amp_new = np.exp(-0.5 * ((wavelength - self.line_wavelength) / self.line_width)**2)

        seds += amplitude * amp_new

        # sed units are erg/s/Angstrom - need to turn them into flux density compatible units
        units = uu.erg / uu.s / uu.hz / uu.cm ** 2

        seds = seds / (4 * np.pi * self.luminosity_distance ** 2) * lambda_to_nu(1.)

        # add units
        flux_density = seds << units

        # convert to mJy
        flux_density = flux_density.to(uu.mJy)
        return flux_density