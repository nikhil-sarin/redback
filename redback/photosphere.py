# This is mostly from mosfit/photospheres but rewritten considerably.
from typing import Any, Union

import numpy as np
from redback.constants import *

class CocoonPhotosphere(object):

    DIFFUSION_CONSTANT = solar_mass / (4*np.pi*speed_of_light*km_cgs)
    RADIUS_CONSTANT = km_cgs * day_to_s
    STEF_CONSTANT = 4 * np.pi * sigma_sb
    reference = 'https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract'
    def __init__(self, time: np.ndarray, luminosity: np.ndarray, tau_diff: float,
                 t_thin: float, vej:np.ndarray, nn: Union[float, int], **kwargs: None) -> None:
        """
        Cocoon Photosphere

        :param time: source frame time in days
        :type time: numpy.ndarray
        :param luminosity: luminosity in ergs/s
        :type luminosity: numpy.ndarray
        :param tau_diff: diffusion time in days
        :type tau_diff: float
        :param t_thin: time to become optically thin in days
        :type t_thin: float
        :param vej: ejecta velocity
        :type vej: numpy.ndarray
        :param nn: density power law index
        :type nn: Union[float, int]
        :param kwargs: Additional keyword arguments
        """
        self.time = time
        self.luminosity = luminosity
        self.tau_diff = tau_diff
        self.t_thin = t_thin
        self.r_photosphere = np.array([])
        self.photosphere_temperature = np.array([])
        self.vej = vej
        self.nn = nn
        self.calculate_photosphere_properties()

    @property
    def set_vphoto(self):
        return self.vej * (self.time / self.t_thin) ** (-2./(self.nn + 3))
    def calculate_r_photosphere(self) -> None:
        self.r_photosphere = self.RADIUS_CONSTANT * self.set_vphoto * self.time

    def calculate_photosphere_temperature(self) -> None:
        self.photosphere_temperature = (self.luminosity / (self.STEF_CONSTANT * self.r_photosphere ** 2))**(0.25)

    def calculate_photosphere_properties(self) -> tuple:
        self.calculate_r_photosphere()
        self.calculate_photosphere_temperature()
        return self.photosphere_temperature, self.r_photosphere

class TemperatureFloor(object):

    RADIUS_CONSTANT = km_cgs * day_to_s
    STEF_CONSTANT = 4 * np.pi * sigma_sb
    reference = "https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract"

    def __init__(self, time: np.ndarray, luminosity: np.ndarray,
                 vej: np.ndarray, temperature_floor: Union[float, int], **kwargs: None) -> None:
        """
        Photosphere with a floor temperature and effective blackbody otherwise

        :param time: source frame time in days
        :type time: numpy.ndarray
        :param luminosity: luminosity
        :type luminosity: numpy.ndarray
        :param vej: ejecta velocity in km/s
        :type vej: numpy.ndarray
        :param temperature_floor: floor temperature in kelvin
        :type temperature_floor: Union[float, int]
        """
        self.time = time
        self.luminosity = luminosity
        self.v_ejecta = vej
        self.temperature_floor = temperature_floor
        self.r_photosphere = np.array([])
        self.photosphere_temperature = np.array([])
        self.radius_squared = np.array([])
        self.rec_radius_squared = np.array([])
        self.calculate_photosphere_properties()

    def set_radius_squared(self) -> None:
        self.radius_squared = (self.RADIUS_CONSTANT * self.v_ejecta * self.time) ** 2

    def set_rec_radius_squared(self) -> None:
        self.rec_radius_squared = self.luminosity / (self.STEF_CONSTANT * self.temperature_floor ** 4)

    @property
    def mask(self) -> np.ndarray:
        return self.radius_squared <= self.rec_radius_squared

    def calculate_r_photosphere(self) -> None:
        self.r_photosphere = self.rec_radius_squared ** 0.5
        self.r_photosphere[self.mask] = self.radius_squared[self.mask] ** 0.5

    def calculate_photosphere_temperature(self) -> None:
        self.photosphere_temperature = np.zeros(len(self.time))
        self.photosphere_temperature[self.mask] = \
            (self.luminosity[self.mask] / (self.STEF_CONSTANT * self.radius_squared[self.mask])) ** 0.25
        self.photosphere_temperature[~self.mask] = self.temperature_floor

    def calculate_photosphere_properties(self) -> tuple:
        self.set_radius_squared()
        self.set_rec_radius_squared()
        self.calculate_r_photosphere()
        self.calculate_photosphere_temperature()
        return self.photosphere_temperature, self.r_photosphere


class TDEPhotosphere(object):

    STEF_CONSTANT = 4 * np.pi * sigma_sb
    reference = "https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract"

    def __init__(self, time: np.ndarray, luminosity: np.ndarray, mass_bh: float, mass_star: float, star_radius: float,
                 tpeak: float, beta: float, rph_0: float, lphoto: float, **kwargs: None) -> None:
        """
        Photosphere that expands/recedes as a power law of Mdot

        :param time: time in source frame in days
        :param luminosity: luminosity
        :param mass_bh: black hole mass in solar masses
        :param mass_star: star mass in solar masses
        :param star_radius: star radius in solar radii
        :param tpeak: peak time in days
        :param beta: dmdt power law slope
        :param rph_0: initial photosphere radius
        :param lphoto: initial photosphere luminosity
        """
        self.time = time
        self.luminosity = luminosity
        self.mass_bh = mass_bh
        self.mass_star = mass_star
        self.star_radius = star_radius
        self.tpeak = tpeak
        self.beta = beta
        self.rph_0 = rph_0
        self.lphoto = lphoto

        self.calculate_photosphere_properties()

    @property
    def kappa_t(self) -> float:
        # Assume solar metallicity for now
        # 0.2*(1 + X) = mean Thomson opacity
        return 0.2 * (1 + 0.74)

    @property
    def star_radius_si(self) -> float:
        return self.star_radius * solar_radius

    @property
    def mass_bh_si(self) -> float:
        return self.mass_bh * solar_mass

    @property
    def rt(self) -> float:
        return (self.mass_bh / self.mass_star) ** (1. / 3.) * self.star_radius_si

    @property
    def rp(self) -> float:
        return self.rt / self.beta

    @property
    def a_p(self) -> float:
        return (graviational_constant * self.mass_bh_si * (self.tpeak * day_to_s / np.pi) ** 2) ** (1. / 3.)

    @property
    def a_t(self) -> float:
        """Semi-major axis of material that accretes at self.time,
        only calculate for times after first mass accretion"""
        return (graviational_constant * self.mass_bh_si * (self.time * day_to_s / np.pi) ** 2) ** (1. / 3.)

    @property
    def r_photo_min(self) -> float:
        return self.r_isco

    @property
    def r_photo_max(self) -> float:
        return self.rp + 2 * self.a_t

    @property
    def r_isco(self) -> float:
        return 6 * graviational_constant * self.mass_bh_si / (speed_of_light ** 2)

    @property
    def eddington_luminosity(self) -> float:
        return 4 * np.pi * graviational_constant * self.mass_bh_si * speed_of_light / self.kappa_t

    @property
    def r_photosphere(self) -> float:
        """adding rphotmin on to rphot for soft min
        also creating soft max -- inverse( 1/rphot + 1/rphotmax)"""
        rphot = self.rph_0 * self.a_p * (self.luminosity / self.eddington_luminosity) ** self.lphoto
        return (rphot * self.r_photo_max) / (rphot + self.r_photo_max) + self.r_photo_min

    @property
    def photosphere_temperature(self) -> float:
        return (self.luminosity / (self.r_photosphere ** 2 * self.STEF_CONSTANT)) ** 0.25

    def calculate_photosphere_properties(self) -> tuple:
        return self.photosphere_temperature, self.r_photosphere, self.rp


class DenseCore(object):

    STEF_CONSTANT = 4 * np.pi * sigma_sb
    reference = '.'

    def __init__(
            self, time: np.ndarray, luminosity: np.ndarray, mej: float, vej: float, kappa: float,
            envelope_slope: float = 10., **kwargs: None) -> None:
        """
        Photosphere with a dense core and a low-mass envelope.

        :param time: time in source frame in days
        :param luminosity: luminosity
        :param mej: ejecta mass
        :param vej: ejecta velocity in km/s
        :param kappa: opacity
        :param envelope_slope: envelope slope, default = 10
        """
        self.time = time
        self.luminosity = luminosity
        self.mej = mej
        self.vej = vej
        self.kappa = kappa
        self.envelope_slope = envelope_slope

        self.radius = np.zeros(len(self.time))
        self.rho_core = np.zeros(len(self.time))
        self.mask_3 = None

        self.r_photosphere = []
        self.photosphere_temperature = []
        self.calculate_photosphere_properties()

    @property
    def peak_luminosity_index(self) -> Any:
        return np.argmax(self.luminosity)

    @property
    def temperature_last(self) -> float:
        return 1e5

    def set_radius(self) -> None:
        self.radius = self.vej * km_cgs * self.time * day_to_s

    def set_rho_core(self) -> None:
        self.rho_core = (3.0 * self.mej * solar_mass / (4.0 * np.pi * self.radius ** 3))

    @property
    def tau_core(self) -> np.ndarray:
        return self.kappa * self.rho_core * self.radius

    @property
    def tau_e(self) -> np.ndarray:
        # Attach power-law envelope of negligible mass
        return self.kappa * self.rho_core * self.radius / (self.envelope_slope - 1.0)

    @property
    def mask_1(self) -> np.ndarray:
        return self.tau_e > (2.0 / 3.0)

    @property
    def mask_2(self) -> np.ndarray:
        return self.tau_core > 1.

    @property
    def mask_4(self) -> np.ndarray:
        # select all arrays after peak_luminosity_index
        return np.arange(self.peak_luminosity_index, len(self.luminosity), 1)

    @property
    def mask_all(self) -> np.ndarray:
        return self.mask_2 & self.mask_3 & self.mask_4

    def calculate_photosphere_properties(self) -> tuple:
        self.set_radius()
        self.set_rho_core()

        self.r_photosphere = np.zeros(len(self.time))
        self.r_photosphere[self.mask_1] = \
            (2.0 * (self.envelope_slope - 1.0) /
             (3.0 * self.kappa * self.rho_core[self.mask_1] * self.radius[self.mask_1] ** self.envelope_slope)) ** \
            (1.0 / (1.0 - self.envelope_slope))
        self.r_photosphere[~self.mask_1] = \
            self.envelope_slope * self.radius[~self.mask_1] / \
            (self.envelope_slope - 1.0) - 2.0 / (
                    3.0 * self.kappa * self.rho_core[~self.mask_1])

        self.photosphere_temperature = np.zeros(len(self.time))
        self.photosphere_temperature[self.mask_2] = \
            (self.luminosity[self.mask_2] / (self.r_photosphere[self.mask_2] ** 2 * self.STEF_CONSTANT)) ** 0.25
        self.photosphere_temperature[~self.mask_2] = self.temperature_last

        self.r_photosphere[~self.mask_2] = \
            (self.luminosity[~self.mask_2] /
             (self.photosphere_temperature[~self.mask_2] ** 4 * self.STEF_CONSTANT)) ** 0.5

        self.mask_3 = self.photosphere_temperature > self.temperature_last
        self.photosphere_temperature[self.mask_all] = self.temperature_last
        self.r_photosphere[self.mask_all] = \
            (self.luminosity[self.mask_all] /
             (self.photosphere_temperature[self.mask_all] ** 4 * self.STEF_CONSTANT)) ** 0.5

        return self.photosphere_temperature, self.r_photosphere
