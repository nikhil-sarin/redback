# This is mostly from mosfit/photospheres

import numpy as np
from redback.constants import *

class TemperatureFloor(object):
    def __init__(self, time, luminosity, vej, temperature_floor, **kwargs):
        """
        Photosphere with a floor temperature and effective blackbody otherwise

        :param time: source frame time in days
        :param luminosity: luminosity
        :param vej: ejecta velocity in km/s
        :param temperature_floor: floor temperature in kelvin
        """
        self.time = time
        self.luminosity = luminosity
        self.v_ejecta = vej
        self.temperature_floor = temperature_floor
        self.r_photosphere = []
        self.photosphere_temperature = []
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract'
        self.photosphere_temperature, self.r_photosphere = self.calculate_photosphere_properties()


    def calculate_photosphere_properties(self):
        radius_constant = km_cgs * day_to_s
        stef_constant = 4*np.pi * sigma_sb

        radius_squared = (radius_constant * self.v_ejecta * self.time) ** 2
        rec_radius_squared = self.luminosity/ (stef_constant * self.temperature_floor ** 4)

        temperature = np.zeros(len(self.time))

        mask = radius_squared < rec_radius_squared
        r_photosphere = radius_squared**0.5

        temperature[mask] = (self.luminosity[mask] / (stef_constant * radius_squared[mask])) ** 0.25
        r_photosphere[mask] = rec_radius_squared[mask]**0.5
        temperature[~mask] = self.temperature_floor

        return temperature, r_photosphere


class TDEphotosphere(object):
    def __init__(self, time, luminosity, mass_bh, mass_star, star_radius, tpeak, beta, rph_0, lphoto, **kwargs):
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

        self.r_photosphere = []
        self.photosphere_temperature = []
        self.rp = []
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract'

        self.photosphere_temperature, self.r_photosphere, self.rp = self.calculate_photosphere_properties()

    def calculate_photosphere_properties(self):
        stef_constant = 4*np.pi * sigma_sb

        star_radius = self.star_radius * solar_radius

        # Assume solar metallicity for now
        # 0.2*(1 + X) = mean Thomson opacity
        kappa_t = 0.2 * (1 + 0.74)
        tpeak = self.tpeak

        Ledd = (4 * np.pi * graviational_constant * self.mass_bh * solar_mass * speed_of_light / kappa_t)

        rt = (self.mass_bh / self.mass_star) ** (1. / 3.) * star_radius
        rp = rt / self.beta

        r_isco = 6 * graviational_constant * self.mass_bh * solar_mass / (speed_of_light**2)
        rphotmin = r_isco

        a_p = (graviational_constant * self.mass_bh * solar_mass * ((tpeak) * day_to_s / np.pi) ** 2) ** (1. / 3.)

        # semi-major axis of material that accretes at self._times,
        # only calculate for times after first mass accretion
        a_t = (graviational_constant * self.mass_bh * solar_mass * ((self.times) * day_to_s / np.pi) ** 2) ** (1. / 3.)

        rphotmax = rp + 2 * a_t

        # adding rphotmin on to rphot for soft min
        # also creating soft max -- inverse( 1/rphot + 1/rphotmax)
        rphot = self.rph_0 * a_p * (self.luminosity / Ledd) ** self.lphoto
        r_photosphere = (rphot * rphotmax) / (rphot + rphotmax) + rphotmin

        temperature = (self.luminosity / (r_photosphere ** 2 * stef_constant)) ** 0.25

        return temperature, r_photosphere, rp

class Densecore(object):
    def __init__(self, time, luminosity, mej, vej, kappa, envelope_slope=10, **kwargs):
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
        self.reference = '.'

        self.r_photosphere = []
        self.photosphere_temperature = []
        self.photosphere_temperature, self.r_photosphere = self.calculate_photosphere_properties()

    def calculate_photosphere_properties(self):
        stef_constant = 4 * np.pi * sigma_sb
        peak_luminosity_index = np.argmax(self.luminosity)
        temperature_last = 1e5

        radius = self.vej * km_cgs * self.time * day_to_s
        rho_core = (3.0 * self.mej * solar_mass / (4.0 * np.pi * radius ** 3))
        tau_core = self.kappa * rho_core * radius

        # Attach power-law envelope of negligible mass
        tau_e = self.kappa * rho_core * radius / (self.envelope_slope - 1.0)

        r_photosphere = np.zeros(len(self.time))
        temp = np.zeros(len(self.time))

        mask1 = tau_e > (2.0/3.0)
        r_photosphere[mask1] = (2.0 * (self.envelope_slope - 1.0) / (3.0 * self.kappa * rho_core[mask1] * radius[mask1] ** self.envelope_slope)) ** (
                    1.0 / (1.0 - self.envelope_slope))
        r_photosphere[~mask1] = self.envelope_slope * radius[~mask1] / (self.envelope_slope - 1.0) - 2.0 / (3.0 * self.kappa * rho_core[~mask1])

        mask2 = tau_core > 1.
        temp[mask2] = (self.luminosity[mask2] /(r_photosphere[mask2]**2 * stef_constant))**0.25
        temp[~mask2] = temperature_last
        r_photosphere[~mask2] = (self.luminosity[~mask2] / (temp[~mask2] ** 4 * stef_constant)) ** 0.5

        mask3 = temp > temperature_last

        # select all arrays after peak_luminosity_index
        mask4 = np.arange(peak_luminosity_index, len(self.luminosity), 1)
        mask_all = (mask2) & (mask3) & (mask4)

        temp[mask_all] = temperature_last
        r_photosphere[mask_all] = (self.luminosity[mask_all] / (temp[mask_all]** 4 * stef_constant))**0.5

        return temp, r_photosphere
