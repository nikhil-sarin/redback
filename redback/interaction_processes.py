# This is mostly from mosfit/transforms but rewritten to be modular

import numpy as np
from scipy.interpolate import interp1d
from redback.constants import *

class Diffusion(object):
    def __init__(self, time, dense_times, luminosity, kappa, kappa_gamma, mej, vej, **kwargs):
        """
        :param time: source frame time in days
        :param dense_times: dense time array in days
        :param dense_luminosity: luminosity
        :param kappa: opacity
        :param kappa_gamma: gamma-ray opacity
        :param mej: ejecta mass
        :param vej: ejecta velocity
        Adds new attributes for tau_diffusion and new luminosity accounting for the
        interaction process at the time values
        """
        self.kappa = kappa
        self.kappa_gamma = kappa_gamma
        self.luminosity = luminosity
        self.time = time
        self.dense_times = dense_times
        self.m_ejecta = mej
        self.v_ejecta = vej
        self.reference = 'https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract'
        self.tau_d = []
        self.new_luminosity = []

        self.tau_d, self.new_luminosity = self.convert_input_luminosity()

    def convert_input_luminosity(self):
        timesteps = 100
        minimum_log_spacing = -3
        diffusion_constant = 2.0 * solar_mass / (13.7 * speed_of_light * km_cgs)
        trapping_constant = 3.0 * solar_mass / (4*np.pi * km_cgs ** 2)

        tau_diff = np.sqrt(diffusion_constant * self.kappa * self.m_ejecta / self.v_ejecta) / day_to_s
        trap_coeff = (trapping_constant * self.kappa_gamma * self.m_ejecta / (self.v_ejecta ** 2)) / day_to_s ** 2

        min_te = np.min(self.dense_times)
        tb = max(0.0, min_te)
        luminosity_interpolator = interp1d(self.dense_times, self.luminosity, copy=False, assume_sorted=True)

        uniq_times = np.unique(self.time[(self.time >= tb) & (self.time <= self.dense_times[-1])])
        lu = len(uniq_times)

        num = int(round(timesteps / 2.0))
        lsp = np.logspace(np.log10(tau_diff /self.dense_times[-1]) + minimum_log_spacing, 0, num)
        xm = np.unique(np.concatenate((lsp, 1 - lsp)))

        int_times = np.clip(tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb, self.dense_times[-1])

        int_te2s = int_times[:, -1] ** 2
        int_lums = luminosity_interpolator(int_times)
        int_args = int_lums * int_times * np.exp((int_times ** 2 - int_te2s.reshape(lu, 1)) / tau_diff**2)
        int_args[np.isnan(int_args)] = 0.

        uniq_lums = np.trapz(int_args, int_times, axis=1)
        uniq_lums *= -2.0 * np.expm1(-trap_coeff / int_te2s) / tau_diff**2

        new_lums = uniq_lums[np.searchsorted(uniq_times, self.time)]

        return tau_diff, new_lums

class AsphericalDiffusion(object):
    def __init__(self, time, dense_times, luminosity, kappa, kappa_gamma, mej, vej, area_projection, area_reference, **kwargs):
        """
        :param time: source frame time in days
        :param dense_times: dense time array in days
        :param luminosity: luminosity
        :param kappa: opacity
        :param kappa_gamma: gamma-ray opacity
        :param mej: ejecta mass
        :param vej: ejecta velocity
        :param area_projection: projected area of cocoon/polar ejecta
        :param area_reference: remaining reference area i.e., the equitorial ejecta
        Adds new attributes for tau_diffusion and new luminosity accounting for the interaction process
        """
        self.kappa = kappa
        self.kappa_gamma = kappa_gamma
        self.luminosity = luminosity
        self.time = time
        self.dense_times = dense_times
        self.m_ejecta = mej
        self.v_ejecta = vej
        self.area_projection = area_projection
        self.area_reference = area_reference
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2020ApJ...897..150D/abstract'
        self.tau_d = []
        self.new_luminosity = []

        self.tau_d, self.new_luminosity = self.convert_input_luminosity()

    def convert_input_luminosity(self):
        timesteps = 100
        minimum_log_spacing = -3
        diffusion_constant = 2.0 * solar_mass / (13.7 * speed_of_light * km_cgs)
        trapping_constant = 3.0 * solar_mass / (4*np.pi * km_cgs ** 2)

        tau_diff = np.sqrt(diffusion_constant * self.kappa * self.m_ejecta / self.v_ejecta) / day_to_s
        trap_coeff = (trapping_constant * self.kappa_gamma * self.m_ejecta / (self.v_ejecta ** 2)) / day_to_s ** 2

        min_te = min(self.dense_times)
        tb = max(0.0, min_te)
        luminosity_interpolator = interp1d(self.dense_times, self.luminosity, copy=False,assume_sorted=True)

        uniq_times = np.unique(self.time[(self.time >= tb) & (self.time <= self.dense_times[-1])])
        lu = len(uniq_times)

        num = int(round(timesteps / 2.0))
        lsp = np.logspace(np.log10(tau_diff /self.dense_times[-1]) + minimum_log_spacing, 0, num)
        xm = np.unique(np.concatenate((lsp, 1 - lsp)))

        int_times = np.clip(tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb, self.dense_times[-1])

        int_te2s = int_times[:, -1] ** 2
        int_lums = luminosity_interpolator(int_times)
        int_args = int_lums * int_times * np.exp((int_times ** 2 - int_te2s.reshape(lu, 1)) / tau_diff**2)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.trapz(int_args, int_times, axis=1)
        uniq_lums *= -2.0 * np.expm1(-trap_coeff / int_te2s) / tau_diff**2

        uniq_lums *= (1 + 1.4 * (2 + uniq_times/tau_diff/0.59) / (1 + np.exp(uniq_times/tau_diff/0.59)) *
                      (self.area_projection/self.area_reference - 1))

        new_lums = uniq_lums[np.searchsorted(uniq_times, self.time)]

        return tau_diff, new_lums

class CSMDiffusion(object):
    def __init__(self, time, dense_times, luminosity, kappa, r_photosphere, mass_csm_threshold, csm_mass, **kwargs):
        """
        :param time: source frame time in days
        :param dense_times: dense time array in days
        :param luminosity: luminosity
        :param kappa: opacity
        :param csm_mass: csm mass in solar masses
        :param mej: ejecta mass in solar masses
        :param r0: radius of csm shell in AU
        :param eta: csm density profile exponent
        :param rho: csm density profile amplitude
        Adds new attribute for luminosity accounting for the interaction process
        """
        self.time = time
        self.dense_times = dense_times
        self.luminosity = luminosity
        self.kappa = kappa
        self.r_photosphere = r_photosphere
        self.mass_csm_threshold = mass_csm_threshold
        self.csm_mass = csm_mass * solar_mass
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract'

        self.new_luminosity = []
        self.new_luminosity = self.convert_input_luminosity()

    def convert_input_luminosity(self):
        timesteps = 3000
        minimum_log_spacing = -3

        # Derived parameters

        # photosphere radius
        r_photosphere = self.r_photosphere

        # mass of the optically thick CSM (tau > 2/3).
        mass_csm_threshold = self.mass_csm_threshold

        beta = 4. * np.pi ** 3. / 9.
        t0 = self.kappa * (mass_csm_threshold) / (beta * speed_of_light * r_photosphere) / day_to_s

        min_te = min(self.dense_times)
        tb = max(0.0, min_te)
        luminosity_interpolator = interp1d(self.dense_times, self.luminosity, copy=False,assume_sorted=True)
        uniq_times = np.unique(self.time[(self.time >= tb) & (self.time <= self.dense_times[-1])])
        lu = len(uniq_times)

        num = int(round(timesteps / 2.0))
        lsp = np.logspace(np.log10(t0 /self.dense_times[-1]) + minimum_log_spacing, 0, num)
        xm = np.unique(np.concatenate((lsp, 1 - lsp)))

        int_times = tb + (uniq_times.reshape(lu, 1) - tb) * xm
        int_tes = int_times[:, -1]

        int_lums = luminosity_interpolator(int_times)
        int_args = int_lums * np.exp((int_times) / t0)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.trapz(int_args, int_times, axis=1)
        uniq_lums *= np.exp(-int_tes/t0)/t0
        new_lums = uniq_lums[np.searchsorted(uniq_times, self.time)]
        return new_lums

class Viscous(object):
    def __init__(self, time, dense_times, luminosity, t_viscous, **kwargs):
        """
        :param time: source frame time in days
        :param dense_times: dense time array in days
        :param luminosity: luminosity
        :param t_viscous: viscous timescale
        Adds new attribute for luminosity accounting for the interaction process
        """
        self.luminosity = luminosity
        self.time = time
        self.dense_times = dense_times
        self.tvisc = t_viscous
        self.reference = ''
        self.new_luminosity = []

        self.new_luminosity = self.convert_input_luminosity()

    def convert_input_luminosity(self):
        timesteps = 1000
        minimum_log_spacing = -3

        min_te = min(self.dense_times)
        tb = max(0.0, min_te)
        luminosity_interpolator = interp1d(self.dense_times, self.luminosity, copy=False,assume_sorted=True)

        uniq_times = np.unique(self.time[(self.time >= tb) & (self.time <= self.dense_times[-1])])
        lu = len(uniq_times)

        num = int(round(timesteps / 2.0))
        lsp = np.logspace(np.log10(self.tvisc /self.dense_times[-1]) +minimum_log_spacing, 0, num)
        xm = np.unique(np.concatenate((lsp, 1 - lsp)))

        int_times = np.clip(tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb, self.dense_times[-1])

        int_tes = int_times[:, -1]
        int_lums = luminosity_interpolator(int_times)
        int_args = int_lums * np.exp((int_times - int_tes.reshape(lu, 1)) / self.tvisc)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.trapz(int_args, int_times, axis=1)/self.tvisc

        new_lums = uniq_lums[np.searchsorted(uniq_times, self.time)]

        return new_lums
