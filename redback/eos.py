import numpy as np
from redback.constants import *
from redback.utils import logger
import astropy.constants as cc
import astropy.units as uu
from scipy.interpolate import interp1d

try:
    import lalsimulation as lalsim
except ModuleNotFoundError as e:
    logger.warning(e)
    logger.warning('lalsimulation is not installed. Some EOS based models will not work.'
                   'Either use bilby eos or pass your own EOS generation class to the model')

class PiecewisePolytrope(object):
    def __init__(self, log_p, gamma_1, gamma_2, gamma_3):
        """
        :param log_p: log central pressure in SI units
        :param gamma_1: polytrope index 1
        :param gamma_2: polytrope index 2
        :param gamma_3: polytrope index 3
        """
        self.log_p = log_p
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3

    def maximum_mass(self):
        """
        :return: maximum non-rotating mass in solar masses (Mtov) for the equation of state
        """
        polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(self.log_p, self.gamma_1,
                                                                         self.gamma_2, self.gamma_3)
        polytrope = lalsim.CreateSimNeutronStarFamily(polytrope)
        maximum_mass = lalsim.SimNeutronStarMaximumMass(polytrope)/cc.M_sun.si.value
        return maximum_mass

    def maximum_speed_of_sound(self):
        """
        :return: maximum speed of sound in units of c
        """
        polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(self.log_p, self.gamma_1,
                                                                         self.gamma_2, self.gamma_3)
        max_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(polytrope)
        max_speed_of_sound = lalsim.SimNeutronStarEOSSpeedOfSound(
            max_enthalpy, polytrope)
        return max_speed_of_sound/cc.c.si.value

    def radius_of_mass(self, mass):
        """
        :param mass: mass array in solar masses
        :return: return radius in meters
        """

        m1 = mass * cc.M_sun.si.value
        polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(self.log_p, self.gamma_1,
                                                                         self.gamma_2, self.gamma_3)
        polytrope = lalsim.CreateSimNeutronStarFamily(polytrope)

        m_tmp = []
        r_tmp = []
        for mm in m1:
            try:
                r_tmp.append(lalsim.SimNeutronStarRadius(mm, polytrope))
                m_tmp.append(mm)
            except RuntimeError:
                pass
        return np.array(r_tmp)


    def lambda_of_central_pressure(self, central_pressure):
        """
        :param central_pressure:
        :return: mass in solar masses and dimensionless tidal deformability
        """
        polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(self.log_p, self.gamma_1,
                                                                         self.gamma_2, self.gamma_3)
        radius, mass, k2 = lalsim.SimNeutronStarTOVODEIntegrate(central_pressure, polytrope)

        Lambda = (2 / 3) * k2 * (cc.c.si.value ** 2 * radius / (cc.G.si.value * mass)) ** 5
        return mass, Lambda

    def lambda_array_of_central_pressure(self, central_pressure_array, maximum_mass_lower_limit=2.01):
        """
        :param central_pressure_array: array of central pressure in SI units
        :param maximum_mass_lower_limit: 2.01 solar masses, Throw out EOS's that are below this value.
        Users should enforce this at the prior level.
        :return: dimensionless tidal deformability
        """

        tmp = np.array([self.lambda_of_central_pressure(pp) for pp in central_pressure_array])
        mass = tmp[:, 0]
        lambdas = tmp[:, 1]

        arg_maximum_mass = np.argmax(mass)
        max_mass = mass[arg_maximum_mass]

        if max_mass < maximum_mass_lower_limit:
            raise ValueError("Maximum mass for this EOS is lower than 2.01, please choose a more realistic EOS")

        # Choose masses between 1. and maximum mass
        args = np.argwhere((mass >= 1.)).flatten()
        mass = mass[args[0]:arg_maximum_mass]
        lambdas = lambdas[args[0]:arg_maximum_mass]

        return mass, lambdas, max_mass

    def lambda_of_mass(self, central_pressure, mass, maximum_mass_lower_limit=2.01):
        """
        :param central_pressure: central pressure in SI units
        :param mass: neutron star masses in solar masses
        :param maximum_mass_limit:
        :return: lambda for the given mass array
        """
        mass_tmp, Lambda_tmp, max_mass = self.lambda_array_of_central_pressure(central_pressure, maximum_mass_lower_limit)

        if hasattr(mass, '__len__'):
            if mass.shape == (len(mass),):  # i.e. np.array([mass_1])
                interpolated_Lambda = interp1d(
                    mass_tmp, Lambda_tmp, fill_value='extrapolate')

                Lambda = interpolated_Lambda(mass)
                args = np.argwhere(Lambda >= 0).flatten()
                args_2 = np.argwhere(Lambda < 0).flatten()

                # We append zeros after a NS collapses, i.e, for masses > M_tov
                Lambda = np.append(Lambda[args], np.zeros(len(args_2)))

            elif mass.shape == (2, len(mass[0])):  # i.e. np.array([mass_1,mass_2])
                interpolated_Lambda = interp1d(
                    mass_tmp, Lambda_tmp, fill_value='extrapolate')
                Lambda = np.zeros([2, len(mass[0])])
                for ii in range(2):
                    Lambda_tmp_2 = interpolated_Lambda(mass[ii])
                    args = np.argwhere(Lambda_tmp_2 >= 0).flatten()
                    args_2 = np.argwhere(Lambda_tmp_2 < 0).flatten()

                    # We append zeros after a NS collapses, i.e, for masses > M_tov
                    Lambda[ii] = np.append(Lambda_tmp_2[args], np.zeros(len(args_2)))

        else:
            # if mass is a float
            interpolated_Lambda = interp1d(
                mass_tmp, Lambda_tmp, fill_value='extrapolate')
            Lambda = interpolated_Lambda(mass)
            if Lambda < 0:
                Lambda = 0


        return Lambda, max_mass



