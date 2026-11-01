import numpy as np
from redback.constants import *
from redback.utils import logger
import astropy.constants as cc
import astropy.units as uu
import warnings


def _import_lalsimulation():
    """Import lalsimulation while suppressing its IPython stdio redirection warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Wswiglal-redir-stdio.*", category=UserWarning)
        import lalsimulation as lalsim
    return lalsim

try:
    lalsim = _import_lalsimulation()
except ModuleNotFoundError as e:
    logger.warning(e)
    logger.warning('lalsimulation is not installed. Some EOS based models will not work.'
                   ' Please use bilby eos or pass your own EOS generation class to the model')


def _create_neutron_star_family(equation_of_state):
    """Create a minimal neutron-star family across LALSimulation APIs."""
    try:
        return lalsim.CreateSimNeutronStarFamily(equation_of_state, 1)
    except TypeError:
        return lalsim.CreateSimNeutronStarFamily(equation_of_state)

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
        polytrope = _create_neutron_star_family(polytrope)
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
        polytrope = _create_neutron_star_family(polytrope)

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
        :param central_pressure: Central pressure in SI units
        :return: mass in solar masses and dimensionless tidal deformability
        """
        polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(self.log_p, self.gamma_1,
                                                                         self.gamma_2, self.gamma_3)
        radius, mass, k2 = lalsim.SimNeutronStarTOVODEIntegrate(central_pressure, polytrope)

        Lambda = (2 / 3) * k2 * (cc.c.si.value ** 2 * radius / (cc.G.si.value * mass)) ** 5
        mass = mass / cc.M_sun.si.value
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
        :param maximum_mass_lower_limit: Maximum mass lower limit in solar masses (default 2.01)
        :return: lambda for the given mass array
        """
        del central_pressure  # Retained for API compatibility.
        equation_of_state = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(
            self.log_p, self.gamma_1, self.gamma_2, self.gamma_3)
        family = _create_neutron_star_family(equation_of_state)
        max_mass = (
            lalsim.SimNeutronStarMaximumMass(family) / cc.M_sun.si.value)
        if max_mass < maximum_mass_lower_limit:
            raise ValueError(
                "Maximum mass for this EOS is lower than 2.01, please choose "
                "a more realistic EOS")

        masses = np.asarray(mass, dtype=float)
        scalar_input = masses.ndim == 0
        lambdas = np.zeros_like(masses, dtype=float)
        for index in np.ndindex(masses.shape):
            if masses[index] > max_mass:
                continue
            mass_si = masses[index] * cc.M_sun.si.value
            try:
                radius = lalsim.SimNeutronStarRadius(mass_si, family)
                love_number = lalsim.SimNeutronStarLoveNumberK2(
                    mass_si, family)
            except RuntimeError:
                continue
            lambdas[index] = (
                2.0 / 3.0 * love_number
                * (cc.c.si.value ** 2 * radius
                   / (cc.G.si.value * mass_si)) ** 5)

        if scalar_input:
            return float(lambdas), max_mass
        return lambdas, max_mass


