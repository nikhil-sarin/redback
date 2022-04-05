import numpy as np
from redback.constants import *
import redback.transient_models as tm

def slsn_constraint(parameters):
    """
    Place constraints on the magnetar rotational energy being larger than the total output energy,
    and the that nebula phase does not begin till at least a 100 days.

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    mej = parameters['mej'] * solar_mass
    vej = parameters['vej'] * km_cgs
    kappa = parameters['kappa']
    mass_ns = parameters['mass_ns']
    p0 = parameters['p0']
    kinetic_energy = 0.5 * mej * vej**2
    rotational_energy = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
    tnebula =  np.sqrt(3 * kappa * mej / (4 * np.pi * vej ** 2)) / 86400
    neutrino_energy = 1e51
    total_energy = kinetic_energy + neutrino_energy
    # ensure rotational energy is greater than total output energy
    converted_parameters['erot_constraint'] = rotational_energy - total_energy
    # ensure t_nebula is greater than 100 days
    converted_parameters['t_nebula_min'] = tnebula - 100
    return converted_parameters


def csm_constraints():
    pass

def basic_magnetar_powered_sn_constraints():
    pass

def general_magnetar_powered_sn_constraints():
    pass

def tde_constraints(parameters):
    """
    Constraint so that the pericenter radius is larger than the schwarzchild radius of the black hole.

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    rp = parameters['pericenter_radius']
    mass_bh = parameters['mass_bh']
    schwarzchild_radius = (2 * graviational_constant * mass_bh * solar_mass /(speed_of_light**2))
    converted_parameters['disruption_radius'] = rp - schwarzchild_radius
    return converted_parameters

def magnetar_driven_kilonova_constraints():
    pass

def nuclear_burning_constraints():
    pass

def simple_fallback_constraints():
    pass

def eos_constraints():
    pass

def max_mass_constraints():
    pass