import numpy as np
from redback.constants import *

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

def basic_magnetar_powered_sn_constraints(parameters):
    """
    Constraint so that magnetar rotational energy is larger than ejecta kinetic energy

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    mej = parameters['mej'] * solar_mass
    vej = parameters['vej'] * km_cgs
    mass_ns = parameters['mass_ns']
    p0 = parameters['p0']
    kinetic_energy = 0.5 * mej * vej**2
    rotational_energy = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
    # ensure rotational energy is greater than total output energy
    converted_parameters['erot_constraint'] = rotational_energy - kinetic_energy
    return converted_parameters

def general_magnetar_powered_sn_constraints(parameters):
    """
    Constraint so that magnetar rotational energy is larger than ejecta kinetic energy

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    mej = parameters['mej'] * solar_mass
    vej = parameters['vej'] * km_cgs
    kinetic_energy = 0.5 * mej * vej ** 2
    l0 = parameters['l0']
    tau = parameters['tsd']
    rotational_energy = 2*l0*tau
    # ensure rotational energy is greater than total output energy
    converted_parameters['erot_constraint'] = rotational_energy - kinetic_energy
    return converted_parameters

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

def simple_fallback_constraints():
    pass

def csm_constraints():
    pass

def magnetar_driven_kilonova_constraints():
    pass

def nuclear_burning_constraints():
    pass

def piecewise_polytrope_eos_constraints(parameters):
    """
    Constraint on piecewise-polytrope EOS to enforce causality and max mass

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    log_p = parameters['log_p']
    gamma_1 = parameters['gamma_1']
    gamma_2 = parameters['gamma_2']
    gamma_3 = parameters['gamma_3']
    maximum_eos_mass = calc_max_mass(log_p=log_p, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)
    converted_parameters['maximum_eos_mass'] = maximum_eos_mass

    maximum_speed_of_sound = calc_speed_of_sound(log_p=log_p, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)
    converted_parameters['maximum_speed_of_sound'] = maximum_speed_of_sound
    return converted_parameters

@np.vectorize
def calc_max_mass(log_p, gamma_1, gamma_2, gamma_3, **kwargs):
    import toast
    maximum_eos_mass = toast.piecewise_polytrope.maximum_mass(
        log_p=log_p, Gamma_1=gamma_1, Gamma_2=gamma_2, Gamma_3=gamma_3)
    return maximum_eos_mass

@np.vectorize
def calc_speed_of_sound(log_p, gamma_1, gamma_2, gamma_3, **kwargs):
    import toast
    maximum_speed_of_sound = toast.piecewise_polytrope.maximum_speed_of_sound(
        log_p=log_p, Gamma_1=gamma_1, Gamma_2=gamma_2, Gamma_3=gamma_3)
    return maximum_speed_of_sound