import numpy as np
import redback.eos as eos
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

def nuclear_burning_constraints(parameters):
    """
    Constraint so that nuclear burning energy is greater than kinetic energy.

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    mej = parameters['mej'] * solar_mass
    vej = parameters['vej'] * km_cgs
    fnickel = parameters['fnickel']
    kinetic_energy = 0.5 * mej * (vej / 2.0) ** 2
    excess_constant = -(56.0 / 4.0 * 2.4249 - 53.9037) / proton_mass * mev_cgs
    emax = excess_constant * mej * fnickel
    converted_parameters['emax_constraint'] = emax - kinetic_energy
    return converted_parameters

def simple_fallback_constraints(parameters):
    """
    Constraint on the fall back energy being larger than the kinetic energy,
    and the that nebula phase does not begin till at least a 100 days.

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    mej = parameters['mej'] * solar_mass
    vej = parameters['vej'] * km_cgs
    kappa = parameters['kappa']
    l0 = parameters['l0']
    t0 = parameters['t_0']
    kinetic_energy = 0.5 * mej * vej**2
    tnebula =  np.sqrt(3 * kappa * mej / (4 * np.pi * vej ** 2)) / 86400
    e_fallback = l0 * 5./2./(t0 * day_to_s)**(2./3.)
    neutrino_energy = 1e51
    total_energy = e_fallback + neutrino_energy
    # ensure total energy is greater than kinetic energy
    converted_parameters['en_constraint'] = total_energy - kinetic_energy
    # ensure t_nebula is greater than 100 days
    converted_parameters['t_nebula_min'] = tnebula - 100
    return converted_parameters

def csm_constraints(parameters):
    """
    Constraint so that photospheric radius is within the csm and the
    diffusion time is less than the shock crossing time.

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    from redback.utils import get_csm_properties
    converted_parameters = parameters.copy()
    mej = parameters['mej']
    csm_mass = parameters['csm_mass']
    kappa = parameters['kappa']
    r0 = parameters['r0']
    vej = parameters['vej']
    nn = parameters.get('nn', 12)
    delta = parameters.get('delta', 1)
    eta = parameters['eta']
    rho = parameters['rho']

    mej = mej * solar_mass
    csm_mass = csm_mass * solar_mass
    r0 = r0 * au_cgs
    vej = vej * km_cgs
    Esn = 3. * vej ** 2 * mej / 10.

    csm_properties = get_csm_properties(nn, eta)
    AA = csm_properties.AA
    Bf = csm_properties.Bf
    qq = rho * r0 ** eta
    # outer CSM shell radius
    radius_csm = ((3.0 - eta) / (4.0 * np.pi * qq) * csm_mass + r0 ** (3.0 - eta)) ** (
            1.0 / (3.0 - eta))
    # photosphere radius
    r_photosphere = abs((-2.0 * (1.0 - eta) / (3.0 * kappa * qq) +
                         radius_csm ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))

    # mass of the optically thick CSM (tau > 2/3).
    mass_csm_threshold = np.abs(4.0 * np.pi * qq / (3.0 - eta) * (
            r_photosphere ** (3.0 - eta) - r0 ** (3.0 - eta)))

    g_n = (1.0 / (4.0 * np.pi * (nn - delta)) * (
            2.0 * (5.0 - delta) * (nn - 5.0) * Esn) ** ((nn - 3.) / 2.0) / (
                   (3.0 - delta) * (nn - 3.0) * mej) ** ((nn - 5.0) / 2.0))

    tshock = ((radius_csm - r0) / Bf / (AA * g_n / qq) ** (
                       1. / (nn - eta))) ** ((nn - eta) /(nn - 3))

    diffusion_time = np.sqrt(2. * kappa * mass_csm_threshold /(vej * 13.7 * 3.e10))
    # ensure shock crossing time is greater than diffusion time
    converted_parameters['shock_time'] = tshock - diffusion_time
    # ensure photospheric radius is within the csm i.e., r_photo < radius_csm and r_photo > r0
    converted_parameters['photosphere_constraint_1'] = radius_csm - r_photosphere
    converted_parameters['photosphere_constraint_2'] = r_photosphere - r0
    return converted_parameters

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
    polytrope = eos.PiecewisePolytrope(log_p=log_p, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)
    maximum_eos_mass = polytrope.maximum_mass()
    return maximum_eos_mass

@np.vectorize
def calc_speed_of_sound(log_p, gamma_1, gamma_2, gamma_3, **kwargs):
    polytrope = eos.PiecewisePolytrope(log_p=log_p, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)
    maximum_speed_of_sound = polytrope.maximum_speed_of_sound()
    return maximum_speed_of_sound
