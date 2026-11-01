import numpy as np
import redback.eos as eos
import redback.ejecta_relations as ejr
from redback.constants import *
from redback.utils import calc_tfb
from scipy.interpolate import interp1d

BARNES_KASEN_MASS_MIN = 1e-3
BARNES_KASEN_MASS_MAX = 1e-1
BARNES_KASEN_VELOCITY_MIN = 0.1
BARNES_KASEN_VELOCITY_MAX = 0.4

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
    erot_ratio = total_energy/rotational_energy
    converted_parameters['e_rot_constraint'] = erot_ratio
    # ensure t_nebula is greater than 100 days
    tnebula_constraint = tnebula - 100
    converted_parameters['t_nebula_min'] = tnebula_constraint
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
    converted_parameters['erot_constraint'] = kinetic_energy/rotational_energy
    return converted_parameters


def nickel_mixing_constraints(parameters):
    """Require the nickel-bearing region to contain the requested nickel mass."""
    converted_parameters = parameters.copy()
    converted_parameters['nickel_fraction_in_mixed_region'] = (
        parameters['f_nickel'] / parameters['f_mixing'])
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
    converted_parameters['erot_constraint'] = kinetic_energy/rotational_energy
    return converted_parameters
    
def vacuum_dipole_magnetar_powered_supernova_constraints(parameters):
    """
    Constraint so that magnetar rotational energy is smaller than some number

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    l0 = parameters['l0']
    tau = parameters['tau_sd']
    rotational_energy = l0*tau
    # ensure rotational energy is less than the maximum spin down energy
    converted_parameters['erot_constraint'] = rotational_energy/1e53
    return converted_parameters        
    
def general_magnetar_powered_supernova_constraints(parameters):
    """
    Constraint so that magnetar rotational energy is smaller than some number

    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    l0 = parameters['l0']
    tau = parameters['tau_sd']
    nn = parameters['nn']    
    rotational_energy = (nn-1)*l0*tau/2.0
    # ensure rotational energy is less than the maximum spin down energy
    converted_parameters['erot_constraint'] = rotational_energy/1e53
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
    schwarzchild_radius = (2 * graviational_constant * mass_bh * solar_mass /(speed_of_light**2))/au_cgs
    disruption_ratio = schwarzchild_radius/rp
    converted_parameters['disruption_radius'] = disruption_ratio
    return converted_parameters

def gaussianrise_tde_constraints(parameters):
    """
    Constraint on beta, eta and peak time for gaussian rise TDE model
    :param parameters: dictionary of parameters
    :return: converted_parameters dictionary where the violated samples are thrown out
    """
    converted_parameters = parameters.copy()
    ms = parameters['stellar_mass']
    mbh6 = parameters['mbh_6']
    betamax = 12.*(ms**(7./15.))*(mbh6**(-2./3.))
    tfb = calc_tfb(binding_energy_const=0.8, mbh_6=mbh6,stellar_mass=ms)/86400
    tfb_obs = tfb * (1 + parameters['redshift'])
    converted_parameters['beta_high'] = converted_parameters['beta']/betamax
    converted_parameters['tfb_max'] = converted_parameters['peak_time']/tfb_obs
    return converted_parameters


def cooling_envelope_constraints(parameters):
    """
    Constraint on cooling-envelope TDE parameters.

    The eta and beta bounds follow the cooling-envelope example. Gaussian-rise variants also require
    that the Gaussian value at the fallback stitching point is finite enough to normalise.
    """
    converted_parameters = parameters.copy()
    ms = parameters['stellar_mass']
    mbh6 = parameters['mbh_6']
    etamin = 0.01 * (ms ** (-7. / 15.)) * (mbh6 ** (2. / 3.))
    betamax = 12. * (ms ** (7. / 15.)) * (mbh6 ** (-2. / 3.))
    with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
        eta_min_ratio = etamin / parameters['eta']
        beta_max_ratio = parameters['beta'] / betamax
    converted_parameters['eta_min_ratio'] = np.nan_to_num(
        eta_min_ratio, nan=np.inf, posinf=np.inf, neginf=np.inf)
    converted_parameters['beta_max_ratio'] = np.nan_to_num(
        beta_max_ratio, nan=np.inf, posinf=np.inf, neginf=np.inf)

    if 'sigma_t' in parameters:
        redshift = parameters.get('redshift', 0.)
        xi = parameters.get('xi', 1.)
        binding_energy_const = parameters.get('binding_energy_const', 0.8)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
            transition_time = xi * calc_tfb(binding_energy_const, mbh6, ms) * (1. + redshift)
            peak_time = parameters['peak_time'] * day_to_s
            sigma_t = parameters['sigma_t'] * day_to_s
            gaussian_stitching_tail = np.abs(transition_time - peak_time) / sigma_t
        converted_parameters['gaussian_stitching_tail'] = np.nan_to_num(
            gaussian_stitching_tail, nan=np.inf, posinf=np.inf, neginf=np.inf)
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
    fnickel = parameters['f_nickel']
    kinetic_energy = 0.5 * mej * (vej / 2.0) ** 2
    excess_constant = -(56.0 / 4.0 * 2.4249 - 53.9037) / proton_mass * mev_cgs
    emax = excess_constant * mej * fnickel
    converted_parameters['emax_constraint'] = kinetic_energy/emax
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
    t0 = parameters['t_0_turn']
    kinetic_energy = 0.5 * mej * vej**2
    tnebula =  np.sqrt(3 * kappa * mej / (4 * np.pi * vej ** 2)) / 86400
    e_fallback = l0 * 5./2./(t0 * day_to_s)**(2./3.)
    neutrino_energy = 1e51
    total_energy = e_fallback + neutrino_energy
    # ensure total energy is greater than kinetic energy
    converted_parameters['en_constraint'] = kinetic_energy/total_energy
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
    converted_parameters = parameters.copy()
    mej = parameters['mej']
    csm_mass = parameters['csm_mass']
    kappa = parameters['kappa']
    r0 = parameters['r0']
    if 'vej' not in parameters:
        with np.errstate(divide='ignore', invalid='ignore'):
            vej = np.sqrt(2.0 * parameters['ek'] / (np.asarray(mej, dtype=float) * solar_mass)) / km_cgs
    else:
        vej = parameters['vej']
    nn = parameters.get('nn', np.ones_like(np.asarray(mej, dtype=float)) * 12.)
    delta = parameters.get('delta', np.ones_like(np.asarray(mej, dtype=float)))
    eta = parameters['eta']
    rho = parameters['rho']

    mej = np.asarray(mej, dtype=float) * solar_mass
    csm_mass = np.asarray(csm_mass, dtype=float) * solar_mass
    kappa = np.asarray(kappa, dtype=float)
    r0 = np.asarray(r0, dtype=float) * au_cgs
    vej = np.asarray(vej, dtype=float) * km_cgs
    nn = np.asarray(nn, dtype=float)
    delta = np.asarray(delta, dtype=float)
    eta = np.asarray(eta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    Esn = 3. * vej ** 2 * mej / 10.

    ns = [6, 7, 8, 9, 10, 12, 14]
    Bfs = [1.377, 1.299, 1.267, 1.250, 1.239, 1.226, 1.218]
    As = [0.62, 0.27, 0.15, 0.096, 0.067, 0.038, 0.025]

    Bf_func = interp1d(ns, Bfs)
    A_func = interp1d(ns, As)

    with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
        Bf = Bf_func(nn)
        AA = A_func(nn)

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
                           1. / (nn - eta))) ** ((nn - eta) / (nn - 3))

        diffusion_time = np.sqrt(2. * kappa * mass_csm_threshold / (vej * 13.7 * 3.e10))
        shock_time = diffusion_time / tshock
        photosphere_constraint_1 = r_photosphere / radius_csm
        photosphere_constraint_2 = r0 / r_photosphere

    shock_time = np.nan_to_num(shock_time, nan=np.inf, posinf=np.inf, neginf=np.inf)
    photosphere_constraint_1 = np.nan_to_num(photosphere_constraint_1, nan=np.inf, posinf=np.inf, neginf=np.inf)
    photosphere_constraint_2 = np.nan_to_num(photosphere_constraint_2, nan=np.inf, posinf=np.inf, neginf=np.inf)
    # ensure shock crossing time is greater than diffusion time
    converted_parameters['shock_time'] = shock_time
    # ensure photospheric radius is within the csm i.e., r_photo < radius_csm and r_photo > r0
    converted_parameters['photosphere_constraint_1'] = photosphere_constraint_1
    converted_parameters['photosphere_constraint_2'] = photosphere_constraint_2
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


def _maybe_scalar(array):
    array = np.asarray(array)
    if array.shape == ():
        return array.item()
    return array


def _polytrope_two_component_bns_ejecta_quantities(mass_1, mass_2, log_p, gamma_1, gamma_2, gamma_3, zeta):
    central_pressure = np.logspace(np.log10(4e32), np.log10(2.5e35), 70)
    inputs = np.broadcast_arrays(
        np.asarray(mass_1), np.asarray(mass_2), np.asarray(log_p),
        np.asarray(gamma_1), np.asarray(gamma_2), np.asarray(gamma_3), np.asarray(zeta))
    shape = inputs[0].shape
    output_shape = shape if shape else (1,)
    dynamical_mej = np.empty(output_shape, dtype=float)
    disk_wind_mej = np.empty(output_shape, dtype=float)
    ejecta_velocity = np.empty(output_shape, dtype=float)

    for index in np.ndindex(output_shape):
        array_index = index if shape else ()
        try:
            eos_model = eos.PiecewisePolytrope(
                log_p=inputs[2][array_index],
                gamma_1=inputs[3][array_index],
                gamma_2=inputs[4][array_index],
                gamma_3=inputs[5][array_index])
            mtov = eos_model.maximum_mass()
            lambda_1, _ = eos_model.lambda_of_mass(central_pressure=central_pressure, mass=float(inputs[0][array_index]))
            lambda_2, _ = eos_model.lambda_of_mass(central_pressure=central_pressure, mass=float(inputs[1][array_index]))
            ejecta_relation = ejr.TwoComponentBNS(
                mass_1=inputs[0][array_index], mass_2=inputs[1][array_index],
                lambda_1=lambda_1, lambda_2=lambda_2,
                mtov=mtov, zeta=inputs[6][array_index])
            dynamical_mej[index] = ejecta_relation.dynamical_mej
            disk_wind_mej[index] = ejecta_relation.disk_wind_mej
            ejecta_velocity[index] = ejecta_relation.ejecta_velocity
        except Exception:
            dynamical_mej[index] = np.nan
            disk_wind_mej[index] = np.nan
            ejecta_velocity[index] = np.nan

    if not shape:
        return dynamical_mej[0], disk_wind_mej[0], ejecta_velocity[0]
    return _maybe_scalar(dynamical_mej.reshape(shape)), _maybe_scalar(disk_wind_mej.reshape(shape)), \
        _maybe_scalar(ejecta_velocity.reshape(shape))


def _safe_constraint_ratio(numerator, denominator):
    with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
        ratio = np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
    return np.nan_to_num(ratio, nan=np.inf, posinf=np.inf, neginf=np.inf)


def _add_barnes_kasen_ejecta_constraints(converted_parameters, prefix, mej, vej):
    converted_parameters[f'{prefix}_mej_min'] = _safe_constraint_ratio(BARNES_KASEN_MASS_MIN, mej)
    converted_parameters[f'{prefix}_mej_max'] = _safe_constraint_ratio(mej, BARNES_KASEN_MASS_MAX)
    converted_parameters[f'{prefix}_vej_min'] = _safe_constraint_ratio(BARNES_KASEN_VELOCITY_MIN, vej)
    converted_parameters[f'{prefix}_vej_max'] = _safe_constraint_ratio(vej, BARNES_KASEN_VELOCITY_MAX)


def one_component_bns_ejecta_relation_constraints(parameters):
    """Constrain BNS ejecta-relation outputs to the Barnes-Kasen calibration domain."""
    converted_parameters = parameters.copy()
    ejecta_relation = ejr.OneComponentBNSNoProjection(
        mass_1=parameters['mass_1'], mass_2=parameters['mass_2'],
        lambda_1=parameters['lambda_1'], lambda_2=parameters['lambda_2'])
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'ejecta', ejecta_relation.ejecta_mass, ejecta_relation.ejecta_velocity)
    return converted_parameters


def one_component_bns_ejecta_relation_projection_constraints(parameters):
    """Constrain projected BNS ejecta-relation outputs to the Barnes-Kasen calibration domain."""
    converted_parameters = parameters.copy()
    ejecta_relation = ejr.OneComponentBNSProjection(
        mass_1=parameters['mass_1'], mass_2=parameters['mass_2'],
        lambda_1=parameters['lambda_1'], lambda_2=parameters['lambda_2'])
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'ejecta', ejecta_relation.ejecta_mass, ejecta_relation.ejecta_velocity)
    return converted_parameters


def two_component_bns_ejecta_relation_constraints(parameters):
    """Constrain two-component BNS ejecta-relation outputs to the Barnes-Kasen calibration domain."""
    converted_parameters = parameters.copy()
    ejecta_relation = ejr.TwoComponentBNS(
        mass_1=parameters['mass_1'], mass_2=parameters['mass_2'],
        lambda_1=parameters['lambda_1'], lambda_2=parameters['lambda_2'],
        mtov=parameters['mtov'], zeta=parameters['zeta'])
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'dynamical_ejecta',
        ejecta_relation.dynamical_mej, ejecta_relation.ejecta_velocity)
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'disk_wind_ejecta',
        ejecta_relation.disk_wind_mej, parameters['vej_2'])
    return converted_parameters


def one_component_nsbh_ejecta_relation_constraints(parameters):
    """Constrain NSBH ejecta-relation outputs to the Barnes-Kasen calibration domain."""
    converted_parameters = parameters.copy()
    ejecta_relation = ejr.OneComponentNSBH(
        mass_bh=parameters['mass_bh'], mass_ns=parameters['mass_ns'],
        chi_bh=parameters['chi_bh'], lambda_ns=parameters['lambda_ns'])
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'ejecta', ejecta_relation.ejecta_mass, ejecta_relation.ejecta_velocity)
    return converted_parameters


def two_component_nsbh_ejecta_relation_constraints(parameters):
    """Constrain two-component NSBH ejecta-relation outputs to the Barnes-Kasen calibration domain."""
    converted_parameters = parameters.copy()
    ejecta_relation = ejr.TwoComponentNSBH(
        mass_bh=parameters['mass_bh'], mass_ns=parameters['mass_ns'],
        chi_bh=parameters['chi_bh'], lambda_ns=parameters['lambda_ns'], zeta=parameters['zeta'])
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'dynamical_ejecta',
        ejecta_relation.dynamical_mej, ejecta_relation.ejecta_velocity)
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'disk_wind_ejecta',
        ejecta_relation.disk_wind_mej, parameters['vej_2'])
    return converted_parameters


def polytrope_eos_two_component_bns_constraints(parameters):
    """Constrain EOS and derived BNS ejecta outputs for polytrope-eos kilonova models."""
    converted_parameters = piecewise_polytrope_eos_constraints(parameters)
    dynamical_mej, disk_wind_mej, ejecta_velocity = _polytrope_two_component_bns_ejecta_quantities(
        mass_1=parameters['mass_1'], mass_2=parameters['mass_2'],
        log_p=parameters['log_p'], gamma_1=parameters['gamma_1'],
        gamma_2=parameters['gamma_2'], gamma_3=parameters['gamma_3'],
        zeta=parameters['zeta'])
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'dynamical_ejecta', dynamical_mej, ejecta_velocity)
    _add_barnes_kasen_ejecta_constraints(
        converted_parameters, 'disk_wind_ejecta', disk_wind_mej, parameters['vej_2'])
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
