import numpy as np
import os

import bilby.core.prior
from bilby.core.prior import Constraint, PriorDict

import redback.model_library
import redback.constraints
from redback.utils import logger

_constraint_settings = {
    'slsn': (
        redback.constraints.slsn_constraint,
        {'e_rot_constraint': (0, 1), 't_nebula_min': (0, 400)}
    ),
    'slsn_bolometric': (
        redback.constraints.slsn_constraint,
        {'e_rot_constraint': (0, 1), 't_nebula_min': (0, 400)}
    ),
    'basic_magnetar_powered': (
        redback.constraints.basic_magnetar_powered_sn_constraints,
        {'erot_constraint': (0, 1)}
    ),
    'general_magnetar_slsn': (
        redback.constraints.general_magnetar_powered_sn_constraints,
        {'erot_constraint': (0, 1)}
    ),
    'gaussianrise_tde': (
        redback.constraints.gaussianrise_tde_constraints,
        {'beta_high': (0, 1), 'tfb_max': (0, 1)}
    ),
    # eta and beta validity depends on mbh_6 and stellar_mass; pass constraint=True to enforce this.
    'cooling_envelope': (
        redback.constraints.cooling_envelope_constraints,
        {'eta_min_ratio': (0, 1), 'beta_max_ratio': (0, 1)}
    ),
    'gaussianrise_cooling_envelope': (
        redback.constraints.cooling_envelope_constraints,
        {'eta_min_ratio': (0, 1), 'beta_max_ratio': (0, 1), 'gaussian_stitching_tail': (0, 35)}
    ),
    'gaussianrise_cooling_envelope_bolometric': (
        redback.constraints.cooling_envelope_constraints,
        {'eta_min_ratio': (0, 1), 'beta_max_ratio': (0, 1), 'gaussian_stitching_tail': (0, 35)}
    ),
    'bpl_cooling_envelope': (
        redback.constraints.cooling_envelope_constraints,
        {'eta_min_ratio': (0, 1), 'beta_max_ratio': (0, 1)}
    ),
    'smooth_exponential_powerlaw_cooling_envelope_bolometric': (
        redback.constraints.cooling_envelope_constraints,
        {'eta_min_ratio': (0, 1), 'beta_max_ratio': (0, 1)}
    ),
    'arnett': (
        redback.constraints.nuclear_burning_constraints,
        {'emax_constraint': (0, 1)}
    ),
    'nickelmixing': (
        redback.constraints.nickel_mixing_constraints,
        {'nickel_fraction_in_mixed_region': (0, 1),
         'nickel_mixing_order': (0, np.inf)}
    ),
    'nickelmixing_bolometric': (
        redback.constraints.nickel_mixing_constraints,
        {'nickel_fraction_in_mixed_region': (0, 1),
         'nickel_mixing_order': (0, np.inf)}
    ),
    'tde_analytical': (
        redback.constraints.simple_fallback_constraints,
        {'en_constraint': (0, 1), 't_nebula_min': (0, 400)}
    ),
    'csm_interaction': (
        redback.constraints.csm_constraints,
        {'shock_time': (0, 0.5), 'photosphere_constraint_1': (0, 1), 'photosphere_constraint_2': (0, 1)}
    ),
    'csm_nickel': (
        redback.constraints.csm_constraints,
        {'shock_time': (0, 0.5), 'photosphere_constraint_1': (0, 1), 'photosphere_constraint_2': (0, 1)}
    ),
    'polytrope_eos_two_component_bns': (
        redback.constraints.polytrope_eos_two_component_bns_constraints,
        {'maximum_eos_mass': (1.5, 5), 'maximum_speed_of_sound': (0, 1.15),
         'dynamical_ejecta_mej_min': (0, 1), 'dynamical_ejecta_mej_max': (0, 1),
         'dynamical_ejecta_vej_min': (0, 1), 'dynamical_ejecta_vej_max': (0, 1),
         'disk_wind_ejecta_mej_min': (0, 1), 'disk_wind_ejecta_mej_max': (0, 1),
         'disk_wind_ejecta_vej_min': (0, 1), 'disk_wind_ejecta_vej_max': (0, 1)}
    ),
    'one_component_ejecta_relation': (
        redback.constraints.one_component_bns_ejecta_relation_constraints,
        {'ejecta_mej_min': (0, 1), 'ejecta_mej_max': (0, 1),
         'ejecta_vej_min': (0, 1), 'ejecta_vej_max': (0, 1)}
    ),
    'one_component_ejecta_relation_projection': (
        redback.constraints.one_component_bns_ejecta_relation_projection_constraints,
        {'ejecta_mej_min': (0, 1), 'ejecta_mej_max': (0, 1),
         'ejecta_vej_min': (0, 1), 'ejecta_vej_max': (0, 1)}
    ),
    'two_component_bns_ejecta_relation': (
        redback.constraints.two_component_bns_ejecta_relation_constraints,
        {'dynamical_ejecta_mej_min': (0, 1), 'dynamical_ejecta_mej_max': (0, 1),
         'dynamical_ejecta_vej_min': (0, 1), 'dynamical_ejecta_vej_max': (0, 1),
         'disk_wind_ejecta_mej_min': (0, 1), 'disk_wind_ejecta_mej_max': (0, 1),
         'disk_wind_ejecta_vej_min': (0, 1), 'disk_wind_ejecta_vej_max': (0, 1)}
    ),
    'one_component_nsbh_ejecta_relation': (
        redback.constraints.one_component_nsbh_ejecta_relation_constraints,
        {'ejecta_mej_min': (0, 1), 'ejecta_mej_max': (0, 1),
         'ejecta_vej_min': (0, 1), 'ejecta_vej_max': (0, 1)}
    ),
    'two_component_nsbh_ejecta_relation': (
        redback.constraints.two_component_nsbh_ejecta_relation_constraints,
        {'dynamical_ejecta_mej_min': (0, 1), 'dynamical_ejecta_mej_max': (0, 1),
         'dynamical_ejecta_vej_min': (0, 1), 'dynamical_ejecta_vej_max': (0, 1),
         'disk_wind_ejecta_mej_min': (0, 1), 'disk_wind_ejecta_mej_max': (0, 1),
         'disk_wind_ejecta_vej_min': (0, 1), 'disk_wind_ejecta_vej_max': (0, 1)}
    ),
}


def _apply_constraints(priors, model):
    setting = _constraint_settings.get(model)
    if setting is None:
        logger.warning(f"No built-in constraints are configured for model {model}. Returning unconstrained priors.")
        return priors
    conversion_function, constraint_bounds = setting
    priors.conversion_function = conversion_function
    for name, bounds in constraint_bounds.items():
        priors[name] = Constraint(*bounds, name=name)
    return priors


def get_priors(model, times=None, y=None, yerr=None, dt=None, constraint=False, **kwargs):
    """
    Get the prior for the given model. If the model is a prompt model, the times, y, and yerr must be provided.

    :param model: String referring to a name of a model implemented in Redback.
    :param times: Time array
    :param y: Y values, arbitrary units
    :param yerr: Error on y values, arbitrary units
    :param dt: time interval
    :param constraint: If True, attach built-in conversion constraints for models with configured constraints.
    :param kwargs: Extra arguments to be passed to the prior function
    :return: priors: PriorDict object
    """
    prompt_prior_functions = dict(gaussian_prompt=get_gaussian_priors, skew_gaussian=get_skew_gaussian_priors,
                                  skew_exponential=get_skew_exponential_priors, fred=get_fred_priors,
                                  fred_extended=get_fred_extended_priors)

    if model in redback.model_library.modules_dict['prompt_models']:
        if times is None:
            times = np.array([0, 100])
        if y is None:
            y = np.array([1, 1e6])
        if yerr is None:
            yerr = np.array([1, 1e3])
        if dt is None:
            dt = np.ones(len(times))
        rate = y * dt
        priors = prompt_prior_functions[model](times=times, y=rate, yerr=yerr)
        priors['background_rate'] = bilby.core.prior.LogUniform(minimum=np.min(rate), maximum=np.max(rate),
                                                                name='background_rate')
        if constraint:
            _apply_constraints(priors, model)
        return priors

    priors = PriorDict()

    if model in redback.model_library.base_models_dict:
        logger.info(f'Setting up prior for base model {model}.')
        logger.info(f'You will need to explicitly set a prior on t0 and or extinction if relevant')

    # Try loading from main priors folder first
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', f'{model}.prior')
        priors.from_file(filename)
        if constraint:
            _apply_constraints(priors, model)
        return priors
    except FileNotFoundError:
        pass  # Continue to try the non_default_priors folder

    # Try loading from non_default_priors subfolder
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', 'non_default_priors', f'{model}.prior')
        priors.from_file(filename)
        if constraint:
            _apply_constraints(priors, model)
        return priors
    except FileNotFoundError:
        pass  # Continue to try plugin prior providers

    # Try plugin prior providers
    from redback.model_library import plugin_prior_providers
    for provider in plugin_prior_providers:
        try:
            result = provider(model)
            if result is not None:
                if constraint:
                    _apply_constraints(result, model)
                return result
        except Exception as e:
            logger.warning(f"Plugin prior provider failed for model '{model}': {e}")

    logger.warning(f'No prior file found for model {model} in either priors or non_default_priors folders. '
                   f'Perhaps you also want to set up the prior for the base model? '
                   f'Or you may need to set up your prior explicitly.')
    logger.info('Returning Empty PriorDict.')
    return priors


def get_prompt_priors(model, times, y, yerr, **kwargs):
    if model == 'gaussian':
        get_gaussian_priors(times=times, y=y, yerr=yerr, **kwargs)


def get_gaussian_priors(times, y, yerr, **kwargs):
    dt = np.min(np.diff(times))
    duration = times[-1] - times[0]
    priors = bilby.core.prior.PriorDict()
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=np.min(yerr), maximum=np.max(y),
                                                      name='amplitude', latex_label=r'$A$')
    priors['sigma'] = bilby.core.prior.LogUniform(minimum=3*dt, maximum=duration, name="sigma", latex_label=r"$\sigma$")
    priors['t_0'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[-1], name="t_0", latex_label=r"$t_0$")
    return priors


def get_skew_gaussian_priors(times, y, yerr, **kwargs):
    priors = get_gaussian_priors(times=times, y=y, yerr=yerr, **kwargs)
    for latex_label, part in zip(
            [r"$\sigma_{\mathrm{rise}}$", r"$\sigma_{\mathrm{fall}}$"], ['rise', 'fall']):
        priors[f'sigma_{part}'] = bilby.core.prior.LogUniform(
            minimum=priors['sigma'].minimum, maximum=priors['sigma'].maximum,
            name=f"sigma_{part}", latex_label=latex_label)
    del priors['sigma']
    return priors


def get_skew_exponential_priors(times, y, yerr, **kwargs):
    priors = get_gaussian_priors(times=times, y=y, yerr=yerr, **kwargs)
    for latex_label, part in zip(
            [r"$\tau_{\mathrm{rise}}$", r"$\tau_{\mathrm{fall}}$"], ['rise', 'fall']):
        priors[f'tau_{part}'] = bilby.core.prior.LogUniform(
            minimum=priors['sigma'].minimum, maximum=priors['sigma'].maximum,
            name=f"tau_{part}", latex_label=latex_label)
    del priors['sigma']
    return priors


def get_fred_priors(times, y, yerr, **kwargs):
    priors = bilby.core.prior.PriorDict()
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=np.min(yerr), maximum=np.max(y),
                                                      name='amplitude', latex_label=r'$A$')
    priors['tau'] = bilby.core.prior.Uniform(minimum=1e-3, maximum=1e3, name="t_0", latex_label=r"$t_0$")
    priors['psi'] = bilby.core.prior.Uniform(minimum=1e-3, maximum=1e3, name=r"\psi")
    priors['delta'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[-1], name=r"\delta")
    return priors


def get_fred_extended_priors(times, y, yerr, **kwargs):
    priors = get_fred_priors(times=times, y=y, yerr=yerr, **kwargs)
    priors['gamma'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e3, name=r"$\gamma$")
    priors['nu'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e3, name=r"$\nu$")
    return priors
