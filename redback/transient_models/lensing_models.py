from inspect import isfunction
import numpy as np
import redback.utils
from redback.utils import logger, citation_wrapper
import redback.sed as sed

# Lensing base models for different transient types
lensing_afterglow_base_models = ['tophat', 'cocoon', 'gaussian',
                                 'kn_afterglow', 'cone_afterglow',
                                 'gaussiancore', 'gaussian',
                                 'smoothpowerlaw', 'powerlawcore',
                                 'tophat','tophat_from_emulator',
                                 'tophat_redback', 'gaussian_redback', 'twocomponent_redback',
                                 'powerlaw_redback', 'alternativepowerlaw_redback', 'doublegaussian_redback',
                                 'tophat_redback_refreshed', 'gaussian_redback_refreshed',
                                 'twocomponent_redback_refreshed','powerlaw_redback_refreshed',
                                 'alternativepowerlaw_redback_refreshed', 'doublegaussian_redback_refreshed',
                                 'jetsimpy_tophat', 'jetsimpy_gaussian', 'jetsimpy_powerlaw']

lensing_general_synchrotron_models = ['pwn', 'kilonova_afterglow_redback', 'kilonova_afterglow_nakarpiran',
                                      'thermal_synchrotron_lnu', 'thermal_synchrotron_fluxdensity',
                                      'tde_synchrotron', 'synchrotron_massloss', 'synchrotron_ism',
                                      'synchrotron_pldensity', 'thermal_synchrotron_v2_lnu',
                                      'thermal_synchrotron_v2_fluxdensity']
lensing_stellar_interaction_models = ['wr_bh_merger']
lensing_integrated_flux_afterglow_models = lensing_afterglow_base_models

lensing_supernova_base_models = ['sn_exponential_powerlaw', 'arnett', 'shock_cooling_and_arnett',
                                 'basic_magnetar_powered', 'slsn', 'magnetar_nickel',
                                 'csm_interaction', 'csm_nickel', 'type_1a', 'type_1c',
                                 'general_magnetar_slsn','general_magnetar_driven_supernova', 'sn_fallback',
                                 'csm_shock_and_arnett', 'shocked_cocoon_and_arnett',
                                 'csm_shock_and_arnett_two_rphots', 'nickelmixing',
                                 'sn_nickel_fallback', 'shockcooling_morag_and_arnett',
                                 'shockcooling_sapirandwaxman_and_arnett',
                                 'csm_shock_and_arnett', 'shocked_cocoon_and_arnett',
                                 'csm_shock_and_arnett_two_rphots', 'typeII_surrogate_sarin25']
lensing_kilonova_base_models = ['nicholl_bns', 'mosfit_rprocess', 'mosfit_kilonova',
                                'power_law_stratified_kilonova','bulla_bns_kilonova',
                                'bulla_nsbh_kilonova', 'kasen_bns_kilonova','two_layer_stratified_kilonova',
                                'three_component_kilonova_model', 'two_component_kilonova_model',
                                'one_component_kilonova_model', 'one_component_ejecta_relation',
                                'one_component_ejecta_relation_projection', 'two_component_bns_ejecta_relation',
                                'polytrope_eos_two_component_bns', 'one_component_nsbh_ejecta_relation',
                                'two_component_nsbh_ejecta_relation','one_comp_kne_rosswog_heatingrate',
                                'two_comp_kne_rosswog_heatingrate','metzger_kilonova_model']

lensing_tde_base_models = ['tde_analytical', 'tde_semianalytical', 'gaussianrise_cooling_envelope',
                           'cooling_envelope', 'bpl_cooling_envelope', 'tde_fallback',
                           'fitted', 'fitted_pldecay', 'fitted_expdecay', 'stream_stream_tde']
lensing_magnetar_driven_base_models = ['basic_mergernova', 'general_mergernova', 'general_mergernova_thermalisation',
                                       'general_mergernova_evolution', 'metzger_magnetar_driven_kilonova_model',
                                       'general_metzger_magnetar_driven', 'general_metzger_magnetar_driven_thermalisation',
                                       'general_metzger_magnetar_driven_evolution']
lensing_shock_powered_base_models = ['shocked_cocoon', 'shock_cooling', 'csm_shock_breakout',
                                     'shockcooling_morag', 'shockcooling_sapirandwaxman']

lensing_model_library = {'kilonova': lensing_kilonova_base_models,
                         'supernova': lensing_supernova_base_models,
                         'general_synchrotron': lensing_general_synchrotron_models,
                         'stellar_interaction': lensing_stellar_interaction_models,
                         'afterglow': lensing_afterglow_base_models,
                         'tde': lensing_tde_base_models,
                         'magnetar_driven': lensing_magnetar_driven_base_models,
                         'shock_powered': lensing_shock_powered_base_models,
                         'integrated_flux_afterglow': lensing_integrated_flux_afterglow_models}

model_library = {'supernova': 'supernova_models', 'afterglow': 'afterglow_models',
                 'magnetar_driven': 'magnetar_driven_ejecta_models', 'tde': 'tde_models',
                 'kilonova': 'kilonova_models', 'shock_powered': 'shock_powered_models',
                 'integrated_flux_afterglow': 'afterglow_models',
                 'stellar_interaction': 'stellar_interaction_models',
                 'general_synchrotron': 'general_synchrotron_models'}

def _get_correct_function(base_model, model_type=None):
    """
    Gets the correct function to use for the base model specified

    :param base_model: string or a function
    :param model_type: type of model, could be None if using a function as input
    :return: function; function to evaluate
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    lensing_base_models = lensing_model_library[model_type]
    module_library = model_library[model_type]

    if isfunction(base_model):
        function = base_model

    elif base_model not in lensing_base_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict[module_library][base_model]
    else:
        raise ValueError("Not a valid base model.")

    return function


def _perform_lensing(time, flux_density_or_spectra_function, nimages, **kwargs):
    """
    Apply gravitational lensing effect to create multiple images with time delays and magnifications

    :param time: time array in days (observer frame)
    :param flux_density_or_spectra_function: function that evaluates the base model at given times
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must include for each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless)
    :return: combined flux density from all lensed images
    """
    # Initialize output as zeros
    lensed_output = None

    # Sum contributions from all images
    for i in range(1, nimages + 1):
        dt_key = f'dt_{i}'
        mu_key = f'mu_{i}'

        # Get time delay and magnification for this image
        dt = kwargs.get(dt_key, 0.0 if i == 1 else 0.0)
        mu = kwargs.get(mu_key, 1.0 if i == 1 else 0.0)

        # Shift time by the time delay
        shifted_time = time - dt

        # Evaluate base model at shifted time
        image_flux = flux_density_or_spectra_function(shifted_time)

        # Apply magnification and add to total
        if lensed_output is None:
            lensed_output = mu * image_flux
        else:
            lensed_output += mu * image_flux

    return lensed_output


def _evaluate_lensing_model(time, nimages=2, model_type=None, **kwargs):
    """
    Generalized evaluate lensing function that creates multiple images with time delays and magnifications

    :param time: time in days (observer frame)
    :param nimages: number of lensed images (default: 2)
    :param model_type: None, or one of the types implemented
    :param kwargs: Must include all parameters for base_model plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless)
    :return: flux/magnitude/spectra with lensing applied
    """
    base_model = kwargs['base_model']
    if kwargs['base_model'] in ['thin_shell_supernova', 'homologous_expansion_supernova']:
        kwargs['base_model'] = kwargs.get('submodel', 'arnett_bolometric')

    if kwargs['output_format'] == 'flux_density':
        temp_kwargs = kwargs.copy()
        temp_kwargs['output_format'] = 'flux_density'

        # Remove lensing parameters from temp_kwargs to avoid passing them to base model
        for i in range(1, nimages + 1):
            temp_kwargs.pop(f'dt_{i}', None)
            temp_kwargs.pop(f'mu_{i}', None)

        function = _get_correct_function(base_model=base_model, model_type=model_type)

        # Create a function that evaluates the base model
        def evaluate_base_model(t):
            return function(t, **temp_kwargs)

        # Apply lensing effect
        flux_density = _perform_lensing(
            time=time,
            flux_density_or_spectra_function=evaluate_base_model,
            nimages=nimages,
            **kwargs
        )

        return flux_density

    else:
        temp_kwargs = kwargs.copy()
        temp_kwargs['output_format'] = 'spectra'
        time_obs = time

        # Remove lensing parameters from temp_kwargs
        for i in range(1, nimages + 1):
            temp_kwargs.pop(f'dt_{i}', None)
            temp_kwargs.pop(f'mu_{i}', None)

        function = _get_correct_function(base_model=base_model, model_type=model_type)

        # For spectra output, we need to handle it differently
        # Get the first image to determine the structure
        spectra_tuple = function(time, **temp_kwargs)

        # Initialize combined spectra
        flux_density = np.zeros_like(spectra_tuple.spectra)
        lambdas = spectra_tuple.lambdas
        time_observer_frame = spectra_tuple.time

        # Sum contributions from all images
        for i in range(1, nimages + 1):
            dt_key = f'dt_{i}'
            mu_key = f'mu_{i}'

            dt = kwargs.get(dt_key, 0.0 if i == 1 else 0.0)
            mu = kwargs.get(mu_key, 1.0 if i == 1 else 0.0)

            # Shift time by the time delay
            shifted_time = time - dt

            # Evaluate base model at shifted time
            image_spectra = function(shifted_time, **temp_kwargs)

            # Apply magnification and add to total
            flux_density += mu * image_spectra.spectra

        return sed.get_correct_output_format_from_spectra(
            time=time_obs,
            time_eval=time_observer_frame,
            spectra=flux_density,
            lambda_array=lambdas,
            **kwargs
        )


@citation_wrapper('redback')
def lensing_with_function(time, nimages=2, **kwargs):
    """
    Gravitational lensing model when using your own specified function

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied

    Example:
        For a doubly-imaged system (nimages=2):
        kwargs = {
            'base_model': 'arnett',
            'dt_1': 0.0,  # first image (reference)
            'mu_1': 1.0,  # first image magnification
            'dt_2': 10.0,  # second image delayed by 10 days
            'mu_2': 0.5,  # second image magnification
            ... # other parameters for 'arnett' model
        }
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type=None, **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_supernova_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in supernova_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied

    Example:
        For multiply-imaged supernova (nimages=4):
        kwargs = {
            'base_model': 'arnett',
            'dt_1': 0.0, 'mu_1': 1.5,
            'dt_2': 5.0, 'mu_2': 1.2,
            'dt_3': 15.0, 'mu_3': 0.8,
            'dt_4': 20.0, 'mu_4': 0.6,
            ... # other parameters for 'arnett' model
        }
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='supernova', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_kilonova_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in kilonova_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='kilonova', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_tde_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in tde_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='tde', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_shock_powered_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in shock_powered_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='shock_powered', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_magnetar_driven_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in magnetar_driven_ejecta_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='magnetar_driven', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_stellar_interaction_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in stellar_interaction_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='stellar_interaction', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_general_synchrotron_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in general_synchrotron_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='general_synchrotron', **kwargs)
    return output


@citation_wrapper('redback')
def lensing_with_afterglow_base_model(time, nimages=2, **kwargs):
    """
    Gravitational lensing with models implemented in afterglow_models

    :param time: time in observer frame in days
    :param nimages: number of lensed images (default: 2)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        For each image i (1 to nimages):
        - dt_i: time delay for image i in days (observer frame)
        - mu_i: magnification factor for image i (dimensionless, can be negative for parity flip)
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with lensing applied
        Note that only sed variant models can take magnitude as an output

    Example:
        For doubly-imaged GRB afterglow:
        kwargs = {
            'base_model': 'tophat',
            'dt_1': 0.0, 'mu_1': 2.0,
            'dt_2': 30.0, 'mu_2': 1.5,
            ... # other parameters for 'tophat' model
        }
    """
    output = _evaluate_lensing_model(time=time, nimages=nimages, model_type='afterglow', **kwargs)
    return output
