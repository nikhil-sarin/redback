from inspect import isfunction
import numpy as np

from redback.transient_models.fireball_models import predeceleration
from redback.utils import logger, calc_ABmag_from_flux_density, citation_wrapper
import astropy.units as uu

def _get_correct_function(base_model, model_type):
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    extinction_afterglow_base_models = ['tophat', 'cocoon', 'gaussian',
                                        'kn_afterglow', 'cone_afterglow',
                                        'gaussiancore', 'gaussian',
                                        'smoothpowerlaw', 'powerlawcore',
                                        'tophat']
    extinction_supernova_base_models = ['sn_exponential_powerlaw', 'arnett',
                                        'basic_magnetar_powered', 'slsn', 'magnetar_nickel',
                                        'csm_interaction', 'csm_nickel', 'type_1a', 'type_1c',
                                        'general_magnetar_slsn']
    extinction_kilonova_base_models = ['mosfit_bns', 'mosfit_rprocess', 'mosfit_kilonova',
                                       'power_law_stratified_kilonova', 'two_layer_stratified_kilonova',
                                       'three_component_kilonova_model', 'two_component_kilonova_model',
                                       'one_component_kilonova_model', 'one_component_ejecta_relation_model',
                                       'metzger_kilonova_model']
    extinction_tde_base_models = ['tde_analytical', 'tde_semianalytical']
    extinction_magnetar_boosted_base_models = ['mergernova', 'metzger_magnetar_boosted_kilonova_model']

    extinction_model_library = {'kilonova': extinction_kilonova_base_models,
                                'supernova': extinction_supernova_base_models,
                                'afterglow': extinction_afterglow_base_models,
                                'tde': extinction_tde_base_models,
                                'magnetar_boosted': extinction_magnetar_boosted_base_models}

    model_library = {'supernova':'supernova_models', 'afterglow':'afterglow_models',
                     'magnetar_boosted':'magnetar_boosted_ejecta_models', 'tde':'tde_models',
                     'kilonova':'kilonova_models'}

    extinction_base_models = extinction_model_library[model_type]
    module_libary = model_library[model_type]

    if isfunction(base_model):
        function = base_model

    elif base_model not in extinction_base_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict[module_libary][base_model]
    else:
        raise ValueError("Not a valid base model.")

    return function

def _perform_extinction(flux_density, frequency, av, r_v):
    import extinction  # noqa
    # convert to angstrom
    frequency = (frequency * uu.Hz).to(uu.Angstrom).value
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=r_v)
    flux_density = extinction.apply(mag_extinction, flux_density)
    return flux_density


@citation_wrapper('redback')
def extinction_with_supernova_base_model(time, av, **kwargs):
    """
    :param time: time in observer frame in days
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    base_model = kwargs['base_model']
    frequency = kwargs['frequency']
    temp_kwargs = kwargs.copy()
    temp_kwargs['output_format'] = 'flux_density'
    function = _get_correct_function(base_model=base_model, model_type='supernova')
    flux_density = function(time, **temp_kwargs)
    r_v = kwargs.get('r_v', 3.1)
    flux_density = _perform_extinction(flux_density=flux_density, frequency=frequency, av=av, r_v=r_v)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('redback')
def extinction_with_kilonova_base_model(time, av, **kwargs):
    """
    :param time: time in observer frame in days
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    base_model = kwargs['base_model']
    frequency = kwargs['frequency']
    temp_kwargs = kwargs.copy()
    temp_kwargs['output_format'] = 'flux_density'
    function = _get_correct_function(base_model=base_model, model_type='kilonova')
    flux_density = function(time, **temp_kwargs)
    r_v = kwargs.get('r_v', 3.1)
    flux_density = _perform_extinction(flux_density=flux_density, frequency=frequency, av=av, r_v=r_v)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('redback')
def extinction_with_tde_base_model(time, av, **kwargs):
    """
    :param time: time in observer frame in days
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    base_model = kwargs['base_model']
    frequency = kwargs['frequency']
    temp_kwargs = kwargs.copy()
    temp_kwargs['output_format'] = 'flux_density'
    function = _get_correct_function(base_model=base_model, model_type='tde')
    flux_density = function(time, **temp_kwargs)
    r_v = kwargs.get('r_v', 3.1)
    flux_density = _perform_extinction(flux_density=flux_density, frequency=frequency, av=av, r_v=r_v)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('Sarin et al. in prep....')
def extinction_with_magnetar_boosted_base_model(time, av, **kwargs):
    """
    :param time: time in observer frame in days
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    base_model = kwargs['base_model']
    frequency = kwargs['frequency']
    temp_kwargs = kwargs.copy()
    temp_kwargs['output_format'] = 'flux_density'
    function = _get_correct_function(base_model=base_model, model_type='magnetar_boosted')
    flux_density = function(time, **temp_kwargs)
    r_v = kwargs.get('r_v', 3.1)
    flux_density = _perform_extinction(flux_density=flux_density, frequency=frequency, av=av, r_v=r_v)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def extinction_with_afterglow_base_model(time, av, **kwargs):
    """
    :param time: time in observer frame in seconds
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    base_model = kwargs['base_model']
    frequency = kwargs['frequency']
    temp_kwargs = kwargs.copy()
    temp_kwargs['output_format'] = 'flux_density'
    function = _get_correct_function(base_model=base_model, model_type='afterglow')
    flux_density = function(time, **temp_kwargs)
    r_v = kwargs.get('r_v', 3.1)
    flux_density = _perform_extinction(flux_density=flux_density, frequency=frequency, av=av, r_v=r_v)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def extinction_afterglow_galactic_dust_to_gas_ratio(time, lognh, factor=2.21, **kwargs):
    """
    :param time: time in observer frame in seconds
    :param lognh: log10 hydrogen column density
    :param factor: factor to convert nh to av i.e., av = nh/factor
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor
    output = extinction_with_afterglow_base_model(time=time, av=av, **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def extinction_with_predeceleration(time, lognh, factor, **kwargs):
    """
    :param time: time in some unit.
    :param lognh: host galaxy column density
    :param factor: extinction factor
    :param kwargs: all params
    :return: flux or magnitude with extinction applied depending on kwargs
    """
    pass
    # import extinction  # noqa
    # lc = predeceleration(time, **kwargs)
    # lc = np.nan_to_num(lc)
    # factor = factor * 1e21
    # nh = 10 ** lognh
    # av = nh / factor
    # frequency = kwargs['frequency']
    # # convert to angstrom
    # frequency = (frequency * uu.Hz).to(uu.Angstrom).value
    # mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
    # lc = extinction.apply(mag_extinction, lc, inplace=True)
    # if kwargs['output_format'] == 'flux_density':
    #     return lc
    # elif kwargs['output_format'] == 'magnitude':
    #     return calc_ABmag_from_flux_density(lc).value
