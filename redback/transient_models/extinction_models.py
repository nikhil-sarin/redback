from inspect import isfunction
import numpy as np
import redback.utils
from redback.transient_models.fireball_models import predeceleration
from redback.utils import logger, calc_ABmag_from_flux_density, citation_wrapper, lambda_to_nu
import astropy.units as uu
import redback.sed as sed
from redback.constants import day_to_s

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
    # Import model library in function to avoid circular dependency.
    from redback.model_library import modules, modules_dict, plugin_module_model_types

    if isfunction(base_model):
        return base_model

    if not isinstance(base_model, str):
        raise ValueError("base_model must be a string name or a callable function")

    if model_type is None:
        # Search all modules for the model name
        for module_name, models in modules_dict.items():
            if base_model in models:
                return models[base_model]
        raise ValueError(
            f"Model '{base_model}' not found in any module. "
            f"Ensure the model name is correct and the module is loaded."
        )

    if model_type not in model_library:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available types: {list(model_library.keys())}"
        )
    module_name = model_library[model_type]
    if module_name not in modules_dict:
        raise ValueError(
            f"Module '{module_name}' for model type '{model_type}' not found. "
            f"Available modules: {list(modules_dict.keys())}"
        )
    module_models = modules_dict[module_name]
    if base_model in module_models:
        return module_models[base_model]

    builtin_module_names = {module.__name__.split('.')[-1] for module in modules}
    for plugin_name, plugin_models in modules_dict.items():
        if plugin_name in builtin_module_names:
            continue
        if model_type not in plugin_module_model_types.get(plugin_name, set()):
            continue
        if base_model in plugin_models:
            return plugin_models[base_model]

    plugin_module_names = [
        name for name in modules_dict
        if name not in builtin_module_names and model_type in plugin_module_model_types.get(name, set())
    ]
    raise ValueError(
        f"Model '{base_model}' not found in '{module_name}' or {model_type} plugin modules. "
        f"Available models in '{module_name}': {list(module_models.keys())}. "
        f"Available {model_type} plugin modules: {plugin_module_names}"
    )

def _perform_extinction(flux_density, angstroms, av_host, rv_host, av_mw=0.0, rv_mw=3.1,
                        host_law='fitzpatrick99', mw_law='fitzpatrick99', **kwargs):
    """
    Apply host galaxy and/or Milky Way extinction to flux density

    :param flux_density: flux density in mJy outputted by the model
    :param angstroms: wavelength in angstroms (observer frame)
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param rv_host: extinction parameter for host galaxy (default 3.1)
    :param av_mw: V-band extinction from Milky Way in magnitudes
    :param rv_mw: extinction parameter for Milky Way (default 3.1)
    :param redshift: source redshift (needed for host extinction)
    :param host_law: extinction law for host galaxy
                     ('fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89')
    :param mw_law: extinction law for Milky Way
                   ('fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89')
    :param kwargs: additional parameters for specific extinction laws
    :return: flux density with extinction applied
    """
    import extinction

    redshift = kwargs['redshift']
    if isinstance(angstroms, float):
        angstroms = np.array([angstroms])

    # Available extinction laws
    extinction_laws = {
        'fitzpatrick99': extinction.fitzpatrick99,
        'fm07': extinction.fm07,
        'calzetti00': extinction.calzetti00,
        'odonnell94': extinction.odonnell94,
        'ccm89': extinction.ccm89
    }

    # Validate extinction laws
    if host_law not in extinction_laws:
        raise ValueError(f"Unknown host extinction law: {host_law}. "
                         f"Available: {list(extinction_laws.keys())}")
    if mw_law not in extinction_laws:
        raise ValueError(f"Unknown MW extinction law: {mw_law}. "
                         f"Available: {list(extinction_laws.keys())}")

    flux_extincted = flux_density.copy() if hasattr(flux_density, 'copy') else np.array(flux_density)

    # Apply host galaxy extinction (in rest frame)
    if av_host > 0:
        # Convert observer frame to rest frame wavelengths
        angstroms_rest = angstroms / (1 + redshift)

        # Get host extinction law function
        host_extinction_func = extinction_laws[host_law]

        # Calculate extinction - handle different function signatures
        try:
            if host_law in ['fitzpatrick99', 'fm07', 'odonnell94', 'ccm89']:
                mag_extinction_host = host_extinction_func(angstroms_rest, av_host, rv_host)
            elif host_law == 'calzetti00':
                # Calzetti law doesn't use R_V parameter
                mag_extinction_host = host_extinction_func(angstroms_rest, av_host)
        except Exception as e:
            raise ValueError(f"Error applying {host_law} extinction law: {e}")

        # Cap extreme extinction values
        if av_host < 10:
            mask = mag_extinction_host > 10
            mag_extinction_host[mask] = 0

        # Apply host extinction
        flux_extincted = extinction.apply(mag_extinction_host, flux_extincted)

    # Apply Milky Way extinction (in observer frame)
    if av_mw > 0:
        # MW extinction applies to observed wavelengths
        mw_extinction_func = extinction_laws[mw_law]

        # Calculate extinction
        try:
            if mw_law in ['fitzpatrick99', 'fm07', 'odonnell94', 'ccm89']:
                mag_extinction_mw = mw_extinction_func(angstroms, av_mw, rv_mw)
            elif mw_law == 'calzetti00':
                mag_extinction_mw = mw_extinction_func(angstroms, av_mw)
        except Exception as e:
            raise ValueError(f"Error applying {mw_law} extinction law: {e}")

        # Cap extreme extinction values
        if av_mw < 10:
            mask = mag_extinction_mw > 10
            mag_extinction_mw[mask] = 0

        # Apply MW extinction
        flux_extincted = extinction.apply(mag_extinction_mw, flux_extincted)

    return flux_extincted


def _evaluate_extinction_model(time, av_host, av_mw=0.0, model_type=None, **kwargs):
    """
    Generalised evaluate extinction function with host and MW extinction

    :param time: time in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param av_mw: V-band extinction from Milky Way in magnitudes
    :param model_type: None, or one of the types implemented
    :param kwargs: Must include all parameters for base_model plus:
        - redshift: source redshift (required for host extinction)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
    :return: flux/magnitude with extinction applied
    """
    base_model = kwargs['base_model']
    if kwargs['base_model'] in ['thin_shell_supernova', 'homologous_expansion_supernova']:
        kwargs['base_model'] = kwargs.get('submodel', 'arnett_bolometric')

    # Extract extinction parameters
    rv_host = kwargs.pop('rv_host', 3.1)
    rv_mw = kwargs.pop('rv_mw', 3.1)
    host_law = kwargs.pop('host_law', 'fitzpatrick99')
    mw_law = kwargs.pop('mw_law', 'fitzpatrick99')

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        if isinstance(frequency, float):
            frequency = np.ones(len(time)) * frequency

        angstroms = redback.utils.nu_to_lambda(frequency)

        temp_kwargs = kwargs.copy()
        temp_kwargs['output_format'] = 'flux_density'
        function = _get_correct_function(base_model=base_model, model_type=model_type)
        flux_density = function(time, **temp_kwargs)

        # Apply extinction
        flux_density = _perform_extinction(
            flux_density=flux_density,
            angstroms=angstroms,
            av_host=av_host,
            rv_host=rv_host,
            av_mw=av_mw,
            rv_mw=rv_mw,
            host_law=host_law,
            mw_law=mw_law,
            **kwargs
        )
        return flux_density

    else:
        temp_kwargs = kwargs.copy()
        temp_kwargs['output_format'] = 'spectra'
        time_obs = time
        function = _get_correct_function(base_model=base_model, model_type=model_type)
        spectra_tuple = function(time, **temp_kwargs)

        flux_density = spectra_tuple.spectra
        lambdas = spectra_tuple.lambdas
        time_observer_frame = spectra_tuple.time

        # Apply extinction
        flux_density = _perform_extinction(
            flux_density=flux_density,
            angstroms=lambdas,
            av_host=av_host,
            rv_host=rv_host,
            av_mw=av_mw,
            rv_mw=rv_mw,
            host_law=host_law,
            mw_law=mw_law,
            **kwargs
        )

        return sed.get_correct_output_format_from_spectra(
            time=time_obs,
            time_eval=time_observer_frame,
            spectra=flux_density,
            lambda_array=lambdas,
            **kwargs
        )

@citation_wrapper('redback')
def extinction_with_function(time, av_host, **kwargs):
    """
    Extinction model when using your own specified function

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type=None, **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_supernova_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in supernova_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='supernova', **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_kilonova_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in kilonova_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='kilonova', **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_tde_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in tde_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='tde', **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_shock_powered_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in shock_powered_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='shock_powered', **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_magnetar_driven_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in magnetar_driven_ejecta_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='magnetar_driven', **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_stellar_interaction_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in stellar_interaction_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='stellar_interaction', **kwargs)
    return output

@citation_wrapper('redback')
def extinction_with_general_synchrotron_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in general_synchrotron_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='general_synchrotron', **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def extinction_with_afterglow_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in afterglow_models

    :param time: time in observer frame in days
    :param av_host: V-band extinction from host galaxy in magnitudes
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
        Note that only sed variant models can take magnitude as an output
    """
    output = _evaluate_extinction_model(time=time, av_host=av_host, model_type='afterglow', **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021arXiv210601556S/abstract')
def extinction_afterglow_galactic_dust_to_gas_ratio(time, lognh, factor=2.21, **kwargs):
    """
    Extinction with afterglow models using galactic dust-to-gas ratio

    :param time: time in observer frame in days
    :param lognh: log10 hydrogen column density
    :param factor: factor to convert nh to av i.e., av = nh/factor (default 2.21)
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model'] plus:
        - redshift: source redshift (required)
        - av_mw: MW V-band extinction in magnitudes (default 0.0)
        - rv_host: host R_V parameter (default 3.1)
        - rv_mw: MW R_V parameter (default 3.1)
        - host_law: host extinction law (default 'fitzpatrick99')
        - mw_law: MW extinction law (default 'fitzpatrick99')
        Available extinction laws: 'fitzpatrick99', 'fm07', 'calzetti00', 'odonnell94', 'ccm89'
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux', 'spectra' with extinction applied
    """
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor
    output = extinction_with_afterglow_base_model(time=time, av_host=av, **kwargs)
    return output


def _extinction_with_predeceleration(time, lognh, factor, **kwargs):
    """
    :param time: time in some unit.
    :param lognh: host galaxy column density
    :param factor: extinction factor
    :param kwargs: all params
    :return: set by output format kwarg - 'flux_density', 'magnitude', 'flux' with extinction applied
    """
    import extinction  # noqa
    lc = predeceleration(time, **kwargs)
    lc = np.nan_to_num(lc)
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor
    frequency = kwargs['frequency']
    # convert to angstrom
    frequency = (frequency * uu.Hz).to(uu.Angstrom).value
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
    lc = extinction.apply(mag_extinction, lc, inplace=True)
    if kwargs['output_format'] == 'flux_density':
        return lc
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(lc).value
