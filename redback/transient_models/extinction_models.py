from inspect import isfunction
import numpy as np
import redback.utils
from redback.transient_models.fireball_models import predeceleration
from redback.utils import logger, calc_ABmag_from_flux_density, citation_wrapper, lambda_to_nu
import astropy.units as uu
import redback.sed as sed
from redback.constants import day_to_s

extinction_afterglow_base_models = ['tophat', 'cocoon', 'gaussian',
                                    'kn_afterglow', 'cone_afterglow',
                                    'gaussiancore', 'gaussian',
                                    'smoothpowerlaw', 'powerlawcore',
                                    'tophat','tophat_from_emulator',
                                    'tophat_redback', 'gaussian_redback', 'twocomponent_redback',
                                    'powerlaw_redback', 'alternativepowerlaw_redback', 'doublegaussian_redback',
                                    'tophat_redback_refreshed', 'gaussian_redback_refreshed',
                                    'twocomponent_redback_refreshed','powerlaw_redback_refreshed',
                                    'alternativepowerlaw_redback_refreshed', 'doublegaussian_redback_refreshed',
                                    'jetsimpy_tophat', 'jetsimpy_gaussian', 'jetsimpy_powerlaw',
                                    'vegas_gaussian', 'vegas_powerlaw', 'vegas_tophat',
                                    'vegas_twocomponent', 'vegas_powerlaw_wing', 'vegas_step',
                                    'vegas_steppowerlaw']

extinction_general_synchrotron_models = ['pwn', 'kilonova_afterglow_redback', 'kilonova_afterglow_nakarpiran',
                                         'thermal_synchrotron_lnu', 'thermal_synchrotron_fluxdensity',
                                         'tde_synchrotron', 'synchrotron_massloss', 'synchrotron_ism',
                                         'synchrotron_pldensity', 'thermal_synchrotron_v2_lnu',
                                         'thermal_synchrotron_v2_fluxdensity']
extinction_stellar_interaction_models = ['wr_bh_merger']
extinction_integrated_flux_afterglow_models = extinction_afterglow_base_models

extinction_supernova_base_models = ['sn_exponential_powerlaw', 'arnett', 'shock_cooling_and_arnett',
                                    'basic_magnetar_powered', 'slsn', 'magnetar_nickel',
                                    'csm_interaction', 'csm_nickel', 'type_1a', 'type_1c',
                                    'general_magnetar_slsn','general_magnetar_driven_supernova', 'sn_fallback',
                                    'csm_shock_and_arnett', 'shocked_cocoon_and_arnett',
                                    'csm_shock_and_arnett_two_rphots', 'nickelmixing',
                                    'sn_nickel_fallback', 'shockcooling_morag_and_arnett',
                                    'shockcooling_sapirwaxman_and_arnett',
                                    'csm_shock_and_arnett', 'shocked_cocoon_and_arnett',
                                    'csm_shock_and_arnett_two_rphots', 'typeII_surrogate_sarin25',
                                    'shocked_cocoon_csm_and_arnett']
extinction_kilonova_base_models = ['nicholl_bns', 'mosfit_rprocess', 'mosfit_kilonova',
                                   'power_law_stratified_kilonova','bulla_bns_kilonova',
                                   'bulla_nsbh_kilonova', 'kasen_bns_kilonova','two_layer_stratified_kilonova',
                                   'three_component_kilonova_model', 'two_component_kilonova_model',
                                   'one_component_kilonova_model', 'one_component_ejecta_relation',
                                   'one_component_ejecta_relation_projection', 'two_component_bns_ejecta_relation',
                                   'polytrope_eos_two_component_bns', 'one_component_nsbh_ejecta_relation',
                                   'two_component_nsbh_ejecta_relation','one_comp_kne_rosswog_heatingrate',
                                   'two_comp_kne_rosswog_heatingrate','metzger_kilonova_model']

extinction_tde_base_models = ['tde_analytical', 'tde_semianalytical', 'gaussianrise_cooling_envelope',
                              'cooling_envelope', 'bpl_cooling_envelope', 'tde_fallback',
                              'fitted', 'fitted_pldecay', 'fitted_expdecay', 'stream_stream_tde']
extinction_magnetar_driven_base_models = ['basic_mergernova', 'general_mergernova', 'general_mergernova_thermalisation',
                                          'general_mergernova_evolution', 'metzger_magnetar_driven_kilonova_model',
                                          'general_metzger_magnetar_driven', 'general_metzger_magnetar_driven_thermalisation',
                                          'general_metzger_magnetar_driven_evolution']
extinction_shock_powered_base_models = ['shocked_cocoon', 'shock_cooling', 'csm_shock_breakout',
                                        'shockcooling_morag', 'shockcooling_sapirandwaxman', 'shocked_cocoon_csm']
extinction_stellar_interaction_models = ['wr_bh_merger']

extinction_model_library = {'kilonova': extinction_kilonova_base_models,
                            'supernova': extinction_supernova_base_models,
                            'general_synchrotron': extinction_general_synchrotron_models,
                            'stellar_interaction': extinction_stellar_interaction_models,
                            'afterglow': extinction_afterglow_base_models,
                            'tde': extinction_tde_base_models,
                            'magnetar_driven': extinction_magnetar_driven_base_models,
                            'shock_powered': extinction_shock_powered_base_models,
                            'stellar_interaction': extinction_stellar_interaction_models,
                            'integrated_flux_afterglow': extinction_integrated_flux_afterglow_models}                            

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
def extinction_with_stellar_interaction_base_model(time, av_host, **kwargs):
    """
    Extinction with models implemented in stellar_interaction_models

    :param time: time in observer frame in days
    :param av_host: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
        and r_v, default is 3.1
    :return: set by kwargs['output_format'] - 'flux_density', 'magnitude', 'flux' with extinction applied
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
