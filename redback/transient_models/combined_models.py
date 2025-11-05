import redback.transient_models.extinction_models as em
import redback.transient_models as tm
from redback.utils import nu_to_lambda
from redback.utils import citation_wrapper

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract, https://ui.adsabs.harvard.edu/abs/2020ApJ...891..152H/abstract')
def tophat_and_twolayerstratified(time, redshift, av, thv, loge0, thc, logn0, p, logepse,
                                  logepsb, ksin, g0, mej, vej_1, vej_2, kappa, beta, **kwargs):
    
    """
    function to combine the flux density signals of a tophat afterglow and a two layer stratified kilonova with extinction

    Parameters
    ----------
    time
        time in days in observer frame
    redshift
        source redshift
    av
        V-band extinction from host galaxy in magnitudes
    thv
        viewing angle in radians
    loge0
        log10 on axis isotropic equivalent energy
    thc
        half width of jet core/jet opening angle in radians
    beta
        power law index of density profile
    logn0
        log10 number density of ISM in cm^-3
    p
        electron distribution power law index. Must be greater than 2.
    logepse
        log10 fraction of thermal energy in electrons
    logepsb
        log10 fraction of thermal energy in magnetic field
    ksin
        fraction of electrons that get accelerated
    g0
        initial lorentz factor
    mej
        ejecta mass in solar masses
    vej_1
        velocity of inner shell in c
    vej_2
        velocity of outer shell in c
    kappa
        constant gray opacity
    kwargs
        Additional keyword arguments e.g., for extinction or the models
    r_v
        extinction parameter, defaults to 3.1
    spread
        whether jet can spread, defaults to False
    latres
        latitudinal resolution for structured jets, defaults to 2
    tres
        time resolution of shock evolution, defaults to 100
    spectype
        whether to have inverse compton, defaults to 0, i.e., no inverse compton. Change to 1 for including inverse compton emission.
    frequency
        frequency to calculate - Must be same length as time array or a single number

    Returns
    -------
        flux density signal with extinction added
    """
    kwargs['output_format'] = 'flux_density'
    afterglow = tm.afterglow_models.tophat(time=time, redshift=redshift, thv=thv, loge0=loge0, thc=thc, logn0=logn0,
                                           p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0, **kwargs)
    kilonova = tm.kilonova_models.two_layer_stratified_kilonova(time=time, redshift=redshift, mej=mej, vej_1=vej_1,
                                                                vej_2=vej_2, kappa=kappa, beta=beta, **kwargs)
    combined = afterglow+kilonova
    r_v = kwargs.get('r_v', 3.1)
    # correct for extinction
    angstroms = nu_to_lambda(kwargs['frequency'])
    combined = em._perform_extinction(flux_density=combined, angstroms=angstroms, av_host=av, rv_host=r_v,
                                      redshift=redshift, **kwargs)
    return combined

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract, redback')
def tophat_and_twocomponent(time, redshift, av, thv, loge0, thc, logn0,
                            p, logepse, logepsb, ksin, g0, mej_1, vej_1,
                            temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    """
    function to combine the flux density signals of a tophat afterglow and a two component kilonova with extinction added

    Parameters
    ----------
    time
        time in days in observer frame
    redshift
        source redshift
    av
        V-band extinction from host galaxy in magnitudes
    thv
        viewing angle in radians
    loge0
        log10 on axis isotropic equivalent energy
    thc
        half width of jet core/jet opening angle in radians
    beta
        index for power-law structure, theta^-b
    logn0
        log10 number density of ISM in cm^-3
    p
        electron distribution power law index. Must be greater than 2.
    logepse
        log10 fraction of thermal energy in electrons
    logepsb
        log10 fraction of thermal energy in magnetic field
    ksin
        fraction of electrons that get accelerated
    g0
        initial lorentz factor
    mej_1
        ejecta mass in solar masses of first component
    vej_1
        minimum initial velocity of first component
    kappa_1
        gray opacity of first component
    temperature_floor_1
        floor temperature of first component
    mej_2
        ejecta mass in solar masses of second component
    vej_2
        minimum initial velocity of second component
    temperature_floor_2
        floor temperature of second component
    kappa_2
        gray opacity of second component
    kwargs
        Additional keyword arguments e.g., for extinction or the models
    r_v
        extinction parameter, defaults to 3.1
    spread
        whether jet can spread, defaults to False
    latres
        latitudinal resolution for structured jets, defaults to 2
    tres
        time resolution of shock evolution, defaults to 100
    spectype
        whether to have inverse compton, defaults to 0, i.e., no inverse compton. Change to 1 for including inverse compton emission.
    frequency
        frequency to calculate - Must be same length as time array or a single number

    Returns
    -------
        flux density signal with extinction added
    """
    
    kwargs['output_format'] = 'flux_density'
    afterglow = tm.afterglow_models.tophat(time=time, redshift=redshift, thv=thv, loge0=loge0, thc=thc, logn0=logn0,
                                           p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0, **kwargs)
    kilonova = tm.kilonova_models.two_component_kilonova_model(time=time, redshift=redshift, av=av,
                                                      mej_1=mej_1, vej_1=vej_1, temperature_floor_1=temperature_floor_1,
                                                      kappa_1=kappa_1, mej_2=mej_2, vej_2=vej_2,
                                                      temperature_floor_2=temperature_floor_2, kappa_2=kappa_2, **kwargs)
    
    combined = afterglow + kilonova
    r_v = kwargs.get('r_v', 3.1)
    # correct for extinction
    angstroms = nu_to_lambda(kwargs['frequency'])
    combined = em._perform_extinction(flux_density=combined, angstroms=angstroms, av_host=av, rv_host=r_v,
                                      redshift=redshift, **kwargs)
    return combined

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def tophat_and_arnett(time, av, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, f_nickel, mej, **kwargs):
    
    """
    function to combine the flux density signals of a tophat afterglow and an arnett supernova with extinction added

    Parameters
    ----------
    time
        time in days in observer frame
    redshift
        source redshift
    av
        V-band extinction from host galaxy in magnitudes
    thv
        viewing angle in radians
    loge0
        log10 on axis isotropic equivalent energy
    thc
        half width of jet core/jet opening angle in radians
    beta
        index for power-law structure, theta^-b
    logn0
        log10 number density of ISM in cm^-3
    p
        electron distribution power law index. Must be greater than 2.
    logepse
        log10 fraction of thermal energy in electrons
    logepsb
        log10 fraction of thermal energy in magnetic field
    ksin
        fraction of electrons that get accelerated
    g0
        initial lorentz factor
    f_nickel
        fraction of nickel mass
    mej
        total ejecta mass in solar masses
    kwargs
        Additional keyword arguments Must include all the kwargs required by the specific interaction_process, photosphere, sed methods used e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor and any other kwargs for the specific models.
    r_v
        extinction parameter, defaults to 3.1
    spread
        whether jet can spread, defaults to False
    latres
        latitudinal resolution for structured jets, defaults to 2
    tres
        time resolution of shock evolution, defaults to 100
    spectype
        whether to have inverse compton, defaults to 0, i.e., no inverse compton. Change to 1 for including inverse compton emission.
    interaction_process
        Default is Diffusion. Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    photosphere
        Default is TemperatureFloor. kwargs must have vej or relevant parameters if using different photosphere model
    sed
        Default is blackbody.
    frequency
        frequency to calculate - Must be same length as time array or a single number

    Returns
    -------
        flux density with extinction added
    """
    
    kwargs['output_format'] = 'flux_density'
    afterglow = tm.afterglow_models.tophat(time=time, redshift=redshift, thv=thv, loge0=loge0, thc=thc, logn0=logn0,
                                           p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0, **kwargs)
    kwargs['base_model'] = 'arnett'
    supernova = tm.supernova_models.arnett(time=time, redshift=redshift, f_nickel=f_nickel, mej=mej, **kwargs)
    combined = afterglow + supernova
    r_v = kwargs.get('r_v', 3.1)
    # correct for extinction
    angstroms = nu_to_lambda(kwargs['frequency'])
    combined = em._perform_extinction(flux_density=combined, angstroms=angstroms, av_host=av, rv_host=r_v,
                                      redshift=redshift, **kwargs)
    return combined

@citation_wrapper('redback, and any citations for the specific model you use')
def afterglow_and_optical(time, redshift, av, **model_kwargs):
    """
    function to combine the signals of any afterglow and any other optical transient with extinction added

    Parameters
    ----------
    time
        time in days in observer frame
    redshift
        source redshift
    av
        V-band extinction from host galaxy in magnitudes
    model_kwargs
        kwargs shared by models frequency and r_v (extinction paramater defaults to 3.1)
    afterglow_kwargs
        dictionary of  parameters required by the afterglow transient model specified by 'base_model' and any additional keyword arguments. Refer to model documentation for details.
    optical_kwargs
        dictionary of parameters required by the optical transient model specifed by 'base_model' and any additional keyword arguments. Note the base model must correspond to the given model type. Refer to model documentation for details.

    Returns
    -------
        flux density signal with extinction added
    """

    from redback.model_library import all_models_dict
    optical_kwargs = model_kwargs['optical_kwargs']
    afterglow_kwargs = model_kwargs['afterglow_kwargs']
    model_kwargs['output_format']= model_kwargs.get('output_format', 'flux_density')

    _afterglow_kwargs = afterglow_kwargs.copy()
    _afterglow_kwargs.update(model_kwargs)

    _optical_kwargs = optical_kwargs.copy()
    _optical_kwargs.update(model_kwargs)

    afterglow_function = all_models_dict[_afterglow_kwargs['base_model']]
    afterglow = afterglow_function(time=time, redshift=redshift,  **_afterglow_kwargs)

    optical_function = all_models_dict[_optical_kwargs['base_model']]
    optical = optical_function(time=time, redshift=redshift, **_optical_kwargs)

    combined = afterglow + optical
    r_v = model_kwargs.get('r_v', 3.1)
    # correct for extinction
    angstroms = nu_to_lambda(model_kwargs['frequency'])
    combined = em._perform_extinction(flux_density=combined, angstroms=angstroms, av_host=av, rv_host=r_v,
                                      redshift=redshift, **kwargs)
    return combined
    