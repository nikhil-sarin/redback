import redback.transient_models.extinction_models as em
import redback.transient_models as tm
from redback.utils import nu_to_lambda
from redback.utils import citation_wrapper

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract, https://ui.adsabs.harvard.edu/abs/2020ApJ...891..152H/abstract')
def tophat_and_twolayerstratified(time, redshift, av, thv, loge0, thc, logn0, p, logepse,
                                  logepsb, ksin, g0, mej, vej_1, vej_2, kappa, beta, **kwargs):
    """
    Combined tophat afterglow and two-layer stratified kilonova with extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in days in observer frame.
    redshift : float
        Source redshift.
    av : float
        V-band extinction from host galaxy in magnitudes.
    thv : float
        Viewing angle in radians.
    loge0 : float
        Log10 on-axis isotropic equivalent energy.
    thc : float
        Half width of jet core/jet opening angle in radians.
    logn0 : float
        Log10 number density of ISM in cm^-3.
    p : float
        Electron distribution power law index (must be > 2).
    logepse : float
        Log10 fraction of thermal energy in electrons.
    logepsb : float
        Log10 fraction of thermal energy in magnetic field.
    ksin : float
        Fraction of electrons that get accelerated.
    g0 : float
        Initial Lorentz factor.
    mej : float
        Ejecta mass in solar masses.
    vej_1 : float
        Velocity of inner shell in c.
    vej_2 : float
        Velocity of outer shell in c.
    kappa : float
        Constant gray opacity.
    beta : float
        Power law index of density profile.
    **kwargs : dict
        Additional keyword arguments:

        - r_v : float, optional
            Extinction parameter (default 3.1).
        - spread : bool, optional
            Whether jet can spread (default False).
        - latres : int, optional
            Latitudinal resolution for structured jets (default 2).
        - tres : int, optional
            Time resolution of shock evolution (default 100).
        - spectype : int, optional
            Whether to have inverse Compton: 0 = no, 1 = yes (default 0).
        - l0, ts, q : float, optional
            Energy injection parameters (default 0).
        - frequency : float or np.ndarray
            Frequency to calculate (must be same length as time array or a single number).

    Returns
    -------
    np.ndarray
        Flux density signal with extinction added.
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
    Combined tophat afterglow and two-component kilonova with extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in days in observer frame.
    redshift : float
        Source redshift.
    av : float
        V-band extinction from host galaxy in magnitudes.
    thv : float
        Viewing angle in radians.
    loge0 : float
        Log10 on-axis isotropic equivalent energy.
    thc : float
        Half width of jet core/jet opening angle in radians.
    logn0 : float
        Log10 number density of ISM in cm^-3.
    p : float
        Electron distribution power law index (must be > 2).
    logepse : float
        Log10 fraction of thermal energy in electrons.
    logepsb : float
        Log10 fraction of thermal energy in magnetic field.
    ksin : float
        Fraction of electrons that get accelerated.
    g0 : float
        Initial Lorentz factor.
    mej_1 : float
        Ejecta mass in solar masses of first component.
    vej_1 : float
        Minimum initial velocity of first component.
    temperature_floor_1 : float
        Floor temperature of first component.
    kappa_1 : float
        Gray opacity of first component.
    mej_2 : float
        Ejecta mass in solar masses of second component.
    vej_2 : float
        Minimum initial velocity of second component.
    temperature_floor_2 : float
        Floor temperature of second component.
    kappa_2 : float
        Gray opacity of second component.
    **kwargs : dict
        Additional keyword arguments:

        - r_v : float, optional
            Extinction parameter (default 3.1).
        - spread : bool, optional
            Whether jet can spread (default False).
        - latres : int, optional
            Latitudinal resolution for structured jets (default 2).
        - tres : int, optional
            Time resolution of shock evolution (default 100).
        - spectype : int, optional
            Whether to have inverse Compton: 0 = no, 1 = yes (default 0).
        - l0, ts, q : float, optional
            Energy injection parameters (default 0).
        - frequency : float or np.ndarray
            Frequency to calculate (must be same length as time array or a single number).

    Returns
    -------
    np.ndarray
        Flux density signal with extinction added.
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
    Combined tophat afterglow and Arnett supernova with extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in days in observer frame.
    av : float
        V-band extinction from host galaxy in magnitudes.
    redshift : float
        Source redshift.
    thv : float
        Viewing angle in radians.
    loge0 : float
        Log10 on-axis isotropic equivalent energy.
    thc : float
        Half width of jet core/jet opening angle in radians.
    logn0 : float
        Log10 number density of ISM in cm^-3.
    p : float
        Electron distribution power law index (must be > 2).
    logepse : float
        Log10 fraction of thermal energy in electrons.
    logepsb : float
        Log10 fraction of thermal energy in magnetic field.
    ksin : float
        Fraction of electrons that get accelerated.
    g0 : float
        Initial Lorentz factor.
    f_nickel : float
        Fraction of nickel mass.
    mej : float
        Total ejecta mass in solar masses.
    **kwargs : dict
        Additional keyword arguments:

        - r_v : float, optional
            Extinction parameter (default 3.1).
        - spread : bool, optional
            Whether jet can spread (default False).
        - latres : int, optional
            Latitudinal resolution for structured jets (default 2).
        - tres : int, optional
            Time resolution of shock evolution (default 100).
        - spectype : int, optional
            Whether to have inverse Compton: 0 = no, 1 = yes (default 0).
        - l0, ts, q : float, optional
            Energy injection parameters (default 0).
        - interaction_process : class, optional
            Default is Diffusion. Can also be None or another interaction process.
        - photosphere : class, optional
            Default is TemperatureFloor.
        - sed : class, optional
            Default is Blackbody.
        - kappa : float
            Opacity.
        - kappa_gamma : float
            Gamma-ray opacity.
        - vej : float
            Ejecta velocity in km/s.
        - temperature_floor : float
            Floor temperature.
        - frequency : float or np.ndarray
            Frequency to calculate (must be same length as time array or a single number).

    Returns
    -------
    np.ndarray
        Flux density with extinction added.
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
    Combined generic afterglow and optical transient with extinction.

    Parameters
    ----------
    time : np.ndarray
        Time in days in observer frame.
    redshift : float
        Source redshift.
    av : float
        V-band extinction from host galaxy in magnitudes.
    **model_kwargs : dict
        Additional keyword arguments:

        - afterglow_kwargs : dict
            Dictionary of parameters required by the afterglow transient model specified by 'base_model'
            and any additional keyword arguments.
        - optical_kwargs : dict
            Dictionary of parameters required by the optical transient model specified by 'base_model'
            and any additional keyword arguments.
        - frequency : float or np.ndarray
            Frequency to calculate.
        - r_v : float, optional
            Extinction parameter (default 3.1).
        - output_format : str, optional
            Output format (default 'flux_density').

    Returns
    -------
    np.ndarray
        Flux density signal with extinction added.
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
    