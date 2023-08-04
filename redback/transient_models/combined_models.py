import redback.transient_models.extinction_models
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
def tophat_and_twolayerstratified(time, redshift, av, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, mej, vej_1, vej_2, kappa, beta, **kwargs):
    
    """
    function to combine the flux density signals of a tophat afterglow and a two layer stratified kilonova with extinction
    
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param av: absolute mag extinction
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param beta: index for power-law structure, theta^-b
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param mej: ejecta mass in solar masses
    :param vej_1: velocity of inner shell in c
    :param vej_2: velocity of outer shell in c
    :param kappa: constant gray opacity
    :param beta: power law index of density profile
    :param kwargs: Additional keyword arguments
    :param r_v: extinction parameter, defaults to 3.1
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param frequency: frequency to calculate - Must be same length as time array or a single number
    :return: flux density signal with extinction added
    
    """
    kwargs['output_format']='flux_density'
    kwargs['base_model']='tophat'
    afterglow = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=redshift, av=av,
         thv=thv, loge0=loge0 , thc= thc, logn0=logn0, p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0,
        **kwargs)
    kwargs['base_model']='two_layer_stratified_kilonova'
    kilonova = redback.transient_models.extinction_models.extinction_with_kilonova_base_model(time=time, redshift=redshift, av=av,
          mej=mej, vej_1=vej_1, vej_2=vej_2, kappa=kappa, beta=beta, **kwargs)
    
    combined = afterglow+kilonova
    return combined


def tophat_and_twocomponent(time, redshift, av, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    """
    function to combine the flux density signals of a tophat afterglow and a two component kilonova with extinction added
    
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param av: absolute mag extinction
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param beta: index for power-law structure, theta^-b
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param mej_1: ejecta mass in solar masses of first component
    :param vej_1: minimum initial velocity of first component
    :param kappa_1: gray opacity of first component
    :param temperature_floor_1: floor temperature of first component
    :param mej_2: ejecta mass in solar masses of second component
    :param vej_2: minimum initial velocity of second component
    :param temperature_floor_2: floor temperature of second component
    :param kappa_2: gray opacity of second component
    :param kwargs: Additional keyword arguments
    :param r_v: extinction parameter, defaults to 3.1
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param frequency: frequency to calculate - Must be same length as time array or a single number
    :return: flux density signal with extinction added
    """
    
    kwargs['output_format']='flux_density'
    kwargs['base_model']='tophat'
    afterglow = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=redshift, av=av,
          thv=thv, loge0=loge0 , thc= thc, logn0=logn0, p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0,
        **kwargs)
    kwargs['base_model']='two_component_kilonova_model'
    kilonova = redback.transient_models.extinction_models.extinction_with_kilonova_base_model(time=time, redshift=redshift, av=av,
          mej_1=mej_1, vej_1=vej_2, temperature_floor_1=temperature_floor_1, kappa_1=kappa_1,             mej_2=mej_2, vej_2=vej_2, temperature_floor_2=temperature_floor_2, kappa_2=kappa_2, **kwargs)
    
    combined = afterglow + kilonova
    return combined


def tophat_and_arnett(time, av, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, f_nickel, mej, **kwargs):
    
    """
    function to combine the flux density signals of a tophat afterglow and an arnett supernova with extinction added
    
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param av: absolute mag extinction
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param beta: index for power-law structure, theta^-b
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: Additional keyword arguments
        Must include all the kwargs required by the specific interaction_process, photosphere, sed methods used
        e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :param r_v: extinction parameter, defaults to 3.1
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity, or another interaction process.
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param frequency: frequency to calculate - Must be same length as time array or a single number
    :return: flux density with extinction added
    """
    
    kwargs['output_format']='flux_density'
    kwargs['base_model']='tophat'
    afterglow = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=redshift, av=av,
         thv=thv, loge0=loge0 , thc= thc, logn0=logn0, p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0,
        **kwargs)
    kwargs['base_model']='arnett'
    supernova = redback.transient_models.extinction_models.extinction_with_supernova_base_model(time=time, redshift=redshift, av=av,
         f_nickel=f_nickel, mej=mej, **kwargs)
    
    combined = afterglow + supernova
    return combined

def afterglow_and_optical(time, redshift, av, model_type, afterglow_kwargs, optical_kwargs, **shared_kwargs):
    
    """
    function to combine the signals of any afterglow and any other optical transient with extinction added
    
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param av: absolute mag extinction
    :param model_type: specify type of optical transient model- 'supernova', 'tde', 'kilonova', 'magnetar_driven', 'shock_powered'
    :param afterglow_kwargs: dictionary of  parameters required by the afterglow transient model specified by 'base_model'
        and any additional keyword arguments. Refer to model documentation for details.
    :param optical_kwargs: dictionary of parameters required by the optical transient model specifed by 'base_model'
        and any additional keyword arguments. Note the base model must correspond to the given model type. Refer to model documentation
        for details.
    :param shared_kwargs: kwargs shared by models e.g. output_format, frequency, bands, r_v (extinction paramater defaults to 3.1)
    :return: set by shared_kwargs output format - 'flux_density' or 'magnitude' with extinction added
        note that only afterglow_models_sed allow for magnitude outputs
    """
    
    afterglow_kwargs.update(shared_kwargs)
    optical_kwargs.update(shared_kwargs)
   
    afterglow = redback.transient_models.extinction_models._evaluate_extinction_model(time=time, redshift=redshift, av=av,
                                                                                   model_type='afterglow', **afterglow_kwargs)
    
    optical= redback.transient_models.extinction_models._evaluate_extinction_model(time=time, redshift=redshift, av=av,
                                                                   model_type=model_type, **optical_kwargs)
  
    combined= afterglow + optical
    return combined
    