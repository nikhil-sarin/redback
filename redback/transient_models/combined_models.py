import redback.transient_models.extinction_models

def tophat_and_twolayerstratified(time, redshift, av, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, mej, vej_1, vej_2, kappa, beta, **kwargs):
    
    """
    function to combine the signals of a tophat afterglow and a two layer stratified kilonova with extinction
    
    takes all params required by each individual model
    params shared by models: time, redshift, av, relevant kwargs (do not pass to function twice)
    
    returns the combined signal as spcified by the 'output_format' kwarg
    """
    
    afterglow = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=redshift, av=av,
        base_model='tophat',  thv=thv, loge0=loge0 , thc= thc, logn0=logn0, p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0,
        **kwargs)
 
    kilonova = redback.transient_models.extinction_models.extinction_with_kilonova_base_model(time=time, redshift=redshift, av=av,
        base_model='two_layer_stratified_kilonova',  mej=mej, vej_1=vej_1, vej_2=vej_2, kappa=kappa, beta=beta, **kwargs)
    
    combined = afterglow+kilonova
    
    return combined


def tophat_and_twocomponent(time, redshift, av, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    """
    function to combine the signals of a tophat afterglow and a two component kilonova with extinction
    
    takes all params required by each individual model
    params shared by models: time, redshift, av, relevant kwargs (do not pass to function twice)
    
    returns the combined signal as spcified by the 'output_format' kwarg
    """
    
    afterglow = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=redshift, av=av,
        base_model='tophat',  thv=thv, loge0=loge0 , thc= thc, logn0=logn0, p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0,
        **kwargs)
    
    kilonova = redback.transient_models.extinction_models.extinction_with_kilonova_base_model(time=time, redshift=redshift, av=av,
        base_model='two_component_kilonova_model',  mej_1=mej_1, vej_1=vej_2, temperature_floor_1=temperature_floor_1, kappa_1=kappa_1,             mej_2=mej_2, vej_2=vej_2, temperature_floor_2=temperature_floor_2, kappa_2=kappa_2, **kwargs)
    
    combined = afterglow+kilonova
    
    return combined


def tophat_and_arnett(time, av, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, f_nickel, mej, **kwargs):
    
    """
    function to combine the signals of a tophat afterglow and an arnett supernova with extinction
    
    takes all params required by each individual model
    params shared by models: time, redshift, av, relevant kwargs (do not pass to function twice)
    
    returns the combined signal as spcified by the 'output_format' kwarg
    """
    
    afterglow = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=redshift, av=av,
        base_model='tophat',  thv=thv, loge0=loge0 , thc= thc, logn0=logn0, p=p, logepse=logepse, logepsb=logepsb, ksin=ksin, g0=g0,
        **kwargs)
    
    supernova = redback.transient_models.extinction_models.extinction_with_supernova_base_model(time=time, redshift=redshift, av=av,
        base_model='arnett',  f_nickel=f_nickel, mej=mej, **kwargs)
    
    combined = afterglow + supernova
    
    return combined