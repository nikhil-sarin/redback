from . import extinction_models

def t0_extinction_models(time, lognh, factor, **kwargs):
    """
    :param time: time in mjd or other unit. Note is not a reference time.
    :param lognh: hydrogen column density
    :param factor: prefactor for extinction
    :param kwargs: Must include t0 parameter which is in the same units as the data.
    :return: magnitude
    """
    time = (time - kwargs['t0']) * 86400
    magnitude = extinction_models.extinction_with_afterglow_base_model(time=time, lognh=lognh, factor=factor, **kwargs)
    return magnitude
