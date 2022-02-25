import numpy as np
from redback.transient_models.phenominological_models import exponential_powerlaw

def thermal_synchrotron():
    """
    From Margalit paper ...

    :return:
    """
    pass

def exponential_powerlaw_bolometric(time, lbol_0, alpha_1, alpha_2, tpeak, **kwargs):
    """
    :param time: rest frame time in seconds
    :param lbol_0: bolometric luminosity scale
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak: peak time in seconds
    :param kwargs:
    :return: bolometric_luminosity
    """
    lbol = exponential_powerlaw(time, a_1=lbol_0, alpha_1=alpha_1, alpha_2=alpha_2,
                                tpeak=tpeak, **kwargs)
    return lbol

def sne_phenominological(time, ):
    pass


