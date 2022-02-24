import numpy as np


def predeceleration(time, aa, mm, t0, **kwargs):
    """
    :param time:
    :param aa:
    :param mm: deceleration powerlaw gradient; typically 3 but depends on physics
    :param t0: time GRB went off.
    :param kwargs:
    :return: deceleration powerlaw; units are arbitrary and dependent on a_1.
    """
    return aa * (time - t0)**mm


def one_component_fireball_model(time, a_1, alpha_1, **kwargs):
    """
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param kwargs:
    :return:
    """
    return a_1 * time ** alpha_1

