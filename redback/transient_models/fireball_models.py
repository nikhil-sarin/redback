from redback.utils import citation_wrapper

@citation_wrapper('redback')
def predeceleration(time, aa, mm, t0, **kwargs):
    """
    :param time: time array in seconds
    :param aa: amplitude term for powerlaw
    :param mm: deceleration powerlaw gradient; typically 3 but depends on physics
    :param t0: time GRB went off.
    :param kwargs: None
    :return: deceleration powerlaw; units are arbitrary and dependent on a_1.
    """
    return aa * (time - t0)**mm

@citation_wrapper('redback')
def one_component_fireball_model(time, a_1, alpha_1, **kwargs):
    """
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param kwargs:
    :return: powerlaw; units are arbitrary and dependent on a_1.
    """
    return a_1 * time ** alpha_1

