from redback.utils import citation_wrapper

@citation_wrapper('redback')
def predeceleration(time, aa, mm, t0, **kwargs):
    """
    Predeceleration powerlaw model.

    Parameters
    ----------
    time : array_like
        Time array in seconds
    aa : float
        Amplitude term for powerlaw
    mm : float
        Deceleration powerlaw gradient; typically 3 but depends on physics
    t0 : float
        Time GRB went off
    kwargs : dict, optional
        Additional keyword arguments

    Returns
    -------
    array_like
        Deceleration powerlaw; units are arbitrary and dependent on aa
    """
    return aa * (time - t0)**mm

@citation_wrapper('redback')
def one_component_fireball_model(time, a_1, alpha_1, **kwargs):
    """
    One component fireball powerlaw model.

    Parameters
    ----------
    time : array_like
        Time array for power law
    a_1 : float
        Power law decay amplitude
    alpha_1 : float
        Power law decay exponent
    kwargs : dict, optional
        Additional keyword arguments

    Returns
    -------
    array_like
        Powerlaw; units are arbitrary and dependent on a_1
    """
    return a_1 * time ** alpha_1

