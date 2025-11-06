from redback.utils import citation_wrapper

@citation_wrapper('redback')
def predeceleration(time, aa, mm, t0, **kwargs):
    """
    Pre-deceleration power law model.

    Parameters
    ----------
    time : np.ndarray
        Time array in seconds.
    aa : float
        Amplitude term for power law.
    mm : float
        Deceleration power law gradient (typically 3 but depends on physics).
    t0 : float
        Time GRB went off.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    np.ndarray
        Deceleration power law (units are arbitrary and dependent on aa).
    """
    return aa * (time - t0)**mm

@citation_wrapper('redback')
def one_component_fireball_model(time, a_1, alpha_1, **kwargs):
    """
    One-component fireball model (simple power law).

    Parameters
    ----------
    time : np.ndarray
        Time array for power law.
    a_1 : float
        Power law decay amplitude.
    alpha_1 : float
        Power law decay exponent.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    np.ndarray
        Power law (units are arbitrary and dependent on a_1).
    """
    return a_1 * time ** alpha_1

