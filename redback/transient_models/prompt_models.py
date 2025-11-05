import numpy as np


def gaussian_prompt(times, amplitude, t_0, sigma, **kwargs):
    """
    Gaussian prompt emission model.

    Parameters
    ----------
    times : array_like
        Time array
    amplitude : float
        Amplitude of the Gaussian
    t_0 : float
        Central time of the Gaussian peak
    sigma : float
        Width (standard deviation) of the Gaussian
    kwargs : dict, optional
        Additional keyword arguments
        
        dt : float, optional
            Time bin width (default: 1)

    Returns
    -------
    array_like
        Flux values at given times
    """
    dt = kwargs.get('dt', 1)
    return amplitude * np.exp(-(times - t_0) ** 2 / (2 * sigma ** 2)) * dt


def skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall, **kwargs):
    """
    Skewed Gaussian prompt emission model with different rise and fall widths.

    Parameters
    ----------
    times : array_like
        Time array
    amplitude : float
        Amplitude of the Gaussian
    t_0 : float
        Central time of the peak
    sigma_rise : float
        Width (standard deviation) of the rise (before t_0)
    sigma_fall : float
        Width (standard deviation) of the fall (after t_0)
    kwargs : dict, optional
        Additional keyword arguments
        
        dt : float, optional
            Time bin width (default: 1)

    Returns
    -------
    array_like
        Flux values at given times
    """
    dt = kwargs.get('dt', 1)
    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = gaussian_prompt(
        times=times[before_burst_indices], amplitude=amplitude, t_0=t_0, sigma=sigma_rise)
    envelope[after_burst_indices] = gaussian_prompt(
        times=times[after_burst_indices], amplitude=amplitude, t_0=t_0, sigma=sigma_fall)
    return envelope * dt


def skew_exponential(times, amplitude, t_0, tau_rise, tau_fall, **kwargs):
    """
    Skewed exponential prompt emission model with different rise and fall timescales.

    Parameters
    ----------
    times : array_like
        Time array
    amplitude : float
        Amplitude of the exponential
    t_0 : float
        Central time of the peak
    tau_rise : float
        Rise timescale (before t_0)
    tau_fall : float
        Fall timescale (after t_0)
    kwargs : dict, optional
        Additional keyword arguments
        
        dt : float, optional
            Time bin width (default: 1)

    Returns
    -------
    array_like
        Flux values at given times
    """
    dt = kwargs.get('dt', 1)

    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_0) / tau_rise)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_0) / tau_fall)
    return envelope * dt


def fred(times, amplitude, psi, tau, delta, **kwargs):
    """
    Fast Rise Exponential Decay (FRED) prompt emission model.

    Parameters
    ----------
    times : array_like
        Time array
    amplitude : float
        Amplitude of the FRED profile
    psi : float
        Shape parameter controlling the asymmetry
    tau : float
        Timescale parameter
    delta : float
        Time offset
    kwargs : dict, optional
        Additional keyword arguments
        
        dt : float, optional
            Time bin width (default: 1)

    Returns
    -------
    array_like
        Flux values at given times
    """
    dt = kwargs.get('dt', 1)
    frac = (times - delta) / tau
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi * (frac + 1 / frac)) * np.exp(2 * psi) * dt


def fred_extended(times, amplitude, psi, tau, delta, gamma, nu, **kwargs):
    """
    Extended Fast Rise Exponential Decay (FRED) prompt emission model with additional shape parameters.

    Parameters
    ----------
    times : array_like
        Time array
    amplitude : float
        Amplitude of the FRED profile
    psi : float
        Shape parameter controlling the asymmetry
    tau : float
        Timescale parameter
    delta : float
        Time offset
    gamma : float
        Exponent for the rise component
    nu : float
        Exponent for the decay component
    kwargs : dict, optional
        Additional keyword arguments
        
        dt : float, optional
            Time bin width (default: 1)

    Returns
    -------
    array_like
        Flux values at given times
    """
    dt = kwargs.get('dt', 1)
    frac = (times - delta) / tau
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi ** gamma * frac ** gamma - psi ** nu / frac ** nu) * np.exp(2 * psi) * dt
