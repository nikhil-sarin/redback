import numpy as np


def gaussian_prompt(times, amplitude, t_0, sigma, **kwargs):
    dt = kwargs.get('dt', 1)
    return amplitude * np.exp(-(times - t_0) ** 2 / (2 * sigma ** 2)) * dt


def skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall, **kwargs):
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
    dt = kwargs.get('dt', 1)

    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_0) / tau_rise)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_0) / tau_fall)
    return envelope * dt


def fred(times, amplitude, psi, tau, delta, **kwargs):
    dt = kwargs.get('dt', 1)
    frac = (times - delta) / tau
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi * (frac + 1 / frac)) * np.exp(2 * psi) * dt


def fred_extended(times, amplitude, psi, tau, delta, gamma, nu, **kwargs):
    dt = kwargs.get('dt', 1)
    frac = (times - delta) / tau
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi ** gamma * frac ** gamma - psi ** nu / frac ** nu) * np.exp(2 * psi) * dt
