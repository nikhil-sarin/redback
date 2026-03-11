"""
Example custom models for Redback.

This module demonstrates how to create models that can be registered as Redback plugins.
"""

import numpy as np


def example_exponential_model(time, amplitude, decay_time, **kwargs):
    """
    Simple exponential decay model.

    This is a minimal example model to demonstrate the plugin system.

    :param time: Time array in days
    :param amplitude: Initial amplitude
    :param decay_time: Exponential decay timescale in days
    :param kwargs: Additional keyword arguments
    :return: Flux as a function of time
    """
    return amplitude * np.exp(-time / decay_time)


def example_gaussian_model(time, peak_time, peak_amplitude, width, **kwargs):
    """
    Gaussian profile model.

    Example model showing a Gaussian light curve.

    :param time: Time array in days
    :param peak_time: Time of peak in days
    :param peak_amplitude: Amplitude at peak
    :param width: Gaussian width (sigma) in days
    :param kwargs: Additional keyword arguments
    :return: Flux as a function of time
    """
    return peak_amplitude * np.exp(-0.5 * ((time - peak_time) / width) ** 2)


def example_combined_model(time, amp_exp, decay_time, amp_gauss, peak_time, width, **kwargs):
    """
    Combined exponential and Gaussian model.

    Demonstrates how to combine multiple components.

    :param time: Time array in days
    :param amp_exp: Exponential component amplitude
    :param decay_time: Exponential decay time
    :param amp_gauss: Gaussian component amplitude
    :param peak_time: Gaussian peak time
    :param width: Gaussian width
    :param kwargs: Additional keyword arguments
    :return: Combined flux as a function of time
    """
    exp_component = example_exponential_model(time, amp_exp, decay_time)
    gauss_component = example_gaussian_model(time, peak_time, amp_gauss, width)
    return exp_component + gauss_component
