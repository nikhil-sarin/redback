"""
Utility functions for transient models to reduce code duplication.

This module provides common functions used across multiple model files for:
- Parameter setup and defaults
- Output format handling
- Common calculations

These utilities help maintain consistency and reduce code duplication across the codebase.
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from collections import namedtuple
import astropy.units as uu

import redback.interaction_processes as ip
import redback.photosphere as photosphere
import redback.sed as sed
from redback.sed import flux_density_to_spectrum
from redback.utils import calc_kcorrected_properties, lambda_to_nu


def setup_optical_depth_defaults(kwargs):
    """
    Setup common default parameters for interaction process, photosphere, and SED.

    This reduces the repetition of these 3 lines that appear in ~100+ model functions:
        kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
        kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
        kwargs['sed'] = kwargs.get("sed", sed.Blackbody)

    :param kwargs: Dictionary of keyword arguments (modified in place)
    :return: None (kwargs is modified in place)
    """
    kwargs['interaction_process'] = kwargs.get('interaction_process', ip.Diffusion)
    kwargs['photosphere'] = kwargs.get('photosphere', photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get('sed', sed.Blackbody)


def get_cosmology_defaults(redshift, kwargs):
    """
    Extract cosmology from kwargs and compute luminosity distance.

    This pattern appears in ~100+ model functions.

    :param redshift: Source redshift
    :param kwargs: Dictionary containing optional 'cosmology' key
    :return: tuple of (cosmology object, luminosity distance in cgs)
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    return cosmology, dl


def setup_photosphere_sed_defaults(kwargs):
    """
    Setup default photosphere and SED classes.

    This pattern appears in every model function that handles output formats.

    :param kwargs: Dictionary of keyword arguments
    :return: tuple of (photosphere_class, sed_class)
    """
    photosphere_class = kwargs.get('photosphere', photosphere.TemperatureFloor)
    sed_class = kwargs.get('sed', sed.Blackbody)
    return photosphere_class, sed_class


def compute_photosphere_and_sed(time, lbol, frequency, photosphere_class, sed_class, dl, **kwargs):
    """
    Common pattern for computing photosphere properties and SED.

    This 3-line pattern appears in almost every model function.

    :param time: Time array
    :param lbol: Bolometric luminosity
    :param frequency: Frequency array
    :param photosphere_class: Photosphere class to use
    :param sed_class: SED class to use
    :param dl: Luminosity distance in cgs
    :param kwargs: Additional arguments passed to photosphere and SED
    :return: tuple of (photosphere object, SED object)
    """
    photo = photosphere_class(time=time, luminosity=lbol, **kwargs)
    sed_obj = sed_class(
        temperature=photo.photosphere_temperature,
        r_photosphere=photo.r_photosphere,
        frequency=frequency,
        luminosity_distance=dl)
    return photo, sed_obj
