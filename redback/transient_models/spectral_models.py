import numpy
from astropy.cosmology import Planck18 as cosmo
import redback.constants as cc
from redback.utils import lambda_to_nu, fnu_to_flambda
import redback.sed as sed
import redback.transient_models.phenomenological_models as pm

def _get_blackbody_spectrum(angstrom, temperature, r_photosphere, distance):
    """
    :param angstrom: wavelength array in angstroms
    :param temperature: temperature in Kelvin
    :param r_photosphere: photosphere radius in cm
    :param distance: distance in cm
    :return: flux in ergs/s/cm^2/angstrom
    """
    frequency = lambda_to_nu(angstrom)
    flux_density = sed.blackbody_to_flux_density(frequency=frequency,
                                                  temperature=temperature,
                                                  r_photosphere=r_photosphere,
                                                  dl=distance)
    flux_density = fnu_to_flambda(f_nu=flux_density, wavelength_A=angstrom)
    return flux_density.value

def _get_powerlaw_spectrum(angstrom, alpha, aa):
    """
    :param angstrom: wavelength array in angstroms
    :param alpha: power law index
    :param aa: normalization
    :return: flux in ergs/s/cm^2/angstrom
    """
    return aa*angstrom**alpha

def powerlaw_spectrum_with_absorption_and_emission_lines(angstroms, alpha, aa, lc1, ls1,
                                                         v1, lc2, ls2, v2, **kwargs):
    """
    A power law spectrum with one absorption line and one emission line.
    One can add more lines if needed. Or turn the line strength to zero to remove the line.

    :param angstroms: wavelength array in angstroms
    :param alpha: power law index
    :param aa: normalization
    :param lc1: center of emission line
    :param ls1: strength of emission line
    :param v1: velocity of emission line
    :param lc2: center of absorption line
    :param ls2: strength of absorption line
    :param v2: velocity of absorption line
    :return: flux in ergs/s/cm^2/angstrom
    """
    flux = _get_powerlaw_spectrum(angstrom=angstroms, alpha=alpha, aa=aa)
    fp1 = pm.line_spectrum_with_velocity_dispersion(angstroms, lc1, ls1, v1)
    fp2 = pm.line_spectrum_with_velocity_dispersion(angstroms, lc2, ls2, v2)
    return flux + fp1 - fp2


def blackbody_spectrum_with_absorption_and_emission_lines(angstroms, redshift,
                                                          rph, temp,
                                                          lc1, ls1, v1,
                                                          lc2, ls2, v2, **kwargs):
    """
    A blackbody spectrum with one absorption line and one emission line.
    One can add more lines if needed. Or turn the line strength to zero to remove the line.

    :param angstroms: wavelength array in angstroms
    :param redshift: redshift
    :param rph: photosphere radius in cm
    :param temp: photosphere temperature in Kelvin
    :param lc1: center of emission line
    :param ls1: strength of emission line
    :param v1: velocity of emission line
    :param lc2: center of absorption line
    :param ls2: strength of absorption line
    :param v2: velocity of absorption line
    :return: flux in ergs/s/cm^2/angstrom
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    flux = _get_blackbody_spectrum(angstrom=angstroms, distance=dl,
                                  r_photosphere=rph, temperature=temp)
    fp1 = pm.line_spectrum_with_velocity_dispersion(angstroms, lc1, ls1, v1)
    fp2 = pm.line_spectrum_with_velocity_dispersion(angstroms, lc2, ls2, v2)
    return flux + fp1 - fp2
