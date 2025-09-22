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
    :return: flux in units set by normalisation
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

def blackbody_spectrum_at_z(angstroms, redshift, rph, temp, **kwargs):
    """
    A blackbody spectrum at a given redshift, properly accounting for redshift effects.

    :param angstroms: wavelength array in angstroms in obs frame
    :param redshift: redshift
    :param rph: photosphere radius in cm (rest frame)
    :param temp: photosphere temperature in Kelvin (rest frame)
    :return: flux in ergs/s/cm^2/angstrom in obs frame
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    # Convert observed wavelengths to rest frame
    angstroms_rest = angstroms / (1 + redshift)

    # Calculate blackbody in rest frame
    flux_rest = _get_blackbody_spectrum(angstrom=angstroms_rest, distance=dl,
                                        r_photosphere=rph, temperature=temp)

    # Apply redshift corrections:
    # - Surface brightness dimming: factor of (1+z)
    # - Wavelength interval stretching: dλ_obs = dλ_rest × (1+z)
    # Combined effect: divide by (1+z)
    flux_obs = flux_rest / (1 + redshift)

    return flux_obs

def powerlaw_plus_blackbody_spectrum_at_z(angstroms, redshift, pl_amplitude, pl_slope, pl_evolution_index,
                                          temperature_0, radius_0, temp_rise_index, temp_decline_index,
                                          temp_peak_time, radius_rise_index, radius_decline_index,
                                          radius_peak_time, time, **kwargs):
    """
    A powerlaw + blackbody spectrum at a given redshift and time, properly accounting for redshift effects.

    :param angstroms: wavelength array in angstroms in obs frame
    :param redshift: source redshift
    :param pl_amplitude: power law amplitude at reference wavelength at t=1 day (erg/s/cm^2/Angstrom)
    :param pl_slope: power law slope (F_lambda ∝ lambda^slope)
    :param pl_evolution_index: power law time evolution F_pl(t) ∝ t^(-pl_evolution_index)
    :param temperature_0: initial blackbody temperature in Kelvin at t=1 day (rest frame)
    :param radius_0: initial blackbody radius in cm at t=1 day (rest frame)
    :param temp_rise_index: temperature rise T(t) ∝ t^temp_rise_index for t < temp_peak_time
    :param temp_decline_index: temperature decline T(t) ∝ t^(-temp_decline_index) for t > temp_peak_time
    :param temp_peak_time: time in days when temperature peaks
    :param radius_rise_index: radius rise R(t) ∝ t^radius_rise_index for t < radius_peak_time
    :param radius_decline_index: radius decline R(t) ∝ t^(-radius_decline_index) for t > radius_peak_time
    :param radius_peak_time: time in days when radius peaks
    :param time: time in observer frame in days
    :param kwargs: Additional parameters
    :param reference_wavelength: wavelength for power law amplitude normalization in Angstroms (default 5000)
    :param cosmology: Cosmology object for luminosity distance calculation
    :return: flux in ergs/s/cm^2/angstrom in obs frame
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    reference_wavelength = kwargs.get('reference_wavelength', 5000.0)  # Angstroms

    # Convert observed wavelengths to rest frame
    angstroms_rest = angstroms / (1 + redshift)

    # Convert observed time to rest frame
    time_rest = time / (1 + redshift)

    # Convert wavelengths to frequencies for the SED calculation
    frequency_rest = lambda_to_nu(wavelength=angstroms_rest)

    # Calculate evolving temperature and radius at this time
    temperature, radius = pm._powerlaw_blackbody_evolution(time=time_rest, temperature_0=temperature_0, radius_0=radius_0,
                                                           temp_rise_index=temp_rise_index,
                                                           temp_decline_index=temp_decline_index,
                                                           temp_peak_time=temp_peak_time,
                                                           radius_rise_index=radius_rise_index,
                                                           radius_decline_index=radius_decline_index,
                                                           radius_peak_time=radius_peak_time)

    # Create combined SED in rest frame
    sed_combined = sed.PowerlawPlusBlackbody(temperature=temperature, r_photosphere=radius,
                                         pl_amplitude=pl_amplitude, pl_slope=pl_slope,
                                         pl_evolution_index=pl_evolution_index, time=time_rest,
                                         reference_wavelength=reference_wavelength,
                                         frequency=frequency_rest, luminosity_distance=dl)

    # Get flux density in rest frame (F_nu)
    flux_density_rest = sed_combined.flux_density  # erg/s/cm^2/Hz

    # Convert from F_nu to F_lambda in rest frame
    flux_lambda_rest = fnu_to_flambda(f_nu=flux_density_rest, wavelength_A=angstroms_rest)

    # Apply redshift corrections to get observed frame flux:
    # - Surface brightness dimming: factor of (1+z)
    # - Wavelength interval stretching: dλ_obs = dλ_rest × (1+z)
    # Combined effect: divide by (1+z)
    flux_lambda_obs = flux_lambda_rest / (1 + redshift)

    # Convert to plain values if needed
    if hasattr(flux_lambda_obs, 'value'):
        return flux_lambda_obs.value
    else:
        return flux_lambda_obs