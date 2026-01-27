import numpy
from astropy.cosmology import Planck18 as cosmo
import redback.constants as cc
from redback.utils import lambda_to_nu, fnu_to_flambda, citation_wrapper
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


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1993ApJ...413..281B/abstract')
def band_function_high_energy(energies_keV, redshift, log10_norm, alpha, beta, e_peak, **kwargs):
    """
    Band function (Band et al. 1993) spectrum.

    :param energies_keV: energy array in keV (observer frame)
    :param redshift: redshift
    :param log10_norm: log10 photon flux normalization at 100 keV (photons/cm^2/s/keV)
    :param alpha: low-energy photon index
    :param beta: high-energy photon index
    :param e_peak: peak energy in keV
    :return: flux density in mJy
    """
    energies_rest = numpy.asarray(energies_keV) * (1 + redshift)

    norm = 10 ** log10_norm
    e_break = (alpha - beta) * e_peak / (2 + alpha)

    photon_flux = numpy.zeros_like(energies_rest)
    low_e = energies_rest < e_break
    if numpy.any(low_e):
        photon_flux[low_e] = norm * (energies_rest[low_e] / 100.0) ** alpha * numpy.exp(-energies_rest[low_e] / e_peak)

    high_e = energies_rest >= e_break
    if numpy.any(high_e):
        photon_flux[high_e] = norm * ((alpha - beta) * e_peak / (100.0 * (2 + alpha))) ** (alpha - beta) * \
                              numpy.exp(beta - alpha) * (energies_rest[high_e] / 100.0) ** beta

    keV_to_Hz = 2.417989e17
    keV_to_erg = 1.60218e-9
    energy_flux = photon_flux * energies_rest * keV_to_erg
    flux_density_erg = energy_flux * keV_to_Hz
    flux_density_mjy = flux_density_erg * 1e26 / (1 + redshift)

    return flux_density_mjy


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1978ppap.book.....R/abstract')
def blackbody_high_energy(energies_keV, redshift, r_photosphere_rs, kT, **kwargs):
    """
    Blackbody spectrum for high-energy transients.

    :param energies_keV: energy array in keV (observer frame)
    :param redshift: redshift
    :param r_photosphere_rs: photosphere radius in solar radii (rest frame)
    :param kT: temperature in keV (rest frame)
    :return: flux density in mJy
    """
    energies_rest = numpy.asarray(energies_keV) * (1 + redshift)

    solar_radius_cm = 6.957e10
    radius_cm = r_photosphere_rs * solar_radius_cm
    keV_to_erg = 1.60218e-9
    cosmology = kwargs.get('cosmology', cosmo)
    dl_cm = cosmology.luminosity_distance(redshift).cgs.value
    energy_erg = energies_rest * keV_to_erg

    temperature_k = kT * 1.16045e7
    frequency_rest = energy_erg / cc.h.cgs.value

    flux_density = sed.blackbody_to_flux_density(temperature=temperature_k,
                                                 r_photosphere=radius_cm,
                                                 dl=dl_cm,
                                                 frequency=frequency_rest)

    if hasattr(flux_density, 'value'):
        flux_density = flux_density.value

    return flux_density * 1e26 / (1 + redshift)
