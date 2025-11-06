import numpy
from astropy.cosmology import Planck18 as cosmo
import redback.constants as cc
from redback.utils import lambda_to_nu, fnu_to_flambda
import redback.sed as sed
import redback.transient_models.phenomenological_models as pm

def _get_blackbody_spectrum(angstrom, temperature, r_photosphere, distance):
    """
    Calculate blackbody spectrum.

    Parameters
    ----------
    angstrom : np.ndarray
        Wavelength array in angstroms.
    temperature : float
        Temperature in Kelvin.
    r_photosphere : float
        Photosphere radius in cm.
    distance : float
        Distance in cm.

    Returns
    -------
    np.ndarray
        Flux in ergs/s/cm^2/angstrom.
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
    Calculate power law spectrum.

    Parameters
    ----------
    angstrom : np.ndarray
        Wavelength array in angstroms.
    alpha : float
        Power law index.
    aa : float
        Normalization.

    Returns
    -------
    np.ndarray
        Flux in units set by normalization.
    """
    return aa*angstrom**alpha

def powerlaw_spectrum_with_absorption_and_emission_lines(angstroms, alpha, aa, lc1, ls1,
                                                         v1, lc2, ls2, v2, **kwargs):
    """
    Power law spectrum with absorption and emission lines.

    Parameters
    ----------
    angstroms : np.ndarray
        Wavelength array in angstroms.
    alpha : float
        Power law index.
    aa : float
        Normalization.
    lc1 : float
        Center of emission line.
    ls1 : float
        Strength of emission line.
    v1 : float
        Velocity of emission line.
    lc2 : float
        Center of absorption line.
    ls2 : float
        Strength of absorption line.
    v2 : float
        Velocity of absorption line.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    np.ndarray
        Flux in ergs/s/cm^2/angstrom.

    Notes
    -----
    One can add more lines if needed or set line strength to zero to remove a line.
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
    Blackbody spectrum with absorption and emission lines.

    Parameters
    ----------
    angstroms : np.ndarray
        Wavelength array in angstroms.
    redshift : float
        Redshift.
    rph : float
        Photosphere radius in cm.
    temp : float
        Photosphere temperature in Kelvin.
    lc1 : float
        Center of emission line.
    ls1 : float
        Strength of emission line.
    v1 : float
        Velocity of emission line.
    lc2 : float
        Center of absorption line.
    ls2 : float
        Strength of absorption line.
    v2 : float
        Velocity of absorption line.
    **kwargs : dict
        Additional keyword arguments:

        - cosmology : astropy.cosmology object, optional
            Cosmology to use (defaults to Planck18).

    Returns
    -------
    np.ndarray
        Flux in ergs/s/cm^2/angstrom.

    Notes
    -----
    One can add more lines if needed or set line strength to zero to remove a line.
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
    Blackbody spectrum at a given redshift with proper redshift corrections.

    Parameters
    ----------
    angstroms : np.ndarray
        Wavelength array in angstroms in observer frame.
    redshift : float
        Redshift.
    rph : float
        Photosphere radius in cm (rest frame).
    temp : float
        Photosphere temperature in Kelvin (rest frame).
    **kwargs : dict
        Additional keyword arguments:

        - cosmology : astropy.cosmology object, optional
            Cosmology to use (defaults to Planck18).

    Returns
    -------
    np.ndarray
        Flux in ergs/s/cm^2/angstrom in observer frame.
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
    Power law plus blackbody spectrum at redshift with time evolution.

    Parameters
    ----------
    angstroms : np.ndarray
        Wavelength array in angstroms in observer frame.
    redshift : float
        Source redshift.
    pl_amplitude : float
        Power law amplitude at reference wavelength at t=1 day (erg/s/cm^2/Angstrom).
    pl_slope : float
        Power law slope (F_lambda ∝ lambda^slope).
    pl_evolution_index : float
        Power law time evolution F_pl(t) ∝ t^(-pl_evolution_index).
    temperature_0 : float
        Initial blackbody temperature in Kelvin at t=1 day (rest frame).
    radius_0 : float
        Initial blackbody radius in cm at t=1 day (rest frame).
    temp_rise_index : float
        Temperature rise T(t) ∝ t^temp_rise_index for t < temp_peak_time.
    temp_decline_index : float
        Temperature decline T(t) ∝ t^(-temp_decline_index) for t > temp_peak_time.
    temp_peak_time : float
        Time in days when temperature peaks.
    radius_rise_index : float
        Radius rise R(t) ∝ t^radius_rise_index for t < radius_peak_time.
    radius_decline_index : float
        Radius decline R(t) ∝ t^(-radius_decline_index) for t > radius_peak_time.
    radius_peak_time : float
        Time in days when radius peaks.
    time : float
        Time in observer frame in days.
    **kwargs : dict
        Additional keyword arguments:

        - reference_wavelength : float, optional
            Wavelength for power law amplitude normalization in Angstroms (default 5000).
        - cosmology : astropy.cosmology object, optional
            Cosmology for luminosity distance calculation (defaults to Planck18).

    Returns
    -------
    np.ndarray
        Flux in ergs/s/cm^2/angstrom in observer frame.
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