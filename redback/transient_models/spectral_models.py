import numpy
import numpy as np
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


def voigt_profile(wavelength, lambda_center, amplitude, sigma_gaussian, gamma_lorentz, continuum=0.0):
    """
    Voigt profile: convolution of Gaussian and Lorentzian line profiles

    Useful for modeling spectral lines where both thermal broadening (Gaussian) and
    natural/pressure broadening (Lorentzian) are important.

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    lambda_center : float
        Central wavelength of the line in Angstroms
    amplitude : float
        Peak amplitude of the profile (can be negative for absorption)
    sigma_gaussian : float
        Gaussian width parameter in Angstroms (thermal broadening)
    gamma_lorentz : float
        Lorentzian HWHM parameter in Angstroms (natural/pressure broadening)
    continuum : float
        Continuum level (default 0.0)

    Returns
    -------
    flux : array
        Voigt profile at each wavelength

    References
    ----------
    - Schreier 2018 (JQSRT, 213, 13) - Voigt function review
    - Approximation based on Faddeeva function

    Notes
    -----
    The Voigt profile is defined as the real part of the Faddeeva function:
    V(x, y) = Re[w(z)] where z = (x + iy)/sqrt(2)
    with x = (wavelength - lambda_center)/(sigma_gaussian * sqrt(2))
    and y = gamma_lorentz / (sigma_gaussian * sqrt(2))

    Examples
    --------
    >>> # H-alpha line with thermal and pressure broadening
    >>> wave = np.linspace(6560, 6570, 1000)
    >>> flux = voigt_profile(wave, lambda_center=6563.0, amplitude=1.0,
    ...                      sigma_gaussian=0.5, gamma_lorentz=0.2)
    """
    from scipy.special import wofz

    # Convert to dimensionless variables
    x = (wavelength - lambda_center) / (sigma_gaussian * np.sqrt(2))
    y = gamma_lorentz / (sigma_gaussian * np.sqrt(2))

    # Faddeeva function (scaled complex error function)
    z = x + 1j * y
    w = wofz(z)

    # Voigt profile is the real part, normalized
    voigt = np.real(w) / (sigma_gaussian * np.sqrt(2 * np.pi))

    # Scale by amplitude and add continuum
    flux = continuum + amplitude * voigt / np.max(voigt)

    return flux


def gaussian_line_profile(wavelength, lambda_center, amplitude, sigma, continuum=0.0):
    """
    Pure Gaussian line profile

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    lambda_center : float
        Central wavelength in Angstroms
    amplitude : float
        Peak amplitude (negative for absorption)
    sigma : float
        Standard deviation in Angstroms
    continuum : float
        Continuum level

    Returns
    -------
    flux : array
        Gaussian profile

    Examples
    --------
    >>> wave = np.linspace(6550, 6575, 500)
    >>> flux = gaussian_line_profile(wave, 6563, -0.5, 2.0, continuum=1.0)
    """
    profile = np.exp(-0.5 * ((wavelength - lambda_center) / sigma)**2)
    flux = continuum + amplitude * profile
    return flux


def lorentzian_line_profile(wavelength, lambda_center, amplitude, gamma, continuum=0.0):
    """
    Pure Lorentzian (Cauchy) line profile

    Natural line shape for radiative decay processes.

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    lambda_center : float
        Central wavelength in Angstroms
    amplitude : float
        Peak amplitude (negative for absorption)
    gamma : float
        Half-width at half-maximum (HWHM) in Angstroms
    continuum : float
        Continuum level

    Returns
    -------
    flux : array
        Lorentzian profile

    Examples
    --------
    >>> wave = np.linspace(6550, 6575, 500)
    >>> flux = lorentzian_line_profile(wave, 6563, -0.3, 1.5, continuum=1.0)
    """
    profile = gamma**2 / ((wavelength - lambda_center)**2 + gamma**2)
    flux = continuum + amplitude * profile
    return flux


def p_cygni_profile(wavelength, lambda_rest, tau_sobolev, v_phot,
                    continuum_flux, source_function='thermal', v_max=None, **kwargs):
    """
    P-Cygni line profile using Sobolev approximation

    Creates characteristic emission + blueshifted absorption from expanding envelope

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms (observer frame)
    lambda_rest : float
        Rest wavelength of transition in Angstroms (e.g., 6355 for Si II)
    tau_sobolev : float
        Sobolev optical depth (0.1-100)
    v_phot : float
        Photospheric velocity in km/s (5000-20000 for SNe)
    continuum_flux : array or float
        Continuum flux level
    source_function : str or float
        'thermal' (Planckian), 'scattering' (electron scat), or float value
    v_max : float, optional
        Maximum velocity of envelope in km/s (default: 1.5 * v_phot)

    Returns
    -------
    flux : array
        Spectrum with P-Cygni profile

    References
    ----------
    - Kasen & Branch 2001 (ApJ, 560, 439) - Analytic inversion
    - Jeffery & Branch 1990 - P-Cygni atlas
    - Branch et al. 2002 (ApJ, 566, 1005) - Direct analysis of SN spectra

    Notes
    -----
    Sobolev approximation valid when:
    v_thermal << v_phot  (typically 10 km/s << 10,000 km/s)

    The profile consists of:
    - Blueshifted absorption trough (v < 0, |v| < v_max)
    - Emission peak (v > 0, near rest wavelength)

    Examples
    --------
    >>> # Si II 6355 line at 10,000 km/s
    >>> wave = np.linspace(5000, 7000, 1000)
    >>> flux = p_cygni_profile(wave, lambda_rest=6355, tau_sobolev=3.0,
    ...                        v_phot=10000, continuum_flux=1.0)
    """
    c_kms = 299792.458  # speed of light in km/s

    if v_max is None:
        v_max = 1.5 * v_phot

    # Velocity shift from line center (negative = blueshift)
    v = c_kms * (wavelength - lambda_rest) / lambda_rest

    # Source function ratio
    if source_function == 'thermal':
        S = 0.5  # Typical thermal source function ratio
    elif source_function == 'scattering':
        S = 0.3  # Pure scattering source function
    else:
        S = float(source_function)

    # Initialize flux array
    flux = np.ones_like(wavelength, dtype=float) * continuum_flux

    # P-Cygni profile calculation
    # Absorption component (blueshifted side, -v_max < v < 0)
    # Absorption occurs for material moving toward us
    absorption_region = (v < 0) & (v > -v_max)

    if np.any(absorption_region):
        # Optical depth profile: peaks at v ~ v_phot
        # Use a profile that has maximum absorption at photospheric velocity
        v_abs = np.abs(v[absorption_region])

        # Gaussian profile centered at v_phot
        tau_profile = tau_sobolev * np.exp(-((v_abs - v_phot) / (0.3 * v_phot))**2)

        # Pure absorption: I = I_continuum * exp(-tau)
        # With partial source function filling: I = I_cont * exp(-tau) + S * B * (1 - exp(-tau))
        flux[absorption_region] = continuum_flux * (
            np.exp(-tau_profile) + S * (1 - np.exp(-tau_profile))
        )

    # Emission component (redshifted side, 0 < v < v_phot)
    emission_region = (v > 0) & (v < 0.5 * v_phot)

    if np.any(emission_region):
        # Emission from resonance scattering in receding material
        v_em = v[emission_region]

        # Emission strength decreases away from rest wavelength
        emission_factor = np.exp(-(v_em / (0.2 * v_phot))**2)

        # Add emission above continuum
        flux[emission_region] = continuum_flux * (1 + 0.3 * S * tau_sobolev * emission_factor)

    return flux


def elementary_p_cygni_profile(wavelength, lambda_rest, v_absorption,
                                absorption_depth, emission_strength, v_width,
                                continuum_flux=1.0):
    """
    Elementary P-Cygni profile with simple parameterization

    Simpler model for quick fits, using Gaussian absorption and emission components.

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    lambda_rest : float
        Rest wavelength of transition in Angstroms
    v_absorption : float
        Velocity of absorption minimum in km/s (positive value)
    absorption_depth : float
        Depth of absorption trough (0 to 1, fraction of continuum)
    emission_strength : float
        Strength of emission peak (relative to continuum)
    v_width : float
        Velocity width of features in km/s
    continuum_flux : float
        Continuum flux level

    Returns
    -------
    flux : array
        P-Cygni profile

    Examples
    --------
    >>> wave = np.linspace(6000, 6700, 1000)
    >>> flux = elementary_p_cygni_profile(wave, lambda_rest=6355,
    ...                                    v_absorption=11000,
    ...                                    absorption_depth=0.4,
    ...                                    emission_strength=0.2,
    ...                                    v_width=1500)
    """
    c_kms = 299792.458

    # Convert velocity to wavelength shift
    lambda_absorption = lambda_rest * (1 - v_absorption / c_kms)
    sigma_lambda = lambda_rest * v_width / c_kms

    # Gaussian absorption component
    absorption = absorption_depth * np.exp(-0.5 * ((wavelength - lambda_absorption) / sigma_lambda)**2)

    # Gaussian emission component (centered near rest wavelength, slightly redshifted)
    lambda_emission = lambda_rest * (1 + 0.5 * v_width / c_kms)
    emission = emission_strength * np.exp(-0.5 * ((wavelength - lambda_emission) / (1.5 * sigma_lambda))**2)

    # Combine: continuum - absorption + emission
    flux = continuum_flux * (1 - absorption + emission)

    return flux


def multi_line_p_cygni_spectrum(wavelength, redshift, continuum_model,
                                line_list, v_phot, **kwargs):
    """
    Full spectrum with multiple P-Cygni profiles

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    redshift : float
        Source redshift
    continuum_model : str or callable
        'blackbody', 'powerlaw', or custom function
    line_list : list of dict
        Each dict has: {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}
    v_phot : float
        Photospheric velocity in km/s

    Returns
    -------
    spectrum : array
        Full spectrum with P-Cygni profiles for all lines

    Examples
    --------
    >>> # Type Ia SN spectrum with Si II, Ca II, Fe II
    >>> lines = [
    ...     {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0},
    ...     {'ion': 'Ca II H&K', 'lambda': 3934, 'tau': 5.0},
    ...     {'ion': 'Fe II', 'lambda': 5169, 'tau': 2.0}
    ... ]
    >>> spectrum = multi_line_p_cygni_spectrum(
    ...     wavelength=wave, redshift=0.01,
    ...     continuum_model='blackbody',
    ...     line_list=lines, v_phot=12000,
    ...     temperature=10000, r_phot=1e15
    ... )
    """
    cosmology = kwargs.get('cosmology', cosmo)

    # Get continuum
    if continuum_model == 'blackbody':
        continuum = blackbody_spectrum_at_z(
            wavelength, redshift,
            kwargs['r_phot'], kwargs['temperature']
        )
    elif continuum_model == 'powerlaw':
        continuum = _get_powerlaw_spectrum(
            wavelength, kwargs['alpha'], kwargs['aa']
        )
    elif callable(continuum_model):
        continuum = continuum_model(wavelength, **kwargs)
    else:
        raise ValueError(f"Unknown continuum model: {continuum_model}")

    # Start with continuum
    spectrum = continuum.copy()

    # Add each P-Cygni line
    for line in line_list:
        # Redshift correction - lines are formed in source rest frame
        lambda_rest = line['lambda']
        lambda_obs = lambda_rest * (1 + redshift)

        # Get P-Cygni profile (ratio to continuum)
        line_profile = p_cygni_profile(
            wavelength, lambda_obs,
            tau_sobolev=line['tau'],
            v_phot=v_phot,
            continuum_flux=1.0,  # normalized
            **kwargs
        )

        # Multiply continuum by line transmission
        spectrum *= line_profile

    return spectrum


def synow_line_model(wavelength, lambda_rest, tau_ref, v_phot, v_max,
                     aux_depth=0.0, temp_exc=10000.0, **kwargs):
    """
    SYNOW-style parameterized line profile

    Based on the SYNOW code for direct analysis of SN spectra.

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    lambda_rest : float
        Rest wavelength of transition in Angstroms
    tau_ref : float
        Reference optical depth at photosphere
    v_phot : float
        Photospheric velocity in km/s
    v_max : float
        Maximum velocity of line-forming region in km/s
    aux_depth : float
        Auxiliary parameter for detached shells (default 0)
    temp_exc : float
        Excitation temperature in K (default 10000)

    Returns
    -------
    flux : array
        SYNOW-style line profile (normalized to continuum=1)

    References
    ----------
    - Fisher et al. 1997 (ApJ, 481, L89) - SYNOW introduction
    - Branch et al. 2002 (ApJ, 566, 1005) - SYNOW applications

    Examples
    --------
    >>> wave = np.linspace(5800, 6800, 1000)
    >>> flux = synow_line_model(wave, lambda_rest=6355, tau_ref=5.0,
    ...                          v_phot=10000, v_max=25000)
    """
    c_kms = 299792.458

    # Velocity at each wavelength (relative to line center)
    v = c_kms * (wavelength - lambda_rest) / lambda_rest

    # Optical depth as function of velocity
    # tau(v) = tau_ref * (v_phot / v)^n for v > v_phot
    # Common value n = 7 for exponential atmosphere
    n_power = kwargs.get('n_power', 7)

    # Initialize transmission
    transmission = np.ones_like(wavelength, dtype=float)

    # Absorption occurs at blueshifted wavelengths
    # v < 0 means blueshift
    abs_region = (v < -v_phot) & (v > -v_max)

    v_abs = np.abs(v[abs_region])
    tau_v = tau_ref * (v_phot / v_abs)**n_power

    # Apply absorption
    transmission[abs_region] = np.exp(-tau_v)

    # Emission component (simplified)
    # Emission fills in from resonance scattering
    em_region = (v > -v_phot) & (v < v_phot)

    # Source function ratio (S/I_c)
    W = kwargs.get('dilution_factor', 0.5)  # Geometric dilution

    # Add weak emission
    transmission[em_region] = 1.0 + W * tau_ref * 0.1 * np.exp(-((v[em_region])/v_phot)**2)

    return transmission


def blackbody_spectrum_with_p_cygni_lines(angstroms, redshift, rph, temp,
                                           line_list, v_phot, **kwargs):
    """
    Blackbody spectrum with multiple P-Cygni line profiles

    Convenience function combining blackbody continuum with P-Cygni lines.

    Parameters
    ----------
    angstroms : array
        Wavelength array in angstroms in observer frame
    redshift : float
        Source redshift
    rph : float
        Photosphere radius in cm
    temp : float
        Photosphere temperature in Kelvin
    line_list : list of dict
        Each dict: {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}
    v_phot : float
        Photospheric velocity in km/s

    Returns
    -------
    flux : array
        Flux in ergs/s/cm^2/angstrom

    Examples
    --------
    >>> wave = np.linspace(3500, 8500, 2000)
    >>> lines = [
    ...     {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0},
    ...     {'ion': 'S II', 'lambda': 5640, 'tau': 2.0},
    ...     {'ion': 'Ca II', 'lambda': 3945, 'tau': 5.0}
    ... ]
    >>> flux = blackbody_spectrum_with_p_cygni_lines(
    ...     wave, redshift=0.01, rph=1e15, temp=11000,
    ...     line_list=lines, v_phot=11000
    ... )
    """
    return multi_line_p_cygni_spectrum(
        wavelength=angstroms,
        redshift=redshift,
        continuum_model='blackbody',
        line_list=line_list,
        v_phot=v_phot,
        r_phot=rph,
        temperature=temp,
        **kwargs
    )


def spectrum_with_voigt_absorption_lines(wavelength, continuum_flux, line_params_list):
    """
    Add multiple Voigt absorption lines to a continuum

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    continuum_flux : array or float
        Continuum flux level
    line_params_list : list of dict
        Each dict: {'lambda': 6563, 'depth': 0.5, 'sigma': 1.0, 'gamma': 0.3}
        - lambda: Central wavelength
        - depth: Fractional depth of absorption (0 to 1)
        - sigma: Gaussian width in Angstroms
        - gamma: Lorentzian HWHM in Angstroms

    Returns
    -------
    flux : array
        Spectrum with Voigt absorption lines

    Examples
    --------
    >>> wave = np.linspace(6500, 6700, 1000)
    >>> lines = [
    ...     {'lambda': 6563, 'depth': 0.3, 'sigma': 2.0, 'gamma': 0.5},
    ...     {'lambda': 6583, 'depth': 0.1, 'sigma': 1.5, 'gamma': 0.3}
    ... ]
    >>> flux = spectrum_with_voigt_absorption_lines(wave, 1.0, lines)
    """
    from scipy.special import wofz

    if np.isscalar(continuum_flux):
        flux = np.ones_like(wavelength) * continuum_flux
    else:
        flux = continuum_flux.copy()

    for line in line_params_list:
        lambda_c = line['lambda']
        depth = line['depth']
        sigma = line['sigma']
        gamma = line['gamma']

        # Voigt profile calculation
        x = (wavelength - lambda_c) / (sigma * np.sqrt(2))
        y = gamma / (sigma * np.sqrt(2))
        z = x + 1j * y
        w = wofz(z)
        voigt = np.real(w) / (sigma * np.sqrt(2 * np.pi))

        # Normalize and apply absorption
        voigt_norm = voigt / np.max(voigt)
        flux *= (1 - depth * voigt_norm)

    return flux