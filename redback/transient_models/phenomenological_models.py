import numpy as np
from redback.utils import citation_wrapper
from redback.constants import speed_of_light_si
from redback.model_utils import get_cosmology_defaults


def smooth_exponential_powerlaw(time, a_1, tpeak, alpha_1, alpha_2, smoothing_factor, **kwargs):
    """
    Smoothed version of exponential power law

    :param time: time array in seconds
    :param a_1: exponential amplitude scale
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak: peak time in seconds
    :param smoothing_factor: controls transition smoothness (higher = smoother)
    :param kwargs: Additional parameters
    :return: In whatever units set by a_1
    """
    t_norm = time / tpeak

    # Smooth transition function using tanh or similar
    transition = 0.5 * (1 + np.tanh(smoothing_factor * np.log(t_norm)))

    # Pre-peak behavior
    pre_peak = a_1 * (t_norm ** alpha_1)

    # Post-peak behavior
    post_peak = a_1 * (t_norm ** alpha_2)

    # Smooth combination
    return pre_peak * (1 - transition) + post_peak * transition

def exp_rise_powerlaw_decline(t, t0, m_peak, tau_rise, alpha, t_peak, **kwargs):
    """
    Compute a smooth light-curve model (in magnitudes) with an exponential rise
    transitioning into a power-law decline, with a smooth (blended) peak.
    In all filters the shape is determined by the same t0, tau_rise, alpha, and t_peak;
    only m_peak differs from filter to filter.

    For t < t0, the function returns np.nan.
    For t >= t0, the model is constructed as a blend of:

      Rising phase:
          m_rise(t)  = m_peak + 1.086 * ((t_peak - t) / tau_rise)
      Declining phase:
          m_decline(t) = m_peak + 2.5 * alpha * log10((t - t0)/(t_peak - t0))

    A smooth transition is achieved by the switching (weight) function:

          weight(t) = 0.5 * [1 + tanh((t - t_peak)/delta)]

    so that the final magnitude is:

          m(t) = (1 - weight(t)) * m_rise(t) + weight(t) * m_decline(t)

    At t = t_peak, weight = 0.5 and both m_rise and m_decline equal m_peak,
    ensuring a smooth peak.

    Parameters
    ----------
    t : array_like
        1D array of times (e.g., in modified Julian days) at which to evaluate the model.
    t0 : float
        Start time of the transient event (e.g., explosion), in MJD.
    m_peak : float or array_like
        Peak magnitude(s) at t = t_peak. If an array is provided, each element is taken
        to correspond to a different filter.
    tau_rise : float
        Characteristic timescale (in days) for the exponential rise.
    alpha : float
        Power-law decay index governing the decline.
    t_peak : float
        Time (in MJD) at peak brightness (must satisfy t_peak > t0).
    delta : float, optional
        Smoothing parameter (in days) controlling the width of the transition around t_peak.
        If not provided, defaults to 50% of (t_peak - t0).

    Returns
    -------
    m_model : ndarray
        If m_peak is an array (multiple filters), returns a 2D array of shape (n_times, n_filters);
        if m_peak is a scalar, returns a 1D array (with NaN for t < t0).

    Examples
    --------
    Single filter:

    >>> t = np.linspace(58990, 59050, 300)
    >>> model1 = exp_rise_powerlaw_decline(t, t0=59000, m_peak=17.0, tau_rise=3.0,
    ...                                     alpha=1.5, t_peak=59010)

    Multiple filters (e.g., g, r, i bands):

    >>> t = np.linspace(58990, 59050, 300)
    >>> m_peaks = np.array([17.0, 17.5, 18.0])
    >>> model_multi = exp_rise_powerlaw_decline(t, t0=59000, m_peak=m_peaks, tau_rise=3.0,
    ...                                          alpha=1.5, t_peak=59010)
    >>> print(model_multi.shape)  # Expected shape: (300, 3)
    """
    # Convert t to a numpy array and force 1D.
    t = np.asarray(t).flatten()
    delta = kwargs.get('delta', 0.5)

    # Define default smoothing parameter delta if not provided.
    #     if delta is None:
    delta = (t_peak - t0) * delta  # default: 50% of the interval [t0, t_peak]

    # Ensure m_peak is at least 1D (so a scalar becomes an array of length 1).
    m_peak = np.atleast_1d(m_peak)
    n_filters = m_peak.shape[0]
    n_times = t.shape[0]

    # Preallocate model magnitude array with shape (n_times, n_filters)
    m_model = np.full((n_times, n_filters), np.nan, dtype=float)

    # Create a mask for times t >= t0.
    valid = t >= t0
    # Reshape t into a column vector for broadcasting: shape (n_times, 1)
    t_col = t.reshape(-1, 1)

    # Compute the switching (weight) function: weight = 0 when t << t_peak, 1 when t >> t_peak.
    weight = 0.5 * (1 + np.tanh((t_col - t_peak) / delta))

    # Rising phase model: for t < t_peak the flux is rising toward peak.
    # m_rise = m_peak + 1.086 * ((t_peak - t) / tau_rise)
    m_rise = m_peak[None, :] + 1.086 * ((t_peak - t_col) / tau_rise)

    # Declining phase model: power-law decline in flux gives a logarithmic increase in magnitude.
    # m_decline = m_peak + 2.5 * alpha * log10((t - t0)/(t_peak - t0))
    ratio = (t_col - t0) / (t_peak - t0)
    m_decline = m_peak[None, :] + 2.5 * alpha * np.log10(ratio)

    # Blend the two components using the switching weight.
    # For t << t_peak, tanh term ≈ -1 so weight ~ 0 and m ~ m_rise.
    # For t >> t_peak, tanh term ≈ +1 so weight ~ 1 and m ~ m_decline.
    m_blend = (1 - weight) * m_rise + weight * m_decline

    # Update m_model for valid times (t >= t0). For t < t0, m_model remains NaN.
    m_model[valid, :] = m_blend[valid, :]

    # If m_peak was given as a scalar, return a 1D array.
    if n_filters == 1:
        return m_model.flatten()
    return m_model

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract')
def bazin_sne(time, aa, bb, t0, tau_rise, tau_fall, **kwargs):
    """
    Bazin function for CCSN light curves with vectorized inputs.

    :param time: time array in arbitrary units
    :param aa: array (or float) of normalisations, if array this is unique to each 'band'
    :param bb: array (or float) of additive constants, if array this is unique to each 'band'
    :param t0: start time
    :param tau_rise: exponential rise time
    :param tau_fall: exponential fall time
    :return: matrix of flux values in units set by AA
    """
    if isinstance(aa, float):
        aa_values = [aa]
        bb_values = [bb]
    else:
        aa_values = aa
        bb_values = bb

    if len(aa_values) != len(bb_values):
        raise ValueError("Length of aa_values and bb_values must be the same.")

    # Compute flux for all aa and bb values
    flux_matrix = np.array([
        aa * (np.exp(-((time - t0) / tau_fall)) / (1 + np.exp(-(time - t0) / tau_rise))) + bb
        for aa, bb in zip(aa_values, bb_values)
    ])
    if isinstance(aa, float):
        return flux_matrix[0]
    else:
        return flux_matrix

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def villar_sne(time, aa, cc, t0, tau_rise, tau_fall, gamma, nu, **kwargs):
    """
    Villar function for SN light curves

    :param time: time array in arbitrary units
    :param aa: normalisation on the Villar function, amplotude
    :param cc: additive constant, baseline flux
    :param t0: "start" time
    :param tau_rise: exponential rise time
    :param tau_fall: exponential fall time
    :param gamma: plateau duration
    :param nu: related to beta and between 0 an 1; nu = -beta/gamma / A
    :param kwargs:
    :return: flux in units set by AA
    """
    mask1 = time < t0 + gamma
    mask2 = (time >= t0 + gamma)
    flux = np.zeros_like(time)
    norm = cc + (aa / (1 + np.exp(-(time - t0)/tau_rise)))
    flux[mask1] = norm[mask1] * (1 - (nu * ((time[mask1] - t0)/gamma)))
    flux[mask2] = norm[mask2] * ((1 - nu) * np.exp(-((time[mask2] - t0 - gamma)/tau_fall)))
    return np.concatenate((flux[mask1], flux[mask2]))

def evolving_blackbody(time, redshift, temperature_0, radius_0,
                            temp_rise_index, temp_decline_index, temp_peak_time,
                            radius_rise_index, radius_decline_index, radius_peak_time,
                            reference_time=1.0, **kwargs):
    """
    Blackbody spectrum with piecewise evolving temperature and radius

    :param time: time in observer frame in days
    :param redshift: source redshift
    :param temperature_0: initial blackbody temperature in Kelvin at reference_time
    :param radius_0: initial blackbody radius in cm at reference_time
    :param temp_rise_index: temperature rise T(t) ∝ t^temp_rise_index for t < temp_peak_time
    :param temp_decline_index: temperature decline T(t) ∝ t^(-temp_decline_index) for t > temp_peak_time
    :param temp_peak_time: time in days when temperature peaks
    :param radius_rise_index: radius rise R(t) ∝ t^radius_rise_index for t < radius_peak_time
    :param radius_decline_index: radius decline R(t) ∝ t^(-radius_decline_index) for t > radius_peak_time
    :param radius_peak_time: time in days when radius peaks
    :param reference_time: reference time for temperature_0, radius_0, and pl_amplitude in days (defaults to 1.0)
    :param kwargs: Additional parameters
    :param frequency: Required if output_format is 'flux_density'
    :param bands: Required if output_format is 'magnitude' or 'flux'
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional wavelength array in Angstroms to evaluate SED
    :param cosmology: Cosmology object for luminosity distance calculation
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from astropy.cosmology import Planck18 as cosmo
    from astropy import units as uu
    from redback.utils import lambda_to_nu, calc_kcorrected_properties
    import redback.sed as sed
    from redback.sed import flux_density_to_spectrum
    from collections import namedtuple
    cosmology, dl = get_cosmology_defaults(redshift, kwargs)
    reference_wavelength = kwargs.get('reference_wavelength', 5000.0)  # Angstroms

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        # Calculate evolving temperature and radius
        temperature, radius = _powerlaw_blackbody_evolution(time=time, temperature_0=temperature_0, radius_0=radius_0,
                                                            temp_rise_index=temp_rise_index,
                                                            temp_decline_index=temp_decline_index,
                                                            temp_peak_time=temp_peak_time,
                                                            radius_rise_index=radius_rise_index,
                                                            radius_decline_index=radius_decline_index,
                                                            radius_peak_time=radius_peak_time,
                                                            reference_time=reference_time)

        # Create combined SED with time-evolving power law
        sed_combined = sed.Blackbody(temperature=temperature, r_photosphere=radius,
                                                 frequency=frequency, luminosity_distance=dl)
        flux_density = sed_combined.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)

        # Calculate evolving temperature and radius
        temperature, radius = _powerlaw_blackbody_evolution(time=time, temperature_0=temperature_0, radius_0=radius_0,
                                                            temp_rise_index=temp_rise_index,
                                                            temp_decline_index=temp_decline_index,
                                                            temp_peak_time=temp_peak_time,
                                                            radius_rise_index=radius_rise_index,
                                                            radius_decline_index=radius_decline_index,
                                                            radius_peak_time=radius_peak_time,
                                                            reference_time=reference_time)

        # Create combined SED with time-evolving power law
        sed_combined = sed.Blackbody(temperature=temperature, r_photosphere=radius,
                                                 frequency=frequency[:, None], luminosity_distance=dl)
        fmjy = sed_combined.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

def evolving_blackbody_with_features(time, redshift, temperature_0, radius_0,
                                     temp_rise_index, temp_decline_index, temp_peak_time,
                                     radius_rise_index, radius_decline_index, radius_peak_time,
                                     reference_time=1.0, **kwargs):
    """
    Blackbody spectrum with piecewise evolving temperature and radius, plus time-dependent spectral features

    :param time: time in observer frame in days
    :param redshift: source redshift
    :param temperature_0: initial blackbody temperature in Kelvin at reference_time
    :param radius_0: initial blackbody radius in cm at reference_time
    :param temp_rise_index: temperature rise T(t) ∝ t^temp_rise_index for t < temp_peak_time
    :param temp_decline_index: temperature decline T(t) ∝ t^(-temp_decline_index) for t > temp_peak_time
    :param temp_peak_time: time in days when temperature peaks
    :param radius_rise_index: radius rise R(t) ∝ t^radius_rise_index for t < radius_peak_time
    :param radius_decline_index: radius decline R(t) ∝ t^(-radius_decline_index) for t > radius_peak_time
    :param radius_peak_time: time in days when radius peaks
    :param reference_time: reference time for temperature_0, radius_0 in days (defaults to 1.0)
    :param kwargs: Additional parameters
    :param frequency: Required if output_format is 'flux_density'
    :param bands: Required if output_format is 'magnitude' or 'flux'
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional wavelength array in Angstroms to evaluate SED
    :param cosmology: Cosmology object for luminosity distance calculation

    Feature Parameters (dynamically numbered):
    Features are defined by groups of parameters with pattern: {param}_feature_{N}
    where N starts from 1. All features with the same N are grouped together.

    Required for each feature N:
    :param rest_wavelength_feature_N: Central wavelength in Angstroms
    :param sigma_feature_N: Gaussian width in Angstroms
    :param amplitude_feature_N: Amplitude (negative=absorption, positive=emission)
    :param t_start_feature_N: Start time in source-frame days
    :param t_end_feature_N: End time in source-frame days

    Optional for each feature N (smooth mode only):
    :param t_rise_feature_N: Rise time in source-frame days (default: 2.0)
    :param t_fall_feature_N: Fall time in source-frame days (default: 5.0)

    General parameters:
    :param evolution_mode: 'smooth' or 'sharp' (default: 'smooth')
    :param use_default_features: If True and no custom features found, use defaults (default: False)

    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from astropy.cosmology import Planck18 as cosmo
    from astropy import units as uu
    from redback.utils import lambda_to_nu, calc_kcorrected_properties
    import redback.sed as sed
    from redback.transient_models.supernova_models import build_spectral_feature_list
    from collections import namedtuple
    import numpy as np

    cosmology, dl = get_cosmology_defaults(redshift, kwargs)

    # Build feature list from numbered parameters
    feature_list = build_spectral_feature_list(**kwargs)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        # Calculate evolving temperature and radius
        temperature, radius = _powerlaw_blackbody_evolution(
            time=time, temperature_0=temperature_0, radius_0=radius_0,
            temp_rise_index=temp_rise_index,
            temp_decline_index=temp_decline_index,
            temp_peak_time=temp_peak_time,
            radius_rise_index=radius_rise_index,
            radius_decline_index=radius_decline_index,
            radius_peak_time=radius_peak_time,
            reference_time=reference_time
        )

        # Convert time from days to seconds for feature application
        time_seconds = time * 24 * 3600

        # Create SED with spectral features
        sed_combined = sed.BlackbodyWithSpectralFeatures(
            temperature=temperature,
            r_photosphere=radius,
            frequency=frequency,
            luminosity_distance=dl,
            time=time_seconds,
            feature_list=feature_list,
            evolution_mode=kwargs.get('evolution_mode', 'smooth')
        )
        flux_density = sed_combined.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)

    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(
            frequency=lambda_to_nu(lambda_observer_frame),
            redshift=redshift,
            time=time_observer_frame
        )

        # Calculate evolving temperature and radius
        temperature, radius = _powerlaw_blackbody_evolution(
            time=time, temperature_0=temperature_0, radius_0=radius_0,
            temp_rise_index=temp_rise_index,
            temp_decline_index=temp_decline_index,
            temp_peak_time=temp_peak_time,
            radius_rise_index=radius_rise_index,
            radius_decline_index=radius_decline_index,
            radius_peak_time=radius_peak_time,
            reference_time=reference_time
        )

        # Convert time from days to seconds for feature application
        time_seconds = time * 24 * 3600

        # Create SED with spectral features
        sed_combined = sed.BlackbodyWithSpectralFeatures(
            temperature=temperature,
            r_photosphere=radius,
            frequency=frequency[:, None],
            luminosity_distance=dl,
            time=time_seconds,
            feature_list=feature_list,
            evolution_mode=kwargs.get('evolution_mode', 'smooth')
        )
        fmjy = sed_combined.flux_density.T
        spectra = sed.flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)

        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(
                time=time_observer_frame,
                lambdas=lambda_observer_frame,
                spectra=spectra
            )
        else:
            return sed.get_correct_output_format_from_spectra(
                time=time_obs,
                time_eval=time_observer_frame,
                spectra=spectra,
                lambda_array=lambda_observer_frame,
                **kwargs
            )

def powerlaw_plus_blackbody(time, redshift, pl_amplitude, pl_slope, pl_evolution_index, temperature_0, radius_0,
                            temp_rise_index, temp_decline_index, temp_peak_time,
                            radius_rise_index, radius_decline_index, radius_peak_time,
                            reference_time=1.0, **kwargs):
    """
    Power law + blackbody spectrum with piecewise evolving temperature and radius

    :param time: time in observer frame in days
    :param redshift: source redshift
    :param pl_amplitude: power law amplitude at reference wavelength at reference_time (erg/s/cm^2/Angstrom)
    :param pl_slope: power law slope (F_lambda ∝ lambda^slope)
    :param pl_evolution_index: power law time evolution F_pl(t) ∝ t^(-pl_evolution_index)
    :param temperature_0: initial blackbody temperature in Kelvin at reference_time
    :param radius_0: initial blackbody radius in cm at reference_time
    :param temp_rise_index: temperature rise T(t) ∝ t^temp_rise_index for t < temp_peak_time
    :param temp_decline_index: temperature decline T(t) ∝ t^(-temp_decline_index) for t > temp_peak_time
    :param temp_peak_time: time in days when temperature peaks
    :param radius_rise_index: radius rise R(t) ∝ t^radius_rise_index for t < radius_peak_time
    :param radius_decline_index: radius decline R(t) ∝ t^(-radius_decline_index) for t > radius_peak_time
    :param radius_peak_time: time in days when radius peaks
    :param reference_time: reference time for temperature_0, radius_0, and pl_amplitude in days (defaults to 1.0)
    :param kwargs: Additional parameters
    :param reference_wavelength: wavelength for power law amplitude normalization in Angstroms (default 5000)
    :param frequency: Required if output_format is 'flux_density'
    :param bands: Required if output_format is 'magnitude' or 'flux'
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional wavelength array in Angstroms to evaluate SED
    :param cosmology: Cosmology object for luminosity distance calculation
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from astropy.cosmology import Planck18 as cosmo
    from astropy import units as uu
    from redback.utils import lambda_to_nu, calc_kcorrected_properties
    import redback.sed as sed
    from redback.sed import flux_density_to_spectrum
    from collections import namedtuple

    cosmology, dl = get_cosmology_defaults(redshift, kwargs)
    reference_wavelength = kwargs.get('reference_wavelength', 5000.0)  # Angstroms

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        # Calculate evolving temperature and radius
        temperature, radius = _powerlaw_blackbody_evolution(time=time, temperature_0=temperature_0, radius_0=radius_0,
                                                            temp_rise_index=temp_rise_index,
                                                            temp_decline_index=temp_decline_index,
                                                            temp_peak_time=temp_peak_time,
                                                            radius_rise_index=radius_rise_index,
                                                            radius_decline_index=radius_decline_index,
                                                            radius_peak_time=radius_peak_time,
                                                            reference_time=reference_time)

        # Create combined SED with time-evolving power law
        sed_combined = sed.PowerlawPlusBlackbody(temperature=temperature, r_photosphere=radius,
                                                 pl_amplitude=pl_amplitude, pl_slope=pl_slope,
                                                 pl_evolution_index=pl_evolution_index, time=time,
                                                 reference_wavelength=reference_wavelength,
                                                 frequency=frequency, luminosity_distance=dl)
        flux_density = sed_combined.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
        time_temp = np.geomspace(0.1, 3000, 300)  # in days
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)

        # Calculate evolving temperature and radius
        temperature, radius = _powerlaw_blackbody_evolution(time=time, temperature_0=temperature_0, radius_0=radius_0,
                                                            temp_rise_index=temp_rise_index,
                                                            temp_decline_index=temp_decline_index,
                                                            temp_peak_time=temp_peak_time,
                                                            radius_rise_index=radius_rise_index,
                                                            radius_decline_index=radius_decline_index,
                                                            radius_peak_time=radius_peak_time,
                                                            reference_time=reference_time)

        # Create combined SED with time-evolving power law
        sed_combined = sed.PowerlawPlusBlackbody(temperature=temperature, r_photosphere=radius,
                                                 pl_amplitude=pl_amplitude, pl_slope=pl_slope,
                                                 pl_evolution_index=pl_evolution_index, time=time,
                                                 reference_wavelength=reference_wavelength,
                                                 frequency=frequency[:, None], luminosity_distance=dl)
        fmjy = sed_combined.flux_density.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


def _powerlaw_blackbody_evolution(time, temperature_0, radius_0, temp_rise_index, temp_decline_index,
                                  temp_peak_time, radius_rise_index, radius_decline_index, radius_peak_time,
                                  reference_time=1.0, **kwargs):
    """
    Calculate evolving temperature and radius with piecewise power-law evolution

    :param time: time array in days
    :param temperature_0: initial temperature at reference_time
    :param radius_0: initial radius at reference_time
    :param temp_rise_index: temperature rise index
    :param temp_decline_index: temperature decline index
    :param temp_peak_time: time when temperature peaks
    :param radius_rise_index: radius rise index
    :param radius_decline_index: radius decline index
    :param radius_peak_time: time when radius peaks
    :param reference_time: reference time for temperature_0 and radius_0 (defaults to 1.0 day)
    :return: temperature and radius values (scalars if time is scalar)
    """
    time = np.atleast_1d(time)

    # Temperature evolution
    temp_peak = temperature_0 * (temp_peak_time / reference_time) ** temp_rise_index
    rise_mask_temp = time <= temp_peak_time
    decline_mask_temp = ~rise_mask_temp

    temperature = np.zeros_like(time)
    temperature[rise_mask_temp] = temperature_0 * (time[rise_mask_temp] / reference_time) ** temp_rise_index
    temperature[decline_mask_temp] = temp_peak * (time[decline_mask_temp] / temp_peak_time) ** (-temp_decline_index)

    # Radius evolution
    radius_peak = radius_0 * (radius_peak_time / reference_time) ** radius_rise_index
    rise_mask_radius = time <= radius_peak_time
    decline_mask_radius = ~rise_mask_radius

    radius = np.zeros_like(time)
    radius[rise_mask_radius] = radius_0 * (time[rise_mask_radius] / reference_time) ** radius_rise_index
    radius[decline_mask_radius] = radius_peak * (time[decline_mask_radius] / radius_peak_time) ** (
        -radius_decline_index)

    # Return scalars if input was scalar
    if len(time) == 1:
        return temperature[0], radius[0]
    else:
        return temperature, radius

def fallback_lbol(time, logl1, tr, **kwargs):
    """
    :param time: time in seconds
    :param logl1: luminosity scale in log 10 ergs
    :param tr: transition time for flat luminosity to power-law decay
    :return: lbol
    """
    l1 = 10**logl1
    time = time * 86400
    tr = tr * 86400
    lbol = l1 * time**(-5./3.)
    lbol[time < tr] = l1 * tr**(-5./3.)
    return lbol

def line_spectrum(wavelength, line_amp, cont_amp, x0, **kwargs):
    """
    A gaussian to add or subtract from a continuum spectrum to mimic absorption or emission lines

    :param wavelength: wavelength array in whatever units
    :param line_amp: line amplitude scale
    :param cont_amp: Continuum amplitude scale
    :param x0: Position of emission line
    :return: spectrum in whatever units set by line_amp
    """
    spectrum = line_amp / cont_amp * np.exp(-(wavelength - x0) ** 2. / (2 * cont_amp ** 2) )
    return spectrum

def line_spectrum_with_velocity_dispersion(angstroms, wavelength_center, line_strength, velocity_dispersion):
    """
    A Gaussian line profile with velocity dispersion

    :param angstroms: wavelength array in angstroms or arbitrary units
    :param wavelength_center: center of the line in angstroms
    :param line_strength: line amplitude scale
    :param velocity_dispersion: velocity in m/s
    :return: spectrum in whatever units set by line_strength
    """

    # Calculate the Doppler shift for each wavelength using Gaussian profile
    intensity = line_strength * np.exp(-0.5 * ((angstroms - wavelength_center) / wavelength_center * speed_of_light_si / velocity_dispersion) ** 2)
    return intensity

def gaussian_rise(time, a_1, peak_time, sigma_t, **kwargs):
    """
    :param time: time array in whatver time units
    :param a_1: gaussian rise amplitude scale
    :param peak_time: peak time in whatever units
    :param sigma_t: the sharpness of the Gaussian
    :return: In whatever units set by a_1 
    """
    total = a_1 * np.exp(-(time - peak_time)**2. / (2 * sigma_t ** 2))
    return total

def exponential_powerlaw(time, a_1, alpha_1, alpha_2, tpeak, **kwargs):
    """
    :param time: time array in seconds
    :param a_1: exponential amplitude scale
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak: peak time in seconds
    :param kwargs:
    :return: In whatever units set by a_1
    """
    total = a_1 * (1 - np.exp(-time/tpeak))**alpha_1 * (time/tpeak)**(-alpha_2)
    return total


def two_component_powerlaw(time, a_1, alpha_1,
                           delta_time_one, alpha_2, **kwargs):
    """
    Two component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :return: In whatever units set by a_1
    """
    time_one = delta_time_one
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    w = np.where(time < time_one)
    x = np.where(time > time_one)

    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2

    total = np.concatenate((f1, f2))

    return total


def three_component_powerlaw(time, a_1, alpha_1,
                             delta_time_one, alpha_2,
                             delta_time_two, alpha_3, **kwargs):
    """
    Three component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :return: In whatever units set by a_1
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where(time > time_two)
    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2
    f3 = amplitude_three * time[y] ** alpha_3

    total = np.concatenate((f1, f2, f3))
    return total


def four_component_powerlaw(time, a_1, alpha_1, delta_time_one,
                            alpha_2, delta_time_two,
                            alpha_3, delta_time_three,
                            alpha_4, **kwargs):
    """
    Four component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :return: In whatever units set by a_1
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where((time_two < time) & (time < time_three))
    z = np.where(time > time_three)
    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2
    f3 = amplitude_three * time[y] ** alpha_3
    f4 = amplitude_four * time[z] ** alpha_4

    total = np.concatenate((f1, f2, f3, f4))

    return total


def five_component_powerlaw(time, a_1, alpha_1,
                            delta_time_one, alpha_2,
                            delta_time_two, alpha_3,
                            delta_time_three, alpha_4,
                            delta_time_four, alpha_5, **kwargs):
    """
    Five component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    :return: In whatever units set by a_1
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four

    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** alpha_4 / (time_four ** alpha_5)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where(time > time_four)

    f1 = a_1 * time[u] ** alpha_1
    f2 = amplitude_two * time[v] ** alpha_2
    f3 = amplitude_three * time[w] ** alpha_3
    f4 = amplitude_four * time[x] ** alpha_4
    f5 = amplitude_five * time[y] ** alpha_5

    total = np.concatenate((f1, f2, f3, f4, f5))

    return total


def six_component_powerlaw(time, a_1, alpha_1,
                           delta_time_one, alpha_2,
                           delta_time_two, alpha_3,
                           delta_time_three, alpha_4,
                           delta_time_four, alpha_5,
                           delta_time_five, alpha_6, **kwargs):
    """
    six component powerlaw model

    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of first power law
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    :param delta_time_five: time between fourth and fifth power laws
    :param alpha_6: power law decay exponent for sixth power law
    :return: In whatever units set by a_1
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four
    time_five = time_four + delta_time_five

    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** alpha_4 / (time_four ** alpha_5)
    amplitude_six = amplitude_five * time_five ** alpha_5 / (time_five ** alpha_6)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where((time_four < time) & (time < time_five))
    z = np.where(time > time_five)

    f1 = a_1 * time[u] ** alpha_1
    f2 = amplitude_two * time[v] ** alpha_2
    f3 = amplitude_three * time[w] ** alpha_3
    f4 = amplitude_four * time[x] ** alpha_4
    f5 = amplitude_five * time[y] ** alpha_5
    f6 = amplitude_six * time[z] ** alpha_6

    total = np.concatenate((f1, f2, f3, f4, f5, f6))
    return total