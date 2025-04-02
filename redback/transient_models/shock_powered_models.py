import numpy as np
from collections import namedtuple
from scipy import special
from scipy.interpolate import interp1d
import redback.constants as cc
from astropy import units as uu
import redback.sed as sed
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, lambda_to_nu


def _shockcooling_morag(time, v_shock, m_env, f_rho_m, radius, kappa):
    """
    Compute shock‑cooling parameters from the model of Morag, Sapir, & Waxman.

    This function calculates the bolometric luminosity, effective temperature, and
    effective photospheric radius based on a shock‑cooling model. The model uses
    several scaling relations. For example, the breakout time is given by:

    .. math::

       t_{\mathrm{br}} = t_{\mathrm{br,0}}\, R^{1.26}\, v_s^{-1.13}\, (f_\rho M)^{-0.13}\,,

    where the coefficient :math:`t_{\mathrm{br,0}} = 0.036` days, and similar
    relations exist for the breakout luminosity :math:`L_{\mathrm{br}}` and the breakout
    color temperature :math:`T_{\mathrm{col,br}}`. A transparency time is also defined:

    .. math::

       t_{\mathrm{tr}} = t_{\mathrm{tr,0}}\, \sqrt{\frac{\kappa\, M_{\mathrm{env}}}{v_s}}\,,

    with :math:`t_{\mathrm{tr,0}} = 19.5` days.

    A dimensionless time is defined as

    .. math::

       \tilde{t} = \frac{t - t_{\mathrm{exp}}}{t_{\mathrm{br}}}\,.

    The bolometric luminosity is then computed using

    .. math::

       L(t) = L_{\mathrm{br}}\left\{\tilde{t}^{-4/3} + A\,\exp\left[-\left(\frac{a\,t}{t_{\mathrm{tr}}}\right)^\alpha\right]\tilde{t}^{-0.17}\right\}\,,

    and the color temperature by

    .. math::

       T_{\mathrm{col}}(t) = T_{\mathrm{col,br}}\, \min\left(0.97\,\tilde{t}^{-1/3},\; \tilde{t}^{-0.45}\right)\,.

    The effective temperature in Kelvin is obtained by converting the color temperature
    (in eV) using the Boltzmann constant and the photospheric radius is estimated via the
    Stefan–Boltzmann law

    .. math::

       R_{\mathrm{bb}} = \frac{1}{T^2}\,\sqrt{\frac{L}{4\pi\,\sigma_{\mathrm{SB}}}}\,.

    :param time: Time at which to evaluate the model (in days).
    :type time: float or array_like
    :param v_shock: Shock speed in units of 10^(8.5) cm/s.
    :type v_shock: float
    :param m_env: Envelope mass in solar masses.
    :type m_env: float
    :param f_rho_m: Product of the dimensionless factor f_ρ and the ejecta mass in solar masses.
    :type f_rho_m: float
    :param radius: Progenitor radius in units of 10^(13) cm.
    :type radius: float
    :param t_exp: Explosion epoch in days. Default is 0.
    :type t_exp: float
    :param kappa: Opacity relative to the electron scattering opacity. Default is 1.
    :type kappa: float

    :return: A namedtuple ``ShockCoolingResult`` with the following fields:
             - luminosity: Bolometric luminosity in erg/s.
             - t_photosphere: Effective temperature in Kelvin.
             - r_photosphere: Effective photospheric radius in cm.
             - min_time: Minimum time for which the model is valid in days.
             - max_time: Maximum time for which the model is valid in days.
    :rtype: ShockCoolingResult
    """
    # Normalization constants
    v_norm = 10 ** 8.5  # (cm/s) for shock speed normalization, roughly 3.16e8 cm/s.
    kappa_norm = 0.34  # (cm²/g) fiducial opacity.
    R_norm = 1e13  # (cm) normalization for progenitor radius.

    # Convert absolute inputs to dimensionless units required by the model
    v_shock = v_shock / v_norm  # Dimensionless shock speed.
    kappa = kappa / kappa_norm  # Dimensionless opacity.
    radius = radius / R_norm  # Dimensionless progenitor radius.

    # --- Model coefficients ---
    t_br_0 = 0.036  # days (0.86 h)
    L_br_0 = 3.69e42  # erg/s
    T_col_br_0 = 8.19  # eV
    t_tr_0 = 19.5  # days
    A = 0.9
    a_value = 2.0
    alpha = 0.5
    t07ev0 = 6.86

    # --- Physical constants ---
    k_B_ev = 8.617333262e-5  # Boltzmann constant in eV/K
    sigma_sb = cc.sigma_sb  # Stefan-Boltzmann constant in erg/(s cm^2 K^4)
    c3 = 1.0 / np.sqrt(4.0 * np.pi * sigma_sb)

    # --- Compute intermediate scales ---
    t_br = t_br_0 * np.power(radius, 1.26) * np.power(v_shock, -1.13) * np.power(f_rho_m, -0.13)
    L_br = L_br_0 * np.power(radius, 0.78) * np.power(v_shock, 2.11) * np.power(f_rho_m, 0.11) * np.power(kappa, -0.89)
    T_col_br = T_col_br_0 * np.power(radius, -0.32) * np.power(v_shock, 0.58) * np.power(f_rho_m, 0.03) * np.power(kappa, -0.22)
    t_tr = t_tr_0 * np.sqrt(kappa * m_env / v_shock)

    # --- Adjust time for the explosion epoch and compute dimensionless time ---
    ttilde = time / t_br

    # --- Compute bolometric luminosity ---
    luminosity = L_br * (np.power(ttilde, -4.0 / 3.0) +
                         A * np.exp(-np.power(a_value * time / t_tr, alpha)) * np.power(ttilde, -0.17))

    # --- Compute color temperature ---
    T_col = T_col_br * np.minimum(0.97 * np.power(ttilde, -1.0 / 3.0),
                                  np.power(ttilde, -0.45))

    # Convert color temperature from eV to Kelvin.
    t_photosphere = T_col / k_B_ev

    # --- Compute effective photospheric radius using the Stefan-Boltzmann law ---
    r_photosphere = c3 * np.sqrt(luminosity) / np.power(t_photosphere, 2)

    min_time = 0.012 * radius

    t07ev = t07ev0 * radius ** 0.56 * v_shock ** 0.16 * kappa ** -0.61 * f_rho_m ** -0.06
    max_time = np.minimum(t07ev, t_tr / a_value)
    ShockCoolingResult = namedtuple('ShockCoolingResult', ['t_photosphere', 'r_photosphere',
                                                           'luminosity', 'min_time', 'max_time'])
    return ShockCoolingResult(t_photosphere=t_photosphere, r_photosphere=r_photosphere, luminosity=luminosity,
                              min_time=min_time, max_time=max_time)

@citation_wrapper('https://academic.oup.com/mnras/article/522/2/2764/7086123#443111844')
def shockcooling_morag_bolometric(time, v_shock, m_env, f_rho_m, radius, kappa, **kwargs):
    """
    Bolometric lightcurve following the Morag, Sapir, & Waxman model.

    :param time: time in source frame in days
    :param v_shock: shock speed in km/s
    :param m_env: envelope mass in solar masses
    :param f_rho_m: f_rho * M (with M in solar masses). f_rho is typically, of order unity
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :return: bolometric luminosity in erg/s
    """
    v_shock = v_shock * 1e5
    radius = radius * 1e13
    output = _shockcooling_morag(time, v_shock, m_env, f_rho_m, radius, kappa)
    lum = output.luminosity
    return lum

@citation_wrapper('https://academic.oup.com/mnras/article/522/2/2764/7086123#443111844')
def shockcooling_morag(time, redshift, v_shock, m_env, f_rho_m, radius, kappa, **kwargs):
    """
    Lightcurve following the Morag, Sapir, & Waxman model

    :param time: time in observer frame in days
    :param redshift: redshift
    :param v_shock: shock speed in km/s
    :param m_env: envelope mass in solar masses
    :param f_rho_m: f_rho * M (with M in solar masses). f_rho is typically, of order unity
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :param time_temp: Optional argument to set your desired time array (in source frame days) to evaluate the model on.
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    v_shock = v_shock * 1e5
    radius = radius * 1e13
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_temp = kwargs.get('time_temp', np.linspace(0.01, 60, 100))
    time_obs = time
    output = _shockcooling_morag(time_temp, v_shock, m_env, f_rho_m, radius, kappa)
    if kwargs['output_format'] == 'namedtuple':
        return output
    elif kwargs['output_format'] == 'flux_density':
        time = time_obs
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=output.t_photosphere)
        rad_func = interp1d(time_temp, y=output.r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)

        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = sed.blackbody_to_flux_density(temperature=output.t_photosphere,
                                             r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)
def _shockcooling_sapirandwaxman(time, v_shock, m_env, f_rho_m, radius, kappa, nn=1.5, RW=False):
    """
    Calculate shock-cooling properties following the Sapir & Waxman (and Rabinak & Waxman) model

    The model equations (with time t in days) are given as follows:

    Pre-exponential luminosity:

    .. math::
       L_{RW} = L_0 \left[ \frac{t^2 \, v_{\mathrm{dim}}}{f_{\rho} \, \kappa_{\mathrm{dim}}}\right]^{-\epsilon_2}
       v_{\mathrm{dim}}^2 \frac{R_{\mathrm{dim}}}{\kappa_{\mathrm{dim}}},

    where

    .. math::
       v_{\mathrm{dim}} = \frac{v_s}{10^{8.5}\,\mathrm{cm/s}}, \quad
       \kappa_{\mathrm{dim}} = \frac{\kappa}{0.34\,\mathrm{cm^2/g}}, \quad
       R_{\mathrm{dim}} = \frac{R}{10^{13}\,\mathrm{cm}}.

    The transparency timescale is defined as:

    .. math::
       t_{\mathrm{tr}} = 19.5 \, \left[\frac{\kappa_{\mathrm{dim}} \, M_{\mathrm{env}}}{v_{\mathrm{dim}}}\right]^{1/2}.

    The full luminosity is then:

    .. math::
       L = L_{RW} \, A \, \exp\left[-\left(\frac{a\, t}{t_{\mathrm{tr}}}\right)^{\alpha}\right].

    The photospheric temperature in eV is given by:

    .. math::
       T_{\mathrm{ph}} = T_0 \left[\frac{t^2 \, v_{\mathrm{dim}}^2}{f_{\rho} \, \kappa_{\mathrm{dim}}}\right]^{\epsilon_1}
       \kappa_{\mathrm{dim}}^{-0.25} \, t^{-0.5} \, R_{\mathrm{dim}}^{0.25},

    and a color correction:

    .. math::
       T_{\mathrm{col}} = T_{\mathrm{ph}} \; \times \; \text{(color correction factor)}.

    Finally, converting from eV to Kelvin:

    .. math::
       T_{\mathrm{K}} = \frac{T_{\mathrm{col}}}{k_B}, \quad \text{with } k_B = 8.61733 \times 10^{-5}\,\mathrm{eV/K},

    and the photospheric radius is derived using the Stefan–Boltzmann relation:

    .. math::
       R_{\mathrm{bb}} = \frac{\sqrt{L/(4\pi\sigma)}}{T_{\mathrm{K}}^2} \quad
       \text{with } \sigma = 5.670374419 \times 10^{-5}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,K^{-4}}.

    :param time: Time (in days) at which to evaluate the model.
    :type time: float or array-like
    :param v_shock: Shock speed in cm/s.
    :type v_shock: float
    :param m_env: Envelope mass in solar masses.
    :type m_env: float
    :param f_rho_m: The product :math:`f_{\\rho} \, M` (with M in solar masses). Typically of order unity.
    :type f_rho_m: float
    :param radius: Progenitor radius in cm.
    :type radius: float
    :param kappa: Ejecta opacity in cm²/g (e.g., approximately 0.34 for pure electron scattering).
    :type kappa: float
    :param nn: The polytropic index of the progenitor. Must be either 1.5 (default) or 3.0.
    :type nn: float, optional
    :param RW: If True, use the simplified Rabinak & Waxman formulation (sets a = 0 and adjusts the temperature correction factor).
    :type RW: bool, optional
    :return: A named tuple with the following fields:
             - **t_photosphere**: Color temperature in Kelvin,
             - **r_photosphere**: Derived photospheric radius in cm,
             - **luminosity**: Bolometric luminosity in erg/s.
            - **min_time**: Minimum time for which the model is valid in days.
            - **max_time**: Maximum time for which the model is valid in days.
    :rtype: namedtuple
    """
    # Normalization constants
    v_norm = 10 ** 8.5  # (cm/s) for shock speed normalization, roughly 3.16e8 cm/s.
    kappa_norm = 0.34  # (cm²/g) fiducial opacity.
    R_norm = 1e13  # (cm) normalization for progenitor radius.

    # Set parameters based on the chosen polytropic index n.
    if nn == 1.5:
        AA = 0.94
        a_val = 1.67
        alpha = 0.8
        epsilon_1 = 0.027
        epsilon_2 = 0.086
        L_0 = 2.0e42  # erg/s
        T_0 = 1.61  # eV
        Tph_to_Tcol = 1.1
    elif nn == 3.0:
        AA = 0.79
        a_val = 4.57
        alpha = 0.73
        epsilon_1 = 0.016
        epsilon_2 = 0.175
        L_0 = 2.1e42  # erg/s
        T_0 = 1.69  # eV
        Tph_to_Tcol = 1.0
    else:
        raise ValueError("n can only be 1.5 or 3.0.")

    if RW:
        a_val = 0.0
        Tph_to_Tcol = 1.2


    # Convert absolute inputs to dimensionless units required by the model
    v_dim = v_shock / v_norm  # Dimensionless shock speed.
    kappa_dim = kappa / kappa_norm  # Dimensionless opacity.
    R_dim = radius / R_norm  # Dimensionless progenitor radius.

    # Pre-exponential luminosity
    L_RW = L_0 * np.power(time ** 2 * v_dim / (f_rho_m * kappa_dim), -epsilon_2) \
           * np.power(v_dim, 2) * R_dim / kappa_dim

    # Transparency timescale in days
    t_tr = 19.5 * np.power((kappa_dim * m_env / v_dim), 0.5)

    # Full luminosity with exponential cutoff
    lum = L_RW * AA * np.exp(-np.power(a_val * time / t_tr, alpha))

    # Photospheric temperature in eV
    T_ph = T_0 * np.power(time ** 2 * np.power(v_dim, 2) / (f_rho_m * kappa_dim), epsilon_1) \
           * np.power(kappa_dim, -0.25) * np.power(time, -0.5) * np.power(R_dim, 0.25)
    # Apply the color correction factor
    T_col = T_ph * Tph_to_Tcol

    # Convert temperature from eV to Kelvin
    k_B = 8.61733e-5  # Boltzmann constant in eV/K
    temperature_K = T_col / k_B

    min_time = 0.2 * R_dim / v_dim * np.maximum(0.5, R_dim ** 0.4 * (f_rho_m * kappa) ** -0.2 * v_dim ** -0.7)
    max_time = 7.4 * (R_dim / kappa_dim) ** 0.55
    #
    # # turn luminosity at times < min_time to 0 and at times > max_time to a small number
    # lum = np.where(time < min_time, 1e4, lum)
    # lum = np.where(time > max_time, 1e4, lum)

    # Calculate the photospheric radius using the Stefan–Boltzmann law
    sigma = cc.sigma_sb  # Stefan–Boltzmann constant [erg s^-1 cm^-2 K^-4]
    radius_cm = np.sqrt(lum / (4 * np.pi * sigma)) / (temperature_K ** 2)

    ShockCoolingResult = namedtuple('ShockCoolingResult', ['t_photosphere', 'r_photosphere',
                                                           'luminosity', 'min_time', 'max_time'])
    return ShockCoolingResult(t_photosphere=temperature_K, r_photosphere=radius_cm, luminosity=lum,
                              min_time=min_time, max_time=max_time)


@citation_wrapper('https://iopscience.iop.org/article/10.3847/1538-4357/aa64df')
def shockcooling_sapirandwaxman_bolometric(time, v_shock, m_env, f_rho_m, radius, kappa, **kwargs):
    """
    Bolometric lightcurve following the Sapir & Waxman (and Rabinak & Waxman) model.

    :param time: time in source frame in days
    :param v_shock: shock speed in km/s
    :param m_env: envelope mass in solar masses
    :param f_rho_m: f_rho * M (with M in solar masses). f_rho is typically, of order unity
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :param n: index of progenitor density profile, 1.5 (default) or 3.0
    :param RW: If True, use the simplified Rabinak & Waxman formulation (off by default)
    :return: bolometric luminosity in erg/s
    """
    n = kwargs.get('n', 1.5)
    v_shock = v_shock * 1e5
    RW = kwargs.get('RW', False)
    radius = radius * 1e13
    output = _shockcooling_sapirandwaxman(time, v_shock, m_env, f_rho_m, radius, kappa, nn=n, RW=RW)
    lum = output.luminosity
    return lum

@citation_wrapper('https://iopscience.iop.org/article/10.3847/1538-4357/aa64df')
def shockcooling_sapirandwaxman(time, redshift, v_shock, m_env, f_rho_m, radius, kappa, **kwargs):
    """
    Lightcurve following the Sapir & Waxman (and Rabinak & Waxman) model

    :param time: time in observer frame in days
    :param redshift: redshift
    :param v_shock: shock speed in km/s
    :param m_env: envelope mass in solar masses
    :param f_rho_m: f_rho * M (with M in solar masses). f_rho is typically, of order unity
    :param radius: star/envelope radius in units of 10^13 cm
    :param kappa: opacity in cm^2/g
    :param kwargs: Additional parameters required by model
    :param time_temp: Optional argument to set your desired time array (in source frame days) to evaluate the model on.
    :param n: index of progenitor density profile, 1.5 (default) or 3.0
    :param RW: If True, use the simplified Rabinak & Waxman formulation (off by default)
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """

    n = kwargs.get('n', 1.5)
    v_shock = v_shock * 1e5
    RW = kwargs.get('RW', False)
    radius = radius * 1e13
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_temp = kwargs.get('time_temp', np.linspace(0.01, 60, 100))
    time_obs = time
    output = _shockcooling_sapirandwaxman(time_temp, v_shock, m_env, f_rho_m, radius, kappa, nn=n, RW=RW)
    if kwargs['output_format'] == 'namedtuple':
        return output
    elif kwargs['output_format'] == 'flux_density':
        time = time_obs
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=output.t_photosphere)
        rad_func = interp1d(time_temp, y=output.r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)

        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = sed.blackbody_to_flux_density(temperature=output.t_photosphere,
                                             r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract')
def _csm_shock_breakout(time, csm_mass, v_min, beta, kappa, shell_radius, shell_width_ratio, **kwargs):
    """
    Dense CSM shock breakout and cooling model From Margalit 2022

    :param time: time in days
    :param csm_mass: mass of CSM shell in g
    :param v_min: minimum velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param kappa: opacity in cm^2/g
    :param shell_radius: radius of shell in 10^14 cm
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :return: namedtuple with lbol, r_photosphere, and temperature
    """
    v0 = v_min * 1e5
    e0 = 0.5 * csm_mass * v0**2
    velocity = v0/beta
    shell_radius *= 1e14
    shell_width = shell_width_ratio * shell_radius
    tdyn = shell_radius / velocity
    tshell = shell_width / velocity
    time = time * cc.day_to_s

    tda = (3 * kappa * csm_mass / (4 * np.pi * cc.speed_of_light * velocity)) ** 0.5

    term1 = ((tdyn + tshell + time) ** 3 - (tdyn + beta * time) ** 3) ** (2 / 3)
    term2 = ((tdyn + tshell) ** 3 - tdyn ** 3) ** (1 / 3)
    term3 = (1 + (1 - beta) * time / tshell) ** (
                -3 * (tdyn / tda) ** 2 * ((1 - beta - beta * tshell / tdyn) ** 2) / (1 - beta) ** 3)
    term4 = np.exp(-time * ((1 - beta ** 3) * time + (2 - 4 * beta * (beta + 1)) * tshell + 6 * (1 - beta ** 2) * tdyn) / (
                2 * (1 - beta) ** 2 * tda ** 2))

    lbol = e0 * term1 / (tda ** 2 * (tshell + (1 - beta) * time) ** 2) * term2 * term3 * term4

    volume = 4./3. * np.pi * velocity**3 * ((tdyn + tshell + time)**3 - (tdyn + beta*time)**3)
    radius = velocity * (tdyn + tshell + time)
    rphotosphere = radius - 2*volume/(3 * kappa * csm_mass)
    teff = (lbol / (4 * np.pi * rphotosphere ** 2 * cc.sigma_sb)) ** 0.25
    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature', 'time_temp',
                                   'tdyn', 'tshell', 'e0', 'tda', 'velocity'])
    output.lbol = lbol
    output.r_photosphere = rphotosphere
    output.temperature = teff
    output.tdyn = tdyn
    output.tshell = tshell
    output.e0 = e0
    output.tda = tda
    output.velocity = velocity
    output.time_temp = time
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract')
def csm_shock_breakout_bolometric(time, csm_mass, v_min, beta, kappa, shell_radius, shell_width_ratio, **kwargs):
    """
    Dense CSM shock breakout and cooling model From Margalit 2022

    :param time: time in days in source frame
    :param csm_mass: mass of CSM shell in solar masses
    :param v_min: minimum velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param kappa: opacity in cm^2/g
    :param shell_radius: radius of shell in 10^14 cm
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: Additional parameters required by model
    :return: bolometric luminosity
    """
    csm_mass = csm_mass * cc.solar_mass
    time_temp = np.linspace(1e-2, 200, 300)  # days
    outputs = _csm_shock_breakout(time_temp, v_min=v_min, beta=beta,
                                  kappa=kappa, csm_mass=csm_mass, shell_radius=shell_radius,
                                  shell_width_ratio=shell_width_ratio, **kwargs)
    func = interp1d(time_temp, outputs.lbol, fill_value='extrapolate')
    return func(time)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...933..238M/abstract')
def csm_shock_breakout(time, redshift, csm_mass, v_min, beta, kappa, shell_radius, shell_width_ratio, **kwargs):
    """
    Dense CSM shock breakout and cooling model From Margalit 2022

    :param time: time in days in observer frame
    :param redshift: redshift
    :param csm_mass: mass of CSM shell in solar masses
    :param v_min: minimum velocity in km/s
    :param beta: velocity ratio in c (beta < 1)
    :param kappa: opacity in cm^2/g
    :param shell_radius: radius of shell in 10^14 cm
    :param shell_width_ratio: shell width ratio (deltaR/R0)
    :param kwargs: Additional parameters required by model
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    csm_mass = csm_mass * cc.solar_mass
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_temp = np.linspace(1e-2, 60, 300) #days
    time_obs = time
    outputs = _csm_shock_breakout(time_temp, v_min=v_min, beta=beta,
                                  kappa=kappa, csm_mass=csm_mass, shell_radius=shell_radius,
                                  shell_width_ratio=shell_width_ratio, **kwargs)
    if kwargs['output_format'] == 'namedtuple':
        return outputs
    elif kwargs['output_format'] == 'flux_density':
        time = time_obs
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=outputs.temperature)
        rad_func = interp1d(time_temp, y=outputs.r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = sed.blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)

        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = sed.blackbody_to_flux_density(temperature=outputs.temperature,
                                         r_photosphere=outputs.r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                           lambdas=lambda_observer_frame,
                                                                           spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                          spectra=spectra, lambda_array=lambda_observer_frame,
                                                          **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def _shock_cooling(time, mass, radius, energy, **kwargs):
    """
    :param time: time in source frame in seconds
    :param mass: mass of extended material in solar masses
    :param radius: radius of extended material in cm
    :param energy: energy of extended material in ergs
    :param kwargs: extra parameters to change physics
    :param nn: density power law slope
    :param delta: inner density power law slope
    :return: namedtuple with lbol, r_photosphere, and temperature
    """
    nn = kwargs.get('nn',10)
    delta = kwargs.get('delta',1.1)
    kk_pow = (nn - 3) * (3 - delta) / (4 * np.pi * (nn - delta))
    kappa = 0.2
    mass = mass * cc.solar_mass
    vt = (((nn - 5) * (5 - delta) / ((nn - 3) * (3 - delta))) * (2 * energy / mass))**0.5
    td = ((3 * kappa * kk_pow * mass) / ((nn - 1) * vt * cc.speed_of_light))**0.5

    prefactor = np.pi * (nn - 1) / (3 * (nn - 5)) * cc.speed_of_light * radius * vt**2 / kappa
    lbol_pre_td = prefactor * np.power(td / time, 4 / (nn - 2))
    lbol_post_td = prefactor * np.exp(-0.5 * (time * time / td / td - 1))
    lbol = np.zeros(len(time))
    lbol[time < td] = lbol_pre_td[time < td]
    lbol[time >= td] = lbol_post_td[time >= td]

    tph = np.sqrt(3 * kappa * kk_pow * mass / (2 * (nn - 1) * vt * vt))
    r_photosphere_pre_td = np.power(tph / time, 2 / (nn - 1)) * vt * time
    r_photosphere_post_td = (np.power((delta - 1) / (nn - 1) * ((time / td) ** 2 - 1) + 1, -1 / (delta + 1)) * vt * time)
    r_photosphere = r_photosphere_pre_td + r_photosphere_post_td

    sigmaT4 = lbol / (4 * np.pi * r_photosphere**2)
    temperature = np.power(sigmaT4 / cc.sigma_sb, 0.25)

    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature', 'td'])
    output.lbol = lbol
    output.r_photosphere = r_photosphere
    output.temperature = temperature
    output.td = td
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract')
def _shocked_cocoon_nicholl(time, kappa, mejecta, vejecta, cos_theta_cocoon, shocked_fraction, nn, tshock):
    """
    Shocked cocoon model from Nicholl et al. 2021

    :param time: time in source frame in days
    :param kappa: opacity
    :param mejecta: ejecta in solar masses
    :param vejecta: ejecta velocity in units of c (speed of light)
    :param cos_theta_cocoon: cosine of the cocoon opening angle
    :param shocked_fraction: fraction of the ejecta that is shocked
    :param nn: ejecta power law density profile
    :param tshock: time of shock in source frame in seconds
    :return: luminosity
    """
    ckm = 3e10 / 1e5
    vejecta = vejecta * ckm
    diffusion_constant = cc.solar_mass / (4 * np.pi * cc.speed_of_light * cc.km_cgs)
    num = cc.speed_of_light / cc.km_cgs
    rshock = cc.speed_of_light * tshock
    mshocked = shocked_fraction * mejecta
    theta = np.arccos(cos_theta_cocoon)
    taudiff = np.sqrt(diffusion_constant * kappa * mshocked / vejecta) / cc.day_to_s

    tthin = (num / vejecta) ** 0.5 * taudiff

    l0 = (theta **2 / 2)**(1. / 3.) * (mshocked * cc.solar_mass *
                                       vejecta * cc.km_cgs * rshock / (taudiff * cc.day_to_s)**2)

    lbol = l0 * (time / taudiff)**-(4/(nn+2)) * (1 + np.tanh(tthin-time))/2.
    output = namedtuple('output', ['lbol', 'tthin', 'taudiff','mshocked'])
    output.lbol = lbol
    output.tthin = tthin
    output.taudiff = taudiff
    output.mshocked = mshocked
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def shock_cooling_bolometric(time, log10_mass, log10_radius, log10_energy, **kwargs):
    """
    :param time: time in source frame in seconds
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param kwargs: extra parameters to change physics
    :param nn: density power law slope
    :param delta: inner density power law slope
    :return: bolometric_luminosity
    """
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy
    output = _shock_cooling(time, mass=mass, radius=radius, energy=energy, **kwargs)
    return output.lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def shock_cooling(time, redshift, log10_mass, log10_radius, log10_energy, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param log10_mass: log10 mass of extended material in solar masses
    :param log10_radius: log10 radius of extended material in cm
    :param log10_energy: log10 energy of extended material in ergs
    :param kwargs: extra parameters to change physics and other settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param nn: density power law slope
    :param delta: inner density power law slope
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = _shock_cooling(time*cc.day_to_s, mass=mass, radius=radius, energy=energy, **kwargs)
        flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                             dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        time_temp = np.linspace(1e-2, 60, 100)
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))

        time_observer_frame = time_temp
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = _shock_cooling(time=time * cc.day_to_s, mass=mass, radius=radius, energy=energy, **kwargs)
        fmjy = sed.blackbody_to_flux_density(temperature=output.temperature,
                                             r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

def _c_j(p):
    """
    :param p: electron power law slope
    :return: prefactor for emissivity
    """
    term1 = (special.gamma((p+5.0)/4.0)/special.gamma((p+7.0)/4.0))
    term2 = special.gamma((3.0*p+19.0)/12.0)
    term3 = special.gamma((3.0*p-1.0)/12.0)*((p-2.0)/(p+1.0))
    term4 = 3.0**((2.0*p-1.0)/2.0)
    term5 = 2.0**(-(7.0-p)/2.0)*np.pi**(-0.5)
    return term1*term2*term3*term4*term5

def _c_alpha(p):
    """
    :param p: electron power law slope
    :return: prefactor for absorption coefficient
    """
    term1 = (special.gamma((p+6.0)/4.0)/special.gamma((p+8.0)/4.0))
    term2 = special.gamma((3.0*p+2.0)/12.0)
    term3 = special.gamma((3.0*p+22.0)/12.0)*(p-2.0)*3.0**((2.0*p-5.0)/2.0)
    term4 = 2.0**(p/2.0)*np.pi**(3.0/2.0)
    return term1*term2*term3*term4


def _g_theta(theta,p):
    """
    :param theta: dimensionless electron temperature
    :param p: electron power law slope
    :return: correction term for power law electron distribution
    """
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    gtheta = ((p-1.0)*(1e0+aa*theta)/((p-1.0)*gamma_m - p+2.0))*(gamma_m/(3.0*theta))**(p-1.0)
    return gtheta

def _low_freq_jpl_correction(x,theta,p):
    """
    :param x: dimensionless frequency
    :param theta: dimensionless electron temperature
    :param p: electron power law slope
    :return: low-frequency correction to power-law emissivity
    """
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    # synchrotron constant in x<<x_m limit
    Cj_low = -np.pi**1.5*(p-2.0)/( 2.0**(1.0/3.0)*3.0**(1.0/6.0)*(3.0*p-1.0)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Cj_low/_c_j(p))*(gamma_m/(3.0*theta))**(-(3.0*p-1.0)/3.0)*x**((3.0*p-1.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    val = (1e0 + corr**(-s))**(-1.0/s)
    return val

def _low_freq_apl_correction(x,theta,p):
    """
    :param x: dimensionless frequency
    :param theta: dimensionless electron temperature
    :param p: electron power law slope
    :return: low-frequency correction to power-law absorption coefficient
    """
    aa = (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
    gamma_m = 1e0 + aa * theta
    # synchrotron constant in x<<x_m limit
    Calpha_low = -2.0**(8.0/3.0)*np.pi**(7.0/2.0)*(p+2.0)*(p-2.0)/( 3.0**(19.0/6.0)*(3.0*p+2)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Calpha_low/_c_alpha(p))*(gamma_m/(3.0*theta))**(-(3.0*p+2.0)/3.0)*x**((3.0*p+2.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    val = ( 1e0 + corr**(-s) )**(-1.0/s)
    return val

def _emissivity_pl(x, nism, bfield, theta, xi, p, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param z_cool: normalised cooling lorentz factor
    :return: synchrotron emissivity of power-law electrons
    """
    val = _c_j(p)*(cc.qe**3/(cc.electron_mass*cc.speed_of_light**2))*xi*nism*bfield*_g_theta(theta=theta,p=p)*x**(-(p-1.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= _low_freq_jpl_correction(x=x,theta=theta,p=p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    emissivity_pl = val
    return emissivity_pl

def _emissivity_thermal(x, nism, bfield, theta, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param z_cool: normalised cooling lorentz factor
    :return: synchrotron emissivity of thermal electrons
    """
    ff = 2.0*theta**2/special.kn(2,1.0/theta)
    ix = 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))
    val = (3.0**0.5/(8.0*np.pi))*(cc.qe**3/(cc.electron_mass*cc.speed_of_light**2))*ff*nism*bfield*x*ix
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum(1e0, (z0/z_cool)**(-1))
    return val

def _alphanu_th(x, nism, bfield, theta, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param z_cool: normalised cooling lorentz factor
    :return: Synchrotron absorption coeff of thermal electrons
    """
    ff = 2.0 * theta ** 2 / special.kn(2, 1.0 / theta)
    ix = 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))
    val = (np.pi*3.0**(-3.0/2.0))*cc.qe*(nism/(theta**5*bfield))*ff*x**(-1.0)*ix
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    return val

def _alphanu_pl(x, nism, bfield, theta, xi, p, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param z_cool: normalised cooling lorentz factor
    :return: Synchrotron absorption coeff of power-law electrons
    """
    val = _c_alpha(p)*cc.qe*(xi*nism/(theta**5*bfield))*_g_theta(theta,p=p)*x**(-(p+4.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= _low_freq_apl_correction(x,theta,p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, (z0/z_cool)**(-1) )
    return val

def _tau_nu(x, nism, radius, bfield, theta, xi, p, z_cool):
    """
    :param x: dimensionless frequency
    :param nism: electron number density in emitting region (cm^-3)
    :param radius: characteristic size of the emitting region (in cm)
    :param bfield: magnetic field strength in Gauss
    :param theta: dimensionless electron temperature
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param z_cool: normalised cooling lorentz factor
    :return: Total (thermal+non-thermal) synchrotron optical depth
    """
    alphanu_pl = _alphanu_pl(x=x,nism=nism,bfield=bfield,theta=theta,xi=xi,p=p,z_cool=z_cool)
    alphanu_thermal = _alphanu_th(x=x, nism=nism, bfield=bfield,theta=theta,z_cool=z_cool)
    val = radius*(alphanu_thermal + alphanu_pl)
    return val

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def _shocked_cocoon(time, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa):
    """
    :param time: source frame time in days
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in c
    :param eta: slope for ejecta density profile
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass shocked
    :param cos_theta_cocoon: cocoon opening angle
    :param kappa: opacity
    :return: namedtuple with lbol, r_photosphere, and temperature
    """
    c_kms = cc.speed_of_light / cc.km_cgs
    vej = vej * c_kms
    diff_const = cc.solar_mass / (4*np.pi * cc.speed_of_light * cc.km_cgs)
    rshock = tshock * cc.speed_of_light
    shocked_mass = mej * shocked_fraction
    theta = np.arccos(cos_theta_cocoon)
    tau_diff = np.sqrt(diff_const * kappa * shocked_mass / vej) / cc.day_to_s

    t_thin = (c_kms / vej) ** 0.5 * tau_diff

    l0 = (theta ** 2 / 2) ** (1 / 3) * (shocked_mass * cc.solar_mass *
                                        vej * cc.km_cgs * rshock / (tau_diff * cc.day_to_s) ** 2)

    lbol = l0 * (time/tau_diff)**(-4/(eta+2)) * (1 + np.tanh(t_thin - time))/2

    v_photosphere = vej * (time / t_thin) ** (-2. / (eta + 3))
    r_photosphere = cc.km_cgs * cc.day_to_s * v_photosphere * time
    temperature = (lbol / (4.0 * np.pi * cc.sigma_sb * r_photosphere**2))**0.25

    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature'])
    output.lbol = lbol
    output.r_photosphere = r_photosphere
    output.temperature = temperature
    return output
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon_bolometric(time, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa, **kwargs):
    """
    :param time: source frame time in days
    :param mej: ejecta mass in solar masses
    :param vej: ejecta mass in km/s
    :param eta: slope for ejecta density profile
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass shocked
    :param cos_theta_cocoon: cocoon opening angle
    :param kappa: opacity
    :param kwargs: None
    :return: bolometric_luminosity
    """
    output = _shocked_cocoon(time, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa)
    return output.lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon(time, redshift, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in c
    :param eta: slope for ejecta density profile
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass shocked
    :param cos_theta_cocoon: cocoon opening angle
    :param kappa: opacity
    :param kwargs: Extra parameters used by function
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = _shocked_cocoon(time=time, mej=mej, vej=vej, eta=eta,
                                 tshock=tshock, shocked_fraction=shocked_fraction,
                                 cos_theta_cocoon=cos_theta_cocoon, kappa=kappa)
        flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                                     dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 60000, 100))
        time_temp = np.linspace(1e-2, 100, 100)
        time_observer_frame = time_temp
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        output = _shocked_cocoon(time=time, mej=mej, vej=vej, eta=eta,
                                 tshock=tshock, shocked_fraction=shocked_fraction,
                                 cos_theta_cocoon=cos_theta_cocoon, kappa=kappa)
        fmjy = sed.blackbody_to_flux_density(temperature=output.temperature,
                                         r_photosphere=output.r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...928..122M/abstract')
def csm_truncation_shock():
    """
    Multi-zone version of Margalit 2022 model for CSM shock breakout and cooling one zone model is implemented as
    csm_shock_breakout

    :return:
    """
    raise NotImplementedError("This model is not yet implemented.")
