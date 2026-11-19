from redback.constants import *
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar, _evolving_gw_and_em_magnetar
import numpy as np
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from collections import namedtuple
import astropy.units as uu # noqa
import astropy.constants as cc # noqa
from redback.utils import calc_kcorrected_properties, interpolated_barnes_and_kasen_thermalisation_efficiency, \
    electron_fraction_from_kappa, citation_wrapper, lambda_to_nu, velocity_from_lorentz_factor, get_optimal_time_array
from redback.sed import blackbody_to_flux_density, get_correct_output_format_from_spectra, \
    flux_density_to_spectrum, blackbody_to_spectrum


DynamicsOutput = namedtuple(
    'DynamicsOutput',
    ['lorentz_factor', 'bolometric_luminosity', 'comoving_temperature',
     'radius', 'doppler_factor', 'tau', 'time', 'kinetic_energy',
     'erad_total', 'thermalisation_efficiency', 'r_photosphere'],
)

def _ejecta_dynamics_and_interaction(time, mej, beta, ejecta_radius, kappa, n_ism,
                                     magnetar_luminosity, pair_cascade_switch, use_gamma_ray_opacity, **kwargs):
    """
    :param time: time in source frame
    :param mej: ejecta mass in solar masses
    :param beta: initial ejecta velocity in c
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param magnetar_luminosity: evaluated magnetar luminosity in source frame
    :param pair_cascade_switch: whether to account for pair cascade losses
    :param use_gamma_ray_opacity: whether to use gamma ray opacity to calculate thermalisation efficiency
    :param kwargs: Additional parameters
    :param use_r_process: whether to use r-process
    :param kappa_gamma: Gamma-ray opacity for leakage efficiency, only used if use_gamma_ray_opacity = True
    :param thermalisation_efficiency: magnetar thermalisation efficiency only used if use_gamma_ray_opacity = False
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param f_nickel: nickel fraction (if not using r_process)
    :return: named tuple with 'lorentz_factor', 'bolometric_luminosity', 'comoving_temperature',
            'radius', 'doppler_factor', 'tau', 'time', 'kinetic_energy',
            'erad_total', 'thermalisation_efficiency'
    """
    time = np.asarray(time, dtype=float)
    mag_lum = np.asarray(magnetar_luminosity, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing one-dimensional array")
    if mag_lum.shape != time.shape:
        raise ValueError("magnetar_luminosity must have the same shape as time")
    if not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one")

    use_r_process = kwargs.get('use_r_process', True)
    ejecta_albedo = kwargs.get('ejecta_albedo', 0.5)
    pair_cascade_fraction = kwargs.get('pair_cascade_fraction', 0.05)
    mass = mej * solar_mass
    initial_internal_energy = kwargs.get(
        'initial_internal_energy', 0.5 * beta ** 2 * mass * speed_of_light ** 2)
    initial_volume = (4 / 3) * np.pi * ejecta_radius ** 3
    initial_gamma = 1 / np.sqrt(1 - beta ** 2)

    requested_method = kwargs.get('dynamics_method')
    method = requested_method or 'RK45'
    rtol = kwargs.get('dynamics_rtol', 1e-6)
    atol = kwargs.get('dynamics_atol', None)
    if atol is None:
        atol = np.array([
            1e-10,
            max(1.0, ejecta_radius * rtol * 1e-2),
            max(1.0, initial_volume * rtol * 1e-2),
            max(1.0, initial_internal_energy * rtol * 1e-2),
        ])

    t0_comoving = 1.3
    tsigma_comoving = 0.11
    ni56_lum = 6.45e43
    co56_lum = 1.45e43
    ni56_life = 8.8 * day_to_s
    co56_life = 111.3 * day_to_s

    def _state_quantities(t, state):
        gamma, radius, volume, internal_energy = state
        gamma = max(gamma, 1.0 + np.finfo(float).eps)
        radius = max(radius, np.finfo(float).tiny)
        volume = max(volume, np.finfo(float).tiny)
        internal_energy = max(internal_energy, np.finfo(float).tiny)
        current_beta = np.sqrt(1 - gamma ** -2)
        doppler = 1 / (gamma * (1 - current_beta))
        comoving_time = doppler * t
        rad_denom = (1 / 2) - (1 / 3.141592654) * np.arctan((comoving_time - t0_comoving) / tsigma_comoving)
        if use_r_process:
            radioactive_luminosity = 4e49 * (mass / (2e33) * 1e2) * rad_denom ** 1.3
        else:
            nickel_mass = kwargs.get('f_nickel', 0) * mej
            radioactive_luminosity = nickel_mass * (
                ni56_lum * np.exp(-comoving_time / ni56_life)
                + co56_lum * np.exp(-comoving_time / co56_life))
        optical_depth = kappa * (mass / volume) * (radius / gamma)
        diffusion_factor = 1.0 if optical_depth <= 1 else optical_depth
        emitted_luminosity = internal_energy * speed_of_light / (diffusion_factor * radius / gamma)
        if optical_depth <= 1:
            temperature = (internal_energy / (radiation_constant * volume)) ** 0.25
        else:
            temperature = (internal_energy / (radiation_constant * volume * optical_depth)) ** 0.25
        velocity = velocity_from_lorentz_factor(gamma)
        if use_gamma_ray_opacity:
            kappa_gamma = kwargs["kappa_gamma"]
            prefactor = 3 * kappa_gamma * mass / (4 * np.pi * velocity ** 2)
            thermalisation_efficiency = 1 - np.exp(-prefactor * t ** -2)
        else:
            thermalisation_efficiency = kwargs["thermalisation_efficiency"]
        return (gamma, radius, volume, internal_energy, current_beta, doppler,
                radioactive_luminosity, optical_depth, emitted_luminosity,
                temperature, thermalisation_efficiency)

    def _derivative(t, state):
        (gamma, radius, volume, internal_energy, current_beta, doppler,
         radioactive_luminosity, _, emitted_luminosity, _,
         thermalisation_efficiency) = _state_quantities(t, state)
        magnetar_power = np.interp(t, time, mag_lum)
        swept_mass = (4 / 3) * np.pi * radius ** 3 * n_ism * proton_mass
        pressure = internal_energy / (3 * volume)
        comoving_volume_rate = 4 * np.pi * radius ** 2 * current_beta * speed_of_light
        radius_rate = current_beta * speed_of_light / (1 - current_beta)
        swept_mass_rate = 4 * np.pi * radius ** 2 * n_ism * proton_mass * radius_rate
        comoving_internal_energy_rate = (
            thermalisation_efficiency * doppler ** -2 * magnetar_power
            + radioactive_luminosity - emitted_luminosity
            - pressure * comoving_volume_rate)
        energy_rate = (
            thermalisation_efficiency * magnetar_power
            + doppler ** 2 * (radioactive_luminosity - emitted_luminosity))
        gamma_rate = (
            energy_rate - gamma * doppler * comoving_internal_energy_rate
            - (gamma ** 2 - 1) * speed_of_light ** 2 * swept_mass_rate
        ) / (mass * speed_of_light ** 2 + internal_energy
             + 2 * gamma * swept_mass * speed_of_light ** 2)
        return [gamma_rate, radius_rate, comoving_volume_rate * doppler,
                comoving_internal_energy_rate * doppler]

    initial_state = [initial_gamma, ejecta_radius, initial_volume, initial_internal_energy]

    def _solve(selected_method):
        return solve_ivp(
            _derivative, (time[0], time[-1]), initial_state,
            method=selected_method, t_eval=time, rtol=rtol, atol=atol,
        )

    def _valid_solution(candidate):
        return (
            candidate.success
            and candidate.y.ndim == 2
            and candidate.y.shape[1] == len(time)
            and np.all(np.isfinite(candidate.y))
            and np.all(candidate.y[0] >= 1.0)
            and np.all(candidate.y[1:3] > 0.0)
            and np.all(candidate.y[3] >= 0.0)
        )

    solution = _solve(method)
    if requested_method is None and not _valid_solution(solution):
        # Explicit Runge-Kutta methods can step slightly through E_int = 0 for
        # weak engines at late times. LSODA handles that stiff boundary well.
        solution = _solve('LSODA')

    if not solution.success or solution.y.shape[1] != len(time):
        raise RuntimeError(f"Ejecta dynamics integration failed: {solution.message}")
    if not _valid_solution(solution):
        raise RuntimeError("Ejecta dynamics integration produced an invalid physical state")

    lorentz_factor, radius, _, _ = solution.y
    quantities = [_state_quantities(t, solution.y[:, index]) for index, t in enumerate(time)]
    doppler_factor = np.array([value[5] for value in quantities])
    tau = np.array([value[7] for value in quantities])
    comoving_emitted_luminosity = np.array([value[8] for value in quantities])
    comoving_temperature = np.array([value[9] for value in quantities])
    thermalisation_efficiency = np.array([value[10] for value in quantities])
    bolometric_luminosity = comoving_emitted_luminosity * doppler_factor ** 2

    if pair_cascade_switch:
        tlife_t = (0.6/(1 - ejecta_albedo))*(pair_cascade_fraction/0.1)**0.5 * (mag_lum/1.0e45)**0.5 \
                  * (velocity_from_lorentz_factor(lorentz_factor)/(0.3*speed_of_light))**(0.5) \
                  * (time/day_to_s)**(-0.5)
        bolometric_luminosity = bolometric_luminosity / (1.0 + tlife_t)
        comoving_temperature = (
            bolometric_luminosity / (4.0 * np.pi * radius ** 2 * sigma_sb)) ** 0.25

    return DynamicsOutput(
        lorentz_factor=lorentz_factor,
        bolometric_luminosity=bolometric_luminosity,
        comoving_temperature=comoving_temperature,
        radius=radius,
        doppler_factor=doppler_factor,
        tau=tau,
        time=time,
        kinetic_energy=(lorentz_factor - 1) * mass * speed_of_light ** 2,
        erad_total=np.trapezoid(bolometric_luminosity, x=time),
        thermalisation_efficiency=thermalisation_efficiency,
        r_photosphere=radius,
    )

def _comoving_blackbody_to_flux_density(dl, frequency, radius, temperature, doppler_factor):
    """
    :param dl: luminosity distance in cm
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number
    :param radius: ejecta radius in cm
    :param temperature: comoving temperature in K
    :param doppler_factor: doppler_factor
    :return: flux_density
    """
    ## adding units back in to ensure dimensions are correct
    frequency = frequency * uu.Hz
    radius = radius * uu.cm
    dl = dl * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 2 * np.pi * planck * frequency ** 3 * radius ** 2
    denom = dl ** 2 * speed_of_light ** 2 * doppler_factor ** 2
    frac = 1. / (np.exp((planck * frequency) / (boltzmann_constant * temperature * doppler_factor)) - 1)
    flux_density = num / denom * frac
    return flux_density


def _comoving_blackbody_to_luminosity(frequency, radius, temperature, doppler_factor):
    """
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number
    :param radius: ejecta radius in cm
    :param temperature: comoving temperature in K
    :param doppler_factor: doppler_factor
    :return: luminosity
    """
    ## adding units back in to ensure dimensions are correct
    frequency = frequency * uu.Hz
    radius = radius * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 8 * np.pi ** 2 * planck * frequency ** 4 * radius ** 2
    denom = speed_of_light ** 2 * doppler_factor ** 2
    frac = 1. / (np.exp((planck * frequency) / (boltzmann_constant * temperature * doppler_factor)) - 1)
    luminosity = num / denom * frac
    return luminosity

def _processing_other_formats(dl, output, redshift, time_obs, time_temp, **kwargs):
    """
    Function to process the output of the dynamics function into other formats

    :param dl: luminosity distance in cm
    :param output: dynamics output
    :param redshift: source redshift
    :param time_obs: observed time array in days
    :param time_temp: temporary time array in seconds where output is evaluated
    :param kwargs: extra arguments
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :return: returns the correct output format
    """
    lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
    time_observer_frame = time_temp * (1. + redshift)
    frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                 redshift=redshift, time=time_observer_frame)
    if kwargs['use_relativistic_blackbody']:
        fmjy = _comoving_blackbody_to_flux_density(dl=dl, frequency=frequency[:, None], radius=output.radius,
                                                   temperature=output.comoving_temperature,
                                                   doppler_factor=output.doppler_factor)
        fmjy = fmjy
        fmjy = fmjy.T
        spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
    else:
        spectra = blackbody_to_spectrum(
            temperature=output.temperature,
            r_photosphere=output.r_photosphere,
            frequency=frequency[:, None],
            dl=dl,
            redshift=redshift,
            lambda_observer_frame=lambda_observer_frame
        )
    if kwargs['output_format'] == 'spectra':
        return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                      lambdas=lambda_observer_frame,
                                                                      spectra=spectra)
    else:
        return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                      spectra=spectra, lambda_array=lambda_observer_frame,
                                                      **kwargs)

def _process_flux_density(dl, output, redshift, time, time_temp, **kwargs):
    """
    Function to process the output of the dynamics function into flux density

    :param dl: luminosity distance in cm
    :param output: dynamics output
    :param redshift: source redshift
    :param time_obs: observed time array in days
    :param time_temp: temporary time array in seconds where output is evaluated
    :param kwargs: extra arguments
    :return: returns the correct output format
    """
    frequency = kwargs['frequency']
    time = time * day_to_s
    # convert to source frame time and frequency
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    if kwargs['use_relativistic_blackbody']:
        temp_func = interp1d(time_temp, y=output.comoving_temperature)
        rad_func = interp1d(time_temp, y=output.radius)
        d_func = interp1d(time_temp, y=output.doppler_factor)
        temp = temp_func(time)
        rad = rad_func(time)
        df = d_func(time)
        flux_density = _comoving_blackbody_to_flux_density(dl=dl, frequency=frequency, radius=rad, temperature=temp,
                                                       doppler_factor=df)
        flux_density = flux_density * (1 + redshift)
    else:
        temp_func = interp1d(time_temp, y=output.temperature)
        rad_func = interp1d(time_temp, y=output.r_photosphere)
        temp = temp_func(time)
        rad = rad_func(time)
        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=rad, frequency=frequency, dl=dl)
    return flux_density.to(uu.mJy).value * (1 + redshift)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...776L..40Y/abstract')
def basic_mergernova(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, p0, bp,
                     mass_ns, theta_pb, thermalisation_efficiency, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param p0: initial spin period in milliseconds
    :param bp: polar magnetic field strength in units of 10^14 Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes in radians
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is False
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', False)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = get_optimal_time_array(1e-4, 1e8, 500) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = basic_magnetar(time=time_temp, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              thermalisation_efficiency=thermalisation_efficiency,
                                              pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=False, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_mergernova(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
               thermalisation_efficiency, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = get_optimal_time_array(1e-4, 1e8, 500) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              thermalisation_efficiency=thermalisation_efficiency,
                                              pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=False, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_mergernova_thermalisation(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
               kappa_gamma, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = get_optimal_time_array(1e-4, 1e8, 500) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_mergernova_evolution(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, logbint,
                                 logbext, p0, chi0, radius, logmoi, kappa_gamma, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param logbint: log10 internal magnetic field in G
    :param logbext: log10 external magnetic field in G
    :param p0: spin period in s
    :param chi0: initial inclination angle
    :param radius: radius of NS in KM
    :param logmoi: log10 moment of inertia of NS
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    kwargs['use_relativistic_blackbody'] = True

    time_temp = get_optimal_time_array(1e-4, 1e8, 500) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    bint = 10 ** logbint
    bext = 10 ** logbext
    radius = radius * km_cgs
    moi = 10 ** logmoi
    output = _evolving_gw_and_em_magnetar(time=time_temp, bint=bint, bext=bext, p0=p0, chi0=chi0, radius=radius, moi=moi)
    magnetar_luminosity = output.Edot_d
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              kappa_gamma=kappa_gamma, pair_cascade_switch=pair_cascade_switch,
                                              use_gamma_ray_opacity=True, **kwargs)
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

def _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                          **kwargs):
    """
    :param time: time in source frame
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :return: luminosity
    """
    time_temp = get_optimal_time_array(1e-4, 1e8, 500) #in source frame
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                              beta=beta, ejecta_radius=ejecta_radius,
                                              kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                              thermalisation_efficiency=thermalisation_efficiency,
                                              pair_cascade_switch=False, use_gamma_ray_opacity=False)
    temp_func = interp1d(time_temp, y=output.comoving_temperature)
    rad_func = interp1d(time_temp, y=output.radius)
    d_func = interp1d(time_temp, y=output.doppler_factor)
    tau_func = interp1d(time_temp, y=output.tau)
    temp = temp_func(time)
    rad = rad_func(time)
    df = d_func(time)
    optical_depth = tau_func(time)
    frequency = kwargs['frequency']
    trapped_ejecta_lum = _comoving_blackbody_to_luminosity(frequency=frequency, radius=rad,
                                                          temperature=temp, doppler_factor=df)
    lsd = magnetar_only(time, l0=l0, tau=tau_sd, nn=nn)
    lum = np.exp(-optical_depth) * lsd + trapped_ejecta_lum
    return lum


def _trapped_magnetar_flux(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                           thermalisation_efficiency, photon_index, **kwargs):
    """
    :param time: time in observer frame in seconds
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output flux density or AB magnitude
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' used to calculate k correction and convert from luminosity to flux
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: integrated flux
    """
    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    kwargs['frequency'] = frequency
    lum = _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                                **kwargs)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    kcorr = (1. + redshift) ** (photon_index - 2)
    flux = lum / (4 * np.pi * dl ** 2 * kcorr)
    return flux

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...835....7S/abstract')
def trapped_magnetar(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn, thermalisation_efficiency,
                     **kwargs):
    """
    :param time: time in source frame or observer frame depending on output format in seconds
    :param redshift: redshift - not used if evaluating luminosity
    :param mej: ejecta mass in solar units
    :param beta: initial ejecta velocity
    :param ejecta_radius: initial ejecta radius
    :param kappa: opacity
    :param n_ism: ism number density
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: 'output_format' - whether to output luminosity or flux
    :param kwargs: 'frequency' in Hertz to evaluate the mergernova emission - use a typical X-ray frequency
    :param kwargs: 'photon_index' only used if calculating the flux lightcurve
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: luminosity or integrated flux
    """
    if kwargs['output_format'] == 'luminosity':
        return _trapped_magnetar_lum(time, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                     thermalisation_efficiency, **kwargs)
    elif kwargs['output_format'] == 'flux':
        return _trapped_magnetar_flux(time, redshift, mej, beta, ejecta_radius, kappa, n_ism, l0, tau_sd, nn,
                                      thermalisation_efficiency, **kwargs)

def _general_metzger_magnetar_driven_kilonova_model(time, mej, vej, beta, kappa, magnetar_luminosity,
                                                    use_gamma_ray_opacity, **kwargs):
    """
    :param time: time array to evaluate model on in source frame in seconds
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa: opacity
    :param magnetar_luminosity: evaluated magnetar luminosity in source frame
    :param pair_cascade_switch: whether to account for pair cascade losses
    :param use_gamma_ray_opacity: whether to use gamma ray opacity to calculate thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param kappa_gamma: Gamma-ray opacity for leakage efficiency, only used if use_gamma_ray_opacity = True
    :param thermalisation_efficiency: magnetar thermalisation efficiency only used if use_gamma_ray_opacity = False
    :param neutron_precursor_switch: whether to have neutron precursor emission, default True
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer.
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :return: named tuple with 'lorentz_factor', 'bolometric_luminosity', 'temperature',
                'r_photosphere', 'kinetic_energy','erad_total', 'thermalisation_efficiency'
    """
    pair_cascade_switch = kwargs.get('pair_cascade_switch', True)
    ejecta_albedo = kwargs.get('ejecta_albedo', 0.5)
    pair_cascade_fraction = kwargs.get('pair_cascade_fraction', 0.01)
    neutron_precursor_switch = kwargs.get('neutron_precursor_switch', True)
    magnetar_heating = kwargs.get('magnetar_heating', 'first_layer')
    vmax = kwargs.get('vmax', 0.7)

    tdays = time/day_to_s
    time_len = len(time)
    mass_len = 200

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    electron_fraction = electron_fraction_from_kappa(kappa)
    t0 = 1.3 #seconds
    sig = 0.11  #seconds
    tau_neutron = 900  #seconds

    # convert to astrophysical units
    m0 = mej * solar_mass
    v0 = vej * speed_of_light
    ek_tot_0 = 0.5 * m0 * v0 ** 2

    # set up mass and velocity layers and normalize them
    vmin = vej
    vel = np.linspace(vmin, vmax, mass_len)
    m_array = mej * (vel/vmin)**(-beta)
    total_mass = np.sum(m_array)
    normalised_mass = m_array * (mej/ total_mass)
    m_array = normalised_mass
    v_m = vel * speed_of_light

    # set up arrays
    time_array = np.tile(time, (mass_len, 1))
    e_th_array = np.tile(e_th, (mass_len, 1))
    edotr = np.zeros((mass_len, time_len))

    time_mask = time > t0
    time_1 = time_array[:, time_mask]
    time_2 = time_array[:, ~time_mask]
    edotr[:,time_mask] = 2.1e10 * e_th_array[:, time_mask] * ((time_1/ (3600. * 24.)) ** (-1.3))
    edotr[:, ~time_mask] = 4.0e18 * (0.5 - (1. / np.pi) * np.arctan((time_2 - t0) / sig)) ** (1.3) * e_th_array[:,~time_mask]

    lsd = magnetar_luminosity

    # set up empty arrays
    energy_v = np.zeros((mass_len, time_len))
    lum_rad = np.zeros((mass_len, time_len))
    qdot_rp = np.zeros((mass_len, time_len))
    td_v = np.zeros((mass_len, time_len))
    tau = np.zeros((mass_len, time_len))
    v_photosphere = np.zeros(time_len)
    v0_array = np.zeros(time_len)
    qdot_magnetar = np.zeros(time_len)
    r_photosphere = np.zeros(time_len)

    if neutron_precursor_switch == True:
        neutron_mass = 1e-4 * solar_mass
        neutron_mass_fraction = (2.0 / np.pi) * (1.0 - electron_fraction) * np.arctan(neutron_mass / m_array / solar_mass)
        rprocess_mass_fraction = 1.0 - neutron_mass_fraction
        initial_neutron_mass_fraction_array = np.tile(neutron_mass_fraction, (time_len, 1)).T
        rprocess_mass_fraction_array = np.tile(rprocess_mass_fraction, (time_len, 1)).T
        neutron_mass_fraction_array = initial_neutron_mass_fraction_array*np.exp(-time_array / tau_neutron)
        edotn = 3.2e14 * neutron_mass_fraction_array
        edotr = edotn + edotr
        kappa_n = 0.4 * (1.0 - neutron_mass_fraction_array - rprocess_mass_fraction_array)
        kappa = kappa * rprocess_mass_fraction_array
        kappa = kappa_n + kappa

    dt = np.diff(time)
    dm = np.abs(np.diff(m_array))

    #initial conditions
    energy_v[:, 0] = 0.5 * m_array*v_m**2
    lum_rad[:, 0] = 0
    qdot_rp[:, 0] = 0
    kinetic_energy = ek_tot_0

    # solve ODE using euler method for all mass shells v
    for ii in range(time_len - 1):
        # # evolve the velocity due to pdv work of central shell of mass M and thermal energy Ev0
        kinetic_energy = kinetic_energy + (energy_v[0, ii] / time[ii]) * dt[ii]
        # kinetic_energy = kinetic_energy + (np.sum(energy_v[:, ii]) / time[ii]) * dt[ii]
        v0 = (2 * kinetic_energy / m0) ** 0.5
        v0_array[ii] = v0
        v_m = v0 * (m_array / (mej)) ** (-1 / beta)
        v_m[v_m > 3e10] = speed_of_light

        if use_gamma_ray_opacity:
            kappa_gamma = kwargs["kappa_gamma"]
            prefactor = 3 * kappa_gamma * mej / (4 * np.pi * vej**2)
            thermalisation_efficiency = 1 - np.exp(-prefactor * time[ii] ** -2)
        else:
            thermalisation_efficiency = kwargs["thermalisation_efficiency"]
        qdot_magnetar[ii] = thermalisation_efficiency * lsd[ii]

        if magnetar_heating == 'all_layers':
            if neutron_precursor_switch:
                td_v[:-1, ii] = (kappa[:-1, ii] * m_array[:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            else:
                td_v[:-1, ii] = (kappa * m_array[:-1] * solar_mass * 3) / (4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            lum_rad[:-1, ii] = energy_v[:-1, ii] / (td_v[:-1, ii] + time[ii] * (v_m[:-1] / speed_of_light))
            energy_v[:-1, ii + 1] = (qdot_magnetar[ii] + edotr[:-1, ii] * dm * solar_mass - (energy_v[:-1, ii] / time[ii]) - lum_rad[:-1, ii]) * dt[ii] + energy_v[:-1, ii]

        # first mass layer
        # only bottom layer i.e., 0'th mass layer gets magnetar contribution
        if magnetar_heating == 'first_layer':
            if neutron_precursor_switch:
                td_v[0, ii] = (kappa[0, ii] * m_array[0] * solar_mass * 3) / (
                            4 * np.pi * v_m[0] * speed_of_light * time[ii] * beta)
                td_v[1:-1, ii] = (kappa[1:-1, ii] * m_array[1:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[1:-1] * speed_of_light * time[ii] * beta)
            else:
                td_v[0, ii] = (kappa * m_array[0] * solar_mass * 3) / (4 * np.pi * v_m[0] * speed_of_light * time[ii] * beta)
                td_v[1:-1, ii] = (kappa * m_array[1:-1] * solar_mass * 3) / (
                            4 * np.pi * v_m[1:-1] * speed_of_light * time[ii] * beta)

            lum_rad[0, ii] = energy_v[0, ii] / (td_v[0, ii] + time[ii] * (v_m[0] / speed_of_light))
            energy_v[0, ii + 1] = (qdot_magnetar[ii] + edotr[0, ii] * dm[0] * solar_mass - (energy_v[0, ii] / time[ii]) - lum_rad[0, ii]) * dt[ii] + energy_v[0, ii]
            # other layers
            lum_rad[1:-1, ii] = energy_v[1:-1, ii] / (td_v[1:-1, ii] + time[ii] * (v_m[1:-1] / speed_of_light))
            energy_v[1:-1, ii + 1] = (edotr[1:-1, ii] * dm[1:] * solar_mass - (energy_v[1:-1, ii] / time[ii]) - lum_rad[1:-1, ii]) * dt[ii] + energy_v[1:-1, ii]

        if neutron_precursor_switch:
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa[:-1, ii] / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))
        else:
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))

        tau[mass_len - 1, ii] = tau[mass_len - 2, ii]
        photosphere_index = np.argmin(np.abs(tau[:, ii] - 1))
        v_photosphere[ii] = v_m[photosphere_index]
        r_photosphere[ii] = v_photosphere[ii] * time[ii]

    bolometric_luminosity = np.sum(lum_rad, axis=0)

    if pair_cascade_switch == True:
        tlife_t = (0.6/(1 - ejecta_albedo))*(pair_cascade_fraction/0.1)**0.5 * (lsd/1.0e45)**0.5 \
                  * (v0/(0.3*speed_of_light))**(0.5) * (time/day_to_s)**(-0.5)
        bolometric_luminosity = bolometric_luminosity / (1.0 + tlife_t)

    temperature = (bolometric_luminosity / (4.0 * np.pi * (r_photosphere) ** (2.0) * sigma_sb)) ** (0.25)

    gamma_beta = v0_array/speed_of_light
    lorentz_factor = 1/(np.sqrt(1 - gamma_beta**2))
    dynamics_output = namedtuple(
        'dynamics_output',
        ['lorentz_factor', 'bolometric_luminosity', 'temperature',
         'r_photosphere', 'kinetic_energy', 'erad_total',
         'thermalisation_efficiency'],
    )
    return dynamics_output(
        lorentz_factor=lorentz_factor,
        bolometric_luminosity=bolometric_luminosity,
        temperature=temperature,
        r_photosphere=r_photosphere,
        kinetic_energy=(lorentz_factor - 1) * m0 * speed_of_light ** 2,
        erad_total=np.trapezoid(bolometric_luminosity, x=time),
        thermalisation_efficiency=qdot_magnetar / lsd,
    )

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017LRR....20....3M/abstract')
def metzger_magnetar_driven_kilonova_model(time, redshift, mej, vej, beta, kappa_r, p0, bp,
                                           mass_ns, theta_pb, thermalisation_efficiency, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param p0: initial spin period in milliseconds
    :param bp: polar magnetic field strength in units of 10^14 Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes in radians
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    use_gamma_ray_opacity = False
    kwargs['use_relativistic_blackbody'] = False

    time_temp = get_optimal_time_array(1e-4, 1e7, 300) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = basic_magnetar(time=time_temp, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             thermalisation_efficiency=thermalisation_efficiency,
                                                             **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_metzger_magnetar_driven(time, redshift, mej, vej, beta, kappa_r, l0,
                                    tau_sd, nn, thermalisation_efficiency, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param thermalisation_efficiency: magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default true
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    use_gamma_ray_opacity = False
    kwargs['use_relativistic_blackbody'] = False

    time_temp = get_optimal_time_array(1e-4, 1e7, 300) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             thermalisation_efficiency=thermalisation_efficiency,
                                                             **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_metzger_magnetar_driven_thermalisation(time, redshift, mej, vej, beta, kappa_r, l0,
                                    tau_sd, nn, kappa_gamma, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param l0: initial magnetar X-ray luminosity
    :param tau_sd: magnetar spin down damping timescale
    :param nn: braking index
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default true
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    kwargs['use_relativistic_blackbody'] = False
    use_gamma_ray_opacity = True

    time_temp = get_optimal_time_array(1e-4, 1e7, 300) #in source frame
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             kappa_gamma=kappa_gamma, **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220514159S/abstract')
def general_metzger_magnetar_driven_evolution(time, redshift, mej, vej, beta, kappa_r, logbint,
                                 logbext, p0, chi0, radius, logmoi, kappa_gamma, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param logbint: log10 internal magnetic field in G
    :param logbext: log10 external magnetic field in G
    :param p0: spin period in s
    :param chi0: initial inclination angle
    :param radius: radius of NS in KM
    :param logmoi: log10 moment of inertia of NS
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency
    :param kwargs: Additional parameters
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param neutron_precursor_switch: whether to have neutron precursor emission, default true
    :param pair_cascade_switch: whether to account for pair cascade losses, default is True
    :param magnetar_heating: whether magnetar heats all layers or just the bottom layer. default first layer only
    :param vmax: maximum initial velocity of mass layers, default is 0.7c
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    use_gamma_ray_opacity = True
    kwargs['use_relativistic_blackbody'] = False

    time_temp = get_optimal_time_array(1e-4, 1e7, 500) #in source frame
    bint = 10 ** logbint
    bext = 10 ** logbext
    radius = radius * km_cgs
    moi = 10 ** logmoi
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    output = _evolving_gw_and_em_magnetar(time=time_temp, bint=bint, bext=bext, p0=p0, chi0=chi0, radius=radius, moi=moi)
    magnetar_luminosity = output.Edot_d
    output = _general_metzger_magnetar_driven_kilonova_model(time=time_temp, mej=mej, vej=vej, beta=beta, kappa=kappa_r,
                                                             magnetar_luminosity=magnetar_luminosity,
                                                             use_gamma_ray_opacity=use_gamma_ray_opacity,
                                                             kappa_gamma=kappa_gamma, **kwargs)
    time_obs = time
    if kwargs['output_format'] == 'flux_density':
        return _process_flux_density(dl=dl, output=output, redshift=redshift,
                                     time=time, time_temp=time_temp, **kwargs)
    else:
        return _processing_other_formats(dl=dl, output=output, redshift=redshift,
                                         time_obs=time_obs, time_temp=time_temp, **kwargs)
