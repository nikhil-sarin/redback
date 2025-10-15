import numpy as np
from collections import namedtuple
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo
from redback.utils import calc_kcorrected_properties, citation_wrapper, lambda_to_nu, get_cosmology_from_kwargs
from redback.constants import *
from scipy.interpolate import interp1d

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...932...84M')
def _wr_bh_merger(time, M_star, M_bh, M_fast, M_pre, v_fast, v_slow, alpha, eta, theta, phi_0, kappa_s, kappa_f, kappa_x, N, **kwargs): 
    """
    Parameters:
    :param time: time in source frame in seconds
    :param M_star: Mass of the Wolf-Rayet star in solar masses
    :param M_bh: Mass of the black hole in solar masses
    :param M_fast: Mass of the fast component in solar masses
    :param M_pre: Mass of the pre-merger CSM in solar masses
    :param v_fast: Velocity of the fast component in units of c
    :param v_slow: Velocity of the slow component in km/s
    :param alpha: Viscosity parameter
    :param eta: Efficiency of conversion of accretion energy to radiation
    :param theta: Disk aspect ratio
    :param phi_0: Solid angle of the slow component
    :param kappa_s: Opacity of the slow component
    :param kappa_f: Opacity of the fast component
    :param kappa_x: Opacity of the x-ray component
    :param N: Number of pre-merger orbits
    :param kwargs: Additional parameters
    """
    # Calculate constants
    M_acc = M_acc = 0.05 * (M_bh/10.0)**0.6 * (M_star/10.0)**0.65
    M_slow = M_star - M_fast - M_acc 

    t_visc = 0.55 * day_to_s * (alpha / 0.1)**-1 * (M_star / 10.0)**0.87 * (M_bh / 10.0)**-0.5 * (theta / 0.33)**-2.0
    L_acc0 = 1.6e44 * (M_bh / 10.0)**0.03 * (M_star / 10.0)**1.63 * (eta / 1e-2) * (alpha / 0.1)**-1.1 * (theta / 0.33)**-2.3
    L_acc_tvisc = L_acc0 * (t_visc / (3.0 * day_to_s))**-2.1 * np.exp(-1)
    vesc_trun = 9e13 * (N / 100) * (M_star / 10.0)**0.58

    # Time-dependent properties
    mask = time < t_visc
    e_slow = 0.5 * (M_slow * solar_mass) * (v_slow * km_cgs)**2
    e_fast = ((1.0 - v_fast**2)**-0.5 - 1.0) * (M_fast * solar_mass) * speed_of_light**2.0

    rad_slow = v_slow * km_cgs * time
    t_diff_slow = (M_slow * solar_mass) * kappa_s / (4.0 * np.pi * rad_slow * speed_of_light)
    t_lc_slow = rad_slow / speed_of_light

    rad_fast = v_fast * speed_of_light * time
    t_diff_fast = (M_fast * solar_mass) * kappa_f / (4.0 * np.pi * rad_fast * speed_of_light)
    t_lc_fast = rad_fast / speed_of_light
    tau_x = kappa_x * (M_fast * solar_mass) / (4.0 * np.pi * rad_fast**2)

    L_sh = 0.5 * (M_pre * solar_mass) * (v_slow * km_cgs)**3 / (vesc_trun) * np.exp(-v_slow * km_cgs * time / vesc_trun)
    L_acc = L_acc0 * (time / (3.0 * day_to_s))**-2.1 * np.exp(-(time / t_visc)**0.28)
    L_acc[mask] = L_acc_tvisc
    L_acc_th = phi_0 * (1 - np.exp(-tau_x)) * L_acc + phi_0 * L_acc
    L_x = phi_0 * np.exp(-tau_x) * L_acc

    # Initialize arrays
    energy_fast = [e_fast]
    energy_slow = [e_slow]
    L_opt_sh = []
    L_opt_rep = []

    for i in range(len(time)):
        if i > 0:
            dt = time[i] - time[i - 1]
            de_slow_dt = -e_slow / time[i] - lum_opt_sh + L_sh[i]
            de_fast_dt = -e_fast / time[i] - lum_opt_rep + L_acc_th[i]
            e_slow += de_slow_dt * dt
            if e_slow < 0:
                e_slow = 0
            e_fast += de_fast_dt * dt    
            if e_fast < 0:
                e_fast = 0
            

        lum_opt_sh = e_slow / (t_lc_slow[i] + t_diff_slow[i])
        lum_opt_rep = e_fast / (t_lc_fast[i] + t_diff_fast[i])
        L_opt_sh.append(lum_opt_sh)
        L_opt_rep.append(lum_opt_rep)
        energy_fast.append(e_fast)
        energy_slow.append(e_slow)

    dynamics_output = namedtuple('dynamics_output', ['time', 'energy_fast', 'energy_slow', 'rad_fast', 'rad_slow', 
                                                     'reprocessed_luminosity', 'shock_powered_luminosity', 'optical_luminosity', 
                                                     'x_ray_luminosity', 'accretion_luminosity', 'erad_opt_total'])

    dynamics_output.time = time
    dynamics_output.energy_fast = energy_fast
    dynamics_output.energy_slow = energy_slow
    dynamics_output.rad_fast = rad_fast
    dynamics_output.rad_slow = rad_slow
    dynamics_output.reprocessed_luminosity = L_opt_rep
    dynamics_output.shock_powered_luminosity = L_opt_sh
    dynamics_output.optical_luminosity = np.array(L_opt_rep) + np.array(L_opt_sh)
    dynamics_output.x_ray_luminosity = L_x
    dynamics_output.accretion_luminosity = L_acc
    dynamics_output.erad_opt_total = np.trapz(np.array(L_opt_sh) + np.array(L_opt_sh), x=time)
    return dynamics_output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...932...84M')
def wr_bh_merger_bolometric(time, M_star, M_bh, M_fast, M_pre, v_fast, v_slow, alpha, eta, **kwargs):
    """
    Parameters:
    :param time: time in source frame in days
    :param M_star: Mass of the Wolf-Rayet star in solar masses
    :param M_bh: Mass of the black hole in solar masses
    :param M_fast: Mass of the fast component in solar masses
    :param M_pre: Mass of the pre-merger CSM in solar masses
    :param v_fast: Velocity of the fast component in units of c
    :param v_slow: Velocity of the slow component in km/s
    :param alpha: Viscosity parameter
    :param eta: Efficiency of conversion of accretion energy to radiation
    :param kwargs: Additional parameters
    :param output_format: whether to output dynamics or bolometric luminosity
    :param theta: Disk aspect ratio
    :param phi_0: Solid angle of the slow component
    :param kappa_s: Opacity of the slow component
    :param kappa_f: Opacity of the fast component
    :param kappa_x: Opacity of the x-ray component
    :param N: Number of pre-merger orbits
    :return: bolometric luminosity or dynamics output
    """
    theta = kwargs.get('theta', 0.33)
    phi_0 = kwargs.get('phi_0', 0.5)
    kappa_s = kwargs.get('kappa_s', 0.03)
    kappa_f = kwargs.get('kappa_f', 0.2)
    kappa_x = kwargs.get('kappa_x', 0.4)
    N = kwargs.get('N', 30)

    time_temp = np.geomspace(1e0, 1e8, 2000)
    dynamics_output = _wr_bh_merger(time_temp, M_star, M_bh, M_fast, M_pre, v_fast, v_slow, alpha, eta, theta, phi_0, kappa_s, kappa_f, kappa_x, N, **kwargs)
    lbol_func = interp1d(time_temp, y=dynamics_output.optical_luminosity)
    time = time * day_to_s    
    lbol = lbol_func(time)
    if kwargs['output_format'] == 'dynamics_output':
        return dynamics_output
    else:
        return lbol
    
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...932...84M')
def wr_bh_merger(time, redshift, M_star, M_bh, M_fast, M_pre, v_fast, v_slow, alpha, eta, **kwargs):
    """
    Parameters:
    :param time: time in source frame in days
    :param redshift: redshift 
    :param M_star: Mass of the Wolf-Rayet star in solar masses
    :param M_bh: Mass of the black hole in solar masses
    :param M_fast: Mass of the fast component in solar masses
    :param M_pre: Mass of the pre-merger CSM in solar masses
    :param v_fast: Velocity of the fast component in units of c
    :param v_slow: Velocity of the slow component in km/s
    :param alpha: Viscosity parameter
    :param eta: Efficiency of conversion of accretion energy to radiation
    :param kwargs: Additional parameters - Must be all the kwargs required by the specific photosphere, sed methods used
             e.g., for TemperatureFloor: vej (km/s) and temperature_floor
    :param theta: Disk aspect ratio
    :param phi_0: Solid angle of the slow component
    :param kappa_s: Opacity of the slow component
    :param kappa_f: Opacity of the fast component
    :param kappa_x: Opacity of the x-ray component
    :param N: Number of pre-merger orbits
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'dynamics_output', 'spectra', 'flux', 'sncosmo_source'
    """

    theta = kwargs.get('theta', 0.33)
    phi_0 = kwargs.get('phi_0', 0.5)
    kappa_s = kwargs.get('kappa_s', 0.03)
    kappa_f = kwargs.get('kappa_f', 0.2)
    kappa_x = kwargs.get('kappa_x', 0.4)
    N = kwargs.get('N', 30)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = get_cosmology_from_kwargs(kwargs)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    time_temp = np.geomspace(1e0, 1e8, 2000)
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        output = _wr_bh_merger(time_temp, M_star, M_bh, M_fast, M_pre, v_fast, v_slow, alpha, eta, theta, phi_0, kappa_s, kappa_f, kappa_x, N, **kwargs)
        photo_fast = kwargs['photosphere'](time=time_temp/day_to_s, luminosity=np.array(output.reprocessed_luminosity), vej = v_fast * speed_of_light / km_cgs, **kwargs) 
        photo_slow = kwargs['photosphere'](time=time_temp/day_to_s, luminosity=np.array(output.shock_powered_luminosity), vej = v_slow, **kwargs)
        temp_func_fast = interp1d(time_temp/day_to_s, y=photo_fast.photosphere_temperature)
        temp_func_slow = interp1d(time_temp/day_to_s, y=photo_slow.photosphere_temperature)
        rad_func_fast = interp1d(time_temp/day_to_s, y=photo_fast.r_photosphere)
        rad_func_slow = interp1d(time_temp/day_to_s, y=photo_slow.r_photosphere)
        temp_fast = temp_func_fast(time)
        temp_slow = temp_func_slow(time)
        rad_fast = rad_func_fast(time)
        rad_slow = rad_func_slow(time)
        sed_fast = kwargs['sed'](temperature=temp_fast, r_photosphere=rad_fast, 
                                            frequency=frequency, luminosity_distance=dl) 
        sed_slow = kwargs['sed'](temperature=temp_slow, r_photosphere=rad_slow, 
                                            frequency=frequency, luminosity_distance=dl)
        flux_density = sed_fast.flux_density + sed_slow.flux_density
        return flux_density.to(uu.mJy).value
    
    else:
        time_obs = time
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(500, 60000, 200))
        time_temp = np.geomspace(1e0, 1e8, 2000)
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                              redshift=redshift, time=time_observer_frame)
        output = _wr_bh_merger(time_temp, M_star, M_bh, M_fast, M_pre, v_fast, v_slow, alpha, eta, theta, phi_0, kappa_s, kappa_f, kappa_x, N, **kwargs)
        photo_fast = kwargs['photosphere'](time=time_temp/day_to_s, luminosity=np.array(output.reprocessed_luminosity), vej = v_fast * speed_of_light / km_cgs, **kwargs) 
        photo_slow = kwargs['photosphere'](time=time_temp/day_to_s, luminosity=np.array(output.shock_powered_luminosity), vej = v_slow, **kwargs)
        if kwargs['output_format'] == 'dynamics_output':                                               
            dynamics_output = namedtuple('dynamics_output', ['time', 'temperature_fast', 'temperature_slow','r_photosphere_fast',
                                                     'r_photosphere_slow','energy_fast','energy_slow','optical_luminosity',
                                                     'reprocessed_luminosity','shock_powered_luminosity','x_ray_luminosity',
                                                     'accretion_luminosity','erad_opt_total'])
            dynamics_output.time = output.time                                                                    
            dynamics_output.temperature_fast = photo_fast.photosphere_temperature
            dynamics_output.temperature_slow = photo_slow.photosphere_temperature
            dynamics_output.r_photosphere_fast = photo_fast.r_photosphere 
            dynamics_output.r_photosphere_slow = photo_slow.r_photosphere 
            dynamics_output.energy_fast = output.energy_fast
            dynamics_output.energy_slow = output.energy_slow
            dynamics_output.optical_luminosity = output.optical_luminosity
            dynamics_output.reprocessed_luminosity = output.reprocessed_luminosity
            dynamics_output.shock_powered_luminosity = output.shock_powered_luminosity
            dynamics_output.x_ray_luminosity = output.x_ray_luminosity
            dynamics_output.accretion_luminosity = output.accretion_luminosity
            dynamics_output.erad_opt_total = output.erad_opt_total            
            return dynamics_output                
        sed_fast = kwargs['sed'](temperature=photo_fast.photosphere_temperature, r_photosphere=photo_fast.r_photosphere,
                    frequency=frequency[:,None], luminosity_distance=dl)
        sed_slow = kwargs['sed'](temperature=photo_slow.photosphere_temperature, r_photosphere=photo_slow.r_photosphere,
                    frequency=frequency[:,None], luminosity_distance=dl)
        fmjy = sed_fast.flux_density + sed_slow.flux_density
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                              spectra=spectra, lambda_array=lambda_observer_frame,
                                                              **kwargs)