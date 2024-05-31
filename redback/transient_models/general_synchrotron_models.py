import numpy as np
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar
from redback.transient_models.magnetar_driven_ejecta_models import _ejecta_dynamics_and_interaction
from astropy.cosmology import Planck18 as cosmo
from redback.utils import calc_kcorrected_properties, citation_wrapper, logger, get_csm_properties, nu_to_lambda, lambda_to_nu, velocity_from_lorentz_factor
from redback.constants import day_to_s, solar_mass, km_cgs, au_cgs, speed_of_light, qe, electron_mass, proton_mass, sigma_T
from scipy import integrate
from scipy.interpolate import interp1d

def _calc_free_free_abs(frequency, Y_fe, Zbar, mej, radius_2darray, F_nu_2darray):
    """
    :param frequency: frequency to calculate
    :param Y_fe: free electron fraction
    :param Zbar: average proton number
    :param mej: ejecta mass in solar units
    :param radius_2darray: radius of the ejecta at each time
    :param F_nu_2darray: unabsorbed flux density
    :return: absorbed flux density
    """
    n_e = mej * solar_mass * 3 / (4 * np.pi * radius_2darray**3) * Y_fe / proton_mass
    tau_ff = 8.4e-28 * n_e**2 * radius_2darray * Zbar**2 * (frequency / 1.0e10) ** -2.1
    F_nu_2darray = F_nu_2darray * np.exp(-tau_ff)
    
    return F_nu_2darray

def _calc_compton_scat(frequency, Y_e, mej, radius_2darray, F_nu_2darray):
    """
    :param frequency: frequency to calculate
    :param Y_e: electron fraction
    :param mej: ejecta mass in solar units
    :param radius_2darray: radius of the ejecta at each time
    :param F_nu_2darray: unabsorbed flux density
    :return: absorbed flux density
    """
    nu_e = 1.24e20
    x = frequency/nu_e
    msk = (x < 1e-3)
    sigknpre = (1.0 + x)/x**3
    sigkn1 = 2.0 * x * (1.0 + x)/(1.0 + 2.0*x) - np.log(1.0 + 2*x)
    sigkn2 = np.log(1.0 + 2.0*x)/(2.0*x)
    sigkn3 = (1.0 + 3.0*x)/(1.0 + 2.0*x)**2
    sig_kn = (3.0/4.0) * sigma_T * (sigknpre*sigkn1 + sigkn2 - sigkn3)
    sig_kn[msk] = sigma_T
    kappa_comp = sig_kn * Y_e / proton_mass
    tau_comp = 3.0 * kappa_comp * mej * solar_mass / (4.0 * np.pi * radius_2darray**2)
    F_nu_2darray = F_nu_2darray * np.exp(-tau_comp)
    
    return F_nu_2darray

def _calc_photoelectric_abs(frequency, Zbar, mej, radius_2darray, F_nu_2darray):
    """
    :param frequency: frequency to calculate
    :param Zbar: average proton number
    :param mej: ejecta mass in solar units
    :param radius_2darray: radius of the ejecta at each time
    :param F_nu_2darray: unabsorbed flux density
    :return: absorbed flux density
    """
    msk = (frequency > 2.42e15)
    kappa_pe = 2.37 * Zbar**3 * (frequency/2.42e18)**-3
    tau_pe = 3.0 * kappa_pe * mej * solar_mass / (4.0 * np.pi * radius_2darray**2)
    F_nu_2darray[:,msk] = F_nu_2darray[:,msk] * np.exp(-tau_pe[:,msk])
    
    return F_nu_2darray

@citation_wrapper('Omand et al. (2024)')
def pwn(time, redshift, mej, l0, tau_sd, nn, eps_b, gamma_b, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar units
    :param l0: initial magnetar spin-down luminosity (in erg/s)
    :param tau_sd: magnetar spin down damping timescale (in seconds)
    :param nn: braking index
    :param eps_b: magnetization of the PWN
    :param gamma_b: Lorentz factor of electrons at synchrotron break
    :param kwargs: Additional parameters - 
    :param E_sn: supernova explosion energy
    :param kappa: opacity (used only in dynamics)
    :param kappa_gamma: gamma-ray opacity used to calculate magnetar thermalisation efficiency (used only in dynamics)
    :param q1: low energy spectral index (must be < 2)
    :param q2: high energy spectral index (must be > 2)
    :param Zbar: average proton number (used for free-free and photoelectric absorption)
    :param Y_e: electron fraction (used for Compton scattering)
    :param Y_fe: free electron fraction (used for free-free absorption)
    :param pair_cascade_switch: whether to account for pair cascade losses, default is False
    :param ejecta albedo: ejecta albedo; default is 0.5
    :param pair_cascade_fraction: fraction of magnetar luminosity lost to pair cascades; default is 0.05
    :param use_r_process: determine whether the ejecta is composed of r-process material; default is no
    :param frequency: (frequency to calculate - Must be same length as time array or a single number)
    :param f_nickel: Ni^56 mass as a fraction of ejecta mass
    :return: flux density or AB magnitude or dynamics output
    """
    #get parameter values or use defaults
    E_sn = kwargs.get('E_sn', 1.0e51)
    kappa = kwargs.get('kappa', 0.1)
    if 'kappa' in kwargs:
        del kwargs['kappa']
    kappa_gamma = kwargs.get('kappa_gamma', 0.01)
    kwargs['kappa_gamma'] = kappa_gamma
    q1 = kwargs.get('q1',1.5)
    q2 = kwargs.get('q2',2.5)
    Zbar = kwargs.get('Zbar',8.0)
    Y_e = kwargs.get('Y_e',0.5)
    Y_fe = kwargs.get('Y_fe',0.0625)
    
    ejecta_radius = 1.0e11
    epse=1.0-eps_b
    n_ism = 1.0e-5
    dl = cosmo.luminosity_distance(redshift).cgs.value
    pair_cascade_switch = kwargs.get('pair_cascade_switch', False)
    use_r_process = kwargs.get('use_r_process', False)
    nu_M=3.8e22*np.ones(2500)

    #initial values and dynamics
    time_temp = np.geomspace(1e0, 1e10, 2500)
    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    magnetar_luminosity = magnetar_only(time=time_temp, l0=l0, tau=tau_sd, nn=nn)
    v_init = np.sqrt(E_sn / (0.5 * mej * solar_mass)) / speed_of_light
    output = _ejecta_dynamics_and_interaction(time=time_temp, mej=mej,
                                          beta=v_init, ejecta_radius=ejecta_radius,
                                          kappa=kappa, n_ism=n_ism, magnetar_luminosity=magnetar_luminosity,
                                          pair_cascade_switch=pair_cascade_switch,
                                          use_gamma_ray_opacity=True, **kwargs)                                                                                
    vej = velocity_from_lorentz_factor(output.lorentz_factor)/km_cgs 

    #calculating synchrotron quantites
    int_lsd = integrate.cumtrapz(magnetar_luminosity, time_temp,initial=0)
    B_nb = np.sqrt(6.0 * eps_b * int_lsd / output.radius**3)
    B_nb[0] = B_nb[1]
    nu_b = 3.0 / 4.0 / np.pi * gamma_b**2 * qe * B_nb / electron_mass / speed_of_light
    nu_0 = np.minimum(nu_M, nu_b)
    beta1 = np.maximum(1.5, ((2.0 + q1) / 2.0))
    beta2 = (2.0 + q2) / 2.0
    Rb = ((1.0 / (2.0 - q1)) - (1.0 / (2.0 - q2))) #bolometric correction
    F_nu_0 = epse * magnetar_luminosity / (8.0 * np.pi * dl**2 * nu_0 * Rb)
    nu_ssa = (dl**2 * 3.0**1.5 * qe**0.5 * B_nb**0.5 * F_nu_0 * nu_0**(beta1 - 1.0) / (4.0 * np.pi**1.5 * output.radius**2 * speed_of_light**0.5 * electron_mass**1.5))**(2.0 / (2 * beta1 + 3))
    F_nu_ssa = F_nu_0 * (nu_ssa / nu_0) ** (1-beta1)

    #making arrays to vectorize properly
    freq_arr = np.tile(frequency, (2500,1))
    F_nu_0_arr = np.tile(F_nu_0, (len(frequency),1))
    nu_0_arr = np.tile(nu_0, (len(frequency),1))
    F_nu_ssa_arr = np.tile(F_nu_ssa, (len(frequency),1))
    nu_ssa_arr = np.tile(nu_ssa, (len(frequency),1))
    r_arr = np.tile(output.radius, (len(frequency),1))
    nu_b_arr = np.tile(nu_b, (len(frequency),1))

    #calculate synchtron light curves for each desired frequency
    F_nu = F_nu_0_arr.T * (frequency / nu_0_arr.T) ** (1-beta1)

    if (np.max(frequency) > np.min(nu_b)):
        msk = (freq_arr >= nu_b_arr.T)
        F_nu[msk] = F_nu_0_arr.T[msk] * (freq_arr[msk] / nu_0_arr.T[msk]) ** (1-beta2)
        
    if (np.min(frequency) < np.max(nu_ssa)):
        msk = (freq_arr < nu_ssa_arr.T)
        F_nu[msk] = F_nu_ssa_arr.T[msk] * (freq_arr[msk] / nu_ssa_arr.T[msk]) ** 2.5 
        
    F_nu = _calc_free_free_abs(frequency, Y_fe, Zbar, mej, r_arr.T, F_nu)
    F_nu = _calc_compton_scat(frequency, Y_e, mej, r_arr.T, F_nu) 
    if (np.max(frequency) < 2.42e15):
        F_nu = _calc_photoelectric_abs(frequency, Zbar, mej, r_arr.T, F_nu)

    #interpolate for each time
    fnu_func = {}
    fnu_func = interp1d(time_temp/day_to_s, y=F_nu.T)
    fnu = np.diag(fnu_func(time))   
    fmjy = np.array(fnu) / 1.0e-26       
    
    return fmjy
