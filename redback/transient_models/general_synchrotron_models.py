import numpy as np
from redback.transient_models.magnetar_models import magnetar_only, basic_magnetar
from redback.transient_models.magnetar_driven_ejecta_models import _ejecta_dynamics_and_interaction
from redback.transient_models.shock_powered_models import _emissivity_pl, _emissivity_thermal, _tau_nu
from redback.transient_models.afterglow_models import _get_kn_dynamics, _pnu_synchrotron
from astropy.cosmology import Planck18 as cosmo
from redback.utils import calc_kcorrected_properties, citation_wrapper, logger, get_csm_properties, nu_to_lambda, lambda_to_nu, velocity_from_lorentz_factor, calc_ABmag_from_flux_density
from redback.constants import day_to_s, solar_mass, km_cgs, au_cgs, speed_of_light, qe, electron_mass, proton_mass, sigma_T
from scipy import integrate
from scipy.interpolate import interp1d
import astropy.units as uu
from collections import namedtuple

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
    if (np.size(sig_kn) > 1):
        sig_kn[msk] = sigma_T
    elif ((np.size(sig_kn) == 1) and (msk == True)):
        sig_kn = sigma_T   
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
    kappa_pe = 2.37 * (Zbar/6.0)**3 * (frequency/2.42e18)**-3
    tau_pe = 3.0 * kappa_pe * mej * solar_mass / (4.0 * np.pi * radius_2darray**2)
    F_nu_2darray[:,msk] = F_nu_2darray[:,msk] * np.exp(-tau_pe[:,msk])
    
    return F_nu_2darray
    
def _calc_optical_abs(frequency, kappa, mej, radius_2darray, F_nu_2darray):
    """
    :param frequency: frequency to calculate
    :param kappa: thermalization opacity
    :param mej: ejecta mass in solar units
    :param radius_2darray: radius of the ejecta at each time
    :param F_nu_2darray: unabsorbed flux density
    :return: absorbed flux density
    """
    msk = np.logical_and((frequency < 2.42e15),(frequency > 2.42e13))
    tau_opt = 3.0 * kappa * mej * solar_mass / (4.0 * np.pi * radius_2darray**2)
    F_nu_2darray[:,msk] = F_nu_2darray[:,msk] * np.exp(-tau_opt[:,msk])
    
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
    :param kappa: opacity (used only in dynamics and optical absorption)
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
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    pair_cascade_switch = kwargs.get('pair_cascade_switch', False)
    use_r_process = kwargs.get('use_r_process', False)
    nu_M=3.8e22*np.ones(2500)

    #initial values and dynamics
    time_temp = np.geomspace(1e0, 1e10, 2500)
    frequency = kwargs['frequency']
    if (np.size(frequency) == 1):
        frequency = np.ones(len(time))*frequency
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
    int_lsd = integrate.cumulative_trapezoid(magnetar_luminosity, time_temp,initial=0)
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
    F_nu_0_arr = np.tile(F_nu_0, (np.size(frequency),1))
    nu_0_arr = np.tile(nu_0, (np.size(frequency),1))
    F_nu_ssa_arr = np.tile(F_nu_ssa, (np.size(frequency),1))
    nu_ssa_arr = np.tile(nu_ssa, (np.size(frequency),1))
    r_arr = np.tile(output.radius, (np.size(frequency),1))
    nu_b_arr = np.tile(nu_b, (np.size(frequency),1))

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
    F_nu = _calc_optical_abs(frequency, kappa, mej, r_arr.T, F_nu) 
    if (np.max(frequency) > 2.42e15):
        F_nu = _calc_photoelectric_abs(frequency, Zbar, mej, r_arr.T, F_nu)

    #interpolate for each time
    fnu_func = {}
    fnu_func = interp1d(time_temp/day_to_s, y=F_nu.T)
    fnu = np.diag(fnu_func(time))   
    fmjy = np.array(fnu) / 1.0e-26       
    
    return fmjy
    
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract')
def kilonova_afterglow_redback(time, redshift, loge0, mej, logn0, logepse, logepsb, p,
                             **kwargs):
    """
    Calculate the afterglow emission from a kilonova remnant, following the model of Sarin et al. 2022.
    This model was modified by Nikhil Sarin following code provided by Ben Margalit.

    :param time: time in observer frame (days) in observer frame
    :param redshift: source redshift
    :param loge0: log10 of the initial kinetic energy of the ejecta (erg)
    :param mej: ejecta mass (solar masses)
    :param logn0: log10 of the circumburst density (cm^-3)
    :param logepse: log10 of the fraction of shock energy given to electrons
    :param logepsb: log10 of the fraction of shock energy given to magnetic field
    :param p: power-law index of the electron energy distribution
    :param kwargs: Additional keyword arguments
    :param zeta_e: fraction of electrons participating in diffusive shock acceleration. Default is 1.
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    Eej = 10 ** loge0
    Mej = mej * solar_mass
    cosmology = kwargs.get('cosmology', cosmo)
    epsilon_e = 10 ** logepse
    epsilon_B = 10 ** logepsb
    n0 = 10 ** logn0
    zeta_e = kwargs.get('zeta_e', 1.0)
    qe = 4.803e-10

    dl = cosmology.luminosity_distance(redshift).cgs.value
    # calculate blast-wave dynamics
    t, R, beta, Gamma, eden, tobs, beta_sh, Gamma_sh = _get_kn_dynamics(n0=n0, Eej=Eej, Mej=Mej)

    # shock-amplified magnetic field, minimum & cooling Lorentz factors
    B = (8.0 * np.pi * epsilon_B * eden) ** 0.5
    gamma_m = 1.0 + (epsilon_e / zeta_e) * ((p - 2.0) / (p - 1.0)) * (proton_mass / electron_mass) * (Gamma - 1.0)
    gamma_c = 6.0 * np.pi * electron_mass * speed_of_light / (sigma_T * Gamma * t * B ** 2)

    # number of emitting electrons, where zeta_DN is an approximate smooth interpolation between the "standard"
    # and deep-Newtonian regime discussed by Sironi & Giannios (2013)
    zeta_DN = (gamma_m - 1.0) / gamma_m
    Ne = zeta_DN * zeta_e * (4.0 * np.pi / 3.0) * R ** 3 * n0

    # LOS approximation
    mu = 1.0
    blueshift = Gamma * (1.0 - beta * mu)

    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    fnu_func = {}
    for nu in frequency:
        Fnu_opt_thin = _pnu_synchrotron(nu * blueshift * (1.0 + redshift), B, gamma_m, gamma_c, Ne, p) * (1.0 + redshift) / (
                    4.0 * np.pi * dl ** 2 * blueshift)

        # correct for synchrotron self-absorption (approximate treatment, correct up to factors of order unity)
        Fnu_opt_thick = Gamma * (8 * np.pi ** 2 * (nu * blueshift * (1.0 + redshift)) ** 2 / speed_of_light ** 2) * R ** 2 * (
                    1.0 / 3.0) * electron_mass * speed_of_light ** 2 * np.maximum(gamma_m, (
                    2 * np.pi * electron_mass * speed_of_light * nu * blueshift * (1.0 + redshift) / (qe * B)) ** 0.5) * (1.0 + redshift) / (
                                    4.0 * np.pi * dl ** 2 * blueshift)
        # new prescription:
        Fnu = Fnu_opt_thick * (1e0 - np.exp(-Fnu_opt_thin / Fnu_opt_thick))
        # add brute-force optically-thin case to avoid roundoff error in 1e0-np.exp(-x) term (above) when x->0
        Fnu[Fnu == 0.0] = Fnu_opt_thin[Fnu == 0.0]

        fnu_func[nu] = interp1d(tobs/day_to_s, Fnu, bounds_error=False, fill_value='extrapolate')

    # interpolate onto actual observed frequency and time values
    flux_density = []
    for freq, tt in zip(frequency, time):
        flux_density.append(fnu_func[freq](tt))

    fmjy = np.array(flux_density) / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2011Natur.478...82N/abstract')
def kilonova_afterglow_nakarpiran(time, redshift, loge0, mej, logn0, logepse, logepsb, p, **kwargs):
    """
    A kilonova afterglow model based on Nakar & Piran 2011

    :param time: time in days in the observer frame
    :param redshift: source redshift
    :param loge0: initial kinetic energy in erg of ejecta
    :param mej: mass of ejecta in solar masses
    :param logn0: log10 of the number density of the circumburst medium in cm^-3
    :param logepse: log10 of the fraction of energy given to electrons
    :param logepsb: log10 of the fraction of energy given to the magnetic field
    :param p: electron power law index
    :param kwargs: Additional keyword arguments
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    :return:
    """
    Eej = 10 ** loge0
    Mej = mej * solar_mass
    Gamma0 = 1.0 + Eej / (Mej * speed_of_light ** 2)
    vej = speed_of_light * (1.0 - Gamma0 ** (-2)) ** 0.5
    cosmology = kwargs.get('cosmology', cosmo)
    epsilon_e = 10 ** logepse
    epsilon_B = 10 ** logepsb
    n0 = 10 ** logn0
    dl = cosmology.luminosity_distance(redshift).cgs.value

    # in days
    t_dec = 30 * (Eej / 1e49) ** (1.0 / 3.0) * (n0 / 1e0) ** (-1.0 / 3.0) * (vej / speed_of_light) ** (-5.0 / 3.0)

    fnu_dec_dict = {}
    fnu_func = {}
    temp_time = np.linspace(0.1, 100, 200) * t_dec
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    for freq in frequency:
        # Eq. 11 in Nakar & Piran 2011 (in Mjy)
        fnu_dec_dict[freq] = 0.3 * (Eej / 1e49) * n0 ** (0.25 * (p + 1)) * (epsilon_B / 1e-1) ** (0.25 * (p + 1)) * (
                epsilon_e / 1e-1) ** (p - 1) * (vej / speed_of_light) ** (0.5 * (5 * p - 7)) * (freq / 1.4e9) ** (
                          -0.5 * (p - 1)) * (dl / 1e27) ** (-2)
        fnu = fnu_dec_dict[freq] * (temp_time / t_dec) ** 3
        fnu[temp_time > t_dec] = fnu_dec_dict[freq] * (temp_time[temp_time > t_dec] / t_dec) ** (-0.3 * (5 * p - 7))
        fnu_func[freq] = interp1d(temp_time, fnu, bounds_error=False, fill_value='extrapolate')

    # interpolate onto actual observed frequency and time values
    flux_density = []
    for freq, tt in zip(frequency, time):
        flux_density.append(fnu_func[freq](tt))
    fmjy = flux_density * uu.mJy
    if kwargs['output_format'] == 'flux_density':
        return fmjy.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy.value).value
        
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def thermal_synchrotron_lnu(time, logn0, v0, logr0, eta, logepse, logepsb, xi, p, **kwargs):
    """
    :param time: time in source frame in seconds
    :param logn0: log10 initial ambient ism density
    :param v0: initial velocity in c
    :param logr0: log10 initial radius
    :param eta: deceleration slope (r = r0 * (time/t0)**eta; v = v0*(time/t0)**(eta-1))
    :param logepse: log10 epsilon_e; electron thermalisation efficiency
    :param logepsb: log10 epsilon_b; magnetic field amplification efficiency
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param kwargs: extra parameters to change physics/settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param wind_slope: slope for ism density scaling (nism = n0 * (r/r0)**(-wind_slope)). Default is 2
    :param mu: mean molecular weight, default is 0.62
    :param mu_e: mean molecular weight per electron, default is 1.18
    :return: lnu
    """
    v0 = v0 * speed_of_light
    r0 = 10**logr0
    t0 = eta * r0 / v0
    radius = r0 * (time / t0) ** eta
    velocity = v0 * (time/t0)**(eta - 1)
    wind_slope = kwargs.get('wind_slope',2)
    mu = kwargs.get('mu', 0.62)
    mu_e = kwargs.get('mu_e', 1.18)
    n0 = 10 ** logn0
    nism = n0 * (radius / r0) ** (-wind_slope)

    epsilon_T = 10**logepse
    epsilon_B = 10**logepsb

    frequency = kwargs['frequency']

    ne = 4.0*mu_e*nism
    beta = velocity/speed_of_light

    theta0 = epsilon_T * (9.0 * mu * proton_mass / (32.0 * mu_e * electron_mass)) * beta ** 2
    theta = (5.0*theta0-6.0+(25.0*theta0**2+180.0*theta0+36.0)**0.5)/30.0

    bfield = (9.0*np.pi*epsilon_B*nism*mu*proton_mass)**0.5*velocity
    # mean dynamical time:
    td = radius/velocity

    z_cool = (6.0 * np.pi * electron_mass * speed_of_light / (sigma_T * bfield ** 2 * td)) / theta
    normalised_frequency_denom = 3.0*theta**2*qe*bfield/(4.0*np.pi*electron_mass*speed_of_light)
    x = frequency / normalised_frequency_denom

    emissivity_pl = _emissivity_pl(x=x, nism=ne, bfield=bfield, theta=theta, xi=xi, p=p, z_cool=z_cool)

    emissivity_thermal = _emissivity_thermal(x=x, nism=ne, bfield=bfield, theta=theta, z_cool=z_cool)

    emissivity = emissivity_thermal + emissivity_pl

    tau = _tau_nu(x=x, nism=ne, radius=radius, bfield=bfield, theta=theta, xi=xi, p=p, z_cool=z_cool)

    lnu = 4.0 * np.pi ** 2 * radius ** 3 * emissivity * (1e0 - np.exp(-tau)) / tau
    if np.size(x) > 1:
        lnu[tau < 1e-10] = (4.0 * np.pi ** 2 * radius ** 3 * emissivity)[tau < 1e-10]
    elif tau < 1e-10:
        lnu = 4.0 * np.pi ** 2 * radius ** 3 * emissivity
    return lnu

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def thermal_synchrotron_fluxdensity(time, redshift, logn0, v0, logr0, eta, logepse, logepsb,
                                    xi, p, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param logn0: log10 initial ambient ism density
    :param v0: initial velocity in c
    :param logr0: log10 initial radius
    :param eta: deceleration slope (r = r0 * (time/t0)**eta; v = v0*(time/t0)**(eta-1))
    :param logepse: log10 epsilon_e; electron thermalisation efficiency
    :param logepsb: log10 epsilon_b; magnetic field amplification efficiency
    :param xi: fraction of energy carried by power law electrons
    :param p: electron power law slope
    :param kwargs: extra parameters to change physics/settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param wind_slope: slope for ism density scaling (nism = n0 * (r/r0)**(-wind_slope)). Default is 2
    :param mu: mean molecular weight, default is 0.62
    :param mu_e: mean molecular weight per electron, default is 1.18
    :param kwargs: extra parameters to change physics and other settings
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density
    """
    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    new_kwargs = kwargs.copy()
    new_kwargs['frequency'] = frequency
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    lnu = thermal_synchrotron_lnu(time,logn0, v0, logr0, eta, logepse, logepsb, xi, p,**new_kwargs)
    flux_density = lnu / (4.0 * np.pi * dl**2)
    return flux_density
    
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5328G/abstract')
def tde_synchrotron(time, redshift, Mej, vej, logepse, logepsb, p, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param Mej: mass of the emitting region (solar masses)
    :param vej: initial velocity of the outflow (km/s)
    :param logepse: log10 epsilon_e; electron thermalisation efficiency
    :param logepsb: log10 epsilon_b; magnetic field amplification efficiency
    :param p: electron power law slope
    :param kwargs: extra parameters to change physics/settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param geometry: geometry of the outflow.  Either "sphere" or "cone" is supported.
    :param output_format: Whether to output light curves or physical parameters.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density
    """
    frequency = kwargs['frequency']
    geometry = kwargs.get('geometry', 'sphere')
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    beta1 = 5.0 / 2.0
    beta2 = (1.0 - p) / 2.0
    s = 1.47 - (0.21 * p)
    eps_e = 10.0 ** logepse
    eps_b = 10.0 ** logepsb
    if geometry == 'cone':
        F_A = 0.13
        F_V = 1.15
    else:    
        F_A = 1.0
        F_V = 4.0/3.0

    beta = vej * km_cgs / speed_of_light
    E = Mej * solar_mass * (beta * speed_of_light) ** 2.0 / 2.0
    R = time * day_to_s * beta * speed_of_light / ((1 - beta) * (1 + redshift))
    eps = 11.0 * eps_b / (6.0 * eps_e)
    R_eq = R / eps ** (1.0 / 17.0)
    E_eq = E / ((11.0 / 17.0) * eps ** (-6.0 / 17.0) + (6.0 / 17.0) * eps ** (11.0 / 17.0))
    chi_e = 2.0
    xi = 1.0 + (1.0 / eps_e)
    
    R_prefac = (1e17 * (21.8 * 525.0 ** (p - 1.0)) ** (1.0 / (13.0 + 2.0 * p))
            * chi_e ** ((2.0 - p) / (13.0 + 2.0 * p))
            * xi ** (1.0 / (13.0 + 2.0 * p))
            * (dl / 1.0e28) ** (2.0 * (p + 6.0) / (13.0 + 2.0 * p))
            * (1.0 + redshift) ** (-(19.0 + 3.0 * p) / (13.0 + 2.0 * p))
            * F_A ** (-(5.0 + p) / (13.0 + 2.0 * p))
            * F_V ** (-1.0 / (13.0 + 2.0 * p))
            * 4.0 ** (1.0 / (13.0 + 2.0 * p)))           
    E_prefac = (1.3e48 * 21.8 ** ((-2.0 * (p + 1.0)) / (13.0 + 2.0 * p))
            * (525 ** (p - 1.0) * chi_e ** (2.0 - p)) ** (11.0 / (13.0 + 2.0 * p))
            * xi ** (11.0 / (13.0 + 2.0 * p))
            * (dl / 1.0e28) ** (2.0 * (3.0 * p + 14.0) / (13.0 + 2.0 * p))
            * (1.0 + redshift) ** ((-27.0 + 5.0 * p) / (13.0 + 2.0 * p))
            * F_A ** (-(3.0 * (p + 1.0)) / (13.0 + 2.0 * p))
            * F_V ** ((2.0 * (p + 1.0)) / (13.0 + 2.0 * p))
            * 4.0 ** (11.0 / (13.0 + 2.0 * p)))
            
    Fvb = (E_prefac * R_eq / (R_prefac * E_eq)) ** ((2.0 * (p + 4.0)) / (13.0 + 2.0 * p))
    vb = (R_prefac * Fvb ** ((p + 6.0) / (13.0 + 2.0 * p)) / R_eq) * 1.0e10
    
    if kwargs['output_format'] == 'physical_parameters':
        physical_parameters = namedtuple('physical_parameters', ['E', 'R', 'N_e', 'n_e', 'B'])
        gamma_m = 2.0
        gamma_a = (525.0 * Fvb * (dl / 1.0e28) ** 2.0 * (1.0 + redshift) ** -3.0 
                * (vb / 1.0e10) ** -2.0 / (F_A * (R / 1.0e17) ** 2.0))
        N_e = (4.0e54 * Fvb ** 3.0 * (dl / 1.0e28) ** 6.0 * (vb / 1.0e10) ** -5.0
                * (1.0 + redshift) ** -8.0 * F_A ** -2.0 * (R / 1.0e17) ** -4.0
                * (gamma_m / gamma_a) ** (1.0 - p))
        n_e = N_e / (4.0 /3.0 * np.pi * R ** 3.0)
        B = (1.3e-2 * Fvb ** -2.0 * (dl / 1.0e28) ** -4.0 * (vb / 1.0e10) ** 5.0
                * (1.0 + redshift) ** 7.0 * F_A ** 2.0 * (R / 1.0e17) ** 4.0)
        physical_parameters.E = E
        physical_parameters.R = R
        physical_parameters.N_e = N_e
        physical_parameters.n_e = n_e
        physical_parameters.B = B    
        return physical_parameters    
    else:        
        flux_density = Fvb * ((frequency / vb) ** (-beta1 * s) + (frequency / vb) ** (-beta2 * s)) ** (-1.0 / s)
        return flux_density    

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2007ihea.book.....R/abstract, https://ui.adsabs.harvard.edu/abs/2017hsn..book..875C/abstract')
def synchrotron_massloss(time, redshift, v_s, log_Mdot_vwind, logepsb, logepse, p, **kwargs):
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param v_s: velocity of the shock (km/s)
    :param log Mdot_vwind: log10 of the mass loss rate over wind velocity ((solar mass / year)/(km / s))
    :param logepse: log10 epsilon_e; electron thermalisation efficiency
    :param logepsb: log10 epsilon_b; magnetic field amplification efficiency
    :param p: electron power law slope
    :param kwargs: extra parameters to change physics/settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)   

    eps_e = 10.0 ** logepse
    eps_b = 10.0 ** logepsb
    v_cgs = v_s * km_cgs
    t_cgs = time * day_to_s
    r_s = v_cgs * t_cgs
    Mdot_vwind = 10.0 ** log_Mdot_vwind    
    Md_vw_cgs = Mdot_vwind * solar_mass / (day_to_s * 365.24) / km_cgs #g/cm
    nu_0 = 1.253e19
   
    rho_csm = Md_vw_cgs / (4.0 * np.pi * r_s ** 2.0)
    u_b = eps_b * rho_csm * v_cgs ** 2.0
    B = np.sqrt(8 * np.pi * u_b)
    N_0 = 4.0 / 3.0 * np.pi * r_s ** 3.0 * eps_e * rho_csm / proton_mass
    C_0 = 4.0 / 3.0 * N_0 * sigma_T * speed_of_light * u_b 
    nu_L = qe * B / (2 * np.pi * electron_mass * speed_of_light)

    L_nu = C_0 / (2.0 * nu_L) * (frequency / nu_L) ** ((1.0 - p) / 2.0) 
    flux_density = L_nu / (4.0 * np.pi * dl**2) / 1.0e-26

    Fv0 = C_0 / (2.0 * nu_L) / (4.0 * np.pi * dl**2)
    beta = 1.0 - (1.0 - p) / 2.0
    nu_ssa = (dl**2 * 3.0**1.5 * qe**0.5 * B**0.5 * Fv0 * nu_L**(beta - 1.0) / (4.0 * np.pi**1.5 * r_s**2 * speed_of_light**0.5 * electron_mass**1.5))**(2.0 / (2.0 * beta + 3.0))
    Fv_ssa = Fv0 * (nu_ssa / nu_L) ** (1-beta)

    if (np.min(frequency) < np.max(nu_ssa)):
        msk = (frequency < nu_ssa)
        flux_density[msk] = Fv_ssa[msk] * (frequency[msk] / nu_ssa[msk]) ** 2.5 / 1.0e-26 
        
    return flux_density    

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2007ihea.book.....R/abstract, https://ui.adsabs.harvard.edu/abs/2017hsn..book..875C/abstract')    
def synchrotron_ism(time, redshift, v_s, logn0, logepsb, logepse, p, **kwargs):  
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param v_s: velocity of the shock (km/s)
    :param logn0: log10 of the circumburst density (cm^-3)
    :param logepse: log10 epsilon_e; electron thermalisation efficiency
    :param logepsb: log10 epsilon_b; magnetic field amplification efficiency
    :param p: electron power law slope
    :param kwargs: extra parameters to change physics/settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

    n_ism = 10.0 ** logn0
    eps_e = 10.0 ** logepse
    eps_b = 10.0 ** logepsb
    v_cgs = v_s * km_cgs
    t_cgs = time * day_to_s
    r_s = v_cgs * t_cgs
    nu_0 = 1.253e19

    rho_csm = proton_mass * n_ism
    u_b = eps_b * rho_csm * v_cgs ** 2.0 
    B = np.sqrt(8 * np.pi * u_b) 
    N_0 = 4.0 / 3.0 * np.pi * r_s ** 3.0 * eps_e * n_ism 
    C_0 = 4.0 / 3.0 * N_0 * sigma_T * speed_of_light * u_b 
    nu_L = qe * B / (2 * np.pi * electron_mass * speed_of_light)

    L_nu = C_0 / (2.0 * nu_L) * (frequency / nu_L) ** ((1.0 - p) / 2.0)
    flux_density = L_nu / (4.0 * np.pi * dl**2) / 1.0e-26

    Fv0 = C_0 / (2.0 * nu_L) / (4.0 * np.pi * dl**2)
    beta = 1.0 - (1.0 - p) / 2.0
    nu_ssa = (dl**2 * 3.0**1.5 * qe**0.5 * B**0.5 * Fv0 * nu_L**(beta - 1.0) / (4.0 * np.pi**1.5 * r_s**2 * speed_of_light**0.5 * electron_mass**1.5))**(2.0 / (2.0 * beta + 3.0))
    Fv_ssa = Fv0 * (nu_ssa / nu_L) ** (1-beta)

    if (np.min(frequency) < np.max(nu_ssa)):
        msk = (frequency < nu_ssa)
        flux_density[msk] = Fv_ssa[msk] * (frequency[msk] / nu_ssa[msk]) ** 2.5 / 1.0e-26 
        
    return flux_density   

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2007ihea.book.....R/abstract, https://ui.adsabs.harvard.edu/abs/2017hsn..book..875C/abstract')    
def synchrotron_pldensity(time, redshift, v_s, logA, s, logepsb, logepse, p, **kwargs):    
    """
    :param time: time in observer frame in days
    :param redshift: redshift
    :param v_s: velocity of the shock (km/s)
    :param logA: log10 of the circumstellar material density at R=1e15 cm (cm^-3)
    :param s: power law index of the circumstellar material density profile
    :param logepse: log10 epsilon_e; electron thermalisation efficiency
    :param logepsb: log10 epsilon_b; magnetic field amplification efficiency
    :param p: electron power law slope
    :param kwargs: extra parameters to change physics/settings
    :param frequency: frequency to calculate model on - Must be same length as time array or a single number)
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density
    """
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

    A = 10.0 ** logA
    eps_e = 10.0 ** logepse
    eps_b = 10.0 ** logepsb
    v_cgs = v_s * km_cgs
    t_cgs = time * day_to_s
    r_s = v_cgs * t_cgs
    nu_0 = 1.253e19    

    n_csm = A * (r_s / 1e15) **(-s)
    rho_csm = n_csm * proton_mass
    u_b = eps_b * rho_csm * v_cgs ** 2.0
    B = np.sqrt(8 * np.pi * u_b) 
    N_0 = 4.0 / 3.0 * np.pi * r_s ** 3.0 * eps_e * n_csm
    C_0 = 4.0 / 3.0 * N_0 * sigma_T * speed_of_light * u_b
    nu_L = qe * B / (2 * np.pi * electron_mass * speed_of_light)

    L_nu = C_0 / (2.0 * nu_L) * (frequency / nu_L) ** ((1.0 - p) / 2.0)
    flux_density = L_nu / (4.0 * np.pi * dl**2) / 1.0e-26

    Fv0 = C_0 / (2.0 * nu_L) / (4.0 * np.pi * dl**2)
    beta = 1.0 - (1.0 - p) / 2.0
    nu_ssa = (dl**2 * 3.0**1.5 * qe**0.5 * B**0.5 * Fv0 * nu_L**(beta - 1.0) / (4.0 * np.pi**1.5 * r_s**2 * speed_of_light**0.5 * electron_mass**1.5))**(2.0 / (2.0 * beta + 3.0))
    Fv_ssa = Fv0 * (nu_ssa / nu_L) ** (1-beta)

    if (np.min(frequency) < np.max(nu_ssa)):
        msk = (frequency < nu_ssa)
        flux_density[msk] = Fv_ssa[msk] * (frequency[msk] / nu_ssa[msk]) ** 2.5 / 1.0e-26 
        
    return flux_density       