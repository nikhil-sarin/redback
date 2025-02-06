import numpy as np
import pandas as pd

from astropy.table import Table, Column
from scipy.interpolate import interp1d, RegularGridInterpolator
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.integrate import cumulative_trapezoid
from collections import namedtuple
from redback.photosphere import TemperatureFloor, CocoonPhotosphere
from redback.interaction_processes import Diffusion, AsphericalDiffusion

from redback.utils import calc_kcorrected_properties, interpolated_barnes_and_kasen_thermalisation_efficiency, \
    electron_fraction_from_kappa, citation_wrapper, lambda_to_nu, _calculate_rosswogkorobkin24_qdot, \
    kappa_from_electron_fraction
from redback.eos import PiecewisePolytrope
from redback.sed import blackbody_to_flux_density, get_correct_output_format_from_spectra, Blackbody
from redback.constants import *
import redback.ejecta_relations as ejr

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract')
def _nicholl_bns_get_quantities(mass_1, mass_2, lambda_s, kappa_red, kappa_blue,
                                mtov, epsilon, alpha, cos_theta_open, cos_theta, **kwargs):
    """
    Calculates quantities for the Nicholl et al. 2021 BNS model

    :param mass_1: Mass of primary in solar masses
    :param mass_2: Mass of secondary in solar masses
    :param lambda_s: Symmetric tidal deformability i.e, lambda_s = lambda_1 + lambda_2
    :param kappa_red: opacity of the red ejecta
    :param kappa_blue: opacity of the blue ejecta
    :param mtov: Tolman Oppenheimer-Volkoff mass in solar masses
    :param epsilon: fraction of disk that gets unbound/ejected
    :param alpha: Enhancement of blue ejecta by NS surface winds if mtotal < prompt collapse,
                can turn off by setting alpha=1
    :param cos_theta_open: Lanthanide opening angle 
    :param cos_theta: Viewing angle of observer
    :param kwargs: Additional keyword arguments
    :param dynamical_ejecta_error: Error in dynamical ejecta mass, default is 1 i.e., no error in fitting formula
    :param disk_ejecta_error: Error in disk ejecta mass, default is 1 i.e., no error in fitting formula
    :return: Namedtuple with 'mejecta_blue', 'mejecta_red', 'mejecta_purple',
            'vejecta_blue', 'vejecta_red', 'vejecta_purple',
            'vejecta_mean', 'kappa_mean', 'mejecta_dyn',
            'mejecta_total', 'kappa_purple', 'radius_1', 'radius_2',
            'binary_lambda', 'remnant_radius', 'area_blue', 'area_blue_ref',
            'area_red', 'area_red_ref' properties. Masses in solar masses and velocities in units of c
    """
    ckm = 3e10/1e5
    a = 0.07550
    b = np.array([[-2.235, 0.8474], [10.45, -3.251], [-15.70, 13.61]])
    c = np.array([[-2.048, 0.5976], [7.941, 0.5658], [-7.360, -1.320]])
    n_ave = 0.743
    dynamical_ejecta_error = kwargs.get('dynamical_ejecta_error', 1.0)
    disk_ejecta_error = kwargs.get('disk_ejecta_error', 1.0)
    theta_open = np.arccos(cos_theta_open)

    fq = (1 - (mass_2 / mass_1) ** (10. / (3 - n_ave))) / (1 + (mass_2 / mass_1) ** (10. / (3 - n_ave)))

    nume = a
    denom = a

    for i in np.arange(3):
        for j in np.arange(2):
            nume += b[i, j] * (mass_2 / mass_1) ** (j + 1) * lambda_s ** (-(i + 1) / 5)

    for i in np.arange(3):
        for j in np.arange(2):
            denom += c[i, j] * (mass_2 / mass_1) ** (j + 1) * lambda_s ** (-(i + 1) / 5)

    lambda_a = lambda_s * fq * nume / denom
    lambda_1 = lambda_s - lambda_a
    lambda_2 = lambda_s + lambda_a
    m_total = mass_1 + mass_2

    binary_lambda =  16./13 * ((mass_1 + 12*mass_2) * mass_1**4 * lambda_1 +
                (mass_2 + 12*mass_1) * mass_2**4 * lambda_2) / m_total**5
    mchirp = (mass_1 * mass_2)**(3./5) / m_total**(1./5)
    remnant_radius = 11.2 * mchirp * (binary_lambda/800)**(1./6.)

    compactness_1 = 0.360 - 0.0355 * np.log(lambda_1) + 0.000705 * np.log(lambda_1) ** 2
    compactness_2 = 0.360 - 0.0355 * np.log(lambda_2) + 0.000705 * np.log(lambda_2) ** 2

    radius_1 = (graviational_constant * mass_1 * solar_mass / (compactness_1 * speed_of_light ** 2)) / 1e5
    radius_2 = (graviational_constant * mass_2 * solar_mass / (compactness_2 * speed_of_light ** 2)) / 1e5

    # Baryonic masses, Gao 2019
    mass_baryonic_1 = mass_1 + 0.08 * mass_1 ** 2
    mass_baryonic_2 = mass_2 + 0.08 * mass_2 ** 2

    a_1 = -1.35695
    b_1 = 6.11252
    c_1 = -49.43355
    d_1 = 16.1144
    n = -2.5484
    dynamical_ejecta_mass = 1e-3 * (a_1 * ((mass_2 / mass_1) ** (1 / 3) * (1 - 2 * compactness_1) / compactness_2 * mass_baryonic_1 +
                            (mass_1 / mass_2) ** (1 / 3) * (1 - 2 * compactness_2) / compactness_2 * mass_baryonic_2) +
                     b_1 * ((mass_2 / mass_1) ** n * mass_baryonic_1 + (mass_1 / mass_2) ** n * mass_baryonic_2) +
                     c_1 * (mass_baryonic_1 - mass_1 + mass_baryonic_2 - mass_2) + d_1)

    if dynamical_ejecta_mass < 0:
        dynamical_ejecta_mass = 0

    dynamical_ejecta_mass *= dynamical_ejecta_error

    a_4 = 14.8609
    b_4 = -28.6148
    c_4 = 13.9597

    # fraction can't exceed 100%
    f_red = min([a_4 * (mass_1 / mass_2) ** 2 + b_4 * (mass_1 / mass_2) + c_4, 1])

    mejecta_red = dynamical_ejecta_mass * f_red
    mejecta_blue = dynamical_ejecta_mass * (1 - f_red)

    # Velocity of dynamical ejecta
    a_2 = -0.219479
    b_2 = 0.444836
    c_2 = -2.67385

    vdynp = a_2 * ((mass_1 / mass_2) * (1 + c_2 * compactness_1) + (mass_2 / mass_1) * (1 + c_2 * compactness_2)) + b_2

    a_3 = -0.315585
    b_3 = 0.63808
    c_3 = -1.00757

    vdynz = a_3 * ((mass_1 / mass_2) * (1 + c_3 * compactness_1) + (mass_2 / mass_1) * (1 + c_3 * compactness_2)) + b_3

    dynamical_ejecta_velocity = np.sqrt(vdynp ** 2 + vdynz ** 2)

    # average velocity over angular ranges (< and > theta_open)

    theta1 = np.arange(0, theta_open, 0.01)
    theta2 = np.arange(theta_open, np.pi / 2, 0.01)

    vtheta1 = np.sqrt((vdynz * np.cos(theta1)) ** 2 + (vdynp * np.sin(theta1)) ** 2)
    vtheta2 = np.sqrt((vdynz * np.cos(theta2)) ** 2 + (vdynp * np.sin(theta2)) ** 2)

    atheta1 = 2 * np.pi * np.sin(theta1)
    atheta2 = 2 * np.pi * np.sin(theta2)

    vejecta_blue = np.trapz(vtheta1 * atheta1, x=theta1) / np.trapz(atheta1, x=theta1)
    vejecta_red = np.trapz(vtheta2 * atheta2, x=theta2) / np.trapz(atheta2, x=theta2)

    # vejecta_red *= ckm
    # vejecta_blue *= ckm

    # Bauswein 2013, cut-off for prompt collapse to BH
    prompt_threshold_mass = (2.38 - 3.606 * mtov / remnant_radius) * mtov

    if m_total < prompt_threshold_mass:
        mejecta_blue /= alpha

    # Now compute disk ejecta following Coughlin+ 2019

    a_5 = -31.335
    b_5 = -0.9760
    c_5 = 1.0474
    d_5 = 0.05957

    logMdisk = np.max([-3, a_5 * (1 + b_5 * np.tanh((c_5 - m_total / prompt_threshold_mass) / d_5))])

    disk_mass = 10 ** logMdisk

    disk_mass *= disk_ejecta_error

    disk_ejecta_mass = disk_mass * epsilon

    mejecta_purple = disk_ejecta_mass

    # Fit for disk velocity using Metzger and Fernandez
    vdisk_max = 0.15
    vdisk_min = 0.03
    vfit = np.polyfit([mtov, prompt_threshold_mass], [vdisk_max, vdisk_min], deg=1)

    # Get average opacity of 'purple' (disk) component
    # Mass-averaged Ye as a function of remnant lifetime from Lippuner 2017
    # Lifetime related to Mtot using Metzger handbook table 3
    if m_total < mtov:
        # stable NS
        Ye = 0.38
        vdisk = vdisk_max
    elif m_total < 1.2 * mtov:
        # long-lived (>>100 ms) NS remnant Ye = 0.34-0.38,
        # smooth interpolation
        Yfit = np.polyfit([mtov, 1.2 * mtov], [0.38, 0.34], deg=1)
        Ye = Yfit[0] * m_total + Yfit[1]
        vdisk = vfit[0] * m_total + vfit[1]
    elif m_total < prompt_threshold_mass:
        # short-lived (hypermassive) NS, Ye = 0.25-0.34, smooth interpolation
        Yfit = np.polyfit([1.2 * mtov, prompt_threshold_mass], [0.34, 0.25], deg=1)
        Ye = Yfit[0] * m_total + Yfit[1]
        vdisk = vfit[0] * m_total + vfit[1]
    else:
        # prompt collapse to BH, disk is red
        Ye = 0.25
        vdisk = vdisk_min

    # Convert Ye to opacity using Tanaka et al 2019 for Ye >= 0.25:
    a_6 = 2112.0
    b_6 = -2238.9
    c_6 = 742.35
    d_6 = -73.14

    kappa_purple = a_6 * Ye ** 3 + b_6 * Ye ** 2 + c_6 * Ye + d_6

    vejecta_purple = vdisk

    vejecta_mean = (mejecta_purple * vejecta_purple + vejecta_red * mejecta_red +
                    vejecta_blue * mejecta_blue) / (mejecta_purple + mejecta_red + mejecta_blue)

    kappa_mean = (mejecta_purple * kappa_purple + kappa_red * mejecta_red +
                  kappa_blue * mejecta_blue) / (mejecta_purple + mejecta_red + mejecta_blue)

    # Viewing angle and lanthanide-poor opening angle correction from Darbha and Kasen 2020
    ct = (1 - cos_theta_open ** 2) ** 0.5

    if cos_theta > ct:
        area_projected_top = np.pi * ct * cos_theta
    else:
        theta_p = np.arccos(cos_theta_open /
                            (1 - cos_theta ** 2) ** 0.5)
        theta_d = np.arctan(np.sin(theta_p) / cos_theta_open *
                            (1 - cos_theta ** 2) ** 0.5 / np.abs(cos_theta))
        area_projected_top = (theta_p - np.sin(theta_p) * np.cos(theta_p)) - (ct *
                                                                     cos_theta * (theta_d - np.sin(theta_d) * np.cos(
                            theta_d) - np.pi))

    minus_cos_theta = -1 * cos_theta

    if minus_cos_theta < -1 * ct:
        area_projected_bottom = 0
    else:
        theta_p2 = np.arccos(cos_theta_open /
                             (1 - minus_cos_theta ** 2) ** 0.5)
        theta_d2 = np.arctan(np.sin(theta_p2) / cos_theta_open *
                             (1 - minus_cos_theta ** 2) ** 0.5 / np.abs(minus_cos_theta))

        Aproj_bot1 = (theta_p2 - np.sin(theta_p2) * np.cos(theta_p2)) + (ct *
                                                                         minus_cos_theta * (theta_d2 - np.sin(
                    theta_d2) * np.cos(theta_d2)))
        area_projected_bottom = np.max([Aproj_bot1, 0])

    area_projected = area_projected_top + area_projected_bottom

    # Compute reference areas for this opening angle to scale luminosity

    cos_theta_ref = 0.5

    if cos_theta_ref > ct:
        area_ref_top = np.pi * ct * cos_theta_ref
    else:
        theta_p_ref = np.arccos(cos_theta_open /
                                (1 - cos_theta_ref ** 2) ** 0.5)
        theta_d_ref = np.arctan(np.sin(theta_p_ref) / cos_theta_open *
                                (1 - cos_theta_ref ** 2) ** 0.5 / np.abs(cos_theta_ref))
        area_ref_top = (theta_p_ref - np.sin(theta_p_ref) *
                    np.cos(theta_p_ref)) - (ct * cos_theta_ref *
                                            (theta_d_ref - np.sin(theta_d_ref) *
                                             np.cos(theta_d_ref) - np.pi))

    minus_cos_theta_ref = -1 * cos_theta_ref

    if minus_cos_theta_ref < -1 * ct:
        area_ref_bottom = 0
    else:
        theta_p2_ref = np.arccos(cos_theta_open /
                                 (1 - minus_cos_theta_ref ** 2) ** 0.5)
        theta_d2_ref = np.arctan(np.sin(theta_p2_ref) /
                                 cos_theta_open * (1 - minus_cos_theta_ref ** 2) ** 0.5 /
                                 np.abs(minus_cos_theta_ref))

        area_ref_bottom = (theta_p2_ref - np.sin(theta_p2_ref) *
                    np.cos(theta_p2_ref)) + (ct * minus_cos_theta_ref *
                                             (theta_d2_ref - np.sin(theta_d2_ref) *
                                              np.cos(theta_d2_ref)))

    area_ref = area_ref_top + area_ref_bottom

    area_blue = area_projected
    area_blue_ref = area_ref

    area_red = np.pi - area_blue
    area_red_ref = np.pi - area_blue_ref

    output = namedtuple('output', ['mejecta_blue', 'mejecta_red', 'mejecta_purple',
                                   'vejecta_blue', 'vejecta_red', 'vejecta_purple',
                                   'vejecta_mean', 'kappa_mean', 'mejecta_dyn',
                                   'mejecta_total', 'kappa_purple', 'radius_1', 'radius_2',
                                   'binary_lambda', 'remnant_radius', 'area_blue', 'area_blue_ref',
                                   'area_red', 'area_red_ref'])
    output.mejecta_blue = mejecta_blue
    output.mejecta_red = mejecta_red
    output.mejecta_purple = mejecta_purple
    output.vejecta_blue = vejecta_blue
    output.vejecta_red = vejecta_red
    output.vejecta_purple = vejecta_purple
    output.vejecta_mean = vejecta_mean
    output.kappa_mean = kappa_mean
    output.mejecta_dyn = dynamical_ejecta_mass
    output.mejecta_total = dynamical_ejecta_mass + mejecta_purple
    output.kappa_purple = kappa_purple
    output.radius_1 = radius_1
    output.radius_2 = radius_2
    output.binary_lambda = binary_lambda
    output.remnant_radius = remnant_radius
    output.area_blue = area_blue
    output.area_blue_ref = area_blue_ref
    output.area_red = area_red
    output.area_red_ref = area_red_ref
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract')
def nicholl_bns(time, redshift, mass_1, mass_2, lambda_s, kappa_red, kappa_blue,
                mtov, epsilon, alpha, cos_theta, cos_theta_open, cos_theta_cocoon, temperature_floor_1,
                temperature_floor_2, temperature_floor_3, **kwargs):
    """
    Kilonova model from Nicholl et al. 2021, inclides three kilonova components
    + shock heating from cocoon + disk winds from remnant

    :param time: time in days in observer frame
    :param redshift: redshift
    :param mass_1: Mass of primary in solar masses
    :param mass_2: Mass of secondary in solar masses
    :param lambda_s: Symmetric tidal deformability i.e, lambda_s = lambda_1 + lambda_2
    :param kappa_red: opacity of the red ejecta
    :param kappa_blue: opacity of the blue ejecta
    :param mtov: Tolman Oppenheimer-Volkoff mass in solar masses
    :param epsilon: fraction of disk that gets unbound/ejected
    :param alpha: Enhancement of blue ejecta by NS surface winds if mtotal < prompt collapse,
                can turn off by setting alpha=1
    :param cos_theta: Viewing angle of observer
    :param cos_theta_open: Lanthanide opening angle
    :param cos_theta_cocoon: Opening angle of shocked cocoon
    :param temperature_floor_1: Temperature floor of first (blue) component
    :param temperature_floor_2: Temperature floor of second (purple) component
    :param temperature_floor_3: Temperature floor of third (red) component
    :param kwargs: Additional keyword arguments
    :param dynamical_ejecta_error: Error in dynamical ejecta mass, default is 1 i.e., no error in fitting formula
    :param disk_ejecta_error: Error in disk ejecta mass, default is 1 i.e., no error in fitting formula
    :param shocked_fraction: Fraction of ejecta that is shocked by jet, default is 0.2 i.e., 20% of blue ejecta is shocked.
        Use 0. if you want to turn off cocoon emission.
    :param nn: ejecta power law density profile, default is 1.
    :param tshock: time for shock in source frame in seconds, default is 1.7s (see Nicholl et al. 2021)
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param dense_resolution: resolution of the grid that the model is actually evaluated on, default is 300
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from redback.transient_models.shock_powered_models import _shocked_cocoon_nicholl
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    dense_resolution = kwargs.get('dense_resolution', 100)
    time_temp = np.geomspace(0.01, 30, dense_resolution)  # in source frame and days
    kappa_gamma = kwargs.get('kappa_gamma', 10)
    ckm = 3e10/1e5

    if np.max(time) > 20: # in source frame and days
        time_temp = np.geomspace(0.01, np.max(time) + 5, dense_resolution)

    time_obs = time
    shocked_fraction = kwargs.get('shocked_fraction', 0.2)
    nn = kwargs.get('nn', 1)
    tshock = kwargs.get('tshock', 1.7)

    output = _nicholl_bns_get_quantities(mass_1=mass_1, mass_2=mass_2, lambda_s=lambda_s,
                                         kappa_red=kappa_red, kappa_blue=kappa_blue, mtov=mtov,
                                         epsilon=epsilon, alpha=alpha, cos_theta_open=cos_theta_open, 
                                         cos_theta=cos_theta, **kwargs)
    cocoon_output = _shocked_cocoon_nicholl(time=time_temp, kappa=kappa_blue, mejecta=output.mejecta_blue,
                                  vejecta=output.vejecta_blue, cos_theta_cocoon=cos_theta_cocoon,
                                  shocked_fraction=shocked_fraction, nn=nn, tshock=tshock)
    cocoon_photo = CocoonPhotosphere(time=time_temp, luminosity=cocoon_output.lbol,
                                     tau_diff=cocoon_output.taudiff, t_thin=cocoon_output.tthin,
                                     vej=output.vejecta_blue*ckm, nn=nn)
    mejs = [output.mejecta_blue, output.mejecta_purple, output.mejecta_red]
    vejs = [output.vejecta_blue, output.vejecta_purple, output.vejecta_red]
    area_projs = [output.area_blue, output.area_blue, output.area_red]
    area_refs = [output.area_blue_ref, output.area_blue_ref, output.area_red_ref]
    temperature_floors = [temperature_floor_1, temperature_floor_2, temperature_floor_3]
    kappas = [kappa_blue, output.kappa_purple, kappa_red]

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=cocoon_photo.photosphere_temperature)
        rad_func = interp1d(time_temp, y=cocoon_photo.r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        temp = temp_func(time)
        photosphere = rad_func(time)
        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)
        ff = flux_density.value
        ff = np.nan_to_num(ff)
        for x in range(3):
            lbols = _mosfit_kilonova_one_component_lbol(time=time_temp*day_to_s, mej=mejs[x], vej=vejs[x])
            interaction_class = AsphericalDiffusion(time=time_temp, dense_times=time_temp,
                                                    luminosity=lbols, kappa=kappas[x], kappa_gamma=kappa_gamma,
                                                    mej=mejs[x], vej=vejs[x]*ckm, area_projection=area_projs[x],
                                                    area_reference=area_refs[x])
            lbols = interaction_class.new_luminosity
            lbols = np.nan_to_num(lbols)
            photo = TemperatureFloor(time=time_temp, luminosity=lbols,
                                     temperature_floor=temperature_floors[x], vej=vejs[x]*ckm)
            temp_func = interp1d(time_temp, y=photo.photosphere_temperature)
            rad_func = interp1d(time_temp, y=photo.r_photosphere)
            temp = temp_func(time)
            photosphere = rad_func(time)
            flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)
            flux_density = np.nan_to_num(flux_density)
            units = flux_density.unit
            ff += flux_density.value
        ff = ff * units
        return ff.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift) #in days
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=cocoon_photo.photosphere_temperature,
                                         r_photosphere=cocoon_photo.r_photosphere,dl=dl,
                                         frequency=frequency[:,None]).T
        cocoon_spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        cocoon_spectra = np.nan_to_num(cocoon_spectra)
        full_spec = cocoon_spectra.value
        for x in range(3):
            lbols = _mosfit_kilonova_one_component_lbol(time=time_temp*day_to_s, mej=mejs[x], vej=vejs[x])
            interaction_class = AsphericalDiffusion(time=time_temp, dense_times=time_temp,
                                                    luminosity=lbols, kappa=kappas[x], kappa_gamma=kappa_gamma,
                                                    mej=mejs[x], vej=vejs[x]*ckm, area_projection=area_projs[x],
                                                    area_reference=area_refs[x])
            lbols = interaction_class.new_luminosity
            photo = TemperatureFloor(time=time_temp, luminosity=lbols,
                                     temperature_floor=temperature_floors[x], vej=vejs[x]*ckm)
            fmjy = blackbody_to_flux_density(temperature=photo.photosphere_temperature,
                                              r_photosphere=photo.r_photosphere, dl=dl,
                                              frequency=frequency[:, None])
            fmjy = fmjy.T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
            spectra = np.nan_to_num(spectra)
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                          spectra=full_spec, lambda_array=lambda_observer_frame,
                                                          **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract')
def mosfit_rprocess(time, redshift, mej, vej, kappa, kappa_gamma, temperature_floor, **kwargs):
    """
    A single component kilonova model that *should* be exactly the same as mosfit's r-process model.
    Effectively the only difference to the redback one_component model is the inclusion of gamma-ray opacity in diffusion.

    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses of first component
    :param vej: minimum initial velocity of first component in units of c
    :param kappa: gray opacity of first component
    :param temperature_floor: floor temperature of first component
    :param kappa_gamma: gamma-ray opacity
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param dense_resolution: resolution of the grid that the model is actually evaluated on, default is 300
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ckm = 3e10/1e5
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    dense_resolution = kwargs.get('dense_resolution', 300)
    time_temp = np.geomspace(1e-2, 7e6, dense_resolution) # in source frame in seconds
    time_obs = time
    lbols = _mosfit_kilonova_one_component_lbol(time=time_temp,
                                                mej=mej, vej=vej)
    interaction_class = Diffusion(time=time_temp / day_to_s, dense_times=time_temp / day_to_s, luminosity=lbols,
                                  kappa=kappa, kappa_gamma=kappa_gamma, mej=mej, vej=vej*ckm)
    lbols = interaction_class.new_luminosity
    photo = TemperatureFloor(time=time_temp / day_to_s, luminosity=lbols, vej=vej*ckm,
                             temperature_floor=temperature_floor)

    if kwargs['output_format'] == 'flux_density':
        #time = time_obs * day_to_s
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp / day_to_s, y=photo.photosphere_temperature)
        rad_func = interp1d(time_temp / day_to_s, y=photo.r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        temp = temp_func(time)
        photosphere = rad_func(time)
        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp / day_to_s * (1. + redshift) # in days
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=photo.photosphere_temperature,
                                         r_photosphere=photo.r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                          spectra=spectra, lambda_array=lambda_observer_frame,
                                                          **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract')
def mosfit_kilonova(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1,
                    mej_2, vej_2, temperature_floor_2, kappa_2,
                    mej_3, vej_3, temperature_floor_3, kappa_3, kappa_gamma, **kwargs):
    """
    A three component kilonova model that *should* be exactly the same as mosfit's kilonova model or Villar et al. 2017.
    Effectively the only difference to the redback three_component model is the inclusion of gamma-ray opacity in diffusion.
    Note: Villar et al. fix the kappa's of each component to get the desired red/blue/purple components. This is not done here.

    :param time: observer frame time in days
    :param redshift: redshift
    :param mej_1: ejecta mass in solar masses of first component
    :param vej_1: minimum initial velocity of first component in units of c
    :param kappa_1: gray opacity of first component
    :param temperature_floor_1: floor temperature of first component
    :param mej_2: ejecta mass in solar masses of second component
    :param vej_2: minimum initial velocity of second component in units of c
    :param temperature_floor_2: floor temperature of second component
    :param kappa_2: gray opacity of second component
    :param mej_3: ejecta mass in solar masses of third component
    :param vej_3: minimum initial velocity of third component in units of c
    :param temperature_floor_3: floor temperature of third component
    :param kappa_3: gray opacity of third component
    :param kappa_gamma: gamma-ray opacity
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param dense_resolution: resolution of the grid that the model is actually evaluated on, default is 300
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ckm = 3e10/1e5
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    dense_resolution = kwargs.get('dense_resolution', 300)
    time_temp = np.geomspace(1e-2, 7e6, dense_resolution)  # in source frame in s
    time_obs = time
    mej = [mej_1, mej_2, mej_3]
    vej = [vej_1, vej_2, vej_3]
    temperature_floor = [temperature_floor_1, temperature_floor_2, temperature_floor_3]
    kappa = [kappa_1, kappa_2, kappa_3]
    if kwargs['output_format'] == 'flux_density':
        #time = time_obs * day_to_s
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        ff = np.zeros(len(time))
        for x in range(3):
            lbols = _mosfit_kilonova_one_component_lbol(time=time_temp,
                                                        mej=mej[x], vej=vej[x])
            interaction_class = Diffusion(time=time_temp / day_to_s, dense_times=time_temp / day_to_s, luminosity=lbols,
                                          kappa=kappa[x], kappa_gamma=kappa_gamma, mej=mej[x], vej=vej[x]*ckm)
            lbols = interaction_class.new_luminosity
            photo = TemperatureFloor(time=time_temp / day_to_s, luminosity=lbols, vej=vej[x]*ckm,
                                     temperature_floor=temperature_floor[x])
            temp_func = interp1d(time_temp / day_to_s, y=photo.photosphere_temperature)
            rad_func = interp1d(time_temp / day_to_s, y=photo.r_photosphere)
            # convert to source frame time and frequency
            temp = temp_func(time)
            photosphere = rad_func(time)
            flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)
            units = flux_density.unit
            ff += flux_density.value
        ff = ff * units
        return ff.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp / day_to_s * (1. + redshift) # in days
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        full_spec = np.zeros((len(time), len(frequency)))
        for x in range(3):
            lbols = _mosfit_kilonova_one_component_lbol(time=time_temp,
                                                        mej=mej[x], vej=vej[x])
            interaction_class = Diffusion(time=time_temp / day_to_s, dense_times=time_temp / day_to_s, luminosity=lbols,
                                          kappa=kappa[x], kappa_gamma=kappa_gamma, mej=mej[x], vej=vej[x]*ckm)
            lbols = interaction_class.new_luminosity
            photo = TemperatureFloor(time=time_temp / day_to_s, luminosity=lbols, vej=vej[x]*ckm,
                                     temperature_floor=temperature_floor[x])
            fmjy = blackbody_to_flux_density(temperature=photo.photosphere_temperature,
                                             r_photosphere=photo.r_photosphere, frequency=frequency[:, None], dl=dl).T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                        lambdas=lambda_observer_frame,
                                                                        spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                          spectra=full_spec, lambda_array=lambda_observer_frame,
                                                          **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract')
def _mosfit_kilonova_one_component_lbol(time, mej, vej):
    """

    :param time: time in seconds in source frame
    :param mej: mass in solar masses
    :param vej: velocity in units of c
    :return: lbol in erg/s
    """
    tdays = time/day_to_s

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    t0 = 1.3 #seconds
    sig = 0.11  #seconds

    m0 = mej * solar_mass
    lum_in = 4.0e18 * (m0) * (0.5 - np.arctan((time - t0) / sig) / np.pi)**1.3
    lbol = lum_in * e_th
    return lbol

@citation_wrapper("redback,https://ui.adsabs.harvard.edu/abs/2020ApJ...891..152H/abstract")
def power_law_stratified_kilonova(time, redshift, mej, voffset, v0, alpha,
                                  kappa_offset, kappa_0, zeta, beta, **kwargs):
    """
    Assumes a power law distribution of ejecta velocities
    and a power law distribution of kappa corresponding to the ejecta velocity layers for a total of 10 "shells"
    and calculates the kilonova light curve, using kilonova heating rate.
    Velocity distribution is flipped so that faster material is the outermost layer as expected for homologous expansion.

    Must be used with a constraint to avoid prior draws that predict nonsensical luminosities. Or a sensible prior.

    :param time: time in days in observer frame
    :param redshift: redshift
    :param mej: total ejecta mass in solar masses
    :param voffset: minimum ejecta velocity in units of c
    :param v0: velocity normalization in units of c of the power law
    :param alpha: power-law index of the velocity distribution i.e., vel = (xs/v0)^-alpha + voffset.
    :param kappa_offset: minimum kappa value
    :param kappa_0: kappa normalization
    :param zeta: power law index of the kappa distribution i.e., kappa = (xs/kappa_0)^-zeta + kappa_offset
    :param beta: power law index of density profile
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    xs = np.linspace(0.2, 0.5, 10)
    velocity_array = np.flip(xs/v0) ** -alpha + voffset
    xs = np.linspace(0.2, 0.5, 9)
    kappa_array = (xs/kappa_0) ** -zeta + kappa_offset
    output = _kilonova_hr(time=time, redshift=redshift, mej=mej, velocity_array=velocity_array,
                          kappa_array=kappa_array, beta=beta, **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1137L/abstract')
def bulla_bns_kilonova(time, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs):
    """
    Kilonovanet model based on Bulla BNS merger simulations

    :param time: time in days in observer frame
    :param redshift: redshift
    :param mej_dyn: dynamical mass of ejecta in solar masses
    :param mej_disk: disk mass of ejecta in solar masses
    :param phi: half-opening angle of the lanthanide-rich tidal dynamical ejecta in degrees
    :param costheta_obs: cosine of the observers viewing angle
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from redback_surrogates.kilonovamodels import bulla_bns_kilonovanet_spectra as function

    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, time=time, redshift=redshift)
        output = function(time_source_frame=time, redshift=redshift, mej_dyn=mej_dyn,
                          mej_disk=mej_disk, phi=phi, costheta_obs=costheta_obs)
        spectra = output.spectra / (4 * np.pi * dl ** 2)  # to erg/s/cm^2/Angstrom
        spectra = spectra * uu.erg / (uu.s * uu.cm ** 2 * uu.Angstrom)
        fmjy = spectra.to(uu.mJy, equivalencies=uu.spectral_density(wav=output.lambdas * uu.Angstrom)).value
        nu_array = lambda_to_nu(output.lambdas)
        fmjy_func = RegularGridInterpolator((np.unique(time), nu_array), fmjy, bounds_error=False)
        if type(frequency) == float:
            frequency = np.ones(len(time)) * frequency
        points = np.array([time, frequency]).T
        return fmjy_func(points)
    else:
        time_source_frame = np.linspace(0.1, 20, 200)
        output = function(time_source_frame=time_source_frame, redshift=redshift, mej_dyn=mej_dyn,
                          mej_disk=mej_disk, phi=phi, costheta_obs=costheta_obs)
        if kwargs['output_format'] == 'spectra':
            return output
        else:
            time_observer_frame = output.time
            lambda_observer_frame = output.lambdas
            spectra = output.spectra / (4 * np.pi * dl ** 2) # to erg/s/cm^2/Angstrom
            spectra = spectra * uu.erg / (uu.s * uu.cm ** 2 * uu.Angstrom)
            time_obs = time
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                   spectra=spectra, lambda_array=lambda_observer_frame,
                                                   **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1137L/abstract')
def bulla_nsbh_kilonova(time, redshift, mej_dyn, mej_disk, costheta_obs, **kwargs):
    """
    Kilonovanet model based on Bulla NSBH merger simulations

    :param time: time in observer frame in days
    :param redshift: redshift
    :param mej_dyn: dynamical mass of ejecta in solar masses
    :param mej_disk: disk mass of ejecta in solar masses
    :param costheta_obs: cosine of the observers viewing angle
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from redback_surrogates.kilonovamodels import bulla_nsbh_kilonovanet_spectra as function

    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, time=time, redshift=redshift)
        output = function(time_source_frame=time, redshift=redshift, mej_dyn=mej_dyn,
                          mej_disk=mej_disk, costheta_obs=costheta_obs)
        spectra = output.spectra / (4 * np.pi * dl ** 2)  # to erg/s/cm^2/Angstrom
        spectra = spectra * uu.erg / (uu.s * uu.cm ** 2 * uu.Angstrom)
        fmjy = spectra.to(uu.mJy, equivalencies=uu.spectral_density(wav=output.lambdas * uu.Angstrom)).value
        nu_array = lambda_to_nu(output.lambdas)
        fmjy_func = RegularGridInterpolator((np.unique(time), nu_array), fmjy, bounds_error=False)
        if type(frequency) == float:
            frequency = np.ones(len(time)) * frequency
        points = np.array([time, frequency]).T
        return fmjy_func(points)
    else:
        time_source_frame = np.linspace(0.1, 20, 200)
        output = function(time_source_frame=time_source_frame, redshift=redshift, mej_dyn=mej_dyn,
                          mej_disk=mej_disk, costheta_obs=costheta_obs)
        if kwargs['output_format'] == 'spectra':
            return output
        else:
            time_observer_frame = output.time
            lambda_observer_frame = output.lambdas
            spectra = output.spectra / (4 * np.pi * dl ** 2) # to erg/s/cm^2/Angstrom
            spectra = spectra * uu.erg / (uu.s * uu.cm ** 2 * uu.Angstrom)
            time_obs = time
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                   spectra=spectra, lambda_array=lambda_observer_frame,
                                                   **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1137L/abstract')
def kasen_bns_kilonova(time, redshift, mej, vej, chi, **kwargs):
    """
    Kilonovanet model based on Kasen BNS simulations

    :param time: time in days in observer frame
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in units of c
    :param chi: lanthanide fraction
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from redback_surrogates.kilonovamodels import kasen_bns_kilonovanet_spectra as function

    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, time=time, redshift=redshift)
        output = function(time_source_frame=time,redshift=redshift, mej=mej, vej=vej, chi=chi)
        spectra = output.spectra / (4 * np.pi * dl ** 2) # to erg/s/cm^2/Angstrom
        spectra = spectra * uu.erg / (uu.s * uu.cm ** 2 * uu.Angstrom)
        fmjy = spectra.to(uu.mJy, equivalencies=uu.spectral_density(wav=output.lambdas * uu.Angstrom)).value
        nu_array = lambda_to_nu(output.lambdas)
        fmjy_func = RegularGridInterpolator((np.unique(time), nu_array), fmjy, bounds_error=False)
        if type(frequency) == float or type(frequency) == np.float64:
            frequency = np.ones(len(time)) * frequency
        points = np.array([time, frequency]).T
        return fmjy_func(points)
    else:
        time_source_frame = np.linspace(0.1, 20, 200)
        output = function(time_source_frame=time_source_frame, redshift=redshift, mej=mej, vej=vej, chi=chi)
        if kwargs['output_format'] == 'spectra':
            return output
        else:
            time_observer_frame = output.time
            lambda_observer_frame = output.lambdas
            spectra = output.spectra / (4 * np.pi * dl ** 2) # to erg/s/cm^2/Angstrom
            spectra = spectra * uu.erg / (uu.s * uu.cm ** 2 * uu.Angstrom)
            time_obs = time
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                   spectra=spectra, lambda_array=lambda_observer_frame,
                                                   **kwargs)
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...891..152H/abstract')
def two_layer_stratified_kilonova(time, redshift, mej, vej_1, vej_2, kappa, beta, **kwargs):
    """
    Uses kilonova_heating_rate module to model a two layer stratified kilonova

    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej_1: velocity of inner shell in c
    :param vej_2: velocity of outer shell in c
    :param kappa: constant gray opacity
    :param beta: power law index of density profile
    :param kwargs: Additional keyword arguments
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    velocity_array = np.array([vej_1, vej_2])
    output = _kilonova_hr(time, redshift, mej, velocity_array, kappa, beta, **kwargs)
    return output


def _kilonova_hr(time, redshift, mej, velocity_array, kappa_array, beta, **kwargs):
    """
    Uses kilonova_heating_rate module

    :param time: observer frame time in days
    :param redshift: redshift
    :param frequency: frequency to calculate - Must be same length as time array or a single number
    :param mej: ejecta mass
    :param velocity_array: array of ejecta velocities; length >=2
    :param kappa_array: opacities of each shell, length = 1 less than velocity
    :param beta: power law index of density profile
    :param kwargs: Additional keyword arguments
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
        time = time * day_to_s
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        if (isinstance(frequency, (float, int)) == False):
            radio_mask = frequency < 10e10
            frequency[radio_mask]=10e50
        elif frequency < 10e10:
            frequency =10e50

        _, temperature, r_photosphere = _kilonova_hr_sourceframe(time, mej, velocity_array, kappa_array, beta)

        flux_density = blackbody_to_flux_density(temperature=temperature.value, r_photosphere=r_photosphere.value,
                                                 dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = np.geomspace(0.03, 10, 100) * day_to_s
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        _, temperature, r_photosphere = _kilonova_hr_sourceframe(time, mej, velocity_array, kappa_array, beta)
        fmjy = blackbody_to_flux_density(temperature=temperature.value,
                                         r_photosphere=r_photosphere.value, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                          spectra=spectra, lambda_array=lambda_observer_frame,
                                                          **kwargs)

def _kilonova_hr_sourceframe(time, mej, velocity_array, kappa_array, beta):
    """
    Uses kilonova_heating_rate module

    :param time: source frame time in seconds
    :param mej: ejecta mass
    :param velocity_array: array of ejecta velocities; length >=2
    :param kappa_array: opacities of each shell, length = 1 less than velocity
    :param beta: power law index of density profile
    :return: bolometric_luminosity, temperature, photosphere
    """
    if len(velocity_array) < 2:
        raise ValueError("velocity_array must be of length >=2")

    from kilonova_heating_rate import lightcurve

    mej = mej * uu.M_sun
    velocity_array = velocity_array * cc.c
    kappa_array = kappa_array * uu.cm**2 / uu.g
    time = time * uu.s
    time = time.to(uu.day)
    if time.value[0] < 0.02:
        raise ValueError("time in source frame must be larger than 0.02 days for this model")

    bolometric_luminosity, temperature, r_photosphere = lightcurve(time, mass=mej, velocities=velocity_array,
                                                                   opacities=kappa_array, n=beta)
    return bolometric_luminosity, temperature, r_photosphere

@citation_wrapper('redback')
def three_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1,
                                 mej_2, vej_2, temperature_floor_2, kappa_2,
                                   mej_3, vej_3, temperature_floor_3, kappa_3, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej_1: ejecta mass in solar masses of first component
    :param vej_1: minimum initial velocity of first component
    :param kappa_1: gray opacity of first component
    :param temperature_floor_1: floor temperature of first component
    :param mej_2: ejecta mass in solar masses of second component
    :param vej_2: minimum initial velocity of second component
    :param temperature_floor_2: floor temperature of second component
    :param kappa_2: gray opacity of second component
    :param mej_3: ejecta mass in solar masses of third component
    :param vej_3: minimum initial velocity of third component
    :param temperature_floor_3: floor temperature of third component
    :param kappa_3: gray opacity of third component
    :param kwargs: Additional keyword arguments
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
    time_temp = np.geomspace(1e-2, 7e6, 300) # in source frame
    time_obs = time

    mej = [mej_1, mej_2, mej_3]
    vej = [vej_1, vej_2, vej_3]
    temperature_floor = [temperature_floor_1, temperature_floor_2, temperature_floor_3]
    kappa = [kappa_1, kappa_2, kappa_3]

    if kwargs['output_format'] == 'flux_density':
        time = time * day_to_s
        frequency = kwargs['frequency']

        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        ff = np.zeros(len(time))
        for x in range(3):
            temp_kwargs = {}
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej[x], vej[x], kappa[x],
                                                                          **temp_kwargs)
            # interpolate properties onto observation times
            temp_func = interp1d(time_temp, y=temperature)
            rad_func = interp1d(time_temp, y=r_photosphere)
            temp = temp_func(time)
            photosphere = rad_func(time)
            flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)
            units = flux_density.unit
            ff += flux_density.value

        ff = ff * units
        return ff.to(uu.mJy).value

    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        full_spec = np.zeros((len(time), len(frequency)))
        for x in range(3):
            temp_kwargs = {}
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej[x], vej[x], kappa[x],
                                                                          **temp_kwargs)
            fmjy = blackbody_to_flux_density(temperature=temperature,
                                             r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
            fmjy = fmjy.T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                          lambdas=lambda_observer_frame,
                                                                          spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                          spectra=full_spec, lambda_array=lambda_observer_frame,
                                                          **kwargs)


@citation_wrapper('redback')
def two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1,
                                 mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej_1: ejecta mass in solar masses of first component
    :param vej_1: minimum initial velocity of first component
    :param kappa_1: gray opacity of first component
    :param temperature_floor_1: floor temperature of first component
    :param mej_2: ejecta mass in solar masses of second component
    :param vej_2: minimum initial velocity of second component
    :param temperature_floor_2: floor temperature of second component
    :param kappa_2: gray opacity of second component
    :param kwargs: Additional keyword arguments
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
    time_temp = np.geomspace(1e-2, 7e6, 300) # in source frame
    time_obs = time

    mej = [mej_1, mej_2]
    vej = [vej_1, vej_2]
    temperature_floor = [temperature_floor_1, temperature_floor_2]
    kappa = [kappa_1, kappa_2]

    if kwargs['output_format'] == 'flux_density':
        time = time * day_to_s
        frequency = kwargs['frequency']

        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        ff = np.zeros(len(time))
        for x in range(2):
            temp_kwargs = {}
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej[x], vej[x], kappa[x],
                                                                          **temp_kwargs)
            # interpolate properties onto observation times
            temp_func = interp1d(time_temp, y=temperature)
            rad_func = interp1d(time_temp, y=r_photosphere)
            temp = temp_func(time)
            photosphere = rad_func(time)
            flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)
            units = flux_density.unit
            ff += flux_density.value

        ff = ff * units
        return ff.to(uu.mJy).value

    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        full_spec = np.zeros((len(time), len(frequency)))

        for x in range(2):
            temp_kwargs = {}
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej[x], vej[x], kappa[x],
                                                                          **temp_kwargs)
            fmjy = blackbody_to_flux_density(temperature=temperature,
                                             r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
            fmjy = fmjy.T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                           lambdas=lambda_observer_frame,
                                                                           spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                          spectra=full_spec, lambda_array=lambda_observer_frame,
                                                          **kwargs)

@citation_wrapper('redback')
def one_component_ejecta_relation(time, redshift, mass_1, mass_2,
                                  lambda_1, lambda_2, kappa, **kwargs):
    """
    Assumes no velocity projection in the ejecta velocity ejecta relation

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_1: mass of primary in solar masses
    :param mass_2: mass of secondary in solar masses
    :param lambda_1: dimensionless tidal deformability of primary
    :param lambda_2: dimensionless tidal deformability of secondary
    :param kappa: gray opacity
    :param kwargs: Additional keyword arguments
    :param temperature_floor: Temperature floor in K (default 4000)
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.OneComponentBNSNoProjection)
    ejecta_relation = ejecta_relation(mass_1, mass_2, lambda_1, lambda_2)
    mej = ejecta_relation.ejecta_mass
    vej = ejecta_relation.ejecta_velocity
    flux_density = one_component_kilonova_model(time, redshift, mej, vej, kappa, **kwargs)
    return flux_density

@citation_wrapper('redback')
def one_component_ejecta_relation_projection(time, redshift, mass_1, mass_2,
                                             lambda_1, lambda_2, kappa, **kwargs):
    """
    Assumes a velocity projection between the orthogonal and orbital plane

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_1: mass of primary in solar masses
    :param mass_2: mass of secondary in solar masses
    :param lambda_1: dimensionless tidal deformability of primary
    :param lambda_2: dimensionless tidal deformability of secondary
    :param kappa: gray opacity
    :param kwargs: Additional keyword arguments
    :param temperature_floor: Temperature floor in K (default 4000)
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.OneComponentBNSProjection)
    ejecta_relation = ejecta_relation(mass_1, mass_2, lambda_1, lambda_2)
    mej = ejecta_relation.ejecta_mass
    vej = ejecta_relation.ejecta_velocity
    flux_density = one_component_kilonova_model(time, redshift, mej, vej, kappa, **kwargs)
    return flux_density

@citation_wrapper('redback')
def two_component_bns_ejecta_relation(time, redshift, mass_1, mass_2,
                                        lambda_1, lambda_2, mtov, zeta, vej_2, kappa_1, kappa_2, tf_1, tf_2, **kwargs):
    """
    Assumes two kilonova components corresponding to dynamical and disk wind ejecta with properties
    derived using ejecta relation specified by keyword argument.

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_1: mass of primary in solar masses
    :param mass_2: mass of secondary in solar masses
    :param lambda_1: dimensionless tidal deformability of primary
    :param lambda_2: dimensionless tidal deformability of secondary
    :param mtov: Tolman Oppenheimer Volkoff mass in solar masses
    :param zeta: fraction of disk that gets unbound
    :param vej_2: disk wind velocity in c
    :param kappa_1: gray opacity of first component
    :param kappa_2: gracy opacity of second component
    :param tf_1: floor temperature of first component
    :param tf_2: floor temperature of second component
    :param kwargs: additional keyword arguments
    :param ejecta_relation: a class that relates the instrinsic parameters to the kilonova parameters
            default is TwoComponentBNS
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.TwoComponentBNS)
    ejecta_relation = ejecta_relation(mass_1=mass_1, mass_2=mass_2, lambda_1=lambda_1,
                                      lambda_2=lambda_2, mtov=mtov, zeta=zeta)
    mej_1 = ejecta_relation.dynamical_mej
    mej_2 = ejecta_relation.disk_wind_mej
    vej_1 = ejecta_relation.ejecta_velocity

    output = two_component_kilonova_model(time=time, redshift=redshift, mej_1=mej_1,
                                                vej_1=vej_1, temperature_floor_1=tf_1,
                                                kappa_1=kappa_1, mej_2=mej_2, vej_2=vej_2,
                                                temperature_floor_2=tf_2, kappa_2=kappa_2, **kwargs)
    return output

@citation_wrapper('redback')
def polytrope_eos_two_component_bns(time, redshift, mass_1, mass_2,  log_p, gamma_1, gamma_2, gamma_3,
                                    zeta, vej_2, kappa_1, kappa_2, tf_1, tf_2, **kwargs):
    """
    Assumes two kilonova components corresponding to dynamical and disk wind ejecta with properties
    derived using ejecta relation specified by keyword argument and lambda set by polytropic EOS.

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_1: mass of primary in solar masses
    :param mass_2: mass of secondary in solar masses
    :param log_p: log central pressure in SI units
    :param gamma_1: polytrope index 1
    :param gamma_2: polytrope index 2
    :param gamma_3: polytrope index 3
    :param zeta: fraction of disk that gets unbound
    :param vej_2: disk wind velocity in c
    :param kappa_1: gray opacity of first component
    :param kappa_2: gracy opacity of second component
    :param tf_1: floor temperature of first component
    :param tf_2: floor temperature of second component
    :param kwargs: additional keyword arguments
    :param ejecta_relation: a class that relates the instrinsic parameters to the kilonova parameters
            default is TwoComponentBNS
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    central_pressure = np.logspace(np.log10(4e32), np.log10(2.5e35), 70)
    eos = PiecewisePolytrope(log_p=log_p, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)
    mtov = eos.maximum_mass()
    masses = np.array([mass_1, mass_2])
    tidal_deformability, _ = eos.lambda_of_mass(central_pressure=central_pressure, mass=masses)
    lambda_1, lambda_2 = tidal_deformability[0], tidal_deformability[1]
    ejecta_relation = kwargs.get('ejecta_relation', ejr.TwoComponentBNS)
    ejecta_relation = ejecta_relation(mass_1=mass_1, mass_2=mass_2, lambda_1=lambda_1,
                                      lambda_2=lambda_2, mtov=mtov, zeta=zeta)
    mej_1 = ejecta_relation.dynamical_mej
    mej_2 = ejecta_relation.disk_wind_mej
    vej_1 = ejecta_relation.ejecta_velocity

    output = two_component_kilonova_model(time=time, redshift=redshift, mej_1=mej_1,
                                                vej_1=vej_1, temperature_floor_1=tf_1,
                                                kappa_1=kappa_1, mej_2=mej_2, vej_2=vej_2,
                                                temperature_floor_2=tf_2, kappa_2=kappa_2, **kwargs)
    return output

@citation_wrapper('redback')
def one_component_nsbh_ejecta_relation(time, redshift, mass_bh, mass_ns,
                                        chi_bh, lambda_ns, kappa, **kwargs):
    """
    One component NSBH model

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_bh: mass of black hole
    :param mass_2: mass of neutron star
    :param chi_bh: spin of black hole along z axis
    :param lambda_ns: tidal deformability of neutron star
    :param kappa: opacity
    :param kwargs: additional keyword arguments
    :param temperature_floor: floor temperature
    :param ejecta_relation: a class that relates the instrinsic parameters to the kilonova parameters
            default is OneComponentNSBH
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.OneComponentNSBH)
    ejecta_relation = ejecta_relation(mass_bh=mass_bh, mass_ns=mass_ns, chi_bh=chi_bh, lambda_ns=lambda_ns)
    mej = ejecta_relation.ejecta_mass
    vej = ejecta_relation.ejecta_velocity
    output = one_component_kilonova_model(time, redshift, mej, vej, kappa, **kwargs)
    return output

@citation_wrapper('redback')
def two_component_nsbh_ejecta_relation(time, redshift,  mass_bh, mass_ns,
                                        chi_bh, lambda_ns, zeta, vej_2, kappa_1, kappa_2, tf_1, tf_2, **kwargs):
    """
    Two component NSBH model with dynamical and disk wind ejecta

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_bh: mass of black hole
    :param mass_2: mass of neutron star
    :param chi_bh: spin of black hole along z axis
    :param lambda_ns: tidal deformability of neutron star
    :param zeta: fraction of disk that gets unbound
    :param vej_2: disk wind velocity in c
    :param kappa_1: gray opacity of first component
    :param kappa_2: gracy opacity of second component
    :param tf_1: floor temperature of first component
    :param tf_2: floor temperature of second component
    :param kwargs: additional keyword arguments
    :param ejecta_relation: a class that relates the instrinsic parameters to the kilonova parameters
            default is TwoComponentNSBH
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.TwoComponentNSBH)
    ejecta_relation = ejecta_relation(mass_bh=mass_bh, mass_ns=mass_ns, chi_bh=chi_bh,
                                      lambda_ns=lambda_ns, zeta=zeta)
    mej_1 = ejecta_relation.dynamical_mej
    mej_2 = ejecta_relation.disk_wind_mej
    vej_1 = ejecta_relation.ejecta_velocity

    output = two_component_kilonova_model(time=time, redshift=redshift, mej_1=mej_1,
                                                vej_1=vej_1, temperature_floor_1=tf_1,
                                                kappa_1=kappa_1, mej_2=mej_2, vej_2=vej_2,
                                                temperature_floor_2=tf_2, kappa_2=kappa_2, **kwargs)
    return output

@citation_wrapper('redback')
def one_component_kilonova_model(time, redshift, mej, vej, kappa, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param kappa: gray opacity
    :param kwargs: Additional keyword arguments
    :param temperature_floor: Temperature floor in K (default 4000)
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
    time_temp = np.geomspace(1e-3, 7e6, 300) # in source frame
    time_obs = time
    _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej, vej, kappa, **kwargs)

    if kwargs['output_format'] == 'flux_density':
        time = time_obs * day_to_s
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=temperature)
        rad_func = interp1d(time_temp, y=r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)

        return flux_density.to(uu.mJy).value

    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                           lambdas=lambda_observer_frame,
                                                                           spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                          spectra=spectra, lambda_array=lambda_observer_frame,
                                                          **kwargs)


def _calc_new_heating_rate(time, mej, electron_fraction, ejecta_velocity, **kwargs):
    """
    Heating rate prescription following Rosswog and Korobkin 2024

    :param time: time in seconds
    :param mej: ejecta mass in solar masses
    :param electron_fraction: electron fraction
    :param ejecta_velocity: ejecta velocity in c
    :param kwargs: Additional keyword arguments
    :param heating_rate_perturbation: A fudge factor for heating rate to account for uncertainties in the heating rate. Default is 1.0 i.e., no perturbation.
    :param heating_rate_fudge: A fudge factor for each of the terms in the heating rate. Default to 1. i.e., no uncertainty
    Default is 1.0 i.e., no perturbation.
    :return: heating rate in erg/s
    """
    heating_rate_perturbation = kwargs.get('heating_rate_perturbation', 1.0)
    # rescale
    m0 = mej * solar_mass
    qdot_in = _calculate_rosswogkorobkin24_qdot(time, ejecta_velocity, electron_fraction)
    lum_in = qdot_in * m0
    return lum_in * heating_rate_perturbation

def _calculate_rosswogkorobkin24_qdot_formula(time_array, e0, alp, t0, sig, alp1,
                            t1, sig1, c1, tau1, c2, tau2, c3, tau3):
    time = time_array
    c1 = np.exp(c1)
    c2 = np.exp(c2)
    c3 = np.exp(c3)
    tau1 = 1e3 * tau1
    tau2 = 1e5 * tau2
    tau3 = 1e5 * tau3
    term1 = 10. ** (e0 + 18) * (0.5 - np.arctan((time - t0) / sig) / np.pi) ** alp
    term2 = (0.5 + np.arctan((time - t1) / sig1) / np.pi) ** alp1
    term3 = c1 * np.exp(-time / tau1)
    term4 = c2 * np.exp(-time / tau2)
    term5 = c3 * np.exp(-time / tau3)
    lum_in = term1 * term2 + term3 + term4 + term5
    return lum_in

def _one_component_kilonova_rosswog_heatingrate(time, mej, vej, electron_fraction, **kwargs):
    tdays = time/day_to_s
    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    temperature_floor = kwargs.get('temperature_floor', 4000)  # kelvin
    beta = 13.7

    v0 = vej * speed_of_light
    m0 = mej * solar_mass
    kappa = kappa_from_electron_fraction(electron_fraction)
    tdiff = np.sqrt(2.0 * kappa * (m0) / (beta * v0 * speed_of_light))

    lum_in = _calc_new_heating_rate(time, mej, electron_fraction, vej, **kwargs)
    integrand = lum_in * e_th * (time / tdiff) * np.exp(time ** 2 / tdiff ** 2)

    bolometric_luminosity = np.zeros(len(time))
    bolometric_luminosity[1:] = cumulative_trapezoid(integrand, time)
    bolometric_luminosity[0] = bolometric_luminosity[1]
    bolometric_luminosity = bolometric_luminosity * np.exp(-time ** 2 / tdiff ** 2) / tdiff

    temperature = (bolometric_luminosity / (4.0 * np.pi * sigma_sb * v0 ** 2 * time ** 2)) ** 0.25
    r_photosphere = (bolometric_luminosity / (4.0 * np.pi * sigma_sb * temperature_floor ** 4)) ** 0.5

    # check temperature floor conditions
    mask = temperature <= temperature_floor
    temperature[mask] = temperature_floor
    mask = np.logical_not(mask)
    r_photosphere[mask] = v0 * time[mask]
    return bolometric_luminosity, temperature, r_photosphere

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024arXiv240407271S/abstract, https://ui.adsabs.harvard.edu/abs/2024AnP...53600306R/abstract')
def one_comp_kne_rosswog_heatingrate(time, redshift, mej, vej, ye, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param kappa: gray opacity
    :param kwargs: Additional keyword arguments
    :param temperature_floor: Temperature floor in K (default 4000)
    :param heating_rate_perturbation: A fudge factor for heating rate to account for uncertainties in the heating rate.
    Default is 1.0 i.e., no perturbation.
    :param heating_rate_fudge: A fudge factor for each of the terms in the heating rate. Default to 1. i.e., no uncertainty
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
    time_temp = np.geomspace(1e-3, 7e6, 300) # in source frame
    time_obs = time
    _, temperature, r_photosphere = _one_component_kilonova_rosswog_heatingrate(time_temp, mej, vej, ye, **kwargs)

    if kwargs['output_format'] == 'flux_density':
        time = time_obs * day_to_s
        frequency = kwargs['frequency']
        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=temperature)
        rad_func = interp1d(time_temp, y=r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)

        return flux_density.to(uu.mJy).value

    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                           lambdas=lambda_observer_frame,
                                                                           spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                          spectra=spectra, lambda_array=lambda_observer_frame,
                                                          **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024arXiv240407271S/abstract, https://ui.adsabs.harvard.edu/abs/2024AnP...53600306R/abstract')
def two_comp_kne_rosswog_heatingrate(time, redshift, mej_1, vej_1, temperature_floor_1, ye_1,
                                 mej_2, vej_2, temperature_floor_2, ye_2, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej_1: ejecta mass in solar masses of first component
    :param vej_1: minimum initial velocity of first component
    :param kappa_1: gray opacity of first component
    :param temperature_floor_1: floor temperature of first component
    :param mej_2: ejecta mass in solar masses of second component
    :param vej_2: minimum initial velocity of second component
    :param temperature_floor_2: floor temperature of second component
    :param kappa_2: gray opacity of second component
    :param kwargs: Additional keyword arguments
    :param heating_rate_perturbation: A fudge factor for heating rate to account for uncertainties in the heating rate.
    Default is 1.0 i.e., no perturbation.
    :param heating_rate_fudge: A fudge factor for each of the terms in the heating rate. Default to 1. i.e., no uncertainty
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
    time_temp = np.geomspace(1e-2, 7e6, 300) # in source frame
    time_obs = time

    mej = [mej_1, mej_2]
    vej = [vej_1, vej_2]
    temperature_floor = [temperature_floor_1, temperature_floor_2]
    ye = [ye_1, ye_2]

    if kwargs['output_format'] == 'flux_density':
        time = time * day_to_s
        frequency = kwargs['frequency']

        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        ff = np.zeros(len(time))
        for x in range(2):
            temp_kwargs = {}
            if 'heating_rate_fudge' in kwargs:
                temp_kwargs['heating_rate_fudge'] = kwargs['heating_rate_fudge']
            if 'heating_rate_perturbation' in kwargs:
                temp_kwargs['heating_rate_perturbation'] = kwargs['heating_rate_perturbation']
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_rosswog_heatingrate(time_temp, mej[x], vej[x], ye[x],
                                                                          **temp_kwargs)
            # interpolate properties onto observation times
            temp_func = interp1d(time_temp, y=temperature)
            rad_func = interp1d(time_temp, y=r_photosphere)
            temp = temp_func(time)
            photosphere = rad_func(time)
            flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                     dl=dl, frequency=frequency)
            units = flux_density.unit
            ff += flux_density.value

        ff = ff * units
        return ff.to(uu.mJy).value

    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        full_spec = np.zeros((len(time), len(frequency)))

        for x in range(2):
            temp_kwargs = {}
            if 'heating_rate_fudge' in kwargs:
                temp_kwargs['heating_rate_fudge'] = kwargs['heating_rate_fudge']
            if 'heating_rate_perturbation' in kwargs:
                temp_kwargs['heating_rate_perturbation'] = kwargs['heating_rate_perturbation']
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_rosswog_heatingrate(time_temp, mej[x], vej[x], ye[x],
                                                                          **temp_kwargs)
            fmjy = blackbody_to_flux_density(temperature=temperature,
                                             r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
            fmjy = fmjy.T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                           lambdas=lambda_observer_frame,
                                                                           spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                          spectra=full_spec, lambda_array=lambda_observer_frame,
                                                          **kwargs)

def _one_component_kilonova_model(time, mej, vej, kappa, **kwargs):
    """
    :param time: source frame time in seconds
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param kappa_r: gray opacity
    :param kwargs: temperature_floor
    :return: bolometric_luminosity, temperature, r_photosphere
    """
    tdays = time/day_to_s

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    t0 = 1.3 #seconds
    sig = 0.11  #seconds
    temperature_floor = kwargs.get('temperature_floor', 4000) #kelvin

    beta = 13.7

    v0 = vej * speed_of_light
    m0 = mej * solar_mass
    tdiff = np.sqrt(2.0 * kappa * (m0) / (beta * v0 * speed_of_light))
    lum_in = 4.0e18 * (m0) * (0.5 - np.arctan((time - t0) / sig) / np.pi)**1.3
    integrand = lum_in * e_th * (time/tdiff) * np.exp(time**2/tdiff**2)
    bolometric_luminosity = np.zeros(len(time))
    bolometric_luminosity[1:] = cumulative_trapezoid(integrand, time)
    bolometric_luminosity[0] = bolometric_luminosity[1]
    bolometric_luminosity = bolometric_luminosity * np.exp(-time**2/tdiff**2) / tdiff

    temperature = (bolometric_luminosity / (4.0 * np.pi * sigma_sb * v0**2 * time**2))**0.25
    r_photosphere = (bolometric_luminosity / (4.0 * np.pi * sigma_sb * temperature_floor**4))**0.5

    # check temperature floor conditions
    mask = temperature <= temperature_floor
    temperature[mask] = temperature_floor
    mask = np.logical_not(mask)
    r_photosphere[mask] = v0 * time[mask]
    return bolometric_luminosity, temperature, r_photosphere

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017LRR....20....3M/abstract')
def metzger_kilonova_model(time, redshift, mej, vej, beta, kappa, **kwargs):
    """
    :param time: observer frame time in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa: gray opacity
    :param kwargs: Additional keyword arguments
    :param neutron_precursor_switch: Whether to include neutron precursor emission
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
    time_temp = np.geomspace(1e-4, 7e6, 300) # in source frame
    time_obs = time
    bolometric_luminosity, temperature, r_photosphere = _metzger_kilonova_model(time_temp, mej, vej, beta,
                                                                                kappa, **kwargs)

    if kwargs['output_format'] == 'flux_density':
        time = time * day_to_s
        frequency = kwargs['frequency']

        # interpolate properties onto observation times
        temp_func = interp1d(time_temp, y=temperature)
        rad_func = interp1d(time_temp, y=r_photosphere)
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        temp = temp_func(time)
        photosphere = rad_func(time)

        flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                                 dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value

    else:
        lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                           lambdas=lambda_observer_frame,
                                                                           spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                          spectra=spectra, lambda_array=lambda_observer_frame,
                                                          **kwargs)


def _metzger_kilonova_model(time, mej, vej, beta, kappa, **kwargs):
    """
    :param time: time array to evaluate model on in source frame in seconds
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa: gray opacity
    :param kwargs: Additional keyword arguments
    :param neutron_precursor_switch: Whether to include neutron precursor emission
    :return: bolometric_luminosity, temperature, photosphere_radius
    """
    neutron_precursor_switch = kwargs.get('neutron_precursor_switch', True)

    time = time
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

    # set up mass and velocity layers
    vmin = vej
    vmax = kwargs.get('vmax', 0.7)
    vel = np.linspace(vmin, vmax, mass_len)
    m_array = mej * (vel/vmin)**(-beta)
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

    # set up empty arrays
    energy_v = np.zeros((mass_len, time_len))
    lum_rad = np.zeros((mass_len, time_len))
    qdot_rp = np.zeros((mass_len, time_len))
    td_v = np.zeros((mass_len, time_len))
    tau = np.zeros((mass_len, time_len))
    v_photosphere = np.zeros(time_len)
    r_photosphere = np.zeros(time_len)

    if neutron_precursor_switch == True:
        neutron_mass = 1e-8 * solar_mass
        neutron_mass_fraction = 1 - 2*electron_fraction * 2 * np.arctan(neutron_mass / m_array / solar_mass) / np.pi
        rprocess_mass_fraction = 1.0 - neutron_mass_fraction
        initial_neutron_mass_fraction_array = np.tile(neutron_mass_fraction, (time_len, 1)).T
        rprocess_mass_fraction_array = np.tile(rprocess_mass_fraction, (time_len, 1)).T
        neutron_mass_fraction_array = initial_neutron_mass_fraction_array*np.exp(-time_array / tau_neutron)
        edotn = 3.2e14 * neutron_mass_fraction_array
        edotn = edotn * neutron_mass_fraction_array
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

    # solve ODE using euler method for all mass shells v
    for ii in range(time_len - 1):
        if neutron_precursor_switch:
            td_v[:-1, ii] = (kappa[:-1, ii] * m_array[:-1] * solar_mass * 3) / (
                    4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa[:-1, ii] / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))
        else:
            td_v[:-1, ii] = (kappa * m_array[:-1] * solar_mass * 3) / (
                        4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))
        lum_rad[:-1, ii] = energy_v[:-1, ii] / (td_v[:-1, ii] + time[ii] * (v_m[:-1] / speed_of_light))
        energy_v[:-1, ii + 1] = (edotr[:-1, ii] - (energy_v[:-1, ii] / time[ii]) - lum_rad[:-1, ii]) * dt[ii] + energy_v[:-1, ii]
        lum_rad[:-1, ii] = lum_rad[:-1, ii] * dm * solar_mass

        tau[mass_len - 1, ii] = tau[mass_len - 2, ii]
        photosphere_index = np.argmin(np.abs(tau[:, ii] - 1))
        v_photosphere[ii] = v_m[photosphere_index]
        r_photosphere[ii] = v_photosphere[ii] * time[ii]

    bolometric_luminosity = np.sum(lum_rad, axis=0)

    temperature = (bolometric_luminosity / (4.0 * np.pi * (r_photosphere) ** (2.0) * sigma_sb)) ** (0.25)

    return bolometric_luminosity, temperature, r_photosphere

def _generate_single_lightcurve(model, t_ini, t_max, dt, **parameters):
    """
    Generates a single lightcurve for a given `gwemlightcurves` model

    Parameters
    ----------
    model: str
        The `gwemlightcurve` model, e.g. 'DiUj2017'
    t_ini: float
        Starting time of the time array `gwemlightcurves` will calculate values at.
    t_max: float
        End time of the time array `gwemlightcurves` will calculate values at.
    dt: float
        Spacing of time uniform time steps.
    parameters: dict
        Function parameters for the given model.
    Returns
    ----------
    func, func: A bolometric function and the magnitude function.
    """
    from gwemlightcurves.KNModels.table import KNTable

    t = Table()
    for key in parameters.keys():
        val = parameters[key]
        t.add_column(Column(data=[val], name=key))
    t.add_column(Column(data=[t_ini], name="tini"))
    t.add_column(Column(data=[t_max], name="tmax"))
    t.add_column(Column(data=[dt], name="dt"))
    model_table = KNTable.model(model, t)
    return model_table["t"][0], model_table["lbol"][0], model_table["mag"][0]


def _generate_single_lightcurve_at_times(model, times, **parameters):
    """
    Generates a single lightcurve for a given `gwemlightcurves` model

    Parameters
    ----------
    model: str
        The `gwemlightcurve` model, e.g. 'DiUj2017'
    times: array_like
        Times at which we interpolate the `gwemlightcurves` values, in days
    parameters: dict
        Function parameters for the given model.
    Returns
    ----------
    array_like, array_like: bolometric and magnitude arrays.
    """

    tini = times[0]
    tmax = times[-1]
    dt = (tmax - tini)/(len(times) - 1)
    gwem_times, lbol, mag = _generate_single_lightcurve(model=model, t_ini=times[0], t_max=times[-1],
                                                        dt=dt, **parameters)

    lbol = interp1d(gwem_times, lbol)(times)
    new_mag = []
    for m in mag:
        new_mag.append(interp1d(gwem_times, m)(times))
    return lbol, np.array(new_mag)


def _gwemlightcurve_interface_factory(model):
    """
    Generates `bilby`-compatible functions from `gwemlightcurve` models.

    Parameters
    ----------
    model: str
        The `gwemlightcurve` model, e.g. 'DiUj2017'

    Returns
    ----------
    func, func: A bolometric function and the magnitude function.
    """

    default_filters = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

    def interface_bolometric(times, **parameters):
        return _generate_single_lightcurve_at_times(model=model, times=times, **parameters)[0]

    def interface_all_magnitudes(times, **parameters):
        magnitudes = _generate_single_lightcurve_at_times(model=model, times=times, **parameters)[1]
        return pd.DataFrame(magnitudes.T, columns=default_filters)

    def interface_filtered_magnitudes(times, **parameters):
        filters = parameters.get('filters', default_filters)
        all_magnitudes = interface_all_magnitudes(times, **parameters)
        if len(filters) == 1:
            return all_magnitudes[filters[0]]

        filtered_magnitudes = np.zeros(len(times))
        for i, f in enumerate(filters):
            filtered_magnitudes[i] = all_magnitudes[f][i]

        return filtered_magnitudes

    return interface_bolometric, interface_filtered_magnitudes


gwem_DiUj2017_bolometric, gwem_DiUj2017_magnitudes = _gwemlightcurve_interface_factory("DiUj2017")
gwem_SmCh2017_bolometric, gwem_SmCh2017_magnitudes = _gwemlightcurve_interface_factory("SmCh2017")
gwem_Me2017_bolometric, gwem_Me2017_magnitudes = _gwemlightcurve_interface_factory("Me2017")
gwem_KaKy2016_bolometric, gwem_KaKy2016_magnitudes = _gwemlightcurve_interface_factory("KaKy2016")
gwem_WoKo2017_bolometric, gwem_WoKo2017_magnitudes = _gwemlightcurve_interface_factory("WoKo2017")
gwem_BaKa2016_bolometric, gwem_BaKa2016_magnitudes = _gwemlightcurve_interface_factory("BaKa2016")
gwem_Ka2017_bolometric, gwem_Ka2017_magnitudes = _gwemlightcurve_interface_factory("Ka2017")
gwem_Ka2017x2_bolometric, gwem_Ka2017x2_magnitudes = _gwemlightcurve_interface_factory("Ka2017x2")
gwem_Ka2017inc_bolometric, gwem_Ka2017inc_magnitudes = _gwemlightcurve_interface_factory("Ka2017inc")
gwem_Ka2017x2inc_bolometric, gwem_Ka2017x2inc_magnitudes = _gwemlightcurve_interface_factory("Ka2017x2inc")
gwem_RoFe2017_bolometric, gwem_RoFe2017_magnitudes = _gwemlightcurve_interface_factory("RoFe2017")
gwem_Bu2019_bolometric, gwem_Bu2019_magnitudes = _gwemlightcurve_interface_factory("Bu2019")
gwem_Bu2019inc_bolometric, gwem_Bu2019inc_magnitudes = _gwemlightcurve_interface_factory("Bu2019inc")
gwem_Bu2019lf_bolometric, gwem_Bu2019lf_magnitudes = _gwemlightcurve_interface_factory("Bu2019lf")
gwem_Bu2019lr_bolometric, gwem_Bu2019lr_magnitudes = _gwemlightcurve_interface_factory("Bu2019lr")
gwem_Bu2019lm_bolometric, gwem_Bu2019lm_magnitudes = _gwemlightcurve_interface_factory("Bu2019lm")
gwem_Bu2019lw_bolometric, gwem_Bu2019lw_magnitudes = _gwemlightcurve_interface_factory("Bu2019lw")
gwem_Bu2019re_bolometric, gwem_Bu2019re_magnitudes = _gwemlightcurve_interface_factory("Bu2019re")
gwem_Bu2019bc_bolometric, gwem_Bu2019bc_magnitudes = _gwemlightcurve_interface_factory("Bu2019bc")
gwem_Bu2019op_bolometric, gwem_Bu2019op_magnitudes = _gwemlightcurve_interface_factory("Bu2019op")
gwem_Bu2019ops_bolometric, gwem_Bu2019ops_magnitudes = _gwemlightcurve_interface_factory("Bu2019ops")
gwem_Bu2019rp_bolometric, gwem_Bu2019rp_magnitudes = _gwemlightcurve_interface_factory("Bu2019rp")
gwem_Bu2019rps_bolometric, gwem_Bu2019rps_magnitudes = _gwemlightcurve_interface_factory("Bu2019rps")
gwem_Wo2020dyn_bolometric, gwem_Wo2020dyn_magnitudes = _gwemlightcurve_interface_factory("Wo2020dyn")
gwem_Wo2020dw_bolometric, gwem_Wo2020dw_magnitudes = _gwemlightcurve_interface_factory("Wo2020dw")
gwem_Bu2019nsbh_bolometric, gwem_Bu2019nsbh_magnitudes = _gwemlightcurve_interface_factory("Bu2019nsbh")
