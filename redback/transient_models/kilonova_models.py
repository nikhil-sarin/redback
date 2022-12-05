import numpy as np
import pandas as pd

from astropy.table import Table, Column
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo  # noqa
from scipy.integrate import cumtrapz
from collections import namedtuple

from redback.utils import calc_kcorrected_properties, interpolated_barnes_and_kasen_thermalisation_efficiency, \
    electron_fraction_from_kappa, citation_wrapper, lambda_to_nu
from redback.eos import PiecewisePolytrope
from redback.sed import blackbody_to_flux_density, get_correct_output_format_from_spectra
from redback.constants import *
import redback.ejecta_relations as ejr

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract')
def _mosfit_bns(time, mass_1, mass_2, lambda_s, kappa_red, kappa_blue,
               mtov, epsilon, alpha, cos_theta_open, **kwargs):
    a = 0.07550
    b = np.array([[-2.235, 0.8474], [10.45, -3.251], [-15.70, 13.61]])
    c = np.array([[-2.048, 0.5976], [7.941, 0.5658], [-7.360, -1.320]])
    n_ave = 0.743

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
    dynamical_ejecta = 1e-3 * (a_1 * ((mass_2 / mass_1) ** (1 / 3) * (1 - 2 * compactness_1) / compactness_2 * mass_baryonic_1 +
                            (mass_1 / mass_2) ** (1 / 3) * (1 - 2 * compactness_2) / compactness_2 * mass_baryonic_2) +
                     b_1 * ((mass_2 / mass_1) ** n * mass_baryonic_1 + (mass_1 / mass_2) ** n * mass_baryonic_2) +
                     c_1 * (mass_baryonic_1 - mass_1 + mass_baryonic_2 - mass_2) + d_1)

    if dynamical_ejecta < 0:
        dynamical_ejecta = 0

    raise NotImplementedError("This model is not yet implemented.")

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3016N/abstract')
def mosfit_bns(time, redshift, mass_1, mass_2, lambda_s, kappa_red, kappa_blue,
               mtov, epsilon, alpha, cos_theta_open, **kwargs):
    raise NotImplementedError("This model is not yet implemented.")

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract')
def mosfit_rprocess():
    raise NotImplementedError("This model is not yet implemented.")

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract')
def _mosfit_kilonova_one_component(time, mej, vej, kappa, ):
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract')
def mosfit_kilonova():
    raise NotImplementedError("This model is not yet implemented.")

@citation_wrapper("redback,https://ui.adsabs.harvard.edu/abs/2020ApJ...891..152H/abstract")
def power_law_stratified_kilonova(time, redshift, mej, vmin, vmax, alpha,
                                  kappa_min, kappa_max, beta, **kwargs):
    raise NotImplementedError("This model is not yet implemented.")

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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    dl = cosmo.luminosity_distance(redshift).cgs.value
    time_obs = time

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        time = time * day_to_s
        # convert to source frame time and frequency
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)

        _, temperature, r_photosphere = _kilonova_hr_sourceframe(time, mej, velocity_array, kappa_array, beta)

        flux_density = blackbody_to_flux_density(temperature=temperature.value, r_photosphere=r_photosphere.value,
                                                 dl=dl, frequency=frequency)
        return flux_density.to(uu.mJy).value
    else:
        frequency_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 20000, 100))
        time_observer_frame = np.geomspace(0.03, 10, 100) * day_to_s
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(frequency_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        _, temperature, r_photosphere = _kilonova_hr_sourceframe(time, mej, velocity_array, kappa_array, beta)
        fmjy = blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=frequency_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'frequency', 'spectra'])(time=time_observer_frame,
                                                                          frequency=frequency_observer_frame,
                                                                          spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                          spectra=spectra, frequency_array=frequency_observer_frame,
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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    dl = cosmo.luminosity_distance(redshift).cgs.value
    time_temp = np.geomspace(1e-4, 3e6, 300) # in source frame
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
        frequency_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 20000, 100))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(frequency_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        full_spec = np.zeros((len(frequency), len(time)))
        for x in range(3):
            temp_kwargs = {}
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej[x], vej[x], kappa[x],
                                                                          **temp_kwargs)
            fmjy = blackbody_to_flux_density(temperature=temperature,
                                             r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
            fmjy = fmjy.T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=frequency_observer_frame * uu.Angstrom))
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'frequency', 'spectra'])(time=time_observer_frame,
                                                                          frequency=frequency_observer_frame,
                                                                          spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                          spectra=full_spec, frequency_array=frequency_observer_frame,
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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    dl = cosmo.luminosity_distance(redshift).cgs.value
    time_temp = np.geomspace(1e-4, 3e6, 300) # in source frame
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
        frequency_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 20000, 100))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(frequency_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        full_spec = np.zeros((len(frequency), len(time)))

        for x in range(2):
            temp_kwargs = {}
            temp_kwargs['temperature_floor'] = temperature_floor[x]
            _, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej[x], vej[x], kappa[x],
                                                                          **temp_kwargs)
            fmjy = blackbody_to_flux_density(temperature=temperature,
                                             r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
            fmjy = fmjy.T
            spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=frequency_observer_frame * uu.Angstrom))
            units = spectra.unit
            full_spec += spectra.value

        full_spec = full_spec * units
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'frequency', 'spectra'])(time=time_observer_frame,
                                                                           frequency=frequency_observer_frame,
                                                                           spectra=full_spec)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                          spectra=full_spec, frequency_array=frequency_observer_frame,
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
                                        chi_eff, lambda_ns, kappa, **kwargs):
    """
    One component NSBH model

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_bh: mass of black hole
    :param mass_2: mass of neutron star
    :param chi_eff: effective spin of black hole
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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.OneComponentNSBH)
    ejecta_relation = ejecta_relation(mass_bh=mass_bh, mass_ns=mass_ns, chi_eff=chi_eff, lambda_ns=lambda_ns)
    mej = ejecta_relation.ejecta_mass
    vej = ejecta_relation.ejecta_velocity
    output = one_component_kilonova_model(time, redshift, mej, vej, kappa, **kwargs)
    return output

@citation_wrapper('redback')
def two_component_nsbh_ejecta_relation(time, redshift,  mass_bh, mass_ns,
                                        chi_eff, lambda_ns, zeta, vej_2, kappa_1, kappa_2, tf_1, tf_2, **kwargs):
    """
    Two component NSBH model with dynamical and disk wind ejecta

    :param time: observer frame time in days
    :param redshift: redshift
    :param mass_bh: mass of black hole
    :param mass_2: mass of neutron star
    :param chi_eff: effective spin of black hole
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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    ejecta_relation = kwargs.get('ejecta_relation', ejr.TwoComponentNSBH)
    ejecta_relation = ejecta_relation(mass_bh=mass_bh, mass_ns=mass_ns, chi_eff=chi_eff,
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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    dl = cosmo.luminosity_distance(redshift).cgs.value
    time_temp = np.geomspace(1e-4, 3e6, 300) # in source frame
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
        frequency_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 20000, 100))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(frequency_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=frequency_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'frequency', 'spectra'])(time=time_observer_frame,
                                                                           frequency=frequency_observer_frame,
                                                                           spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame/day_to_s,
                                                          spectra=spectra, frequency_array=frequency_observer_frame,
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
    bolometric_luminosity[1:] = cumtrapz(integrand, time)
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
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    dl = cosmo.luminosity_distance(redshift).cgs.value
    time_temp = np.geomspace(1e-4, 1e7, 300) # in source frame
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
        frequency_observer_frame = kwargs.get('frequency_array', np.geomspace(100, 20000, 100))
        time_observer_frame = time_temp * (1. + redshift)
        frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(frequency_observer_frame),
                                                     redshift=redshift, time=time_observer_frame)
        fmjy = blackbody_to_flux_density(temperature=temperature,
                                         r_photosphere=r_photosphere, frequency=frequency[:, None], dl=dl)
        fmjy = fmjy.T
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                     equivalencies=uu.spectral_density(wav=frequency_observer_frame * uu.Angstrom))
        if kwargs['output_format'] == 'spectra':
            return namedtuple('output', ['time', 'frequency', 'spectra'])(time=time_observer_frame,
                                                                          frequency=frequency_observer_frame,
                                                                          spectra=spectra)
        else:
            return get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame / day_to_s,
                                                          spectra=spectra, frequency_array=frequency_observer_frame,
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
