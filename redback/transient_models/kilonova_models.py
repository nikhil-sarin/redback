import numpy as np
import pandas as pd

from astropy.table import Table, Column
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import interpolated_barnes_and_kasen_thermalisation_efficiency, blackbody_to_flux_density, electron_fraction_from_kappa
from redback.constants import *
import astropy.units as uu
from scipy.integrate import cumtrapz

def one_component_kilonova_model(time, redshift, frequencies, mej, vej, kappa, **kwargs):
    """
    :param time: observer frame time
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param kappa_r: opacity
    :param kwargs: output_format
    :return: flux_density or magnitude
    """
    time_temp = np.geomspace(1e-4, 1e7, 300)
    bolometric_luminosity, temperature, r_photosphere = _one_component_kilonova_model(time_temp, mej,
                                                                                      vej, kappa, **kwargs)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    # interpolate properties onto observation times
    temp_func = interp1d(time_temp, y=temperature)
    rad_func = interp1d(time_temp, y=r_photosphere)
    # convert to source frame time and frequency
    time = time / (1 + redshift)
    frequencies = frequencies / (1 + redshift)

    temp = temp_func(time)
    photosphere = rad_func(time)

    flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                             dl=dl, frequencies=frequencies)

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def _one_component_kilonova_model(time, mej, vej, kappa):
    """
    :param time: source frame time
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param kappa_r: opacity
    :param kwargs: output_format
    :return: flux_density or magnitude
    """
    tdays = time/86400

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    t0 = 1.3 #seconds
    sig = 0.11  #seconds
    temperature_floor = 4000 #kelvin
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



def metzger_kilonova_model(time, redshift, frequencies, mej, vej, beta, kappa_r, **kwargs):
    """
    :param time: observer frame time
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param kwargs: neutron_precursor_switch, output_format
    :return: flux_density or magnitude
    """
    time_temp = np.geomspace(1e-4, 1e7, 300)
    bolometric_luminosity, temperature, r_photosphere = _metzger_kilonova_model(time_temp, mej, vej, beta,
                                                                                               kappa_r, **kwargs)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    # interpolate properties onto observation times
    temp_func = interp1d(time_temp, y=temperature)
    rad_func = interp1d(time_temp, y=r_photosphere)
    # convert to source frame time and frequency
    time = time / (1 + redshift)
    frequencies = frequencies / (1 + redshift)

    temp = temp_func(time)
    photosphere = rad_func(time)

    flux_density = blackbody_to_flux_density(temperature=temp, r_photosphere=photosphere,
                                             dl=dl, frequencies=frequencies)

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def _metzger_kilonova_model(time, mej, vej, beta, kappa_r, **kwargs):
    """
    :param time: time array to evaluate model on in source frame
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: minimum initial velocity
    :param beta: velocity power law slope (M=v^-beta)
    :param kappa_r: opacity
    :param kwargs: neutron_precursor_switch
    :return: bolometric_luminosity, temperature, photosphere_radius
    """
    neutron_precursor_switch = kwargs.get('neutron_precursor_switch', True)

    time = time
    tdays = time/86400
    time_len = len(time)
    mass_len = 250

    # set up kilonova physics
    av, bv, dv = interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)
    # thermalisation from Barnes+16
    e_th = 0.36 * (np.exp(-av * tdays) + np.log1p(2.0 * bv * tdays ** dv) / (2.0 * bv * tdays ** dv))
    electron_fraction = electron_fraction_from_kappa(kappa_r)
    t0 = 1.3 #seconds
    sig = 0.11  #seconds
    tau_neutron = 900  #seconds

    # convert to astrophysical units
    m0 = mej * solar_mass
    v0 = vej * speed_of_light
    ek_tot_0 = 0.5 * m0 * v0 ** 2

    # set up mass and velocity layers
    m_array = np.logspace(-8, np.log10(mej), mass_len) #in solar masses
    v_m = v0 * (m_array / (mej)) ** (-1 / beta)
    # don't violate relativity
    v_m[v_m > 3e10] = speed_of_light

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
        kappa_r = kappa_r * rprocess_mass_fraction_array
        kappa_r = kappa_n + kappa_r

    dt = np.diff(time)
    dm = np.diff(m_array)

    #initial conditions
    energy_v[:, 0] = 0.5 * m_array*v_m**2
    lum_rad[:, 0] = 0
    qdot_rp[:, 0] = 0
    kinetic_energy = ek_tot_0

    # solve ODE using euler method for all mass shells v
    for ii in range(time_len - 1):
        if neutron_precursor_switch:
            td_v[:-1, ii] = (kappa_r[:-1, ii] * m_array[:-1] * solar_mass * 3) / (
                    4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
            tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa_r[:-1, ii] / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))
        else:
            td_v[:-1, ii] = (kappa_r * m_array[:-1] * solar_mass * 3) / (
                        4 * np.pi * v_m[:-1] * speed_of_light * time[ii] * beta)
        lum_rad[:-1, ii] = energy_v[:-1, ii] / (td_v[:-1, ii] + time[ii] * (v_m[:-1] / speed_of_light))
        energy_v[:-1, ii + 1] = (edotr[:-1, ii] - (energy_v[:-1, ii] / time[ii]) - lum_rad[:-1, ii]) * dt[ii] + energy_v[:-1, ii]
        lum_rad[:-1, ii] = lum_rad[:-1, ii] * dm * solar_mass
        tau[:-1, ii] = (m_array[:-1] * solar_mass * kappa_r / (4 * np.pi * (time[ii] * v_m[:-1]) ** 2))

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
        Times at which we interpolate the `gwemlightcurves` values
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
    This is currently very inefficient as there is an interpolation step.

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


