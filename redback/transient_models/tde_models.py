import numpy as np
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper
import astropy.units as uu

def _analytic_fallback(time, l0, t_0):
    """
    :param time: time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :return: bolometric luminosity
    """
    mask = time - t_0 > 0.
    lbol = np.zeros(len(time))
    lbol[mask] = l0 / (time[mask] * 86400)**(5./3.)
    lbol[~mask] = l0 / (t_0 * 86400)**(5./3.)
    return lbol

def _semianalytical_fallback():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def _metzger_tde(time, redshift, **kwargs):
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022arXiv220707136M/abstract')
def metzger_tde(time, redshift, **kwargs):
    pass

@citation_wrapper('redback')
def tde_analytical_bolometric(time, l0, t_0,interaction_process = ip.Diffusion,
                                    **kwargs):
    """
    :param time: rest frame time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
                e.g., for Diffusion: kappa, kappa_gamma, mej (solar masses), vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """

    lbol = _analytic_fallback(time=time, l0=l0, t_0=t_0)
    if interaction_process is not None:
        interaction_class = interaction_process(time=time, luminosity=lbol, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('redback')
def tde_analytical(time, redshift, l0, t_0, **kwargs):
    """
    :param time: rest frame time in days
    :param l0: bolometric luminosity at 1 second in cgs
    :param t_0: turn on time in days (after this time lbol decays as 5/3 powerlaw)
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: TemperatureFloor
    :param sed: CutoffBlackbody must have cutoff_wavelength in kwargs or it will default to 3000 Angstrom
    :param kwargs: Must be all the kwargs required by the specific interaction_process
     e.g., for Diffusion TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
    _photosphere = kwargs.get("photosphere", photosphere.TemperatureFloor)
    _sed = kwargs.get("sed", sed.CutoffBlackbody)

    frequency = kwargs['frequency']
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    lbol = tde_analytical_bolometric(time=time, l0=l0, t_0=t_0, interaction_process=_interaction_process, **kwargs)

    photo = _photosphere(time=time, luminosity=lbol, **kwargs)
    sed_1 = _sed(time=time, temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                 frequency=frequency, luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength, luminosity=lbol)

    flux_density = sed_1.flux_density
    flux_density = np.nan_to_num(flux_density)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract')
def tde_semianalytical():
    pass