import numpy as np
from redback.transient_models.phenominological_models import exponential_powerlaw
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties
import astropy.units as uu

def thermal_synchrotron():
    """
    From Margalit paper ...

    :return:
    """
    pass

def exponential_powerlaw_bolometric(time, lbol_0, alpha_1, alpha_2, tpeak_d, interaction_process = ip.Diffusion,
                                    **kwargs):
    """
    :param time: rest frame time in days
    :param lbol_0: bolometric luminosity scale in cgs
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak_d: peak time in days
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
                e.g., for Diffusion: kappa, kappa_gamma, mej (solar masses), vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """
    lbol = exponential_powerlaw(time, a_1=lbol_0, alpha_1=alpha_1, alpha_2=alpha_2,
                                tpeak=tpeak_d, **kwargs)
    if interaction_process is not None:
        interaction_class = interaction_process(time=time, luminosity=lbol, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

def sn_exponential_powerlaw(time, redshift, lbol_0, alpha_1, alpha_2, tpeak_d,
                            interaction_process = ip.Diffusion,
                            photosphere=photosphere.TemperatureFloor,
                            sed=sed.Blackbody,**kwargs):
    """
    :param time: observer frame time in days
    :param lbol_0: bolometric luminosity scale in cgs
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak_d: peak time in days
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: Default is TemperatureFloor.
            kwargs must vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
            e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, mej (solar masses), vej (km/s), floor temperature
    :return: flux_density or magnitude depending on output_format kwarg
    """
    frequency = kwargs['frequency']
    # time = time * 86400
    frequency, time = calc_kcorrected_properties(frequencies=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    lbol = exponential_powerlaw_bolometric(time=time, lbol_0=lbol_0,
                                           alpha_1=alpha_1,alpha_2=alpha_2, tpeak_d=tpeak_d,
                                           interaction_process=interaction_process, **kwargs)
    photo = photosphere(time=time, luminosity=lbol, **kwargs)
    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
              frequencies=frequency, luminosity_distance=dl)

    flux_density = sed_1.flux_density

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def _nickelcobalt_engine(time, f_nickel, mej, **kwargs):
    """
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: None
    :return: bolometric_luminosity
    """
    '1994ApJS...92..527N'
    ni56_lum = 6.45e43
    co56_lum = 1.45e43
    ni56_life = 8.8  # days
    co56_life = 111.3  # days
    nickel_mass = f_nickel * mej
    lbol = nickel_mass * (ni56_lum*np.exp(-time/ni56_life) + co56_lum * np.exp(-time/co56_life))
    return lbol

def arnett_bolometric(time, f_nickel, mej, interaction_process=ip.Diffusion, **kwargs):
    """
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """

    lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
    if interaction_process is not None:
        interaction_class = interaction_process(time=time, luminosity=lbol, mej=mej, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

def arnett(time, redshift, f_nickel, mej, interaction_process=ip.Diffusion,
           photosphere=photosphere.TemperatureFloor,
           sed=sed.Blackbody, **kwargs):
    """
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: Default is TemperatureFloor.
            kwargs must vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    frequency = kwargs['frequency']
    # time = time * 86400
    frequency, time = calc_kcorrected_properties(frequencies=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, interaction_process=interaction_process, **kwargs)
    photo = photosphere(time=time, luminosity=lbol, **kwargs)
    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                frequencies=frequency, luminosity_distance=dl)

    flux_density = sed_1.flux_density

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def magnetar_nickel():
    pass

def superluminous_supernova():
    pass

def basic_magnetar_powered():
    pass

def braking_index_magnetar_powered():
    pass

def csm_interaction():
    pass

def csm_nickel():
    pass

def type_1a():
    pass

def type_1c():
    pass

def homologous_expansion_supernova_model():
    pass

def thin_shell_supernova_model():
    pass

