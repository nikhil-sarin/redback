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
    :param time: rest frame time in seconds
    :param lbol_0: bolometric luminosity scale in cgs
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak_d: peak time in days
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
                e.g., for Diffusion: kappa, kappa_gamma, mej (solar masses), vej (km/s)
    :return: bolometric_luminosity
    """
    lbol = exponential_powerlaw(time, a_1=lbol_0, alpha_1=alpha_1, alpha_2=alpha_2,
                                tpeak=tpeak_d*86400, **kwargs)
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
            e.g., for Diffusion: kappa, kappa_gamma, mej (solar masses), vej (km/s)
    :return: flux_density or magnitude depending on output_format
    """
    frequency = kwargs['frequency']
    time = time * 86400
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


