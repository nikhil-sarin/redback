import numpy as np
from collections import namedtuple
from redback.constants import *
import redback.sed as sed
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper, logger

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def _shock_cooling(time, mass, radius, energy, **kwargs):
    nn = kwargs.get('nn',10)
    delta = kwargs.get('delta',1.1)
    kk_pow = (nn - 3) * (3 - delta) / (4 * np.pi * (nn - delta))
    kappa = 0.2
    vt = (((nn - 5) * (5 - delta) / ((nn - 3) * (3 - delta))) * (2 * energy / mass))**0.5
    td = ((3 * kappa * kk_pow * mass) / ((nn - 1) * vt * speed_of_light))**0.5

    prefactor = np.pi * (nn - 1) / (3 * (nn - 5)) * speed_of_light * radius * vt**2 / kappa
    lbol_pre_td = prefactor * np.power(td / time, 4 / (nn - 2))
    lbol_post_td = prefactor * np.exp(-0.5 * (time * time / td / td - 1))
    lbol = np.zeros(len(time))
    lbol[time < td] = lbol_pre_td[time < td]
    lbol[time >= td] = lbol_post_td[time >= td]

    tph = np.sqrt(3 * kappa * kk_pow * mass / (2 * (nn - 1) * vt * vt))
    r_photosphere_pre_td = np.power(tph / time, 2 / (nn - 1)) * vt * time
    r_photosphere_post_td = (np.power((delta - 1) / (nn - 1) * ((time / td) ** 2 - 1) + 1, -1 / (delta + 1))* vt * time)
    r_photosphere = np.zeros(len(time))
    r_photosphere[time < td] = r_photosphere_pre_td[time < td]
    r_photosphere[time >= td] = r_photosphere_post_td[time >= td]

    sigmaT4 = lbol / (4 * np.pi * r_photosphere**2)
    temperature = np.power(sigmaT4 / sigma_sb, 0.25)

    output = namedtuple('output', ['lbol', 'r_photosphere', 'temperature'])
    output.lbol = lbol
    output.r_photosphere = r_photosphere
    output.temperature = temperature
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def shock_cooling_bolometric(time, log10_mass, log10_radius, log10_energy, **kwargs):
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy
    output = _shock_cooling(time, mass=mass, radius=radius, energy=energy, **kwargs)
    return output.lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
def shock_cooling(time, redshift, log10_mass, log10_radius, log10_energy, **kwargs):
    mass = 10 ** log10_mass
    radius = 10 ** log10_radius
    energy = 10 ** log10_energy

    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    output = _shock_cooling(time, mass=mass, radius=radius, energy=energy, **kwargs)
    flux_density = sed.blackbody_to_flux_density(temperature=output.temperature, r_photosphere=output.r_photosphere,
                                             dl=dl, frequency=frequency)

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def _thermal_synchrotron():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def thermal_synchrotron_bolometric():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...923L..14M/abstract')
def thermal_synchrotron():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def _shocked_cocoon():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon_bolometric():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
def shocked_cocoon():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022ApJ...928..122M/abstract')
def csm_truncation_shock():
    pass
