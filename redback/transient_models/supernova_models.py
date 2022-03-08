import numpy as np
from redback.transient_models.phenomenological_models import exponential_powerlaw
import redback.interaction_processes as ip
import redback.sed as sed
import redback.photosphere as photosphere
from astropy.cosmology import Planck18 as cosmo  # noqa
from redback.utils import calc_kcorrected_properties, citation_wrapper
import astropy.units as uu

def thermal_synchrotron():
    """
    From Margalit paper ...

    :return:
    """
    pass

@citation_wrapper('redback')
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

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def sn_exponential_powerlaw(time, redshift, lbol_0, alpha_1, alpha_2, tpeak_d,
                            interaction_process = ip.Diffusion,
                            photosphere=photosphere.TemperatureFloor,
                            sed=sed.Blackbody,**kwargs):
    """
    :param time: observer frame time in days
    :param redshift: source redshift
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
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    lbol = exponential_powerlaw_bolometric(time=time, lbol_0=lbol_0,
                                           alpha_1=alpha_1,alpha_2=alpha_2, tpeak_d=tpeak_d,
                                           interaction_process=interaction_process, **kwargs)
    photo = photosphere(time=time, luminosity=lbol, **kwargs)
    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
              frequency=frequency, luminosity_distance=dl)

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

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
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

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett(time, redshift, f_nickel, mej, interaction_process=ip.Diffusion,
           photosphere=photosphere.TemperatureFloor,
           sed=sed.Blackbody, **kwargs):
    """
    :param time: time in days
    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    frequency = kwargs['frequency']
    # time = time * 86400
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, interaction_process=interaction_process, **kwargs)
    photo = photosphere(time=time, luminosity=lbol, **kwargs)
    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                frequency=frequency, luminosity_distance=dl)

    flux_density = sed_1.flux_density

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

def _basic_magnetar(time, p0, bp, mass_ns, theta_pb, **kwargs):
    """
    :param time: time in seconds in source frame
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param kwargs: None
    :return: luminosity
    """
    erot = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
    tp = 1.3e5 * bp**(-2) * p0**2 * (mass_ns/1.4)**(3./2.) * (np.sin(theta_pb))**(-2)
    luminosity = 2 * erot / tp / (1. + 2 * time / tp)**2
    return luminosity

@citation_wrapper('redback')
def basic_magnetar_powered_bolometric(time, p0, bp, mass_ns, theta_pb,interaction_process=ip.Diffusion, **kwargs):
    """
    :param time: time in days in source frame
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """
    lbol = _basic_magnetar(time=time*86400, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    if interaction_process is not None:
        interaction_class = interaction_process(time=time, luminosity=lbol, **kwargs)
        lbol = interaction_class.new_luminosity
    return lbol

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract')
def basic_magnetar_powered(time, redshift, p0, bp, mass_ns, theta_pb, interaction_process=ip.Diffusion,
                            photosphere=photosphere.TemperatureFloor, sed=sed.Blackbody,**kwargs):
    """
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    lbol = basic_magnetar_powered_bolometric(time=time, p0=p0,bp=bp, mass_ns=mass_ns, theta_pb=theta_pb,
                                     interaction_process=interaction_process, **kwargs)
    photo = photosphere(time=time, luminosity=lbol, **kwargs)

    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                frequency=frequency, luminosity_distance=dl)

    flux_density = sed_1.flux_density

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('redback')
def slsn_bolometric(time, p0, bp, mass_ns, theta_pb,interaction_process=ip.Diffusion, **kwargs):
    """
    Same as basic magnetar_powered but with constraint on rotational_energy/kinetic_energy and nebula phase

    :param time: time in days in source frame
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param kwargs: Must be all the kwargs required by the specific interaction_process
             e.g., for Diffusion: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: bolometric_luminosity
    """
    return basic_magnetar_powered_bolometric(time=time, p0=p0, bp=bp, mass_ns=mass_ns,
                                             theta_pb=theta_pb, interaction_process=interaction_process, **kwargs)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract')
def slsn(time, redshift, p0, bp, mass_ns, theta_pb, interaction_process=ip.Diffusion,
         photosphere=photosphere.TemperatureFloor, sed=sed.CutoffBlackbody,**kwargs):
    """
    Same as basic magnetar_powered but with constraint on rotational_energy/kinetic_energy and nebula phase

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is CutoffBlackbody.
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    frequency = kwargs['frequency']
    cutoff_wavelength = kwargs.get('cutoff_wavelength', 3000)
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value
    lbol = slsn_bolometric(time=time, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb,
                           interaction_process=interaction_process)
    photo = photosphere(time=time, luminosity=lbol, **kwargs)
    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                frequency=frequency, luminosity_distance=dl, cutoff_wavelength=cutoff_wavelength)

    flux_density = sed_1.flux_density

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def magnetar_nickel(time, redshift, f_nickel, mej, p0, bp, mass_ns, theta_pb, interaction_process=ip.Diffusion,
                    photosphere=photosphere.TemperatureFloor, sed=sed.Blackbody, **kwargs):
    """
    :param time: time in days in observer frame
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param redshift: source redshift
    :param p0: initial spin period
    :param bp: polar magnetic field strength in Gauss
    :param mass_ns: mass of neutron star in solar masses
    :param theta_pb: angle between spin and magnetic field axes
    :param interaction_process: Default is Diffusion.
            Can also be None in which case the output is just the raw engine luminosity
    :param photosphere: Default is TemperatureFloor.
            kwargs must have vej or relevant parameters if using different photosphere model
    :param sed: Default is blackbody.
    :param kwargs: Must be all the kwargs required by the specific interaction_process, photosphere, sed methods used
             e.g., for Diffusion and TemperatureFloor: kappa, kappa_gamma, vej (km/s), temperature_floor
    :return: flux_density or magnitude depending on output_format kwarg
    """
    frequency = kwargs['frequency']
    frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
    dl = cosmo.luminosity_distance(redshift).cgs.value

    lbol_mag = _basic_magnetar(time=time*86400, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
    lbol_arnett = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
    lbol = lbol_mag + lbol_arnett

    if interaction_process is not None:
        interaction_class = interaction_process(time=time, luminosity=lbol, **kwargs)
        lbol = interaction_class.new_luminosity

    photo = photosphere(time=time, luminosity=lbol, **kwargs)

    sed_1 = sed(temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                frequency=frequency, luminosity_distance=dl)

    flux_density = sed_1.flux_density

    if kwargs['output_format'] == 'flux_density':
        return flux_density.to(uu.mJy).value
    elif kwargs['output_format'] == 'magnitude':
        return flux_density.to(uu.ABmag).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract')
def csm_interaction():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def csm_nickel():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def type_1a():
    pass

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
def type_1c():
    pass

@citation_wrapper('redback')
def homologous_expansion_supernova_model_bolometric(time, **kwargs):
    v_ejecta = np.sqrt(10.0 * self._energy * FOE /
                       (3.0 * self._m_ejecta * M_SUN_CGS)) / KM_CGS
    pass

@citation_wrapper('redback')
def homologous_expansion_supernova_model():
    pass

@citation_wrapper('redback')
def thin_shell_supernova_model_bolometric():
    pass

@citation_wrapper('redback')
def thin_shell_supernova_model():
    pass

@citation_wrapper('redback')
def general_magnetar_slsn():
    pass

