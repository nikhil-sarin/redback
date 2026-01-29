"""
VegasAfterglow Models for Redback
==================================

High-performance C++ afterglow models via VegasAfterglow.
Provides millisecond-scale light curve evaluation for all jet structures.

Each model uses a unified interface supporting ISM, Wind, and Hybrid media
through lognism and loga parameters.

Citation: Wang, Chen & Zhang (2026), https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/

Models:
-------
Unified Medium (ISM/Wind/Hybrid via parameters):
    - vegas_tophat
    - vegas_gaussian
    - vegas_powerlaw
    - vegas_powerlaw_wing
    - vegas_two_component
    - vegas_step_powerlaw

Medium Configuration:
---------------------
Pure ISM:    lognism=0.0,    loga=-np.inf
Pure Wind:   lognism=-np.inf, loga=11.0
Hybrid:      lognism=0.0,    loga=11.0 (wind with ISM floor)
"""

from astropy.cosmology import Planck18 as cosmo
from redback.utils import citation_wrapper, calc_ABmag_from_flux_density, bands_to_frequency
from redback.constants import day_to_s
import numpy as np

try:
    from VegasAfterglow import (
        Wind, TophatJet, GaussianJet, PowerLawJet, PowerLawWing,
        TwoComponentJet, StepPowerLawJet, Observer, Radiation, Model, Magnetar
    )
    VEGASAFTERGLOW_AVAILABLE = True
except ImportError:
    VEGASAFTERGLOW_AVAILABLE = False


def _check_vegasafterglow_available():
    """Check if VegasAfterglow is installed"""
    if not VEGASAFTERGLOW_AVAILABLE:
        raise ImportError(
            "VegasAfterglow is not installed. "
            "Install with: pip install VegasAfterglow\n"
            "See: https://github.com/YihanWangAstro/VegasAfterglow"
        )


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/abstract')
def vegas_tophat(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, **kwargs):
    """
    VegasAfterglow tophat jet with unified medium (ISM/Wind/Hybrid)

    High-performance C++ implementation providing ~1ms per light curve evaluation.

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 isotropic equivalent energy [erg]
    :param thc: jet core opening angle in radians
    :param lognism: log10 ISM number density [cm^-3] (use -inf for pure wind)
    :param loga: log10 wind parameter A_* [g/cm] (use -inf for pure ISM)
    :param p: electron power-law index
    :param logepse: log10 electron energy fraction
    :param logepsb: log10 magnetic field energy fraction
    :param g0: initial Lorentz factor
    :param kwargs: Additional parameters
        - frequency: frequency array in Hz (required for flux_density mode)
        - output_format: 'flux_density' or magnitude
        - bands: photometric bands for magnitude calculation
        - cosmology: astropy cosmology object (default: Planck18)
        - phiv: azimuthal viewing angle [rad] (default: 0.0)
        - xie: electron acceleration efficiency (default: 1.0)
        - spreading: enable jet spreading (default: False)
        - duration: shell duration [s] (default: 1.0)
        - wind_k: wind power-law index (default: 2.0)
        - wind_n0: wind transition radius density [cm^-3] (default: inf)
        - reverse_shock: enable reverse shock (default: False)
        - reverse_logepse: reverse shock electron fraction (default: logepse)
        - reverse_logepsb: reverse shock magnetic fraction (default: logepsb)
        - reverse_p: reverse shock electron index (default: p)
        - reverse_xie: reverse shock electron efficiency (default: 1.0)
        - ssc: enable synchrotron self-Compton (default: False)
        - ssc_cooling: enable SSC cooling (default: False)
        - kn: enable Klein-Nishina corrections (default: False)
        - resolutions: (phi_res, theta_res, time_res) tuple (default: (0.3, 1, 10))
        - rtol: relative tolerance for ODE solver (default: 1e-6)
        - axisymmetric: assume axisymmetric jet (default: True)
        - magnetar_L0: magnetar luminosity [erg/s] (optional)
        - magnetar_t0: magnetar spin-down time [s] (optional)
        - magnetar_q: magnetar braking index (optional)
    :return: flux density [erg/cm²/s/Hz] or AB magnitude
    
    Examples:
    ---------
    Pure ISM: lognism=0.0, loga=-np.inf
    Pure Wind: lognism=-np.inf, loga=11.0
    Hybrid: lognism=0.0, loga=11.0 (wind with ISM floor)
    """
    _check_vegasafterglow_available()

    # Convert time to seconds
    time_s = time * day_to_s

    # Get cosmology and luminosity distance
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    # Get frequency
    frequency = kwargs.get('frequency')
    if frequency is None:
        raise ValueError("frequency must be provided in kwargs for flux_density mode")

    # Build unified medium (handles ISM, Wind, and Hybrid)
    n_ism = 0.0 if lognism == -np.inf else 10**lognism
    A_star = 0.0 if loga == -np.inf else 10**loga
    wind_k = kwargs.get('wind_k', 2.0)
    wind_n0 = kwargs.get('wind_n0', float('inf'))
    
    medium = Wind(A_star=A_star, n_ism=n_ism, k=wind_k, n0=wind_n0)

    # Jet parameters
    spreading = kwargs.get('spreading', False)
    duration = kwargs.get('duration', 1.0)
    
    # Magnetar (optional)
    magnetar = None
    if 'magnetar_L0' in kwargs:
        magnetar = Magnetar(
            L0=kwargs['magnetar_L0'],
            t0=kwargs['magnetar_t0'],
            q=kwargs['magnetar_q']
        )
    
    jet = TophatJet(
        theta_c=thc, 
        E_iso=10**loge0, 
        Gamma0=g0, 
        spreading=spreading,
        duration=duration,
        magnetar=magnetar
    )

    # Observer parameters
    phiv = kwargs.get('phiv', 0.0)
    obs = Observer(lumi_dist=dl, z=redshift, theta_obs=thv, phi_obs=phiv)

    # Forward shock radiation
    xie = kwargs.get('xie', 1.0)
    ssc = kwargs.get('ssc', False)
    ssc_cooling = kwargs.get('ssc_cooling', False)
    kn = kwargs.get('kn', False)
    
    rad_fwd = Radiation(
        eps_e=10**logepse, 
        eps_B=10**logepsb, 
        p=p,
        xi_e=xie,
        ssc=ssc,
        ssc_cooling=ssc_cooling,
        kn=kn
    )

    # Reverse shock (optional)
    rad_rvs = None
    if kwargs.get('reverse_shock', False):
        reverse_xie = kwargs.get('reverse_xie', 1.0)
        rad_rvs = Radiation(
            eps_e=10**kwargs.get('reverse_logepse', logepse),
            eps_B=10**kwargs.get('reverse_logepsb', logepsb),
            p=kwargs.get('reverse_p', p),
            xi_e=reverse_xie,
            ssc=ssc,
            ssc_cooling=ssc_cooling,
            kn=kn
        )

    # Model resolution and numerical parameters
    resolutions = kwargs.get('resolutions', (0.3, 1, 10))
    rtol = kwargs.get('rtol', 1e-6)
    axisymmetric = kwargs.get('axisymmetric', True)

    # Create model
    model = Model(
        jet=jet, 
        medium=medium, 
        observer=obs, 
        fwd_rad=rad_fwd, 
        rvs_rad=rad_rvs,
        resolutions=resolutions,
        rtol=rtol,
        axisymmetric=axisymmetric
    )

    # Evaluate
    if kwargs['output_format'] == 'flux_density':
        flux_density = model.flux_density(time_s, frequency)
        return flux_density.total
    else:
        # Magnitude mode
        frequency = bands_to_frequency(kwargs['bands'])
        flux_density = model.flux_density(time_s, frequency)
        return calc_ABmag_from_flux_density(flux_density.total).value


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/abstract')
def vegas_gaussian(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, **kwargs):
    """
    VegasAfterglow Gaussian jet with unified medium (ISM/Wind/Hybrid)

    Parameters same as vegas_tophat. Uses GaussianJet instead of TophatJet.
    """
    _check_vegasafterglow_available()

    # Same setup as tophat
    time_s = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs.get('frequency')
    if frequency is None:
        raise ValueError("frequency must be provided in kwargs")

    # Medium
    n_ism = 0.0 if lognism == -np.inf else 10**lognism
    A_star = 0.0 if loga == -np.inf else 10**loga
    medium = Wind(
        A_star=A_star, 
        n_ism=n_ism, 
        k=kwargs.get('wind_k', 2.0), 
        n0=kwargs.get('wind_n0', float('inf'))
    )

    # Jet (Gaussian instead of Tophat)
    magnetar = None
    if 'magnetar_L0' in kwargs:
        magnetar = Magnetar(L0=kwargs['magnetar_L0'], t0=kwargs['magnetar_t0'], q=kwargs['magnetar_q'])
    
    jet = GaussianJet(
        theta_c=thc, 
        E_iso=10**loge0, 
        Gamma0=g0, 
        spreading=kwargs.get('spreading', False),
        duration=kwargs.get('duration', 1.0),
        magnetar=magnetar
    )

    # Rest same as tophat
    obs = Observer(lumi_dist=dl, z=redshift, theta_obs=thv, phi_obs=kwargs.get('phiv', 0.0))
    
    xie = kwargs.get('xie', 1.0)
    ssc = kwargs.get('ssc', False)
    ssc_cooling = kwargs.get('ssc_cooling', False)
    kn = kwargs.get('kn', False)
    
    rad_fwd = Radiation(eps_e=10**logepse, eps_B=10**logepsb, p=p, xi_e=xie, ssc=ssc, ssc_cooling=ssc_cooling, kn=kn)
    
    rad_rvs = None
    if kwargs.get('reverse_shock', False):
        rad_rvs = Radiation(
            eps_e=10**kwargs.get('reverse_logepse', logepse),
            eps_B=10**kwargs.get('reverse_logepsb', logepsb),
            p=kwargs.get('reverse_p', p),
            xi_e=kwargs.get('reverse_xie', 1.0),
            ssc=ssc, ssc_cooling=ssc_cooling, kn=kn
        )

    model = Model(
        jet=jet, medium=medium, observer=obs, fwd_rad=rad_fwd, rvs_rad=rad_rvs,
        resolutions=kwargs.get('resolutions', (0.3, 1, 10)),
        rtol=kwargs.get('rtol', 1e-6),
        axisymmetric=kwargs.get('axisymmetric', True)
    )

    if kwargs['output_format'] == 'flux_density':
        return model.flux_density(time_s, frequency).total
    else:
        frequency = bands_to_frequency(kwargs['bands'])
        return calc_ABmag_from_flux_density(model.flux_density(time_s, frequency).total).value


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/abstract')
def vegas_powerlaw(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, ke, kg, **kwargs):
    """
    VegasAfterglow power-law jet with unified medium (ISM/Wind/Hybrid)

    Parameters same as vegas_tophat plus:
    :param ke: energy power-law index
    :param kg: Lorentz factor power-law index
    """
    _check_vegasafterglow_available()

    time_s = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs.get('frequency')
    if frequency is None:
        raise ValueError("frequency must be provided in kwargs")

    n_ism = 0.0 if lognism == -np.inf else 10**lognism
    A_star = 0.0 if loga == -np.inf else 10**loga
    medium = Wind(A_star=A_star, n_ism=n_ism, k=kwargs.get('wind_k', 2.0), n0=kwargs.get('wind_n0', float('inf')))

    magnetar = None
    if 'magnetar_L0' in kwargs:
        magnetar = Magnetar(L0=kwargs['magnetar_L0'], t0=kwargs['magnetar_t0'], q=kwargs['magnetar_q'])
    
    jet = PowerLawJet(
        theta_c=thc, E_iso=10**loge0, Gamma0=g0, k_e=ke, k_g=kg,
        spreading=kwargs.get('spreading', False),
        duration=kwargs.get('duration', 1.0),
        magnetar=magnetar
    )

    obs = Observer(lumi_dist=dl, z=redshift, theta_obs=thv, phi_obs=kwargs.get('phiv', 0.0))
    
    xie = kwargs.get('xie', 1.0)
    ssc = kwargs.get('ssc', False)
    ssc_cooling = kwargs.get('ssc_cooling', False)
    kn = kwargs.get('kn', False)
    
    rad_fwd = Radiation(eps_e=10**logepse, eps_B=10**logepsb, p=p, xi_e=xie, ssc=ssc, ssc_cooling=ssc_cooling, kn=kn)
    
    rad_rvs = None
    if kwargs.get('reverse_shock', False):
        rad_rvs = Radiation(
            eps_e=10**kwargs.get('reverse_logepse', logepse),
            eps_B=10**kwargs.get('reverse_logepsb', logepsb),
            p=kwargs.get('reverse_p', p),
            xi_e=kwargs.get('reverse_xie', 1.0),
            ssc=ssc, ssc_cooling=ssc_cooling, kn=kn
        )

    model = Model(
        jet=jet, medium=medium, observer=obs, fwd_rad=rad_fwd, rvs_rad=rad_rvs,
        resolutions=kwargs.get('resolutions', (0.3, 1, 10)),
        rtol=kwargs.get('rtol', 1e-6),
        axisymmetric=kwargs.get('axisymmetric', True)
    )

    if kwargs['output_format'] == 'flux_density':
        return model.flux_density(time_s, frequency).total
    else:
        frequency = bands_to_frequency(kwargs['bands'])
        return calc_ABmag_from_flux_density(model.flux_density(time_s, frequency).total).value


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/abstract')
def vegas_powerlaw_wing(time, redshift, thv, loge0_w, thc, lognism, loga, p, logepse, logepsb, g0_w, ke, kg, **kwargs):
    """
    VegasAfterglow power-law wing jet with unified medium (ISM/Wind/Hybrid)

    :param loge0_w: log10 isotropic energy for wing [erg]
    :param g0_w: initial Lorentz factor for wing
    :param ke: energy power-law index
    :param kg: Lorentz factor power-law index
    """
    _check_vegasafterglow_available()

    time_s = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs.get('frequency')
    if frequency is None:
        raise ValueError("frequency must be provided in kwargs")

    n_ism = 0.0 if lognism == -np.inf else 10**lognism
    A_star = 0.0 if loga == -np.inf else 10**loga
    medium = Wind(A_star=A_star, n_ism=n_ism, k=kwargs.get('wind_k', 2.0), n0=kwargs.get('wind_n0', float('inf')))

    jet = PowerLawWing(
        theta_c=thc, E_iso_w=10**loge0_w, Gamma0_w=g0_w, k_e=ke, k_g=kg,
        spreading=kwargs.get('spreading', False),
        duration=kwargs.get('duration', 1.0)
    )

    obs = Observer(lumi_dist=dl, z=redshift, theta_obs=thv, phi_obs=kwargs.get('phiv', 0.0))
    
    xie = kwargs.get('xie', 1.0)
    ssc = kwargs.get('ssc', False)
    ssc_cooling = kwargs.get('ssc_cooling', False)
    kn = kwargs.get('kn', False)
    
    rad_fwd = Radiation(eps_e=10**logepse, eps_B=10**logepsb, p=p, xi_e=xie, ssc=ssc, ssc_cooling=ssc_cooling, kn=kn)
    
    rad_rvs = None
    if kwargs.get('reverse_shock', False):
        rad_rvs = Radiation(
            eps_e=10**kwargs.get('reverse_logepse', logepse),
            eps_B=10**kwargs.get('reverse_logepsb', logepsb),
            p=kwargs.get('reverse_p', p),
            xi_e=kwargs.get('reverse_xie', 1.0),
            ssc=ssc, ssc_cooling=ssc_cooling, kn=kn
        )

    model = Model(
        jet=jet, medium=medium, observer=obs, fwd_rad=rad_fwd, rvs_rad=rad_rvs,
        resolutions=kwargs.get('resolutions', (0.3, 1, 10)),
        rtol=kwargs.get('rtol', 1e-6),
        axisymmetric=kwargs.get('axisymmetric', True)
    )

    if kwargs['output_format'] == 'flux_density':
        return model.flux_density(time_s, frequency).total
    else:
        frequency = bands_to_frequency(kwargs['bands'])
        return calc_ABmag_from_flux_density(model.flux_density(time_s, frequency).total).value


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/abstract')
def vegas_two_component(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, 
                        theta_w, loge0_w, g0_w, **kwargs):
    """
    VegasAfterglow two-component jet with unified medium (ISM/Wind/Hybrid)

    :param theta_w: wide component opening angle [rad]
    :param loge0_w: log10 wide component isotropic energy [erg]
    :param g0_w: wide component initial Lorentz factor
    """
    _check_vegasafterglow_available()

    time_s = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs.get('frequency')
    if frequency is None:
        raise ValueError("frequency must be provided in kwargs")

    n_ism = 0.0 if lognism == -np.inf else 10**lognism
    A_star = 0.0 if loga == -np.inf else 10**loga
    medium = Wind(A_star=A_star, n_ism=n_ism, k=kwargs.get('wind_k', 2.0), n0=kwargs.get('wind_n0', float('inf')))

    magnetar = None
    if 'magnetar_L0' in kwargs:
        magnetar = Magnetar(L0=kwargs['magnetar_L0'], t0=kwargs['magnetar_t0'], q=kwargs['magnetar_q'])
    
    jet = TwoComponentJet(
        theta_c=thc, E_iso=10**loge0, Gamma0=g0,
        theta_w=theta_w, E_iso_w=10**loge0_w, Gamma0_w=g0_w,
        spreading=kwargs.get('spreading', False),
        duration=kwargs.get('duration', 1.0),
        magnetar=magnetar
    )

    obs = Observer(lumi_dist=dl, z=redshift, theta_obs=thv, phi_obs=kwargs.get('phiv', 0.0))
    
    xie = kwargs.get('xie', 1.0)
    ssc = kwargs.get('ssc', False)
    ssc_cooling = kwargs.get('ssc_cooling', False)
    kn = kwargs.get('kn', False)
    
    rad_fwd = Radiation(eps_e=10**logepse, eps_B=10**logepsb, p=p, xi_e=xie, ssc=ssc, ssc_cooling=ssc_cooling, kn=kn)
    
    rad_rvs = None
    if kwargs.get('reverse_shock', False):
        rad_rvs = Radiation(
            eps_e=10**kwargs.get('reverse_logepse', logepse),
            eps_B=10**kwargs.get('reverse_logepsb', logepsb),
            p=kwargs.get('reverse_p', p),
            xi_e=kwargs.get('reverse_xie', 1.0),
            ssc=ssc, ssc_cooling=ssc_cooling, kn=kn
        )

    model = Model(
        jet=jet, medium=medium, observer=obs, fwd_rad=rad_fwd, rvs_rad=rad_rvs,
        resolutions=kwargs.get('resolutions', (0.3, 1, 10)),
        rtol=kwargs.get('rtol', 1e-6),
        axisymmetric=kwargs.get('axisymmetric', True)
    )

    if kwargs['output_format'] == 'flux_density':
        return model.flux_density(time_s, frequency).total
    else:
        frequency = bands_to_frequency(kwargs['bands'])
        return calc_ABmag_from_flux_density(model.flux_density(time_s, frequency).total).value


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2026JHEAp..5000490W/abstract')
def vegas_step_powerlaw(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, 
                        loge0_w, g0_w, ke, kg, **kwargs):
    """
    VegasAfterglow step power-law jet with unified medium (ISM/Wind/Hybrid)

    :param loge0_w: log10 wide component isotropic energy [erg]
    :param g0_w: wide component initial Lorentz factor
    :param ke: energy power-law index
    :param kg: Lorentz factor power-law index
    """
    _check_vegasafterglow_available()

    time_s = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    frequency = kwargs.get('frequency')
    if frequency is None:
        raise ValueError("frequency must be provided in kwargs")

    n_ism = 0.0 if lognism == -np.inf else 10**lognism
    A_star = 0.0 if loga == -np.inf else 10**loga
    medium = Wind(A_star=A_star, n_ism=n_ism, k=kwargs.get('wind_k', 2.0), n0=kwargs.get('wind_n0', float('inf')))

    magnetar = None
    if 'magnetar_L0' in kwargs:
        magnetar = Magnetar(L0=kwargs['magnetar_L0'], t0=kwargs['magnetar_t0'], q=kwargs['magnetar_q'])
    
    jet = StepPowerLawJet(
        theta_c=thc, E_iso=10**loge0, Gamma0=g0,
        E_iso_w=10**loge0_w, Gamma0_w=g0_w, k_e=ke, k_g=kg,
        spreading=kwargs.get('spreading', False),
        duration=kwargs.get('duration', 1.0),
        magnetar=magnetar
    )

    obs = Observer(lumi_dist=dl, z=redshift, theta_obs=thv, phi_obs=kwargs.get('phiv', 0.0))
    
    xie = kwargs.get('xie', 1.0)
    ssc = kwargs.get('ssc', False)
    ssc_cooling = kwargs.get('ssc_cooling', False)
    kn = kwargs.get('kn', False)
    
    rad_fwd = Radiation(eps_e=10**logepse, eps_B=10**logepsb, p=p, xi_e=xie, ssc=ssc, ssc_cooling=ssc_cooling, kn=kn)
    
    rad_rvs = None
    if kwargs.get('reverse_shock', False):
        rad_rvs = Radiation(
            eps_e=10**kwargs.get('reverse_logepse', logepse),
            eps_B=10**kwargs.get('reverse_logepsb', logepsb),
            p=kwargs.get('reverse_p', p),
            xi_e=kwargs.get('reverse_xie', 1.0),
            ssc=ssc, ssc_cooling=ssc_cooling, kn=kn
        )

    model = Model(
        jet=jet, medium=medium, observer=obs, fwd_rad=rad_fwd, rvs_rad=rad_rvs,
        resolutions=kwargs.get('resolutions', (0.3, 1, 10)),
        rtol=kwargs.get('rtol', 1e-6),
        axisymmetric=kwargs.get('axisymmetric', True)
    )

    if kwargs['output_format'] == 'flux_density':
        return model.flux_density(time_s, frequency).total
    else:
        frequency = bands_to_frequency(kwargs['bands'])
        return calc_ABmag_from_flux_density(model.flux_density(time_s, frequency).total).value
