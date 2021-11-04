from astropy.cosmology import Planck18 as cosmo  # noqa

from redback.utils import logger

try:
    import afterglowpy as afterglow

    jettype_dict = {'tophat': afterglow.jet.TopHat, 'gaussian': afterglow.jet.Gaussian,
                    'powerlaw_w_core': afterglow.jet.PowerLawCore, 'gaussian_w_core': afterglow.jet.GaussianCore,
                    'cocoon': afterglow.Spherical, 'smooth_power_law': afterglow.jet.PowerLaw,
                    'cone': afterglow.jet.Cone}
    spectype_dict = {'no_inverse_compton': 0, 'inverse_compton': 1}
except ModuleNotFoundError as e:
    logger.warning(e)
    afterglow = None


def cocoon(time, redshift, umax, umin, loge0, k, mej, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['cocoon']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'uMax': umax, 'Er': e0,
         'uMin': umin, 'k': k, 'MFast_solar': mej, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def kn_afterglow(time, redshift, umax, umin, loge0, k, mej, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['cocoon']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'uMax': umax, 'Er': e0,
         'uMin': umin, 'k': k, 'MFast_solar': mej, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def cone_afterglow(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['cone']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def gaussiancore(time, redshift, thv, loge0, thc, thw, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['gaussian_w_core']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']

    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def gaussian(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['gaussian']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def smoothpowerlaw(time, redshift, thv, loge0, thw, thc, beta, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['smooth_power_law']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw, 'b': beta}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def powerlawcore(time, redshift, thv, loge0, thw, thc, beta, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['powerlaw_w_core']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw, 'b': beta}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density


def tophat(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    dl = cosmo.luminosity_distance(redshift).cgs.value
    spread = kwargs['spread']
    latres = kwargs['latres']
    tres = kwargs['tres']
    jettype = jettype_dict['tophat']
    spectype = kwargs['spectype']
    frequency = kwargs['frequency']

    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': 0, 'q': 0, 'ts': 0, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    return flux_density
