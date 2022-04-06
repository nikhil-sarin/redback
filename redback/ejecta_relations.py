import numpy as np
from astropy import constants as cc
from redback.utils import citation_wrapper


class OneComponentBNSNoProjection(object):
    """
    Relations to connect intrinsic GW parameters to extrinsic kilonova parameters from Dietrich and Ujevic 2017
    for a one component BNS kilonova model.
    """
    def __init__(self, mass_1, mass_2, lambda_1, lambda_2):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.c1 = calc_compactness_from_lambda(self.lambda_1)
        self.c2 = calc_compactness_from_lambda(self.lambda_2)
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2017CQGra..34j5014D/abstract'
        self.ejecta_mass = self.calculate_ejecta_mass()
        self.ejecta_velocity = self.calculate_ejecta_velocity()

    @property
    def calculate_ejecta_velocity(self):
        c1 = self.c1
        c2 = self.c2

        a = -0.3090
        b = 0.657
        c = -1.879
        vej = a * (self.mass_1 / self.mass_2) * (1 + c * c1) + a * (self.mass_2 / self.mass_1) * (1 + c * c2) + b
        return vej

    @property
    def calculate_ejecta_mass(self):
        c1 = self.c1
        c2 = self.c2

        a = -0.0719
        b = 0.2116
        d = -2.42
        n = -2.905

        log10_mej = a * (self.mass_1 * (1 - 2 * c1) / c1 + self.mass_2 * (1 - 2 * c2) / c2) + b * \
                    (self.mass_1 * (self.mass_2 / self.mass_1) ** n + self.mass_2 * (self.mass_1 / self.mass_2) ** n) + d

        mej = 10 ** log10_mej
        return mej

class OneComponentNSBH(object):
    """
    Relations to connect intrinsic GW parameters to extrinsic kilonova parameters
    for a neutron star black hole merger with one component (zone) from Kawaguchi et al. 2016.
    """
    def __init__(self, mass_bh, mass_ns, chi_eff, lambda_ns):
        self.mass_bh = mass_bh
        self.mass_ns = mass_ns
        self.mass_ratio = mass_bh/mass_ns
        self.chi_eff = chi_eff
        self.lambda_ns = lambda_ns
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2016ApJ...825...52K/abstract'
        self.risco = self.isco_radius()
        self.ejecta_velocity = self.calculate_ejecta_velocity()
        self.ejecta_mass = self.calculate_ejecta_mass()

    @property
    def isco_radius(self):
        chi = self.chi_eff
        z1 = 1 + ((1 - chi * chi) ** (1 / 3.0)) * (((1 + chi) ** (1 / 3.0)) + (1 - chi) ** (1 / 3.0))
        z2 = (3 * chi * chi + z1 * z1) ** (1 / 2.0)
        risco = 3 + z2 - np.sign(chi) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2.0)
        return risco

    @property
    def calculate_ejecta_velocity(self):
        vej = 1.5333330951369120e-2*self.mass_ratio+0.19066667068621043
        return vej

    @property
    def calculate_ejecta_mass(self):
        a1 = -2.269e-3
        a2 = 4.464e-2
        a3 = 2.431
        a4 = -0.4159
        n1 = 1.352
        n2 = 0.2497

        compactness = calc_compactness_from_lambda(self.lambda_ns)
        baryonic_mass = calc_baryonic_mass(mass=self.mass_ns, compactness=compactness)

        tmp1 = self.risco * (self.mass_ratio ** n1) * a1
        tmp2 = (self.mass_ratio ** n2) * (1 - 2 * compactness) * a2 / compactness
        tmp3 = (1 - self.mass_ns / baryonic_mass) * a3 + a4
        mej = baryonic_mass * np.maximum(tmp1 + tmp2 + tmp3, 0)
        return mej


def calc_compactness_from_lambda(lambda_1):
    """
    :param lambda_1: dimensionless tidal deformability
    :return: compactness
    """
    c1 = 0.371 - 0.0391 * np.log(lambda_1) + 0.001056 * np.log(lambda_1) ** 2
    return c1


def calc_compactness(mass, radius):
    """
    :param mass: in solar masses
    :param radius: in meters
    :return: compactness
    """
    mass_si = mass * cc.M_sun.si.value
    radius_si = radius #meters
    g_si = cc.G.value
    c_si = cc.c.value
    compactness=g_si*mass_si / ( c_si**2 *radius_si)
    return compactness


def calc_baryonic_mass_eos_insensitive(mass_g, radius_14):
    """
    :param mass_g: gravitational mass in solar mass
    :param radius_14: radius of 1.4 M_sun neutron star in meters
    :return: baryonic mass
    """
    mb = mass_g + radius_14**(-1.) * mass_g**2
    return mb


def calc_baryonic_mass(mass, compactness):
    """
    :param mass: mass in solar masses
    :param compactness: NS compactness
    :return: baryonic mass
    """
    mb = mass*(1 + 0.8857853174243745*compactness**1.2082383572002926)
    return mb

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017CQGra..34j5014D/abstract')
def calc_vrho(mass_1,mass_2,lambda_1,lambda_2):
    """
    Average velocity in the orbital plane

    :param mass_1: mass of primary neutron star
    :param mass_2: mass of secondary neutron star
    :param lambda_1: tidal deformability of primary neutron star
    :param lambda_2: tidal deformability of secondary neutron star
    :return: average velocity in the orbital plane
    """
    a=-0.219479
    b=0.444836
    c=-2.67385
    compactness_1 = calc_compactness_from_lambda(lambda_1)
    compactness_2 = calc_compactness_from_lambda(lambda_2)

    return ((mass_1/mass_2)*(1.0+c*compactness_1)+(mass_2/mass_1)*(1.0+c*compactness_2))*a+b

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017CQGra..34j5014D/abstract')
def calc_vz(mass_1,mass_2,lambda_1,lambda_2):
    """
    Velocity orthogonal to the orbital plane (z direction)

    :param mass_1: mass of primary neutron star
    :param mass_2: mass of secondary neutron star
    :param lambda_1: tidal deformability of primary neutron star
    :param lambda_2: tidal deformability of secondary neutron star
    :return: average velocity orthogonal to the orbital plane
    """
    a=-0.315585
    b=0.63808
    c=-1.00757
    compactness_1 = calc_compactness_from_lambda(lambda_1)
    compactness_2 = calc_compactness_from_lambda(lambda_2)

    return ((mass_1/mass_2)*(1.0+c*compactness_1)+(mass_2/mass_1)*(1.0+c*compactness_2))*a+b