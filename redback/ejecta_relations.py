import numpy as np
from astropy import constants as cc
from redback.utils import citation_wrapper

class OneComponentBNSNoProjection(object):
    def __init__(self, mass_1, mass_2, lambda_1, lambda_2):
        """
        Relations to connect intrinsic GW parameters to extrinsic kilonova parameters from Dietrich and Ujevic 2017
        for a one component BNS kilonova model assuming no orbital plane or orthogonal projection.

        :param mass_1: mass of primary neutron star
        :param mass_2: mass of secondary neutron star
        :param lambda_1: tidal deformability of primary neutron star
        :param lambda_2: tidal deformability of secondary neutron star
        """
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.c1 = calc_compactness_from_lambda(self.lambda_1)
        self.c2 = calc_compactness_from_lambda(self.lambda_2)
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2017CQGra..34j5014D/abstract'
        self.vrho = calc_vrho(mass_1=self.mass_1, mass_2=self.mass_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        self.vz = calc_vz(mass_1=self.mass_1, mass_2=self.mass_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        self.ejecta_mass = self.calculate_ejecta_mass
        self.ejecta_velocity = self.calculate_ejecta_velocity
        self.qej = self.calculate_qej
        self.phej = self.calculate_phej

    @property
    def calculate_ejecta_velocity(self):
        """
        Calculate ejecta velocity assuming no orbital plane or orthogonal projection

        :return: ejecta velocity in c
        """
        c1 = self.c1
        c2 = self.c2

        a = -0.3090
        b = 0.657
        c = -1.879
        vej = a * (self.mass_1 / self.mass_2) * (1 + c * c1) + a * (self.mass_2 / self.mass_1) * (1 + c * c2) + b
        return vej

    @property
    def calculate_ejecta_mass(self):
        """
        Calculate ejecta mass assuming one single component

        :return: ejecta mass in solar masses
        """
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

    @property
    def calculate_qej(self):
        """
        Polar opening angle

        :return: polar opening angle
        """
        tmp1 = 3. * self.vz + np.sqrt(9 * self.vz**2 + 4 * self.vrho**2)
        qej = ((2.0 ** (4.0 / 3.0)) * self.vrho**2 + (2. * self.vrho**2 * tmp1) ** (2.0 / 3.0)) / ((self.vrho ** 5.0) * tmp1) ** (
                    1.0 / 3.0)
        return qej

    @property
    def calculate_phej(self):
        """
        azimuthal opening angle

        :return: azimuthal opening angle
        """
        return 4.0 * self.qej * np.pi / 2.0

class OneComponentBNSProjection(object):
    def __init__(self, mass_1, mass_2, lambda_1, lambda_2):
        """
        Relations to connect intrinsic GW parameters to extrinsic kilonova parameters from Dietrich and Ujevic 2017
        for a one component BNS kilonova model assuming orbital plane and orthogonal projection.

        :param mass_1: mass of primary neutron star
        :param mass_2: mass of secondary neutron star
        :param lambda_1: tidal deformability of primary neutron star
        :param lambda_2: tidal deformability of secondary neutron star
        """
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.c1 = calc_compactness_from_lambda(self.lambda_1)
        self.c2 = calc_compactness_from_lambda(self.lambda_2)
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2017CQGra..34j5014D/abstract'
        self.vrho = calc_vrho(mass_1=self.mass_1, mass_2=self.mass_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        self.vz = calc_vz(mass_1=self.mass_1, mass_2=self.mass_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        self.ejecta_mass = self.calculate_ejecta_mass
        self.ejecta_velocity = self.calculate_ejecta_velocity
        self.qej = self.calculate_qej
        self.phej = self.calculate_phej

    @property
    def calculate_ejecta_velocity(self):
        """
        Calculate ejecta velocity assuming no orbital plane or orthogonal projection

        :return: ejecta velocity in c
        """
        vej = (self.vrho**2.0+self.vz**2.0)**0.5
        return vej

    @property
    def calculate_ejecta_mass(self):
        """
        Calculate ejecta mass assuming one single component

        :return: ejecta mass in solar masses
        """
        a = -1.35695
        b = 6.11252
        c = -49.43355
        d = 16.1144
        n = -2.5484

        c1 = self.c1
        c2 = self.c2
        mb1 = calc_baryonic_mass(mass=self.mass_1, compactness=c1)
        mb2 = calc_baryonic_mass(mass=self.mass_2, compactness=c2)

        tmp1 = ((mb1 * ((self.mass_2 / self.mass_1) ** (1.0 / 3.0)) * (1.0 - 2.0 * c1) / c1) + (
                    mb2 * ((self.mass_1 / self.mass_2) ** (1.0 / 3.0)) * (1.0 - 2.0 * c2) / c2)) * a
        tmp2 = (mb1 * ((self.mass_2 / self.mass_1) ** n) + mb2 * ((self.mass_1 / self.mass_2) ** n)) * b
        tmp3 = (mb1 * (1.0 - self.mass_1 / mb1) + mb2 * (1.0 - self.mass_2 / mb2)) * c

        mej = np.maximum(tmp1 + tmp2 + tmp3 + d, 0) / 1000.0
        return mej

    @property
    def calculate_qej(self):
        """
        Polar opening angle

        :return: polar opening angle
        """
        tmp1 = 3. * self.vz + np.sqrt(9 * self.vz**2 + 4 * self.vrho**2)
        qej = ((2.0 ** (4.0 / 3.0)) * self.vrho**2 + (2. * self.vrho**2 * tmp1) ** (2.0 / 3.0)) / ((self.vrho ** 5.0) * tmp1) ** (
                    1.0 / 3.0)
        return qej

    @property
    def calculate_phej(self):
        """
        azimuthal opening angle

        :return: azimuthal opening angle
        """
        return 4.0 * self.qej * np.pi / 2.0

class TwoComponentBNS(object):
    def __init__(self, mass_1, mass_2, lambda_1, lambda_2, mtov, zeta):
        """
        Relations to connect intrinsic GW parameters to extrinsic kilonova parameters from Coughlin+2019
        for a two component BNS kilonova model.

        :param mass_1: mass of primary neutron star
        :param mass_2: mass of secondary neutron star
        :param lambda_1: tidal deformability of primary neutron star
        :param lambda_2: tidal deformability of secondary neutron star
        :param mtov: Tolman Oppenheimer Volkoff maximum neutron star mass
        :param zeta: fraction of disk that gets unbound
        """
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.mtov = mtov
        self.zeta = zeta
        self.c1 = calc_compactness_from_lambda(self.lambda_1)
        self.c2 = calc_compactness_from_lambda(self.lambda_2)
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2019MNRAS.489L..91C/abstract'
        self.vrho = calc_vrho(mass_1=self.mass_1, mass_2=self.mass_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        self.vz = calc_vz(mass_1=self.mass_1, mass_2=self.mass_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        self.dynamical_mej = self.calculate_dynamical_ejecta_mass
        self.disk_wind_mej = self.calculate_disk_wind_mass
        self.ejecta_velocity = self.calculate_ejecta_velocity
        self.qej = self.calculate_qej
        self.phej = self.calculate_phej

    @property
    def calculate_ejecta_velocity(self):
        """
        Calculate ejecta velocity of the dynamical ejecta

        :return: ejecta velocity in c
        """
        c1 = self.c1
        c2 = self.c2

        a = -0.3090
        b = 0.657
        c = -1.879

        vej = a * (self.mass_1 / self.mass_2) * (1 + c * c1) + a * (self.mass_2 / self.mass_1) * (1 + c * c2) + b
        return vej

    @property
    def calculate_dynamical_ejecta_mass(self):
        """
        Calculate the disk wind and dynamical ejecta masses

        :return: ejecta mass in solar masses
        """
        c1 = self.c1
        c2 = self.c2
        m1 = self.mass_1
        m2 = self.mass_2

        a = -0.0719
        b = 0.2116
        d = -2.42
        n = -2.905

        log10_mej = a * (m1 * (1 - 2 * c1) / c1 + m2 * (1 - 2 * c2) / c2) + b * (
                    m1 * (m2 / m1) ** n + m2 * (m1 / m2) ** n) + d

        mej_dynamical = 10 ** log10_mej
        return mej_dynamical

    @property
    def calculate_disk_wind_mass(self):
        """
        Calculate the disk wind and dynamical ejecta masses

        :return: ejecta mass in solar masses
        """
        q = self.mass_1 / self.mass_2

        lambdatilde = (16.0 / 13.0) * \
                      (self.lambda_2 + self.lambda_1 * (q ** 5) + 12 * self.lambda_1 * (q ** 4) + 12 * self.lambda_2 * q) / (
                    (q + 1) ** 5)
        mc = ((self.mass_1 * self.mass_2) ** (3. / 5.)) * ((self.mass_1 + self.mass_2) ** (-1. / 5.))

        mTOV = self.mtov
        R16 = mc * (lambdatilde / 0.0042) ** (1.0 / 6.0)
        mth = (2.38 - 3.606 * mTOV / R16) * mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        mtot = self.mass_1 + self.mass_2
        mdisk = a * (1 + b * np.tanh((c - mtot / mth) / d))

        mdisk = 10 ** mdisk

        mej_disk_wind = self.zeta * mdisk
        return mej_disk_wind

    @property
    def calculate_qej(self):
        """
        Polar opening angle

        :return: polar opening angle
        """
        tmp1 = 3. * self.vz + np.sqrt(9 * self.vz**2 + 4 * self.vrho**2)
        qej = ((2.0 ** (4.0 / 3.0)) * self.vrho**2 + (2. * self.vrho**2 * tmp1) ** (2.0 / 3.0)) / ((self.vrho ** 5.0) * tmp1) ** (
                    1.0 / 3.0)
        return qej

    @property
    def calculate_phej(self):
        """
        azimuthal opening angle

        :return: azimuthal opening angle
        """
        return 4.0 * self.qej * np.pi / 2.0


class TwoComponentNSBH(object):
    def __init__(self, mass_bh, mass_ns, chi_eff, lambda_ns, zeta):
        """
        Relations to connect intrinsic GW parameters to extrinsic kilonova parameters
        for a neutron star black hole merger with two components using relations from Kawaguchi et al. 2016.
        and Foucart et al. 2018

        :param mass_bh: mass of black hole
        :param mass_2: mass of neutron star
        :param chi_eff: effective spin of black hole
        :param lambda_ns: tidal deformability of neutron star
        :param zeta: fraction of disk that gets unbound
        """
        self.mass_bh = mass_bh
        self.mass_ns = mass_ns
        self.mass_ratio = mass_bh/mass_ns
        self.chi_eff = chi_eff
        self.lambda_ns = lambda_ns
        self.zeta = zeta
        self.reference = ['https://ui.adsabs.harvard.edu/abs/2016ApJ...825...52K/abstract',
                          'https://ui.adsabs.harvard.edu/abs/2018PhRvD..98h1501F/abstract']
        self.risco = self.rcap_isco
        self.ejecta_velocity = self.calculate_ejecta_velocity
        self.dynamical_mej = self.calculate_dynamical_ejecta_mass
        self.disk_wind_mej = self.calculate_disk_wind_mass

    @property
    def rcap_isco(self):
        """
        Calculate the normalized ISCO radius for a given BH spin.

        :return: Normalized radius of the Innermost Stable Circular Orbit
        """
        chi_bh = self.chi_eff
        z1 = 1 + (1 - chi_bh ** 2) ** (1 / 3) * (
                (1 + chi_bh) ** (1 / 3) + (1 - chi_bh) ** (1 / 3))
        z2 = np.sqrt(3 * chi_bh ** 2 + z1 ** 2)
        risco = 3 + z2 - np.sign(chi_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
        return risco

    @property
    def calculate_ejecta_velocity(self):
        """
        Calculate ejecta velocity

        :return: ejecta velocity in c
        """
        vej = 1.5333330951369120e-2*self.mass_ratio+0.19066667068621043
        return vej

    @property
    def calculate_dynamical_ejecta_mass(self):
        """
        Calculate ejecta mass

        :return: ejecta mass in solar masses
        """
        compactness = calc_compactness_from_lambda(self.lambda_ns)
        baryonic_mass = calc_baryonic_mass(mass=self.mass_ns, compactness=compactness)

        # Compute mass_dyn. Formula from Kawaguchi et al., 2016
        a1, a2, a3, a4, n1, n2 = 4.464e-2, 2.269e-3, 2.431, -0.4159, 0.2497, 1.352
        term_a1 = a1 * (self.mass_ratio ** n1) * (1 - 2 * compactness) / compactness
        term_a2 = -a2 * (self.mass_ratio ** n2) * self.risco
        term_a3 = a3 * (1 - self.mass_ns / baryonic_mass)
        term_a4 = a4
        mass_dyn = baryonic_mass * np.maximum(term_a1 + term_a2 + term_a3 + term_a4, 0)
        return mass_dyn

    @property
    def calculate_disk_wind_mass(self):
        """
        Calculate ejecta mass

        :return: ejecta mass in solar masses
        """
        compactness = calc_compactness_from_lambda(self.lambda_ns)
        baryonic_mass = calc_baryonic_mass(mass=self.mass_ns, compactness=compactness)
        rho = (15 * self.lambda_ns) ** (-1 / 5)
        eta = self.mass_ratio / (1 + self.mass_ratio) ** 2

        # Compute mass_out. Formula from ^Foucart et al., 2018
        alpha, beta, gamma, delta = 0.308, 0.124, 0.283, 1.536
        term_alpha = alpha * (1 - 2 * rho) / (eta ** (1 / 3))
        term_beta = -beta * self.risco * rho / eta
        term_gamma = gamma
        mej_disk = baryonic_mass * (np.maximum(term_alpha + term_beta + term_gamma, 0.0)) ** delta
        mej_disk_wind = self.zeta * mej_disk
        return mej_disk_wind

class OneComponentNSBH(object):
    def __init__(self, mass_bh, mass_ns, chi_eff, lambda_ns):
        """
        Relations to connect intrinsic GW parameters to extrinsic kilonova parameters
        for a neutron star black hole merger with one component (zone) from Kawaguchi et al. 2016.

        :param mass_bh: mass of black hole
        :param mass_2: mass of neutron star
        :param chi_eff: effective spin of black hole
        :param lambda_ns: tidal deformability of neutron star
        """
        self.mass_bh = mass_bh
        self.mass_ns = mass_ns
        self.mass_ratio = mass_bh/mass_ns
        self.chi_eff = chi_eff
        self.lambda_ns = lambda_ns
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2016ApJ...825...52K/abstract'
        self.risco = self.isco_radius
        self.ejecta_velocity = self.calculate_ejecta_velocity
        self.ejecta_mass = self.calculate_ejecta_mass

    @property
    def isco_radius(self):
        """
        Calculate innermost stable orbit radius

        :return: isco radius
        """
        chi = self.chi_eff
        z1 = 1 + ((1 - chi * chi) ** (1 / 3.0)) * (((1 + chi) ** (1 / 3.0)) + (1 - chi) ** (1 / 3.0))
        z2 = (3 * chi * chi + z1 * z1) ** (1 / 2.0)
        risco = 3 + z2 - np.sign(chi) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2.0)
        return risco

    @property
    def calculate_ejecta_velocity(self):
        """
        Calculate ejecta velocity

        :return: ejecta velocity in c
        """
        vej = 1.5333330951369120e-2*self.mass_ratio+0.19066667068621043
        return vej

    @property
    def calculate_ejecta_mass(self):
        """
        Calculate ejecta mass

        :return: ejecta mass in solar masses
        """
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