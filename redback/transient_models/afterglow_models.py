from astropy.cosmology import Planck18 as cosmo  # noqa
from inspect import isfunction
from redback.utils import logger, citation_wrapper, calc_ABmag_from_flux_density, lambda_to_nu, bands_to_frequency
from redback.constants import day_to_s, speed_of_light, solar_mass, proton_mass, electron_mass, sigma_T
from redback.sed import get_correct_output_format_from_spectra
import astropy.units as uu
import numpy as np
from collections import namedtuple
from scipy.special import erf
from scipy.interpolate import interp1d
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

jet_spreading_models = ['tophat', 'cocoon', 'gaussian',
                          'kn_afterglow', 'cone_afterglow',
                          'gaussiancore', 'gaussian',
                          'smoothpowerlaw', 'powerlawcore',
                          'tophat']

class RedbackAfterglows():
    def __init__(self, k, n, epsb, epse, g0, ek, thc, thj, tho, p, exp, time, freq, redshift, Dl,
                 extra_structure_parameter_1,extra_structure_parameter_2, method='TH', res=100, steps=int(500), xiN=1):
        """
        A general class for afterglow models implemented directly in redback.
        This class is not meant to be used directly but instead via the interface for each specific model.
        The afterglows are based on the method shown in Lamb, Mandel & Resmi 2018 and other papers.
        Script was originally written by En-Tzu Lin <entzulin@gapp.nthu.edu.tw> and Gavin Lamb <g.p.lamb@ljmu.ac.uk>
        and modified and implemented into redback by Nikhil Sarin <nsarin.astro@gmail.com>.
        Includes wind-like mediums, expansion and multiple jet structures.

        :param k:
        :param n: ISM, ambient number density
        :param epsb: magnetic fraction
        :param epse: electron fraction
        :param g0: initial Lorentz factor
        :param ek: kinetic energy
        :param thc: core angle
        :param thj: jet outer angle. For tophat jets thc=thj
        :param tho: observers viewing angle
        :param p: electron power-law index
        :param exp: Boolean for whether to include sound speed expansion
        :param time: lightcurve time steps
        :param freq: lightcurve frequencies
        :param redshift: source redshift
        :param Dl: luminosity distance
        :param extra_structure_parameter_1: Extra structure specific parameter #1.
            Specifically, this parameter sets;
            The index on energy for power-law jets.
            The fractional energy contribution for the Double Gaussian (must be less than 1).
            The energy fraction  for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        :param extra_structure_parameter_2: Extra structure specific parameter #2.
            Specifically, this parameter sets;
            The index on lorentz factor for power-law jets.
            The lorentz factor for second Gaussian (must be less than 1).
            The lorentz factor  for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        :param method: Type of jet structure to use. Defaults to 'TH' for tophat jet.
            Other options are '2C', 'GJ', 'PL', 'PL2', 'DG'. Corresponding to two component, gaussian jet, powerlaw,
            alternative powerlaw and double Gaussian.
        :param res: resolution
        :param steps: number of steps used to resolve Gamma and dm
        :param XiN: fraction of electrons that get accelerated
        """
        self.k = k
        if self.k == 0:
            self.n = n
        elif self.k == 2:
            self.n = n * 3e35  # n \equiv A*
        self.epsB = epsb
        self.epse = epse
        self.g0 = g0
        self.ek = ek
        self.thc = thc
        self.thj = thj
        self.tho = tho
        self.p = p
        self.exp = exp
        self.t = time
        self.freq = freq
        self.z = redshift
        self.Dl = Dl
        self.method = method
        self.s = extra_structure_parameter_1
        self.a = extra_structure_parameter_2
        self.res = res
        self.steps = steps
        self.xiN = xiN

        ### Set up physical constants
        self.mp = 1.6726231e-24  # g, mass of proton
        self.me = 9.1093897e-28  # g, mass of electron
        self.cc = 2.99792453e10  # cm s^-1, speed of light
        self.qe = 4.8032068e-10  # esu, electron charge
        self.c2 = self.cc * self.cc
        self.sigT = (self.qe * self.qe / (self.me * self.c2)) ** 2 * (8 * np.pi / 3)  # Thomson cross-section
        self.fourpi = 4 * np.pi
        self.is_expansion = self.exp

    def calc_erf_numba(self, x):
        return np.array([erf(i) for i in x])

    def get_lightcurve(self):
        if (self.k != 0) and (self.k != 2):
            raise ValueError("k must either be 0 or 2")
        if (self.p < 1.2) or (self.p > 3.4):
            raise ValueError("p is out of range, 1.2 < p < 3.4")
        # parameters p, x_p, and phi_p from Wijers & Galama 1999
        pxf = np.array([[1.0, 3.0, 0.41], [1.2, 1.4, 0.44], [1.4, 1.1, 0.48], [1.6, 0.86, 0.53], [1.8, 0.725, 0.56],
                        [2.0, 0.637, 0.59], [2.2, 0.579, 0.612], [2.5, 0.520, 0.630], [2.7, 0.487, 0.641],
                        [3.0, 0.451, 0.659],
                        [3.2, 0.434, 0.660], [3.4, 0.420, 0.675]])
        xp = np.interp(self.p, pxf[:, 0], pxf[:, 1])  # dimensionless spectral peak
        Fx = np.interp(self.p, pxf[:, 0], pxf[:, 2])  # dimensionless peak flux
        nu0 = np.unique(self.freq)  # unique frequencies in the sample, if loading a data array for frequencies
        nu = nu0 * (1 + self.z)  # rest frame frequency
        nu = np.array(nu)
        Omi, thi, phii, rotstep, latstep = self.get_segments(thj=self.thj, res=self.res)
        Gs, Ei = self.get_structure(gamma=self.g0, en=self.ek,
                                    thi=thi, thc=self.thc, method=self.method, s=self.s, a=self.a, thj=self.thj)
        G, SM, ghat = self.get_gamma(G0=Gs, Eps=Ei, therm=0., steps=self.steps, n0=self.n, k=self.k)
        Obsa = self.get_obsangle(phii=phii, thi=thi, tho=self.tho)
        # calculate the afterglow flux
        Flux, tobs = self.calc_afterglow(G=G, SM=SM, Dl=self.Dl, p=self.p, xp=xp, Fx=Fx,
                                         EB=self.epsB, Ee=self.epse, Gs=Gs, Omi=Omi, Ei=Ei,
                                         n=self.n, k=self.k, tho=self.tho, thi=thi, phii=phii,
                                         thj=self.thj, ghat=ghat, rotstep=rotstep,
                                         latstep=latstep, Obsa=Obsa, nu=nu,
                                         steps=self.steps, XiN=self.xiN)
        # sums all the flux at the same observer times
        LC = self.calc_lightcurve(time=self.t, tobs=tobs, Flux=Flux,
                                  nu_size=nu.size, thi_size=thi.size, phii_size=phii.size,
                                  freq=self.freq, nu0=nu0)
        return LC

    def get_segments(self, thj, res):
        ### parameter setting
        latstep = thj/res #lateral step from centre to edge - fixed angular width
        rotstep = 2.*np.pi/res #rotational step 
        Nlatstep = int(res) #integer number of lateral steps
        Nrotstep = int(2*np.pi/rotstep) #interger number of rotational steps
        Omi = np.empty((Nlatstep*Nrotstep)) #defines an array for the solid angle of each segment
        phi = np.linspace(rotstep,Nrotstep*rotstep,Nrotstep) #rotation angles from 0-2pi
        for i in range(0,Nlatstep):
            Omi[i*Nrotstep:(i+1)*Nrotstep] = (phi-(phi-rotstep))*(np.cos((i*latstep))-np.cos(((i+1)*latstep))) #defines the solid angle of each segment
        thi = np.linspace(latstep-latstep/2.,Nlatstep*latstep-latstep/2.,Nlatstep) #lateral angle to centre of segment
        phii = np.linspace(rotstep-rotstep/2.,Nrotstep*rotstep-rotstep/2.,Nrotstep)#rotational angle to centre of each segment
        return Omi, thi, phii, rotstep, latstep

    def get_structure(self, gamma, en, thi, thc, method, s, a, thj):
        if gamma == 1.0:
            raise ValueError("Gamma must not equal 1!!!")
        Gs = np.full(shape=thi.size, fill_value=gamma)
        Ei = np.full(shape=thi.size, fill_value=en)
        if method == "TH":  # Top-hat
            thj_ = np.where(thc < thj, thc, thj)
            fac = (self.calc_erf_numba(-(thi - thj_) * 1000.) * 0.5) + 0.5
            Gs = (Gs - 1) * fac + 1.0000000000001
            Ei *= fac
        elif method == "Gaussian" or method == "GJ":  # Gaussian
            fac = np.exp(-0.5 * (thi / thc) ** 2)
            Gs = (Gs - 1) * fac + 1.0000000000001
            Ei *= fac
        elif method == "PL":  # Power-law
            for i, thi_t in enumerate(thi):
                if thi_t >= thc:
                    fac_s = (thc / thi_t) ** (s)
                    fac_a = (thc / thi_t) ** (a)
                    Ei[i] *= fac_s
                    Gs[i] = (Gs[i] - 1.) * fac_a + 1.000000000000001
        elif method == "PL-alt" or method == "PL2":  # alternative powerlaw
            fac = (1 + (thi / thc) ** 2) ** 0.5
            Gs = 1.000000000001 + (Gs - 1) * fac ** (-a)
            Ei *= fac ** (-s)
        elif method == "Two-Component" or method == "2C":  # Two-Component
            if a < 1:
                raise ValueError("a must be > 1 for two-component model")
            for i, thi_t in enumerate(thi):
                if thi_t > thc:
                    Gs[i] = a
                    Ei[i] *= s
        elif method == "Double-Gaussian" or method == "DG":  # Double Gaussian
            if (a > 1) or (s > 1):
                raise ValueError("a and s must be < 1 for double Gaussian model")
            fac = self.double_gaussian_beam(thi, thc, thj, s, a)
            Ei *= fac
            fac_gs = self.double_gaussian_lf(thi, thc, thj, s, a)
            Gs = (Gs - 1) * fac_gs + 1.0000000000001
        else:  # default
            pass
        return Gs, Ei

    def gaussian_beam(self, thi, thc):
        return np.exp(-0.5 * (thi / thc) ** 2)

    def double_gaussian_beam(self, thi, thc, thj, s, a):
        return (1. - s) * self.gaussian_beam(thi, thc) + s * self.gaussian_beam(thi, thj)

    def double_gaussian_lf(self, thi, thc, thj, s, a):
        out = self.double_gaussian_beam(thi, thc, thj, s, a) / self.double_gaussian_beam(thi, thc, thj, s / a, a)
        out[np.isnan(out)] = a
        return out

    def get_gamma(self, G0, Eps, therm, steps, n0, k):
        ### parameter setting
        Rmin = 1e10  # cm
        Rmax = 1e24  # cm
        Nmin = (self.fourpi / (3. - k)) * n0 * Rmin ** (3. - k)  # min number
        Nmax = (self.fourpi / (3. - k)) * n0 * Rmax ** (3. - k)  # max number
        dlogm = np.log10(Nmin * self.mp)
        h = np.log10(Nmax / Nmin) / steps  # step size in dlog(m)
        # constant factor
        fac = -h * np.log(10)
        M = Eps / (G0 * self.c2)  # explosion rest mass

        ### 4th-order Runge-Kutta integration
        def RK4(ghat, dm_rk4, G_rk4):
            ghatm1 = ghat - 1.
            dm_base10 = 10 ** dm_rk4
            G_rk4_sq = G_rk4 * G_rk4
            _G_rk4 = 1. / G_rk4
            return fac * dm_base10 * (ghat * (G_rk4_sq - 1.) - ghatm1 * (G_rk4 - _G_rk4)) \
                / (M + dm_base10 * (therm + (1. - therm) * (2. * ghat * G_rk4 - ghatm1 * (1 + 1. / G_rk4_sq))))

        # arrays for the state at each step
        state_G = np.empty(shape=(steps, G0.size))
        state_dm = np.empty(shape=(steps, G0.size))
        state_gH = np.empty(shape=(steps, G0.size))
        # initial values
        G = G0
        dm = np.full(shape=G.size, fill_value=dlogm)
        # main loop
        for i in range(steps):
            # store the G and dm
            state_G[i] = G
            state_dm[i] = dm
            G2m1 = G * G - 1.0
            G2m1_root = G2m1 ** 0.5
            # temperature of shocked matter
            theta = G2m1_root * (G2m1_root + 1.07 * G2m1) / (3. * (1 + G2m1_root + 1.07 * G2m1))
            z = theta / (0.24 + theta)
            # adiabatic index
            ghat = ((((((
                                1.07136 * z - 2.39332) * z + 2.32513) * z - 0.96583) * z + 0.18203) * z - 1.21937) * z + 5.) / 3.
            # 4th order Runge Kutta to solve ODE from Pe'er 2012
            F1 = RK4(ghat, dm, G)
            F2 = RK4(ghat, dm + 0.5 * h, G + 0.5 * F1)
            F3 = RK4(ghat, dm + 0.5 * h, G + 0.5 * F2)
            F4 = RK4(ghat, dm + h, G + F3)
            # update state
            G = G + (F1 + 2. * (F2 + F3) + F4) / 6. + 1e-15
            dm = dm + h
            # store the ghat
            state_gH[i] = ghat
        return state_G.T, 10. ** state_dm.T, state_gH.T

    def get_obsangle(self, phii, thi, tho):
        phi = 0.0  # we assume rotational symmetry - phi is therefore arbitrary
        sin_thi = np.sin(thi)
        cos_thi = np.cos(thi)
        f1 = np.sin(phi) * np.sin(tho) * np.sin(phii) * sin_thi.reshape(-1, 1)
        f2 = np.cos(phi) * np.sin(tho) * np.cos(phii) * sin_thi.reshape(-1, 1)
        Obsa = np.cos(tho) * cos_thi.reshape(-1, 1) + f2 + f1
        Obsa = np.arccos(Obsa)
        return Obsa.ravel()

    def calc_afterglow_step1(self, G, dm, p, xp, Fx, EB, Ee, n, k, thi, ghat, rotstep, latstep, xiN):
        rotstep = np.full(1, rotstep)
        latstep = np.full(1, latstep)
        Gm1 = G - 1.0
        G2 = G * G
        beta = (1 - 1. / G2) ** 0.5  # normalised velocity at each radial step
        Ne = dm / self.mp  # number of swept up electrons
        ### side-ways expansion
        cs = (self.c2 * ghat * (ghat - 1) * Gm1 / (1 + ghat * Gm1)) ** 0.5  # sound speed
        te = np.arcsin(cs / (self.cc * (G2 - 1.) ** 0.5))  # equivalent form for angle due to spreading
        # prepare ex and OmG in this function
        if self.is_expansion:
            ex = te / (G2) # expansion
            fac = 0.5 * latstep
            OmG = rotstep * (np.cos(thi - fac) - np.cos(ex/self.res + thi + fac))  # equivalent form for linear spacing
        else:
            ex = np.ones(te.size)  # no expansion
            fac = 0.5 * latstep
            OmG = rotstep * (np.cos(thi - fac) - np.cos(thi + fac))  # equivalent form for linear spacing
        # prepare R
        size = G.size
        exponent = ((1 - np.cos(latstep + ex[0])) / (1 - np.cos(latstep + ex[:size]))) ** (1/2)
        R = ((3. - k) * Ne[:size] / (self.fourpi * n)) ** (1. / (3. - k))
        R[1:] = np.diff(R) * exponent[1:size] ** (1. / (3. - k))
        R = np.cumsum(R)

        n0 = n * R ** (-k)
        ### forward shock
        # parameters for synchrotron emission
        B = (2 * self.fourpi * EB * n0 * self.mp * self.c2 * (
                    (ghat * G + 1.) / (ghat - 1.)) * Gm1) ** 0.5  # magnetic field strength
        gmm = (1.5 * self.fourpi * self.qe / (self.sigT * B)) ** (0.5)
        if p > 2:
            gp = (p - 2) / (p - 1)
        elif p == 2:
            gp = 1 / np.log(gmm / ((Ee / xiN) * Gm1 * (self.mp / self.me)))
        if p >= 2:
            gm = gp * (Ee / xiN) * Gm1 * (self.mp / self.me)
        else:
            gm = ((2 - p) / (p - 1) * (self.mp / self.me) * (Ee / xiN) * Gm1 * gmm ** (p - 2)) ** (1 / (p - 1))

        nump = 3. * xp * gm * gm * self.qe * B / (
                    self.fourpi * self.me * self.cc)  # characteristic synchrotron frequency co-moving
        Pp = xiN * Fx * self.me * self.c2 * self.sigT * B / (
                    3. * self.qe)  # 3.**0.5*q**3.*B/(me*c**2.)#  # synchrotron power per electron co-moving
        KT = gm * self.me * self.c2
        return beta, Ne, OmG, R, B, gm, nump, Pp, KT

    def calc_afterglow_step2(self, Dl, Om0, rotstep, latstep, Obsa, beta, Ne, OmG, R, B, gm, nump, Pp, KT, G):
        Dl2 = Dl * Dl
        NO  = Om0 * Ne / self.fourpi   # initial electrons per segment
        cos_Obsa = np.cos(Obsa)
        if self.is_expansion:
            Om   = np.maximum(Om0, OmG)  # solid angle at each step given expansion condition
            thii = np.arccos(1. - Om / rotstep)
        else:
            Om   = OmG  # array for solid angle
            thii = np.arccos(1. - Om / rotstep)
        size = G.size
        R_diff = np.diff(R,prepend=0)
        dt = R_diff*(1./beta[:size]-np.cos(Obsa))/self.cc
        dto = R_diff*(1./beta[:size]-1.)/self.cc
        tobs = np.cumsum(dt)
        tobso = np.cumsum(dto)
        ""
        ### forward shock
        dop  = 1. / (G * (1. - beta * cos_Obsa))  # Doppler factor
        # parameters for synchrotron emission
        gc = 6. * np.pi * self.me * self.cc / (G * self.sigT * B * B * tobso)  # gamma_c
        nucp = 0.286 * 3. * gc * gc * self.qe * B / (self.fourpi * self.me * self.cc)  # cooling frequency co-moving
        num = dop * nump  # observer frame synchrotron frequency
        nuc = dop * nucp  # observer frame cooling frequency
        # maximum synchrotron flux including emission area correction for 1/G > jet opening angle
        Fmax = NO * Pp * dop * dop * dop / (self.fourpi * Dl2)
        ### self-absorption
        FBB = 2 * Om * np.cos(thii) * dop * KT * R * R / (self.c2 * Dl2)
        return FBB, Fmax, nuc, num, tobs

    def get_ag(self, FBB, nuc, num, nu1, Fmax, p):
        Fluxt = np.zeros((num.size)) #array for flux at a given frequency with time
        #Observed flux at each step
        #Fast
        F1 = (nuc < num) & (nu1 < nuc)
        Fluxt[F1] = Fmax[F1] * (nu1/nuc[F1])**(1./3.)
        F2 = (nuc < nu1) & (nuc < num)
        Fluxt[F2] = Fmax[F2] * (nu1/nuc[F2])**(-1./2.)
        F3 = (num < nu1) & (nuc < num)
        Fluxt[F3] = Fmax[F3] * (num[F3]/nuc[F3])**(-1./2.)*(nu1/num[F3])**(-p/2.)
        #Slow
        S1 = (num < nuc) & (nu1 < num)
        Fluxt[S1] = Fmax[S1] * (nu1/num[S1])**(1./3.)
        S2 = (num < nu1) & (num < nuc)
        Fluxt[S2] = Fmax[S2] * (nu1/num[S2])**(-(p-1.)/2.)
        S3 = (nuc < nu1) & (num < nuc)
        Fluxt[S3] = Fmax[S3] * (nuc[S3]/num[S3])**(-(p-1.)/2.)*(nu1/nuc[S3])**(-p/2.)
        #SSA
        FBB = FBB*nu1**2.*np.maximum(1,(nu1/num)**0.5)
        Fluxt = np.minimum(FBB,Fluxt)
        return Fluxt

    def calc_afterglow(self, G, SM, Dl, p, xp, Fx, EB, Ee, Gs, Omi, Ei, n, k, tho, thi, phii, thj, ghat, rotstep,
                       latstep, Obsa, nu, steps, XiN):
        Flux = np.empty(shape=(nu.size, steps, thi.size * phii.size))
        tobs = np.empty(shape=(steps, thi.size * phii.size))
        kk = 0
        for i in range(thi.size):
            beta, Ne, OmG, R, B, gm, nump, Pp, KT = self.calc_afterglow_step1(G[i, :], SM[i, :], p, xp, Fx, EB, Ee, n, k,
                                                                         thi[i], ghat[i, :], rotstep, latstep, XiN)
            for j in range(phii.size):
                FBB, Fmax, nuc, num, tobs[:, kk] = self.calc_afterglow_step2(Dl, Omi[kk], rotstep, latstep, Obsa[kk], beta,
                                                                        Ne, OmG, R, B, gm, nump, Pp, KT, G[i, :])
                if nu.size > 1:
                    for h in range(0, nu.size):
                        Flux[h, :, kk] = self.get_ag(FBB, nuc, num, nu[h], Fmax, p)
                elif nu.size == 1:
                    Flux[0, :, kk] = self.get_ag(FBB, nuc, num, nu, Fmax, p)
                kk += 1
        return Flux, tobs

    def calc_lightcurve(self, time, tobs, Flux, nu_size, thi_size, phii_size, freq, nu0):
        LC = np.zeros((freq.size))
        # forward shock lightcurve at each observation time
        for h in range(nu_size):
            FF = np.zeros((len(time[(freq == nu0[h])])))
            for i in range(thi_size * phii_size):
                FF += np.interp(time[(freq == nu0[h])] / (1 + self.z), tobs[:, i], Flux[h, :, i])
            LC[(freq == nu0[h])] = FF
        return LC * (1 + self.z)

class RedbackAfterglowsRefreshed(RedbackAfterglows):
    def __init__(self, k, n, epsb, epse, g0, g1, ek, et, s1, thc, thj, tho, p, exp, time, freq, redshift, Dl,
                 extra_structure_parameter_1, extra_structure_parameter_2,
                 method='TH', res=100, steps=int(500), xiN=1):

        """
        A general class for refreshed afterglow models implemented directly in redback.
        This class is not meant to be used directly but instead via the interface for each specific model.
        The afterglows are based on the method shown in Lamb, Mandel & Resmi 2018 and other papers.
        Script was originally written by En-Tzu Lin <entzulin@gapp.nthu.edu.tw> and Gavin Lamb <g.p.lamb@ljmu.ac.uk>
        and modified and implemented into redback by Nikhil Sarin <nsarin.astro@gmail.com>.
        Includes wind-like mediums, expansion and multiple jet structures.

        :param k:
        :param n: ISM, ambient number density
        :param epsb: magnetic fraction
        :param epse: electron fraction
        :param g0: initial Lorentz factor
        :param g1: Lorentz factor of shell at start of energy injection
        :param ek: kinetic energy
        :param et: factor by which total kinetic energy is larger
        :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
            Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
        :param thc: core angle
        :param thj: jet outer angle. For tophat jets thc=thj
        :param tho: observers viewing angle
        :param p: electron power-law index
        :param exp: Boolean for whether to include sound speed expansion
        :param time: lightcurve time steps
        :param freq: lightcurve frequencies
        :param redshift: source redshift
        :param Dl: luminosity distance
        :param extra_structure_parameter_1: Extra structure specific parameter #1.
            Specifically, this parameter sets;
            The index on energy for power-law jets.
            The fractional energy contribution for the Double Gaussian (must be less than 1).
            The energy fraction  for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        :param extra_structure_parameter_2: Extra structure specific parameter #2.
            Specifically, this parameter sets;
            The index on lorentz factor for power-law jets.
            The lorentz factor for second Gaussian (must be less than 1).
            The lorentz factor  for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        :param method: Type of jet structure to use. Defaults to 'TH' for tophat jet.
            Other options are '2C', 'GJ', 'PL', 'PL2', 'DG'. Corresponding to two component, gaussian jet, powerlaw,
            alternative powerlaw and double Gaussian.
        :param res: resolution
        :param steps: number of steps used to resolve Gamma and dm
        :param XiN: fraction of electrons that get accelerated
        """

        super().__init__(k=k, n=n, epsb=epsb, epse=epse, g0=g0, ek=ek, thc=thc, thj=thj,
                         tho=tho, p=p, exp=exp, time=time, freq=freq, redshift=redshift,
                         Dl=Dl, extra_structure_parameter_1=extra_structure_parameter_1,
                         extra_structure_parameter_2=extra_structure_parameter_2, method=method,
                         res=res, steps=steps, xiN=xiN)
        self.G1 = g1
        self.Et = et
        self.s1 = s1

    def get_gamma_refreshed(self, G0, G1, Eps, Eps2, s1, therm, steps, n0, k):
        Eps0 = Eps
        # solves blastwave dynamics and gives the Lorentz factor and swept-up mass at each step
        # Gamma0, blast energy per steradian, fraction thermal radiated
        # therm = 0 for adiabatic solution
        n = n0
        Rmin = 1.e10  # cm
        Rmax = 1.e24  # cm
        Nmin = (self.fourpi / 3.) * n * Rmin ** (3 - k)  # min number
        Nmax = (self.fourpi / 3.) * n * Rmax ** (3 - k)  # min number
        # assumption is no sideways expansion - the radius is not determined by the O(4)RK only mass and Gamma
        dlogm = np.log10(Nmin * self.mp)
        h = (np.log10(Nmax) - np.log10(Nmin)) / steps  # step size in dlog(m)
        G = np.ones(steps + 1)  # set up arrays
        dm = np.zeros(steps)
        G[0] = G0  # initial Gamma
        dm[0] = dlogm  #
        gH = np.zeros(steps)
        M = Eps / (G0 * self.cc ** 2.)  # explosion rest mass
        for i in range(0, steps, 1):
            theta = ((G[i] ** 2. - 1) ** 0.5) / 3. * (((G[i] ** 2. - 1) ** 0.5 + 1.07 * (G[i] ** 2. - 1)) / (
                        1 + (G[i] ** 2. - 1) ** 0.5 + 1.07 * (G[i] ** 2. - 1)))  # temperature of shocked matter
            z = theta / (0.24 + theta)
            ghat = (
                               5 - 1.21937 * z + 0.18203 * z ** 2. - 0.96583 * z ** 3. + 2.32513 * z ** 4. - 2.39332 * z ** 5. + 1.07136 * z ** 6.) / 3.  # adiabatic index
            dm[i] = (dlogm + i * h)
            # 4th order Runge Kutta to solve ODE from Pe'er 2012
            F1 = -1. * h * np.log(10) * (10. ** dm[i]) * (
                        ghat * (G[i] ** 2. - 1) - (ghat - 1) * G[i] * (1 - G[i] ** -2.)) / (
                             M + therm * (10. ** dm[i]) + (1. - therm) * 10. ** (dm[i]) * (
                                 2. * ghat * G[i] - (ghat - 1) * (1 + G[i] ** -2.)))  #

            F2 = -1. * h * np.log(10) * (10. ** (dm[i] + h / 2.)) * (
                        ghat * ((G[i] + F1 / 2.) ** 2. - 1) - (ghat - 1) * (G[i] + F1 / 2.) * (
                            1 - (G[i] + F1 / 2.) ** -2.)) / (
                             M + therm * (10. ** (dm[i] + h / 2.)) + (1. - therm) * (10. ** (dm[i] + h / 2.)) * (
                                 2. * ghat * (G[i] + F1 / 2.) - (ghat - 1) * (1 + (G[i] + F1 / 2.) ** -2.)))  #

            F3 = -1. * h * np.log(10) * (10. ** (dm[i] + h / 2.)) * (
                        ghat * ((G[i] + F2 / 2.) ** 2. - 1) - (ghat - 1) * (G[i] + F2 / 2.) * (
                            1 - (G[i] + F2 / 2.) ** -2.)) / (
                             M + therm * (10. ** (dm[i] + h / 2.)) + (1. - therm) * (10. ** (dm[i] + h / 2.)) * (
                                 2. * ghat * (G[i] + F2 / 2.) - (ghat - 1) * (1 + (G[i] + F2 / 2.) ** -2.)))  #

            F4 = -1. * h * np.log(10) * (10. ** (dm[i] + h)) * (
                        ghat * ((G[i] + F3) ** 2. - 1) - (ghat - 1) * (G[i] + F3) * (1 - (G[i] + F3) ** -2.)) / (
                             M + therm * (10. ** (dm[i] + h)) + (1. - therm) * (10. ** (dm[i] + h)) * (
                                 2. * ghat * (G[i] + F3) - (ghat - 1) * (1 + (G[i] + F3) ** -2.)))  #
            gH[i] = ghat
            G[i + 1] = G[i] + (1. / 6.) * (F1 + 2. * F2 + 2. * F3 + F4)
            if G[i + 1] <= G1:
                Eps1 = Eps
                Eps = min(Eps0 * ((G[i + 1] ** 2 - 1) ** 0.5 / (G1 ** 2 - 1) ** 0.5) ** -s1, Eps2)  #
                M += (Eps - Eps1) / (G[i] * self.cc ** 2.)  # new explosion rest mass

        return G, 10. ** dm, gH

    def get_lightcurve(self):
        if (self.k != 0) and (self.k != 2):
            raise ValueError("k must either be 0 or 2")
        if (self.p < 1.2) or (self.p > 3.4):
            raise ValueError("p is out of range, 1.2 < p < 3.4")
        # parameters p, x_p, and phi_p from Wijers & Galama 1999
        pxf = np.array([[1.0, 3.0, 0.41], [1.2, 1.4, 0.44], [1.4, 1.1, 0.48], [1.6, 0.86, 0.53], [1.8, 0.725, 0.56],
                        [2.0, 0.637, 0.59], [2.2, 0.579, 0.612], [2.5, 0.520, 0.630], [2.7, 0.487, 0.641],
                        [3.0, 0.451, 0.659],
                        [3.2, 0.434, 0.660], [3.4, 0.420, 0.675]])
        xp = np.interp(self.p, pxf[:, 0], pxf[:, 1])  # dimensionless spectral peak
        Fx = np.interp(self.p, pxf[:, 0], pxf[:, 2])  # dimensionless peak flux
        nu0 = np.unique(self.freq)  # unique frequencies in the sample, if loading a data array for frequencies
        nu = nu0 * (1 + self.z)  # rest frame frequency
        nu = np.array(nu)
        Omi, thi, phii, rotstep, latstep = self.get_segments(thj=self.thj, res=self.res)
        Gs, Ei = self.get_structure(gamma=self.g0, en=self.ek,
                                    thi=thi, thc=self.thc, method=self.method, s=self.s, a=self.a, thj=self.thj)
        G = np.empty((thi.size, self.steps))
        SM = np.empty((thi.size, self.steps))
        ghat = np.empty((thi.size, self.steps))
        for i in range(thi.size):
            E2 = self.Et * Ei[i]
            Gg, dM, gh = self.get_gamma_refreshed(Gs[i], self.G1, Ei[i], E2 * Ei[i] / self.ek, self.s1, 0.,
                                        self.steps, self.n, self.k)
            G[i, :], SM[i, :], ghat[i, :] = Gg[0:Gg.size - 1], dM[0:Gg.size - 1], gh[0:Gg.size - 1]
        Obsa = self.get_obsangle(phii=phii, thi=thi, tho=self.tho)
        # calculate the afterglow flux
        Flux, tobs = self.calc_afterglow(G=G, SM=SM, Dl=self.Dl, p=self.p, xp=xp, Fx=Fx,
                                         EB=self.epsB, Ee=self.epse, Gs=Gs, Omi=Omi, Ei=Ei,
                                         n=self.n, k=self.k, tho=self.tho, thi=thi, phii=phii,
                                         thj=self.thj, ghat=ghat, rotstep=rotstep,
                                         latstep=latstep, Obsa=Obsa, nu=nu,
                                         steps=self.steps, XiN=self.xiN)
        # sums all the flux at the same observer times
        LC = self.calc_lightcurve(time=self.t, tobs=tobs, Flux=Flux,
                                  nu_size=nu.size, thi_size=thi.size, phii_size=phii.size,
                                  freq=self.freq, nu0=nu0)
        return LC

def _pnu_synchrotron(nu, B, gamma_m, gamma_c, Ne, p):
    """

    :param nu: frequency in Hz
    :param B: magnetic field in G
    :param gamma_m: minimum Lorentz factor of electrons
    :param gamma_c: electron Lorentz factor at which the cooling is important
    :param Ne: Number of emitting electrons
    :param p: power law index of the electron energy distribution
    :return: Pnu
    """
    qe = 4.80320425e-10  # electron charge in CGS
    Pnu_max = Ne * (electron_mass * speed_of_light ** 2 * sigma_T / (3.0 * qe)) * B
    nu_m = qe * B * gamma_m ** 2 / (2.0 * np.pi * electron_mass * speed_of_light)
    nu_c = qe * B * gamma_c ** 2 / (2.0 * np.pi * electron_mass * speed_of_light)

    Pnu = np.zeros_like(gamma_m)

    # slow cooling
    cooling_msk = (nu_m <= nu_c)

    msk = (nu < nu_m) * cooling_msk
    Pnu[msk] = Pnu_max[msk] * (nu[msk] / nu_m[msk]) ** (1.0 / 3.0)
    msk = (nu_m <= nu) * (nu <= nu_c) * cooling_msk
    Pnu[msk] = Pnu_max[msk] * (nu[msk] / nu_m[msk]) ** (-0.5 * (p - 1.0))
    msk = (nu_c < nu) * cooling_msk
    Pnu[msk] = Pnu_max[msk] * (nu_c[msk] / nu_m[msk]) ** (-0.5 * (p - 1.0)) * (nu[msk] / nu_c[msk]) ** (-0.5 * p)

    # fast cooling
    cooling_msk = (nu_c < nu_m)

    msk = (nu < nu_c) * cooling_msk
    Pnu[msk] = Pnu_max[msk] * (nu[msk] / nu_c[msk]) ** (1.0 / 3.0)
    msk = (nu_c <= nu) * (nu <= nu_m) * cooling_msk
    Pnu[msk] = Pnu_max[msk] * (nu[msk] / nu_c[msk]) ** (-0.5)
    msk = (nu_m < nu) * cooling_msk
    Pnu[msk] = Pnu_max[msk] * (nu_m[msk] / nu_c[msk]) ** (-0.5) * (nu[msk] / nu_m[msk]) ** (-0.5 * p)

    return Pnu

def _get_kn_dynamics(n0, Eej, Mej):
    """
    Calculates blast-wave hydrodynamics. Based on Pe'er (2012) with a numerical correction
    factor to ensure asymptotic convergence to Sedov-Taylor solution (see also Nava et al. 2013; Huang et al. 1999)

    :param n0: ISM density in cm^-3
    :param Eej: Ejecta energy in erg
    :param Mej: ejecta mass in g
    :return: Dynamical outputs - t, R, beta, Gamma, eden, tobs, beta_sh, Gamma_sh.
    """
    from scipy import integrate

    # calculate initial Lorentz factor & velocity from mass and energy
    Gamma0 = 1.0 + Eej / (Mej * speed_of_light ** 2)
    v0 = speed_of_light * (1.0 - Gamma0 ** (-2)) ** 0.5

    # characteristic, maximum, and starting times
    tdec = (3 * Eej / (4 * np.pi * proton_mass * speed_of_light ** 2 * n0 * Gamma0 * (Gamma0 - 1.0) * v0 ** 3)) ** (1.0 / 3.0)
    tmax = 1e3 * tdec
    t0 = 1e-4 * tdec

    # maximum numer of timesteps before abort
    N = 100000
    # maximal fractional difference in variables between timesteps
    s = 0.002

    # initiate variables
    t = np.zeros(N)
    R = np.zeros_like(t)
    Gamma = np.zeros_like(t)
    beta = np.zeros_like(t)
    m = np.zeros_like(t)
    Gamma_sh = np.zeros_like(t)
    beta_sh = np.zeros_like(t)

    t[0] = t0
    Gamma[0] = Gamma0
    beta[0] = v0 / speed_of_light
    R[0] = v0 * t[0]
    m[0] = (4.0 * np.pi / 3.0) * n0 * proton_mass * R[0] ** 3

    g = (4.0 + Gamma[0] ** (-1)) / 3.0
    Gamma_sh[0] = ((Gamma[0] + 1.0) * (g * (Gamma[0] - 1.0) + 1.0) ** 2 / (
                g * (2.0 - g) * (Gamma[0] - 1.0) + 2.0)) ** 0.5
    beta_sh[0] = (1.0 - Gamma_sh[0] ** (-2)) ** 0.5

    # integrate equations
    i = 0
    while t[i] < tmax and i < N - 1:
        # time derivative of variables (time is measured in lab frame)
        Rdot = beta_sh[i] * speed_of_light
        mdot = 4.0 * np.pi * n0 * proton_mass * R[i] ** 2 * Rdot
        Gammadot = -mdot * (g * (Gamma[i] ** 2 - 1.0) - (g - 1.0) * Gamma[i] * beta[i] ** 2) / (
                    Mej + m[i] * (2.0 * g * Gamma[i] - (g - 1.0) * (Gamma[i] ** (-2) + 1.0)))

        # calculate next timestep based on allowed tollerance "s" (or exit condition)
        dt = min(s * R[i] / Rdot, s * np.abs((Gamma[i] - 1.0) / Gammadot), s * m[i] / mdot, tmax - t[i])

        # update variables
        t[i + 1] = t[i] + dt
        R[i + 1] = R[i] + dt * Rdot
        Gamma[i + 1] = Gamma[i] + dt * Gammadot
        m[i + 1] = m[i] + dt * mdot
        beta[i + 1] = (1.0 - Gamma[i + 1] ** (-2)) ** 0.5
        # effectiv adiabatic index, smoothly interpolating between 4/3 in the ultra-relativistic limit and 5/3 in the Newtonian regime
        g = (4.0 + Gamma[i + 1] ** (-1)) / 3.0
        Gamma_sh[i + 1] = ((Gamma[i + 1] + 1.0) * (g * (Gamma[i + 1] - 1.0) + 1.0) ** 2 / (
                    g * (2.0 - g) * (Gamma[i + 1] - 1.0) + 2.0)) ** 0.5
        beta_sh[i + 1] = (1.0 - Gamma_sh[i + 1] ** (-2)) ** 0.5
        i += 1

    # trim arrays to end at last point of iteration
    i += 1
    t = t[:i]
    R = R[:i]
    beta = beta[:i]
    Gamma = Gamma[:i]
    beta_sh = beta_sh[:i]
    Gamma_sh = Gamma_sh[:i]

    g = (4.0 + Gamma ** (-1)) / 3.0
    # calculate post-shock thermal energy density (Blandford & McKee 1976).
    # Assumes cold upstream matter: enthalpy = mass density * c^2
    eden = (Gamma - 1.0) * ((g * Gamma + 1.0) / (g - 1.0)) * n0 * proton_mass * speed_of_light ** 2

    # convert from lab frame to observer time. Approximate expression acounting for radial+azimuthal time-of-flight effect
    # (eq. 26 from Nava et al. 2013; see also Waxman 1997)
    tobs = R / (Gamma ** 2 * (1.0 + beta) * speed_of_light) + np.insert(
        integrate.cumulative_trapezoid((1.0 - beta_sh) / beta_sh, x=R) / speed_of_light, 0, 0.0)

    return t, R, beta, Gamma, eden, tobs, beta_sh, Gamma_sh

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def tophat_redback(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A tophat model implemented directly in redback. Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet opening angle in radians
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'TH'
    s, a = 0.01, 0.5

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglows(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thc, tho=thv, p=p, exp=exp,
                                time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=s, extra_structure_parameter_2=a,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def gaussian_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A Gaussian structure afterglow model implemented directly in redback. Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'GJ'
    s, a = 0.01, 0.5

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglows(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=s, extra_structure_parameter_2=a,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def twocomponent_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A two component model implemented directly in redback. Tophat till thc and then second component till thj.
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Fraction of energy in the outer sheath of the jet. Defaults to 0.01
    :param aa: Lorentz factor outside the core. Defaults to 4.
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = '2C'
    ss = kwargs.get('ss', 0.01)
    aa = kwargs.get('aa', 4)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglows(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def powerlaw_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A Classic powerlaw structured jet implemented directly in redback.
    Tophat with powerlaw energy proportional to theta^ss and lorentz factor proportional to theta^aa outside core.
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Index of energy outside core. Defaults to -3
    :param aa: Index of Lorentz factor outside the core. Defaults to -3
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'PL'
    ss = kwargs.get('ss', 3)
    aa = kwargs.get('aa', -3)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglows(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def alternativepowerlaw_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    An alternative powerlaw structured jet implemented directly in redback. Profile follows (theta/thc^2)^0.5^(-s or -a).
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Index of energy outside core. Defaults to 3
    :param aa: Index of Lorentz factor outside the core. Defaults to 3
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'PL2'
    ss = kwargs.get('ss', 3)
    aa = kwargs.get('aa', 3)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglows(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def doublegaussian_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    Double Gaussian structured jet implemented directly in redback.
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Fractional contribution of energy to second Gaussian. Defaults to 0.1, must be less than 1.
    :param aa: Lorentz factor for second Gaussian, must be less than 1.
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'DG'
    ss = kwargs.get('ss', 0.1)
    aa = kwargs.get('aa', 0.5)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglows(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def tophat_redback_refreshed(time, redshift, thv, loge0, thc, g1, et, s1,
                             logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A Refreshed tophat model implemented directly in redback. Based on Lamb et al. 2019
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet opening angle in radians
    :param g1: Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
    :param et: factor by which total kinetic energy is larger
    :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'TH'
    s, a = 0.01, 0.5

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglowsRefreshed(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, g1=g1, et=et, s1=s1,
                                          thc=thc, thj=thc, tho=thv, p=p, exp=exp,time=time, freq=frequency,
                                          redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=s, extra_structure_parameter_2=a,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def gaussian_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                               logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A Refreshed Gaussian structured jet model implemented directly in redback. Based on Lamb et al. 2019
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param g1: Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
    :param et: factor by which total kinetic energy is larger
    :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'GJ'
    s, a = 0.01, 0.5

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglowsRefreshed(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj,
                                 tho=thv, p=p, exp=exp, g1=g1, et=et, s1=s1,
                                time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=s, extra_structure_parameter_2=a,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def twocomponent_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                                   logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A refreshed two component model implemented directly in redback. Tophat till thc and then second component till thj.
    Based on Lamb et al. 2019 and other work.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param g1: Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
    :param et: factor by which total kinetic energy is larger
    :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Fraction of energy in the outer sheath of the jet. Defaults to 0.01
    :param aa: Lorentz factor outside the core. Defaults to 4.
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = '2C'
    ss = kwargs.get('ss', 0.01)
    aa = kwargs.get('aa', 4.)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglowsRefreshed(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0, thc=thc, thj=thj,
                                          tho=thv, p=p, exp=exp, g1=g1, et=et, s1=s1, time=time, freq=frequency,
                                          redshift=redshift, Dl=dl, method=method, extra_structure_parameter_1=ss,
                                          extra_structure_parameter_2=aa, res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def powerlaw_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                               logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A Classic refreshed powerlaw structured jet implemented directly in redback.
    Tophat with powerlaw energy proportional to theta^ss and lorentz factor proportional to theta^aa outside core.
    Based on Lamb et al. 2019 and other work.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param g1: Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
    :param et: factor by which total kinetic energy is larger
    :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Index of energy outside core. Defaults to -3
    :param aa: Index of Lorentz factor outside the core. Defaults to -3
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'PL'
    ss = kwargs.get('ss', 3)
    aa = kwargs.get('aa', -3)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglowsRefreshed(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, g1=g1, et=et, s1=s1,
                                 ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def alternativepowerlaw_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                                          logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    An alternative refreshed powerlaw structured jet implemented directly in redback. Profile follows (theta/thc^2)^0.5^(-s or -a).
    Based on Lamb et al. 2019.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param g1: Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
    :param et: factor by which total kinetic energy is larger
    :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Index of energy outside core. Defaults to 3
    :param aa: Index of Lorentz factor outside the core. Defaults to 3
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'PL2'
    ss = kwargs.get('ss', 3)
    aa = kwargs.get('aa', 3)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglowsRefreshed(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, g1=g1, et=et, s1=s1,
                                          ek=e0, thc=thc, thj=thj, tho=thv, p=p, exp=exp,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def doublegaussian_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                                     logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    Double Gaussian structured, refreshed jet implemented directly in redback.
    Based on Lamb et al. 2019 and other work.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    :param time: time in days
    :param redshift: source redshift
    :param thv: observer viewing angle in radians
    :param loge0: jet energy in \log_{10} ergs
    :param thc: jet core size in radians
    :param thj: jet edge in radians (thc < thj < pi/2)
    :param g1: Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
    :param et: factor by which total kinetic energy is larger
    :param s1: index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
    :param logn0: ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
    :param p: electron power law index
    :param logepse: partition fraction in electrons
    :param logepsb: partition fraction in magnetic field
    :param g0: initial lorentz factor
    :param xiN: fraction of electrons that get accelerated. Defaults to 1.
    :param kwargs: additional keyword arguments
    :param res: resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
    :param steps: number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    :param k: power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    :param expansion: 0 or 1 to dictate whether to include expansion effects. Defaults to 1
    :param ss: Fractional contribution of energy to second Gaussian. Defaults to 0.1, must be less than 1.
    :param aa: Lorentz factor for second Gaussian, must be less than 1.
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    exp = kwargs.get('expansion', 1)
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    nism = 10 ** logn0
    e0 = 10 ** loge0
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    method = 'DG'
    ss = kwargs.get('ss', 0.1)
    aa = kwargs.get('aa', 0.5)

    # Set resolution dynamically
    sep = max(thv - thc, 0)
    order = min(int((2 - 10 * sep) * thc * g0), 100)
    default_res = max(10, order)
    res = kwargs.get('res', default_res)
    steps = kwargs.get('steps', 250)
    ag_class = RedbackAfterglowsRefreshed(k=k, n=nism, epse=epse, epsb=epsb, g0=g0, ek=e0,
                                 thc=thc, thj=thj, tho=thv, p=p, exp=exp, g1=g1, et=et, s1=s1,
                                 time=time, freq=frequency, redshift=redshift, Dl=dl, method=method,
                                 extra_structure_parameter_1=ss, extra_structure_parameter_2=aa,
                                 res=res, xiN=xiN, steps=steps)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def cocoon(time, redshift, umax, umin, loge0, k, mej, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A cocoon afterglow model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param umax: initial outflow 4 velocity maximum
    :param umin: minimum outflow 4 velocity
    :param loge0: log10 fidicial energy in velocity distribution E(>u) = E0u^-k in erg
    :param k: power law index of energy velocity distribution
    :param mej: mass of material at umax in solar masses
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['cocoon']
    frequency = kwargs['frequency']
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'uMax': umax, 'Er': e0,
         'uMin': umin, 'k': k, 'MFast_solar': mej, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def kilonova_afterglow(time, redshift, umax, umin, loge0, k, mej, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A kilonova afterglow model from afterglowpy, similar to cocoon but with constraints.

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param umax: initial outflow 4 velocity maximum
    :param umin: minimum outflow 4 velocity
    :param loge0: log10 fidicial energy in velocity distribution E(>u) = E0u^-k in erg
    :param k: power law index of energy velocity distribution
    :param mej: mass of material at umax in solar masses
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    output = cocoon(time=time, redshift=redshift, umax=umax, umin=umin, loge0=loge0,
                    k=k, mej=mej, logn0=logn0,p=p,logepse=logepse,logepsb=logepsb,
                    ksin=ksin, g0=g0, **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def cone_afterglow(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A cone afterglow model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thw: wing truncation angle of jet thw = thw*thc
    :param thc: half width of jet core in radians
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['cone']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def gaussiancore(time, redshift, thv, loge0, thc, thw, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A gaussiancore model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thw: wing truncation angle of jet thw = thw*thc
    :param thc: half width of jet core in radians
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['gaussian_w_core']
    frequency = kwargs['frequency']

    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def gaussian(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A gaussian structured jet model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thw: wing truncation angle of jet thw = thw*thc
    :param thc: half width of jet core in radians
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['gaussian']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def smoothpowerlaw(time, redshift, thv, loge0, thw, thc, beta, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A smoothpowerlaw structured jet model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thw: wing truncation angle of jet thw = thw*thc
    :param thc: half width of jet core in radians
    :param beta: index for power-law structure, theta^-b
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['smooth_power_law']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw, 'b': beta}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def powerlawcore(time, redshift, thv, loge0, thw, thc, beta, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A power law with core structured jet model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thw: wing truncation angle of jet thw = thw*thc
    :param thc: half width of jet core in radians
    :param beta: index for power-law structure, theta^-b
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['powerlaw_w_core']
    frequency = kwargs['frequency']
    thw = thw * thc
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb

    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres, 'thetaWing': thw, 'b': beta}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def tophat(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """
    A tophat jet model from afterglowpy

    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param ksin: fraction of electrons that get accelerated
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models. assuming a monochromatic
    """
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    spread = kwargs.get('spread', False)
    latres = kwargs.get('latres', 2)
    tres = kwargs.get('tres', 100)
    spectype = kwargs.get('spectype', 0)
    l0 = kwargs.get('L0', 0)
    q = kwargs.get('q', 0)
    ts = kwargs.get('ts', 0)
    jettype = jettype_dict['tophat']
    frequency = kwargs['frequency']
    e0 = 10 ** loge0
    n0 = 10 ** logn0
    epse = 10 ** logepse
    epsb = 10 ** logepsb
    Z = {'jetType': jettype, 'specType': spectype, 'thetaObs': thv, 'E0': e0,
         'thetaCore': thc, 'n0': n0, 'p': p, 'epsilon_e': epse, 'epsilon_B': epsb,
         'xi_N': ksin, 'd_L': dl, 'z': redshift, 'L0': l0, 'q': q, 'ts': ts, 'g0': g0,
         'spread': spread, 'latRes': latres, 'tRes': tres}
    flux_density = afterglow.fluxDensity(time, frequency, **Z)
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value


def tophat_from_emulator(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, **kwargs):
    """
    Evaluate a tophat afterglow model using an mpl regressor. Note that this model predicts for a fixed redshift = 0.01 and fixed ksin = 1.
    This tophat model does not include all of the ususal kwargs

    :param time: time in days in observer frame, should be in range 0.1 to 300
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param frequency: frequency of the band to view in- single number or same length as time array
    :param output_format: Whether to output flux density or AB mag, specified by 'flux_density' or 'magnitude'
    :return: flux density or AB mag predicted by emulator. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models
    """

    from redback_surrogates.afterglowmodels import tophat_emulator
    
    z1=0.01
    z2= redshift
    frequency= np.log10(kwargs['frequency'])    
    flux_density = tophat_emulator(new_time=time/(1+z2), thv=thv, loge0=loge0, thc=thc, logn0=logn0, p=p,
                                            logepse=logepse, logepsb=logepsb, g0=g0,frequency=frequency)
        
    #scaling flux density with redshift
    dl1 = cosmo.luminosity_distance(z1)
    dl2 = cosmo.luminosity_distance(z2)
    scale_factor = ((dl1**2)*(1+z1)) / (dl2**2)
    flux_density=flux_density*scale_factor
    
    if kwargs['output_format'] == 'flux_density':
        return flux_density
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def afterglow_models_with_energy_injection(time, **kwargs):
    """
    A base class for afterglowpy models with energy injection.

    :param time: time in days in observer frame
    :param kwargs: Additional keyword arguments
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param base_model: A string to indicate the type of jet model to use.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']
    if isfunction(base_model):
        function = base_model
    elif base_model not in jet_spreading_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")
    kwargs['ts'] = kwargs['ts'] * day_to_s
    kwargs['spread'] = True
    output = function(time, **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def afterglow_models_with_jet_spread(time, **kwargs):
    """
    A base class for afterglow models with jet spreading. Note, with these models you cannot sample in g0.

    :param time: time in days in observer frame
    :param kwargs: Additional keyword arguments
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param l0, ts, q: energy injection parameters, defaults to 0
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param base_model: A string to indicate the type of jet model to use.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
        For a proper calculation of the magntitude use the sed variant models.
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']
    if isfunction(base_model):
        function = base_model
    elif base_model not in jet_spreading_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")
    kwargs['spread'] = True
    kwargs.pop('g0')
    output = function(time, **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def afterglow_models_sed(time, **kwargs):
    """
    A base class for afterglowpy models for bandpass magnitudes/flux/spectra/sncosmo source.

    :param time: time in days in observer frame
    :param kwargs: Additional keyword arguments, must be all parameters required by the base model and the following:
    :param base_model: A string to indicate the type of jet model to use.
    :param spread: whether jet can spread, defaults to False
    :param latres: latitudinal resolution for structured jets, defaults to 2
    :param tres: time resolution of shock evolution, defaults to 100
    :param spectype: whether to have inverse compton, defaults to 0, i.e., no inverse compton.
        Change to 1 for including inverse compton emission.
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :return: set by output format - 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']
    if isfunction(base_model):
        function = base_model
    elif base_model not in jet_spreading_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")
    temp_kwargs = kwargs.copy()
    temp_kwargs['spread'] = kwargs.get('spread', False)
    lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 200))
    frequency = lambda_to_nu(lambda_observer_frame)
    time_observer_frame = np.linspace(0, np.max(time), 300)
    times_mesh, frequency_mesh = np.meshgrid(time_observer_frame, frequency)
    temp_kwargs['frequency'] = frequency_mesh
    temp_kwargs['output_format'] = 'flux_density'
    output = function(times_mesh, **temp_kwargs).T
    fmjy = output * uu.mJy
    spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                 equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom))
    if kwargs['output_format'] == 'spectra':
        return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                    lambdas=lambda_observer_frame,
                                                                    spectra=spectra)
    else:
        return get_correct_output_format_from_spectra(time=time, time_eval=time_observer_frame,
                                                      spectra=spectra, lambda_array=lambda_observer_frame,
                                                      **kwargs)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024ApJS..273...17W/abstract')
def jetsimpy_tophat(time, redshift, thv, loge0, thc, nism, A, p, logepse, logepsb, g0, **kwargs):
    """
    A tophat jet model from jetsimpy
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param nism: number density of ISM in cm^-3 (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
    :param A: wind density scale (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
    :param p: electron distribution power law index.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
    """
    import jetsimpy #Can not use models unless jetsimpy is downloaded
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    P = dict(Eiso = 10 ** loge0, lf = g0, theta_c = thc, n0 = nism, A = A, eps_e = 10 ** logepse, eps_b = 10 ** logepsb, p = p, theta_v = thv, d = dl*3.24078e-25, z = redshift) #make a param dict
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        flux_density = jetsimpy.FluxDensity_tophat(time, frequency, P)
        return flux_density   
    else:
        frequency = bands_to_frequency(kwargs['bands'])       
        flux_density = jetsimpy.FluxDensity_tophat(time, frequency, P)
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024ApJS..273...17W/abstract')
def jetsimpy_gaussian(time, redshift, thv, loge0, thc, nism, A, p, logepse, logepsb, g0, **kwargs):
    """
    A gaussian jet model from jetsimpy
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param nism: number density of ISM in cm^-3 (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
    :param A: wind density scale (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
    :param p: electron distribution power law index.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param g0: initial lorentz factor
    :param kwargs: Additional keyword arguments
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
    """
    import jetsimpy #Can not use models unless jetsimpy is downloaded
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    P = dict(Eiso = 10 ** loge0, lf = g0, theta_c = thc, n0 = nism, A = A, eps_e = 10 ** logepse, eps_b = 10 ** logepsb, p = p, theta_v = thv, d = dl*3.24078e-25, z = redshift) #make a param dict
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        flux_density = jetsimpy.FluxDensity_gaussian(time, frequency, P)
        return flux_density   
    else:
        frequency = bands_to_frequency(kwargs['bands'])       
        flux_density = jetsimpy.FluxDensity_gaussian(time, frequency, P)
        return calc_ABmag_from_flux_density(flux_density).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2024ApJS..273...17W/abstract')
def jetsimpy_powerlaw(time, redshift, thv, loge0, thc, nism, A, p, logepse, logepsb, g0, s, **kwargs):
    """
    A power-law jet model from jetsimpy
    :param time: time in days in observer frame
    :param redshift: source redshift
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param nism: number density of ISM in cm^-3 (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
    :param A: wind density scale (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
    :param p: electron distribution power law index.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param g0: initial lorentz factor
    :param s: power-law jet slope
    :param kwargs: Additional keyword arguments
    :param output_format: Whether to output flux density or AB mag
    :param frequency: frequency in Hz for the flux density calculation
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.
    """
    import jetsimpy #Can not use models unless jetsimpy is downloaded
    time = time * day_to_s
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    P = dict(Eiso = 10 ** loge0, lf = g0, theta_c = thc, n0 = nism, A = A, eps_e = 10 ** logepse, eps_b = 10 ** logepsb, p = p, theta_v = thv, d = dl*3.24078e-25, z = redshift, s = s) #make a param dict
    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        flux_density = jetsimpy.FluxDensity_powerlaw(time, frequency, P)
        return flux_density   
    else:
        frequency = bands_to_frequency(kwargs['bands'])       
        flux_density = jetsimpy.FluxDensity_powerlaw(time, frequency, P)
        return calc_ABmag_from_flux_density(flux_density).value
