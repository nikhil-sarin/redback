from astropy.cosmology import Planck18 as cosmo  # noqa
from inspect import isfunction
from redback.utils import logger, citation_wrapper, calc_ABmag_from_flux_density, lambda_to_nu, bands_to_frequency
from redback.constants import day_to_s, speed_of_light, solar_mass, proton_mass, electron_mass, sigma_T
from redback.sed import get_correct_output_format_from_spectra
import astropy.units as uu
import numpy as np
from collections import namedtuple
from scipy.special import erf
from redback.wrappers import cond_jit

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


# Physical constants (as module-level constants for Numba)
MP = 1.6726231e-24  # g, mass of proton
ME = 9.1093897e-28  # g, mass of electron
CC = 2.99792453e10  # cm s^-1, speed of light
QE = 4.8032068e-10  # esu, electron charge
C2 = CC * CC
SIGT = (QE * QE / (ME * C2)) ** 2 * (8 * np.pi / 3)  # Thomson cross-section
FOURPI = 4 * np.pi
DAY_TO_S = 86400.0


# Numba-compiled utility functions
@cond_jit(nopython=True, fastmath=True, cache=True)
def get_segments_numba(thj, res):
    """Calculate jet segments - matches Python exactly"""
    latstep = thj / res
    rotstep = 2.0 * np.pi / res
    Nlatstep = int(res)
    Nrotstep = int(2 * np.pi / rotstep)

    Omi = np.zeros(Nlatstep * Nrotstep)
    phi = np.linspace(rotstep, Nrotstep * rotstep, Nrotstep)

    # Match the Python loop exactly
    for i in range(Nlatstep):
        start_idx = i * Nrotstep
        end_idx = (i + 1) * Nrotstep
        for j in range(Nrotstep):
            Omi[start_idx + j] = (phi[j] - (phi[j] - rotstep)) * (np.cos(i * latstep) - np.cos((i + 1) * latstep))

    thi = np.linspace(latstep - latstep / 2., Nlatstep * latstep - latstep / 2., Nlatstep)
    phii = np.linspace(rotstep - rotstep / 2., Nrotstep * rotstep - rotstep / 2., Nrotstep)

    return Omi, thi, phii, rotstep, latstep


@cond_jit(nopython=True, fastmath=True, cache=True)
def erf_approximation_numba(x):
    """Numba-compatible erf approximation that matches scipy.special.erf closely"""
    # Use a high-precision approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    result = np.zeros_like(x)
    for i in range(len(x)):
        sign = 1.0 if x[i] >= 0 else -1.0
        x_abs = abs(x[i])

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)

        result[i] = sign * y

    return result


@cond_jit(nopython=True, fastmath=True, cache=True)
def get_structure_numba(gamma, en, thi, thc, method_id, s, a, thj):
    """Calculate jet structure - matches Python exactly"""
    n = thi.shape[0]
    Gs = np.full(n, gamma)
    Ei = np.full(n, en)

    if method_id == 1:  # Top-hat - match Python implementation exactly
        thj_used = min(thc, thj)
        # Use the erf approximation that matches Python
        erf_arg = -(thi - thj_used) * 1000.0
        fac = (erf_approximation_numba(erf_arg) * 0.5) + 0.5
        Gs = (Gs - 1) * fac + 1.0000000000001
        Ei *= fac

    elif method_id == 2:  # Gaussian
        fac = np.exp(-0.5 * (thi / thc) ** 2)
        Gs = (Gs - 1.0) * fac + 1.0000000000001
        Ei *= fac

    elif method_id == 3:  # Power-law
        for i in range(n):
            if thi[i] >= thc:
                fac_s = (thc / thi[i]) ** s
                fac_a = (thc / thi[i]) ** a
                Ei[i] *= fac_s
                Gs[i] = (Gs[i] - 1.0) * fac_a + 1.000000000000001

    elif method_id == 4:  # Alternative powerlaw
        fac = (1 + (thi / thc) ** 2) ** 0.5
        Gs = 1.000000000001 + (Gs - 1.0) * fac ** (-a)
        Ei *= fac ** (-s)

    elif method_id == 5:  # Two-Component
        for i in range(n):
            if thi[i] > thc:
                Gs[i] = a
                Ei[i] *= s

    elif method_id == 6:  # Double Gaussian
        for i in range(n):
            # Energy structure
            Ei[i] = Ei[i] * ((1.0 - s) * np.exp(-0.5 * (thi[i] / thc) ** 2) +
                             s * np.exp(-0.5 * (thi[i] / thj) ** 2))

            # Lorentz factor structure
            temp = ((1.0 - s) * np.exp(-0.5 * (thi[i] / thc) ** 2) +
                    s * np.exp(-0.5 * (thi[i] / thj) ** 2)) / \
                   ((1.0 - s / a) * np.exp(-0.5 * (thi[i] / thc) ** 2) +
                    (s / a) * np.exp(-0.5 * (thi[i] / thj) ** 2))

            if not np.isfinite(temp):
                Gs[i] = a
            else:
                Gs[i] = (Gs[i] - 1.0) * temp + 1.0000000000001

    return Gs, Ei


@cond_jit(nopython=True, fastmath=True, cache=True)
def get_obsangle_numba(phii, thi, tho):
    """Calculate observer angles - matches Python exactly"""
    phi = 0.0  # rotational symmetry
    sin_thi = np.sin(thi)
    cos_thi = np.cos(thi)

    # Match the Python broadcasting exactly
    n_thi = thi.shape[0]
    n_phi = phii.shape[0]
    Obsa = np.zeros(n_thi * n_phi)

    idx = 0
    for i in range(n_thi):
        for j in range(n_phi):
            f1 = np.sin(phi) * np.sin(tho) * np.sin(phii[j]) * sin_thi[i]
            f2 = np.cos(phi) * np.sin(tho) * np.cos(phii[j]) * sin_thi[i]
            ang = np.cos(tho) * cos_thi[i] + f2 + f1

            # Ensure valid range for acos
            if ang > 1.0:
                ang = 1.0
            elif ang < -1.0:
                ang = -1.0

            Obsa[idx] = np.arccos(ang)
            idx += 1

    return Obsa


@cond_jit(nopython=True, fastmath=True, cache=True)
def RK4_step_numba(ghat, dm_rk4, G_rk4, M, fac, therm):
    """Single RK4 step - matches Python exactly"""
    ghatm1 = ghat - 1.0
    dm_base10 = 10.0 ** dm_rk4
    G_rk4_sq = G_rk4 * G_rk4
    _G_rk4 = 1.0 / G_rk4

    return fac * dm_base10 * (ghat * (G_rk4_sq - 1.0) - ghatm1 * (G_rk4 - _G_rk4)) / \
        (M + dm_base10 * (therm + (1.0 - therm) * (2.0 * ghat * G_rk4 -
                                                   ghatm1 * (1 + 1.0 / G_rk4_sq))))


@cond_jit(nopython=True, fastmath=True, cache=True)
def get_gamma_numba(G0, Eps, therm, steps, n0, k):
    """Calculate gamma evolution - matches Python exactly"""
    n_g0 = G0.shape[0]
    steps_int = int(steps)

    # Parameter setting - match Python exactly
    Rmin = 1e10
    Rmax = 1e24
    Nmin = (FOURPI / (3.0 - k)) * n0 * Rmin ** (3.0 - k)
    Nmax = (FOURPI / (3.0 - k)) * n0 * Rmax ** (3.0 - k)
    dlogm = np.log10(Nmin * MP)
    h = np.log10(Nmax / Nmin) / float(steps_int)
    fac = -h * np.log(10.0)

    M = Eps / (G0 * C2)

    # Arrays for state
    state_G = np.zeros((steps_int, n_g0), dtype=np.float64)
    state_dm = np.zeros((steps_int, n_g0), dtype=np.float64)
    state_gH = np.zeros((steps_int, n_g0), dtype=np.float64)

    # Initial values
    G = G0.astype(np.float64).copy()
    dm = np.full(n_g0, dlogm, dtype=np.float64)

    # Main loop - match Python exactly
    for i in range(steps_int):
        # Store current values
        state_G[i] = G
        state_dm[i] = dm

        # Calculate intermediate values
        G2m1 = G * G - 1.0
        G2m1_root = np.sqrt(G2m1)

        # Temperature and adiabatic index - match Python exactly
        theta = G2m1_root * (G2m1_root + 1.07 * G2m1) / (3.0 * (1 + G2m1_root + 1.07 * G2m1))
        z = theta / (0.24 + theta)
        ghat = ((((((1.07136 * z - 2.39332) * z + 2.32513) * z -
                   0.96583) * z + 0.18203) * z - 1.21937) * z + 5.0) / 3.0

        state_gH[i] = ghat

        # 4th order Runge-Kutta - match Python exactly
        F1 = RK4_step_numba(ghat, dm, G, M, fac, therm)
        F2 = RK4_step_numba(ghat, dm + 0.5 * h, G + 0.5 * F1, M, fac, therm)
        F3 = RK4_step_numba(ghat, dm + 0.5 * h, G + 0.5 * F2, M, fac, therm)
        F4 = RK4_step_numba(ghat, dm + h, G + F3, M, fac, therm)

        # Update state - match Python exactly
        G = G + (F1 + 2.0 * (F2 + F3) + F4) / 6.0 + 1e-15
        dm = dm + h

    return state_G.T, (10.0 ** state_dm).T, state_gH.T


@cond_jit(nopython=True, fastmath=True, cache=True)
def calc_afterglow_step1_numba(G, dm, p, xp, Fx, EB, Ee, n, k, thi, ghat, rotstep, latstep, xiN, is_expansion, a1, res):
    """Calculate synchrotron parameters - matches Python exactly"""
    size = G.shape[0]

    # Ensure all variables are properly typed arrays
    Gm1 = G - 1.0
    G2 = G * G
    beta = np.sqrt(1 - 1.0 / G2)
    Ne = dm / MP

    # Side-ways expansion - match Python exactly
    cs = np.sqrt(C2 * ghat * (ghat - 1) * Gm1 / (1 + ghat * Gm1))
    te = np.arcsin(cs / (CC * np.sqrt(G2 - 1.0)))

    # Initialize OmG as array from the start
    OmG = np.zeros(size, dtype=np.float64)

    # Expansion effects - match Python logic exactly
    if is_expansion:
        ex = te / G ** (a1 + 1)
        fac = 0.5 * latstep
        for i in range(size):
            OmG[i] = rotstep * (np.cos(thi - fac) - np.cos(ex[i] / res + thi + fac))
    else:
        ex = np.ones(size, dtype=np.float64)  # Make sure ex is an array
        fac = 0.5 * latstep
        omg_val = rotstep * (np.cos(thi - fac) - np.cos(thi + fac))
        for i in range(size):
            OmG[i] = omg_val

    # Calculate R - match Python exactly
    exponent = np.ones(size, dtype=np.float64)
    for i in range(1, size):
        exponent[i] = ((1 - np.cos(latstep + ex[0])) / (1 - np.cos(latstep + ex[i]))) ** (1.0 / 2.0)

    R = ((3.0 - k) * Ne[:size] / (FOURPI * n)) ** (1.0 / (3.0 - k))

    # Match Python: R[1:] = np.diff(R) * exponent[1:size] ** (1. / (3. - k))
    R_orig = R.copy()
    for i in range(1, size):
        R[i] = (R_orig[i] - R_orig[i - 1]) * exponent[i] ** (1.0 / (3.0 - k))

    R = np.cumsum(R)  # Match Python: R = np.cumsum(R)

    n0 = n * R ** (-k)

    # Forward shock parameters - ensure all are arrays
    B = np.sqrt(2 * FOURPI * EB * n0 * MP * C2 * ((ghat * G + 1.0) / (ghat - 1.0)) * Gm1)
    gmm = np.sqrt(1.5 * FOURPI * QE / (SIGT * B))

    # Calculate gm - ensure it's always an array
    gm = np.zeros(size, dtype=np.float64)

    if p > 2:
        gp = (p - 2) / (p - 1)
        gm[:] = gp * (Ee / xiN) * Gm1 * (MP / ME)
    elif p == 2:
        for i in range(size):
            gp_i = 1 / np.log(gmm[i] / ((Ee / xiN) * Gm1[i] * (MP / ME)))
            gm[i] = gp_i * (Ee / xiN) * Gm1[i] * (MP / ME)
    else:  # p < 2
        for i in range(size):
            gm[i] = ((2 - p) / (p - 1) * (MP / ME) * (Ee / xiN) * Gm1[i] * gmm[i] ** (p - 2)) ** (1 / (p - 1))

    # Ensure all output arrays are properly formed
    nump = 3.0 * xp * gm * gm * QE * B / (FOURPI * ME * CC)
    Pp = xiN * Fx * ME * C2 * SIGT * B / (3.0 * QE)
    KT = gm * ME * C2

    return beta, Ne, OmG, R, B, gm, nump, Pp, KT


@cond_jit(nopython=True, fastmath=True, cache=True)
def calc_afterglow_step2_numba(Dl, Om0, rotstep, latstep, Obsa, beta, Ne, OmG, R, B, gm, nump, Pp, KT, G, is_expansion):
    """Calculate emission properties - matches Python exactly"""
    Dl2 = Dl * Dl
    NO = Om0 * Ne / FOURPI  # Match Python: NO = Om0 * Ne / self.fourpi
    cos_Obsa = np.cos(Obsa)

    size = G.shape[0]

    # Match Python expansion logic exactly
    Om = np.zeros(size, dtype=np.float64)
    thii = np.zeros(size, dtype=np.float64)

    if is_expansion:
        for i in range(size):
            Om[i] = max(Om0, OmG[i])  # Match Python: Om = np.maximum(Om0, OmG)
            thii[i] = np.arccos(1.0 - Om[i] / rotstep)
    else:
        for i in range(size):
            Om[i] = OmG[i]  # Match Python: Om = OmG
            thii[i] = np.arccos(1.0 - Om[i] / rotstep)

    # Match Python: R_diff = np.diff(R,prepend=0)
    R_diff = np.zeros(size, dtype=np.float64)
    R_diff[0] = R[0]
    for i in range(1, size):
        R_diff[i] = R[i] - R[i - 1]

    dt = R_diff * (1.0 / beta[:size] - cos_Obsa) / CC
    dto = R_diff * (1.0 / beta[:size] - 1.0) / CC

    tobs = np.cumsum(dt)
    tobso = np.cumsum(dto)

    # Forward shock - match Python exactly
    dop = 1.0 / (G * (1.0 - beta * cos_Obsa))
    gc = 6.0 * np.pi * ME * CC / (G * SIGT * B * B * tobso)
    nucp = 0.286 * 3.0 * gc * gc * QE * B / (FOURPI * ME * CC)
    num = dop * nump
    nuc = dop * nucp
    Fmax = NO * Pp * dop * dop * dop / (FOURPI * Dl2)

    # Self-absorption - match Python exactly
    FBB = 2 * Om * np.cos(thii) * dop * KT * R * R / (C2 * Dl2)

    return FBB, Fmax, nuc, num, tobs


@cond_jit(nopython=True, fastmath=True, cache=True)
def get_ag_numba(FBB, nuc, num, nu1, Fmax, p):
    """Calculate flux at frequency - matches Python exactly with overlapping conditions"""
    Fluxt = np.zeros(num.shape[0], dtype=np.float64)

    # Allow overlapping conditions where later ones can overwrite earlier ones
    for i in range(num.shape[0]):
        # Fast cooling regime
        if (nuc[i] < num[i]) and (nu1 < nuc[i]):
            Fluxt[i] = Fmax[i] * (nu1 / nuc[i]) ** (1.0 / 3.0)
        if (nuc[i] < nu1) and (nuc[i] < num[i]):
            Fluxt[i] = Fmax[i] * (nu1 / nuc[i]) ** (-1.0 / 2.0)
        if (num[i] < nu1) and (nuc[i] < num[i]):
            Fluxt[i] = Fmax[i] * (num[i] / nuc[i]) ** (-1.0 / 2.0) * (nu1 / num[i]) ** (-p / 2.0)
        # Slow cooling regime
        if (num[i] < nuc[i]) and (nu1 < num[i]):
            Fluxt[i] = Fmax[i] * (nu1 / num[i]) ** (1.0 / 3.0)
        if (num[i] < nu1) and (num[i] < nuc[i]):
            Fluxt[i] = Fmax[i] * (nu1 / num[i]) ** (-(p - 1.0) / 2.0)
        if (nuc[i] < nu1) and (num[i] < nuc[i]):
            Fluxt[i] = Fmax[i] * (nuc[i] / num[i]) ** (-(p - 1.0) / 2.0) * (nu1 / nuc[i]) ** (-p / 2.0)

    # Self-absorption
    FBB_adj = FBB * nu1 ** 2.0 * np.maximum(1.0, (nu1 / num) ** 0.5)
    for i in range(len(Fluxt)):
        Fluxt[i] = min(FBB_adj[i], Fluxt[i])

    return Fluxt

@cond_jit(nopython=True, fastmath=True, cache=True)
def get_gamma_refreshed_numba(G0, G1, Eps, Eps2, s1, therm, steps, n0, k):
    """Calculate gamma evolution with energy injection - matches Python exactly"""
    Eps0 = Eps
    n = n0

    # Parameter setting - match Python exactly
    Rmin = 1e10
    Rmax = 1e24
    Nmin = (FOURPI / 3.0) * n * Rmin ** (3.0 - k)
    Nmax = (FOURPI / 3.0) * n * Rmax ** (3.0 - k)

    dlogm = np.log10(Nmin * MP)
    h = (np.log10(Nmax) - np.log10(Nmin)) / steps

    # Arrays for state
    steps_int = int(steps)
    G = np.ones(steps_int + 1, dtype=np.float64)
    dm = np.zeros(steps_int, dtype=np.float64)
    gH = np.zeros(steps_int, dtype=np.float64)

    # Initial values
    G[0] = G0
    dm[0] = dlogm
    M = Eps / (G0 * C2)

    # Main loop - match Python exactly
    for i in range(steps_int):
        # Calculate intermediate values
        G2m1 = G[i] * G[i] - 1.0
        G2m1_root = np.sqrt(G2m1)

        # Temperature and adiabatic index - match Python exactly
        theta = G2m1_root / 3.0 * ((G2m1_root + 1.07 * G2m1) / (1 + G2m1_root + 1.07 * G2m1))
        z = theta / (0.24 + theta)
        ghat = (5.0 - 1.21937 * z + 0.18203 * z ** 2 - 0.96583 * z ** 3 +
                2.32513 * z ** 4 - 2.39332 * z ** 5 + 1.07136 * z ** 6) / 3.0

        dm[i] = dlogm + i * h
        gH[i] = ghat

        # Helper values for RK4
        dm_10 = 10.0 ** dm[i]
        h_log10 = h * np.log(10.0)

        # 4th order Runge-Kutta - match Python exactly
        F1 = -h_log10 * dm_10 * (ghat * G2m1 - (ghat - 1.0) * G[i] * (1.0 - 1.0 / G[i] ** 2)) / \
             (M + therm * dm_10 + (1.0 - therm) * dm_10 * (2.0 * ghat * G[i] - (ghat - 1.0) * (1.0 + 1.0 / G[i] ** 2)))

        G_F1_2 = G[i] + F1 / 2.0
        dm_h2_10 = 10.0 ** (dm[i] + h / 2.0)
        F2 = -h_log10 * dm_h2_10 * (ghat * (G_F1_2 ** 2 - 1.0) - (ghat - 1.0) * G_F1_2 * (1.0 - 1.0 / G_F1_2 ** 2)) / \
             (M + therm * dm_h2_10 + (1.0 - therm) * dm_h2_10 * (
                         2.0 * ghat * G_F1_2 - (ghat - 1.0) * (1.0 + 1.0 / G_F1_2 ** 2)))

        G_F2_2 = G[i] + F2 / 2.0
        F3 = -h_log10 * dm_h2_10 * (ghat * (G_F2_2 ** 2 - 1.0) - (ghat - 1.0) * G_F2_2 * (1.0 - 1.0 / G_F2_2 ** 2)) / \
             (M + therm * dm_h2_10 + (1.0 - therm) * dm_h2_10 * (
                         2.0 * ghat * G_F2_2 - (ghat - 1.0) * (1.0 + 1.0 / G_F2_2 ** 2)))

        G_F3 = G[i] + F3
        dm_h_10 = 10.0 ** (dm[i] + h)
        F4 = -h_log10 * dm_h_10 * (ghat * (G_F3 ** 2 - 1.0) - (ghat - 1.0) * G_F3 * (1.0 - 1.0 / G_F3 ** 2)) / \
             (M + therm * dm_h_10 + (1.0 - therm) * dm_h_10 * (
                         2.0 * ghat * G_F3 - (ghat - 1.0) * (1.0 + 1.0 / G_F3 ** 2)))

        # Update state
        G[i + 1] = G[i] + (F1 + 2.0 * F2 + 2.0 * F3 + F4) / 6.0

        # Energy injection - match Python exactly
        if G[i + 1] <= G1:
            Eps1 = Eps
            beta_ratio = np.sqrt(G[i + 1] ** 2 - 1.0) / np.sqrt(G1 ** 2 - 1.0)
            Eps = min(Eps0 * (beta_ratio ** (-s1)), Eps2)
            M += (Eps - Eps1) / (G[i] * C2)

    # Return arrays matching the Python version
    G_out = G[:-1]  # Remove the last element to match Python size
    dm_out = 10.0 ** dm
    gH_out = gH

    return G_out, dm_out, gH_out

# Main class with Numba acceleration
class RedbackAfterglows:
    def __init__(self, k, n, epsb, epse, g0, ek, thc, thj, tho, p, exp, time, freq, redshift, Dl,
                 extra_structure_parameter_1, extra_structure_parameter_2, method='TH', res=100, steps=int(500), xiN=1,
                 a1=1):
        """
        A general class for afterglow models implemented directly in redback.

        This class is not meant to be used directly but instead via the interface for each specific model.
        The afterglows are based on the method shown in Lamb, Mandel & Resmi 2018 and other papers.
        Script was originally written by En-Tzu Lin <entzulin@gapp.nthu.edu.tw> and Gavin Lamb <g.p.lamb@ljmu.ac.uk>
        and modified and implemented into redback by Nikhil Sarin <nsarin.astro@gmail.com>.
        Includes wind-like mediums, expansion and multiple jet structures.
        Includes SSA and uses Numba for acceleration (or numpy if Numba is not installed).

        Parameters
        ----------
        k : int
            0 or 2 for constant or wind density.
        n : float
            ISM, ambient number density.
        epsb : float
            Magnetic fraction.
        epse : float
            Electron fraction.
        g0 : float
            Initial Lorentz factor.
        ek : float
            Kinetic energy.
        thc : float
            Core angle.
        thj : float
            Jet outer angle. For tophat jets thc=thj.
        tho : float
            Observer's viewing angle.
        p : float
            Electron power-law index.
        exp : bool
            Boolean for whether to include sound speed expansion.
        time : np.ndarray
            Lightcurve time steps.
        freq : np.ndarray
            Lightcurve frequencies.
        redshift : float
            Source redshift.
        Dl : float
            Luminosity distance.
        extra_structure_parameter_1 : float
            Extra structure specific parameter #1.
            Specifically, this parameter sets the index on energy for power-law jets,
            the fractional energy contribution for the Double Gaussian (must be less than 1),
            or the energy fraction for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        extra_structure_parameter_2 : float
            Extra structure specific parameter #2.
            Specifically, this parameter sets the index on Lorentz factor for power-law jets,
            the Lorentz factor for second Gaussian (must be less than 1),
            or the Lorentz factor for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        method : str, optional
            Type of jet structure to use. Defaults to 'TH' for tophat jet.
            Other options are '2C', 'GJ', 'PL', 'PL2', 'DG'. Corresponding to two component, gaussian jet, powerlaw,
            alternative powerlaw and double Gaussian.
        res : int, optional
            Resolution. Default is 100.
        steps : int, optional
            Number of steps used to resolve Gamma and dm. Default is 500.
        xiN : float, optional
            Fraction of electrons that get accelerated. Default is 1.
        a1 : float, optional
            The expansion description, a1 = 0 sound speed, a1 = 1 Granot & Piran 2012. Default is 1.
        """
        self.k = float(k)
        if self.k == 0:
            self.n = float(n)
        elif self.k == 2:
            self.n = float(n * 3e35)
        else:
            self.n = float(n)

        self.epsB = float(epsb)
        self.epse = float(epse)
        self.g0 = float(g0)
        self.ek = float(ek)
        self.thc = float(thc)
        self.thj = float(thj)
        self.tho = float(tho)
        self.p = float(p)
        self.exp = bool(exp)
        self.t = time
        self.freq = freq
        self.z = float(redshift)
        self.Dl = float(Dl)
        self.method = method
        self.s = float(extra_structure_parameter_1)
        self.a = float(extra_structure_parameter_2)
        self.res = float(res)
        self.steps = int(steps)
        self.xiN = float(xiN)
        self.a1 = float(a1)
        self.is_expansion = self.exp

        # Method ID for Numba
        self.method_id = self._method_to_id(method)

    def _method_to_id(self, method):
        """Convert method string to integer for Numba"""
        method_map = {"TH": 1, "Gaussian": 2, "GJ": 2, "PL": 3, "PL-alt": 4, "PL2": 4,
                      "Two-Component": 5, "2C": 5, "Double-Gaussian": 6, "DG": 6}
        return method_map.get(method, 1)

    def get_lightcurve(self):
        if (self.k != 0) and (self.k != 2):
            raise ValueError("k must either be 0 or 2")
        if (self.p < 1.2) or (self.p > 3.4):
            raise ValueError("p is out of range, 1.2 < p < 3.4")

        # Parameters from Wijers & Galama 1999
        pxf = np.array([[1.0, 3.0, 0.41], [1.2, 1.4, 0.44], [1.4, 1.1, 0.48], [1.6, 0.86, 0.53], [1.8, 0.725, 0.56],
                        [2.0, 0.637, 0.59], [2.2, 0.579, 0.612], [2.5, 0.520, 0.630], [2.7, 0.487, 0.641],
                        [3.0, 0.451, 0.659], [3.2, 0.434, 0.660], [3.4, 0.420, 0.675]])
        xp = np.interp(self.p, pxf[:, 0], pxf[:, 1])  # dimensionless spectral peak
        Fx = np.interp(self.p, pxf[:, 0], pxf[:, 2])  # dimensionless peak flux
        nu0 = np.unique(self.freq)
        nu = nu0 * (1 + self.z)
        nu = np.array(nu)

        # Use Numba-accelerated functions
        Omi, thi, phii, rotstep, latstep = get_segments_numba(self.thj, self.res)
        Gs, Ei = get_structure_numba(self.g0, self.ek, thi, self.thc, self.method_id, self.s, self.a, self.thj)
        G, SM, ghat = get_gamma_numba(Gs, Ei, 0.0, self.steps, self.n, self.k)
        Obsa = get_obsangle_numba(phii, thi, self.tho)

        # Calculate afterglow flux using Numba
        Flux, tobs = self.calc_afterglow_numba(G, SM, self.Dl, self.p, xp, Fx,
                                               self.epsB, self.epse, Gs, Omi, Ei,
                                               self.n, self.k, self.tho, thi, phii,
                                               self.thj, ghat, rotstep, latstep, Obsa, nu,
                                               self.steps, self.xiN)

        # Calculate final lightcurve
        LC = self.calc_lightcurve(self.t, tobs, Flux, nu.size, thi.size, phii.size, self.freq, nu0)
        return LC

    def calc_afterglow_numba(self, G, SM, Dl, p, xp, Fx, EB, Ee, Gs, Omi, Ei, n, k, tho, thi, phii, thj, ghat, rotstep,
                             latstep, Obsa, nu, steps, XiN):
        """Numba-accelerated afterglow calculation"""
        Flux = np.empty((nu.size, steps, thi.size * phii.size))
        tobs = np.empty((steps, thi.size * phii.size))

        kk = 0
        for i in range(thi.size):
            beta, Ne, OmG, R, B, gm, nump, Pp, KT = calc_afterglow_step1_numba(
                G[i, :], SM[i, :], p, xp, Fx, EB, Ee, n, k, thi[i], ghat[i, :],
                rotstep, latstep, XiN, self.is_expansion, self.a1, self.res)

            for j in range(phii.size):
                FBB, Fmax, nuc, num, tobs[:, kk] = calc_afterglow_step2_numba(
                    Dl, Omi[kk], rotstep, latstep, Obsa[kk], beta, Ne, OmG, R, B, gm, nump, Pp, KT, G[i, :],
                    self.is_expansion)

                if nu.size > 1:
                    for h in range(nu.size):
                        Flux[h, :, kk] = get_ag_numba(FBB, nuc, num, nu[h], Fmax, p)
                elif nu.size == 1:
                    Flux[0, :, kk] = get_ag_numba(FBB, nuc, num, nu[0], Fmax, p)
                kk += 1

        return Flux, tobs

    def calc_lightcurve(self, time, tobs, Flux, nu_size, thi_size, phii_size, freq, nu0):
        LC = np.zeros(freq.size)
        # forward shock lightcurve at each observation time
        for h in range(nu_size):
            FF = np.zeros(len(time[(freq == nu0[h])]))
            for i in range(thi_size * phii_size):
                FF += np.interp(time[(freq == nu0[h])] / (1 + self.z), tobs[:, i], Flux[h, :, i])
            LC[(freq == nu0[h])] = FF
        return LC * (1 + self.z)

class RedbackAfterglowsRefreshed(RedbackAfterglows):
    def __init__(self, k, n, epsb, epse, g0, g1, ek, et, s1, thc, thj, tho, p, exp, time, freq, redshift, Dl,
                 extra_structure_parameter_1, extra_structure_parameter_2,
                 method='TH', res=100, steps=int(500), xiN=1, a1=1):

        """
        A general class for refreshed afterglow models implemented directly in redback.

        This class is not meant to be used directly but instead via the interface for each specific model.
        The afterglows are based on the method shown in Lamb, Mandel & Resmi 2018 and other papers.
        Script was originally written by En-Tzu Lin <entzulin@gapp.nthu.edu.tw> and Gavin Lamb <g.p.lamb@ljmu.ac.uk>
        and modified and implemented into redback by Nikhil Sarin <nsarin.astro@gmail.com>.
        Includes wind-like mediums, expansion and multiple jet structures.
        Includes SSA and uses Numba for acceleration (or numpy if Numba is not installed).

        Parameters
        ----------
        k : int
            0 or 2 for constant or wind density.
        n : float
            ISM, ambient number density.
        epsb : float
            Magnetic fraction.
        epse : float
            Electron fraction.
        g0 : float
            Initial Lorentz factor.
        g1 : float
            Lorentz factor of shell at start of energy injection.
        ek : float
            Kinetic energy.
        et : float
            Factor by which total kinetic energy is larger.
        s1 : float
            Index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
            Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020.
        thc : float
            Core angle.
        thj : float
            Jet outer angle. For tophat jets thc=thj.
        tho : float
            Observer's viewing angle.
        p : float
            Electron power-law index.
        exp : bool
            Boolean for whether to include sound speed expansion.
        time : np.ndarray
            Lightcurve time steps.
        freq : np.ndarray
            Lightcurve frequencies.
        redshift : float
            Source redshift.
        Dl : float
            Luminosity distance.
        extra_structure_parameter_1 : float
            Extra structure specific parameter #1.
            Specifically, this parameter sets the index on energy for power-law jets,
            the fractional energy contribution for the Double Gaussian (must be less than 1),
            or the energy fraction for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        extra_structure_parameter_2 : float
            Extra structure specific parameter #2.
            Specifically, this parameter sets the index on Lorentz factor for power-law jets,
            the Lorentz factor for second Gaussian (must be less than 1),
            or the Lorentz factor for the outer sheath for two-component jets (must be less than 1).
            Unused for tophat or Gaussian jets.
        method : str, optional
            Type of jet structure to use. Defaults to 'TH' for tophat jet.
            Other options are '2C', 'GJ', 'PL', 'PL2', 'DG'. Corresponding to two component, gaussian jet, powerlaw,
            alternative powerlaw and double Gaussian.
        res : int, optional
            Resolution. Default is 100.
        steps : int, optional
            Number of steps used to resolve Gamma and dm. Default is 500.
        xiN : float, optional
            Fraction of electrons that get accelerated. Default is 1.
        a1 : float, optional
            The expansion description, a1 = 0 sound speed, a1 = 1 Granot & Piran 2012. Default is 1.
        """

        super().__init__(k=k, n=n, epsb=epsb, epse=epse, g0=g0, ek=ek, thc=thc, thj=thj,
                         tho=tho, p=p, exp=exp, time=time, freq=freq, redshift=redshift,
                         Dl=Dl, extra_structure_parameter_1=extra_structure_parameter_1,
                         extra_structure_parameter_2=extra_structure_parameter_2, method=method,
                         res=res, steps=steps, xiN=xiN, a1=a1)
        self.G1 = float(g1)
        self.Et = float(et)
        self.s1 = float(s1)

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

        # Use Numba-accelerated functions from parent class
        Omi, thi, phii, rotstep, latstep = get_segments_numba(self.thj, self.res)
        Gs, Ei = get_structure_numba(self.g0, self.ek, thi, self.thc, self.method_id, self.s, self.a, self.thj)

        # Use Numba-accelerated refreshed gamma calculation
        G = np.empty((thi.size, self.steps))
        SM = np.empty((thi.size, self.steps))
        ghat = np.empty((thi.size, self.steps))

        for i in range(thi.size):
            E2 = self.Et * Ei[i]
            Gg, dM, gh = get_gamma_refreshed_numba(Gs[i], self.G1, Ei[i], E2 * Ei[i] / self.ek,
                                                   self.s1, 0.0, self.steps, self.n, self.k)
            G[i, :] = Gg
            SM[i, :] = dM
            ghat[i, :] = gh

        Obsa = get_obsangle_numba(phii, thi, self.tho)

        # Calculate afterglow flux using Numba from parent class
        Flux, tobs = self.calc_afterglow_numba(G, SM, self.Dl, self.p, xp, Fx,
                                               self.epsB, self.epse, Gs, Omi, Ei,
                                               self.n, self.k, self.tho, thi, phii,
                                               self.thj, ghat, rotstep, latstep, Obsa, nu,
                                               self.steps, self.xiN)

        # Calculate final lightcurve
        LC = self.calc_lightcurve(time=self.t, tobs=tobs, Flux=Flux,
                                  nu_size=nu.size, thi_size=thi.size, phii_size=phii.size,
                                  freq=self.freq, nu0=nu0)
        return LC

def _pnu_synchrotron(nu, B, gamma_m, gamma_c, Ne, p):
    """Parameters
----------
nu : float
    frequency in Hz
B : float
    magnetic field in G
gamma_m : float
    minimum Lorentz factor of electrons
gamma_c : float
    electron Lorentz factor at which the cooling is important
Ne : float
    Number of emitting electrons
p : float
    power law index of the electron energy distribution

Returns
-------
float or np.ndarray
    Pnu
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
    """Calculates blast-wave hydrodynamics. Based on Pe'er (2012) with a numerical correction
    factor to ensure asymptotic convergence to Sedov-Taylor solution (see also Nava et al. 2013; Huang et al. 1999)

Parameters
----------
n0 : float
    ISM density in cm^-3
Eej : float
    Ejecta energy in erg
Mej : float
    ejecta mass in g

Returns
-------
float or np.ndarray
    Dynamical outputs - t, R, beta, Gamma, eden, tobs, beta_sh, Gamma_sh.
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
    A tophat model implemented directly in redback.

    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet opening angle in radians.
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def gaussian_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A Gaussian structure afterglow model implemented directly in redback.

    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default,
            but can be set manually to a specific number.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def twocomponent_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    A two component model implemented directly in redback.

    Tophat till thc and then second component till thj.
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - ss : float
            Fraction of energy in the outer sheath of the jet. Defaults to 0.01.
        - aa : float
            Lorentz factor outside the core. Defaults to 4.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
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

    Tophat with powerlaw energy proportional to theta^ss and Lorentz factor proportional to theta^aa outside core.
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - ss : float
            Index of energy outside core. Defaults to 3.
        - aa : float
            Index of Lorentz factor outside the core. Defaults to -3.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract')
def alternativepowerlaw_redback(time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """
    An alternative powerlaw structured jet implemented directly in redback.

    Profile follows (theta/thc^2)^0.5^(-s or -a).
    Based on Lamb, Mandel & Resmi 2018 and other work.
    Look at the RedbackAfterglow class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - ss : float
            Index of energy outside core. Defaults to 3.
        - aa : float
            Index of Lorentz factor outside the core. Defaults to 3.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
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

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - ss : float
            Fractional contribution of energy to second Gaussian. Defaults to 0.1, must be less than 1.
        - aa : float
            Lorentz factor for second Gaussian, must be less than 1.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
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
    A Refreshed tophat model implemented directly in redback.

    Based on Lamb et al. 2019. Look at the RedbackAfterglowRefreshed class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet opening angle in radians.
    g1 : float
        Lorentz factor of shell at start of energy injection. 2 <= g1 < g0.
    et : float
        Factor by which total kinetic energy is larger.
    s1 : float
        Index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020.
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
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
    A Refreshed Gaussian structured jet model implemented directly in redback.

    Based on Lamb et al. 2019. Look at the RedbackAfterglowRefreshed class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    g1 : float
        Lorentz factor of shell at start of energy injection. 2 <= g1 < g0.
    et : float
        Factor by which total kinetic energy is larger.
    s1 : float
        Index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020.
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
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
    A refreshed two component model implemented directly in redback.

    Tophat till thc and then second component till thj.
    Based on Lamb et al. 2019 and other work.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

    Parameters
    ----------
    time : np.ndarray
        Time in days.
    redshift : float
        Source redshift.
    thv : float
        Observer viewing angle in radians.
    loge0 : float
        Jet energy in log10 ergs.
    thc : float
        Jet core size in radians.
    thj : float
        Jet edge in radians (thc < thj < pi/2).
    g1 : float
        Lorentz factor of shell at start of energy injection. 2 <= g1 < g0.
    et : float
        Factor by which total kinetic energy is larger.
    s1 : float
        Index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
        Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020.
    logn0 : float
        ISM number density in log10 cm^-3 or log10 A* for wind-like density profile.
    p : float
        Electron power law index.
    logepse : float
        Partition fraction in electrons (log10).
    logepsb : float
        Partition fraction in magnetic field (log10).
    g0 : float
        Initial Lorentz factor.
    xiN : float
        Fraction of electrons that get accelerated. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments:

        - frequency : np.ndarray or float
            Frequency in Hz for the flux density calculation (required).
        - output_format : str
            'flux_density' or 'magnitude'.
        - res : int
            Resolution - set dynamically based on afterglow properties by default.
        - steps : int
            Number of steps used to resolve Gamma and dm. Defaults to 250.
        - k : int
            Power law index of density profile. Defaults to 0 for constant density.
            Can be set to 2 for wind-like density profile.
        - expansion : int
            0 or 1 to dictate whether to include expansion effects. Defaults to 1.
        - ss : float
            Fraction of energy in the outer sheath of the jet. Defaults to 0.01.
        - aa : float
            Lorentz factor outside the core. Defaults to 4.
        - cosmology : astropy.cosmology
            Cosmology to use for luminosity distance calculation. Defaults to Planck18.

    Returns
    -------
    float or np.ndarray
        Flux density in mJy or AB magnitude based on output_format.

    Notes
    -----
    This gives the monochromatic magnitude at the effective frequency for the band.
    For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                          extra_structure_parameter_2=aa, res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def powerlaw_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                               logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """A Classic refreshed powerlaw structured jet implemented directly in redback.
    Tophat with powerlaw energy proportional to theta^ss and lorentz factor proportional to theta^aa outside core.
    Based on Lamb et al. 2019 and other work.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

Parameters
----------
time : np.ndarray
    time in days
redshift : float
    source redshift
thv : float
    observer viewing angle in radians
loge0 : float
    jet energy in \log_{10} ergs
thc : float
    jet core size in radians
thj : float
    jet edge in radians (thc < thj < pi/2)
g1 : float
    Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
et : float
    factor by which total kinetic energy is larger
s1 : float
    index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
    Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
logn0 : float
    ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
p : float
    electron power law index
logepse : float
    partition fraction in electrons
logepsb : float
    partition fraction in magnetic field
g0 : float
    initial lorentz factor
xiN : float
    fraction of electrons that get accelerated. Defaults to 1.
**kwargs : dict
    Additional keyword arguments:

    - res : type
        resolution - set dynamically based on afterglow properties by default,
        but can be set manually to a specific number.
    - steps : type
        number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    - k : type
        power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    - expansion : type
        0 or 1 to dictate whether to include expansion effects. Defaults to 1
    - ss : type
        Index of energy outside core. Defaults to -3
    - aa : type
        Index of Lorentz factor outside the core. Defaults to -3
    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def alternativepowerlaw_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                                          logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """An alternative refreshed powerlaw structured jet implemented directly in redback. Profile follows (theta/thc^2)^0.5^(-s or -a).
    Based on Lamb et al. 2019.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

Parameters
----------
time : np.ndarray
    time in days
redshift : float
    source redshift
thv : float
    observer viewing angle in radians
loge0 : float
    jet energy in \log_{10} ergs
thc : float
    jet core size in radians
thj : float
    jet edge in radians (thc < thj < pi/2)
g1 : float
    Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
et : float
    factor by which total kinetic energy is larger
s1 : float
    index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
    Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
logn0 : float
    ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
p : float
    electron power law index
logepse : float
    partition fraction in electrons
logepsb : float
    partition fraction in magnetic field
g0 : float
    initial lorentz factor
xiN : float
    fraction of electrons that get accelerated. Defaults to 1.
**kwargs : dict
    Additional keyword arguments:

    - res : type
        resolution - set dynamically based on afterglow properties by default,
        but can be set manually to a specific number.
    - steps : type
        number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    - k : type
        power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    - expansion : type
        0 or 1 to dictate whether to include expansion effects. Defaults to 1
    - ss : type
        Index of energy outside core. Defaults to 3
    - aa : type
        Index of Lorentz factor outside the core. Defaults to 3
    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('redback, https://ui.adsabs.harvard.edu/abs/2019ApJ...883...48L/abstract')
def doublegaussian_redback_refreshed(time, redshift, thv, loge0, thc, thj, g1, et, s1,
                                     logn0, p, logepse, logepsb, g0, xiN, **kwargs):
    """Double Gaussian structured, refreshed jet implemented directly in redback.
    Based on Lamb et al. 2019 and other work.
    Look at the RedbackAfterglowRefreshed class for more details/implementation.

Parameters
----------
time : np.ndarray
    time in days
redshift : float
    source redshift
thv : float
    observer viewing angle in radians
loge0 : float
    jet energy in \log_{10} ergs
thc : float
    jet core size in radians
thj : float
    jet edge in radians (thc < thj < pi/2)
g1 : float
    Lorentz factor of shell at start of energy injection. 2 <= g1 < g0
et : float
    factor by which total kinetic energy is larger
s1 : float
    index for energy injection; typically between 0--10, some higher values, ~<30, are supported for some structures.
    Values of ~10 are consistent with a discrete shock interaction, see Lamb, Levan & Tanvir 2020
logn0 : float
    ism number density in \log_{10} cm^-3 or \log_{10} A* for wind-like density profile
p : float
    electron power law index
logepse : float
    partition fraction in electrons
logepsb : float
    partition fraction in magnetic field
g0 : float
    initial lorentz factor
xiN : float
    fraction of electrons that get accelerated. Defaults to 1.
**kwargs : dict
    Additional keyword arguments:

    - res : type
        resolution - set dynamically based on afterglow properties by default,
        but can be set manually to a specific number.
    - steps : type
        number of steps used to resolve Gamma and dm. Defaults to 250 but can be set manually.
    - k : type
        power law index of density profile. Defaults to 0 for constant density.
        Can be set to 2 for wind-like density profile.
    - expansion : type
        0 or 1 to dictate whether to include expansion effects. Defaults to 1
    - ss : type
        Fractional contribution of energy to second Gaussian. Defaults to 0.1, must be less than 1.
    - aa : type
        Lorentz factor for second Gaussian, must be less than 1.
    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
    """
    frequency = kwargs['frequency']
    if isinstance(frequency, float):
        frequency = np.ones(len(time)) * frequency
    k = kwargs.get('k', 0)
    a1 = kwargs.get('a1', 1)
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
                                 res=res, xiN=xiN, steps=steps, a1=a1)
    flux_density = ag_class.get_lightcurve()
    fmjy = flux_density / 1e-26
    if kwargs['output_format'] == 'flux_density':
        return fmjy
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(fmjy).value

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def cocoon(time, redshift, umax, umin, loge0, k, mej, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """A cocoon afterglow model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
umax : float
    initial outflow 4 velocity maximum
umin : float
    minimum outflow 4 velocity
loge0 : float
    log10 fidicial energy in velocity distribution E(>u) = E0u^-k in erg
mej : float
    mass of material at umax in solar masses
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - k : type
        power law index of energy velocity distribution
    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A kilonova afterglow model from afterglowpy, similar to cocoon but with constraints.

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
umax : float
    initial outflow 4 velocity maximum
umin : float
    minimum outflow 4 velocity
loge0 : float
    log10 fidicial energy in velocity distribution E(>u) = E0u^-k in erg
mej : float
    mass of material at umax in solar masses
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - k : type
        power law index of energy velocity distribution
    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
    """
    output = cocoon(time=time, redshift=redshift, umax=umax, umin=umin, loge0=loge0,
                    k=k, mej=mej, logn0=logn0,p=p,logepse=logepse,logepsb=logepsb,
                    ksin=ksin, g0=g0, **kwargs)
    return output

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2020ApJ...896..166R/abstract')
def cone_afterglow(time, redshift, thv, loge0, thw, thc, logn0, p, logepse, logepsb, ksin, g0, **kwargs):
    """A cone afterglow model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thw : float
    wing truncation angle of jet thw = thw*thc
thc : float
    half width of jet core in radians
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A gaussiancore model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thw : float
    wing truncation angle of jet thw = thw*thc
thc : float
    half width of jet core in radians
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A gaussian structured jet model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thw : float
    wing truncation angle of jet thw = thw*thc
thc : float
    half width of jet core in radians
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A smoothpowerlaw structured jet model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thw : float
    wing truncation angle of jet thw = thw*thc
thc : float
    half width of jet core in radians
beta : float
    index for power-law structure, theta^-b
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A power law with core structured jet model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thw : float
    wing truncation angle of jet thw = thw*thc
thc : float
    half width of jet core in radians
beta : float
    index for power-law structure, theta^-b
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A tophat jet model from afterglowpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thc : float
    half width of jet core/jet opening angle in radians
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
ksin : float
    fraction of electrons that get accelerated
g0 : float
    initial lorentz factor
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models. assuming a monochromatic

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2025MNRAS.539.3319W/abstract')
def tophat_from_emulator(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, **kwargs):
    """Evaluate a tophat afterglow model using an mpl regressor. Note that this model predicts for a fixed redshift = 0.01 and fixed ksin = 1.
    This tophat model does not include all of the ususal kwargs

Parameters
----------
time : np.ndarray
    time in days in observer frame, should be in range 0.1 to 300
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thc : float
    half width of jet core/jet opening angle in radians
logn0 : float
    log10 number density of ISM in cm^-3
p : float
    electron distribution power law index. Must be greater than 2.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
g0 : float
    initial lorentz factor
**kwargs : dict
    Additional keyword arguments:

    - frequency : type
        frequency of the band to view in- single number or same length as time array
    - output_format : type
        Whether to output flux density or AB mag, specified by 'flux_density' or 'magnitude'

Returns
-------
float or np.ndarray
    flux density or AB mag predicted by emulator. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A base class for afterglowpy models with energy injection.

Parameters
----------
time : np.ndarray
    time in days in observer frame
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
base_model : float
    A string to indicate the type of jet model to use.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A base class for afterglow models with jet spreading. Note, with these models you cannot sample in g0.

Parameters
----------
time : np.ndarray
    time in days in observer frame
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
base_model : float
    A string to indicate the type of jet model to use.
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band. For a proper calculation of the magntitude use the sed variant models.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A base class for afterglowpy models for bandpass magnitudes/flux/spectra/sncosmo source.

Parameters
----------
time : np.ndarray
    time in days in observer frame
base_model : float
    A string to indicate the type of jet model to use.
spread : float
    whether jet can spread, defaults to False
latres : float
    latitudinal resolution for structured jets, defaults to 2
tres : float
    time resolution of shock evolution, defaults to 100
spectype : float
    whether to have inverse compton, defaults to 0, i.e., no inverse compton.
    Change to 1 for including inverse compton emission.
**kwargs : dict
    Additional keyword arguments:

    - bands : type
        Required if output_format is 'magnitude' or 'flux'.
    - output_format : type
        'magnitude', 'spectra', 'flux', 'sncosmo_source'
    - lambda_array : type
        Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.

Returns
-------
float or np.ndarray
    set by output format - 'magnitude', 'spectra', 'flux', 'sncosmo_source'
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
    spectra = fmjy.to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
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
    """A tophat jet model from jetsimpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thc : float
    half width of jet core/jet opening angle in radians
nism : float
    number density of ISM in cm^-3 (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
A : float
    wind density scale (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
p : float
    electron distribution power law index.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
g0 : float
    initial lorentz factor
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A gaussian jet model from jetsimpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thc : float
    half width of jet core/jet opening angle in radians
nism : float
    number density of ISM in cm^-3 (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
A : float
    wind density scale (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
p : float
    electron distribution power law index.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
g0 : float
    initial lorentz factor
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
    """A power-law jet model from jetsimpy

Parameters
----------
time : np.ndarray
    time in days in observer frame
redshift : float
    source redshift
thv : float
    viewing angle in radians
loge0 : float
    log10 on axis isotropic equivalent energy
thc : float
    half width of jet core/jet opening angle in radians
nism : float
    number density of ISM in cm^-3 (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
A : float
    wind density scale (ntot = A * (r / 1e17)^-2 + nism (cm^-3))
p : float
    electron distribution power law index.
logepse : float
    log10 fraction of thermal energy in electrons
logepsb : float
    log10 fraction of thermal energy in magnetic field
g0 : float
    initial lorentz factor
s : float
    power-law jet slope
**kwargs : dict
    Additional keyword arguments:

    - output_format : type
        Whether to output flux density or AB mag
    - frequency : type
        frequency in Hz for the flux density calculation
    - cosmology : type
        Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.

Returns
-------
float or np.ndarray
    flux density or AB mag. Note this is going to give the monochromatic magnitude at the effective frequency for the band.

Notes
-----
This gives the monochromatic magnitude at the effective frequency for the band.
For a proper calculation of the magnitude use the sed variant models.
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
