from __future__ import annotations

import numpy as np

_MJY_TO_FNU = 1e-26  # erg / s / cm^2 / Hz
_PLANCK_ERG_S = 6.62607015e-27  # erg * s
_KEV_TO_HZ = 2.417989e17  # Hz / keV


def mjy_to_fnu(mjy: np.ndarray) -> np.ndarray:
    """
    Convert mJy to F_nu in erg/s/cm^2/Hz.
    """
    return np.asarray(mjy, dtype=float) * _MJY_TO_FNU


def mjy_to_photon_flux_per_keV(mjy: np.ndarray, energy_keV: np.ndarray) -> np.ndarray:
    """
    Convert flux density (mJy) to photon flux density per keV.

    Formula:
        N_E(keV) = F_nu / (h * E_keV)
    where F_nu is in erg/s/cm^2/Hz, h in erg*s, and E_keV is energy in keV.
    """
    fnu = mjy_to_fnu(mjy)
    energy_keV = np.asarray(energy_keV, dtype=float)
    return fnu / (_PLANCK_ERG_S * energy_keV)


def mjy_to_energy_flux_per_keV(mjy: np.ndarray) -> np.ndarray:
    """
    Convert flux density (mJy) to energy flux density per keV (erg/s/cm^2/keV).
    """
    fnu = mjy_to_fnu(mjy)
    return fnu * _KEV_TO_HZ
