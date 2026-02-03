from __future__ import annotations

import numpy as np

from redback.spectral.conversions import mjy_to_photon_flux_per_keV
from redback.spectral.response import ResponseMatrix, EffectiveArea


def _integrate_photon_flux_per_bin(photon_flux_per_keV: np.ndarray, energy_edges_keV: np.ndarray) -> np.ndarray:
    """
    Integrate photon flux density over energy bins (counts/s/cm^2 per bin).
    """
    widths = energy_edges_keV[1:] - energy_edges_keV[:-1]
    return photon_flux_per_keV * widths


def fold_spectrum(
    model_flux_mjy: np.ndarray,
    energy_edges_keV: np.ndarray,
    rmf: ResponseMatrix | None = None,
    arf: EffectiveArea | None = None,
    exposure: float = 1.0,
    areascal: float = 1.0,
) -> np.ndarray:
    """
    Fold a model spectrum through ARF/RMF and exposure to produce expected counts.

    :param model_flux_mjy: Flux density (mJy) evaluated at bin centers in energy space.
    :param energy_edges_keV: Energy bin edges in keV.
    :param rmf: Optional response matrix for redistribution.
    :param arf: Optional effective area.
    :param exposure: Exposure time in seconds.
    :param areascal: Area scaling factor (dimensionless).
    :return: Expected counts per channel (if RMF) or per energy bin (if no RMF).
    """
    energy_centers = 0.5 * (energy_edges_keV[:-1] + energy_edges_keV[1:])
    photon_flux_per_keV = mjy_to_photon_flux_per_keV(model_flux_mjy, energy_centers)
    photon_flux_per_bin = _integrate_photon_flux_per_bin(photon_flux_per_keV, energy_edges_keV)

    if arf is not None:
        area = arf.evaluate(energy_centers)
        photon_flux_per_bin = photon_flux_per_bin * area

    photon_flux_per_bin = photon_flux_per_bin * exposure * areascal

    if rmf is None:
        return photon_flux_per_bin

    return rmf.apply(photon_flux_per_bin)
