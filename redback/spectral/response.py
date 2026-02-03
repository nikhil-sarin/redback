from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ResponseMatrix:
    e_min: np.ndarray
    e_max: np.ndarray
    channel: np.ndarray
    emin_chan: np.ndarray
    emax_chan: np.ndarray
    matrix: np.ndarray

    @property
    def energy_centers(self) -> np.ndarray:
        return 0.5 * (self.e_min + self.e_max)

    def apply(self, photon_flux_per_keV: np.ndarray) -> np.ndarray:
        return self.matrix @ photon_flux_per_keV


@dataclass
class EffectiveArea:
    e_min: np.ndarray
    e_max: np.ndarray
    area: np.ndarray

    @property
    def energy_centers(self) -> np.ndarray:
        return 0.5 * (self.e_min + self.e_max)

    def evaluate(self, energy_keV: np.ndarray) -> np.ndarray:
        return np.interp(energy_keV, self.energy_centers, self.area, left=0.0, right=0.0)
