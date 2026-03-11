from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import redback
from redback.spectral.dataset import SpectralDataset


@dataclass
class CountsSpectrumTransient:
    """
    High-energy spectral transient wrapper for OGIP-style data.
    """
    dataset: SpectralDataset
    name: str = "spectral_transient"

    def __post_init__(self):
        self.dataset.name = self.name
        self.data_mode = "spectrum_counts"
        self.directory_structure = redback.get_data.directory.spectrum_directory_structure(transient=self.name)

    @classmethod
    def from_ogip(
        cls,
        pha: str,
        rmf: Optional[str] = None,
        arf: Optional[str] = None,
        bkg: Optional[str] = None,
        spectrum_index: Optional[int] = None,
        name: Optional[str] = None,
        energy_edges_keV=None,
    ) -> "CountsSpectrumTransient":
        dataset = SpectralDataset.from_ogip(
            pha=pha,
            rmf=rmf,
            arf=arf,
            bkg=bkg,
            spectrum_index=spectrum_index,
            name=name or "spectral_transient",
            energy_edges_keV=energy_edges_keV,
        )
        return cls(dataset=dataset, name=dataset.name)

    @classmethod
    def from_ogip_directory(
        cls,
        directory: str,
        pha: Optional[str] = None,
        bkg: Optional[str] = None,
        rmf: Optional[str] = None,
        arf: Optional[str] = None,
        spectrum_index: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "CountsSpectrumTransient":
        dataset = SpectralDataset.from_ogip_directory(
            directory=directory,
            pha=pha,
            bkg=bkg,
            rmf=rmf,
            arf=arf,
            spectrum_index=spectrum_index,
            name=name or "spectral_transient",
        )
        return cls(dataset=dataset, name=dataset.name)

    @classmethod
    def from_simulator(cls, sim, time_bins, name: Optional[str] = None) -> "CountsSpectrumTransient":
        dataset = SpectralDataset.from_simulator(sim=sim, time_bins=time_bins)
        dataset.name = name or "spectral_transient"
        return cls(dataset=dataset, name=dataset.name)

    def plot_data(self, **kwargs):
        return self.dataset.plot_spectrum_data(**kwargs)

    def plot_fit(self, **kwargs):
        return self.dataset.plot_spectrum_fit(**kwargs)

    def plot_lightcurve(self, **kwargs):
        return self.dataset.plot_lightcurve(**kwargs)

    def compute_band_flux(self, *args, **kwargs):
        return self.dataset.compute_band_flux(*args, **kwargs)
