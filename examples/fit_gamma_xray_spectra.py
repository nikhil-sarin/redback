"""
Fit a simulated gamma/X-ray count spectrum with redback's spectral fitting API.

This mirrors examples/simulate_gamma_xray_spectra_ray.py but adds a fitting step.
"""

import numpy as np

from redback.simulate_transients import SimulateHighEnergyTransient
from redback.spectral.dataset import SpectralDataset
from redback.transient_models import spectral_models
from redback.sampler import fit_model

redshift = 0.2
energy_edges = np.logspace(np.log10(10.0), np.log10(1500.0), 60)
time_range = (0.1, 30.0)


def effective_area(energy_keV):
    peak_area = 120.0  # cm^2
    energy_keV = np.asarray(energy_keV)
    area = peak_area * np.exp(-((np.log10(energy_keV) - 2) ** 2) / 0.5)
    area = np.where(energy_keV < 15, 20.0, area)
    area = np.where(energy_keV > 900, 30.0, area)
    return area


def background_rate(energy_keV):
    energy_keV = np.asarray(energy_keV)
    return 0.005 / energy_keV**0.3  # counts/s/keV


keV_to_Hz = 2.417989e17


def band_function_transient(times, **kwargs):
    frequencies = kwargs.get('frequency')
    energies_keV = frequencies / keV_to_Hz
    spectrum_mjy = spectral_models.band_function_high_energy(
        energies_keV=energies_keV,
        redshift=redshift,
        log10_norm=-2.5,
        alpha=-1.0,
        beta=-2.3,
        e_peak=200.0,
    )
    time_profile = np.exp(-0.5 * ((times - 6.0) / 2.0) ** 2)
    return spectrum_mjy * time_profile


sim = SimulateHighEnergyTransient(
    model=band_function_transient,
    parameters={},
    energy_edges=energy_edges,
    time_range=time_range,
    effective_area=effective_area,
    background_rate=background_rate,
    time_resolution=0.01,
    seed=42,
)

# Build a spectrum in a 1s window around the peak
time_bins = np.array([5.5, 6.5])
dataset = SpectralDataset.from_simulator(sim, time_bins=time_bins)

result = fit_model(
    transient=dataset,
    model=band_function_transient,
    sampler="dynesty",
    statistic="wstat",
    nlive=500,
)

print(result)
