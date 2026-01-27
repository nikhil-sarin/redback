"""
Simulate high-energy transients with photon counting statistics

This example demonstrates the SimulateHighEnergyTransient class for generating
realistic X-ray/gamma-ray observations, including:
- Time-tagged events (TTE): Individual photon arrival times and energies
- Binned photon counts: Count rates in time bins and energy channels
- Energy-dependent detector response and background
- Proper Poisson statistics

We compare two built-in high-energy spectral models:
1. Band function (GRB-like prompt emission)
2. Blackbody (thermal X-ray/gamma-ray emission)

Both are plotted on the same count-spectrum panel for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from redback.simulate_transients import SimulateHighEnergyTransient
from redback.transient_models import spectral_models

# ==========================================================================
# Configuration
# ==========================================================================
redshift = 0.2

# Energy bin edges in keV (GBM-like range)
energy_edges = np.logspace(np.log10(10.0), np.log10(1500.0), 60)

# Observation time setup
time_range = (0.1, 30.0)

# Effective area: approximate energy-dependent response
def effective_area(energy_keV):
    peak_area = 120.0  # cm^2
    energy_keV = np.asarray(energy_keV)
    area = peak_area * np.exp(-((np.log10(energy_keV) - 2) ** 2) / 0.5)
    area = np.where(energy_keV < 15, 20.0, area)
    area = np.where(energy_keV > 900, 30.0, area)
    return area

# Background rate: higher at low energies
def background_rate(energy_keV):
    energy_keV = np.asarray(energy_keV)
    return 0.005 / energy_keV**0.3  # counts/s/keV

# Common time profile (Gaussian pulse)
# Keeps the model spectral-only while giving a transient light curve.
def gaussian_time_profile(times, t_peak, sigma_t):
    return np.exp(-0.5 * ((times - t_peak) / sigma_t) ** 2)

# ==========================================================================
# Built-in spectral models wrapped with a time profile
# ==========================================================================

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

    time_profile = gaussian_time_profile(times, t_peak=6.0, sigma_t=2.0)
    return spectrum_mjy * time_profile


def blackbody_transient(times, **kwargs):
    frequencies = kwargs.get('frequency')
    energies_keV = frequencies / keV_to_Hz

    spectrum_mjy = spectral_models.blackbody_high_energy(
        energies_keV=energies_keV,
        redshift=redshift,
        r_photosphere_rs=0.5,
        kT=5,
    )

    time_profile = gaussian_time_profile(times, t_peak=6.0, sigma_t=2.0)
    return spectrum_mjy * time_profile

# ==========================================================================
# Simulators
# ==========================================================================

sim_band = SimulateHighEnergyTransient(
    model=band_function_transient,
    parameters={},
    energy_edges=energy_edges,
    time_range=time_range,
    effective_area=effective_area,
    background_rate=background_rate,
    time_resolution=0.01,
    seed=42,
)

sim_blackbody = SimulateHighEnergyTransient(
    model=blackbody_transient,
    parameters={},
    energy_edges=energy_edges,
    time_range=time_range,
    effective_area=effective_area,
    background_rate=background_rate,
    time_resolution=0.01,
    seed=7,
)

# ==========================================================================
# Generate binned counts and TTE
# ==========================================================================

# Log-spaced bins for prompt emission
log_time_bins = np.logspace(np.log10(time_range[0]), np.log10(time_range[1]), 40)

band_lc = sim_band.generate_binned_counts(time_bins=log_time_bins, energy_integrated=True)
blackbody_lc = sim_blackbody.generate_binned_counts(time_bins=log_time_bins, energy_integrated=True)

# Count spectrum in a short window around the pulse peak
peak_time = 6.0
dt_spectrum = 1.0
spectrum_bins = np.array([peak_time - 0.5 * dt_spectrum, peak_time + 0.5 * dt_spectrum])

band_spectrum = sim_band.generate_binned_counts(time_bins=spectrum_bins, energy_integrated=False)
blackbody_spectrum = sim_blackbody.generate_binned_counts(time_bins=spectrum_bins, energy_integrated=False)

# Generate TTE for Band function example
tte_band = sim_band.generate_time_tagged_events(max_events=150000)

print(f"Band TTE events: {len(tte_band)} (source={np.sum(~tte_band['is_background'])}, "
      f"background={np.sum(tte_band['is_background'])})")

# Quick checks: expected source/background counts in the spectrum window
energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
bin_widths = energy_edges[1:] - energy_edges[:-1]
eff_area = effective_area(energy_centers)
bkg_rate = background_rate(energy_centers)

expected_bkg_counts = np.sum(bkg_rate * eff_area * bin_widths * dt_spectrum)

band_flux = sim_band._evaluate_model_flux(np.full_like(energy_centers, peak_time), energy_centers)
blackbody_flux = sim_blackbody._evaluate_model_flux(np.full_like(energy_centers, peak_time), energy_centers)
expected_band_counts = np.sum(band_flux * eff_area * bin_widths * dt_spectrum)
expected_blackbody_counts = np.sum(blackbody_flux * eff_area * bin_widths * dt_spectrum)

print(f"Expected source counts (1s window at peak): Band={expected_band_counts:.1f}, "
      f"Blackbody={expected_blackbody_counts:.1f}")
print(f"Expected background-only counts (1s window at peak): {expected_bkg_counts:.1f}")
print(f"Expected total counts (1s window at peak): Band={expected_band_counts + expected_bkg_counts:.1f}, "
      f"Blackbody={expected_blackbody_counts + expected_bkg_counts:.1f}")

# ==========================================================================
# Plotting
# ==========================================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 10))

# Panel 1: Energy-integrated light curves
ax1 = axes[0]
ax1.errorbar(
    band_lc['time_center'],
    band_lc['count_rate'],
    yerr=band_lc['count_rate_error'],
    fmt='o',
    markersize=4,
    alpha=0.7,
    label='Band function (counts/s)'
)
ax1.errorbar(
    blackbody_lc['time_center'],
    blackbody_lc['count_rate'],
    yerr=blackbody_lc['count_rate_error'],
    fmt='s',
    markersize=4,
    alpha=0.7,
    label='Blackbody (counts/s)'
)
ax1.set_xscale('log')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Count rate (counts/s)')
ax1.set_title('Energy-integrated light curves')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')
ax1.legend()

# Panel 2: Count spectrum comparison
ax2 = axes[1]

band_spec = band_spectrum.groupby('energy_channel').agg({
    'counts': 'sum',
    'energy_center': 'first',
    'energy_low': 'first',
    'energy_high': 'first',
}).reset_index()
blackbody_spec = blackbody_spectrum.groupby('energy_channel').agg({
    'counts': 'sum',
    'energy_center': 'first',
    'energy_low': 'first',
    'energy_high': 'first',
}).reset_index()

band_spec['bin_width'] = band_spec['energy_high'] - band_spec['energy_low']
blackbody_spec['bin_width'] = blackbody_spec['energy_high'] - blackbody_spec['energy_low']

band_spec['rate_density'] = band_spec['counts'] / dt_spectrum / band_spec['bin_width']
blackbody_spec['rate_density'] = blackbody_spec['counts'] / dt_spectrum / blackbody_spec['bin_width']

background_rate_density = bkg_rate * eff_area

ax2.step(
    band_spec['energy_center'],
    band_spec['rate_density'],
    where='mid',
    linewidth=2,
    label='Band total (source+background)'
)
ax2.step(
    blackbody_spec['energy_center'],
    blackbody_spec['rate_density'],
    where='mid',
    linewidth=2,
    label='Blackbody total (source+background)'
)
ax2.step(
    energy_centers,
    background_rate_density,
    where='mid',
    linewidth=1.5,
    color='0.4',
    linestyle='--',
    label='Background only'
)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Energy (keV)')
ax2.set_ylabel('Counts/s/keV')
ax2.set_title('Count spectrum comparison (totals vs background near peak)')
ax2.grid(True, alpha=0.3, which='both', linestyle='--')
ax2.legend()

plt.tight_layout()
plt.savefig('gamma_ray_simulation_results.png', dpi=200)
print("Saved plot to gamma_ray_simulation_results.png")
