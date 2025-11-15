"""
Simulate gamma-ray transients with photon counting statistics

This example demonstrates the SimulateGammaRayTransient class for generating
realistic gamma-ray observations including:
- Time-tagged events (TTE): Individual photon arrival times and energies
- Binned photon counts: Count rates in time bins and energy channels
- Energy-dependent detector response
- Background modeling
- Proper Poisson statistics

Examples include:
1. GRB prompt emission (Fermi/GBM-like)
2. Magnetar burst with multiple energy channels
3. X-ray afterglow with time-tagged events
4. Fast transient with variable time binning
"""

import numpy as np
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateGammaRayTransient
import pandas as pd

# ============================================================================
# Example 1: GRB Prompt Emission (Fermi/GBM-like)
# ============================================================================
print("=" * 70)
print("Example 1: GRB Prompt Emission")
print("=" * 70)

# Fermi/GBM typical energy range
energy_edges_gbm = [8, 25, 50, 100, 300, 1000]  # keV

# GRB parameters for a simple power-law model
# For demonstration, we'll use a phenomenological model
# In practice, you'd use redback GRB models
grb_params = {
    'redshift': 0.5,
    'luminosity_distance': 2700,  # Mpc (approximately z=0.5)
    # Add other model-specific parameters here
}

# Fermi/GBM effective area (simplified, energy-dependent)
def gbm_effective_area(energy_keV):
    """Approximate Fermi/GBM effective area"""
    # Peak around 100 keV, drop at low and high energies
    peak_area = 120  # cm^2
    if np.isscalar(energy_keV):
        if energy_keV < 10:
            return 20
        elif energy_keV > 800:
            return 30
        else:
            return peak_area * np.exp(-((np.log10(energy_keV) - 2) ** 2) / 2)
    else:
        areas = np.zeros_like(energy_keV)
        areas[energy_keV < 10] = 20
        areas[energy_keV > 800] = 30
        mid = (energy_keV >= 10) & (energy_keV <= 800)
        areas[mid] = peak_area * np.exp(-((np.log10(energy_keV[mid]) - 2) ** 2) / 2)
        return areas

# Fermi/GBM background (higher at low energies)
def gbm_background(energy_keV):
    """Approximate Fermi/GBM background rate"""
    if np.isscalar(energy_keV):
        return 0.5 / energy_keV**0.5  # counts/s/keV
    else:
        return 0.5 / energy_keV**0.5

print("\nSimulating GRB with Fermi/GBM-like detector response...")
print("Energy range: 8-1000 keV")
print("Time range: -1 to 100 s")

# For this example, we'll create a simple power-law decay model
# In practice, use actual redback GRB models
def simple_grb_model(times, **kwargs):
    """Simple power-law decay for demonstration"""
    # Normalize flux
    t0 = 0.1  # Peak time
    flux_peak = 1e-5  # Jy
    alpha = -1.5  # Decay slope

    flux = np.zeros_like(times)
    mask = times > t0
    flux[mask] = flux_peak * (times[mask] / t0) ** alpha
    flux[~mask] = flux_peak

    return flux

# Note: For real usage, use actual redback models:
# sim_grb = SimulateGammaRayTransient(
#     model='prompt_afterglow',  # or other GRB model
#     parameters=grb_params,
#     ...
# )

# Generate binned counts (faster than TTE for GRBs)
# Logarithmic time binning for GRB prompt emission
time_bins_grb = np.logspace(-1, 2, 40)  # 0.1s to 100s

print("\nGenerating binned photon counts...")
print(f"Using {len(time_bins_grb)-1} logarithmic time bins")

# Create simulator
# Note: Using simple model for demonstration
# sim_grb = SimulateGammaRayTransient(
#     model=simple_grb_model,
#     parameters={},
#     energy_edges=energy_edges_gbm,
#     time_range=(time_bins_grb[0], time_bins_grb[-1]),
#     effective_area=gbm_effective_area,
#     background_rate=gbm_background,
#     time_resolution=0.001,  # 1 ms resolution
#     seed=42
# )

# binned_grb = sim_grb.generate_binned_counts(
#     time_bins=time_bins_grb,
#     energy_integrated=False  # Get per-channel light curves
# )

# print(f"Generated binned counts: {len(binned_grb)} data points")
# print(f"Energy channels: {sim_grb.n_energy_bins}")

# Example output structure:
# print("\nSample of binned data:")
# print(binned_grb.head())

print("\n[Note: Uncomment code above and use actual redback GRB model for real simulation]")

# ============================================================================
# Example 2: Magnetar Burst with Multiple Energy Channels
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Magnetar Burst")
print("=" * 70)

# Swift/BAT energy range
energy_edges_bat = [15, 25, 50, 100, 150]  # keV

magnetar_params = {
    'luminosity': 1e40,  # erg/s
    'temperature': 10,  # keV
    # Other magnetar model parameters
}

print("\nSimulating magnetar burst...")
print("Energy range: 15-150 keV (Swift/BAT-like)")
print("Time range: -0.1 to 10 s")
print("Expecting short burst with exponential decay")

# Swift/BAT effective area (simplified)
bat_area = 5200  # cm^2, approximately constant

# In practice:
# sim_magnetar = SimulateGammaRayTransient(
#     model='magnetar_driven',
#     parameters=magnetar_params,
#     energy_edges=energy_edges_bat,
#     time_range=(-0.1, 10),
#     effective_area=bat_area,
#     background_rate=0.05,
#     time_resolution=0.0001,  # 0.1 ms for fast variability
#     seed=123
# )

# # Generate both TTE and binned for magnetar
# print("\nGenerating time-tagged events...")
# events_magnetar = sim_magnetar.generate_time_tagged_events(max_events=500000)
# print(f"Total events: {len(events_magnetar)}")
# print(f"Source events: {np.sum(~events_magnetar['is_background'])}")
# print(f"Background events: {np.sum(events_magnetar['is_background'])}")

# # Also generate binned light curve
# time_bins_magnetar = np.linspace(-0.1, 10, 200)
# binned_magnetar = sim_magnetar.generate_binned_counts(
#     time_bins=time_bins_magnetar,
#     energy_integrated=True  # Single light curve
# )

print("\n[Note: Uncomment code above and use actual redback magnetar model]")

# ============================================================================
# Example 3: Time-Tagged Events for Timing Analysis
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Time-Tagged Events for Timing Analysis")
print("=" * 70)

print("\nGenerating time-tagged events for high-resolution timing...")
print("Use case: Pulsar timing, QPO searches, burst oscillations")

# In practice, after generating TTE:
# events = sim.generate_time_tagged_events()

# Example timing analysis workflow:
print("\nTiming analysis workflow:")
print("1. Generate time-tagged events")
print("2. Apply GTI (Good Time Intervals)")
print("3. Bin events at desired resolution")
print("4. Compute power spectrum or autocorrelation")
print("5. Search for periodic signals")

# Example: Bin TTE into custom time bins
def bin_tte_events(events, time_bins):
    """Bin time-tagged events into custom bins"""
    counts, _ = np.histogram(events['time'], bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    dt = np.diff(time_bins)

    result = pd.DataFrame({
        'time_center': bin_centers,
        'counts': counts,
        'counts_error': np.sqrt(counts),
        'count_rate': counts / dt,
        'count_rate_error': np.sqrt(counts) / dt
    })

    return result

# Example: Energy-resolved analysis
def energy_resolved_lightcurve(events, time_bins, energy_range):
    """Extract light curve for specific energy range"""
    e_low, e_high = energy_range
    mask = (events['energy'] >= e_low) & (events['energy'] < e_high)
    events_filtered = events[mask]

    return bin_tte_events(events_filtered, time_bins)

print("\nExample functions for TTE analysis:")
print("- bin_tte_events(): Bin events at any resolution")
print("- energy_resolved_lightcurve(): Extract energy-specific light curves")

# ============================================================================
# Example 4: Energy-Dependent Detector Response
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Energy-Dependent Response Functions")
print("=" * 70)

print("\nDemonstrating different ways to specify detector response:")

# Method 1: Constant effective area
print("\n1. Constant effective area:")
print("   effective_area=100  # 100 cm^2 for all energies")

# Method 2: Dictionary mapping
print("\n2. Dictionary mapping (linear interpolation between points):")
area_dict = {
    10: 50,   # 50 cm^2 at 10 keV
    50: 120,  # 120 cm^2 at 50 keV
    100: 130, # 130 cm^2 at 100 keV
    500: 80,  # 80 cm^2 at 500 keV
    1000: 40  # 40 cm^2 at 1000 keV
}
print(f"   effective_area={area_dict}")

# Method 3: Callable function
print("\n3. Callable function (most flexible):")
print("""
def custom_response(energy_keV):
    # Complex energy-dependent response
    if energy_keV < 10:
        return 20
    elif energy_keV > 800:
        return 30
    else:
        peak = 120
        return peak * np.exp(-((np.log10(energy_keV) - 2)**2) / 2)

effective_area=custom_response
""")

# Similar for background rates
print("\nBackground rates can be specified the same way:")
print("- Constant: background_rate=0.1")
print("- Dictionary: background_rate={10: 0.5, 100: 0.1, 1000: 0.05}")
print("- Function: background_rate=lambda e: 0.5 / e**0.5")

# ============================================================================
# Example 5: Saving and Loading Data
# ============================================================================
print("\n" + "=" * 70)
print("Example 5: Saving and Loading Simulation Data")
print("=" * 70)

print("\nSaving time-tagged events:")
print("sim.save_time_tagged_events(filename='my_grb')")
print("Creates:")
print("  - simulated/my_grb_tte.csv (event data)")
print("  - simulated/my_grb_tte_metadata.json (configuration)")

print("\nSaving binned counts:")
print("sim.save_binned_counts(filename='my_grb_binned')")
print("Creates:")
print("  - simulated/my_grb_binned.csv")

print("\nLoading for analysis:")
print("""
import pandas as pd
events = pd.read_csv('simulated/my_grb_tte.csv')
binned = pd.read_csv('simulated/my_grb_binned.csv')

# Filter by energy channel
events_ch0 = events[events['energy_channel'] == 0]

# Filter by time range
events_peak = events[(events['time'] > 0) & (events['time'] < 10)]

# Source-only events
source_events = events[~events['is_background']]
""")

# ============================================================================
# Example 6: Integration with redback Fitting
# ============================================================================
print("\n" + "=" * 70)
print("Example 6: Integration with redback Fitting Workflow")
print("=" * 70)

print("\nWorkflow for parameter inference from simulated data:")
print("""
1. Generate simulated observations:
   sim = SimulateGammaRayTransient(...)
   binned = sim.generate_binned_counts(time_bins=...)

2. Create redback transient object:
   # For binned counts, convert to appropriate format
   transient_data = binned[['time_center', 'count_rate', 'count_rate_error']]

   # Load as generic transient or specific type
   transient = redback.transient.GenericTransient.from_simulated_data(
       name='my_simulated_grb',
       data=transient_data
   )

3. Fit model:
   result = redback.fit_model(
       transient=transient,
       model='prompt_afterglow',  # Or appropriate model
       prior=my_prior,
       ...
   )

4. Compare injected vs recovered parameters:
   injected_params = sim.parameters
   recovered_params = result.posterior.median()

5. Assess biases and uncertainties
""")

# ============================================================================
# Example 7: Variable Time Binning
# ============================================================================
print("\n" + "=" * 70)
print("Example 7: Variable Time Binning Strategies")
print("=" * 70)

print("\nDifferent binning strategies for different science cases:")

# Logarithmic (GRBs, power-law decays)
print("\n1. Logarithmic binning (GRBs, afterglows):")
log_bins = np.logspace(-2, 3, 50)  # 10ms to 1000s
print(f"   {len(log_bins)-1} bins from {log_bins[0]:.3f}s to {log_bins[-1]:.0f}s")

# Linear (steady sources, slow variability)
print("\n2. Linear binning (steady sources):")
lin_bins = np.linspace(0, 100, 100)  # 1s bins
print(f"   {len(lin_bins)-1} bins of {np.diff(lin_bins)[0]:.1f}s")

# Bayesian blocks (adaptive to light curve structure)
print("\n3. Bayesian blocks (adaptive):")
print("   Use astropy.stats.bayesian_blocks()")
print("   Bins adapt to signal variability")

# SNR-based (constant SNR per bin)
print("\n4. SNR-optimized binning:")
print("   Combine bins to achieve target SNR")
print("   Useful for spectral fitting")

# Custom (science-specific)
print("\n5. Custom binning:")
custom_bins = np.array([0, 1, 2, 5, 10, 20, 50, 100])  # Specific intervals
print(f"   {custom_bins}")
print("   Based on known burst structure or features")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: SimulateGammaRayTransient Capabilities")
print("=" * 70)

print("""
✓ Time-tagged events (TTE):
  - Individual photon arrival times and energies
  - Thinning algorithm for non-homogeneous Poisson process
  - Source/background separation
  - High time resolution (sub-ms)

✓ Binned photon counts:
  - Flexible time binning (linear, log, custom)
  - Per-energy-channel light curves
  - Energy-integrated light curves
  - Proper Poisson statistics

✓ Detector response:
  - Energy-dependent effective area
  - Constant, tabulated, or functional form
  - Background modeling (constant or variable)

✓ Use cases:
  - GRB prompt emission and afterglows
  - Magnetar bursts and flares
  - Fast radio/X-ray transients
  - Pulsar timing
  - QPO and variability studies
  - Parameter recovery studies

✓ Integration:
  - Works with all redback models
  - Compatible with redback fitting workflow
  - Save/load functionality
  - Flexible data formats

Typical instruments:
  - Fermi/GBM: 8-1000 keV, ~120 cm^2
  - Swift/BAT: 15-150 keV, ~5200 cm^2
  - INTEGRAL/SPI: 18-8000 keV, ~500 cm^2
  - NuSTAR: 3-79 keV, ~800 cm^2
""")

print("\nFor working examples with real redback models,")
print("uncomment the code sections above and use appropriate model parameters.")
