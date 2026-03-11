"""
Example: Radio and X-ray Transient Simulation

This demonstrates using redback simulation tools for radio and X-ray transients,
including GRB afterglows, TDEs, and other multi-wavelength phenomena.
Keep in mind this is just an example usage; you can adapt a lot of the code below to be more specific to your needs.
Look at the documentation for more details on each class and function.

Uses:
- SimulateGenericTransient for frequency-based observations
- PopulationSynthesizer for parameter generation
- Custom cadences for radio/X-ray monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateGenericTransient, PopulationSynthesizer
import redback

print("="*70)
print("RADIO AND X-RAY TRANSIENT SIMULATION")
print("="*70)

# ============================================================================
# Example 1: GRB Afterglow - Radio Observations
# ============================================================================
print("\n" + "="*70)
print("Example 1: GRB Afterglow Radio Observations (Tophat Jet)")
print("="*70)

# GRB afterglow parameters
grb_params = {
    'redshift': 0.5,
    'thv': 0.2,  # Viewing angle
    'loge0': 53.0,  # log10(E_iso / erg)
    'thc': 0.1,  # Jet opening angle
    'logn0': 0.0,  # log10(n_ISM / cm^-3)
    'p': 2.2,  # Electron spectral index
    'logepse': -1.0,  # log10(epsilon_e)
    'logepsb': -2.0,  # log10(epsilon_B)
    'ksin': 1.0,  # Fraction of electrons accelerated
    'g0': 1000  # Initial Lorentz factor
}

# Radio frequencies (Hz)
# VLA: 1.4 GHz, 5 GHz, 15 GHz
radio_frequencies = np.array([1.4e9, 5e9, 15e9])

# Observation times (days after burst)
times = np.array([1, 3, 7, 14, 30, 60, 100, 200, 365])

# Total data points
n_points = len(times) * len(radio_frequencies)

print(f"\nObserving at {len(radio_frequencies)} frequencies")
print(f"  1.4 GHz (L-band)")
print(f"  5 GHz (C-band)")
print(f"  15 GHz (Ku-band)")
print(f"\nObservation times: {len(times)} epochs")

# Simulate using SimulateGenericTransient
sim_grb = SimulateGenericTransient(
    model='tophat',  # Tophat jet afterglow model
    parameters=grb_params,
    times=times,
    model_kwargs={'frequency': radio_frequencies, 'output_format': 'flux_density'},
    data_points=n_points,
    multiwavelength_transient=True,  # Different frequencies
    noise_term=0.1,  # 10% fractional uncertainty
    noise_type='gaussianmodel',
    seed=42
)

print(f"\nGenerated {len(sim_grb.data)} radio observations")
print(f"\nSample observations:")
print(sim_grb.data[['time', 'frequency', 'true_output', 'output', 'output_error']].head(10))

# Calculate SNR
sim_grb.data['snr'] = sim_grb.data['output'] / sim_grb.data['output_error']

# Apply 5-sigma detection threshold
detected = sim_grb.data[sim_grb.data['snr'] >= 5]
print(f"\nDetections (SNR >= 5): {len(detected)}/{len(sim_grb.data)}")


# ============================================================================
# Example 2: X-ray Monitoring of TDE
# ============================================================================
print("\n" + "="*70)
print("Example 2: X-ray Monitoring of a TDE-like Tophat Jet")
print("="*70)

# TDE-like jet parameters (using the same tophat model)
tde_params = {
    'redshift': 0.01,
    'thv': 0.3,
    'loge0': 50.0,  # Lower energy than GRB
    'thc': 0.2,
    'logn0': -1.0,
    'p': 2.2,
    'logepse': -1.2,
    'logepsb': -3.0,
    'ksin': 1.0,
    'g0': 5.0  # Mildly relativistic
}

# X-ray energy bands (keV converted to Hz)
# Soft: 0.3-2 keV, Hard: 2-10 keV
keV_to_Hz = 2.417989e17  # Conversion factor
xray_frequencies = np.array([
    1.15 * keV_to_Hz,  # 1.15 keV (mid-point of soft band)
    6.0 * keV_to_Hz    # 6 keV (mid-point of hard band)
])

# Dense monitoring early, sparse later
early_times = np.linspace(0.1, 10, 20)   # Every ~0.5 days
middle_times = np.linspace(10, 50, 20)   # Every ~2 days
late_times = np.linspace(50, 200, 15)    # Every ~10 days
xray_times = np.concatenate([early_times, middle_times, late_times])

n_xray_points = len(xray_times) * len(xray_frequencies)

print(f"\nX-ray monitoring:")
print(f"  Soft band: 0.3-2 keV")
print(f"  Hard band: 2-10 keV")
print(f"  Total epochs: {len(xray_times)}")
print(f"  Early (0-10 days): dense monitoring")
print(f"  Late (>50 days): sparse monitoring")

sim_xray = SimulateGenericTransient(
    model='tophat',
    parameters=tde_params,
    times=xray_times,
    model_kwargs={'frequency': xray_frequencies, 'output_format': 'flux_density'},
    data_points=n_xray_points,
    multiwavelength_transient=True,
    noise_term=0.02,  # 2% uncertainty
    noise_type='gaussianmodel',
    seed=123
)

print(f"\nGenerated {len(sim_xray.data)} X-ray observations")

# Analyze light curve
sim_xray.data['snr'] = sim_xray.data['output'] / sim_xray.data['output_error']
detected_xray = sim_xray.data[sim_xray.data['snr'] >= 2]  # 2-sigma threshold

print(f"Detections (SNR >= 2): {len(detected_xray)}/{len(sim_xray.data)}")


# ============================================================================
# Plotting: Radio + X-ray Light Curves
# ============================================================================
print("\n" + "="*70)
print("Plotting Radio and X-ray Light Curves")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Radio plot
ax_radio = axes[0]
for freq in radio_frequencies:
    freq_data = sim_grb.data[sim_grb.data['frequency'] == freq]
    ax_radio.errorbar(
        freq_data['time'],
        freq_data['output'],
        yerr=freq_data['output_error'],
        fmt='o',
        label=f"{freq/1e9:.1f} GHz",
        alpha=0.7
    )
ax_radio.set_xscale('log')
ax_radio.set_yscale('log')
ax_radio.set_xlabel('Time (days)')
ax_radio.set_ylabel('Flux Density (Jy)')
ax_radio.set_title('GRB Radio Afterglow (Tophat)')
ax_radio.legend()
ax_radio.grid(True, alpha=0.3)

# X-ray plot
ax_xray = axes[1]
for freq in xray_frequencies:
    freq_data = sim_xray.data[sim_xray.data['frequency'] == freq]
    ax_xray.errorbar(
        freq_data['time'],
        freq_data['output'],
        yerr=freq_data['output_error'],
        fmt='o',
        label=f"{freq/keV_to_Hz:.2f} keV",
        alpha=0.7
    )
ax_xray.set_xscale('log')
ax_xray.set_yscale('log')
ax_xray.set_xlabel('Time (days)')
ax_xray.set_ylabel('Flux Density (Jy)')
ax_xray.set_title('TDE X-ray Jet (Tophat)')
ax_xray.legend()
ax_xray.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('radio_xray_simulation_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot to radio_xray_simulation_results.png")


# ============================================================================
# Example 3: Multi-Frequency Radio Monitoring with Cadence
# ============================================================================
print("\n" + "="*70)
print("Example 3: Structured Radio Monitoring Campaign")
print("="*70)

# Generate a kilonova for radio follow-up
synth = PopulationSynthesizer(
    model='kilonova_afterglow',
    rate=1e-6,
    seed=42
)

kn_params_df = synth.generate_population(n_events=1, z_max=0.1)
kn_params = kn_params_df.iloc[0].to_dict()

print(f"\nKilonova at z={kn_params['redshift']:.4f}")

# Radio frequencies for kilonova follow-up
# VLA/ALMA: 6 GHz, 22 GHz, 100 GHz
kn_frequencies = np.array([6e9, 22e9, 100e9])

# Structured cadence: early frequent, then sparse
# Days: 1, 3, 5, 7, 10, 14, 21, 30, 50, 100
kn_times = np.array([1, 3, 5, 7, 10, 14, 21, 30, 50, 100])

n_kn_points = len(kn_times) * len(kn_frequencies)

print(f"\nRadio follow-up campaign:")
print(f"  Frequencies: 6 GHz (C-band), 22 GHz (K-band), 100 GHz (mm)")
print(f"  Epochs: {len(kn_times)} over 100 days")
print(f"  Total observations: {n_kn_points}")

sim_kn_radio = SimulateGenericTransient(
    model='kilonova_afterglow',
    parameters=kn_params,
    times=kn_times,
    model_kwargs={'frequency': kn_frequencies, 'output_format': 'flux_density'},
    data_points=n_kn_points,
    multiwavelength_transient=True,
    noise_term=0.2,
    noise_type='gaussianmodel',
    seed=456
)

print(f"\nGenerated {len(sim_kn_radio.data)} kilonova radio observations")

# Calculate detection statistics per frequency
for freq in kn_frequencies:
    freq_data = sim_kn_radio.data[sim_kn_radio.data['frequency'] == freq]
    freq_data['snr'] = freq_data['output'] / freq_data['output_error']
    n_det = np.sum(freq_data['snr'] >= 5)
    print(f"  {freq/1e9:.0f} GHz: {n_det}/{len(freq_data)} detections")


# ============================================================================
# Example 4: Population of Radio Transients
# ============================================================================
print("\n" + "="*70)
print("Example 4: Population of GRB Radio Afterglows")
print("="*70)

# Generate population of GRBs
grb_synth = PopulationSynthesizer(
    model='cone_afterglow',
    rate=1e-7,  # Lower rate for GRBs
    seed=789
)

grb_population = grb_synth.generate_population(n_events=5, z_max=1.0)

print(f"\nSimulating {len(grb_population)} GRBs in radio")

# Simple monitoring: 1.4 GHz at 1, 7, 30 days
monitoring_freq = np.array([1.4e9])
monitoring_times = np.array([1, 7, 30])

detection_count = 0
total_obs = 0

for idx in range(len(grb_population)):
    grb = grb_population.iloc[idx].to_dict()

    # Simulate radio observations
    sim = SimulateGenericTransient(
        model='cone_afterglow',
        parameters=grb,
        times=monitoring_times,
        model_kwargs={'frequency': monitoring_freq, 'output_format': 'flux_density'},
        data_points=len(monitoring_times),
        multiwavelength_transient=False,  # Single frequency
        noise_term=0.2,
        noise_type='gaussianmodel',
        seed=789 + idx
    )

    # Check for detections
    sim.data['snr'] = sim.data['output'] / sim.data['output_error']
    n_det = np.sum(sim.data['snr'] >= 5)
    total_obs += len(sim.data)

    if n_det > 0:
        detection_count += 1
        print(f"  GRB {idx+1} (z={grb['redshift']:.2f}): {n_det}/3 detections")

print(f"\nOverall: {detection_count}/{len(grb_population)} GRBs detected")
print(f"Total observations: {total_obs}")


# ============================================================================
# Example 5: Saving Radio/X-ray Data
# ============================================================================
print("\n" + "="*70)
print("Example 5: Saving Radio/X-ray Observations")
print("="*70)

# Save GRB radio data
sim_grb.save_transient('grb_radio_example')
print("\nSaved GRB radio data:")
print("  simulated/grb_radio_example.csv")
print("  simulated/grb_radio_example_injection_parameters.csv")

# The saved data can be loaded and analyzed with redback
loaded_data = pd.read_csv('simulated/grb_radio_example.csv')
print(f"\nLoaded {len(loaded_data)} observations")
print("\nColumns:", list(loaded_data.columns))


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Radio/X-ray Simulation with Redback")
print("="*70)
print("""
SimulateGenericTransient supports frequency-based observations:

Key Features:
✓ Works with any frequency (radio, mm, X-ray)
✓ Multi-frequency observations (multiwavelength_transient=True)
✓ Flexible cadences via time array
✓ Multiple noise models (gaussian, gaussianmodel, SNRbased)
✓ Compatible with PopulationSynthesizer

Usage Pattern:
```python
sim = SimulateGenericTransient(
    model='cone_afterglow',
    parameters=params,
    times=observation_times,
    model_kwargs={
        'frequency': frequencies_in_Hz,
        'output_format': 'flux_density'
    },
    data_points=n_observations,
    multiwavelength_transient=True,  # If multiple frequencies
    noise_type='gaussianmodel'  # or 'gaussian', 'SNRbased'
)
```

Output:
- time: Observation time
- frequency: Observation frequency (Hz)
- true_output: Model flux density (Jy)
- output: Observed flux density with noise
- output_error: Uncertainty

Integration with Population:
1. Generate parameters with PopulationSynthesizer
2. Loop over population
3. Simulate radio/X-ray for each event
4. Analyze detection statistics
""")

print("\n" + "="*70)
print("All radio/X-ray examples completed!")
print("="*70)
