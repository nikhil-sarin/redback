"""
Example: Using SimulateTransientWithCadence for survey planning

This demonstrates the new SimulateTransientWithCadence class that bridges
the gap between simple simulation and full survey simulation.

Perfect for:
1. Using PopulationSynthesizer parameters
2. Testing different cadence strategies
3. SNR-based detection modeling
4. Survey planning without full pointing databases
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from redback.simulate_transients import PopulationSynthesizer, SimulateTransientWithCadence
import redback

print("="*70)
print("SIMULATION WITH CADENCE AND SNR CUTS")
print("="*70)

# ============================================================================
# Example 1: Basic Usage with PopulationSynthesizer
# ============================================================================
print("\n" + "="*70)
print("Example 1: Single Transient with Simple Cadence")
print("="*70)

# Generate a single kilonova using PopulationSynthesizer
synth = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,
    seed=42
)

# Generate one event
params = synth.generate_population(n_events=1, z_max=0.02)
single_event = params.iloc[0].to_dict()

print(f"\nEvent parameters:")
print(f"  Redshift: {single_event['redshift']:.4f}")
print(f"  t0 (MJD): {single_event['t0_mjd_transient']:.1f}")
if 'mej' in single_event:
    print(f"  Ejecta mass: {single_event['mej']:.4f} M_sun")

# Define simple cadence: observe every 2 days in g,r,i
cadence_config = {
    'bands': ['lsstg', 'lsstr', 'lssti'],
    'cadence_days': 2.0,  # Every 2 days
    'duration_days': 30,   # For 30 days
    'limiting_mags': {
        'lsstg': 24.5,
        'lsstr': 24.5,
        'lssti': 24.0
    }
}

# Simulate observations
sim = SimulateTransientWithCadence(
    model='one_component_kilonova_model',
    parameters=single_event,
    cadence_config=cadence_config,
    snr_threshold=5,
    noise_type='limiting_mag',
    seed=42
)

print(f"\nGenerated {len(sim.observations)} observations")
print(f"Detected {len(sim.detected_observations)} observations (SNR >= 5)")

print(f"\nFirst few observations:")
print(sim.observations[['time_since_t0', 'band', 'magnitude', 'snr', 'detected']].head(10))

# Quick plot for the first transient
print("\nSaving example light curve plot...")
fig, ax = plt.subplots(figsize=(7, 4))
for band in cadence_config['bands']:
    band_data = sim.observations[sim.observations['band'] == band]
    ax.scatter(band_data['time_since_t0'], band_data['magnitude'], label=band, alpha=0.7)
ax.invert_yaxis()
ax.set_xlabel('Time (days)')
ax.set_ylabel('AB Magnitude')
ax.set_title('Simulated Kilonova Light Curve')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cadence_simulation_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot to cadence_simulation_results.png")


# ============================================================================
# Example 2: Different Cadences Per Band
# ============================================================================
print("\n" + "="*70)
print("Example 2: Different Cadences Per Band")
print("="*70)

# lsstg: every 3 days, lsstr: every 1 day, lssti: every 5 days
cadence_per_band = {
    'bands': ['lsstg', 'lsstr', 'lssti'],
    'cadence_days': {'lsstg': 3, 'lsstr': 1, 'lssti': 5},  # Different per band
    'duration_days': 50,
    'limiting_mags': {'lsstg': 24.5, 'lsstr': 24.5, 'lssti': 24.0}
}

sim2 = SimulateTransientWithCadence(
    model='one_component_kilonova_model',
    parameters=single_event,
    cadence_config=cadence_per_band,
    snr_threshold=5,
    seed=42
)

print(f"\nTotal observations: {len(sim2.observations)}")
for band in ['lsstg', 'lsstr', 'lssti']:
    n_band = len(sim2.observations[sim2.observations['band'] == band])
    n_det = len(sim2.detected_observations[sim2.detected_observations['band'] == band])
    print(f"  {band}-band: {n_band} observations, {n_det} detected")


# ============================================================================
# Example 3: Alternating Band Sequence (Like Real Surveys)
# ============================================================================
print("\n" + "="*70)
print("Example 3: Alternating Band Sequence")
print("="*70)

# Observe in g-r-i-g-r-i pattern every night
cadence_sequence = {
    'bands': ['lsstg', 'lsstr', 'lssti'],
    'cadence_days': 1.0,  # Base cadence
    'duration_days': 20,
    'limiting_mags': {'lsstg': 24.5, 'lsstr': 24.5, 'lssti': 24.0},
    'band_sequence': ['lsstg', 'lsstr', 'lssti']  # Repeating pattern
}

sim3 = SimulateTransientWithCadence(
    model='one_component_kilonova_model',
    parameters=single_event,
    cadence_config=cadence_sequence,
    snr_threshold=5,
    seed=42
)

print(f"\nGenerated {len(sim3.observations)} observations in sequence")
print(f"\nObservation pattern (first 15):")
print(sim3.observations[['time_since_t0', 'band', 'detected']].head(15))


# ============================================================================
# Example 4: Population Simulation with Cadence
# ============================================================================
print("\n" + "="*70)
print("Example 4: Simulate Population with Cadence")
print("="*70)

# Generate 10 events
population_params = synth.generate_population(n_events=10, z_max=0.1)

print(f"\nSimulating {len(population_params)} events with cadence...")

# Simple cadence for all
simple_cadence = {
    'bands': ['lsstr'],  # Just r-band for speed
    'cadence_days': 3,
    'duration_days': 40,
    'limiting_mags': {'lsstr': 24.5}
}

n_detected_events = 0
total_detections = 0

for idx in range(len(population_params)):
    event = population_params.iloc[idx].to_dict()

    sim = SimulateTransientWithCadence(
        model='one_component_kilonova_model',
        parameters=event,
        cadence_config=simple_cadence,
        snr_threshold=5,
        seed=42 + idx
    )

    n_det = len(sim.detected_observations)
    total_detections += n_det

    if n_det > 0:
        n_detected_events += 1

print(f"\nResults:")
print(f"  Events with >=1 detection: {n_detected_events}/{len(population_params)}")
print(f"  Total detections: {total_detections}")
print(f"  Average detections per event: {total_detections/len(population_params):.1f}")


# ============================================================================
# Example 5: Testing Different SNR Thresholds
# ============================================================================
print("\n" + "="*70)
print("Example 5: Impact of SNR Threshold")
print("="*70)

test_cadence = {
    'bands': ['lsstr'],
    'cadence_days': 2,
    'duration_days': 30,
    'limiting_mags': {'lsstr': 24.5}
}

snr_thresholds = [3, 5, 7, 10]

print(f"\nDetections vs SNR threshold:")
for snr_thresh in snr_thresholds:
    sim = SimulateTransientWithCadence(
        model='one_component_kilonova_model',
        parameters=single_event,
        cadence_config=test_cadence,
        snr_threshold=snr_thresh,
        seed=42
    )

    n_det = len(sim.detected_observations)
    print(f"  SNR >= {snr_thresh}: {n_det} detections")


# ============================================================================
# Example 6: Delayed Start (Transient Found Later)
# ============================================================================
print("\n" + "="*70)
print("Example 6: Delayed Observation Start")
print("="*70)

# Start observing 5 days after explosion
delayed_cadence = {
    'bands': ['lsstg', 'lsstr', 'lssti'],
    'cadence_days': 1,
    'duration_days': 20,
    'start_offset_days': 5,  # Start 5 days after t0
    'limiting_mags': {'lsstg': 24.5, 'lsstr': 24.5, 'lssti': 24.0}
}

sim_delayed = SimulateTransientWithCadence(
    model='one_component_kilonova_model',
    parameters=single_event,
    cadence_config=delayed_cadence,
    snr_threshold=5,
    seed=42
)

print(f"\nObservations starting 5 days after explosion:")
print(f"  First observation at t = {sim_delayed.observations['time_since_t0'].min():.1f} days")
print(f"  Total detections: {len(sim_delayed.detected_observations)}")


# ============================================================================
# Example 7: Saving and Loading
# ============================================================================
print("\n" + "="*70)
print("Example 7: Saving Results")
print("="*70)

# Simulate and save
sim_to_save = SimulateTransientWithCadence(
    model='one_component_kilonova_model',
    parameters=single_event,
    cadence_config=cadence_config,
    snr_threshold=5,
    seed=42
)

sim_to_save.save_transient('example_kilonova')
print("\nSaved transient data to:")
print("  simulated/example_kilonova_observations.csv")
print("  simulated/example_kilonova_parameters.csv")
print("  simulated/example_kilonova_cadence_config.json")

# Load it back
loaded_obs = pd.read_csv('simulated/example_kilonova_observations.csv')
print(f"\nLoaded {len(loaded_obs)} observations from file")


# ============================================================================
# Example 8: Integration with Full Workflow
# ============================================================================
print("\n" + "="*70)
print("Example 8: Complete Workflow")
print("="*70)

print("""
A workflow combining all tools:

1. Generate population with PopulationSynthesizer:
   - Specify rate, rate evolution, cosmology
   - Sample parameters from priors
   - Get redshifts, sky positions, explosion times

2. Simulate observations with SimulateTransientWithCadence:
   - Specify realistic cadence
   - Apply SNR cuts
   - Get detected observations

3. Analyze results:
   - Detection efficiency vs redshift
   - Optimal cadence strategies
   - SNR distributions

4. Feed to redback analysis:
   - Use detected_observations as input data
   - Fit with redback.fit_model()
   - Infer parameters

Example code:
```python
# 1. Generate population
synth = PopulationSynthesizer(model='kilonova', rate=1e-6)
params = synth.generate_population(n_events=100)

# 2. Simulate each with cadence
detected_events = []
for idx in range(len(params)):
    sim = SimulateTransientWithCadence(
        model='kilonova',
        parameters=params.iloc[idx].to_dict(),
        cadence_config=my_survey_cadence,
        snr_threshold=5
    )
    if len(sim.detected_observations) > 0:
        detected_events.append(sim.detected_observations)

# 3. Analyze
print(f"Detection rate: {len(detected_events)/len(params):.1%}")

# 4. Fit with redback (for detected events)
for obs_df in detected_events:
    transient = redback.Transient.from_simulated_optical_data(obs_df)
    result = redback.fit_model(transient, model='kilonova', ...)
```
""")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
SimulateTransientWithCadence Features:

✓ Works with PopulationSynthesizer parameters
✓ Flexible cadence specification (per-band or alternating)
✓ SNR-based detection modeling
✓ Realistic noise from limiting magnitudes
✓ No pointing database required
✓ Easy to test different survey strategies

Key Parameters:
- cadence_days: Observation frequency (float or dict)
- limiting_mags: 5-sigma depths per band
- snr_threshold: Detection threshold
- band_sequence: Alternating band pattern (optional)
- start_offset_days: Delayed start (optional)

Output:
- observations: All scheduled observations
- detected_observations: Only SNR >= threshold
- Saves to CSV for later analysis
""")

print("\n" + "="*70)
print("All examples completed!")
print("="*70)
