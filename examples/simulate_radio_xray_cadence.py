"""
Example: Radio/X-ray Observations with SimulateTransientWithCadence

This demonstrates the NEW capability to simulate radio and X-ray observations
with realistic cadences and SNR-based detection cuts using SimulateTransientWithCadence.

Perfect for:
- Radio/X-ray follow-up campaigns
- Multi-frequency monitoring strategies
- Survey planning for radio/X-ray facilities
- Testing detection thresholds
"""

import numpy as np
import pandas as pd
from redback.simulate_transients import PopulationSynthesizer, SimulateTransientWithCadence
import redback

print("="*70)
print("RADIO/X-RAY OBSERVATIONS WITH CADENCE & SNR CUTS")
print("="*70)

# ============================================================================
# Example 1: GRB Afterglow Radio Monitoring with Cadence
# ============================================================================
print("\n" + "="*70)
print("Example 1: GRB Afterglow Radio Monitoring Campaign")
print("="*70)

# GRB parameters
grb_params = {
    'redshift': 0.5,
    'thv': 0.2,
    'loge0': 53.0,
    'thc': 0.1,
    'logn0': 0.0,
    'p': 2.2,
    'logepse': -1.0,
    'logepsb': -2.0,
    'g0': 1000,
    't0_mjd_transient': 60000.0
}

# Radio cadence config: VLA monitoring at 1.4, 5, 15 GHz
radio_cadence = {
    'frequencies': [1.4e9, 5e9, 15e9],  # Hz
    'cadence_days': 7,  # Weekly observations
    'duration_days': 200,  # Monitor for 200 days
    'sensitivity': {
        1.4e9: 0.05,  # 50 μJy at L-band
        5e9: 0.03,    # 30 μJy at C-band
        15e9: 0.04    # 40 μJy at Ku-band
    }
}

print("\nRadio monitoring configuration:")
print(f"  Frequencies: 1.4, 5, 15 GHz")
print(f"  Cadence: {radio_cadence['cadence_days']} days")
print(f"  Duration: {radio_cadence['duration_days']} days")
print(f"  Sensitivities: {list(radio_cadence['sensitivity'].values())} mJy")

# Simulate radio observations
sim_grb_radio = SimulateTransientWithCadence(
    model='afterglow',
    parameters=grb_params,
    cadence_config=radio_cadence,
    snr_threshold=5,
    noise_type='sensitivity',
    observation_mode='radio',
    seed=42
)

print(f"\nGenerated {len(sim_grb_radio.observations)} radio observations")
print(f"Detected {len(sim_grb_radio.detected_observations)} (SNR >= 5)")

# Show sample observations
print(f"\nSample observations:")
print(sim_grb_radio.observations[['time_since_t0', 'frequency', 'model_flux_density',
                                    'flux_density', 'snr', 'detected']].head(10))

# Detection statistics per frequency
for freq in radio_cadence['frequencies']:
    freq_obs = sim_grb_radio.observations[sim_grb_radio.observations['frequency'] == freq]
    n_det = (freq_obs['detected'] == True).sum()
    print(f"  {freq/1e9:.1f} GHz: {n_det}/{len(freq_obs)} detections")


# ============================================================================
# Example 2: Kilonova Radio Follow-up with Variable Cadence
# ============================================================================
print("\n" + "="*70)
print("Example 2: Kilonova Radio - Dense Early, Sparse Late")
print("="*70)

# Generate kilonova with PopulationSynthesizer
synth = PopulationSynthesizer(
    model='kilonova_afterglow',
    rate=1e-6,
    seed=123
)

kn_params_df = synth.generate_population(n_events=1, z_max=0.1)
kn_params = kn_params_df.iloc[0].to_dict()

print(f"\nKilonova at z={kn_params['redshift']:.4f}")

# Variable cadence: frequent early, sparse late
kn_radio_cadence = {
    'frequencies': [6e9, 22e9],  # 6 GHz, 22 GHz
    'cadence_days': {
        6e9: 3,   # C-band every 3 days
        22e9: 5   # K-band every 5 days
    },
    'duration_days': 100,
    'sensitivity': 0.02  # 20 μJy for all frequencies
}

sim_kn_radio = SimulateTransientWithCadence(
    model='kilonova_afterglow',
    parameters=kn_params,
    cadence_config=kn_radio_cadence,
    snr_threshold=5,
    noise_type='sensitivity',
    observation_mode='radio',
    seed=456
)

print(f"\nTotal observations: {len(sim_kn_radio.observations)}")
print(f"Detections: {len(sim_kn_radio.detected_observations)}")


# ============================================================================
# Example 3: X-ray Monitoring of TDE
# ============================================================================
print("\n" + "="*70)
print("Example 3: X-ray Monitoring of Tidal Disruption Event")
print("="*70)

# TDE parameters
tde_params = {
    'redshift': 0.03,
    't0_mjd_transient': 60100.0
}

# X-ray cadence: dense early, sparse late
# Convert keV to Hz: E(keV) × 2.417989e17 = ν(Hz)
keV_to_Hz = 2.417989e17

xray_cadence = {
    'frequencies': [1.0 * keV_to_Hz, 5.0 * keV_to_Hz],  # Soft and hard X-ray
    'cadence_days': {
        1.0 * keV_to_Hz: 2,  # Soft band every 2 days
        5.0 * keV_to_Hz: 3   # Hard band every 3 days
    },
    'duration_days': 150,
    'start_offset_days': 0.1,  # Start 0.1 days after discovery
    'sensitivity': 1e-14  # erg/cm^2/s (typical Swift sensitivity)
}

print("\nX-ray monitoring:")
print(f"  Soft band (~1 keV): every 2 days")
print(f"  Hard band (~5 keV): every 3 days")
print(f"  Duration: 150 days")

sim_tde_xray = SimulateTransientWithCadence(
    model='tde_analytical',
    parameters=tde_params,
    cadence_config=xray_cadence,
    snr_threshold=3,  # Lower threshold for X-ray
    noise_type='sensitivity',
    observation_mode='xray',
    seed=789
)

print(f"\nTotal observations: {len(sim_tde_xray.observations)}")
print(f"Detections (SNR >= 3): {len(sim_tde_xray.detected_observations)}")


# ============================================================================
# Example 4: Multi-Frequency Radio with Alternating Pattern
# ============================================================================
print("\n" + "="*70)
print("Example 4: Alternating Frequency Pattern")
print("="*70)

# Observe in rotating pattern: 1.4 → 5 → 15 GHz
rotating_cadence = {
    'frequencies': [1.4e9, 5e9, 15e9],
    'cadence_days': 2,  # Base cadence
    'duration_days': 60,
    'frequency_sequence': [1.4e9, 5e9, 15e9],  # Rotate through frequencies
    'sensitivity': 0.05  # Same for all
}

sim_rotating = SimulateTransientWithCadence(
    model='afterglow',
    parameters=grb_params,
    cadence_config=rotating_cadence,
    snr_threshold=5,
    observation_mode='radio',
    seed=999
)

print(f"\nRotating frequency pattern (every 2 days):")
print(sim_rotating.observations[['time_since_t0', 'frequency', 'detected']].head(12))


# ============================================================================
# Example 5: Population of Radio Transients
# ============================================================================
print("\n" + "="*70)
print("Example 5: Population of GRB Radio Afterglows")
print("="*70)

# Generate population
grb_synth = PopulationSynthesizer(
    model='afterglow',
    rate=1e-7,
    seed=111
)

grb_pop = grb_synth.generate_population(n_events=10, z_max=2.0)

print(f"\nSimulating {len(grb_pop)} GRBs with radio follow-up")

# Simple radio monitoring
simple_radio_cadence = {
    'frequencies': [5e9],  # Just 5 GHz
    'cadence_days': 7,
    'duration_days': 100,
    'sensitivity': 0.05
}

detection_count = 0
for idx in range(len(grb_pop)):
    grb = grb_pop.iloc[idx].to_dict()

    sim = SimulateTransientWithCadence(
        model='afterglow',
        parameters=grb,
        cadence_config=simple_radio_cadence,
        snr_threshold=5,
        observation_mode='radio',
        seed=111 + idx
    )

    n_det = len(sim.detected_observations)
    if n_det > 0:
        detection_count += 1
        print(f"  GRB {idx+1} (z={grb['redshift']:.2f}): {n_det} detections")

print(f"\nDetection rate: {detection_count}/{len(grb_pop)} = {100*detection_count/len(grb_pop):.1f}%")


# ============================================================================
# Example 6: Saving Radio/X-ray Observations
# ============================================================================
print("\n" + "="*70)
print("Example 6: Saving Radio/X-ray Data")
print("="*70)

sim_grb_radio.save_transient('grb_radio_cadence')
print("\nSaved GRB radio data:")
print("  simulated/grb_radio_cadence_observations.csv")
print("  simulated/grb_radio_cadence_parameters.csv")
print("  simulated/grb_radio_cadence_cadence_config.json")

# Load and inspect
loaded = pd.read_csv('simulated/grb_radio_cadence_observations.csv')
print(f"\nLoaded {len(loaded)} observations")
print("\nColumns:", list(loaded.columns))


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Radio/X-ray with SimulateTransientWithCadence")
print("="*70)
print("""
SimulateTransientWithCadence now supports radio and X-ray!

Key Features:
✓ Realistic cadences (uniform, per-frequency, alternating)
✓ SNR-based detection cuts
✓ Sensitivity-based noise modeling
✓ Works with PopulationSynthesizer
✓ Multi-wavelength campaigns

Observation Modes:
- observation_mode='optical': Bands and magnitudes
- observation_mode='radio': Frequencies and flux densities
- observation_mode='xray': Frequencies and flux/energy densities

Cadence Config for Radio/X-ray:
```python
cadence_config = {
    'frequencies': [1.4e9, 5e9, 15e9],  # Hz
    'cadence_days': 7,  # or dict for per-frequency
    'duration_days': 200,
    'sensitivity': 0.05,  # Jy (or dict for per-frequency)
    'start_offset_days': 0,  # optional
    'frequency_sequence': [...]  # optional, for alternating
}
```

Usage:
```python
sim = SimulateTransientWithCadence(
    model='afterglow',
    parameters=params,
    cadence_config=radio_cadence,
    snr_threshold=5,
    noise_type='sensitivity',  # Use 'sensitivity' for radio/X-ray
    observation_mode='radio',  # or 'xray'
)
```

Output Columns (Radio/X-ray):
- time_since_t0, time_mjd
- frequency (Hz)
- model_flux_density (Jy or erg/cm^2/s)
- flux_density (observed, with noise)
- flux_density_error
- snr
- detected (bool)
- sensitivity (survey limit)

Perfect for:
- Radio/X-ray follow-up planning
- Multi-frequency monitoring strategies
- Testing detection thresholds
- Survey optimization
- Population studies
""")

print("\n" + "="*70)
print("All radio/X-ray cadence examples completed!")
print("="*70)
