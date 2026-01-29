"""
Example script demonstrating the modular PopulationSynthesizer workflow

This shows how the refactored PopulationSynthesizer provides:
1. Pure parameter generation (generate_population)
2. Flexible post-processing (apply_detection_criteria)
3. Rate inference (infer_rate)
4. Integration with redback simulation tools

The key advantage is separation of concerns:
- Generate parameters → use with ANY simulation tool
- Post-process separately with custom criteria
- Infer rates independently
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from redback.simulate_transients import PopulationSynthesizer, TransientPopulation
import redback

print("="*70)
print("MODULAR POPULATION SYNTHESIS WORKFLOW")
print("="*70)

# ============================================================================
# Example 1: Pure Parameter Generation
# ============================================================================
print("\n" + "="*70)
print("Example 1: Pure Parameter Generation")
print("="*70)

# Create synthesizer
synth = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,  # Gpc^-3 yr^-1
    rate_evolution='constant',
    cosmology='Planck18',
    seed=42
)

# Generate just the parameters (no detection modeling, no light curves)
params_df = synth.generate_population(
    n_events=5,
    z_max=0.5,
    time_range=(60000, 60365.25)  # MJD range for 1 year
)

print(f"\nGenerated {len(params_df)} events")
print(f"Columns: {list(params_df.columns)}")
if len(params_df) > 0:
    print(f"\nFirst event parameters:")
    print(params_df.iloc[0])
else:
    print("\nNo events generated; increase n_events or rate.")

# This DataFrame can now be passed to ANY simulation tool!
print(f"\nThis DataFrame is ready to pass to:")
print(f"  - redback.SimulateOpticalTransient")
print(f"  - Your own simulation code")
print(f"  - External tools")


# ============================================================================
# Example 2: Integration with Redback SimulateOpticalTransient
# ============================================================================
print("\n" + "="*70)
print("Example 2: Using Parameters with SimulateOpticalTransient")
print("="*70)

# Generate a small sample
params_df_small = synth.generate_population(n_events=3, z_max=0.2)

print(f"\nGenerated {len(params_df_small)} events for simulation")
print("\nSimulating Rubin survey observations...")
from redback.simulate_transients import SimulateOpticalTransient

simulator = SimulateOpticalTransient.simulate_transient_population_in_rubin(
    model='one_component_kilonova_model',
    parameters=params_df_small.to_dict('list'),  # Convert to dict format
    model_kwargs={'output_format': 'sncosmo_source'}
)

print("Simulation complete.")
print(f"  Events simulated: {len(simulator.parameters)}")
print(f"  Observation tables: {len(simulator.list_of_observations)}")
print(f"  Observations in first event: {len(simulator.list_of_observations[0])}")


# ============================================================================
# Example 3: Custom Detection Criteria
# ============================================================================
print("\n" + "="*70)
print("Example 3: Custom Detection Criteria")
print("="*70)

# Generate population
params_df = synth.generate_population(n_events=100, z_max=1.0)

# Define custom detection function based on redshift
def simple_redshift_cut(row, z_threshold=0.3):
    """Simple: detect if z < threshold"""
    return row['redshift'] < z_threshold

# Apply detection criteria
params_with_det = synth.apply_detection_criteria(
    params_df,
    detection_function=simple_redshift_cut,
    z_threshold=0.3
)

n_detected = params_with_det['detected'].sum()
print(f"\nSimple redshift cut (z < 0.3):")
print(f"  Total: {len(params_with_det)}")
print(f"  Detected: {n_detected} ({100*n_detected/len(params_with_det):.1f}%)")


# Example 3b: Probabilistic detection based on redshift
def probabilistic_detection(row, z_50=0.5):
    """Detection probability decreases with redshift"""
    # Sigmoid: 50% at z_50, falling off at higher z
    prob = 1.0 / (1.0 + np.exp(5 * (row['redshift'] - z_50)))
    return prob  # Return probability (0-1)

params_with_prob = synth.apply_detection_criteria(
    params_df,
    detection_function=probabilistic_detection,
    z_50=0.4
)

n_detected = params_with_prob['detected'].sum()
print(f"\nProbabilistic detection (50% at z=0.4):")
print(f"  Total: {len(params_with_prob)}")
print(f"  Detected: {n_detected} ({100*n_detected/len(params_with_prob):.1f}%)")


# Example 3c: Complex multi-criterion detection
def complex_detection(row, z_max=1.0, mej_min=0.01):
    """Multiple criteria: redshift AND ejecta mass"""
    z_ok = row['redshift'] < z_max
    mej_ok = row.get('mej', 1.0) > mej_min  # Check if mej exists
    return z_ok and mej_ok  # Both must be true

params_complex = synth.apply_detection_criteria(
    params_df,
    detection_function=complex_detection,
    z_max=0.5,
    mej_min=0.02
)

n_detected = params_complex['detected'].sum()
print(f"\nComplex detection (z<0.5 AND mej>0.02):")
print(f"  Total: {len(params_complex)}")
print(f"  Detected: {n_detected} ({100*n_detected/len(params_complex):.1f}%)")
print("You could of course just do this manually as well, the results are all in the DataFrame.")


# ============================================================================
# Example 4: Rate Inference from Detected Sample
# ============================================================================
print("\n" + "="*70)
print("Example 4: Rate Inference from Detected Sample")
print("Note: This is a simplified demo for illustration purposes, not a full statistical treatment.")
print("="*70)

# Generate a "true" population with known rate
true_rate = 2.0  # Gpc^-3 yr^-1 (set for a stable demo sample)
synth_true = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=true_rate,
    rate_evolution='constant',
    seed=123
)

# Generate population (fixed sample size for a stable demo)
true_pop = synth_true.generate_population(n_events=200, z_max=1.0)
print(f"\nTrue rate: {true_rate:.2e} Gpc^-3 yr^-1")
print(f"Generated {len(true_pop)} events (fixed sample)")

# Apply some detection efficiency (redshift-dependent)
def efficiency_detection(row):
    """Efficiency drops with redshift"""
    return 1.0 / (1.0 + (row['redshift'] / 0.3)**3)

detected_pop = synth_true.apply_detection_criteria(
    true_pop,
    detection_function=efficiency_detection
)

# Get detected sample
detected_only = detected_pop[detected_pop['detected'] == True]
if len(detected_pop) > 0:
    print(f"Detected {len(detected_only)} events ({100*len(detected_only)/len(detected_pop):.1f}%)")
else:
    print("Detected 0 events (no population generated)")

# Define efficiency function for rate inference
def efficiency_func(z):
    """Must match what was used in detection"""
    return 1.0 / (1.0 + (z / 0.3)**3)

if len(detected_only) > 0:
    # Infer rate
    rate_results = synth_true.infer_rate(
        observed_sample=detected_only,
        efficiency_function=efficiency_func,
        z_bins=10
    )

    print(f"\nRate Inference Results:")
    print(f"  Inferred rate: {rate_results['rate_ml']:.2e} ± {rate_results['rate_uncertainty']:.2e} Gpc^-3 yr^-1")
    print(f"  True rate:     {true_rate:.2e} Gpc^-3 yr^-1")
    print(f"  Fractional error: {abs(rate_results['rate_ml'] - true_rate)/true_rate:.1%}")
else:
    print("\nRate inference skipped (no detected events).")


# ============================================================================
# Example 5: Analyzing Detection Efficiency vs Redshift
# ============================================================================
print("\n" + "="*70)
print("Example 5: Analyzing Detection Efficiency vs Redshift")
print("="*70)

# Generate large sample
synth_large = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,
    rate_evolution='constant',
    seed=999
)

large_pop = synth_large.generate_population(n_events=500, z_max=2.0)

# Apply redshift-dependent detection
def survey_detection(row, z_50=0.8):
    """Realistic survey: sharp cutoff around z_50"""
    return 1.0 / (1.0 + np.exp(10 * (row['redshift'] - z_50)))

detected_large = synth_large.apply_detection_criteria(
    large_pop,
    detection_function=survey_detection,
    z_50=0.8
)

# Analyze efficiency in redshift bins
z_bins = np.linspace(0, 2, 11)
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

print(f"\nDetection efficiency vs redshift:")
print(f"{'z_center':<10} {'N_total':<10} {'N_detected':<12} {'Efficiency':<12}")
print("-" * 50)

for i in range(len(z_centers)):
    mask = (detected_large['redshift'] >= z_bins[i]) & (detected_large['redshift'] < z_bins[i+1])
    n_in_bin = mask.sum()
    if n_in_bin > 0:
        n_det_in_bin = (detected_large[mask]['detected'] == True).sum()
        eff = n_det_in_bin / n_in_bin
        print(f"{z_centers[i]:<10.2f} {n_in_bin:<10} {n_det_in_bin:<12} {eff:<12.3f}")

# Plot efficiency curve
efficiencies = []
for i in range(len(z_centers)):
    mask = (detected_large['redshift'] >= z_bins[i]) & (detected_large['redshift'] < z_bins[i+1])
    n_in_bin = mask.sum()
    if n_in_bin > 0:
        n_det_in_bin = (detected_large[mask]['detected'] == True).sum()
        efficiencies.append(n_det_in_bin / n_in_bin)
    else:
        efficiencies.append(0.0)

plt.figure(figsize=(7, 4))
plt.plot(z_centers, efficiencies, marker='o')
plt.xlabel('Redshift')
plt.ylabel('Detection Efficiency')
plt.title('Detection Efficiency vs Redshift')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('population_detection_efficiency.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to population_detection_efficiency.png")


# ============================================================================
# Example 6: Evolving Rate Population
# ============================================================================
print("\n" + "="*70)
print("Example 6: Population with Evolving Rate")
print("="*70)

# Create synthesizer with powerlaw evolution
synth_evolving = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,
    rate_evolution='powerlaw',  # R(z) = R0 * (1+z)^2.7
    seed=456
)

# Generate population
evolving_pop = synth_evolving.generate_population(n_events=200, z_max=2.0)

# Analyze redshift distribution
z_bins_evolv = np.linspace(0, 2, 11)
z_counts, _ = np.histogram(evolving_pop['redshift'], bins=z_bins_evolv)
z_centers_evolv = (z_bins_evolv[:-1] + z_bins_evolv[1:]) / 2

print(f"\nGenerated {len(evolving_pop)} events with powerlaw rate evolution")
print(f"\nRedshift distribution:")
print(f"{'z_center':<10} {'N_events':<10} {'Relative rate':<15}")
print("-" * 40)

for i in range(len(z_centers_evolv)):
    rel_rate = (1 + z_centers_evolv[i])**2.7  # Expected relative rate
    print(f"{z_centers_evolv[i]:<10.2f} {z_counts[i]:<10} {rel_rate:<15.2f}")

print(f"\nNote: More events at higher redshift due to R(z) ∝ (1+z)^2.7")


# ============================================================================
# Example 7: Saving and Loading for Later Processing
# ============================================================================
print("\n" + "="*70)
print("Example 7: Saving Parameters for Later Use")
print("="*70)

# Generate population
params_to_save = synth.generate_population(n_events=50, z_max=0.5)

# Save to file
import os
os.makedirs('populations', exist_ok=True)
params_to_save.to_csv('populations/my_population_params.csv', index=False)
print(f"\nSaved {len(params_to_save)} events to populations/my_population_params.csv")

print("\nYou can now:")
print("  1. Load this file in another script")
print("  2. Pass to redback simulation tools")
print("  3. Apply different detection criteria")
print("  4. Use in custom analysis pipelines")

# Load it back
loaded_params = pd.read_csv('populations/my_population_params.csv')
print(f"\nLoaded {len(loaded_params)} events from file")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Key Advantages of Modular Design")
print("="*70)
print("""
1. SEPARATION OF CONCERNS:
   - generate_population() → Pure parameter generation
   - apply_detection_criteria() → Flexible post-processing
   - infer_rate() → Independent rate inference

2. FLEXIBILITY:
   - Parameters can go to any simulation tool e.g., in-house redback or lightcurve lynx, or batched sims 
   - Custom detection logic easily implemented
   - Multiple post-processing passes possible

3. WORKFLOW:
   a) Generate parameters with correct rate, cosmology, priors
   b) Pass to simulation tools (redback or custom)
   c) Apply detection criteria based on survey/science
   d) Analyze results, infer rates

4. INTEGRATION:
   - Works seamlessly with SimulateOpticalTransient
   - Compatible with external tools
   - Easy to extend with custom functions
""")

print("\n" + "="*70)
print("All examples completed successfully!")
print("="*70)
