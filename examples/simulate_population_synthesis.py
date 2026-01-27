"""
Example script demonstrating the PopulationSynthesizer functionality

This shows how to:
1. Create a population synthesizer for different transient types
2. Simulate realistic populations with rate evolution
3. Apply selection effects
4. Analyze populations
5. Infer rates from observed samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from redback.simulate_transients import PopulationSynthesizer, TransientPopulation
import redback

# Example 1: Basic kilonova population with constant rate
print("=" * 60)
print("Example 1: Basic Kilonova Population")
print("=" * 60)

# Create a population synthesizer for kilonovae
synth = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,  # 1e-6 Gpc^-3 yr^-1
    rate_evolution='constant',
    cosmology='Planck18',
    seed=42
)

# Simulate a population with a fixed sample size for a stable demo
population = synth.simulate_population(
    n_events=50,
    include_selection_effects=False
)

print(f"Generated {len(population)} kilonovae")
print(f"Redshift range: {population.redshifts.min():.3f} - {population.redshifts.max():.3f}")
print(f"\nFirst 5 events:")
print(population.parameters.head())

# Save the population
population.save('kilonova_population_10yr.csv')


# Example 2: Supernova population with evolving rate
print("\n" + "=" * 60)
print("Example 2: Supernova Population with Rate Evolution")
print("=" * 60)

# Create synthesizer with SFR-like rate evolution
synth_sn = PopulationSynthesizer(
    model='arnett_bolometric',
    rate=1e-4,  # Higher rate for SNe
    rate_evolution='sfr_like',  # Follows star formation rate
    cosmology='Planck18',
    seed=123
)

# Simulate population with a fixed size to ensure non-zero counts
population_sn = synth_sn.simulate_population(
    n_events=300,
    z_max=1.5
)

print(f"Generated {len(population_sn)} supernovae")
print(f"Redshift range: {population_sn.redshifts.min():.3f} - {population_sn.redshifts.max():.3f}")

# Analyze redshift distribution
edges, counts, centers = population_sn.get_redshift_distribution(bins=15)
print(f"\nRedshift distribution (first 5 bins):")
for i in range(min(5, len(centers))):
    print(f"  z={centers[i]:.2f}: {counts[i]} events")

# Plot redshift distribution
plt.figure(figsize=(7, 4))
plt.bar(centers, counts, width=(edges[1] - edges[0]), alpha=0.7)
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Supernova Redshift Distribution')
plt.tight_layout()
plt.savefig('population_synthesis_redshift.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to population_synthesis_redshift.png")


# Example 3: Population with selection effects (LSST-like survey)
print("\n" + "=" * 60)
print("Example 3: Kilonova Population with Selection Effects")
print("=" * 60)

# Create synthesizer
synth_lsst = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,
    rate_evolution='constant',
    cosmology='Planck18',
    seed=456
)

# Define survey configuration (LSST-like)
lsst_config = {
    'limiting_mag': 24.5,  # 5-sigma depth in r-band
    'bands': ['lsstr'],
    'area_sqdeg': 18000  # Full survey area
}

# Simulate with selection effects (fixed size for stability)
population_detected = synth_lsst.simulate_population(
    n_events=300,
    include_selection_effects=True,
    survey_config=lsst_config
)

print(f"Total transients: {len(population_detected)}")
print(f"Detection fraction: {population_detected.detection_fraction:.2%}")
print(f"Detected transients: {len(population_detected.detected)}")

# Analyze detected vs all
if 'detected' in population_detected.parameters.columns:
    all_z = population_detected.redshifts
    detected_z = population_detected.detected['redshift'].values
    print(f"\nMean redshift (all): {np.mean(all_z):.3f}")
    print(f"Mean redshift (detected): {np.mean(detected_z):.3f}")
    print(f"Max redshift (detected): {np.max(detected_z):.3f}")


# Example 4: Rate inference
print("\n" + "=" * 60)
print("Example 4: Infer Rate from Observed Sample")
print("Note: This is again a a simple example of rate inference. It does not do any hierarchical modeling.")
print("=" * 60)

# Create a "true" population
synth_true = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=600,  # True rate (set for a stable demo)
    rate_evolution='constant',
    seed=789
)

true_population = synth_true.simulate_population(n_events=200)
print(f"True rate: {600:.2e} Gpc^-3 yr^-1")
print(f"Observed {len(true_population)} events")

# Infer the rate back
rate_inference = synth_true.infer_rate(
    observed_sample=true_population,
    efficiency_function=None,  # Assume perfect detection
    z_bins=10
)

print(f"\nInferred rate: {rate_inference['rate_ml']:.2e} ± {rate_inference['rate_uncertainty']:.2e} Gpc^-3 yr^-1")
print(f"Fractional error: {rate_inference['rate_uncertainty']/rate_inference['rate_ml']:.1%}")


# Example 5: Fixed number of events
print("\n" + "=" * 60)
print("Example 5: Generate Fixed Number of Events")
print("=" * 60)

# Sometimes you want a fixed number rather than Poisson draw
synth_fixed = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,
    seed=999
)

population_fixed = synth_fixed.simulate_population(
    n_events=100,  # Exactly 100 events
    z_max=0.5
)

print(f"Generated exactly {len(population_fixed)} events")
print(f"Redshift range: {population_fixed.redshifts.min():.3f} - {population_fixed.redshifts.max():.3f}")


# Example 6: Custom rate evolution
print("\n" + "=" * 60)
print("Example 6: Custom Rate Evolution Function")
print("=" * 60)

# Define a custom rate evolution (e.g., peaked at z=1)
def custom_rate(z):
    """Rate peaks at z=1"""
    base = 1e-6
    return base * np.exp(-(z - 1.0)**2 / 0.5)

synth_custom = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=custom_rate,  # Pass function directly
    cosmology='Planck18',
    seed=111
)

population_custom = synth_custom.simulate_population(
    n_events=200,
    z_max=3.0
)

# Check redshift distribution
edges, counts, centers = population_custom.get_redshift_distribution(bins=20)
peak_idx = np.argmax(counts)
print(f"Generated {len(population_custom)} events")
print(f"Peak of redshift distribution at z ~ {centers[peak_idx]:.2f}")
print(f"Expected peak at z = 1.0")


# Example 7: Analyzing parameter distributions
print("\n" + "=" * 60)
print("Example 7: Parameter Distribution Analysis")
print("=" * 60)

synth_params = PopulationSynthesizer(
    model='one_component_kilonova_model',
    rate=1e-6,
    seed=222
)

population_params = synth_params.simulate_population(n_events=500)

# Analyze ejecta mass distribution (if in parameters)
if 'mej' in population_params.parameters.columns:
    edges, counts, centers = population_params.get_parameter_distribution('mej', bins=20)
    print(f"Ejecta mass (mej) distribution:")
    print(f"  Mean: {population_params.parameters['mej'].mean():.3f} M_sun")
    print(f"  Median: {population_params.parameters['mej'].median():.3f} M_sun")
    print(f"  Range: [{population_params.parameters['mej'].min():.3f}, "
          f"{population_params.parameters['mej'].max():.3f}] M_sun")


# Example 8: Loading saved populations
print("\n" + "=" * 60)
print("Example 8: Save and Load Populations")
print("=" * 60)

# Save with metadata
population.save('example_population.csv', save_metadata=True)
print("Saved population to populations/example_population.csv")

# Load it back
loaded_pop = TransientPopulation.load('example_population.csv')
print(f"Loaded {len(loaded_pop)} events")
print(f"Metadata: {loaded_pop.metadata}")


print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
