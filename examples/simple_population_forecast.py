"""
Simple population generation without rate modeling

This demonstrates the rate_weighted_redshifts=False option, which is useful when:
- You want to forecast N events without committing to a specific rate
- Testing/exploring parameter space
- Using UniformComovingVolume for simple uniform-in-volume sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from bilby.gw.prior import UniformComovingVolume
from redback.simulate_transients import PopulationSynthesizer
import redback

print("="*70)
print("SIMPLE POPULATION FORECAST (No Rate Modeling)")
print("="*70)

# ============================================================================
# Example 1: Using UniformComovingVolume for simple forecasts
# ============================================================================
print("\n" + "="*70)
print("Example 1: N Events Uniformly in Comoving Volume")
print("="*70)

# Create a custom prior with UniformComovingVolume for redshifts
# This samples uniformly in comoving volume (no rate weighting)
custom_prior = redback.priors.get_priors('one_component_kilonova_model')

# Replace the redshift prior with UniformComovingVolume
custom_prior['redshift'] = UniformComovingVolume(
    minimum=0.001,
    maximum=0.5,
    name='redshift',
    cosmology='Planck18'
)

print(f"Redshift prior: {custom_prior['redshift']}")

# Create synthesizer
synth = PopulationSynthesizer(
    model='one_component_kilonova_model',
    prior=custom_prior,
    rate=1e-6,  # Not used when rate_weighted_redshifts=False
    seed=42
)

# Generate 100 events uniformly in volume and time
params = synth.generate_population(
    n_events=100,  # Specify exact number
    time_range=(60000, 60365.25),  # 1 year survey
    rate_weighted_redshifts=False  # Key parameter!
)

print(f"\nGenerated {len(params)} events")
print(f"Redshift range: [{params['redshift'].min():.3f}, {params['redshift'].max():.3f}]")
print(f"Time range: [{params['t0_mjd_transient'].min():.1f}, {params['t0_mjd_transient'].max():.1f}]")

# Analyze redshift distribution
print(f"\nRedshift distribution:")
z_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
z_counts, _ = np.histogram(params['redshift'], bins=z_bins)
for i in range(len(z_bins)-1):
    print(f"  {z_bins[i]:.1f} < z < {z_bins[i+1]:.1f}: {z_counts[i]} events")

print("\nNote: Events are roughly uniform in each bin (uniform in comoving volume)")


# ============================================================================
# Example 2: Comparison with Rate-Weighted Sampling
# ============================================================================
print("\n" + "="*70)
print("Example 2: Comparison - Rate-Weighted vs Uniform")
print("="*70)

# Same setup but rate-weighted
synth_rate = PopulationSynthesizer(
    model='one_component_kilonova_model',
    prior=custom_prior,
    rate=1e-6,
    rate_evolution='powerlaw',  # R(z) ∝ (1+z)^2.7
    seed=42
)

# Generate with rate weighting
params_rate = synth_rate.generate_population(
    n_events=100,
    rate_weighted_redshifts=True  # Use rate weighting
)

# Generate without rate weighting
params_uniform = synth_rate.generate_population(
    n_events=100,
    rate_weighted_redshifts=False  # Uniform in volume
)

print(f"\nRate-weighted (powerlaw R(z) ∝ (1+z)^2.7):")
print(f"  Mean redshift: {params_rate['redshift'].mean():.3f}")
print(f"  Median redshift: {params_rate['redshift'].median():.3f}")

print(f"\nUniform in comoving volume:")
print(f"  Mean redshift: {params_uniform['redshift'].mean():.3f}")
print(f"  Median redshift: {params_uniform['redshift'].median():.3f}")

print("\nNote: Rate-weighted has higher mean z due to (1+z)^2.7 evolution")

# Plot the redshift distributions
plt.figure(figsize=(7, 4))
bins = np.linspace(0, 0.5, 20)
plt.hist(params_rate['redshift'], bins=bins, alpha=0.6, label='Rate-weighted')
plt.hist(params_uniform['redshift'], bins=bins, alpha=0.6, label='Uniform')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Redshift Distributions')
plt.legend()
plt.tight_layout()
plt.savefig('simple_population_forecast_redshift.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to simple_population_forecast_redshift.png")


# ============================================================================
# Example 3: Quick Forecast for Survey Planning
# ============================================================================
print("\n" + "="*70)
print("Example 3: Quick Survey Planning Forecast")
print("="*70)

# You want to know: "If I see ~50 kilonovae in my survey, what will
# the distribution of parameters look like?"

# Set up prior with your survey's redshift reach
forecast_prior = redback.priors.get_priors('one_component_kilonova_model')
forecast_prior['redshift'] = UniformComovingVolume(
    minimum=0.01,
    maximum=0.3,  # Your survey's reach
    name='redshift'
)

synth_forecast = PopulationSynthesizer(
    model='one_component_kilonova_model',
    prior=forecast_prior,
    seed=999
)

# Generate 50 events
forecast_params = synth_forecast.generate_population(
    n_events=50,
    time_range=(60000, 60365.25*3),  # 3 year survey
    rate_weighted_redshifts=False
)

print(f"\nForecast: {len(forecast_params)} events over 3 years")
print(f"\nParameter distributions:")

if 'mej' in forecast_params.columns:
    print(f"  Ejecta mass (mej):")
    print(f"    Range: [{forecast_params['mej'].min():.4f}, {forecast_params['mej'].max():.4f}] M_sun")
    print(f"    Median: {forecast_params['mej'].median():.4f} M_sun")

if 'vej' in forecast_params.columns:
    print(f"  Ejecta velocity (vej):")
    print(f"    Range: [{forecast_params['vej'].min():.3f}, {forecast_params['vej'].max():.3f}] c")
    print(f"    Median: {forecast_params['vej'].median():.3f} c")

print(f"  Redshift:")
print(f"    Range: [{forecast_params['redshift'].min():.3f}, {forecast_params['redshift'].max():.3f}]")
print(f"    Median: {forecast_params['redshift'].median():.3f}")


# ============================================================================
# Example 4: Use with Existing Simulation Tools
# ============================================================================
print("\n" + "="*70)
print("Example 4: Pass to SimulateOpticalTransient")
print("="*70)

# Generate simple parameter set with a low-z prior for easier detection
simple_prior = custom_prior.copy()
simple_prior['redshift'] = UniformComovingVolume(
    minimum=0.001,
    maximum=0.05,
    name='redshift',
    cosmology='Planck18'
)
simple_synth = PopulationSynthesizer(
    model='one_component_kilonova_model',
    prior=simple_prior,
    seed=123
)
simple_params = simple_synth.generate_population(
    n_events=5,
    rate_weighted_redshifts=False
)

print(f"\nGenerated {len(simple_params)} events")
print("\nSimulating Rubin survey observations...")
from redback.simulate_transients import SimulateOpticalTransient

simulator = SimulateOpticalTransient.simulate_transient_population_in_rubin(
    model='one_component_kilonova_model',
    parameters=simple_params.to_dict('list'),
    model_kwargs={'output_format': 'sncosmo_source'},
    end_transient_time=10)

print("Simulation complete.")
print(f"  Events simulated: {len(simulator.parameters)}")
print(f"  Observation tables: {len(simulator.list_of_observations)}")
print(f"  Observations in first event: {len(simulator.list_of_observations[0])}")


# ============================================================================
# Example 5: Error Checking
# ============================================================================
print("\n" + "="*70)
print("Example 5: What Happens Without n_events?")
print("="*70)

print("\nIf you don't specify n_events with rate_weighted_redshifts=False:")
print("You'll get an error (can't calculate from rate without rate weighting)")

try:
    bad_params = synth.generate_population(
        n_years=10,  # Only specify n_years
        rate_weighted_redshifts=False  # But no rate weighting!
    )
except ValueError as e:
    print(f"\n✗ Error (expected): {e}")

print("\nSolution: Always specify n_events when using rate_weighted_redshifts=False")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Two modes of operation:

1. RATE-WEIGHTED (default, rate_weighted_redshifts=True):
   - Redshifts sampled as: R(z) * dVc/dz / (1+z)
   - Number of events from Poisson(rate × volume × time)
   - Use when: doing rate inference, realistic populations

2. SIMPLE FORECAST (rate_weighted_redshifts=False):
   - Redshifts sampled from prior (e.g., UniformComovingVolume)
   - Number of events specified by user
   - Use when: forecasting, testing, survey planning

Key setup for mode 2:
```python
from bilby.gw.prior import UniformComovingVolume

prior['redshift'] = UniformComovingVolume(
    minimum=0.001,
    maximum=0.5,
    name='redshift'
)

params = synth.generate_population(
    n_events=100,
    rate_weighted_redshifts=False
)
```

Events are placed uniformly in time in both cases!
""")

print("\n" + "="*70)
print("All examples completed!")
print("="*70)
