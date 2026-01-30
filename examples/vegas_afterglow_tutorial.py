# VegasAfterglow Integration Tutorial
# This example demonstrates the full capabilities of using VegasAfterglow models in redback

import redback
import bilby
import numpy as np
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateGenericTransient

# ===============================================================================
# 1. Simulate GRB Afterglow Data
# ===============================================================================

print("=" * 80)
print("Step 1: Simulating GRB Afterglow Data with VegasAfterglow")
print("=" * 80)

# Set up parameters for a tophat jet in an ISM
parameters = {
    'redshift': 0.5,
    'thv': 0.3,         # viewing angle in radians  
    'loge0': 52.5,      # log10 isotropic energy [erg]
    'thc': 0.2,         # jet core angle in radians
    'lognism': 0.0,     # log10 ISM density [cm^-3]
    'loga': -10.0,      # log10 wind parameter (very small = pure ISM)
    'p': 2.2,           # electron power-law index
    'logepse': -1.0,    # log10 electron energy fraction
    'logepsb': -2.0,    # log10 magnetic energy fraction
    'g0': 300.0         # initial Lorentz factor
}

# Generate synthetic observations at optical frequency
times = np.logspace(0, 2, 20)  # 1 to 100 days
model_kwargs = {'output_format': 'flux_density', 'frequency': 5e14}  # Optical

simulator = SimulateGenericTransient(
    model='vegas_tophat',
    parameters=parameters,
    times=times,
    data_points=20,
    model_kwargs=model_kwargs,
    noise_term=0.1
)

print(f"Generated {len(simulator.data)} synthetic observations")
print(simulator.data.head())

# ===============================================================================
# 2. Load Data into Redback Afterglow Object
# ===============================================================================

print("\\n" + "=" * 80)
print("Step 2: Loading Data into Redback")
print("=" * 80)

# Create Afterglow object from simulated data
afterglow = redback.transient.Afterglow(
    name='GRB_synthetic',
    data_mode='flux_density',
    time=simulator.data['time'].values,
    flux_density=simulator.data['output'].values,
    flux_density_err=simulator.data['output_error'].values,
    frequency=np.repeat(5e14, len(simulator.data))
)

print(f"Loaded afterglow object: {afterglow.name}")
print(f"Data mode: {afterglow.data_mode}")
print(f"Number of observations: {len(afterglow.x)}")

# Plot the data
afterglow.plot_data()
plt.savefig('vegas_synthetic_data.png', dpi=150, bbox_inches='tight')
print("Data plot saved as: vegas_synthetic_data.png")

# ===============================================================================
# 3. Maximum Likelihood Estimation
# ===============================================================================

print("\\n" + "=" * 80)
print("Step 3: Maximum Likelihood Estimation")
print("=" * 80)

# Get priors for vegas_tophat
priors = redback.priors.get_priors(model='vegas_tophat')

# Do a quick maximum likelihood fit
result_mle = redback.fit_model(
    model='vegas_tophat',
    sampler='dynesty',
    nlive=100,
    transient=afterglow,
    prior=priors,
    sample='rwalk',
    clean=True,
    maxmcmc=1000  # Quick run for demo
)

print("Maximum likelihood parameters:")
for key, value in result_mle.posterior.iloc[result_mle.posterior['log_likelihood'].idxmax()].items():
    if key in parameters:
        print(f"  {key}: {value:.3f} (true: {parameters[key]:.3f})")

# ===============================================================================
# 4. Full Bayesian Inference
# ===============================================================================

print("\\n" + "=" * 80)
print("Step 4: Full Bayesian Parameter Estimation")
print("=" * 80)

# Run full sampling
result = redback.fit_model(
    model='vegas_tophat',
    sampler='dynesty',
    nlive=500,
    transient=afterglow,
    prior=priors,
    sample='rslice',
    clean=True
)

# Plot results
result.plot_corner()
plt.savefig('vegas_corner.png', dpi=150, bbox_inches='tight')
print("Corner plot saved as: vegas_corner.png")

result.plot_lightcurve(random_models=100)
plt.savefig('vegas_lightcurve_fit.png', dpi=150, bbox_inches='tight')
print("Lightcurve fit saved as: vegas_lightcurve_fit.png")

# ===============================================================================
# 5. Compare with Other Redback Models
# ===============================================================================

print("\\n" + "=" * 80)
print("Step 5: Comparing VegasAfterglow with Other Redback Models")
print("=" * 80)

# Compare vegas_tophat with jetsimpy_tophat
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.errorbar(afterglow.x, afterglow.y, yerr=afterglow.y_err, 
            fmt='o', label='Data', color='black', alpha=0.6)

# Plot vegas model
time_plot = np.logspace(-0.5, 2.5, 100)
vegas_flux = redback.model_library.all_models_dict['vegas_tophat'](
    time=time_plot, **parameters, **model_kwargs
)
ax.plot(time_plot, vegas_flux, label='VegasAfterglow (tophat)', linewidth=2)

# Plot jetsimpy for comparison (if available)
try:
    jetsimpy_params = {
        'redshift': parameters['redshift'],
        'thv': parameters['thv'],
        'loge0': parameters['loge0'],
        'thc': parameters['thc'],
        'nism': 10**parameters['lognism'],
        'A': 10**parameters['loga'],
        'p': parameters['p'],
        'logepse': parameters['logepse'],
        'logepsb': parameters['logepsb'],
        'g0': parameters['g0']
    }
    jetsimpy_flux = redback.model_library.all_models_dict['jetsimpy_tophat'](
        time=time_plot, **jetsimpy_params, **model_kwargs
    )
    ax.plot(time_plot, jetsimpy_flux, '--', label='Jetsimpy (tophat)', linewidth=2)
except:
    print("Jetsimpy not available for comparison")

ax.set_xlabel('Time [days]', fontsize=14)
ax.set_ylabel('Flux Density [mJy]', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vegas_model_comparison.png', dpi=150, bbox_inches='tight')
print("Model comparison saved as: vegas_model_comparison.png")

# ===============================================================================
# 6. Advanced Features
# ===============================================================================

print("\\n" + "=" * 80)
print("Step 6: Redback Advanced Features")
print("=" * 80)

print("\\nRedback and VegasAfterglow provide many additional capabilities:")
print("  - Non-detections: Include upper limits in your fits")
print("  - Complex likelihoods: Gaussian processes, mixture models, etc.")
print("  - Spectra fitting: Multi-wavelength spectral analysis")
print("  - Combined models: Kilonova + afterglow, SN + magnetar, etc.")
print("  - Multiple jet structures: tophat, Gaussian, power-law, step, etc.")
print("  - Reverse shocks: Enable with reverse_shock=True")
print("  - SSC radiation: Enable with ssc=True")
print("  - Magnetar energy injection: Add magnetar_L0, magnetar_t0 parameters")

print("\\n" + "=" * 80)
print("Tutorial Complete!")
print("=" * 80)
print("Check out the other examples!")