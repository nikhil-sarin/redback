"""
Multi-Messenger Analysis Example
==================================

This example demonstrates how to use the MultiMessengerTransient class for joint
analysis of transients observed through multiple messengers (optical, X-ray, radio, etc.).

We'll simulate a GW170817-like event with:
1. Optical kilonova emission
2. X-ray afterglow
3. Radio afterglow

And perform both individual and joint parameter estimation.
"""

import numpy as np
import bilby
import redback
from redback.multimessenger import MultiMessengerTransient, create_joint_prior
from redback.transient_models import kilonova_models, afterglow_models

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Step 1: Define true parameters for simulation
# ============================================================================

# Shared parameters (same across all messengers)
true_params = {
    'viewing_angle': 0.4,  # ~23 degrees
    'redshift': 0.01,
    'luminosity_distance': 40.0,  # Mpc
}

# Kilonova-specific parameters (optical)
kilonova_params = {
    'mej': 0.05,  # ejecta mass in solar masses
    'vej': 0.2,   # ejecta velocity (c)
    'kappa': 3.0,  # opacity
    **true_params
}

# Afterglow parameters (X-ray and radio)
afterglow_params = {
    'loge0': 52.0,  # log10 of energy in ergs
    'thc': 0.1,     # jet core angle
    'logn0': 0.0,   # log10 of ISM density
    'p': 2.2,       # electron spectral index
    'logepse': -1.0,  # log10 of epsilon_e
    'logepsb': -2.0,  # log10 of epsilon_B
    'ksin': 1,
    'g0': 1000,
    'thv': true_params['viewing_angle'],  # viewing angle
    **true_params
}

# ============================================================================
# Step 2: Simulate observations
# ============================================================================

print("Simulating multi-messenger observations...")

# Optical kilonova (0.5 - 20 days)
optical_time = np.linspace(0.5, 20, 30)
optical_kwargs = {
    'output_format': 'magnitude',
    'bands': np.array(['bessellux'] * len(optical_time)),
    'frequency': None
}

# We'll use a simple two-component kilonova model
true_optical_mag = kilonova_models.two_component_kilonova_model(
    optical_time,
    mej_1=0.03, vej_1=0.2, kappa_1=1.0,   # blue component
    mej_2=0.02, vej_2=0.15, kappa_2=10.0,  # red component
    redshift=true_params['redshift'],
    **optical_kwargs
)

# Add noise
optical_mag_err = 0.1 * np.ones_like(true_optical_mag)
observed_optical_mag = np.random.normal(true_optical_mag, optical_mag_err)

# Create optical transient object
optical_transient = redback.transient.Transient(
    name='simulated_kilonova',
    time=optical_time,
    magnitude=observed_optical_mag,
    magnitude_err=optical_mag_err,
    bands=optical_kwargs['bands'],
    data_mode='magnitude',
    redshift=true_params['redshift']
)

print(f"  ✓ Simulated optical kilonova ({len(optical_time)} points)")

# X-ray afterglow (1 - 100 days at 2keV ~ 5e17 Hz)
xray_time = np.logspace(np.log10(1), np.log10(100), 20)
xray_frequency = np.ones_like(xray_time) * 5e17  # 2 keV

xray_kwargs = {
    'output_format': 'flux_density',
    'frequency': xray_frequency
}

true_xray_flux = afterglow_models.tophat(
    xray_time,
    **afterglow_params,
    **xray_kwargs
)

# Add noise
xray_flux_err = 0.15 * true_xray_flux
observed_xray_flux = np.random.normal(true_xray_flux, xray_flux_err)

# Create X-ray transient object
xray_transient = redback.transient.Transient(
    name='simulated_xray',
    time=xray_time,
    flux_density=observed_xray_flux,
    flux_density_err=xray_flux_err,
    frequency=xray_frequency,
    data_mode='flux_density',
    redshift=true_params['redshift']
)

print(f"  ✓ Simulated X-ray afterglow ({len(xray_time)} points)")

# Radio afterglow (10 - 200 days at 3 GHz)
radio_time = np.logspace(np.log10(10), np.log10(200), 15)
radio_frequency = np.ones_like(radio_time) * 3e9  # 3 GHz

radio_kwargs = {
    'output_format': 'flux_density',
    'frequency': radio_frequency
}

true_radio_flux = afterglow_models.tophat(
    radio_time,
    **afterglow_params,
    **radio_kwargs
)

# Add noise
radio_flux_err = 0.2 * true_radio_flux
observed_radio_flux = np.random.normal(true_radio_flux, radio_flux_err)

# Create radio transient object
radio_transient = redback.transient.Transient(
    name='simulated_radio',
    time=radio_time,
    flux_density=observed_radio_flux,
    flux_density_err=radio_flux_err,
    frequency=radio_frequency,
    data_mode='flux_density',
    redshift=true_params['redshift']
)

print(f"  ✓ Simulated radio afterglow ({len(radio_time)} points)")

# ============================================================================
# Step 3: Create MultiMessengerTransient object
# ============================================================================

print("\nCreating MultiMessengerTransient object...")

mm_transient = MultiMessengerTransient(
    optical_transient=optical_transient,
    xray_transient=xray_transient,
    radio_transient=radio_transient,
    name='GW170817_like'
)

print(f"  {mm_transient}")

# ============================================================================
# Step 4: Set up priors for joint analysis
# ============================================================================

print("\nSetting up priors for joint analysis...")

# Create prior dictionary
priors = bilby.core.prior.PriorDict()

# Shared parameters
priors['viewing_angle'] = bilby.core.prior.Uniform(0, np.pi/2, 'viewing_angle',
                                                    latex_label=r'$\theta_{\rm obs}$')
priors['redshift'] = true_params['redshift']  # Fixed

# Optical (kilonova) parameters
priors['mej_1'] = bilby.core.prior.Uniform(0.01, 0.1, 'mej_1', latex_label=r'$M_{\rm ej,1}$')
priors['vej_1'] = bilby.core.prior.Uniform(0.1, 0.3, 'vej_1', latex_label=r'$v_{\rm ej,1}$')
priors['kappa_1'] = bilby.core.prior.Uniform(0.5, 5.0, 'kappa_1', latex_label=r'$\kappa_1$')
priors['mej_2'] = bilby.core.prior.Uniform(0.01, 0.1, 'mej_2', latex_label=r'$M_{\rm ej,2}$')
priors['vej_2'] = bilby.core.prior.Uniform(0.05, 0.25, 'vej_2', latex_label=r'$v_{\rm ej,2}$')
priors['kappa_2'] = bilby.core.prior.Uniform(5.0, 20.0, 'kappa_2', latex_label=r'$\kappa_2$')

# Afterglow parameters (shared between X-ray and radio)
priors['loge0'] = bilby.core.prior.Uniform(50, 54, 'loge0', latex_label=r'$\log E_0$')
priors['thc'] = bilby.core.prior.Uniform(0.05, 0.3, 'thc', latex_label=r'$\theta_c$')
priors['logn0'] = bilby.core.prior.Uniform(-3, 2, 'logn0', latex_label=r'$\log n_0$')
priors['p'] = bilby.core.prior.Uniform(2.0, 3.0, 'p', latex_label=r'$p$')
priors['logepse'] = bilby.core.prior.Uniform(-3, 0, 'logepse', latex_label=r'$\log \epsilon_e$')
priors['logepsb'] = bilby.core.prior.Uniform(-4, 0, 'logepsb', latex_label=r'$\log \epsilon_B$')
priors['ksin'] = 1  # Fixed
priors['g0'] = 1000  # Fixed

print("  ✓ Priors configured")

# ============================================================================
# Step 5: Perform individual fits (for comparison)
# ============================================================================

print("\n" + "="*70)
print("INDIVIDUAL FITS (for comparison)")
print("="*70)

# Note: For demonstration, we'll use fast settings (low nlive).
# For real analysis, use nlive >= 2000

individual_models = {
    'optical': 'two_component_kilonova_model',
    'xray': 'tophat',
    'radio': 'tophat'
}

individual_model_kwargs = {
    'optical': optical_kwargs,
    'xray': xray_kwargs,
    'radio': radio_kwargs
}

# For individual fits, we need separate priors for each messenger
optical_priors = bilby.core.prior.PriorDict()
optical_priors.update({k: v for k, v in priors.items()
                      if k in ['redshift', 'mej_1', 'vej_1', 'kappa_1',
                              'mej_2', 'vej_2', 'kappa_2']})

afterglow_priors = bilby.core.prior.PriorDict()
afterglow_priors['thv'] = priors['viewing_angle']  # Map viewing_angle -> thv
afterglow_priors['redshift'] = true_params['redshift']
afterglow_priors.update({k: v for k, v in priors.items()
                        if k in ['loge0', 'thc', 'logn0', 'p',
                                'logepse', 'logepsb', 'ksin', 'g0']})

individual_priors = {
    'optical': optical_priors,
    'xray': afterglow_priors,
    'radio': afterglow_priors
}

# Uncomment to run individual fits (takes time)
# individual_results = mm_transient.fit_individual(
#     models=individual_models,
#     priors=individual_priors,
#     model_kwargs=individual_model_kwargs,
#     nlive=500,  # Low for speed
#     sampler='dynesty',
#     outdir='./outdir_individual',
#     resume=True
# )

print("  (Individual fits commented out for speed - uncomment to run)")

# ============================================================================
# Step 6: Perform joint multi-messenger fit
# ============================================================================

print("\n" + "="*70)
print("JOINT MULTI-MESSENGER FIT")
print("="*70)

# For joint fit, we need to map viewing_angle to thv for afterglow models
# We'll create wrapper functions to handle this

def xray_model(time, viewing_angle, **kwargs):
    """Wrapper to map viewing_angle -> thv for afterglow model"""
    return afterglow_models.tophat(time, thv=viewing_angle, **kwargs)

def radio_model(time, viewing_angle, **kwargs):
    """Wrapper to map viewing_angle -> thv for afterglow model"""
    return afterglow_models.tophat(time, thv=viewing_angle, **kwargs)

joint_models = {
    'optical': 'two_component_kilonova_model',
    'xray': xray_model,
    'radio': radio_model
}

shared_params = ['viewing_angle']

print(f"\nShared parameters: {shared_params}")
print("\nStarting joint analysis...")
print("  (Using low nlive for speed - increase for production)")

# Uncomment to run joint fit (takes time)
# joint_result = mm_transient.fit_joint(
#     models=joint_models,
#     priors=priors,
#     shared_params=shared_params,
#     model_kwargs={
#         'optical': optical_kwargs,
#         'xray': xray_kwargs,
#         'radio': radio_kwargs
#     },
#     nlive=500,  # Low for speed, use >= 2000 for real analysis
#     sampler='dynesty',
#     outdir='./outdir_joint',
#     label='GW170817_like_joint',
#     resume=True,
#     plot=True
# )

print("  (Joint fit commented out for speed - uncomment to run)")

# ============================================================================
# Step 7: Compare results
# ============================================================================

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print("""
After running both individual and joint fits, you can compare:

1. Viewing angle constraints:
   - Individual fits: each messenger constrains viewing_angle independently
   - Joint fit: all messengers constrain viewing_angle together

2. Evidence comparison:
   - Compare Bayes factors to assess if joint model is preferred

3. Parameter correlations:
   - Joint fit reveals correlations across messengers
   - E.g., viewing_angle correlation with optical and radio properties

To plot results:
    joint_result.plot_corner()

To plot individual messenger lightcurves with posteriors:
    redback.analysis.plot_lightcurve(
        transient=optical_transient,
        parameters=joint_result.posterior.sample(100),
        model=kilonova_models.two_component_kilonova_model,
        model_kwargs=optical_kwargs
    )
""")

# ============================================================================
# Example: Using with GW data
# ============================================================================

print("\n" + "="*70)
print("ADVANCED: Including Gravitational Wave Data")
print("="*70)

print("""
To include GW data, construct a bilby.gw likelihood and pass it as
an external likelihood:

Example:
--------
# Set up GW data (following bilby.gw workflow)
import bilby.gw

waveform_generator = bilby.gw.WaveformGenerator(...)
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
interferometers.set_strain_data_from_power_spectral_densities(...)

gw_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
    priors=gw_priors
)

# Create MultiMessengerTransient with GW
mm_transient = MultiMessengerTransient(
    optical_transient=optical_transient,
    xray_transient=xray_transient,
    gw_likelihood=gw_likelihood
)

# Add GW-EM shared parameters to priors
priors['chirp_mass'] = bilby.core.prior.Gaussian(1.2, 0.1)
priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(10, 250)

# Run joint fit
result = mm_transient.fit_joint(
    models={'optical': kilonova_model, 'xray': afterglow_model},
    priors=priors,
    shared_params=['viewing_angle', 'luminosity_distance'],
    ...
)

See examples/joint_grb_gw_example.py for a complete GW+EM example.
""")

print("\n" + "="*70)
print("Example complete!")
print("="*70)
print("\nTo run the actual fits, uncomment the fit_individual() and fit_joint() calls.")
print("Recommended settings for production: nlive=2000, walks=200")
