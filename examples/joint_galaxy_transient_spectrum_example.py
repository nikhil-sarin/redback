"""
Joint Galaxy and Transient Spectrum Fitting Example
====================================================

This example demonstrates how to simultaneously fit both a transient spectrum
and its host galaxy spectrum. This is crucial when the transient is embedded
in a bright host galaxy, as the galaxy contribution can significantly affect
parameter inference.

Applications:
- Supernovae in bright host galaxies
- Tidal disruption events (TDEs)
- Active galactic nuclei (AGN) outbursts
- Any transient with significant host contamination

By jointly fitting both components, we can:
1. Properly account for host galaxy contamination
2. Use galaxy redshift to constrain transient
3. Separate transient flux from galaxy flux
4. Account for galaxy extinction effects
"""

import numpy as np
import bilby
import redback
from redback.multimessenger import MultiMessengerTransient
from redback.transient_models import spectral_models
from redback.transient import Spectrum

# Set random seed for reproducibility
np.random.seed(456)

print("="*70)
print("JOINT GALAXY + TRANSIENT SPECTRUM FITTING")
print("="*70)

# ============================================================================
# Step 1: Define component models
# ============================================================================

print("\n1. Setting up galaxy and transient models...")

def galaxy_spectrum_model(wavelength, redshift, galaxy_temperature,
                         galaxy_luminosity, **kwargs):
    """
    Simple galaxy spectrum model (stellar continuum).
    In reality, you'd use more sophisticated models with emission lines.
    """
    # Simple blackbody for stellar continuum
    # Typical galaxy: T ~ 5000-6000 K (solar-like stars dominate)

    # Luminosity in erg/s, convert to flux at distance
    # For this example, we'll use a simple blackbody scaled by luminosity
    from redback.constants import speed_of_light as c
    from redback.constants import h_planck as h
    from redback.constants import k_B

    wave_rest = wavelength / (1 + redshift)

    # Planck function
    numerator = 2 * h * c**2 / (wave_rest * 1e-8)**5
    exponent = h * c / (wave_rest * 1e-8 * k_B * galaxy_temperature)
    planck = numerator / (np.exp(exponent) - 1)

    # Scale by luminosity (simplified)
    flux = planck * galaxy_luminosity * 1e-40  # Arbitrary scaling for example

    return flux

def transient_spectrum_model(wavelength, redshift, transient_temperature,
                             transient_r_phot, **kwargs):
    """
    Transient spectrum model (e.g., supernova photosphere).
    """
    return spectral_models.blackbody_spectrum(
        wavelength,
        redshift=redshift,
        temperature=transient_temperature,
        r_phot=transient_r_phot
    )

def combined_galaxy_transient_model(wavelength, redshift,
                                   galaxy_temperature, galaxy_luminosity,
                                   transient_temperature, transient_r_phot,
                                   **kwargs):
    """
    Combined model: galaxy + transient.
    The observed spectrum is the sum of both components.
    """
    galaxy_flux = galaxy_spectrum_model(
        wavelength, redshift, galaxy_temperature, galaxy_luminosity
    )

    transient_flux = transient_spectrum_model(
        wavelength, redshift, transient_temperature, transient_r_phot
    )

    return galaxy_flux + transient_flux

print("  ✓ Models defined:")
print("    - Galaxy: Stellar continuum (blackbody)")
print("    - Transient: Photospheric emission (blackbody)")
print("    - Combined: Galaxy + Transient")

# ============================================================================
# Step 2: Simulate observed spectrum (galaxy + transient)
# ============================================================================

print("\n2. Simulating observed spectrum...")

# True parameters
true_params = {
    'redshift': 0.05,
    # Galaxy parameters
    'galaxy_temperature': 5500,    # K (solar-like)
    'galaxy_luminosity': 3.0,      # Arbitrary units
    # Transient parameters (e.g., supernova a few days after peak)
    'transient_temperature': 8000,  # K (hotter than galaxy)
    'transient_r_phot': 5e14,      # cm (photosphere size)
}

# Wavelength range (optical: 3500-9000 Angstroms)
wavelengths = np.linspace(3500, 9000, 150)

# Generate true combined spectrum
true_galaxy_flux = galaxy_spectrum_model(
    wavelengths,
    true_params['redshift'],
    true_params['galaxy_temperature'],
    true_params['galaxy_luminosity']
)

true_transient_flux = transient_spectrum_model(
    wavelengths,
    true_params['redshift'],
    true_params['transient_temperature'],
    true_params['transient_r_phot']
)

true_combined_flux = true_galaxy_flux + true_transient_flux

# Add realistic noise
spectrum_snr = 30  # Signal-to-noise ratio
flux_err = true_combined_flux / spectrum_snr
observed_flux = np.random.normal(true_combined_flux, flux_err)

# Create spectrum object
spectrum_obs = Spectrum(
    angstroms=wavelengths,
    flux_density=observed_flux,
    flux_density_err=flux_err,
    time='Observation epoch',
    name='galaxy_plus_transient'
)

print(f"  ✓ Simulated spectrum:")
print(f"    - Wavelength range: {wavelengths.min():.0f}-{wavelengths.max():.0f} Å")
print(f"    - SNR: {spectrum_snr}")
print(f"    - Galaxy contribution: {true_galaxy_flux.mean():.2e} erg/s/cm²/Å")
print(f"    - Transient contribution: {true_transient_flux.mean():.2e} erg/s/cm²/Å")
print(f"    - Ratio (transient/galaxy): {true_transient_flux.mean()/true_galaxy_flux.mean():.2f}")

# ============================================================================
# Step 3: Set up joint fitting
# ============================================================================

print("\n3. Setting up joint galaxy + transient fit...")

# Create likelihood for combined model
combined_likelihood = redback.likelihoods.GaussianLikelihood(
    x=spectrum_obs.angstroms,
    y=spectrum_obs.flux_density,
    sigma=spectrum_obs.flux_density_err,
    function=combined_galaxy_transient_model,
    kwargs={}
)

# Set up priors
priors = bilby.core.prior.PriorDict()

# Shared parameter: redshift (known from galaxy)
# In real analysis, might have prior from galaxy spectroscopy
priors['redshift'] = bilby.core.prior.Gaussian(
    0.05, 0.001, 'redshift',
    latex_label=r'$z$'
)

# Galaxy parameters
priors['galaxy_temperature'] = bilby.core.prior.Uniform(
    4000, 7000, 'galaxy_temperature',
    latex_label=r'$T_{\rm gal}$ [K]'
)
priors['galaxy_luminosity'] = bilby.core.prior.Uniform(
    0.5, 10.0, 'galaxy_luminosity',
    latex_label=r'$L_{\rm gal}$'
)

# Transient parameters
priors['transient_temperature'] = bilby.core.prior.Uniform(
    5000, 15000, 'transient_temperature',
    latex_label=r'$T_{\rm SN}$ [K]'
)
priors['transient_r_phot'] = bilby.core.prior.LogUniform(
    1e14, 1e16, 'transient_r_phot',
    latex_label=r'$R_{\rm phot}$ [cm]'
)

print("  ✓ Priors configured:")
print("    - Shared: redshift (Gaussian from galaxy)")
print("    - Galaxy: temperature, luminosity")
print("    - Transient: temperature, photosphere radius")

# ============================================================================
# Step 4: Comparison - fit with and without galaxy model
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: With vs. Without Galaxy Model")
print("="*70)

# ============================================================================
# Fit A: Transient only (WRONG - ignores galaxy)
# ============================================================================

print("\nFit A: Transient-only model (incorrect)")
print("  This ignores the galaxy contribution and will give biased results")

transient_only_likelihood = redback.likelihoods.GaussianLikelihood(
    x=spectrum_obs.angstroms,
    y=spectrum_obs.flux_density,
    sigma=spectrum_obs.flux_density_err,
    function=transient_spectrum_model,
    kwargs={}
)

transient_only_priors = bilby.core.prior.PriorDict()
transient_only_priors['redshift'] = priors['redshift']
transient_only_priors['transient_temperature'] = priors['transient_temperature']
transient_only_priors['transient_r_phot'] = priors['transient_r_phot']

# Uncomment to run
# print("  Running sampler (transient-only)...")
# result_transient_only = bilby.run_sampler(
#     likelihood=transient_only_likelihood,
#     priors=transient_only_priors,
#     sampler='dynesty',
#     nlive=500,
#     outdir='./outdir_transient_only',
#     label='transient_only_wrong',
#     resume=True
# )

print("  (Fit commented out for speed - will show biased parameters)")

# ============================================================================
# Fit B: Galaxy + Transient (CORRECT)
# ============================================================================

print("\nFit B: Joint galaxy + transient model (correct)")
print("  This properly accounts for both components")

# Uncomment to run
# print("  Running sampler (joint fit)...")
# result_joint = bilby.run_sampler(
#     likelihood=combined_likelihood,
#     priors=priors,
#     sampler='dynesty',
#     nlive=1000,
#     outdir='./outdir_joint_galaxy_transient',
#     label='joint_galaxy_transient',
#     resume=True,
#     injection_parameters=true_params
# )

print("  (Fit commented out for speed)")

# ============================================================================
# Step 5: Using MultiMessengerTransient framework
# ============================================================================

print("\n" + "="*70)
print("USING MULTIMESSENGER FRAMEWORK")
print("="*70)

print("""
While the above approach uses a combined model function, you can also
use the MultiMessengerTransient framework to treat galaxy and transient
as separate "messengers":

# Create separate likelihoods
galaxy_likelihood = redback.likelihoods.GaussianLikelihood(
    x=wavelengths,
    y=galaxy_flux_estimate,  # Initial estimate from spectrum decomposition
    sigma=galaxy_flux_err,
    function=galaxy_spectrum_model,
    kwargs={}
)

transient_likelihood = redback.likelihoods.GaussianLikelihood(
    x=wavelengths,
    y=transient_flux_estimate,  # Initial estimate
    sigma=transient_flux_err,
    function=transient_spectrum_model,
    kwargs={}
)

# Use MultiMessengerTransient
mm_transient = MultiMessengerTransient(
    custom_likelihoods={
        'galaxy': galaxy_likelihood,
        'transient': transient_likelihood
    },
    name='galaxy_transient_decomposition'
)

However, this requires prior knowledge or decomposition of the spectrum
into galaxy and transient components. The combined model approach shown
above is more straightforward when both components overlap spectrally.
""")

# ============================================================================
# Step 6: Advanced - Including emission lines
# ============================================================================

print("\n" + "="*70)
print("ADVANCED: Including Galaxy Emission Lines")
print("="*70)

print("""
For more realistic galaxy modeling, include emission lines:

def galaxy_with_lines_model(wavelength, redshift,
                            galaxy_temperature, galaxy_luminosity,
                            h_alpha_flux, h_beta_flux, oiii_flux,
                            **kwargs):
    '''Galaxy model with stellar continuum + emission lines'''

    # Stellar continuum
    continuum = galaxy_spectrum_model(
        wavelength, redshift, galaxy_temperature, galaxy_luminosity
    )

    # Add emission lines (Gaussian profiles)
    # H-alpha at 6563 Å
    h_alpha = gaussian_line(wavelength, 6563 * (1 + redshift),
                           h_alpha_flux, width=3.0)

    # H-beta at 4861 Å
    h_beta = gaussian_line(wavelength, 4861 * (1 + redshift),
                          h_beta_flux, width=3.0)

    # [OIII] at 5007 Å
    oiii = gaussian_line(wavelength, 5007 * (1 + redshift),
                        oiii_flux, width=2.0)

    return continuum + h_alpha + h_beta + oiii

def gaussian_line(wavelength, line_center, amplitude, width):
    '''Gaussian emission line profile'''
    return amplitude * np.exp(-0.5 * ((wavelength - line_center) / width)**2)

# Add emission line parameters to priors
priors['h_alpha_flux'] = bilby.core.prior.LogUniform(1e-17, 1e-15)
priors['h_beta_flux'] = bilby.core.prior.LogUniform(1e-18, 1e-16)
priors['oiii_flux'] = bilby.core.prior.LogUniform(1e-18, 1e-16)
""")

# ============================================================================
# Step 7: Best practices
# ============================================================================

print("\n" + "="*70)
print("BEST PRACTICES")
print("="*70)

print("""
1. Pre-processing:
   - If possible, obtain pre-explosion galaxy spectrum for reference
   - Use galaxy spectrum to constrain stellar population, emission lines
   - Subtract galaxy template if available (but beware of systematic errors)

2. Model selection:
   - Start with simple continuum models (blackbody, power law)
   - Add emission/absorption lines as needed
   - Use stellar population synthesis models for galaxy (e.g., FSPS, BC03)
   - Use radiative transfer for transient (e.g., TARDIS, CMFGEN)

3. Parameter constraints:
   - Constrain galaxy parameters from broader wavelength coverage if available
   - Use galaxy redshift as strong prior (typically known accurately)
   - Consider fixing some galaxy parameters if pre-explosion data exists

4. Systematic uncertainties:
   - Account for flux calibration differences between epochs
   - Consider spatial aperture effects (galaxy vs. transient location)
   - Include extinction: both Galactic and host galaxy
   - Check for variability in AGN contribution (if present)

5. Validation:
   - Compare to pre-explosion imaging/spectroscopy
   - Check that galaxy parameters are physically reasonable
   - Verify transient parameters match photometric evolution
   - Use multiple spectra at different epochs to check consistency

6. Alternative approaches:
   - Spectral decomposition: Use tools like STARLIGHT or pPXF first
   - Template subtraction: Subtract scaled galaxy template
   - Spatial separation: If possible, extract spatially separated spectra

7. When to use joint fitting:
   - Transient is bright relative to galaxy (SNR > 10)
   - No pre-explosion spectrum available
   - Transient and galaxy spectra overlap significantly
   - Need self-consistent model for both components
""")

# ============================================================================
# Step 8: Example analysis output
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE ANALYSIS WORKFLOW")
print("="*70)

print("""
After running the fits, compare results:

1. Evidence comparison:
   print(f"Transient-only ln(Z): {result_transient_only.log_evidence:.2f}")
   print(f"Joint model ln(Z): {result_joint.log_evidence:.2f}")
   print(f"Bayes factor: {result_joint.log_evidence - result_transient_only.log_evidence:.2f}")

   # Joint model should have much higher evidence if galaxy is significant

2. Parameter recovery:
   # Check if transient parameters are recovered correctly
   fig = result_joint.plot_corner(
       parameters=['transient_temperature', 'transient_r_phot'],
       truths=true_params
   )

3. Visualize decomposition:
   # Plot the best-fit decomposition
   import matplotlib.pyplot as plt

   best_params = result_joint.posterior.iloc[result_joint.posterior['log_likelihood'].idxmax()]

   galaxy_best = galaxy_spectrum_model(wavelengths, **best_params)
   transient_best = transient_spectrum_model(wavelengths, **best_params)
   combined_best = galaxy_best + transient_best

   plt.figure(figsize=(12, 6))
   plt.errorbar(wavelengths, observed_flux, flux_err,
                fmt='o', alpha=0.3, label='Observed')
   plt.plot(wavelengths, combined_best, 'k-', lw=2, label='Best fit (total)')
   plt.plot(wavelengths, galaxy_best, '--', lw=2, label='Galaxy component')
   plt.plot(wavelengths, transient_best, '--', lw=2, label='Transient component')
   plt.xlabel('Wavelength [Å]')
   plt.ylabel('Flux density')
   plt.legend()
   plt.title('Spectral Decomposition: Galaxy + Transient')

4. Compare biased vs. unbiased results:
   # Show how ignoring galaxy biases transient parameters

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Temperature comparison
   axes[0].hist(result_transient_only.posterior['transient_temperature'],
                bins=30, alpha=0.5, label='Transient-only (biased)')
   axes[0].hist(result_joint.posterior['transient_temperature'],
                bins=30, alpha=0.5, label='Joint (correct)')
   axes[0].axvline(true_params['transient_temperature'],
                   color='k', ls='--', label='True value')
   axes[0].set_xlabel('Transient Temperature [K]')
   axes[0].legend()

   # Radius comparison
   axes[1].hist(np.log10(result_transient_only.posterior['transient_r_phot']),
                bins=30, alpha=0.5, label='Transient-only (biased)')
   axes[1].hist(np.log10(result_joint.posterior['transient_r_phot']),
                bins=30, alpha=0.5, label='Joint (correct)')
   axes[1].axvline(np.log10(true_params['transient_r_phot']),
                   color='k', ls='--', label='True value')
   axes[1].set_xlabel('log Photosphere Radius [cm]')
   axes[1].legend()
""")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
Key points:

1. Host galaxy contamination is crucial for accurate transient spectroscopy
2. Joint fitting allows proper separation of galaxy and transient components
3. Ignoring galaxy leads to biased transient parameters
4. Shared redshift provides strong constraint across components
5. Pre-explosion data is invaluable for constraining galaxy model

When to use this approach:
- Supernova spectroscopy (especially Type Ia in bright galaxies)
- TDEs (distinguishing transient from host AGN)
- Kilonovae (though usually fainter hosts)
- Any transient where host contamination is >10% of total flux

Alternative approaches:
- Template subtraction (if pre-explosion spectrum available)
- Spectral decomposition tools (STARLIGHT, pPXF, etc.)
- Spatial separation (if PSF allows)

The joint fitting approach shown here is most useful when:
- No pre-explosion data available
- Need physical model for both components
- Want to propagate uncertainties properly
- Have time-series spectra to constrain evolution
""")

print("\n" + "="*70)
print("Example complete!")
print("="*70)
print("\nTo run actual fits, uncomment the bilby.run_sampler() calls.")
print("Recommended: nlive >= 1000, check convergence")
print("\nFor production analysis:")
print("  - Use realistic galaxy models (stellar population synthesis)")
print("  - Include emission/absorption lines")
print("  - Account for extinction (Galactic + host)")
print("  - Validate with pre-explosion data if available")
