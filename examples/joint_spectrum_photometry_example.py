"""
Joint Spectrum and Photometry Fitting Example
==============================================

This example demonstrates how to use the MultiMessengerTransient class to jointly
fit spectroscopic and photometric data from the same transient event.

This is a common scenario in transient astronomy where you have:
1. Multi-band photometry (lightcurves) over extended time
2. One or more spectra taken at specific epochs

By fitting them jointly, physical parameters can be better constrained as:
- Photometry constrains time evolution and integrated properties
- Spectra constrain detailed spectral properties and composition

We'll simulate a kilonova with:
- Multi-band optical photometry (ugriz bands)
- A spectrum at ~3 days post-merger
"""

import numpy as np
import bilby
import redback
from redback.multimessenger import MultiMessengerTransient
from redback.transient_models import kilonova_models, spectral_models
from redback.transient import Transient, Spectrum

# Set random seed for reproducibility
np.random.seed(123)

print("="*70)
print("JOINT SPECTRUM + PHOTOMETRY FITTING EXAMPLE")
print("="*70)

# ============================================================================
# Step 1: Define true parameters and simulate observations
# ============================================================================

print("\n1. Simulating kilonova observations...")

# True physical parameters (shared between spectrum and photometry)
true_params = {
    'redshift': 0.01,
    'mej': 0.05,      # ejecta mass (solar masses)
    'vej': 0.2,       # ejecta velocity (c)
    'kappa': 3.0,     # opacity (cm^2/g)
    'temperature_floor': 2000,  # minimum temperature (K)
}

# For the spectrum, we'll also need spectral-specific parameters
true_spectrum_params = {
    'temperature': 6000,  # photospheric temperature at spectrum time
    'r_phot': 3e14,       # photospheric radius (cm)
    **true_params
}

# ============================================================================
# Simulate photometry (multi-band lightcurve)
# ============================================================================

print("  - Simulating multi-band photometry...")

# Time array for photometry (0.5 to 15 days)
phot_times = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15])

# We'll simulate observations in 5 bands (u, g, r, i, z)
bands_list = ['sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz']
all_times = []
all_bands = []
all_mags = []
all_mag_errs = []

for band in bands_list:
    # Simulate observations in this band
    band_array = np.array([band] * len(phot_times))

    # Generate true magnitudes using a simple kilonova model
    model_kwargs = {
        'output_format': 'magnitude',
        'bands': band_array,
        'frequency': None
    }

    true_mags = kilonova_models.arnett_bolometric(
        phot_times,
        **true_params,
        **model_kwargs
    )

    # Add realistic noise
    mag_errs = np.random.uniform(0.05, 0.15, len(phot_times))
    observed_mags = np.random.normal(true_mags, mag_errs)

    all_times.extend(phot_times)
    all_bands.extend(band_array)
    all_mags.extend(observed_mags)
    all_mag_errs.extend(mag_errs)

# Convert to numpy arrays
all_times = np.array(all_times)
all_bands = np.array(all_bands)
all_mags = np.array(all_mags)
all_mag_errs = np.array(all_mag_errs)

# Create photometry transient object
photometry_transient = Transient(
    name='kilonova_photometry',
    time=all_times,
    magnitude=all_mags,
    magnitude_err=all_mag_errs,
    bands=all_bands,
    data_mode='magnitude',
    redshift=true_params['redshift']
)

print(f"    ✓ Created photometry with {len(all_times)} observations in {len(bands_list)} bands")

# ============================================================================
# Simulate spectrum at a specific epoch
# ============================================================================

print("  - Simulating spectrum at ~3 days...")

# Wavelength array for spectrum (3000 - 10000 Angstroms)
wavelengths = np.linspace(3000, 10000, 200)

# Simulate spectrum using a blackbody model
# (In reality, you'd use a more sophisticated model with line features)
true_spectrum_flux = spectral_models.blackbody_spectrum(
    wavelengths,
    redshift=true_spectrum_params['redshift'],
    r_phot=true_spectrum_params['r_phot'],
    temperature=true_spectrum_params['temperature']
)

# Add noise to spectrum
spectrum_snr = 20
spectrum_flux_err = true_spectrum_flux / spectrum_snr
observed_spectrum_flux = np.random.normal(true_spectrum_flux, spectrum_flux_err)

# Create spectrum object
spectrum_epoch = 3.0  # days
spectrum_transient = Spectrum(
    angstroms=wavelengths,
    flux_density=observed_spectrum_flux,
    flux_density_err=spectrum_flux_err,
    time=f"{spectrum_epoch} days",
    name='kilonova_spectrum_3d'
)

print(f"    ✓ Created spectrum at {spectrum_epoch} days with {len(wavelengths)} wavelength points")

# ============================================================================
# Step 2: Create MultiMessengerTransient object
# ============================================================================

print("\n2. Creating MultiMessengerTransient object...")

# We'll treat photometry and spectrum as separate "messengers"
# For this, we'll use custom likelihoods since they're different data types

# Build photometry likelihood
phot_likelihood = redback.likelihoods.GaussianLikelihood(
    x=photometry_transient.time,
    y=photometry_transient.magnitude,
    sigma=photometry_transient.magnitude_err,
    function=kilonova_models.arnett_bolometric,
    kwargs={'output_format': 'magnitude',
            'bands': photometry_transient.bands,
            'frequency': None}
)

# Build spectrum likelihood
spec_likelihood = redback.likelihoods.GaussianLikelihood(
    x=spectrum_transient.angstroms,
    y=spectrum_transient.flux_density,
    sigma=spectrum_transient.flux_density_err,
    function=spectral_models.blackbody_spectrum,
    kwargs={}
)

# Create MultiMessengerTransient with custom likelihoods
mm_transient = MultiMessengerTransient(
    custom_likelihoods={
        'photometry': phot_likelihood,
        'spectrum': spec_likelihood
    },
    name='joint_spectrum_photometry'
)

print(f"  {mm_transient}")

# ============================================================================
# Step 3: Set up priors for joint analysis
# ============================================================================

print("\n3. Setting up priors for joint analysis...")

# Create prior dictionary with both photometry and spectrum parameters
priors = bilby.core.prior.PriorDict()

# Shared parameters (constrained by both photometry and spectrum)
priors['redshift'] = true_params['redshift']  # Fixed (assumed known)
priors['mej'] = bilby.core.prior.Uniform(0.01, 0.1, 'mej',
                                         latex_label=r'$M_{\rm ej}$ [$M_\odot$]')
priors['vej'] = bilby.core.prior.Uniform(0.1, 0.3, 'vej',
                                         latex_label=r'$v_{\rm ej}$ [c]')
priors['kappa'] = bilby.core.prior.Uniform(0.5, 10.0, 'kappa',
                                           latex_label=r'$\kappa$ [cm$^2$/g]')
priors['temperature_floor'] = bilby.core.prior.Uniform(1000, 5000, 'temperature_floor',
                                                       latex_label=r'$T_{\rm floor}$ [K]')

# Spectrum-specific parameters (only constrained by spectrum)
priors['temperature'] = bilby.core.prior.Uniform(3000, 10000, 'temperature',
                                                 latex_label=r'$T_{\rm phot}$ [K]')
priors['r_phot'] = bilby.core.prior.LogUniform(1e13, 1e15, 'r_phot',
                                               latex_label=r'$R_{\rm phot}$ [cm]')

print("  ✓ Priors configured")
print(f"    - Shared parameters: redshift, mej, vej, kappa, temperature_floor")
print(f"    - Spectrum-only parameters: temperature, r_phot")

# ============================================================================
# Step 4: Fit photometry and spectrum individually (for comparison)
# ============================================================================

print("\n" + "="*70)
print("INDIVIDUAL FITS (for comparison)")
print("="*70)

print("\nFitting photometry alone...")
print("  (Using low nlive for speed - increase for production)")

# Photometry-only priors
phot_priors = bilby.core.prior.PriorDict()
phot_priors['redshift'] = true_params['redshift']
phot_priors['mej'] = priors['mej']
phot_priors['vej'] = priors['vej']
phot_priors['kappa'] = priors['kappa']
phot_priors['temperature_floor'] = priors['temperature_floor']

# Uncomment to run photometry-only fit
# phot_result = bilby.run_sampler(
#     likelihood=phot_likelihood,
#     priors=phot_priors,
#     sampler='dynesty',
#     nlive=500,
#     outdir='./outdir_photometry_only',
#     label='photometry_only',
#     resume=True
# )

print("  (Photometry-only fit commented out for speed)")

print("\nFitting spectrum alone...")

# Spectrum-only priors
spec_priors = bilby.core.prior.PriorDict()
spec_priors['redshift'] = true_params['redshift']
spec_priors['temperature'] = priors['temperature']
spec_priors['r_phot'] = priors['r_phot']

# Uncomment to run spectrum-only fit
# spec_result = bilby.run_sampler(
#     likelihood=spec_likelihood,
#     priors=spec_priors,
#     sampler='dynesty',
#     nlive=500,
#     outdir='./outdir_spectrum_only',
#     label='spectrum_only',
#     resume=True
# )

print("  (Spectrum-only fit commented out for speed)")

# ============================================================================
# Step 5: Joint fit of photometry + spectrum
# ============================================================================

print("\n" + "="*70)
print("JOINT PHOTOMETRY + SPECTRUM FIT")
print("="*70)

print("\nNote: In this joint fit:")
print("  - Photometry constrains ejecta mass, velocity, and opacity")
print("  - Spectrum at t=3d constrains temperature and photosphere size")
print("  - Shared 'redshift' ensures consistency")
print("  - Joint constraints are tighter than individual fits")

print("\nStarting joint analysis...")
print("  (Using low nlive for speed - increase for production)")

# For the joint fit, we need to ensure parameter names don't conflict
# Since we're using custom likelihoods, bilby will combine them automatically

# Uncomment to run joint fit
# joint_result = bilby.run_sampler(
#     likelihood=bilby.core.likelihood.JointLikelihood(phot_likelihood, spec_likelihood),
#     priors=priors,
#     sampler='dynesty',
#     nlive=1000,  # Increase for production (>= 2000)
#     walks=100,
#     outdir='./outdir_joint_spec_phot',
#     label='joint_spectrum_photometry',
#     resume=True,
#     plot=True
# )

print("  (Joint fit commented out for speed - uncomment to run)")

# ============================================================================
# Alternative: Using MultiMessengerTransient.fit_joint()
# ============================================================================

print("\n" + "="*70)
print("ALTERNATIVE: Using MultiMessengerTransient.fit_joint()")
print("="*70)

print("""
Since we've already constructed the likelihoods and added them as
custom_likelihoods, we cannot use fit_joint() with models directly.

However, for a cleaner workflow when starting from scratch, you could:

1. Store photometry and spectrum as separate transient objects
2. Pass them to MultiMessengerTransient
3. Use fit_joint() with appropriate model wrappers

Example workflow:
-----------------

# Define model wrappers that handle parameter mapping
def photometry_model(time, mej, vej, kappa, temperature_floor, **kwargs):
    return kilonova_models.arnett_bolometric(
        time, mej=mej, vej=vej, kappa=kappa,
        temperature_floor=temperature_floor, **kwargs
    )

def spectrum_model(wavelength, temperature, r_phot, redshift, **kwargs):
    return spectral_models.blackbody_spectrum(
        wavelength, temperature=temperature,
        r_phot=r_phot, redshift=redshift, **kwargs
    )

# Note: Currently MultiMessengerTransient expects Transient objects for
# photometry, but Spectrum is a different class. For true integration,
# you would either:
# a) Use custom likelihoods (as we did above)
# b) Extend MultiMessengerTransient to handle Spectrum objects
# c) Convert spectrum to pseudo-transient format

For now, the custom likelihood approach shown above is most flexible.
""")

# ============================================================================
# Step 6: Analyzing results
# ============================================================================

print("\n" + "="*70)
print("ANALYZING RESULTS")
print("="*70)

print("""
After running the fits, you can compare:

1. Parameter constraints:
   Individual fits:
   - Photometry alone: weak constraints on temperature/r_phot
   - Spectrum alone: no constraints on time evolution (mej, vej)

   Joint fit:
   - All parameters constrained by complementary information
   - Degeneracies broken by combining data types

2. Plot photometry fit:
   import redback.analysis
   redback.analysis.plot_lightcurve(
       transient=photometry_transient,
       parameters=joint_result.posterior.sample(100),
       model=kilonova_models.arnett_bolometric,
       model_kwargs={'output_format': 'magnitude',
                     'bands': photometry_transient.bands}
   )

3. Plot spectrum fit:
   joint_result.plot_spectrum(model=spectral_models.blackbody_spectrum)

4. Corner plot comparing individual vs. joint:
   # Plot all three results together
   from bilby.core.result import make_pp_plot
   results = [phot_result, spec_result, joint_result]
   labels = ['Photometry only', 'Spectrum only', 'Joint']

   # Compare posteriors
   import corner
   # ... create comparison corner plots ...

5. Evidence comparison:
   print(f"Photometry ln(Z): {phot_result.log_evidence}")
   print(f"Spectrum ln(Z): {spec_result.log_evidence}")
   print(f"Joint ln(Z): {joint_result.log_evidence}")

   # If models are independent, joint evidence should be approximately
   # the sum of individual evidences (if parameters are shared, this
   # provides additional constraints and may increase evidence)
""")

# ============================================================================
# Best Practices for Spectrum + Photometry Joint Fitting
# ============================================================================

print("\n" + "="*70)
print("BEST PRACTICES")
print("="*70)

print("""
1. Time synchronization:
   - Ensure spectrum epoch aligns with photometry time grid
   - Account for any time delays or offsets
   - Use consistent time reference (e.g., explosion time, trigger time)

2. Wavelength/band consistency:
   - Verify spectrum wavelength range covers photometric bands
   - Check that models consistently handle both data types
   - Consider filter transmission functions

3. Model selection:
   - Use physically motivated models that predict both SED and evolution
   - For simple cases: blackbody + light curve parameterization
   - For detailed cases: full radiative transfer models

4. Parameter sharing:
   - Share physical parameters (mass, velocity, composition)
   - Keep epoch-dependent parameters separate (temperature, radius at t_spec)
   - Consider time-dependent relations (e.g., T ~ t^(-a))

5. Multiple spectra:
   - If you have spectra at multiple epochs, add each as a separate likelihood
   - Share underlying physical parameters
   - Allow epoch-dependent parameters to vary (e.g., temperature_1, temperature_2)

6. Systematic uncertainties:
   - Include calibration uncertainties in photometry
   - Account for flux calibration uncertainties in spectra
   - Consider extinction/reddening

7. Model complexity:
   - Start with simple models (single blackbody, simple LC)
   - Add complexity as justified by data quality
   - Use model comparison (Bayes factors) to assess improvements
""")

# ============================================================================
# Example with multiple spectra
# ============================================================================

print("\n" + "="*70)
print("EXTENSION: Multiple Spectra at Different Epochs")
print("="*70)

print("""
If you have spectra at multiple epochs (e.g., t=1d, 3d, 7d), you can:

# Create spectrum objects for each epoch
spectrum_1d = Spectrum(wavelengths, flux_1d, flux_err_1d, time='1 day')
spectrum_3d = Spectrum(wavelengths, flux_3d, flux_err_3d, time='3 days')
spectrum_7d = Spectrum(wavelengths, flux_7d, flux_err_7d, time='7 days')

# Create likelihoods with epoch-dependent parameters
spec_1d_likelihood = redback.likelihoods.GaussianLikelihood(
    x=spectrum_1d.angstroms, y=spectrum_1d.flux_density,
    sigma=spectrum_1d.flux_density_err,
    function=lambda wave, temperature_1d, r_phot_1d, **kw:
        spectral_models.blackbody_spectrum(
            wave, temperature=temperature_1d, r_phot=r_phot_1d, **kw
        )
)

# Similar for 3d and 7d...

# Add all to MultiMessengerTransient
mm_transient = MultiMessengerTransient(
    custom_likelihoods={
        'photometry': phot_likelihood,
        'spectrum_1d': spec_1d_likelihood,
        'spectrum_3d': spec_3d_likelihood,
        'spectrum_7d': spec_7d_likelihood
    }
)

# Set up priors with epoch-dependent parameters
priors = bilby.core.prior.PriorDict()
priors['mej'] = ...  # Shared
priors['vej'] = ...  # Shared
priors['temperature_1d'] = ...  # Epoch-specific
priors['temperature_3d'] = ...  # Epoch-specific
priors['temperature_7d'] = ...  # Epoch-specific
# etc.

# Run joint fit
result = bilby.run_sampler(
    likelihood=bilby.core.likelihood.JointLikelihood(
        phot_likelihood, spec_1d_likelihood,
        spec_3d_likelihood, spec_7d_likelihood
    ),
    priors=priors,
    ...
)
""")

print("\n" + "="*70)
print("Example complete!")
print("="*70)
print("\nTo run the actual fits, uncomment the fit calls above.")
print("Recommended settings for production:")
print("  - nlive >= 2000 (nested sampling)")
print("  - Check convergence with evidence estimation errors")
print("  - Run multiple times with different seeds to verify stability")
