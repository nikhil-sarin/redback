"""
Example: Using Spectral Line Profile Models and Velocity Fitting Tools

This example demonstrates how to:
1. Generate P-Cygni line profiles for supernova spectra
2. Create Voigt absorption profiles
3. Measure expansion velocities from spectral features
4. Track photospheric velocity evolution
5. Explore a SYNOW-style parameterized line profile

These tools are particularly useful for analyzing thermal transients,
but you can adapt them for other sources with expanding atmospheres.

What you will see:
1) Two single-line P-Cygni profiles (Sobolev vs. simple Gaussian).
2) A toy Type Ia spectrum built from multiple P-Cygni lines.
3) Velocity measurements and a time-evolution demo.
4) A SYNOW-style parameterized line profile and how its parameters change shape.
"""

import numpy as np
import matplotlib.pyplot as plt
import redback
from redback.transient_models import spectral_models
from redback.analysis import SpectralVelocityFitter

# =============================================================================
# Example 1: Generate a single P-Cygni profile
# =============================================================================
print("Example 1: Single P-Cygni Profile")
print("=" * 50)

# Wavelength array for Si II 6355 region
wavelength = np.linspace(5800, 6800, 1000)

# Generate P-Cygni profiles with:
# - Rest wavelength: 6355 Angstroms (Si II)
# - Optical depth: 3.0
# - Photospheric velocity: 11,000 km/s
flux_pcygni_sobolev = spectral_models.p_cygni_profile(
    wavelength=wavelength,
    lambda_rest=6355,
    tau_sobolev=0.7,
    v_phot=10500,
    continuum_flux=1.0,
    source_function='scattering',
    emission_scale=0.10,
    v_max=25000
)

flux_pcygni_elementary = spectral_models.elementary_p_cygni_profile(
    wavelength=wavelength,
    lambda_rest=6355,
    v_absorption=10500,
    absorption_depth=0.28,
    emission_strength=0.18,
    v_width=1800
)

# Plot the profiles
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelength, flux_pcygni_sobolev, 'b-', linewidth=2, label='Sobolev')
ax.plot(wavelength, flux_pcygni_elementary, 'r--', linewidth=2, label='Elementary (Gaussian)')
lambda_abs = 6355 * (1 - 10500 / 299792.458)
ax.axvline(lambda_abs, color='gray', linestyle='--', alpha=0.5, label='Absorption minimum')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Continuum')
ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Normalized Flux', fontsize=12)
ax.set_title('P-Cygni Profile: Si II 6355 at v_phot = 11,000 km/s', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('pcygni_single_line.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pcygni_single_line.png")

# =============================================================================
# Example 2: Toy model Type Ia SN spectrum with multiple P-Cygni lines
# =============================================================================
print("\nExample 2: Multi-line Type Ia Supernova Spectrum")
print("=" * 50)

# Define typical Type Ia SN lines near maximum light
line_list = [
    {'ion': 'Ca II H&K', 'lambda': 3945, 'tau': 1.0},
    {'ion': 'Si II 4130', 'lambda': 4130, 'tau': 0.4},
    {'ion': 'Mg II', 'lambda': 4481, 'tau': 0.3},
    {'ion': 'Fe III', 'lambda': 5129, 'tau': 0.3},
    {'ion': 'Fe III', 'lambda': 5156, 'tau': 0.3},
    {'ion': 'S II W', 'lambda': 5454, 'tau': 0.5},
    {'ion': 'S II 5640', 'lambda': 5640, 'tau': 0.55},
    {'ion': 'Si II 5972', 'lambda': 5972, 'tau': 0.55},
    {'ion': 'Si II 6355', 'lambda': 6355, 'tau': 0.6},
    {'ion': 'O I', 'lambda': 7774, 'tau': 0.45},
    {'ion': 'Ca II IR', 'lambda': 8579, 'tau': 0.5},
]

# Full optical wavelength range
wavelength_full = np.linspace(3500, 9000, 3000)

# Generate spectrum with blackbody continuum + P-Cygni lines
spectrum_ia = spectral_models.blackbody_spectrum_with_p_cygni_lines(
    angstroms=wavelength_full,
    redshift=0.01,  # z = 0.01
    rph=1e15,       # Photosphere radius in cm
    temp=8000,      # Temperature in K (steeper blue-to-red decline)
    line_list=line_list,
    v_phot=11000,   # Photospheric velocity in km/s
    source_function='thermal',
    emission_scale=0.12,
    v_max=28000
)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
# Preserve the continuum slope; just scale to unit peak
# Normalize to ~1 near 5500 Å to mimic arbitrary-unit spectra
anchor_flux = np.interp(5500.0, wavelength_full, spectrum_ia)
spec_norm = spectrum_ia / anchor_flux
ax.plot(wavelength_full, spec_norm, 'b-', linewidth=1.5)

# Overplot a SALT2 spectrum at peak using redback's sncosmo interface
salt2_model = redback.transient_models.supernova_models.sncosmo_models(
    time=np.array([0.0]),
    redshift=0.01,
    model_kwargs={'x0': 1e-7, 'x1': 0.9, 'c': 0.0},
    sncosmo_model='salt2',
    peak_time=0.0,
    output_format='sncosmo_source',
    host_extinction=False,
    mw_extinction=False
)
salt2_flux = salt2_model.flux(0.0, wavelength_full)
salt2_anchor = np.interp(5500.0, wavelength_full, salt2_flux)
salt2_norm = salt2_flux / salt2_anchor
ax.plot(wavelength_full, salt2_norm, 'k--', linewidth=1.2, alpha=0.8, label='SALT2 (peak)')

# Mark major features
features = {
    'Ca II HK': 3945,
    'Si II 4130': 4130,
    'S II': 5454,
    'Si II 5972': 5972,
    'Si II 6355': 6355,
    'O I': 7774,
    'Ca II IR': 8579
}
for name, lam in features.items():
    lambda_abs = lam * (1 - 11000 / 299792.458) * 1.01  # blueshift + redshift
    ax.axvline(lambda_abs, color='red', linestyle='--', alpha=0.3)
    ax.text(lambda_abs, 0.95, name, rotation=90, fontsize=9, ha='right')

ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Normalized Flux', fontsize=12)
ax.set_title('Toy model Type Ia Supernova Spectrum', fontsize=14)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(3500, 9000)
ax.grid(True, alpha=0.3)
plt.savefig('type_ia_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: type_ia_spectrum.png")

# =============================================================================
# Example 3: Elementary P-Cygni Profile (Simple Parameterization)
# =============================================================================
print("\nExample 3: Elementary P-Cygni Profile")
print("=" * 50)

wavelength = np.linspace(6000, 6700, 1000)

# Compare physical and elementary models
flux_physical = spectral_models.p_cygni_profile(
    wavelength, lambda_rest=6355, tau_sobolev=3.0,
    v_phot=11000, continuum_flux=1.0, source_function='scattering', emission_scale=0.08, v_max=25000
)

flux_elementary = spectral_models.elementary_p_cygni_profile(
    wavelength, lambda_rest=6355,
    v_absorption=11000,
    absorption_depth=0.35,
    emission_strength=0.15,
    v_width=1500
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelength, flux_physical, 'b-', linewidth=2, label='Physical (Sobolev)')
ax.plot(wavelength, flux_elementary, 'r--', linewidth=2, label='Elementary (Gaussian)')
ax.axvline(6355, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Normalized Flux', fontsize=12)
ax.set_title('Comparison: Physical vs Elementary P-Cygni Models', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('pcygni_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pcygni_comparison.png")

# =============================================================================
# Example 4: Voigt Profile (Gaussian + Lorentzian)
# =============================================================================
print("\nExample 4: Voigt Line Profile")
print("=" * 50)

wavelength_narrow = np.linspace(6555, 6575, 500)

# Pure Gaussian
flux_gaussian = spectral_models.gaussian_line_profile(
    wavelength_narrow, lambda_center=6563, amplitude=-0.5, sigma=2.0, continuum=1.0
)

# Pure Lorentzian
flux_lorentzian = spectral_models.lorentzian_line_profile(
    wavelength_narrow, lambda_center=6563, amplitude=-0.5, gamma=2.0, continuum=1.0
)

# Voigt (combination)
flux_voigt = spectral_models.voigt_profile(
    wavelength_narrow, lambda_center=6563, amplitude=-0.5,
    sigma_gaussian=2.0, gamma_lorentz=1.0, continuum=1.0
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelength_narrow, flux_gaussian, 'b-', linewidth=2, label='Gaussian')
ax.plot(wavelength_narrow, flux_lorentzian, 'g-', linewidth=2, label='Lorentzian')
ax.plot(wavelength_narrow, flux_voigt, 'r--', linewidth=2, label='Voigt')
ax.axvline(6563, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Flux', fontsize=12)
ax.set_title('H-alpha Absorption: Gaussian, Lorentzian, and Voigt Profiles', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('voigt_profile_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: voigt_profile_comparison.png")

# =============================================================================
# Example 5: Measuring Line Velocities with SpectralVelocityFitter
# =============================================================================
print("\nExample 5: Measuring Photospheric Velocities")
print("=" * 50)

# Generate a mock Type Ia SN spectrum
wavelength = np.linspace(3500, 8000, 2000)
lines = [
    {'ion': 'Si II', 'lambda': 6355, 'tau': 1.3},
    {'ion': 'Ca II', 'lambda': 3945, 'tau': 2.2},
    {'ion': 'Fe II', 'lambda': 5169, 'tau': 1.1}
]
spectral_redshift = 0.01
spectrum = spectral_models.blackbody_spectrum_with_p_cygni_lines(
    wavelength, redshift=spectral_redshift, rph=1e15, temp=11000,
    line_list=lines, v_phot=11000, source_function='scattering', emission_scale=0.08, v_max=25000
)

# Add some noise
np.random.seed(42)
noise = 0.02 * np.max(spectrum) * np.random.randn(len(spectrum))
spectrum_noisy = np.nan_to_num(spectrum + noise)
spectrum_noisy = spectrum_noisy / np.percentile(spectrum_noisy, 95)

# Create velocity fitter
fitter = SpectralVelocityFitter(wavelength, spectrum_noisy)

# Measure Si II 6355 velocity using different methods
print("Si II 6355 Velocity Measurements:")
for method in ['min', 'centroid', 'gaussian']:
    v, verr = fitter.measure_line_velocity(6355, method=method)
    print(f"  {method:10s}: {v:.0f} +/- {verr:.0f} km/s")

# Measure multiple lines at once
line_dict = {
    'Si II 6355': 6355,
    'Ca II HK': 3945,
    'Fe II 5169': 5169
}

print("\nAll Line Velocities:")
velocities = fitter.measure_multiple_lines(line_dict, method='min')
for ion_name, (v, verr) in velocities.items():
    print(f"  {ion_name}: {v:.0f} +/- {verr:.0f} km/s")

# Plot spectrum with velocity measurements
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavelength, spectrum_noisy / np.max(spectrum_noisy), 'b-', linewidth=1, alpha=0.7)

# Mark measured velocities
c_kms = 299792.458
colors = ['red', 'green', 'orange']
for (ion_name, rest_wave), color in zip(line_dict.items(), colors):
    v, verr = velocities[ion_name]
    v_abs = -v
    lambda_abs = rest_wave * (1 - v_abs / c_kms)
    ax.axvline(lambda_abs, color=color, linestyle='--', linewidth=2,
               label=f'{ion_name}: {v_abs:.0f} km/s')
    ax.axvline(rest_wave, color=color, linestyle=':', alpha=0.5)

ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Normalized Flux', fontsize=12)
ax.set_title('Photospheric Velocity Measurements from SN Ia Spectrum', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.savefig('velocity_measurements.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: velocity_measurements.png")

# =============================================================================
# Example 6: Photospheric Velocity Evolution
# =============================================================================
print("\nExample 6: Photospheric Velocity Evolution")
print("=" * 50)

# Simulate time series of spectra with decreasing velocity
# Typical SN Ia: v_phot decreases by ~50-100 km/s/day
times = np.array([0, 5, 10, 15, 20, 25])  # Days since maximum
v_phot_evolution = 12000 - 70 * times  # Linear decline ~70 km/s/day

wavelength_list = []
flux_list = []

for t, v_phot in zip(times, v_phot_evolution):
    wave = np.linspace(5800, 6800, 500)
    flux = spectral_models.p_cygni_profile(
        wave, lambda_rest=6355, tau_sobolev=3.0,
        v_phot=v_phot, continuum_flux=1.0, source_function='scattering', emission_scale=0.08, v_max=25000
    )
    # Add noise
    flux += 0.02 * np.random.randn(len(flux))
    wavelength_list.append(wave)
    flux_list.append(flux)

# Measure velocity evolution
times_measured, velocities, errors = SpectralVelocityFitter.photospheric_velocity_evolution(
    wavelength_list, flux_list, times, line_wavelength=6355, method='fit', v_window=20000,
    source_function='scattering', emission_scale=0.08, v_max=25000
)

print("Velocity Evolution (Si II 6355):")
velocities_measured = np.abs(velocities)
v_true = np.abs(v_phot_evolution)
for t, v, err in zip(times_measured, velocities_measured, errors):
    print(f"  Day {t:5.1f}: {v:7.0f} +/- {err:.0f} km/s")

# Calculate velocity gradient
fitter_dummy = SpectralVelocityFitter(wavelength_list[0], flux_list[0])
gradient, gradient_err = fitter_dummy.measure_velocity_gradient(
    wavelength_list, flux_list, times, line_wavelength=6355
)
print(f"\nVelocity gradient: {gradient:.1f} +/- {gradient_err:.1f} km/s/day")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Spectra at different epochs
colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
for t, wave, flux, color in zip(times, wavelength_list, flux_list, colors):
    ax1.plot(wave, flux, color=color, linewidth=1.5, label=f't = {t:.0f} d')
ax1.axvline(6355, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax1.set_ylabel('Normalized Flux', fontsize=12)
ax1.set_title('Si II 6355 Profile Evolution', fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Velocity vs time
ax2.errorbar(times_measured, velocities_measured/1000, yerr=errors/1000,
             fmt='o-', markersize=10, linewidth=2, capsize=5)
ax2.plot(times_measured, v_true/1000, 'r--', linewidth=2,
         alpha=0.7, label='True velocity (magnitude)')
ax2.set_xlabel('Days since maximum', fontsize=12)
ax2.set_ylabel('Photospheric Velocity Magnitude (1000 km/s)', fontsize=12)
ax2.set_title('Photospheric Velocity Evolution', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: velocity_evolution.png")

# =============================================================================
# Example 7: High-Velocity Feature Detection
# =============================================================================
print("\nExample 7: High-Velocity Feature (HVF) Detection")
print("=" * 50)

# Create spectrum with both photospheric and HVF components
wavelength = np.linspace(5800, 6800, 1000)

# Photospheric component at 11,000 km/s
flux_phot = spectral_models.elementary_p_cygni_profile(
    wavelength, lambda_rest=6355,
    v_absorption=11000,
    absorption_depth=0.45,
    emission_strength=0.05,
    v_width=1500
)

# HVF component at 16,000 km/s (weaker)
flux_hvf = spectral_models.elementary_p_cygni_profile(
    wavelength, lambda_rest=6355,
    v_absorption=16000,
    absorption_depth=0.12,
    emission_strength=0.0,
    v_width=1000,
    continuum_flux=1.0
)

# Combined spectrum
spectrum_combined = flux_phot * flux_hvf / 1.0  # Multiplicative for absorption
spectrum_combined += 0.01 * np.random.randn(len(wavelength))

# Detect HVF
fitter = SpectralVelocityFitter(wavelength, spectrum_combined)
has_hvf, v_hvf, v_hvf_err = fitter.identify_high_velocity_features(
    line_rest_wavelength=6355,
    v_phot_expected=11000,
    threshold_factor=1.3
)

print(f"HVF detected: {has_hvf}")
if has_hvf:
    print(f"HVF velocity: {-v_hvf:.0f} +/- {v_hvf_err:.0f} km/s")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelength, spectrum_combined, 'b-', linewidth=2, label='Observed spectrum')

# Mark features
c_kms = 299792.458
lambda_phot = 6355 * (1 - 11000/c_kms)
ax.axvline(lambda_phot, color='green', linestyle='--', linewidth=2,
           label=f'Photospheric: 11,000 km/s')

if has_hvf:
    lambda_hvf = 6355 * (1 + v_hvf/c_kms)
    ax.axvline(lambda_hvf, color='red', linestyle='--', linewidth=2,
               label=f'HVF: {-v_hvf:.0f} km/s')

ax.axvline(6355, color='gray', linestyle=':', alpha=0.5, label='Rest wavelength')
ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Normalized Flux', fontsize=12)
ax.set_title('High-Velocity Feature Detection in Si II 6355', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('hvf_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: hvf_detection.png")

# =============================================================================
# Example 8: SYNOW-style Line Model
# =============================================================================
print("\nExample 8: SYNOW-style Line Profile")
print("=" * 50)

wavelength = np.linspace(5800, 6800, 1000)

# SYNOW-style profile with different power-law indices
transmission_n7 = spectral_models.synow_line_model(
    wavelength, lambda_rest=6355, tau_ref=3.0,
    v_phot=11000, v_max=25000, n_power=7,
    aux_depth=0.15, temp_exc=12000
)

transmission_n10 = spectral_models.synow_line_model(
    wavelength, lambda_rest=6355, tau_ref=3.0,
    v_phot=11000, v_max=25000, n_power=10,
    aux_depth=0.15, temp_exc=12000
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelength, transmission_n7, 'b-', linewidth=2, label='n = 7 (standard)')
ax.plot(wavelength, transmission_n10, 'r--', linewidth=2, label='n = 10 (steeper)')
ax.axvline(6355, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Wavelength (Angstroms)', fontsize=12)
ax.set_ylabel('Transmission', fontsize=12)
ax.set_title('SYNOW-style Line Profile: Power-Law + Detached Shell', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('synow_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: synow_profile.png")

print("\n" + "=" * 50)
print("All examples completed successfully!")
print("Generated plots have been saved to the current directory.")
print("=" * 50)
