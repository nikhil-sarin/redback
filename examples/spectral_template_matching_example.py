"""
Spectral Template Matching Example
===================================

This example demonstrates how to use the SpectralTemplateMatcher class
to match observed spectra against template libraries for supernova classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from redback.analysis import SpectralTemplateMatcher
from redback.transient.transient import Spectrum

# =============================================================================
# Example 1: Using Default Blackbody Templates
# =============================================================================

print("=" * 60)
print("Example 1: Basic Template Matching with Default Templates")
print("=" * 60)

# Create a matcher with default built-in templates
matcher = SpectralTemplateMatcher()
print(f"Loaded {len(matcher.templates)} default templates")

# Create a synthetic observed spectrum (Type Ia-like at z=0.05)
# Simulating a Type Ia SN near maximum light
wavelengths_obs = np.linspace(3500, 9000, 500)
# Create a blackbody-like spectrum with some redshift applied
temp = 11000  # K, typical for Ia near max
z_true = 0.05

# Simple blackbody approximation (in observed frame)
wavelength_rest = wavelengths_obs / (1 + z_true)
h, c, k = 6.626e-27, 3e10, 1.38e-16
wavelength_cm = wavelength_rest * 1e-8
exponent = np.clip((h * c) / (wavelength_cm * k * temp), None, 700)
flux_obs = (1 / wavelength_cm**5) / (np.exp(exponent) - 1)
flux_obs = flux_obs / np.max(flux_obs)
flux_err = 0.05 * flux_obs  # 5% errors

# Create Spectrum object
observed_spectrum = Spectrum(
    angstroms=wavelengths_obs,
    flux_density=flux_obs,
    flux_density_err=flux_err,
    time="0",
    name="Test_SN"
)

# Match the spectrum
result = matcher.match_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
    n_redshift_points=50,
    method='correlation'
)

print(f"\nBest Match Results:")
print(f"  Type: {result['type']}")
print(f"  Phase: {result['phase']} days from maximum")
print(f"  Redshift: {result['redshift']:.4f} (true: {z_true})")
print(f"  Correlation: {result['correlation']:.4f}")
print(f"  Template: {result['template_name']}")

# =============================================================================
# Example 2: Classification with Confidence Metrics
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Full Classification with Type Probabilities")
print("=" * 60)

classification = matcher.classify_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
    n_redshift_points=30,
    top_n=10
)

print(f"\nClassification Results:")
print(f"  Best Type: {classification['best_type']}")
print(f"  Best Phase: {classification['best_phase']} days")
print(f"  Best Redshift: {classification['best_redshift']:.4f}")
print(f"  Correlation: {classification['correlation']:.4f}")
print(f"\nType Probabilities:")
for sn_type, prob in sorted(classification['type_probabilities'].items(),
                             key=lambda x: -x[1]):
    print(f"  {sn_type}: {prob:.2%}")

# =============================================================================
# Example 3: Plotting the Match
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Visualizing the Best Match")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
matcher.plot_match(observed_spectrum, result, axes=ax)
ax.set_title(f"Spectral Template Match\nBest: {result['type']} at z={result['redshift']:.3f}")
plt.tight_layout()
plt.savefig('spectral_match_example.png', dpi=150)
print("Saved plot to: spectral_match_example.png")
plt.close()

# =============================================================================
# Example 4: View Available Template Sources
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Available Template Sources")
print("=" * 60)

sources = SpectralTemplateMatcher.get_available_template_sources()
for name, info in sources.items():
    print(f"\n{name}:")
    print(f"  Description: {info['description']}")
    print(f"  URL: {info['url']}")
    print(f"  Citation: {info['citation']}")

# =============================================================================
# Example 5: Using Custom Templates
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Adding Custom Templates")
print("=" * 60)

# Create a matcher and add custom template
custom_matcher = SpectralTemplateMatcher()

# Add a custom template (e.g., from your own observations)
custom_wavelength = np.linspace(3000, 10000, 1000)
custom_flux = np.exp(-(custom_wavelength - 6500)**2 / (2 * 1000**2))  # Gaussian feature

custom_matcher.add_template(
    wavelength=custom_wavelength,
    flux=custom_flux,
    sn_type='Custom',
    phase=0,
    name='My_Custom_Template'
)

print(f"Added custom template. Total templates: {len(custom_matcher.templates)}")

# =============================================================================
# Example 6: Filtering Templates
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: Filtering Templates by Type and Phase")
print("=" * 60)

# Filter to only Type Ia templates near maximum light
filtered_matcher = matcher.filter_templates(
    types=['Ia'],
    phase_range=(-5, 10)
)

print(f"Original templates: {len(matcher.templates)}")
print(f"Filtered templates (Ia, phase -5 to +10): {len(filtered_matcher.templates)}")

# Match with filtered templates
result_filtered = filtered_matcher.match_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2)
)

if result_filtered:
    print(f"\nFiltered Match Results:")
    print(f"  Type: {result_filtered['type']}")
    print(f"  Phase: {result_filtered['phase']} days")
    print(f"  Redshift: {result_filtered['redshift']:.4f}")

# =============================================================================
# Example 7: Saving Templates for Future Use
# =============================================================================

print("\n" + "=" * 60)
print("Example 7: Saving Templates to Disk")
print("=" * 60)

# Save templates for later reuse
output_dir = './my_templates'
matcher.save_templates(output_dir, format='csv')
print(f"Saved {len(matcher.templates)} templates to {output_dir}/")

# Load them back later with:
# matcher_reloaded = SpectralTemplateMatcher(template_library_path=output_dir)

# =============================================================================
# Example 8: Chi-squared Matching (when errors are available)
# =============================================================================

print("\n" + "=" * 60)
print("Example 8: Chi-squared Based Matching")
print("=" * 60)

result_chi2 = matcher.match_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
    method='chi2'
)

print(f"\nChi-squared Match Results:")
print(f"  Type: {result_chi2['type']}")
print(f"  Phase: {result_chi2['phase']} days")
print(f"  Redshift: {result_chi2['redshift']:.4f}")
print(f"  Chi-squared: {result_chi2.get('chi2', 'N/A'):.2f}")
print(f"  Scale factor: {result_chi2.get('scale_factor', 'N/A'):.4f}")

# =============================================================================
# Example 9: Getting All Matches for Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Example 9: Analyzing All Matches")
print("=" * 60)

all_matches = matcher.match_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
    method='both',
    return_all_matches=True
)

print(f"Total matches tested: {len(all_matches)}")
print("\nTop 5 matches by correlation:")
for i, match in enumerate(all_matches[:5]):
    print(f"  {i+1}. {match['type']} phase={match['phase']:+.0f}d "
          f"z={match['redshift']:.3f} r={match['correlation']:.3f}")

# =============================================================================
# Advanced: Downloading Templates from External Sources
# =============================================================================

print("\n" + "=" * 60)
print("ADVANCED USAGE: Downloading External Templates")
print("=" * 60)

print("""
# Download from Open Supernova Catalog (requires internet):
# matcher = SpectralTemplateMatcher.download_templates_from_osc(
#     sn_types=['Ia', 'II', 'Ib', 'Ic'],
#     max_per_type=10
# )

# Download SESN templates from GitHub:
# matcher = SpectralTemplateMatcher.from_sesn_templates()

# Load SNID templates from local directory:
# matcher = SpectralTemplateMatcher.from_snid_template_directory(
#     '/path/to/snid/templates/'
# )

# Download from any GitHub repository:
# template_dir = SpectralTemplateMatcher.download_github_templates(
#     'https://github.com/dkjmagill/QUB-SNID-Templates'
# )
# matcher = SpectralTemplateMatcher.from_snid_template_directory(template_dir)
""")

print("\nExample complete!")
print("See documentation for more details: docs/spectral_template_matching.txt")
