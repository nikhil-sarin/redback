"""
Spectral Template Matching Example
===================================

This example demonstrates how to use the SpectralTemplateMatcher class
to match observed spectra against template libraries for supernova classification.

The default template library uses sncosmo spectral models (SALT2, SN 1998bw,
Nugent templates) and does not require any external downloads.

For a large, high-quality template library, use the Super-SNID templates
(downloaded automatically on first use):

    matcher = SpectralTemplateMatcher.from_super_snid_templates(
        cache_dir='./my_templates'
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import sncosmo
from redback.analysis import SpectralTemplateMatcher
from redback.transient.transient import Spectrum

# =============================================================================
# Create a toy observed spectrum: SALT2 Type Ia at z=0.05, near maximum light
# =============================================================================

src = sncosmo.get_source('salt2')
z_true = 0.05
wavelengths_obs = np.linspace(3500, 9000, 500)
wave_rest = wavelengths_obs / (1 + z_true)
wave_rest_clipped = np.clip(wave_rest, src.minwave(), src.maxwave())
flux_rest = src.flux(0.0, wave_rest_clipped)
flux_obs = flux_rest / np.max(flux_rest)
np.random.seed(42)
flux_err = 0.05 * np.abs(flux_obs)
flux_obs += np.random.normal(0, flux_err, len(flux_obs))

# Create Spectrum object
observed_spectrum = Spectrum(
    angstroms=wavelengths_obs,
    flux_density=flux_obs,
    flux_density_err=flux_err,
    time="0",
    name="Test_SN_Ia"
)

# =============================================================================
# Example 1: Basic Template Matching with Default Templates
# =============================================================================

print("=" * 60)
print("Example 1: Basic Template Matching with Default Templates")
print("=" * 60)

matcher = SpectralTemplateMatcher()
print(f"Loaded {len(matcher.templates)} default templates")
print(f"Types covered: {sorted(set(t['type'] for t in matcher.templates))}")

result = matcher.match_spectrum(observed_spectrum, redshift_range=(0.0, 0.2))

print(f"\nBest Match Results:")
print(f"  Type:       {result['type']}")
print(f"  Phase:      {result['phase']:+.0f} days from maximum")
print(f"  Redshift:   {result['redshift']:.4f}  (true: {z_true})")
print(f"  rlap:       {result['rlap']:.2f}")
print(f"  Template:   {result['template_name']}")

# =============================================================================
# Example 2: Classification with Probabilities
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Full Classification with Type Probabilities")
print("=" * 60)

classification = matcher.classify_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
    top_n=10
)
print(classification.summary())

# =============================================================================
# Example 3: Visualizing the Best Match
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Visualizing the Best Match")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
matcher.plot_match(observed_spectrum, result, axes=ax)
ax.set_title(
    f"Spectral Template Match  —  "
    f"{result['type']} at z={result['redshift']:.3f}, rlap={result['rlap']:.1f}"
)
plt.tight_layout()
plt.savefig('spectral_match_example.png', dpi=150)
print("Saved plot to: spectral_match_example.png")
plt.close()

# =============================================================================
# Example 4: Filtering Templates
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Filtering Templates by Type and Phase")
print("=" * 60)

ia_matcher = matcher.filter_templates(types=['Ia'], phase_range=(-5, 10))
print(f"All templates:     {len(matcher.templates)}")
print(f"Ia, -5 to +10 d:   {len(ia_matcher.templates)}")

result_ia = ia_matcher.match_spectrum(observed_spectrum, redshift_range=(0.0, 0.2))
if result_ia:
    print(f"\nIa-only match: phase={result_ia['phase']:+.0f}d  z={result_ia['redshift']:.4f}")

# =============================================================================
# Example 5: Super-SNID Templates (downloaded automatically)
# If you use this frequently, set the cache_dir to avoid redownloading.
# This requires internet access on first run and uses the excellent Super-SNID library templates available at:
# https://github.com/dkjmagill/QUB-SNID-Templates
# Please cite the appropriate references if you use these templates in your research.
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Classification with Super-SNID Templates")
print("=" * 60)

print("Downloading Super-SNID template library (cached after first run)...")
super_snid_matcher = SpectralTemplateMatcher.from_super_snid_templates(
    cache_dir='./my_templates'
)
print(f"Loaded {len(super_snid_matcher.templates)} Super-SNID templates")
print(f"Types covered: {sorted(set(t['type'] for t in super_snid_matcher.templates))}")

result_ss = super_snid_matcher.match_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
)
print(f"\nSuper-SNID Best Match:")
print(f"  Type:       {result_ss['type']}")
print(f"  Phase:      {result_ss['phase']:+.0f} days from maximum")
print(f"  Redshift:   {result_ss['redshift']:.4f}  (true: {z_true})")
print(f"  rlap:       {result_ss['rlap']:.2f}")
print(f"  Template:   {result_ss['template_name']}")

classification_ss = super_snid_matcher.classify_spectrum(
    observed_spectrum,
    redshift_range=(0.0, 0.2),
    top_n=10
)
print(f"\nSuper-SNID Classification:")
print(classification_ss.summary())

fig, ax = plt.subplots(figsize=(10, 6))
super_snid_matcher.plot_match(observed_spectrum, result_ss, axes=ax)
ax.set_title(
    f"Super-SNID Match  —  "
    f"{result_ss['type']} at z={result_ss['redshift']:.3f}, rlap={result_ss['rlap']:.1f}"
)
plt.tight_layout()
plt.savefig('spectral_match_supersnid.png', dpi=150)
print("Saved plot to: spectral_match_supersnid.png")
plt.close()

# =============================================================================
# Example 6: Generating Templates from Custom sncosmo Sources
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: Custom sncosmo Template Set")
print("=" * 60)

custom_templates = SpectralTemplateMatcher.generate_sncosmo_templates(
    sources=[
        ('salt2',      'Ia',    [-10, -5, 0, 5, 10, 15, 20]),
        ('v19-1998bw', 'Ic-BL', [0, 5, 10, 15, 20]),
    ]
)
custom_matcher = SpectralTemplateMatcher(templates=custom_templates)
print(f"Custom matcher: {len(custom_matcher.templates)} templates")

# =============================================================================
# Example 7: Saving and Reloading Templates
# =============================================================================

print("\n" + "=" * 60)
print("Example 7: Saving and Reloading Templates")
print("=" * 60)

output_dir = './my_default_templates'
matcher.save_templates(output_dir, format='csv')
print(f"Saved {len(matcher.templates)} templates to {output_dir}/")

reloaded = SpectralTemplateMatcher(template_library_path=output_dir)
print(f"Reloaded {len(reloaded.templates)} templates")

# =============================================================================
# Example 8: Loading SNID Template Files from disk
# =============================================================================

print("\n" + "=" * 60)
print("Example 8: Loading SNID Templates from disk")
print("=" * 60)

print("""
If you have SNID template files (.lnw format) already on disk:

    matcher = SpectralTemplateMatcher.from_snid_template_directory(
        '/path/to/snid/templates-2.0/'
    )
""")

print("Example complete!")
print("See documentation for more details: docs/spectral_template_matching.txt")
