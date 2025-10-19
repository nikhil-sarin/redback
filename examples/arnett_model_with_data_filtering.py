"""
Example: Fitting an Arnett model to supernova data with data filtering

This example demonstrates how to fit an Arnett model to supernova data that contains:
1. Pre-explosion data points (observations before the actual supernova explosion)
2. Erroneous data points from a bad photometric pipeline

We will show multiple methods to filter and clean the data before fitting.
"""

import numpy as np
import pandas as pd
import redback

# =============================================================================
# Step 1: Load supernova data
# =============================================================================

# For this example, we'll use data from the Open Transient Catalog
# You can replace this with your own data following the pattern shown below

sne = "SN1998bw"

# Load data using redback's built-in data getter
# This returns a supernova object with all the data
supernova = redback.supernova.Supernova.from_open_access_catalogue(
    name=sne, 
    data_mode='flux_density', 
    active_bands=["I", "R", "V", "B"]
)

# Alternative: If you have your own data in a CSV file, you can load it like this:
# data = pd.read_csv('your_data.csv')
# time_mjd = data['time_mjd'].values
# magnitude = data['magnitude'].values
# magnitude_err = data['magnitude_err'].values
# bands = data['band'].values
# 
# supernova = redback.supernova.Supernova(
#     name='your_supernova',
#     data_mode='magnitude',
#     time_mjd=time_mjd,
#     magnitude=magnitude,
#     magnitude_err=magnitude_err,
#     bands=bands,
#     use_phase_model=True  # Important for handling unknown explosion time
# )

# =============================================================================
# Step 2: Filter out pre-explosion data points
# =============================================================================

# Method 1: Filter by time (if you know the explosion time)
# Assuming the explosion time is at t=0 in your time array
# We'll only keep data points after the explosion

# For data loaded from catalogs, check what time reference is used
print(f"Data mode: {supernova.data_mode}")
print(f"Using phase model: {supernova.use_phase_model}")

# Get the current data arrays
time_data = supernova.time
flux_density_data = supernova.flux_density
flux_density_err_data = supernova.flux_density_err
bands_data = supernova.bands

# Method 1a: Simple time cut - remove all data before time = 0
# (if your time is already referenced to explosion)
time_threshold = 0  # days after explosion
mask_after_explosion = time_data > time_threshold

# Method 1b: Remove pre-explosion data based on a specific date
# If using MJD and you know the explosion date
# explosion_mjd = 50929.0  # Example MJD of explosion
# mask_after_explosion = supernova.time_mjd > explosion_mjd

# Apply the time mask
time_filtered = time_data[mask_after_explosion]
flux_density_filtered = flux_density_data[mask_after_explosion]
flux_density_err_filtered = flux_density_err_data[mask_after_explosion]
bands_filtered = bands_data[mask_after_explosion]

# =============================================================================
# Step 3: Remove erroneous data points from bad photometric pipeline
# =============================================================================

# Method 2a: Remove outliers using sigma clipping
# This removes data points that are too far from the median

def sigma_clip_data(flux, flux_err, sigma=3.0):
    """
    Remove outliers using sigma clipping.
    
    Parameters
    ----------
    flux : array-like
        Flux density values
    flux_err : array-like
        Flux density errors
    sigma : float, optional
        Number of standard deviations for clipping (default: 3.0)
    
    Returns
    -------
    mask : array-like
        Boolean mask where True indicates good data points
    """
    # Calculate median and standard deviation
    median_flux = np.median(flux)
    std_flux = np.std(flux)
    
    # Create mask for points within sigma standard deviations
    mask = np.abs(flux - median_flux) < sigma * std_flux
    
    return mask

# Apply sigma clipping to each band separately
# This is better because different bands have different flux levels
unique_bands = np.unique(bands_filtered)
mask_good_data = np.zeros(len(flux_density_filtered), dtype=bool)

for band in unique_bands:
    band_mask = bands_filtered == band
    band_flux = flux_density_filtered[band_mask]
    band_flux_err = flux_density_err_filtered[band_mask]
    
    # Apply sigma clipping to this band
    good_points_in_band = sigma_clip_data(band_flux, band_flux_err, sigma=3.0)
    
    # Update the overall mask
    band_indices = np.where(band_mask)[0]
    mask_good_data[band_indices[good_points_in_band]] = True

# Method 2b: Remove points with unreasonably large errors
# Bad photometric pipelines often produce very large error bars
max_relative_error = 0.5  # Remove points with >50% relative error
mask_good_errors = (flux_density_err_filtered / flux_density_filtered) < max_relative_error

# Method 2c: Remove negative or zero flux densities
# These are often errors in the photometric pipeline
mask_positive_flux = flux_density_filtered > 0

# Combine all masks
final_mask = mask_good_data & mask_good_errors & mask_positive_flux

# Apply all masks to create clean data
time_clean = time_filtered[final_mask]
flux_density_clean = flux_density_filtered[final_mask]
flux_density_err_clean = flux_density_err_filtered[final_mask]
bands_clean = bands_filtered[final_mask]

print(f"\nData filtering summary:")
print(f"Original data points: {len(time_data)}")
print(f"After removing pre-explosion data: {len(time_filtered)}")
print(f"After sigma clipping: {np.sum(mask_good_data)}")
print(f"After error filtering: {np.sum(mask_good_errors)}")
print(f"After removing non-positive flux: {np.sum(mask_positive_flux)}")
print(f"Final clean data points: {len(time_clean)}")

# =============================================================================
# Step 4: Create a new transient object with cleaned data
# =============================================================================

# Create a new supernova object with the cleaned data
supernova_clean = redback.supernova.Supernova(
    name=f"{sne}_cleaned",
    data_mode='flux_density',
    time=time_clean,
    flux_density=flux_density_clean,
    flux_density_err=flux_density_err_clean,
    bands=bands_clean,
    active_bands=["I", "R", "V", "B"],
    use_phase_model=False  # Since we've already referenced to explosion time
)

# Plot the cleaned data to visually inspect
print("\nPlotting cleaned data...")
supernova_clean.plot_multiband(filters=["I", "R", "V", "B"])

# =============================================================================
# Step 5: Fit the Arnett model to the cleaned data
# =============================================================================

# Set up the Arnett model
model = "arnett"
sampler = 'dynesty'

# Get default priors for the Arnett model
priors = redback.priors.get_priors(model=model)

# Set the redshift if known (example value)
priors['redshift'] = 1e-2

# Set up model kwargs
# The Arnett model needs to know what frequencies/bands to calculate
model_kwargs = dict(
    frequency=supernova_clean.filtered_frequencies, 
    output_format='flux_density'
)

# Optional: You can modify priors based on your knowledge
# For example, if you know the supernova was particularly energetic:
# priors['mej'] = bilby.core.prior.Uniform(minimum=5, maximum=20, name='mej')
# priors['f_nickel'] = bilby.core.prior.Uniform(minimum=0.1, maximum=1.0, name='f_nickel')

print("\nFitting Arnett model to cleaned data...")
print("This may take several minutes...")

# Fit the model
# Note: For a quick test run, use nlive=100 and resume=False
# For production runs, use nlive=500-1000 and resume=True
result = redback.fit_model(
    transient=supernova_clean, 
    model=model, 
    sampler=sampler, 
    model_kwargs=model_kwargs,
    prior=priors, 
    sample='rslice',  # Sampling method for dynesty
    nlive=100,  # Number of live points (increase for better accuracy)
    resume=True  # Resume from previous run if available
)

# =============================================================================
# Step 6: Analyze and plot results
# =============================================================================

print("\nFitting complete! Generating plots...")

# Plot the corner plot showing parameter posteriors
result.plot_corner()

# Plot the fitted lightcurve with random posterior samples
result.plot_lightcurve(random_models=100, plot_others=False)

# Plot multiband lightcurve showing fit in each band
result.plot_multiband_lightcurve(filters=["I", "R", "V", "B"])

print("\nResults saved to the output directory.")
print("\nKey takeaways:")
print("1. Pre-explosion data was filtered using a time threshold")
print("2. Erroneous data was removed using sigma clipping and error thresholds")
print("3. The Arnett model was successfully fitted to the cleaned data")
print("4. Results show the posterior distributions of physical parameters")
