"""
Example: simulate a kilonova with LightCurveLynx and fit it with redback.

Requires:
    pip install lightcurvelynx

The LSST OpSim database and passband tables are loaded from the
LightCurveLynx base data directory (_LIGHTCURVELYNX_BASE_DATA_DIR).
See the LightCurveLynx docs for how to download those files.

Two fitting approaches are demonstrated:
  1. Basic fit: physical model with redshift fixed to the injected value and
     explosion time known from the simulation (time-since-explosion axis).
  2. Realistic fit: use t0_base_model to sample the explosion epoch t0 as a
     free parameter, combined with extinction_with_kilonova_base_model to
     also fit for host-galaxy dust.  This is the recommended setup for real
     data where neither t0 nor the host extinction is known a priori.

Non-detections (observations below the SNR threshold) are flagged by
results_augment_lightcurves and dropped from the redback transient by default.
To retain them as upper limits, pass include_upper_limits=True to
from_lightcurvelynx.
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby

import redback
from redback import model_library

from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import ObsTableRADECSampler
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.models.redback_models import RedbackWrapperModel
from lightcurvelynx.utils.post_process_results import results_augment_lightcurves
from lightcurvelynx import _LIGHTCURVELYNX_BASE_DATA_DIR

# ---------------------------------------------------------------------------
# 1. Load OpSim and passbands
# ---------------------------------------------------------------------------
filters = ["g", "r", "i", "z"]

opsim_db = OpSim.from_db(_LIGHTCURVELYNX_BASE_DATA_DIR / "opsim" / "baseline_v3.4_10yrs.db")
filter_mask = np.isin(opsim_db["filter"], filters)
opsim_db = opsim_db.filter_rows(filter_mask)
t_min, t_max = opsim_db.time_bounds()
print(f"Loaded OpSim: {len(opsim_db)} rows, MJD [{t_min:.1f}, {t_max:.1f}]")

passband_group = PassbandGroup.from_preset(
    preset="LSST",
    filters=filters,
    units="nm",
    trim_quantile=0.001,
    delta_wave=1,
    table_dir=_LIGHTCURVELYNX_BASE_DATA_DIR / "passbands" / "LSST",
)

# ---------------------------------------------------------------------------
# 2. Build the LightCurveLynx source model
# ---------------------------------------------------------------------------
rb_model = model_library.all_models_dict["one_component_kilonova_model"]

ra_dec_sampler = ObsTableRADECSampler(opsim_db, radius=3.0, node_label="ra_dec_sampler")
time_sampler = NumpyRandomFunc("uniform", low=t_min, high=t_max, node_label="time_sampler")

parameters = {
    "mej": 0.05,
    "redshift": 0.01,
    "temperature_floor": 3000,
    "kappa": 1,
    "vej": 0.2,
}

source = RedbackWrapperModel(
    rb_model,
    parameters=parameters,
    ra=ra_dec_sampler.ra,
    dec=ra_dec_sampler.dec,
    t0=time_sampler,
    node_label="source",
)

# ---------------------------------------------------------------------------
# 3. Simulate lightcurves
# ---------------------------------------------------------------------------
# obs_time_window_offset must start > 0 for explosion-based models (kilonova,
# supernova, etc.) which are only defined for positive times since explosion.
n_sims = 20
lightcurves = simulate_lightcurves(
    source,
    n_sims,
    opsim_db,
    passband_group,
    obs_time_window_offset=(0.1, 20),
)

# Post-process: compute SNR, detection flag, AB mag/magerr, and relative time.
# Observations below min_snr are flagged as non-detections (detection=False).
# These are dropped from the redback transient by default; pass
# include_upper_limits=True to from_lightcurvelynx to keep them as upper limits.
results = results_augment_lightcurves(lightcurves, min_snr=3)

# Pick the first simulated object that has enough detections.
# Map the short filter names used by LightCurveLynx to the LSST band names
# that redback expects (e.g. 'g' -> 'lsstg').
band_mapping = {"g": "lsstg", "r": "lsstr", "i": "lssti", "z": "lsstz"}

chosen_idx = None
for idx in results.index:
    row = results.loc[idx]
    lc = row["lightcurve"].copy()
    lc["filter"] = np.array([band_mapping.get(f, f) for f in lc["filter"]])
    detections = lc[lc["detection"]]
    if len(detections) >= 3:
        chosen_idx = idx
        chosen_lc = lc
        chosen_params = row["params"]
        injected_t0_mjd = float(row["t0"])  # explosion epoch in MJD
        break

if chosen_idx is None:
    raise RuntimeError(
        "No simulated object had enough detections. "
        "Try increasing n_sims or lowering the SNR threshold."
    )

print(f"\nUsing simulated object index {chosen_idx}")
print(f"Injection parameters: {chosen_params}")
print(f"Injected t0 (MJD): {injected_t0_mjd:.2f}")
print(f"Lightcurve ({len(chosen_lc)} rows, {chosen_lc['detection'].sum()} detections):")
print(chosen_lc)

# ---------------------------------------------------------------------------
# 4. Load into a redback transient object
# ---------------------------------------------------------------------------
transient_name = "lynx_kilonova_sim"

# Note that redback and LightCurveLynx only agree on AB magnitudes.
# Redback has a different definition of bandpass flux and flux density.
# from_lightcurvelynx handles the conversion automatically by reconstructing
# from the mag/magerr columns.  If you construct the object manually, work in
# AB magnitudes or ensure units are consistent before fitting.
kn = redback.transient.Kilonova.from_lightcurvelynx(
    name=transient_name,
    data=chosen_lc,
    data_mode="flux_density",
)
kn.plot_data(show=False)
plt.savefig(f"{transient_name}_data.png", dpi=150, bbox_inches="tight")
plt.close()
print("Data plot saved.")

# ---------------------------------------------------------------------------
# 5. Basic fit: physical model with fixed redshift and known explosion time
# ---------------------------------------------------------------------------
model = "one_component_kilonova_model"
sampler = "dynesty"

priors = redback.priors.get_priors(model=model)
# Fix redshift to the injected value (known from simulation).
injected_redshift = chosen_params.get("source.redshift", parameters["redshift"])
priors["redshift"] = float(injected_redshift)

model_kwargs = dict(
    frequency=kn.filtered_frequencies,
    output_format="flux_density",
)

result = redback.fit_model(
    transient=kn,
    model=model,
    sampler=sampler,
    model_kwargs=model_kwargs,
    prior=priors,
    sample="rslice",
    nlive=500,
    resume=True,
)
# This result object is identical to any other redback result object so has
# the methods for plotting/etc.

# ---------------------------------------------------------------------------
# 6. Realistic fit: sample t0 and host extinction with wrapper models
#
# For real data the explosion time is unknown, so we use t0_base_model which
# works on MJD times and subtracts t0 before calling the base model.
# The prior on t0 must have its maximum at or before the first detection —
# t0_base_model returns zero flux for pre-t0 times, so a t0 after the first
# detection would leave the likelihood with no valid data points.
#
# We additionally wrap with extinction_with_kilonova_base_model to fit for
# host-galaxy V-band extinction av_host.  MW extinction av_mw can be fixed
# from a dustmap query and passed as a model_kwarg.
# ---------------------------------------------------------------------------
kn_phase = redback.transient.Kilonova.from_lightcurvelynx(
    name=transient_name,
    data=chosen_lc,
    data_mode="flux_density",
    use_phase_model=True,
)

first_det_mjd = float(chosen_lc[chosen_lc["detection"]]["mjd"].min())
t0_prior_window = 20.0  # days to search before first detection

priors_ext = redback.priors.get_priors(model="one_component_kilonova_model")
priors_ext["redshift"] = float(injected_redshift)
priors_ext["t0"] = bilby.core.prior.Uniform(
    minimum=first_det_mjd - t0_prior_window,
    maximum=first_det_mjd,
    name="t0",
    latex_label="$t_0$ (MJD)",
)
priors_ext["av_host"] = bilby.core.prior.Uniform(
    minimum=0.0, maximum=1.0, name="av_host", latex_label="$A_V^{\\rm host}$"
)

model_kwargs_ext = dict(
    frequency=kn_phase.filtered_frequencies,
    output_format="flux_density",
    base_model="one_component_kilonova_model",
    av_mw=0.0,  # fix MW extinction; in practice query from e.g. dustmaps
)

result_ext = redback.fit_model(
    transient=kn_phase,
    model="extinction_with_kilonova_base_model",
    sampler=sampler,
    model_kwargs=model_kwargs_ext,
    prior=priors_ext,
    sample="rslice",
    nlive=500,
    resume=True,
)
