"""
Example: simulate a kilonova with LightCurveLynx and fit it with redback.

Requires:
    pip install lightcurvelynx

The LSST OpSim database and passband tables are loaded from the
LightCurveLynx base data directory (_LIGHTCURVELYNX_BASE_DATA_DIR).
See the LightCurveLynx docs for how to download those files.
"""

import numpy as np
import matplotlib.pyplot as plt

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
# Simulate until we get at least one object with detections.
n_sims = 20
lightcurves = simulate_lightcurves(
    source,
    n_sims,
    opsim_db,
    passband_group,
    obs_time_window_offset=(0.1, 20),
)

# Post-process: apply SNR cut and augment with mag columns.
results = results_augment_lightcurves(lightcurves, min_snr=3)

# Pick the first simulated object that has detections.
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
        break

if chosen_idx is None:
    raise RuntimeError(
        "No simulated object had enough detections. "
        "Try increasing n_sims or lowering the SNR threshold."
    )

print(f"\nUsing simulated object index {chosen_idx}")
print(f"Injection parameters: {chosen_params}")
print(f"Lightcurve ({len(chosen_lc)} rows, {chosen_lc['detection'].sum()} detections):")
print(chosen_lc)

# ---------------------------------------------------------------------------
# 4. Load into a redback transient object
# ---------------------------------------------------------------------------
transient_name = "lynx_kilonova_sim"

# Note that redback and lightcurve lynx only agree on AB magnitudes.
# Redback has a different definition of bandpass flux and a different definition of flux density.
# If you use the following constructor this is automatically taken care off
# If you are doing transient object construction manually either work in magnitudes or make sure you are making the units consistent before fitting.
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
# 5. Set up priors and fit
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
# This result object is identical to any other redback result object so has the methods for plotting/etc.