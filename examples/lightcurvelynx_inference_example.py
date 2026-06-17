"""
Example: simulate kilonovae with LightCurveLynx and fit them with redback.

Requires:
    pip install lightcurvelynx

The LSST OpSim database and passband tables are loaded from the
LightCurveLynx base data directory (_LIGHTCURVELYNX_BASE_DATA_DIR).
See the LightCurveLynx docs for how to download those files.

Two fitting approaches are demonstrated:
  1. Basic fit: simulate with one_component_kilonova_model directly (no
     extinction, explosion time known from the simulation) and fit with the
     same model on a time-since-explosion axis.
  2. Injection-recovery fit: simulate with extinction_with_kilonova_base_model
     (injecting a known host-galaxy Av) and fit with t0_kilonova_extinction,
     sampling both av_host and t0 as free parameters.  This is the recommended
     setup for real data where neither the explosion epoch nor the host
     extinction are known a priori.

Note on t0 handling: RedbackWrapperModel (from LightCurveLynx) always
subtracts t0 from the MJD observation times before calling the source
function, so the source receives time-since-explosion in days.  This means
the simulation source should be extinction_with_kilonova_base_model (which
takes time-since-explosion), not t0_kilonova_extinction (which takes MJD and
subtracts t0 itself — that would double-subtract t0).  For the fit we use
t0_kilonova_extinction with use_phase_model=True so that redback passes raw
MJD times to the model, which then handles the subtraction.

Non-detections (observations below the SNR threshold) are flagged by
results_augment_lightcurves and dropped from the redback transient by default.
To retain them as upper limits, pass include_upper_limits=True to
from_lightcurvelynx method.

Pre-explosion non-detections: this example uses obs_time_window_offset=(0.1, 20)
so the observation window always starts after the explosion and no pre-explosion
observations are generated.  A fix to RedbackWrapperModel is pending in
lincc-frameworks/lightcurvelynx#880 — once merged, you will be able to use a
negative start offset (e.g. obs_time_window_offset=(-5, 20)) to generate
realistic pre-explosion non-detections; they will automatically receive zero
flux and appear as non-detections in results_augment_lightcurves.

Please also look at the lightcurvelynx documentation to see how you can change things on the
lightcurvelynx side to simulate multiple surveys/etc
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
# 1. Load OpSim and passbands (shared by both simulations)
# ---------------------------------------------------------------------------
filters = ["g", "r", "i", "z"]
band_mapping = {"g": "lsstg", "r": "lsstr", "i": "lssti", "z": "lsstz"}

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

sampler = "dynesty"
n_sims = 20

# ---------------------------------------------------------------------------
# Helper: pick the first simulated object with enough detections
# ---------------------------------------------------------------------------
def pick_event(results, source_label):
    for idx in results.index:
        row = results.loc[idx]
        lc = row["lightcurve"].copy()
        lc["filter"] = np.array([band_mapping.get(f, f) for f in lc["filter"]])
        if lc["detection"].sum() >= 3:
            return idx, lc, row["params"], float(row["t0"])
    raise RuntimeError(
        "No simulated object had enough detections. "
        "Try increasing n_sims or lowering the SNR threshold."
    )


# ===========================================================================
# Simulation 1 + Fit 1: bare kilonova, no extinction, known explosion time
# ===========================================================================

# ---------------------------------------------------------------------------
# 2. Build and run the first simulation
#
# RedbackWrapperModel subtracts t0 (the LightCurveLynx explosion-time sampler)
# from the MJD observation times before calling the source function.  The
# source therefore receives time-since-explosion in days, which is what
# one_component_kilonova_model expects.
# ---------------------------------------------------------------------------
ra_dec_1 = ObsTableRADECSampler(opsim_db, radius=3.0, node_label="ra_dec_1")
t0_sampler_1 = NumpyRandomFunc("uniform", low=t_min, high=t_max, node_label="t0_1")

source_1 = RedbackWrapperModel(
    model_library.all_models_dict["one_component_kilonova_model"],
    parameters={
        "mej": 0.05,
        "redshift": 0.01,
        "temperature_floor": 3000,
        "kappa": 1,
        "vej": 0.2,
    },
    ra=ra_dec_1.ra,
    dec=ra_dec_1.dec,
    t0=t0_sampler_1,
    node_label="source_1",
)

# obs_time_window_offset must start > 0 for explosion-based models — they are
# only defined for positive times since explosion.
lightcurves_1 = simulate_lightcurves(
    source_1, n_sims, opsim_db, passband_group, obs_time_window_offset=(0.1, 20),
)
# Post-process: compute SNR, detection flag, AB mag/magerr.
# Observations below min_snr are flagged as non-detections (detection=False).
# These are dropped from the redback transient by default; pass
# include_upper_limits=True to from_lightcurvelynx to keep them as upper limits.
results_1 = results_augment_lightcurves(lightcurves_1, min_snr=3)

chosen_idx_1, chosen_lc_1, chosen_params_1, injected_t0_mjd_1 = pick_event(results_1, "source_1")
injected_redshift_1 = chosen_params_1.get("source_1.redshift", 0.01)

print(f"\n=== Simulation 1 (no extinction) ===")
print(f"Using simulated object index {chosen_idx_1}")
print(f"Injection parameters: {chosen_params_1}")
print(f"Injected t0 (MJD): {injected_t0_mjd_1:.2f}")
print(f"Lightcurve ({len(chosen_lc_1)} rows, {chosen_lc_1['detection'].sum()} detections):")
print(chosen_lc_1)

# ---------------------------------------------------------------------------
# 3. Load into redback and fit
#
# use_phase_model=False (default): redback converts MJD times to
# time-since-explosion using the known injected t0 before passing to the model.
# Note that redback and LightCurveLynx only agree on AB magnitudes.
# from_lightcurvelynx handles unit conversion automatically.
# ---------------------------------------------------------------------------
kn_1 = redback.transient.Kilonova.from_lightcurvelynx(
    name="lynx_kilonova_basic",
    data=chosen_lc_1,
    data_mode="flux_density",
)
kn_1.plot_data(show=False)
plt.savefig("lynx_kilonova_basic_data.png", dpi=150, bbox_inches="tight")
plt.close()
print("Basic data plot saved.")

priors_1 = redback.priors.get_priors(model="one_component_kilonova_model")
priors_1["redshift"] = float(injected_redshift_1)

result_1 = redback.fit_model(
    transient=kn_1,
    model="one_component_kilonova_model",
    sampler=sampler,
    model_kwargs=dict(frequency=kn_1.filtered_frequencies, output_format="flux_density"),
    prior=priors_1,
    sample="rslice",
    nlive=500,
    resume=True,
)


# ===========================================================================
# Simulation 2 + Fit 2: kilonova with host extinction, t0 and av_host unknown
# ===========================================================================

# ---------------------------------------------------------------------------
# 4. Build and run the second simulation
#
# We inject av_host=0.3 mag of host-galaxy extinction using
# extinction_with_kilonova_base_model as the source.  As above,
# RedbackWrapperModel subtracts t0 before calling the source, so the source
# receives time-since-explosion — exactly what extinction_with_kilonova_base_model
# expects (it applies dust reddening and then dispatches to base_model).
# ---------------------------------------------------------------------------
INJECTED_AV_HOST = 0.3  # mag

ra_dec_2 = ObsTableRADECSampler(opsim_db, radius=3.0, node_label="ra_dec_2")
t0_sampler_2 = NumpyRandomFunc("uniform", low=t_min, high=t_max, node_label="t0_2")

source_2 = RedbackWrapperModel(
    model_library.all_models_dict["extinction_with_kilonova_base_model"],
    parameters={
        "mej": 0.05,
        "redshift": 0.01,
        "temperature_floor": 3000,
        "kappa": 1,
        "vej": 0.2,
        "av_host": INJECTED_AV_HOST,
        "av_mw": 0.0,
        "base_model": "one_component_kilonova_model",
    },
    ra=ra_dec_2.ra,
    dec=ra_dec_2.dec,
    t0=t0_sampler_2,
    node_label="source_2",
)

lightcurves_2 = simulate_lightcurves(
    source_2, n_sims, opsim_db, passband_group, obs_time_window_offset=(0.1, 20),
)
results_2 = results_augment_lightcurves(lightcurves_2, min_snr=3)

chosen_idx_2, chosen_lc_2, chosen_params_2, injected_t0_mjd_2 = pick_event(results_2, "source_2")
injected_redshift_2 = chosen_params_2.get("source_2.redshift", 0.01)

print(f"\n=== Simulation 2 (with host extinction, unknown t0) ===")
print(f"Using simulated object index {chosen_idx_2}")
print(f"Injection parameters: {chosen_params_2}")
print(f"Injected t0 (MJD): {injected_t0_mjd_2:.2f}")
print(f"Injected av_host: {INJECTED_AV_HOST:.2f} mag")
print(f"Lightcurve ({len(chosen_lc_2)} rows, {chosen_lc_2['detection'].sum()} detections):")
print(chosen_lc_2)

# ---------------------------------------------------------------------------
# 5. Load into redback and fit with t0_kilonova_extinction
#
# use_phase_model=True: redback passes raw MJD times to the model.
# t0_kilonova_extinction takes MJD times, subtracts t0, applies host + MW
# dust reddening, and dispatches to base_model.  This is the correct pair
# for data simulated with extinction_with_kilonova_base_model via
# RedbackWrapperModel — the physics is identical; the only difference is that
# the fit model also samples t0 as a free parameter.
#
# The prior on t0 must have its maximum at or before the first detection —
# t0_kilonova_extinction returns zero flux for pre-t0 times, so a t0 after
# the first detection leaves the likelihood with no valid data points.
#
# av_mw can be fixed from a dustmap query (e.g. dustmaps.sfd) and passed as
# a model_kwarg rather than sampled.
# ---------------------------------------------------------------------------
kn_2 = redback.transient.Kilonova.from_lightcurvelynx(
    name="lynx_kilonova_extinction",
    data=chosen_lc_2,
    data_mode="flux_density",
    use_phase_model=True,
)
kn_2.plot_data(show=False)
plt.savefig("lynx_kilonova_extinction_data.png", dpi=150, bbox_inches="tight")
plt.close()
print("Extinction data plot saved.")

first_det_mjd = float(chosen_lc_2[chosen_lc_2["detection"]]["mjd"].min())
t0_prior_window = 20.0  # days to search before first detection

priors_2 = redback.priors.get_priors(model="one_component_kilonova_model")
priors_2["redshift"] = float(injected_redshift_2)
priors_2["t0"] = bilby.core.prior.Uniform(
    minimum=first_det_mjd - t0_prior_window,
    maximum=first_det_mjd,
    name="t0",
    latex_label="$t_0$ (MJD)",
)
priors_2["av_host"] = bilby.core.prior.Uniform(
    minimum=0.0, maximum=1.0, name="av_host", latex_label="$A_V^{\\rm host}$"
)

result_2 = redback.fit_model(
    transient=kn_2,
    model="t0_kilonova_extinction",
    sampler=sampler,
    model_kwargs=dict(
        frequency=kn_2.filtered_frequencies,
        output_format="flux_density",
        base_model="one_component_kilonova_model",
        av_mw=0.0,  # fix MW extinction; in practice query from e.g. dustmaps
    ),
    prior=priors_2,
    sample="rslice",
    nlive=500,
    resume=True,
)

print(f"\nInjected av_host = {INJECTED_AV_HOST:.2f} mag")
print(f"Injected t0 (MJD) = {injected_t0_mjd_2:.2f}")
print("Recovered posteriors:")
print(result_2.posterior[["av_host", "t0"]].describe())
