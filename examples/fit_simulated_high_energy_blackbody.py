"""
End-to-end example: simulate a high-energy blackbody spectrum and recover parameters.
"""

import numpy as np

import redback.priors
from redback.simulate_transients import SimulateHighEnergyTransient
from redback.spectral.dataset import SpectralDataset
from redback.sampler import fit_model
from redback.utils import calc_credible_intervals
from redback.transient_models.spectral_models import blackbody_high_energy


def static_blackbody_model(time, frequency, redshift, r_photosphere_rs, kT, **kwargs):
    """
    Wrapper for SimulateHighEnergyTransient: ignore time, convert frequency -> energy.
    Returns flux density in mJy.
    """
    keV_to_Hz = 2.417989e17
    energies_keV = np.asarray(frequency, dtype=float) / keV_to_Hz
    return blackbody_high_energy(
        energies_keV=energies_keV,
        redshift=redshift,
        r_photosphere_rs=r_photosphere_rs,
        kT=kT,
    )


# Simulation setup
energy_edges = np.array([0.3, 0.5, 1.0, 2.0, 5.0])  # keV
time_range = (0.0, 100.0)  # seconds
true_params = dict(redshift=0.1, r_photosphere_rs=1.0, kT=1.5)

sim = SimulateHighEnergyTransient(
    model=static_blackbody_model,
    parameters=true_params,
    energy_edges=energy_edges,
    time_range=time_range,
    effective_area=150.0,
    background_rate=0.05,
    time_resolution=0.1,
    seed=1234,
)

time_bins = np.linspace(time_range[0], time_range[1], 101)
dataset = SpectralDataset.from_simulator(sim=sim, time_bins=time_bins)
dataset.set_active_interval(0.3, 5.0)

dataset.plot_spectrum_data(
    show=False,
    save=True,
    filename="sim_bb_spectrum.png",
    min_counts=5,
    xscale="linear",
    plot_background=False,
    xlim=(0.3, 5.0),
)

# Fit model
model = "blackbody_high_energy"
prior = redback.priors.get_priors("blackbody_high_energy")
prior["redshift"] = true_params["redshift"]

result = fit_model(
    transient=dataset,
    model=model,
    prior=prior,
    sampler="dynesty",
    statistic="auto",
    nlive=300,
    plot=False,
    clean=True,
    resume=False,
)

result.plot_corner(filename="sim_bb_corner.png", show=False)

# Summarize recovery
posterior = result.posterior
sample_size = min(len(posterior), 1000)
samples = posterior.sample(n=sample_size, random_state=0)

for key in ("r_photosphere_rs", "kT"):
    vals = samples[key].to_numpy()
    lo, hi, med = calc_credible_intervals(vals, interval=0.68)
    print(f"{key}: true={true_params[key]:.3g}, median={med:.3g} (+{hi-med:.3g}/-{med-lo:.3g})")

dataset.plot_spectrum_fit(
    model=model,
    posterior=result.posterior,
    model_kwargs=None,
    filename="sim_bb_fit.png",
    show=False,
    save=True,
    min_counts=5,
    xscale="linear",
    plot_background=False,
    xlim=(0.3, 5.0),
)
