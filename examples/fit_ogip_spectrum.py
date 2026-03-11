"""
Fit an OGIP PHA/RMF spectrum with redback's spectral fitting API.

Uses publicly available Fermi/GBM BGO example data (simulated power-law GRB spectrum)
included in redback's example_data directory. The combined response file (.rsp) provides
the redistribution matrix and effective area in a single FITS file, as is common for
gamma-ray detectors. The data here is from ThreeML's repository.
"""

import os

import redback.priors
from redback.transient.spectral import CountsSpectrumTransient
from redback.sampler import fit_model
from redback.utils import calc_credible_intervals
import numpy as np

# Locate the example data bundled with redback
_here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_here, "example_data")

spec = CountsSpectrumTransient.from_ogip(
    pha=os.path.join(data_dir, "ogip_powerlaw.pha"),
    bkg=os.path.join(data_dir, "ogip_powerlaw.bak"),
    rmf=os.path.join(data_dir, "ogip_powerlaw.rsp"),  # combined RSP (RMF + ARF)
    name="gbm_grb",
)

dataset = spec.dataset
# Restrict to a well-calibrated energy interval (Fermi/GBM BGO: ~200 keV – 40 MeV)
dataset.set_active_interval(200.0, 40000.0)

# Plot the raw count spectrum
spec.plot_data(
    filename="spec_data.png",
    show=False,
    save=True,
    min_counts=5,
    xscale="log",
    yscale="log",
    xlim=(200.0, 40000.0),
)

# We could also plot the count-rate light curve (requires a separate light curve file)
# from redback.spectral.io import read_lc
# lc = read_lc("path/to/source.lc")
# spec.plot_lightcurve(lc=lc, filename="spec_lc.png", show=False, save=True)

model = "powerlaw_high_energy"
prior = redback.priors.get_priors(model)
# Fix redshift to zero for a GRB at unknown redshift (spectral shape only)
prior["redshift"] = 0.0

result = fit_model(
    transient=spec,
    model=model,
    prior=prior,
    sampler="nestle",
    statistic="auto",   # auto-selects wstat (background available)
    nlive=500,
    plot=False,
    clean=True,
    resume=False,
)

result.plot_corner(filename="spec_corner.png", show=False)

# Compute band flux from posterior samples
posterior = result.posterior
sample_size = min(len(posterior), 500)
flux_samples = posterior.sample(n=sample_size, random_state=0)

fluxes = []
for _, row in flux_samples.iterrows():
    params = row.to_dict()
    fluxes.append(
        dataset.compute_band_flux(
            model=model,
            parameters=params,
            energy_min_keV=200.0,
            energy_max_keV=40000.0,
            unabsorbed=False,
        )
    )

fluxes = np.asarray(fluxes)
lo, hi, med = calc_credible_intervals(fluxes, interval=0.68)
print(f"200 keV – 40 MeV flux: {med:.3e} (+{hi-med:.3e}/-{med-lo:.3e}) erg/cm²/s")

# Plot best-fit spectrum with residuals
spec.plot_fit(
    model=model,
    posterior=result.posterior,
    filename="spec_fit.png",
    show=False,
    save=True,
    min_counts=5,
    xscale="log",
    xlim=(200.0, 40000.0),
)
