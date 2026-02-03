"""
Fit an OGIP PHA/RMF/ARF spectrum with redback's spectral fitting API.
"""

from redback.spectral.dataset import SpectralDataset
from redback.spectral.io import read_lc
from redback.transient_models import spectral_models
from redback.sampler import fit_model


def powerlaw_wrapper(times, **kwargs):
    frequencies = kwargs.get("frequency")
    energies_keV = frequencies / 2.417989e17
    return spectral_models.powerlaw_high_energy(
        energies_keV=energies_keV,
        redshift=0.0,
        log10_norm=-2.0,
        alpha=-1.5,
    )


dataset = SpectralDataset.from_ogip(
    pha="ep11900012809wxt3s2.pha",
    bkg="ep11900012809wxt3s2bk.pha",
    rmf="ep11900012809wxt3.rmf",
    arf="ep11900012809wxt3s2.arf",
)
dataset.set_active_interval(0.3, 5.0)

lc = read_lc("ep11900012809wxt3s2.lc")
SpectralDataset.plot_lightcurve(
    time=lc.time,
    rate=lc.rate,
    error=lc.error,
    show=False,
    save=True,
    filename="spec_lc.png",
)

dataset.plot_spectrum_data(show=False, save=True, filename="spec_data.png", min_counts=5,
                           xscale="linear", plot_background=False,
                           xlim=(0.3, 5.0), ylim=(1e-3, 2e-1))

result = fit_model(
    transient=dataset,
    model=powerlaw_wrapper,
    sampler="dynesty",
    statistic="wstat",
    nlive=500,
)

print(result)

dataset.plot_spectrum_fit(
    model=powerlaw_wrapper,
    posterior=result.posterior,
    model_kwargs=None,
    filename="spec_fit.png",
    show=False,
    save=True,
    min_counts=20,
)
