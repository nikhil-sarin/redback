"""
Fit an OGIP PHA/RMF/ARF spectrum with redback's spectral fitting API.
"""

import numpy as np

import redback.priors
from redback.spectral.dataset import SpectralDataset
from redback.spectral.io import read_lc
from redback.sampler import fit_model
from redback.utils import calc_credible_intervals


dataset = SpectralDataset.from_ogip(
    pha="ep11900012809wxt3s2.pha",
    bkg="ep11900012809wxt3s2bk.pha",
    rmf="ep11900012809wxt3.rmf",
    arf="ep11900012809wxt3s2.arf", name='ep_event')

dataset.set_active_interval(0.3, 5.0)

lc = read_lc("ep11900012809wxt3s2.lc")
SpectralDataset.plot_lightcurve(
    lc=lc,
    filename="spec_lc.png",
    show=False,
    save=False,
    xscale="linear",
    yscale="linear",
    min_counts=5,
)

dataset.plot_spectrum_data(show=False, save=False, filename="spec_data.png", min_counts=5,
                           xscale="linear", plot_background=False,
                           xlim=(0.3, 5.0), ylim=(1e-3, 2e-1))

model = "tbabs_powerlaw_high_energy"
prior = redback.priors.get_priors("tbabs_powerlaw_high_energy")
prior['redshift'] = 2.294

result = fit_model(
    transient=dataset,
    model=model,
    prior=prior,
    sampler="pymultinest",
    statistic="wstat",
    nlive=500,
    plot=False,
    clean=False,
    resume=False,
)

result.plot_corner(filename="spec_corner.png", show=False)

posterior = result.posterior
sample_size = min(len(posterior), 500)
flux_samples = posterior.sample(n=sample_size, random_state=0)

absorbed_fluxes = []
unabsorbed_fluxes = []
for _, row in flux_samples.iterrows():
    params = row.to_dict()
    absorbed_fluxes.append(
        dataset.compute_band_flux(model=model, parameters=params, energy_min_keV=0.5, energy_max_keV=10.0)
    )
    unabsorbed_fluxes.append(
        dataset.compute_band_flux(
            model=model,
            parameters=params,
            energy_min_keV=0.5,
            energy_max_keV=10.0,
            unabsorbed=True,
        )
    )

absorbed_fluxes = np.asarray(absorbed_fluxes)
unabsorbed_fluxes = np.asarray(unabsorbed_fluxes)
abs_lo, abs_hi, abs_med = calc_credible_intervals(absorbed_fluxes, interval=0.68)
unabs_lo, unabs_hi, unabs_med = calc_credible_intervals(unabsorbed_fluxes, interval=0.68)

print(
    f"Absorbed 0.5-10 keV flux: {abs_med:.3e} (+{abs_hi-abs_med:.3e}/-{abs_med-abs_lo:.3e}) erg/cm^2/s"
)
print(
    f"Unabsorbed 0.5-10 keV flux: {unabs_med:.3e} (+{unabs_hi-unabs_med:.3e}/-{unabs_med-unabs_lo:.3e}) "
    "erg/cm^2/s"
)

dataset.plot_spectrum_fit(
    model=model,
    posterior=result.posterior,
    model_kwargs=None,
    filename="spec_fit.png",
    show=False,
    save=True,
    min_counts=5,
    xscale="linear",
    plot_background=False,
    xlim=(0.3, 5.0),
    ylim=(1e-3, 2e-1),
)
