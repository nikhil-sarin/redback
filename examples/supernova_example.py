import matplotlib.pyplot as plt

import redback


sampler = 'dynesty'

model = "skew_exponential"

sne = "SN2011kl"

data = redback.getdata.get_supernova_data_from_open_transient_catalog_data(sne)

supernova = redback.supernova.Supernova.from_open_access_catalogue(name=sne, data_mode='flux_density')
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
bands = ["g", "i"]
supernova.plot_multiband(figure=fig, axes=axes, filters=["J", "H", "g", "i"])

# use default priors
priors = redback.priors.get_priors(model=model)
priors['redshift'] = 1e-2
model_kwargs = dict(frequencies=redback.utils.bands_to_frequencies(bands), output_format='flux_density')

result = redback.fit_model(name=sne, transient=supernova, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, data_mode='flux_density', sample='rslice', nlive=200, resume=False)
result.plot_corner()
# returns a Kilonova result object
result.plot_lightcurve(random_models=1000)
