import matplotlib.pyplot as plt

import redback


sampler = 'dynesty'
# lots of different models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'mergernova'

kne = 'at2017gfo'
path = 'KNDir'
# gets the photometry data for AT2017gfo, the KN associated with GW170817
data = redback.get_data.get_kilonova_data_from_open_transient_catalog_data(transient=kne)
# creates a GRBDir with GRB
kilonova = redback.kilonova.Kilonova.from_open_access_catalogue(name=kne, data_mode="flux_density")
# kilonova.flux_density_data = True
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 8))
bands = ["g"]
kilonova.plot_multiband(figure=fig, axes=axes, filters=["g", "r", "i", "z", "y", "J"])

# use default priors
priors = redback.priors.get_priors(model=model)
priors['redshift'] = 1e-2
priors['l0'] = 1e47
priors['n_ism'] = 1e-3
priors['tau_sd'] = 1e3
priors['thermalisation_efficiency'] = 0.3

model_kwargs = dict(frequencies=redback.utils.bands_to_frequencies(bands), output_format='flux_density')

result = redback.fit_model(name=kne, transient=kilonova, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           path=path, prior=priors, data_mode='flux_density', sample='rslice', nlive=200)
result.plot_corner()
# returns a Kilonova result object
result.plot_lightcurve(random_models=1000)
