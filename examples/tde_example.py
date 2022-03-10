import matplotlib.pyplot as plt

import redback

sampler = 'dynesty'
model = 'tde_analytical'
tde = "PS18kh"

data = redback.get_data.get_tidal_disruption_event_data_from_open_transient_catalog_data(tde)

tidal_disruption_event = redback.tde.TDE.from_open_access_catalogue(tidal_disruption_event, data_mode='flux_density')

# lets make a plot of the data
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
tidal_disruption_event.plot_multiband(figure=fig, axes=axes, filters=["J", "H", "g", "i"])

# we are now only going to fit u and r bands. We set this using the transient object and the active bands attribute.
# By default all data is used, this is just for speed in the example or if you only trust some of the data/physics.
bands = ["u", "r"]
tidal_disruption_event.active_bands = bands

# use default priors
priors = redback.priors.get_priors(model=model)
# we know the redshift for PS18kh so we just fix the prior for the redshift to the known value.
# We can do this through the metadata that was downloaded alongside the data, or if you just know it.
priors['redshift'] = tidal_disruption_event.redshift

model_kwargs = dict(frequencies=redback.utils.bands_to_frequency(bands), output_format='flux_density')

# returns a tde result object
result = redback.fit_model(name=tde, transient=tidal_disruption_event, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, data_mode='flux_density', sample='rslice', nlive=200, resume=False)

result.plot_corner()

result.plot_lightcurve(random_models=1000)
