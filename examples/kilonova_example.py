import matplotlib.pyplot as plt

import redback
import bilby

sampler = 'pymultinest'
# lots of different models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'kilonova_nicholl21'

kne = 'at2017gfo'
path = 'KNDir'
# gets the photometry data for AT2017gfo, the KN associated with GW170817
data = redback.getdata.get_open_transient_catalog_data(transient=kne, transient_type='kilonova')
# creates a GRBDir with GRB
kilonova = redback.kilonova.Kilonova.from_open_access_catalogue(name=kne, data_mode="photometry")
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 8))
kilonova.flux_density_data = True
kilonova.plot_multiband(figure=fig, axes=axes, filters=["g", "r", "i", "z", "y", "J"])

# use default priors
priors = redback.priors.get_priors(model=model, data_mode='photometry')

result = redback.fit_model(data_type='kilonova', name=kne, model=model, sampler=sampler,
                           path=path, prior=priors, data_mode='photometry', likelihood='gaussian')
# returns a Kilonova result object
result.plot_lightcurve(random_models=1000)
