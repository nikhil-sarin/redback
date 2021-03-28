import redback
import bilby

sampler = 'pymultinest'
# lots of different models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'kilonova_nicholl21'

kne = ['at2017gfo']
path = 'KNDir'
# gets the photometry data for AT2017gfo, the KN associated with GW170817
data = redback.getdata.get_open_transient_catalog_data(transient=kne, transient_type='kilonova')
# creates a GRBDir with GRB

# use default priors
priors = redback.priors.get_priors(model=model, data_mode='photometry')

result = redback.fit_model(data_type='kilonova', name=kne, model=model, sampler=sampler,
                           path=path, prior=priors, data_mode='photometry', likelihood='gaussian')
# returns a Kilonova result object
result.plot_lightcurve(random_models=1000)
