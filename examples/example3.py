import redback
import bilby
sampler = 'pymultinest'
#lots of different models implemented, including
#afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'band_function'

prompt = ['080430']
path = 'GRBDir'
#gets the photometry data for AT2017gfo, the KN associated with GW170817
redback.getdata.get_prompt_data(prompt, data_mode = 'counts', path = path)
#creates a GRBDir with GRB

#use default priors
priors = redback.priors(model = model, data_mode = 'counts')

result = redback.fit_model(data_type = 'prompt',name = prompt, model = model,
                           sampler = sampler,path = path, prior = priors,
                           data_mode = 'counts', likelihood = 'poisson',
                            bin_size = 'BAT')
#returns a GRB prompt result object
result.plot_lightcurve(random_models = 1000)
