import numpy as np

import bilby
import celerite.terms

import redback

# We implemented many models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'evolving_magnetar'
EvolvingMagnetar = bilby.likelihood.function_to_celerite_mean_model(redback.model_library.all_models_dict[model])

GRB = '070809'
# Flux density, flux data
redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb=GRB, data_mode="flux")
# creates a GRBDir with GRB

# create Luminosity data
afterglow = redback.afterglow.SGRB.from_swift_grb(name=GRB, data_mode='flux',
                                                  truncate=True, truncate_method="prompt_time_error")

# uses an analytical k-correction expression to create luminosity data if not already there.
# Can also use a numerical k-correction through CIAO
afterglow.analytical_flux_to_luminosity()
# afterglow.plot_data()

# use default priors
priors = redback.priors.get_priors(model=model)
mean_model = EvolvingMagnetar(**priors.sample())
for k, v in priors.copy().items():
    priors[f"mean:{k}"] = v
    del priors[k]
priors["kernel:log_S0"] = bilby.prior.Uniform(-5.0, 10, name="log_S0")
priors["kernel:log_Q"] = bilby.prior.Uniform(np.log(np.sqrt(2)), 10, name="log_Q")
priors["kernel:log_omega0"] = bilby.prior.Uniform(-np.log(afterglow.x[-1] - afterglow.x[0]), 10, name="log_omega0")

kernel = celerite.terms.SHOTerm(log_S0=0.0, log_Q=0.0, log_omega0=0.0)

likelihood = bilby.likelihood.CeleriteLikelihood(
    kernel=kernel, mean_model=mean_model, t=afterglow.x, y=afterglow.y, yerr=afterglow.y_err[0])

likelihood.parameters = priors.sample()
print(likelihood.parameters)

# Call redback.fit_model to run the sampler and obtain GRB result object
result = redback.fit_model(model=model, likelihood=likelihood, sampler='dynesty', nlive=400, transient=afterglow,
                           prior=priors, sample='rslice', resume=True)

result.plot_corner()
result.plot_lightcurve(random_models=100)
result.plot_residual()
