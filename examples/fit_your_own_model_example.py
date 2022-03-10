import bilby.core.prior

import redback
from bilby.core.prior import LogUniform, Uniform


# If your favourite model is not implemented in redback. You can still fit it using redback!
# Now instead of passing a string as the model. You need to pass a python function.

# let's make a simple power law. A power law model is already in redback but this is just an example.
# You can make up any model you like.

# time must be the first element.
def my_favourite_model(time, l0, alpha, **kwargs):
    return l0 * time ** alpha


model = my_favourite_model

# now let's use this model and fit it to the data
GRB = '070809'

# Flux density, flux data
redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb=GRB, data_mode="flux")
# creates a GRBDir with GRB

# create Luminosity data
afterglow = redback.afterglow.SGRB.from_swift_grb(name=GRB, data_mode='flux',
                                                  truncate=True, truncate_method="prompt_time_error")

afterglow.plot_data()

# uses an analytical k-correction expression to create luminosity data if not already there.
# Can also use a numerical k-correction through CIAO
afterglow.analytical_flux_to_luminosity()

# You need to create your own priors for this new model.
# The model has two parameters l0 and alpha. We use bilby priors for this
priors = bilby.core.prior.PriorDict()
priors['l0'] = LogUniform(1e-10, 1e5, 'l0', latex_label=r'$l_{0}$')
priors['alpha'] = Uniform(-7, 0, 'alpha', latex_label=r'$\alpha$')

# Call redback.fit_model to run the sampler and obtain GRB result object
result = redback.fit_model(name=GRB, model=model, sampler='dynesty', nlive=200, transient=afterglow,
                           prior=priors, sample='rslice', resume=False)

result.plot_lightcurve(random_models=1000, model=my_favourite_model)
