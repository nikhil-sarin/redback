from bilby.core.prior import PriorDict, Uniform, Constraint
import matplotlib.pyplot as plt
import numpy as np

import redback

# In this example we show two ways to handle non-detections in redback:
# 1. Using prior constraints (the original method shown below)
# 2. Using the integrated non-detection support (see handling_non_detections.ipynb)
#    where you pass `detections` and `upper_limit_sigma` to the transient object
#    and redback automatically handles upper limits in plotting and fitting.

model = 'evolving_magnetar'
priors = redback.priors.get_priors(model=model)

function = redback.model_library.all_models_dict[model]

# Define limits based on your non-detection
# We do not observe anything above a flux of 1e5mJy after 28days
day_lim =  np.array([28.])
fd_flux_upper_lim = 1e5


def scalarize_sample(sample):
    return {key: np.atleast_1d(value)[0] for key, value in sample.items()}


def constraint_on_flux_time_from_nondetection(params):
    scalar_params = scalarize_sample(params)
    fd_flux = function(day_lim, **scalar_params)
    converted_params = params.copy()
    converted_params['x'] = fd_flux/fd_flux_upper_lim # constrains flux to less than upper limit at specified day
    return converted_params

priors_constrained = PriorDict(conversion_function=constraint_on_flux_time_from_nondetection)
for key in priors:
    priors_constrained[key] = priors[key] 
priors_constrained['x'] = Constraint(0,1)


def draw_scalar_sample(prior):
    return scalarize_sample(prior.sample(size=1))


# Plot draws from original and constrained prior distributions.
time = np.linspace(1,100,50)
for ii in range(100):
    original_prior_draws = function(time, **draw_scalar_sample(priors))
    constrained_prior_draws = function(time, **draw_scalar_sample(priors_constrained))
    plt.semilogy(time, original_prior_draws, c='gray', alpha=0.25)
    plt.semilogy(time, constrained_prior_draws, c='red', alpha=0.25)

plt.axvline(day_lim, linestyle='-', c='k')
plt.axhline(fd_flux_upper_lim, linestyle='--', c='k')
plt.xlim(1,100)
plt.ylabel('Flux density [mJy]')
plt.xlabel('Time [days]')
plt.show()
