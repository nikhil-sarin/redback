import numpy as np

import bilby
import george

import redback

# We implemented many models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'full_magnetar'
magmodel = bilby.likelihood.function_to_george_mean_model(redback.model_library.all_models_dict[model])

GRB = '070809'
# Flux density, flux data
redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb=GRB, data_mode="flux")
# creates a GRBDir with GRB

# create Luminosity data
ag = redback.afterglow.SGRB.from_swift_grb(name=GRB, data_mode='flux',
                                                  truncate=True, truncate_method="prompt_time_error")

# uses an analytical k-correction expression to create luminosity data if not already there.
# Can also use a numerical k-correction through CIAO
ag.analytical_flux_to_luminosity()

# use default priors
priors = redback.priors.get_priors(model=model)

# Now set up the GP likelihood and prior.
# Fitting with a GP requires small modifications to the prior so that George understands
# And then give an estimate of the GP parameters as well.
mean_model = magmodel(**priors.sample())
for k, v in priors.copy().items():
    priors[f"mean:{k}"] = v
    del priors[k]
# Let's set a prior on the kernel parameters
priors["kernel:k1:log_constant"] = Uniform(-50, 30, name="log_A", latex_label=r"$\ln A$")
priors["kernel:k2:metric:log_M_0_0"] = Uniform(-10, 10000, name="log_M_0_0", latex_label=r"$\ln M_{00}$")

# Set up a simple kernel
kernel = 2 * kernels.ExpSquaredKernel(100, ndim=1)
# We must use a George specific likelihood available in bilby.
# Note that GPs can only handle symmetric errors in george so we make an approximation.

likelihood = bilby.core.likelihood.GeorgeLikelihood(
    kernel=kernel, mean_model=mean_model, t=ag.x, y=ag.y, yerr=np.max(ag.y_err, axis=0))

likelihood.parameters = priors.sample()
print(likelihood.parameters)

# The sampling interface is now directly via bilby.
# You could also use the redback interface but
# none of the other redback result functionality will be usable for such analyses so easiest to use bilby directly.
# Standard redback/bilby things
label = 'GP'
outdir = 'testgp'

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label=label,
    sampler="nestle",
    sample="rslice",
    nlive=300,
    clean=False,
)

# Let's see how we did with some plots.
result.plot_corner()

x = np.linspace(ag.x[0], ag.x[-1]+20, 100)
likelihood.gp.compute(ag.x, np.max(ag.y_err, axis=0))
pred_mean, pred_var = likelihood.gp.predict(ag.y, x, return_var=True)
pred_std = np.sqrt(pred_var)

# Let's plot things
ax = ag.plot_data(show=False)

#First we plot the GP prediction
color = "#ff7f0e"
ax.plot(x, pred_mean, color=color, label=r"GP $1\sigma$ prediction")
ax.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3, edgecolor="none")

# Now we plot just the mean model for 50 random posterior samples
samples = [result.posterior.iloc[np.random.randint(len(result.posterior))] for _ in range(50)]
for sample in samples:
    likelihood.set_parameters(sample)
    if not isinstance(likelihood.mean_model, (float, int)):
        trend = likelihood.mean_model.get_value(x)
    ax.loglog(x, trend, color="blue", alpha=0.05)
ax.legend()

# Voila, you can now fit a GP + underlying physical model to your data with redback.
