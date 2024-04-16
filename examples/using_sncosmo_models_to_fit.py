# In this example, we show how one can use models implemented in sncosmo to fit data/or plot lightcurves.
# There is a general interface in redback that provides a redback-friendly interface to all models in sncosmo.

import bilby
import redback
import numpy as np
import matplotlib.pyplot as plt

# Let's first plot the salt2 model using the redback interface

# We first set up the parameters for the model.
sncosmo_model = 'salt2' #must be the same as the sncosmo registered model name
redshift = 0.1
start_time = 55570
time = np.linspace(0.1, 65, 100) + start_time


# We set up a dictionary for any kwargs required by redback.
# These are similar to the kwargs required by other redback models
# e.g., an output format or bands or frequency.
kwargs = {'sncosmo_model': sncosmo_model, 'frequency': 4e14, 'output_format': 'flux_density'}

# We also set the peak time for the model in MJD
kwargs['peak_time'] = 55589.0

# We now need to pass in any extra arguments required by the model itself. For salt2 these are x0, x1, and c
model_kwargs = {'x0': 0.8, 'x1': 0.9, 'c': 0.3}
outs = redback.transient_models.supernova_models.sncosmo_models(time=time, redshift=redshift,
                                                                model_kwargs=model_kwargs, **kwargs)

# Let's plot.

plt.semilogy(time - start_time, outs, label='salt2')
plt.show()

# We can also call the model to evaluate magnitudes instead of flux densities.
kwargs = {'sncosmo_model': sncosmo_model, 'bands': 'ztfr', 'output_format': 'magnitude'}
kwargs['peak_time'] = 55589.0
outs = redback.transient_models.supernova_models.sncosmo_models(time=time, redshift=redshift,
                                                                model_kwargs=model_kwargs, **kwargs)
plt.plot(time - start_time, outs)
plt.gca().invert_yaxis()
plt.show()

# Now that we can see the model works. Let's try to fit some data.
# Let's fit some data to a ztf data
transient = 'ZTF20aamdsjv'
data = redback.get_data.get_lasair_data(transient=transient, transient_type='supernova')

# Set up the redback transient object.
sn = redback.transient.Supernova.from_lasair_data(transient, use_phase_model=True,
                                                  data_mode='magnitude')

# Let's plot the data to ensure everything is set up correctly.
sn.plot_data()

# Now we set up the model and priors. For SNCOSMO model fitting, this interface is slightly different.
# To the standard in redback as we can directly sample t0 without using a phase model.
# Note you will be able get lightcurves from t0 if you just set t0 to zero.
sncosmo_model = 'salt2' #must be the same as the sncosmo registered model name

priors = bilby.core.prior.PriorDict()
priors['redshift'] = 0.061

# Set a prior on t0 to be within 100 days before the first observation
priors['t0'] = bilby.core.prior.Uniform(sn.x[0] - 100, sn.x[0] - 0.01, 't0', latex_label=r'$t_0$')

# Set a prior on the peak time to be within 10 days of the maximum (minimum magnitude)
data_peak = sn.x[np.argmin(sn.y)]
priors['peak_time'] = bilby.core.prior.Uniform(data_peak - 10, data_peak + 10, 'peak_time', latex_label=r'$t_{\rm peak}$')

# Set a prior on the x0, x1, and c parameters i.e., the salt2 model parameters
priors['x0'] = bilby.core.prior.Uniform(1e-10, 1e-1, 'x0', latex_label=r'$x_0$')
priors['x1'] = bilby.core.prior.Normal(0, 1, 'x1', latex_label=r'$x_1$')
priors['c'] = bilby.core.prior.Normal(0, 0.1, 'c', latex_label=r'$c$')

# We set up a dictionary for any kwargs required by redback.
# To make sure redback understands which keywords are model parameters, we need to pass an extra list with the names of the model parameters.
kwargs = {'sncosmo_model': sncosmo_model, 'bands': sn.filtered_sncosmo_bands, 'output_format': 'magnitude',
          'model_kwarg_names': ['x0', 'x1', 'c']}

# Let's fit. Again the interface is similar to the normal interface for redback.
result = redback.fit_model(transient=sn, model='sncosmo_models', prior=priors, model_kwargs=kwargs,
                           sampler='ultranest', nlive=200, plot=False)
ax = result.plot_lightcurve(random_models=50, show=False)
ax.set_xscale('linear')
ax.set_yscale('linear')
plt.show()

# Congratulations, you now have a nice fit with the SNCOSMO salt2 model through redback.