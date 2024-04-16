# In this example, we show how one can use models implemented in sncosmo to fit data.
# There is a general interface in redback that provides a redback-friendly interface to all models in sncosmo.

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