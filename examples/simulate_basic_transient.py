# This example shows how to simulate a generic transient
# i.e., not using a pointing table to create lightcurves of a real survey but rather just random samples of a transient lightcurve.
# Since Redback models are callable functions this is easy to do yourself, but this example shows how to do it using the SimulatedGenericTransient class which eases the process.
import redback
import numpy as np
from redback.simulate_transients import SimulateGenericTransient
# Let's set up the parameters of our model
# We will use the one_component_kilonova_model implemented in redback but any optical model in Redback will work.
bands = ['lsstg', 'lsstr', 'lssti']
times = np.linspace(0.2, 10, 20) # days
model_kwargs = {'bands':bands, 'output_format':'magnitude'}
num_of_data_points = 40 # the number of data points to generate
parameters = {}
parameters['mej'] = 0.05
parameters['redshift'] = 0.01
parameters['temperature_floor'] = 3000
parameters['kappa'] = 1
parameters['vej'] = 0.2
kn_obs = SimulateGenericTransient(model='one_component_kilonova_model', parameters=parameters,
                                  times=times, data_points=num_of_data_points, model_kwargs=model_kwargs,
                                  multiwavelength_transient=True, noise_term=0.02)
# We can print out the data to see what it looks like
print(kn_obs.data)

# But why print when you can load this up into a redback transient and plot.
kilonova = redback.transient.Kilonova(name='my_kilonova', magnitude=kn_obs.data['output'].values,
                                      time=kn_obs.data['time'].values, data_mode='magnitude',
                                      magnitude_err=kn_obs.data['output_error'].values, bands=kn_obs.data['band'].values)
kilonova.plot_data()
# Congratulations, you have made a generic transient lightcurve!