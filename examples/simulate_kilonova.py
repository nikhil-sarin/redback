# This example will show how to simulate a kilonova with a user-generated pointings table

import numpy as np
import redback
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateOpticalTransient
from redback.model_library import all_models_dict


# We first set up a table that contains the pointings of the telescope. 
# This requires setting the number of observations in each filter, average cadence, the cadence scatter, 
# and the limiting magnitudes for each filter.
num_obs = {'lsstg': 10, 'lsstr':10, 'lssti':10}
average_cadence = {'lsstg': 1.5, 'lsstr': 5.0, 'lssti': 2.5}
cadence_scatter = {'lsstg': 0.5, 'lsstr':0.5, 'lssti':0.5}
limiting_magnitudes = {'lsstg': 25.0, 'lsstr': 24.5, 'lssti': 23.0}

# We now use redback to make a pointings table from the above information
pointings = redback.simulate_transients.make_pointing_table_from_average_cadence(
    ra=1.0, dec=1.5, num_obs=num_obs, average_cadence=average_cadence,
    cadence_scatter=cadence_scatter, limiting_magnitudes=limiting_magnitudes, initMJD=59581.0)
print(pointings)

# We now set up the parameters for the kilonova model. 
# We will use the one_component_kilonova_model implemented in redback but any optical model 
# in Redback will work or a user could pass their own model.

model_kwargs = {}
# Load the default prior for this model in redback and sample from it to get 1 set of parameters. 
# We fix a few parameters here to create a nice looking kilonova
parameters = redback.priors.get_priors(model='one_component_kilonova_model').sample()
parameters['mej'] = 0.05
parameters['t0_mjd_transient'] = 59582.0
parameters['redshift'] = 0.01
parameters['t0'] = parameters['t0_mjd_transient']
parameters['temperature_floor'] = 3000
parameters['kappa'] = 1
parameters['vej'] = 0.2

# We also can place the transient on the sky by setting the ra and dec parameters. 
# This will be randomly set from the pointing if not given.
parameters['ra'] = 1.0
parameters['dec'] = 1.5

# We now simulate a kilonova using the SimulateOpticalTransient class.
kn_sim = SimulateOpticalTransient.simulate_transient(model='one_component_kilonova_model',
                                       parameters=parameters, pointings_database=pointings,
                                       survey=None, model_kwargs=model_kwargs, end_transient_time=10.)

# We can print the observations that were simulated to see what the data looks like. 
print(kn_sim.observations)

# We can also save the observations to a file using the save_transient method.
kn_sim.save_transient(name='my_kilonova')

# # We can now load the data into a transient object for plotting and other tasks such as inference.
# kn_object = redback.transient.Kilonova.from_simulated_data(name='230421', data_mode='magnitude',
#                                                            time=kn_sim.inference_observations['time (days)'].values,magnitude=kn_sim.inference_observations['magnitude'].values,
#     magnitude_err=kn_sim.inference_observations['e_magnitude'].values, bands=kn_sim.inference_observations['band'].values)
# ax = kn_object.plot_data()
# plt.show()
