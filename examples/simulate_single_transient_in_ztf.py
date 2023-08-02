# This example will show how to simulate a kilonova with ZTF pointings table.
import redback
from redback.simulate_transients import SimulateOpticalTransient
import matplotlib.pyplot as plt

# We now set up the parameters for the kilonova model.
# We will use the one_component_kilonova_model implemented in redback but any optical model
# in Redback will work or a user could pass their own model.

model_kwargs = {}
# Load the default prior for this model in redback and sample from it to get 1 set of parameters.
# We fix a few parameters here to create a nice looking kilonova
parameters = redback.priors.get_priors(model='one_component_kilonova_model').sample()
parameters['mej'] = 0.05
parameters['t0_mjd_transient'] = 58288
parameters['redshift'] = 0.005
parameters['t0'] = parameters['t0_mjd_transient']
parameters['temperature_floor'] = 3000
parameters['kappa'] = 1
parameters['vej'] = 0.2

# We also can place the transient on the sky by setting the ra and dec parameters.
# This will be randomly set from the pointing if not given.
parameters['ra'] = 3.355395
parameters['dec'] = 0.5820673

# We now simulate a kilonova using the SimulateOpticalTransient class.
# Now specifying a survey string, which will load the pointings table from the tables directory in redback.
# These tables will need to be downloaded from zenodo using the redback.utils if not already present.
# Please look at the documentation for more details.
# Here we will also add some extra noise to the observations.
# This noise source is proportional to the model flux and is added in quadrature to survey noise.
# The error factor can be modified but is 0.02 by default.
kn_sim = SimulateOpticalTransient.simulate_transient_in_ztf(model='one_component_kilonova_model',
                                                              survey='ztf',
                                                              parameters=parameters, model_kwargs=model_kwargs,
                                                              end_transient_time=15., snr_threshold=5.,
                                                            add_source_noise=True)
# We can print the observations that were simulated to see what the data looks like.
print(kn_sim.observations)

# We can also save the observations to a file using the save_transient method.
# This will save the observations to a csv file in a 'simulated' directory alongside the csv file
# specifying the injection parameters.
kn_sim.save_transient(name='my_kilonova')

# We can now load the data into a transient object for plotting and other tasks such as inference.
# Note that this will only the 'detected' data points. The user can add the non-detections back in if they wish.
kn_object = redback.transient.Kilonova.from_simulated_optical_data(name='my_kilonova', data_mode='magnitude')
ax = kn_object.plot_data(show=False)
ax.set_ylim(23, 17)
plt.show()