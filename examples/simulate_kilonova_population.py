# This example will show how to simulate a collection of kilonovae with Rubin
import redback
from redback.simulate_transients import SimulateOpticalTransient
import matplotlib.pyplot as plt
import bilby
import numpy as np

# We now set up the parameters for the kilonova population. An easy way to do this is to use the redback prior.
# We will use the one_component_kilonova_model implemented in redback but any optical model will work.
# A user could also pass their own model
np.random.seed(1346)
prior = redback.priors.get_priors(model='one_component_kilonova_model')

# Let's change the redshift prior to ensure we get some bright kilonovae.
prior['redshift'] = bilby.core.prior.Uniform(0.002, 0.005, 'redshift', latex_label='$z$')
# We can now sample from the prior to get a set of parameters for a kilonova. Let's sample 5 events.
events = 5
parameters = prior.sample(events)
# For each kilonova we also need to provide a starting time. Let's keep this simple.
parameters['t0_mjd_transient'] = np.array([60260, 60280, 60290, 60360, 60320])

# We can also place the transient ourselves on the sky by setting the ra and dec parameters.
# For now we will let redback randomly place the transients from ra and dec that are covered by the Rubin survey pointings.
# Let's print all the values to be sure we are happy with them.
print(parameters)

# We now simulate a kilonova population using the SimulateOpticalTransient class.
# Specifically the class method for simulating a transient population in Rubin.

kn_sim = SimulateOpticalTransient.simulate_transient_population_in_rubin(model='one_component_kilonova_model',
                                                                        survey='Rubin_10yr_baseline',
                                                                        parameters=parameters, model_kwargs={},
                                                                        end_transient_time=10., snr_threshold=5.)
# We can print the observations that were simulated to see what the data looks like.
print(kn_sim.list_of_inference_observations)

# But why print when you can plot?
# First, let's save the observations to a file.
kn_sim.save_transient_population(transient_names=None)
# This will save the transient data to a csv file in the 'simulated' directory alongside the csv file of the injection parameters.
# By default the name of the transient will be event_{}.csv where {} is the event number. But you can also pass a list of names.

# Now we can create redback objects of each of these transients and then plot them.
# Note that this will only plot the 'detected' data points. The user can add the non-detections back in if they wish.

objects = []
for i in range(events):
    kne = redback.transient.Kilonova.from_simulated_optical_data(name='event_{}'.format(i), data_mode='magnitude')
    objects.append(kne)

# Now we can plot the light curves. We can use the redback plot_data method for this but put the transients on one figure.
fig, ax = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
for i, obj in enumerate(objects):
    obj.plot_data(axes=ax[i], data_mode='magnitude', show=False)
plt.ylim(24,18)
plt.show()
