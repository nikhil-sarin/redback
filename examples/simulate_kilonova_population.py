# This example will show how to simulate a collection of kilonovae with Rubin
import redback
from redback.simulate_transients import SimulateOpticalTransient
import matplotlib.pyplot as plt
import bilby
import numpy as np

# We now set up the parameters for the kilonova population. An easy way to do this is to use the redback prior.
# We will use the one_component_kilonova_model implemented in redback but any optical model will work.
# A user could also pass their own model

prior = redback.priors.get_priors(model='one_component_kilonova_model')

# Let's change the redshift prior to ensure we get some bright kilonovae.
prior['redshift'] = bilby.core.prior.Uniform(0.005, 0.01, 'redshift', latex_label='$z$')
# We can now sample from the prior to get a set of parameters for a kilonova. Let's sample 5 events.
events = 5
parameters = prior.sample(events)
# For each kilonova we also need to provide a starting time. Let's keep this simple.
parameters['t0_mjd_transient'] = np.array([60250, 60260, 60270, 60280, 60290])

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

# But why print when you can plot? Firstly though, let's save the observations to a file.
# This will save the transient data to a csv file in the 'simulated' directory alongside the csv file of the injection parameters.
# By default the name of the transient will be event_{}.csv where {} is the event number. But you can also pass a list of names.
kn_sim.save_transient_population(transient_names=None)


