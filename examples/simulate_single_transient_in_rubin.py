import numpy as np
import redback
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateOpticalTransient
from redback.model_library import all_models_dict
import pandas as pd

data = pd.read_csv('../redback/tables/rubin_baseline_nexp1_v1_7_10yrs.tar.gz', compression='gzip')
pointings = data

model_kwargs = {'base_model':'gaussiancore', 'spread':False}
parameters = redback.priors.get_priors(model='one_component_kilonova_model').sample()
parameters['ra'] = 1.0
parameters['dec'] = -0.5
parameters['mej'] = 0.04
parameters['t0_mjd_transient'] = 60582.0
parameters['redshift'] = 0.01
parameters['t0'] = parameters['t0_mjd_transient']
parameters['temperature_floor'] = 3000
parameters['kappa'] = 1
parameters['vej'] = 0.2
sim = SimulateOpticalTransient(model='one_component_kilonova_model', survey='Rubin_10yr_baseline',
                               parameters=parameters, model_kwargs=model_kwargs, end_transient_time=100.)
AG_instance = sim
print(sim.inference_observations)
afterglow = redback.transient.Afterglow(
    name='hi', data_mode='magnitude', time=AG_instance.inference_observations['time (days)'].values,
    magnitude=AG_instance.inference_observations['magnitude'].values,
    magnitude_err=AG_instance.inference_observations['e_magnitude'].values, bands=AG_instance.inference_observations['band'].values)
ax = afterglow.plot_data()
plt.show()
