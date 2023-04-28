import numpy as np
import redback
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateOpticalTransient
from redback.model_library import all_models_dict


num_obs = {'lsstg': 10, 'lsstr':10, 'lssti':10}
average_cadence = {'lsstg': 1.5, 'lsstr': 5.0, 'lssti': 2.5}
cadence_scatter = {'lsstg': 0.5, 'lsstr':0.5, 'lssti':0.5}
limiting_magnitudes = {'lsstg': 25.0, 'lsstr': 24.5, 'lssti': 23.0}
pointings = redback.simulate_transients.make_pointing_table_from_average_cadence(
    ra=1.0, dec=1.5, num_obs=num_obs, average_cadence=average_cadence,
    cadence_scatter=cadence_scatter, limiting_magnitudes=limiting_magnitudes, initMJD=59581.0)
print(pointings)
model_kwargs = {'base_model':'gaussiancore', 'spread':False}
parameters = redback.priors.get_priors(model='one_component_kilonova_model').sample()
parameters['ra'] = 1.0
parameters['dec'] = 1.5
parameters['mej'] = 0.01
parameters['t0_mjd_transient'] = 59582.0
parameters['redshift'] = 0.01
parameters['t0'] = parameters['t0_mjd_transient']
parameters['temperature_floor'] = 3000
parameters['kappa'] = 1
parameters['vej'] = 0.2
print(parameters)

model = 't0_base_model'
func = all_models_dict[model]
base_model = 'one_component_kilonova_model'
model_kwargs = {'base_model':'gaussiancore', 'spread':False}
_model_kwargs = dict(bands=pointings['filter'].values, base_model=base_model, output_format='magnitude')
_model_kwargs.update(parameters)
mag = func(time=pointings['expMJD'].values, **_model_kwargs)
print(mag)

AG_instance = SimulateOpticalTransient(model='one_component_kilonova_model',
                                       parameters=parameters, pointings_database=pointings,
                                       survey=None, model_kwargs=model_kwargs, end_transient_time=10.)
print(AG_instance.observations)
afterglow = redback.transient.Afterglow(
    name='230421', data_mode='magnitude', time=AG_instance.inference_observations['time (days)'].values,
    magnitude=AG_instance.inference_observations['magnitude'].values,
    magnitude_err=AG_instance.inference_observations['e_magnitude'].values, bands=AG_instance.inference_observations['band'].values)
ax = afterglow.plot_data()
plt.show()
