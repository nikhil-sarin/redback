import numpy as np
import redback
from redback.simulate_transients import SimulateOpticalTransient


num_obs = {'lsstg': 5, 'lsstr':5, 'lssti':5}
average_cadence = {'lsstg': 3.0, 'lsstr': 2.0, 'lssti': 2.5}
cadence_scatter = {'lsstg': 0.5, 'lsstr':0.5, 'lssti':0.5}
limiting_magnitudes = {'lsstg': 25.0, 'lsstr': 24.5, 'lssti': 23.0}
pointings = redback.simulate_transients.make_pointing_table_from_average_cadence(
    0.0, 0.0, num_obs, average_cadence, cadence_scatter, limiting_magnitudes)
print(pointings)

model_kwargs = {'base_model':'gaussiancore', 'spread':False}
parameters = redback.priors.get_priors(model='one_component_kilonova_model').sample()
parameters['ra'] = 0.0
parameters['dec'] = 0.0
parameters['t0_mjd_transient'] = 59582.0


AG_instance = SimulateOpticalTransient(model='one_component_kilonova_model', parameters=parameters, pointings_database=pointings, survey=None, model_kwargs=model_kwargs)



afterglow = redback.transient.Afterglow(
    name='example', data_mode='magnitude', time=AG_instance.observations['time (days)'],
    magnitude=AG_instance.observations['magnitude'], magnitude_err=AG_instance.observations['e_magnitude'], bands=AG_instance.observations['band'])

afterglow.plot_data()
