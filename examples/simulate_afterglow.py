import redback
from redback.simulation import SimulateOpticalTransient


num_obs = {'lsstg': 5, 'lsstr':5, 'lssti':5}
average_cadence = {'lsstg': 3.0, 'lsstr': 2.0, 'lssti': 2.5}
cadence_scatter = {'lsstg': 0.5, 'lsstr':0.5, 'lssti':0.5}
limiting_magnitudes = {'lsstg': 25.0, 'lsstr': 24.5, 'lssti': 23.0}
pointings = redback.simulation.make_pointing_table_from_average_cadence(
    0.0, 0.0, num_obs, average_cadence, cadence_scatter, limiting_magnitudes, **kwargs)

model_kwargs = {'base_model':'gaussiancore', 'spread':False}
parameters = redback.prior.get_prior(model='gaussiancore').sample()

AG_instance = SimulateOpticalTransient.simulate_transient('afterglow_models_sed', parameters, pointings, survey=None, model_kwargs)
