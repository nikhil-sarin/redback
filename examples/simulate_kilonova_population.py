import numpy as np
import redback
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateOpticalTransient
from redback.model_library import all_models_dict

import pandas as pd

data = pd.read_csv('../redback/tables/rubin_baseline_nexp1_v1_7_10yrs.tar.gz', compression='gzip')
pointings = data

simsurvey = redback.simulate_transients.SimulateFullOpticalSurvey()

# print(pointings)
# model_kwargs = {'base_model':'gaussiancore', 'spread':False}
# parameters = redback.priors.get_priors(model='one_component_kilonova_model').sample()
# parameters['ra'] = 1.0
# parameters['dec'] = 1.5
# parameters['mej'] = 0.01
# parameters['t0_mjd_transient'] = 59582.0
# parameters['redshift'] = 0.02
# parameters['t0'] = parameters['t0_mjd_transient']
# parameters['temperature_floor'] = 3000
# parameters['kappa'] = 1
# parameters['vej'] = 0.2
# print(parameters)


