import bilby.core.prior
from bilby.core.prior import PriorDict, Uniform, Constraint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback
from redback.constraints import csm_constraints

model = 'csm_interaction'
model_priors = redback.priors.get_priors(model=model)
# model = 'csm_interaction_bolometric'
function = redback.model_library.all_models_dict[model]

priors = bilby.core.prior.PriorDict(conversion_function=csm_constraints)
priors.update(model_priors)
# priors['photosphere_constraint_1'] = Constraint(0, 1)
# priors['photosphere_constraint_2'] = Constraint(0, 1)
priors['csm_mass'] = 58.0
priors['mej'] = 46
priors['vej'] = 5500
priors['r0'] = 617
priors['nn'] = 8.8
priors['delta'] = 0.
priors['rho'] = 1e-14
priors['eta'] = 2
priors['redshift'] = 0.16
samples = pd.DataFrame(priors.sample(10))
time = np.geomspace(0.01, 500, 500)
redshift = 0.01

for x in range(10):
    kwargs = samples.iloc[x]
    kwargs['output_format'] = 'magnitude'
    kwargs['bands'] = ['lsstg']
    # kwargs['interaction_process'] = None
    mag = function(time, **kwargs)
    # plt.loglog(time, mag)
    plt.plot(time, mag)
plt.gca().invert_yaxis()
plt.show()