# In this example, we show how to use the csm_constraints function to apply constraints to the model.
# This works the same way for all other inbuilt constraints

import bilby.core.prior
from bilby.core.prior import Constraint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback

# Import the constraints function
from redback.constraints import csm_constraints


model = 'csm_interaction'
model_priors = redback.priors.get_priors(model=model)
function = redback.model_library.all_models_dict[model]

# Create a PriorDict object with the constraints function provided as a conversion_function
priors = bilby.core.prior.PriorDict(conversion_function=csm_constraints)
# We update this prior instance with the model priors loaded above.
priors.update(model_priors)

# We now apply our constraints. Please refer to the csm constraints function for more information on the constraints
priors['shock_time'] = Constraint(0.6, 0.8)
priors['photosphere_constraint_1'] = Constraint(0, 1)
priors['photosphere_constraint_2'] = Constraint(0, 0.5)

# Please keep in mind that if you sample with fixed parameters that are required in the constraints function,
# you will get an error. You will need to sample with these parameters as well, or modify the constraints function yourself.

priors['kappa'] = 0.34
priors['redshift'] = 0.16
samples = pd.DataFrame(priors.sample(20))
time = np.linspace(200, 900, 500)

redshift = 0.01
for x in range(len(samples)):
    kwargs = samples.iloc[x]
    kwargs['output_format'] = 'magnitude'
    kwargs['bands'] = ['lsstg']
    mag = function(time, **kwargs)
    plt.plot(time, mag)
plt.ylim(21, 30)
plt.gca().invert_yaxis()
plt.show()