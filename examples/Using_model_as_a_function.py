# In this example, we show how a model can be called a function.
# This is useful for plotting, or for use in setting up an analysis.
import redback
import numpy as np
import matplotlib.pyplot as plt
model = 'one_component_kilonova_model'

# use default priors
priors = redback.priors.get_priors(model=model)
priors['redshift'] = 1e-2

model_kwargs = dict(frequency=2e14, output_format='flux_density', bands='sdssi')
time = np.linspace(0.1, 30, 50)
for x in range(100):
    ss = priors.sample()
    ss.update(model_kwargs)
    out = redback.transient_models.kilonova_models.one_component_kilonova_model(time, **ss)
    plt.semilogy(time, out, alpha=0.1, color='red')
plt.xlabel('Time [days]')
plt.ylabel('Flux density [mJy]')
plt.ylim(1e-4, 1e0)
plt.tight_layout()
plt.show()