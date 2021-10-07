import numpy as np

import bilby
import redback

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use("Qt5Agg")

sampler = 'dynesty'
model = 'gaussian'
name = '910505'

redback.getdata.get_prompt_data_from_batse(name, use_default_directory=False)
prompt = redback.transient.prompt.PromptTimeSeries.from_batse_grb_name(name=name)

plt.step(prompt.time, prompt.counts/prompt.bin_size)
plt.show()
plt.clf()

# use default priors
priors = redback.priors.get_priors(model=model, data_mode='counts')
max_counts = np.max(prompt.counts)
dt = prompt.time[1] - prompt.time[0]
duration = prompt.time[-1] - prompt.time[0]

result = redback.fit_model(source_type='prompt', name=name, model=model, transient=prompt, nlive=500,
                           sampler=sampler, prior=priors, data_mode='counts', outdir="GRB_results", clean=True)
# returns a GRB prompt result object
result.plot_lightcurve(random_models=1000)
result.plot_corner()
