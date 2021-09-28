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

redback.getdata.get_data(transient_type="prompt", instrument="batse", event_label=name, use_default_directory=True)
prompt = redback.transient.prompt.PromptTimeSeries.from_batse_grb_name(name=name, trigger_number="148")

plt.step(prompt.time, prompt.counts/prompt.bin_size)
plt.show()
plt.clf()

# use default priors
# priors = redback.priors(model = model, data_mode = 'counts')
max_counts = np.max(prompt.counts)
dt = prompt.time[1] - prompt.time[0]
duration = prompt.time[-1] - prompt.time[0]
priors = bilby.core.prior.PriorDict()
priors["background_rate"] = bilby.prior.Uniform(0, 1e6, name="background_rate")
priors["log_amplitude"] = bilby.prior.Uniform(np.log(1), 30, name="log_amplitude")
priors["t_0"] = bilby.prior.Uniform(prompt.time[0], prompt.time[-1], name="t_0")
# priors["t_0"] = bilby.prior.Uniform(10, 20, name="t_0")
priors["log_sigma"] = bilby.prior.Uniform(np.log(dt), np.log(100*duration), name="log_sigma")

result = redback.fit_model(source_type='prompt', name=name, model=model, transient=prompt, nlive=500,
                           sampler=sampler, prior=priors, data_mode='counts', outdir="GRB_results", clean=True)
# returns a GRB prompt result object
# result.plot_lightcurve(random_models=1000)
print(result.outdir)
print(result.label)
result.plot_corner()
result.outdir = '.'
result.plot_corner()
