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

redback.getdata.get_data(transient_type="prompt", data_source="batse", event_label=name)
prompt = redback.transient.prompt.PromptTimeSeries.from_batse_grb_name(name=name)

# use default priors
priors = redback.priors.get_priors(model=model, data_mode='counts', times=prompt.time,
                                   y=prompt.counts, yerr=prompt.counts_err, dt=prompt.bin_size)

result = redback.fit_model(source_type='prompt', name=name, model=model, transient=prompt, nlive=500,
                           sampler=sampler, prior=priors, data_mode='counts', outdir="GRB_results", sample='rslice')
# returns a GRB prompt result object
result.plot_lightcurve(random_models=1000)
result.plot_corner()
