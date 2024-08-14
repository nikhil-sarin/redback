import redback

sampler = 'dynesty'
model = 'gaussian_prompt'
name = '910505'

data = redback.get_data.get_prompt_data_from_batse(grb=name)
prompt = redback.transient.prompt.PromptTimeSeries.from_batse_grb_name(name=name)

# use default priors
priors = redback.priors.get_priors(model=model, times=prompt.time,
                                   y=prompt.counts, yerr=prompt.counts_err, dt=prompt.bin_size)

result = redback.fit_model(source_type='prompt', model=model, transient=prompt, nlive=500,
                           sampler=sampler, prior=priors, outdir="GRB_results", sample='rslice')
# returns a GRB prompt result object
result.plot_lightcurve(random_models=1000)
result.plot_corner()
