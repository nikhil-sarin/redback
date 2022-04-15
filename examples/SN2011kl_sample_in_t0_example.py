import redback
from bilby.core.prior import Uniform, Gaussian

sampler = 'dynesty'
sne = "SN2011kl"

# we want to sample in t0 and with extinction so we use
model = 't0_supernova_extinction'
# we want to specify a base model for the actual physics
base_model = "arnett"

data = redback.get_data.get_supernova_data_from_open_transient_catalog_data(sne)

# load the data into a supernova transient object which does all the processing and can be used to make plots etc
# we set the data_mode to flux density to use/fit flux density. We could use 'magnitude' or 'luminosity' or flux as well.
# However, for optical transients we recommend fitting in flux_density.
supernova = redback.supernova.Supernova.from_open_access_catalogue(name=sne, data_mode='flux_density', use_phase_model=True)

# lets make a plot of the data
supernova.plot_multiband(filters=["J", "H", "g", "i"])

# we are now only going to fit g and i bands. We set this using the transient object and the active bands attribute
bands = ["g", "H"]
supernova.active_bands = bands

# use default priors
priors = redback.priors.get_priors(model=model)
priors.update(redback.priors.get_priors(model=base_model))
# we know the redshift for SN2011kl so we just fix the prior for the redshift to the known value
priors['redshift'] = 0.677

# we also want to sample in t0 so we need a prior for that and extinction parameters
# we use bilby prior objects for this
# let's set t0 as a Gaussian around the first observation with sigma = 0.5 - There are many other priors to choose from.
# This also demonstates how one can change the latex labels etc
priors['t0'] = Gaussian(data['time'][0], sigma=0.5, name='t0', latex_label=r'$T_{\rm{0}}$')

# We also need a prior on A_v i.e., the total mag extinction.
# Just use uniform prior for now.
priors['av'] = Uniform(0.1, 1, name='av', latex_label=r'$a_{v}$')

model_kwargs = dict(frequency=supernova.filtered_frequencies, output_format='flux_density', base_model=base_model)

# returns a supernova result object
result = redback.fit_model(transient=supernova, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, sample='rslice', nlive=500, resume=True)

result.plot_corner()

result.plot_lightcurve(random_models=100)
