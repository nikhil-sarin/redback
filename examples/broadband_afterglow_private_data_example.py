# What if your data is not public, or if you simulated it yourself.
# You can fit it with redback but now by loading the transient object yourself.
# This example also shows how to fit afterglow models to broadband data.
# In particular, the afterglow of GRB170817A, the GRB that accompanied the first Binary neutron star merger

import redback
import pandas as pd

# load the data file
data = pd.read_csv('example_data/grb_afterglow.csv')
time_d = data['time'].values
flux_density = data['flux'].values
frequency = data['frequency'].values
flux_density_err = data['flux_err'].values

# we now load the afterglow transient object. We are using flux_density data here so we need to use that data mode
data_mode = 'flux_density'

# set some other useful things as variables
name = '170817A'
redshift = 1e-2

afterglow = redback.transient.Afterglow(
    name=name, data_mode=data_mode, time=time_d,
    flux_density=flux_density, flux_density_err=flux_density_err, frequency=frequency)

# Now we have loaded the data up, we can plot it.
afterglow.plot_data()
afterglow.plot_multiband()

# now let's actually fit it with data. We will use all the data and a gaussiancore structured jet from afterglowpy.
# Note this is not a fast example, so we will make some sampling sacrifices for speed.

model = 'gaussiancore'

# use default priors and 'nestle' sampler
sampler = 'dynesty'
priors = redback.priors.get_priors(model=model)
priors['redshift'] = redshift

# We are gonna fix some of the microphysical parameters for speed
priors['logn0'] = -2.6
priors['p'] = 2.16
priors['logepse'] = -1.25
priors['logepsb'] = -3.8
priors['ksin'] = 1.

model_kwargs = dict(frequency=frequency, output_format='flux_density')

# returns a supernova result object
result = redback.fit_model(transient=afterglow, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, sample='rslice', nlive=50, dlogz=10, resume=True)
# plot corner
result.plot_corner()

# plot multiband lightcurve. This will plot a panel for every unique frequency
result.plot_multiband_lightcurve(random_models=100)
