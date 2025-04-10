# This example shows how to deal with outliers in the data by using a mixture model likelhood.

# First let's fake some data.
# We are gonna keep things a bit simple here.

import numpy as np
import matplotlib.pyplot as plt
import redback
import bilby

np.random.seed(12345)

# Set some parameters
tts = np.linspace(0.1, 100, 100)
kappa = 0.07
kappa_gamma = 0.1
mej = 10
vej = 5000
nickel_mass = 1.5
tf = 1600
f_nickel = nickel_mass/mej
tts = np.linspace(0.1, 400, 40)

parameters = {}
parameters['output_format'] = 'magnitude'
parameters['bands'] = np.repeat('ztfg', len(tts))
parameters['kappa'] = kappa
parameters['kappa_gamma'] = kappa_gamma
parameters['f_nickel'] = f_nickel
parameters['mej'] = mej
parameters['vej'] = vej
parameters['temperature_floor'] = tf
parameters['redshift'] = 0.1

# Create an alias to the model func for convenience
func = redback.model_library.all_models_dict['arnett']

# Fake some data
mags = func(tts, **parameters)
sigmas = 0.03 * mags
mag_obs = mags + np.random.normal(0, sigmas, len(tts))

# Now for some of the data we are going to scatter them a lot more, to mimic outliers.
index = [3, 10, 19, 23]
# 4 out of 40 i.e., 90% of the data is inlier - this will be a parameter we can either fix or marginalise over
sigma_out = 3 # mag of additional scatter for these outlier points
mag_obs[index] += np.random.normal(0, sigma_out, len(index))

# Let's add these to our parameters dict so the corner plot can put orange markers at the true values
parameters['sigma_out'] = sigma_out
parameters['alpha'] = 0.9

# lets load up our transient object.

sn = redback.transient.Supernova(name='mixture',
                                 magnitude=mag_obs,
                                 magnitude_err=sigmas,
                                data_mode='magnitude',
                                time=tts, bands=parameters['bands'])
sn.plot_data()

# Let's set up the likelihood

model_kwargs = {'bands':sn.sncosmo_bands, 'output_format':'magnitude'}
likelihood = redback.likelihoods.MixtureGaussianLikelihood(x=sn.x, y=sn.y,
                                                           sigma=sn.y_err,
                                                          function=func,
                                                           kwargs=model_kwargs)


priors = redback.priors.get_priors('arnett')
priors['alpha'] = bilby.core.prior.Uniform(0.85, 0.95, 'alpha')
priors['sigma_out'] = bilby.core.prior.Uniform(1.5, 4.5, 'sigma_out')

result = redback.fit_model(transient=sn, prior=priors, likelihood=likelihood,
                           sampler='nestle', nlive=200, plot=False, clean=False,
                           injection_parameters=parameters, model='arnett',
                           model_kwargs=model_kwargs)
result.plot_corner()

ax = result.plot_lightcurve(show=False)
ax.set_yscale('linear')
plt.show()

# The fits not too bad because the outliers can be dealt with without influencing the fit.
# This is especially useful when the outliers themselves may have smaller intrinsic errors (e.g., if they are brighter)
# This would typically be a problem for a normal Gaussian likelihood, but the mixture model can handle this.

# Let's also go above and beyond, and plot the posterior distributions each data point,
# and quantify whether it is an outlier or not.
# Note our actual index of outliers.

random_samples = 500 # number of random samples to draw from the posterior.
# In principle this should be the full posterior, but this is reasonable approximation
post = np.zeros((random_samples, len(sn.x)))
for x in range(random_samples):
    ss = result.posterior.iloc[np.random.randint(len(result.posterior))]
    ss['output_format'] = 'magnitude'
    function = func
    ss['bands'] = sn.filtered_sncosmo_bands
    model_prediction = function(sn.x, **ss)
    post[x] = likelihood.calculate_outlier_posteriors(model_prediction)

med = np.median(post,axis=0)
std = np.std(post, axis=0)
iis = np.arange(0, len(sn.x), 1)
plt.errorbar(iis, med, std, fmt='o', label='Outlier probability')
plt.scatter(np.array([index]), np.ones(len(index)),
            label='True outliers', color='red', marker='x', s=50)
plt.ylabel('Outlier probability')
plt.xlabel('Data point ID')
plt.legend()
plt.show()

# Voila! We can see that the outliers are indeed outliers, and the model is able to fit the data well.
# Note that with real data, the outliers may not be so obvious, but the model can still fit the data well
# because it can still account for the probability that the particular data point may be an outlier.