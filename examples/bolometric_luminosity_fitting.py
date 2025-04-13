# In this example we show how to use redback to fit bolometric luminosities of different optical transients.
# Note, redback also provides functionality to build a bolometric lightcurve with/without SED corrections from the photometric data.
# Please check the other examples for more details.

# Lets import some stuff
import numpy as np
import matplotlib.pyplot as plt
import redback

# Let's make some fake data using these parameters
times = np.geomspace(0.5, 50, 20)
times = np.append(times, np.linspace(50, 400, 100))
kappa = 0.07
kappa_gamma = 0.1
mej = 10
vej = 5000
nickel_mass = 1.5
tf = 1600
f_nickel = nickel_mass/mej

parameters = {}
parameters['kappa'] = kappa
parameters['kappa_gamma'] = kappa_gamma
parameters['f_nickel'] = f_nickel
parameters['mej'] = mej
parameters['vej'] = vej
parameters['temperature_floor'] = tf

# For convenience we are going to create an alias to the bolometric luminosity function
model = 'arnett_bolometric' # we use this model here, but this could be replaced by any other redback function.
func = redback.model_library.all_models_dict['arnett_bolometric']

# Fake the data.
lbol = func(times, **parameters)
sigmas = 0.1 * lbol
lbol_obs = lbol + np.random.normal(0, sigmas, size=len(lbol))

# Create the supernova transient object
sn = redback.transient.Supernova(name='fake_sn', Lum50=lbol_obs/1e50, time_rest_frame=times, Lum50_err=sigmas/1e50,
                                 data_mode='luminosity')

# Most redback models can also output bolometric luminosities of optical transients, which is often returned in erg/s.
# However, the transient objects assume luminosity in units of 10^50 erg/s.
# A simple workaround for this is to write a small wrapper function that takes the luminosity in erg/s from the function and divides by 10^50.
# And you then use this wrapper function to fit instead.

def wrapper(tt, **kwargs):
    """
    Wrapper function to convert luminosity from erg/s to 10^50 erg/s.
    """
    # Call the original function
    lbol = func(tt, **kwargs)

    # Convert to 10^50 erg/s
    lbol = lbol / 1e50

    return lbol

# Now we can fit the data using the redback fitting function, we also need the priors
priors = redback.priors.get_priors(model)
result = redback.fit_model(transient=sn, prior=priors, model=wrapper,
                           sampler='nestle', nlive=100, plot=False,
                           injection_parameters=parameters)

# Let's plot the results and see how we did
result.plot_lightcurve(model=wrapper)

# Plot the corner plot
result.plot_corner(show=True)
plt.show()