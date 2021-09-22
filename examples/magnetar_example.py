import redback
import bilby
import matplotlib.pyplot as plt

sampler = 'dynesty'
# lots of different models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'evolving_magnetar'

GRB = '070809'
path = 'GRBData'
# Flux density, flux data
redback.getdata.get_afterglow_data_from_swift(GRB, data_mode='flux')
# creates a GRBDir with GRB

# create Luminosity data
afterglow = redback.afterglow.SGRB(GRB, data_mode='flux')
afterglow.load_and_truncate_data()
afterglow.analytical_flux_to_luminosity()

# assert False

# uses an analytical k-correction expression to create luminosity data if not already there.
# Can also use a numerical k-correction through CIAO

# use default priors
priors = redback.redback.priors.get_priors(model=model, data_mode='luminosity')

# alternatively can pass in some priors
# priors = {}
# priors['A_1'] = bilby.core.prior.LogUniform(1e-15, 1e15, 'A_1', latex_label = r'$A_{1}$')
# priors['alpha_1'] = bilby.core.prior.Uniform(-7, -1, 'alpha_1', latex_label = r'$\alpha_{1}$')
# priors['p0'] = bilby.core.prior.Uniform(0.7e-3, 0.1, 'p0', latex_label = r'$P_{0} [s]$')
# priors['mu0'] = bilby.core.prior.Uniform(1e-3, 10, 'mu0', latex_label = r'$\mu_{0} [10^{33} G cm^{3}]$')
# priors['muinf'] = bilby.core.prior.Uniform(1e-3, 10, 'muinf', latex_label = r'$\mu_{\inf} [10^{33} G cm^3]$')
# priors['sinalpha0'] = bilby.core.prior.Uniform(1e-3, 0.99, 'sinalpha0', latex_label = r'$\sin\alpha_{0}$')
# priors['tm'] = bilby.core.prior.Uniform(1e-3, 100, 'tm', latex_label = r'$t_{m} [days]$')
# priors['II'] = bilby.core.prior.LogUniform(1e45, 1e46, 'II', latex_label = r'$I$')


result, data = redback.fit_model(name=GRB, model=model, sampler=sampler, nlive=500,
                                 path=path, prior=priors, data_mode='luminosity')


# returns a GRB result object
result.plot_lightcurve(model=model, random_models=1000)
