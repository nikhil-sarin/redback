import redback
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
# import bilby
# import latex
# import matplotlib
# matplotlib.use("macosx")


# We implemented many models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = 'evolving_magnetar'

GRB = '070809'
# Flux density, flux data
redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb=GRB, data_mode="flux")
# creates a GRBDir with GRB

# create Luminosity data
afterglow = redback.afterglow.SGRB.from_swift_grb(name=GRB, data_mode='flux',
                                                  truncate=True, truncate_method="prompt_time_error")

# uses an analytical k-correction expression to create luminosity data if not already there.
# Can also use a numerical k-correction through CIAO
afterglow.analytical_flux_to_luminosity()
afterglow.plot_data()

# use default priors
priors = redback.priors.get_priors(model=model)

# alternatively can create a dictionary of priors in some priors
# priors = bilby.core.prior.PriorDict()
# priors['a_1'] = bilby.core.prior.LogUniform(1e-15, 1e15, 'A_1', latex_label = r'$A_{1}$')
# priors['alpha_1'] = bilby.core.prior.Uniform(-7, -1, 'alpha_1', latex_label = r'$\alpha_{1}$')
# priors['p0'] = bilby.core.prior.Uniform(0.7e-3, 0.1, 'p0', latex_label = r'$P_{0} [s]$')
# priors['mu0'] = bilby.core.prior.Uniform(1e-3, 10, 'mu0', latex_label = r'$\mu_{0} [10^{33} G cm^{3}]$')
# priors['muinf'] = bilby.core.prior.Uniform(1e-3, 10, 'muinf', latex_label = r'$\mu_{\inf} [10^{33} G cm^3]$')
# priors['sinalpha0'] = bilby.core.prior.Uniform(1e-3, 0.99, 'sinalpha0', latex_label = r'$\sin\alpha_{0}$')
# priors['tm'] = bilby.core.prior.Uniform(1e-3, 100, 'tm', latex_label = r'$t_{m} [days]$')
# priors['II'] = bilby.core.prior.LogUniform(1e45, 1e46, 'II', latex_label = r'$I$')


# Call redback.fit_model to run the sampler and obtain GRB result object
result = redback.fit_model(model=model, sampler='dynesty', nlive=200, transient=afterglow,
                           prior=priors, sample='rslice', clean=True)

result.plot_lightcurve(random_models=100)
result.plot_residual()
# plt.show()