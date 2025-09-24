import bilby
import redback
import numpy as np
import matplotlib.pyplot as plt

# redshift = 0.1
# start_time = 55570
# time = np.linspace(0.1, 65, 100) + start_time
#
# kwargs = {'frequency': 4e14, 'output_format': 'magnitude', 'bands':'ztfr'}
# outs = redback.transient_models.supernova_models.salt2(time=time, redshift=redshift, x0=1e-7, x1=0.9, c=0.3, peak_time=55589, **kwargs)
#
# # Let's plot.
#
# plt.plot(time - start_time, outs, label='salt2')
# plt.gca().invert_yaxis()
# plt.show()

transient = 'ZTF20aamdsjv'
data = redback.get_data.get_lasair_data(transient=transient, transient_type='supernova')

# Set up the redback transient object.
sn = redback.transient.Supernova.from_lasair_data(transient, use_phase_model=True,
                                                  data_mode='magnitude')

priors = bilby.core.prior.PriorDict()
priors['redshift'] = 0.061

# Set a prior on t0 to be within 100 days before the first observation
priors['t0'] = bilby.core.prior.Uniform(sn.x[0] - 100, sn.x[0] - 0.01, 't0', latex_label=r'$t_0$')

# Set a prior on the peak time to be within 10 days of the maximum (minimum magnitude)
data_peak = sn.x[np.argmin(sn.y)]
priors['peak_time'] = bilby.core.prior.Uniform(data_peak - 10, data_peak + 10, 'peak_time', latex_label=r'$t_{\rm peak}$')

# Set a prior on the x0, x1, and c parameters i.e., the salt2 model parameters
priors['x0'] = bilby.core.prior.Uniform(1e-10, 1e-1, 'x0', latex_label=r'$x_0$')
priors['x1'] = bilby.core.prior.Normal(0, 1, 'x1', latex_label=r'$x_1$')
priors['c'] = bilby.core.prior.Normal(0, 0.1, 'c', latex_label=r'$c$')

kwargs = {'bands': sn.filtered_sncosmo_bands, 'output_format': 'magnitude'}

# Let's fit. Again the interface is similar to the normal interface for redback.
result = redback.fit_model(transient=sn, model='salt2', prior=priors, model_kwargs=kwargs,
                           sampler='dynesty', nlive=100, plot=False, clean=True)
ax = result.plot_lightcurve(random_models=50, show=False)
ax.set_xscale('linear')
ax.set_yscale('linear')
plt.show()
