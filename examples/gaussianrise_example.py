import redback
import bilby
tde = "PS18kh"
data = redback.get_data.get_tidal_disruption_event_data_from_open_transient_catalog_data(tde)
tde = redback.tde.TDE.from_open_access_catalogue(tde, data_mode='flux_density', use_phase_model=False)
tde.plot_data()

from bilby.core.prior import LogUniform, Constraint, Uniform

def constraints(parameters):
    converted_parameters = parameters.copy()
    ms = parameters['stellar_mass']
    mbh6 = parameters['mbh_6']
    etamin = 0.01*(ms**(-7./15.))*(mbh6**(2./3.))
    betamax = 12.*(ms**(7./15.))*(mbh6**(-2./3.))
    converted_parameters['eta_low'] = converted_parameters['eta'] - etamin
    converted_parameters['beta_high'] = betamax - converted_parameters['beta']
    return converted_parameters

prior = bilby.core.prior.PriorDict(conversion_function=constraints)
# prior['peak_time_mjd'] = Uniform(58157+20,58157+50, name='peak_time', latex_label = r'$t_{\mathrm{peak}}$~[days]')
prior['peak_time'] = Uniform(10,50, name='peak_time', latex_label = r'$t_{\mathrm{peak}}$~[days]')
prior['sigma_t'] = Uniform(10,50, name='sigma_t', latex_label = r'$\sigma$~[days]')
prior['mbh_6'] = Uniform(1e-3, 10, name='mbh_6', latex_label = r'$M_{\mathrm{BH}}~[10^{6}~M_{\odot}]$')
prior['stellar_mass'] = Uniform(0.1, 5, name='stellar_mass', latex_label = r'$M_{\mathrm{star}} [M_{\odot}]$')
prior['eta'] = LogUniform(1e-3, 0.1, name='eta', latex_label=r'$\eta$')
prior['alpha'] = LogUniform(1e-2, 1e-1, name='alpha', latex_label=r'$\alpha$')
prior['beta'] = Uniform(1, 30, name='beta', latex_label=r'$\beta$')
prior['eta_low'] = Constraint(0,0.1)
prior['beta_high'] = Constraint(0,1)
# prior['t0'] = bilby.core.prior.Uniform(tde.time_mjd[0]-100, tde.time_mjd[0]-0.1, latex_label='t0')
# prior['av'] = Uniform(0.1, 3, name='av', latex_label=r'$a_{v}$')
prior['redshift'] = 0.075
# model = 't0_tde_extinction'
base_model = 'gaussianrise_metzger_tde'
model_kwargs = dict(frequency=tde.filtered_frequencies, output_format='flux_density')
sampler = 'nestle'

result = redback.fit_model(transient=tde, model=base_model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=prior, sample='rslice', nlive=1000, clean=True, plot=False)
result.plot_corner()