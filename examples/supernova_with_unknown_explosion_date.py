from   astropy.io import ascii
import bilby
import pandas as pd
from   matplotlib import rcParams
import redback
from   redback.model_library import all_models_dict
from   redback.likelihoods import GaussianLikelihoodQuadratureNoise
from   redback import filters

# Switching off latex rendering for faster and prettier plots

rcParams['text.usetex']         = False
rcParams['font.family']         = 'DejaVu Sans'
rcParams['mathtext.fontset']    = 'dejavusans'

"""
We will fit the multi-band photometry of the SLSN 2018ibb with the Arnett
model, assuming the entire light curve is powered by radioactive Ni-56 and
Co-56.

We will also:
-Add two filters to Redback
-Use a local data file
-Use a different likelihood function
-Modify priors
-Modify settings of the nested sampler
"""

# 1) Adding filters to SN Cosmo

# Filter response functions can be downloaded from 
# http://svo2.cab.inta-csic.es/theory/fps/

import astropy.units as u
import sncosmo

from   astroquery.svo_fps import SvoFps

filter_list  = SvoFps.get_filter_list(facility='La Silla', instrument='GROND')
filter_label = ['grond::' + x.split('/')[1].split('.')[1] for x in filter_list['filterID']]
plot_label   = ['GROND/' + x.split('/')[1].split('.')[1] for x in filter_list['filterID']]

[filters.add_filter_svo(filter_list[ii], filter_label[ii], plot_label[ii]) for ii in range(len(filter_list))]

# 2) Read in the data and put the data into a Redback SN object

# Load data

data = pd.read_csv('example_data/SN2018ibb_photcat_REDBACK.ascii', sep=' ')

# Creating a Redback supernova object

sn=redback.transient.Supernova(name            = 'SN2018ibb',
                               data_mode       = 'magnitude',
                               time_mjd        = data['MJD'].values,
                               magnitude       = data['MAG'].values,
                               magnitude_err   = data['MAG_ERR'].values,
                               bands           = data['band'].values,
                               use_phase_model = True
                              )

redshift = 0.166

# Plot data

fig = sn.plot_data(show=False)
fig.set_xscale('linear')
fig.legend(loc='center right', ncol=2, bbox_to_anchor=(1.3, 0.25, 0.75, 0.5))
fig.set_xlim(-200, 800)
fig.set_ylim(25, 17)

# 3) Fit data

# We use the phase model family since we do not know the time of explosion.

model = 't0_supernova_extinction'

# Physical model

base_model = 'arnett'

# Putting the model family and the physical together

model_kwargs = dict(
                    bands         = sn.filtered_sncosmo_bands, 
                    base_model    = base_model,
                    output_format = 'magnitude'
                    )

function = all_models_dict[model]

# Set priors and change some default values

priors = redback.priors.get_priors(model=model)
priors.update(redback.priors.get_priors(model=base_model))

# Redshift
priors['redshift'] = redshift

# Explosion date
priors['t0'] = bilby.core.prior.Uniform(   
                                        minimum     = data['MJD'].values.min() - 200,
                                        maximum     = data['MJD'].values.min() - 1, 
                                        name        = 't0',
                                        latex_label = r'$t_{\rm expl.}~\rm (day)$',
                                        unit        = None,
                                        boundary    = None
                                        )

# Opacities
# We assume powering by radioactive nickel and cobalt. Hence, we freeze
# the opacities to theoretically motivated values.

priors['kappa'] = 0.07
priors['kappa_gamma'] = 0.03


# Ejecta mass
priors['mej'] = bilby.core.prior.Uniform(   
                                        minimum     = 1,
                                        maximum     = 260,
                                        name        = 'mej',
                                        latex_label = r'$M_{\rm{ej}}~(M_{\odot})$',
                                        unit        = None,
                                        boundary    = None
                                        )

# Nickel fraction
priors['f_nickel'] = bilby.core.prior.Uniform(   
                                        minimum     = 0,
                                        maximum     = 1,
                                        name        = 'fni',
                                        latex_label = r'$f_{\rm Ni}$',
                                        unit        = None,
                                        boundary    = None
                                        )

# Extinction
priors['av'] = bilby.core.prior.Uniform(
                                        minimum     = 0,
                                        maximum     = 1,
                                        name        = 'av',
                                        latex_label = r'$A_V~\rm (mag)$',
                                        unit        = None,
                                        boundary    = None
                                        )

# Ejecta velocity
priors['vej'] = bilby.core.prior.Uniform(   
                                        minimum     = 1000.0,          
                                        maximum     = 10000.0,      
                                        name        = 'vej',               
                                        latex_label = r'$v_{\rm ej}~\rm (km/s)$',    
                                        unit        = None,
                                        boundary    = None
                                        )

# Temperature floor
priors['temperature_floor'] = bilby.core.prior.Uniform(
                                        minimum     = 5000,            
                                        maximum     = 12000,        
                                        name        = 'temperature_floor', 
                                        latex_label = r'$T~(\rm K)$',                     
                                        unit        = None,
                                        boundary    = None
                                        )

# White noise parameter
priors['sigma'] = bilby.core.prior.Uniform(
                                        minimum     = 0.001,
                                        maximum     = 2,
                                        name        = 'sigma',
                                        latex_label = r'σ',
                                        unit        = None,
                                        boundary    = None
                                        )

# The measurement uncertainties might underestimate the true error.
# We might have a systematic error in the photometry. The model
# also has a systematic uncertainty.

# To add such these additional source of error to our model, we
# use the following likelihood function.

likelihood_func = GaussianLikelihoodQuadratureNoise(   
                                        x        = sn.x,
                                        y        = sn.y,
                                        function = function,
                                        kwargs   = model_kwargs,
                                        sigma_i  = sn.y_err
                                        )

# We are using the nested sampler nestle. We use a larger number of life
# points for a better sampling of the posterior

sampler    = "nestle"
nlive      = 500

# Fit the data

result = redback.fit_model(transient    = sn,
                           model        = model,
                           sampler      = sampler,
                           model_kwargs = model_kwargs,
                           prior        = priors,
                           sample       = 'rslice',
                           nlive        = nlive,
                           resume       = True,
                           likelihood   = likelihood_func,
                           clean        = True,
                           plot         = False
                          )

# Create a corner plot of the posteriors

result.plot_corner(show_titles=True, title_quantiles = [0.16, 0.5, 0.84])

# Display the fit to each band

fig=result.plot_multiband(show=False, random_models=100)

fig[0].set_xscale('linear')
fig[0].set_xlim(-200, 800)
