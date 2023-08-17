# In this example, we show how redback and bilby can be used together to jointly analyse
# a GRB afterglow + GW signal from a binary neutron star merger.
# We will fix some parameters on the GRB afterglow model, with a simple tophat model
# and then use relative binning in bilby to speed up the analysis for this example,
# but this workflow can be extended to full parameter estimation. Or to include kilonova observations as well.

import numpy as np
import bilby
import redback
from astropy.cosmology import Planck18 as cosmo
from redback.transient_models.afterglow_models import tophat
from bilby.core.prior import Uniform
import matplotlib.pyplot as plt

# First, we define the parameters of the GRB and GW signal.
source_redshift = 0.03
source_distance = cosmo.luminosity_distance(source_redshift).value

gw_injection_parameters = dict(chirp_mass=bilby.gw.conversion.component_masses_to_chirp_mass(1.5, 1.3),
                               mass_ratio=1.3/1.5, chi_1=0.02, chi_2=0.02, luminosity_distance=source_distance,
                               theta_jn=0.43, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                               ra=1.375, dec=-1.2108, lambda_1=400, lambda_2=450, fiducial=1, mass_1=1.5, mass_2=1.3)
# We are now going to assume that theta_jn = the observers viewing angle
# i.e., the GRB jet goes along the orbital angular momentum of the binary.
# This is a reasonable and common assumption, for demonstrative purposes,
# we will also assume that the GRB jet energy is proportional the mass of the binary.

# We write a simple function to estimate this energy from the masses.
def get_jet_energy(mass_1, mass_2, fudge):
    total_mass = (mass_1 + mass_2)
    return total_mass * fudge * 2e33 * 3e10**2

fudge_factor = 0.04
grb_injection_parameters = dict(fudge=fudge_factor,
                                theta_jn=gw_injection_parameters['theta_jn'],
                                redshift=source_redshift,
                                thc=0.1, logn0=0.2, p=2.2, logepse=-1, logepsb=-2, ksin=1,
                                g0=1000, chirp_mass=gw_injection_parameters['chirp_mass'],
                                mass_ratio=gw_injection_parameters['mass_ratio'])

# Now as this model is slightly different to the redback tophat model (in assumptions about the jet energy and opening angle).
# We need to write some model function to encapsulate this.
# We will use the redback tophat model but just wrap it in another function to capture the model we have written down above.

def grb_afterglow_model(time, redshift, theta_jn, chirp_mass, mass_ratio, fudge, thc, logn0, p,
                        logepse, logepsb, ksin, g0, **kwargs):
    m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    energy = get_jet_energy(m1, m2, fudge=fudge)
    energy = np.log10(energy)
    return tophat(time=time, redshift=redshift, thv=theta_jn, loge0=energy, thc=thc,
                  logn0=logn0, p=p, logepse=logepse, logepsb=logepsb,
                  ksin=ksin, g0=g0, **kwargs)


# Let's now simulate the GRB afterglow following this model. We could use the redback simulation module
# but for simplicitly we will just call the function above directly and add some noise.
# We will also only simulate at one wavelength (X-ray), evaluating the flux density at 2e17 Hz.

# We will simulate for 150 days, with 50 points.
time = np.linspace(5, 150, 50)
afterglow_kwargs = {}
afterglow_kwargs['output_format'] = 'flux_density'
afterglow_kwargs['frequency'] = np.ones(len(time))*2e17


np.random.seed(123)
true_flux = grb_afterglow_model(time, **grb_injection_parameters, **afterglow_kwargs)
yerr = 0.2*true_flux
observed_flux = np.random.normal(true_flux, yerr)

# Let's load this data up into a redback transient object and see what it looks like.
sim_afterglow = redback.transient.Afterglow(name='simulated', time=time, flux_density=observed_flux,
                                            flux_density_err=yerr, data_mode='flux_density',
                                            frequency=afterglow_kwargs['frequency'])
sim_afterglow.plot_data()

# We can now set up the GRB likelihood. This is the same as the standard workflow for redback,
# but we need to explicitly set up the prior and likelihood.
em_likelihood = redback.likelihoods.GaussianLikelihood(x=sim_afterglow.time,
                                                         y=sim_afterglow.flux_density,
                                                         function=grb_afterglow_model,
                                                         sigma=yerr, kwargs=afterglow_kwargs)
# Set up the prior, and fix some parameters.
em_priors = bilby.core.prior.PriorDict()
em_priors['redshift'] = source_redshift
em_priors['thc'] = Uniform(0.01, 0.2, 'thc', latex_label = r'$\theta_{\mathrm{core}}$')
em_priors['logn0'] = Uniform(-4, 2, 'logn0', latex_label = r'$\log_{10} n_{\mathrm{ism}}$')
em_priors['p'] = Uniform(2,3, 'p', latex_label = r'$p$')
em_priors['fudge'] = Uniform(0.01, 0.1, 'fudge', latex_label = r'$f_{\mathrm{fudge}}$')
em_priors['logepse'] = grb_injection_parameters['logepse']
em_priors['logepsb'] = grb_injection_parameters['logepsb']
em_priors['ksin'] = grb_injection_parameters['ksin']
em_priors['g0'] = grb_injection_parameters['g0']

# We can now set up the GW part of the analysis.
# This follows the standard bilby workflow, specifically the relative binning example available at
# https://git.ligo.org/lscsoft/bilby/-/blob/master/examples/gw_examples/injection_examples/relative_binning.py

duration = 32
sampling_frequency = 2048
start_time = gw_injection_parameters['geocent_time'] + 2 - duration
np.random.seed(123)
# Fixed arguments passed into the source model. The analysis starts at 40 Hz
waveform_arguments = dict(waveform_approximant='IMRPhenomD_NRTidal',
                          reference_frequency=50., minimum_frequency=40.0)
# Create the waveform_generator using a LAL Binary Neutron Star source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use one interferometer
# (LIGO-Hanford (H1).
# These default to their design sensitivity and start at 40 Hz.
interferometers = bilby.gw.detector.InterferometerList(['H1'])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)
interferometers.inject_signal(parameters=gw_injection_parameters,
                              waveform_generator=waveform_generator)

# Set up the fiducial parameters for the relative binning likelihood to be the
# injected parameters.
fiducial_parameters = gw_injection_parameters.copy()
m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(m1, m2)
fiducial_parameters["symmetric_mass_ratio"] = bilby.gw.conversion.component_masses_to_symmetric_mass_ratio(m1, m2)

# Load the default prior for binary neutron stars.
gw_priors = bilby.gw.prior.BNSPriorDict()
# gw_priors.pop('chirp_mass')
# gw_priors.pop('mass_ratio')
# Fix some GW parameters for speed.
for key in ['ra', 'dec', 'chi_1', 'chi_2', 'phase','geocent_time', 'psi']:
    gw_priors[key] = gw_injection_parameters[key]
# Set up a different GW prior on some parameters.
gw_priors['chirp_mass'] = bilby.core.prior.Gaussian(1.215, 0.1, name="chirp_mass", unit="$M_{\\odot}$")
# gw_priors['mass_1'] = Uniform(1.2, 1.6, 'mass_1', latex_label = r'$m_1$')
# gw_priors['mass_2'] = Uniform(1.2, 1.6, 'mass_1', latex_label = r'$m_1$')
gw_priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=10, maximum=250)
gw_priors['theta_jn'] = bilby.core.prior.Uniform(0, 0.8, 'theta_jn', latex_label = r'$\theta_{\mathrm{observer}}$')
gw_priors['lambda_1'] = bilby.gw.prior.Uniform(name='lambda_1', minimum=0, maximum=5000)
gw_priors['lambda_2'] = bilby.gw.prior.Uniform(name='lambda_2', minimum=0, maximum=5000)

# Initialise the likelihood by passing in the interferometer data (ifos) and the waveform generator
GW_likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
    priors=gw_priors,
    distance_marginalization=False,
    fiducial_parameters=fiducial_parameters,
    time_marginalization=False,
    phase_marginalization=False)

# We can now set up the joint likelihood interface.
joint_likelihood = bilby.core.likelihood.JointLikelihood(GW_likelihood, em_likelihood)
priors_emgw = em_priors.copy()
priors_emgw.update(gw_priors)

nlive = 1000
walks = 100
label = 'emgw'
outdir = 'joint_analysis'
clean = True
all_injection_parameters = dict(grb_injection_parameters, **gw_injection_parameters)
result = bilby.run_sampler(joint_likelihood, priors=priors_emgw, label=label, sampler='pymultinest', nlive=nlive,
                            outdir=outdir, plot=True, use_ratio=False, walks = walks, resume=True, clean=clean,
                               injection_parameters = all_injection_parameters)
result.plot_corner(smooth=1.2)

# Keep in mind that the above is not a redback result object, but a standard bilby one.
# To see a fit of the lightcurve or redback specific result attributes/method you need to use the analysis class.

redback.analysis.plot_lightcurve(transient=sim_afterglow, parameters=result.posterior.sample(100),
                                 model=grb_afterglow_model, model_kwargs=afterglow_kwargs)

# Let's now also fit the afterglow and GW on their own so we can easily compare the effect of the joint analysis.

# First the afterglow. We also overwrite the save location to save everything to the same directory.
afterglow_kwargs = {}
afterglow_kwargs['output_format'] = 'flux_density'
afterglow_kwargs['frequency'] = np.ones(len(time))*2e17
em_priors = bilby.core.prior.PriorDict()
em_priors['redshift'] = source_redshift
em_priors['thc'] = Uniform(0.01, 0.2, 'thc', latex_label = r'$\theta_{\mathrm{core}}$')
em_priors['logn0'] = Uniform(-4, 2, 'logn0', latex_label = r'$\log_{10} n_{\mathrm{ism}}$')
em_priors['p'] = Uniform(2,3, 'p', latex_label = r'$p$')
em_priors['fudge'] = Uniform(0.01, 0.1, 'fudge', latex_label = r'$f_{\mathrm{fudge}}$')
em_priors['logepse'] = grb_injection_parameters['logepse']
em_priors['logepsb'] = grb_injection_parameters['logepsb']
em_priors['ksin'] = grb_injection_parameters['ksin']
em_priors['g0'] = grb_injection_parameters['g0']
em_priors['chirp_mass'] = gw_priors['chirp_mass']
em_priors['mass_ratio'] = gw_priors['mass_ratio']
em_priors['mass_1'] = gw_priors['mass_1']
em_priors['mass_2'] = gw_priors['mass_2']
em_priors['theta_jn'] = gw_priors['theta_jn']
em_result = redback.fit_model(model=grb_afterglow_model, transient=sim_afterglow, prior=em_priors, plot=False,
                              sampler='pymultinest', outdir=outdir, label='afterglow_only',
                              injection_parameters=all_injection_parameters, resume=True, clean=clean,
                              model_kwargs=afterglow_kwargs)
redback.analysis.plot_lightcurve(transient=sim_afterglow, parameters=em_result.posterior.sample(100),
                                 model=grb_afterglow_model, model_kwargs=afterglow_kwargs)
em_result.plot_corner(smooth=1.2)

# Now the GW.
gw_result = bilby.run_sampler(GW_likelihood, priors=gw_priors, label='gw_only', sampler='pymultinest', nlive=nlive,
                              outdir=outdir, injection_parameters=all_injection_parameters, resume=True, clean=clean)
gw_result.plot_corner(smooth=1.2)
