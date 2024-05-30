import unittest
import numpy as np
import bilby
from os import listdir
from os.path import dirname
from bilby.core.prior import Constraint, Uniform, LogUniform
import pandas as pd
from pathlib import Path
from shutil import rmtree
import redback

_dirname = dirname(__file__)


class TestLoadPriors(unittest.TestCase):

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def test_load_priors(self):
        for f in self.prior_files:
            prior_dict = bilby.prior.PriorDict()
            prior_dict.from_file(f"{self.path_to_files}{f}")

class TestConstraints(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_slsn_constraint(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.slsn_constraint)
        _prior = redback.priors.get_priors(model='slsn')
        priors.update(_prior)
        priors['erot_constraint'] = Constraint(0, 1)
        priors['t_nebula_min'] = Constraint(0, 400)
        samples = pd.DataFrame(priors.sample(1000))
        mej = samples['mej'] * redback.constants.solar_mass
        vej = samples['vej'] * redback.constants.km_cgs
        kappa = samples['kappa']
        mass_ns = samples['mass_ns']
        p0 = samples['p0']
        samples['erot'] = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
        neutrino_energy = 1e51
        samples['ek'] = 0.5 * mej * vej**2
        samples['etot'] = samples['ek'] + neutrino_energy
        samples['tneb'] = np.sqrt(3 * kappa * mej / (4 * np.pi * vej ** 2)) / 86400
        self.assertTrue(np.all(samples['erot'].values >= samples['etot'].values))
        self.assertTrue(np.all(samples['tneb'].values >= 100))

    def test_basic_magnetar_powered_sn_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.basic_magnetar_powered_sn_constraints)
        _prior = redback.priors.get_priors(model='basic_magnetar_powered')
        priors.update(_prior)
        priors['erot_constraint'] = Constraint(0, 1)
        samples = pd.DataFrame(priors.sample(1000))
        mej = samples['mej'] * redback.constants.solar_mass
        vej = samples['vej'] * redback.constants.km_cgs
        mass_ns = samples['mass_ns']
        p0 = samples['p0']
        samples['erot'] = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
        neutrino_energy = 1e51
        samples['ek'] = 0.5 * mej * vej**2
        samples['etot'] = samples['ek'] + neutrino_energy
        self.assertTrue(np.all(samples['erot'].values >= samples['ek'].values))

    def test_general_magnetar_powered_sn_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.general_magnetar_powered_sn_constraints)
        _prior = redback.priors.get_priors(model='general_magnetar_slsn')
        priors.update(_prior)
        priors['erot_constraint'] = Constraint(0, 1)
        samples = pd.DataFrame(priors.sample(1000))
        mej = samples['mej'] * redback.constants.solar_mass
        vej = samples['vej'] * redback.constants.km_cgs
        l0 = samples['l0']
        tau = samples['tsd']
        samples['erot'] = 2*l0*tau
        samples['ek'] = 0.5 * mej * vej**2
        self.assertTrue(np.all(samples['erot'].values >= samples['ek'].values))

    def test_tde_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.tde_constraints)
        priors['pericenter_radius'] = Uniform(0.1, 100, name='pericenter_radius', latex_label=r'$r_{\mathrm{p}}$~[AU]')
        priors['mass_bh'] = Uniform(1e5, 5e8, name='mass_bh', latex_label=r'$M_{\mathrm{BH}}$~[M$_{\odot}$]')
        priors['disruption_radius'] = Constraint(0, 1)
        samples = pd.DataFrame(priors.sample(1000))
        mass_bh = samples['mass_bh']
        sch_rad = (2 * redback.constants.graviational_constant * mass_bh * redback.constants.solar_mass /
                   (redback.constants.speed_of_light**2)).values / redback.constants.au_cgs
        self.assertTrue(np.all(samples['pericenter_radius'].values >= sch_rad))

    def test_gaussianrise_tde_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.gaussianrise_tde_constraints)
        _prior = redback.priors.get_priors(model='gaussianrise_tde')
        priors.update(_prior)
        priors['peak_time'] = Uniform(20, 60, name='peak_time', latex_label=r'$t_{\mathrm{peak}}$~[days]')
        priors['sigma_t'] = Uniform(5, 10, name='sigma_t', latex_label=r'$\sigma$~[days]')
        priors['mbh_6'] = Uniform(0.3, 20, name='mbh_6', latex_label=r'$M_{\mathrm{BH}}~[10^{6}~M_{\odot}]$')
        priors['stellar_mass'] = Uniform(0.3, 5, name='stellar_mass', latex_label=r'$M_{\mathrm{star}} [M_{\odot}]$')
        priors['eta'] = LogUniform(1e-3, 0.1, name='eta', latex_label=r'$\eta$')
        priors['alpha'] = LogUniform(1e-2, 1e-1, name='alpha', latex_label=r'$\alpha$')
        priors['beta'] = Uniform(1, 30, name='beta', latex_label=r'$\beta$')
        priors['redshift'] = 0.01
        priors['beta_high'] = Constraint(0, 1)
        priors['tfb_max'] = Constraint(0, 1)
        samples = pd.DataFrame(priors.sample(1000))
        ms = samples['stellar_mass']
        mbh6 = samples['mbh_6']
        betamax = 12. * (ms ** (7. / 15.)) * (mbh6 ** (-2. / 3.))
        tfb = redback.utils.calc_tfb(binding_energy_const=0.8, mbh_6=mbh6, stellar_mass=ms) / 86400
        tfb_obs = tfb * (1 + samples['redshift'])
        self.assertTrue(np.all(samples['beta'].values <= betamax))
        self.assertTrue(np.all(tfb_obs >= samples['peak_time']))

    def test_nuclear_burning_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.nuclear_burning_constraints)
        _prior = redback.priors.get_priors(model='arnett')
        priors.update(_prior)
        priors['emax_constraint'] = Constraint(0, 1)
        samples = pd.DataFrame(priors.sample(1000))
        mej = samples['mej'] * redback.constants.solar_mass
        vej = samples['vej'] * redback.constants.km_cgs
        fnickel = samples['f_nickel']
        excess_constant = -(56.0 / 4.0 * 2.4249 - 53.9037) / redback.constants.proton_mass * redback.constants.mev_cgs
        samples['ek'] = 0.5 * mej * (vej / 2.0) **2
        samples['emax'] = excess_constant * mej * fnickel
        self.assertTrue(np.all(samples['emax'].values >= samples['ek'].values))

    def test_simple_fallback_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.simple_fallback_constraints)
        _prior = redback.priors.get_priors(model='tde_analytical')
        priors.update(_prior)
        priors['en_constraint'] = Constraint(0, 1)
        priors['t_nebula_min'] = Constraint(0, 400)
        samples = pd.DataFrame(priors.sample(1000))
        mej = samples['mej'] * redback.constants.solar_mass
        vej = samples['vej'] * redback.constants.km_cgs
        kappa = samples['kappa']
        l0 = samples['l0']
        t0 = samples['t_0_turn']
        samples['efb'] = l0 * 5./2./(t0 * redback.constants.day_to_s)**(2./3.)
        neutrino_energy = 1e51
        samples['ek'] = 0.5 * mej * vej**2
        samples['etot'] = samples['efb'] + neutrino_energy
        samples['tneb'] = np.sqrt(3 * kappa * mej / (4 * np.pi * vej ** 2)) / 86400
        self.assertTrue(np.all(samples['etot'].values >= samples['ek'].values))
        self.assertTrue(np.all(samples['tneb'].values >= 100))

    def test_csm_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.csm_constraints)
        _prior = redback.priors.get_priors(model='csm_interaction')
        priors.update(_prior)
        priors['shock_time'] = Constraint(0, 0.5)
        priors['photosphere_constraint_1'] = Constraint(0, 1)
        priors['photosphere_constraint_2'] = Constraint(0, 1)
        samples = pd.DataFrame(priors.sample(100))
        mej = samples['mej'].values * redback.constants.solar_mass
        csm_mass = samples['csm_mass'].values * redback.constants.solar_mass
        kappa = samples['kappa'].values
        r0 = samples['r0'].values * redback.constants.au_cgs
        vej = samples['vej'].values * redback.constants.km_cgs
        nn = np.ones(len(samples)) * 12
        delta = np.ones(len(samples))
        eta = samples['eta'].values
        rho = samples['rho'].values
        Esn = 3. * vej**2 * mej / 10.

        AA = np.zeros(len(mej))
        Bf = np.zeros(len(mej))
        for x in range(len(mej)):
            csm_properties = redback.utils.get_csm_properties(nn[x], eta[x])
            AA[x] = csm_properties.AA
            Bf[x] = csm_properties.Bf
        qq = rho * r0 ** eta
        # outer CSM shell radius
        radius_csm = ((3.0 - eta) / (4.0 * np.pi * qq) * csm_mass + r0 ** (3.0 - eta)) ** (
                1.0 / (3.0 - eta))
        # photosphere radius
        r_photosphere = abs((-2.0 * (1.0 - eta) / (3.0 * kappa * qq) +
                             radius_csm ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))

        # mass of the optically thick CSM (tau > 2/3).
        mass_csm_threshold = np.abs(4.0 * np.pi * qq / (3.0 - eta) * (
                r_photosphere ** (3.0 - eta) - r0 ** (3.0 - eta)))

        g_n = (1.0 / (4.0 * np.pi * (nn - delta)) * (
                2.0 * (5.0 - delta) * (nn - 5.0) * Esn) ** ((nn - 3.) / 2.0) / (
                       (3.0 - delta) * (nn - 3.0) * mej) ** ((nn - 5.0) / 2.0))

        tshock = ((radius_csm - r0) / Bf / (AA * g_n / qq) ** (
                1. / (nn - eta))) ** ((nn - eta) / (nn - 3))

        diffusion_time = np.sqrt(2. * kappa * mass_csm_threshold / (vej * 13.7 * 3.e10))

        self.assertTrue(np.all(diffusion_time <= tshock))
        self.assertTrue(np.all(r_photosphere <= radius_csm))
        self.assertTrue(np.all(r_photosphere >= r0))

    def test_piecewise_polytrope_eos_constraints(self):
        priors = bilby.prior.PriorDict(conversion_function=redback.constraints.piecewise_polytrope_eos_constraints)
        _prior = redback.priors.get_priors(model='polytrope_eos_two_component_bns')
        priors.update(_prior)
        priors['maximum_eos_mass'] = Constraint(1.5, 5)
        priors['maximum_speed_of_sound'] = Constraint(0, 1.15)
        samples = priors.sample(100)
        max_mass = redback.constraints.calc_max_mass(**samples)
        print(max_mass)
        cs = redback.constraints.calc_speed_of_sound(**samples)
        self.assertTrue(np.all(max_mass >= 1.5))
        self.assertTrue(np.all(max_mass <= 5))
        self.assertTrue(np.all(cs <= 1.15))


class TestCornerPlotPriorSamples(unittest.TestCase):
    outdir = "testing_corner"

    @classmethod
    def setUpClass(cls) -> None:
        Path(cls.outdir).mkdir(exist_ok=True, parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.outdir)

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def get_posterior(self, file):
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(100))

    def get_result(self, file):
        prior = self.get_prior(file=file)
        posterior = self.get_posterior(file=file)
        search_parameter_keys = [k for k, v in prior.items() if
                                 not isinstance(v, (bilby.core.prior.DeltaFunction, bilby.core.prior.Constraint, float, int))]
        fixed_parameter_keys = [k for k, v in prior.items() if isinstance(v, (bilby.core.prior.DeltaFunction, float, int))]
        constraint_parameter_keys = [k for k, v in prior.items() if isinstance(v, bilby.core.prior.Constraint)]
        return bilby.result.Result(label=file, outdir=self.outdir,
                                   search_parameter_keys=search_parameter_keys,
                                   fixed_parameter_keys=fixed_parameter_keys,
                                   constraint_parameter_keys=constraint_parameter_keys, priors=prior,
                                   sampler_kwargs=dict(), injection_parameters=None,
                                   meta_data=None, posterior=posterior, samples=None,
                                   nested_samples=None, log_evidence=0,
                                   log_evidence_err=0, information_gain=0,
                                   log_noise_evidence=0, log_bayes_factor=0,
                                   log_likelihood_evaluations=0,
                                   log_prior_evaluations=0, sampling_time=0, nburn=0,
                                   num_likelihood_evaluations=0, walkers=0,
                                   max_autocorrelation_time=0, use_ratio=False,
                                   version=None)

    def test_plot_priors(self):
        for f in self.prior_files:
            print(f)
            res = self.get_result(file=f)
            res.plot_corner()
