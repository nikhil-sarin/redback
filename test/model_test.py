import unittest
from os import listdir
from os.path import dirname
from unittest import mock

import bilby
import numpy as np

import redback.model_library

_dirname = dirname(__file__)


class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3])
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            if model_name == 'trapped_magnetar':
                kwargs['output_format'] = 'luminosity'
            else:
                kwargs['output_format'] = 'flux_density'
            prior = self.get_prior(file=f)
            sample = prior.sample()
            if model_name == 'polytrope_eos_two_component_bns':
                sample['gamma_1'] = 4.04
                sample['gamma_2'] = 2.159
                sample['gamma_3'] = 3.688
                sample['log_p'] = 33.72
            function = redback.model_library.all_models_dict[model_name]
            ys = function(times, **sample, **kwargs)
            self.assertEqual(len(times), len(ys))

class TestPhaseModels(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3]) + 55855
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            skip_dict = ['bazin_sne', 'villar_sne', 'blackbody_spectrum_with_absorption_and_emission_lines',
                         'powerlaw_spectrum_with_absorption_and_emission_lines']
            if model_name in skip_dict:
                print('Skipping {}'.format(model_name))
                pass
            else:
                if model_name == 'trapped_magnetar':
                    kwargs['output_format'] = 'luminosity'
                else:
                    kwargs['output_format'] = 'flux_density'
                prior = self.get_prior(file=f)
                function = redback.model_library.all_models_dict['t0_base_model']
                kwargs['base_model'] = model_name
                kwargs['t0'] = 55855
                sample = prior.sample()
                if model_name == 'polytrope_eos_two_component_bns':
                    sample['gamma_1'] = 4.04
                    sample['gamma_2'] = 2.159
                    sample['gamma_3'] = 3.688
                    sample['log_p'] = 33.72
                ys = function(times, **sample, **kwargs)
                self.assertEqual(len(times), len(ys))

class TestMagnitudeOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3])
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            if model_name == 'trapped_magnetar':
                kwargs['output_format'] = 'luminosity'
            elif model_name in ['tophat_and_twocomponent', 'tophat_and_twolayerstratified',
                                'tophat_and_arnett', 'tophatredback_and_twolayerstratified']:
                pass
            else:
                kwargs['output_format'] = 'magnitude'
            prior = self.get_prior(file=f)
            sample = prior.sample()
            if model_name == 'polytrope_eos_two_component_bns':
                sample['gamma_1'] = 4.04
                sample['gamma_2'] = 2.159
                sample['gamma_3'] = 3.688
                sample['log_p'] = 33.72
            function = redback.model_library.all_models_dict[model_name]
            ys = function(times, **sample, **kwargs)
            self.assertEqual(len(times), len(ys))

class TestFluxOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova', 'csm_nickel']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'flux'
                prior = self.get_prior(file=f)
                sample = prior.sample()
                if model_name == 'polytrope_eos_two_component_bns':
                    sample['gamma_1'] = 4.04
                    sample['gamma_2'] = 2.159
                    sample['gamma_3'] = 3.688
                    sample['log_p'] = 33.72
                function = redback.model_library.all_models_dict[model_name]
                ys = function(times, **sample, **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))


class TestIntegratedFluxModelFlux(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=np.array([2e14,6e14]), bands='ztfg')
        times = np.array([1, 2, 3])
        valid_models = ['gaussiancore']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                prior = self.get_prior(file=f)
                kwargs['base_model'] = model_name
                sample = prior.sample()
                function = redback.model_library.all_models_dict['integrated_flux_afterglowpy_base_model']
                ys = function(times, **sample, **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestIntegratedFluxModelRate(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=np.array([2e14,6e14]), bands='ztfg')
        times = np.array([1, 2, 3])
        valid_models = ['smoothpowerlaw']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                prior = self.get_prior(file=f)
                kwargs['base_model'] = model_name
                function = redback.model_library.all_models_dict['integrated_flux_rate_model']
                ys = function(times, **prior.sample(), **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestExtinctionModelsFluxDensity(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3]) + 58555
        valid_models = ['arnett', 'slsn','one_component_kilonova_model'
                        'tde_analytical', 'basic_mergernova', 'gaussiancore',
                        'smoothpowerlaw', 'shock_cooling']
        supernova_models = ['arnett', 'slsn']
        kilonova_models = ['one_component_kilonova_model']
        tde_models = ['tde_analytical']
        magnetardriven_models = ['basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'flux_density'
                kwargs['base_model'] = model_name
                prior = self.get_prior(file=f)
                prior['av'] = 0.5
                prior['t0'] = 58555
                if model_name in supernova_models:
                    function = 't0_supernova_extinction'
                elif model_name in kilonova_models:
                    function = 't0_kilonova_extinction'
                elif model_name in tde_models:
                    function = 't0_tde_extinction'
                elif model_name in magnetardriven_models:
                    function = 't0_magnetar_driven_extinction'
                if model_name == 'smoothpowerlaw':
                    function = 't0_afterglow_extinction'
                if model_name == 'gaussiancore':
                    function = 't0_afterglow_extinction_model_d2g'
                    kwargs['lognh'] = 22
                    kwargs['factor'] = 5
                    prior.pop('av')
                if model_name == 'shock_cooling':
                    function = 't0_shock_powered_extinction'
                function = redback.model_library.all_models_dict[function]
                ys = function(times, **prior.sample(), **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestExtinctionModelsMagnitude(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3]) + 58555
        valid_models = ['arnett', 'slsn','one_component_kilonova_model'
                        'tde_analytical', 'basic_mergernova', 'shock_cooling']
        supernova_models = ['arnett', 'slsn']
        kilonova_models = ['one_component_kilonova_model']
        tde_models = ['tde_analytical']
        magnetardriven_models = ['basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'magnitude'
                kwargs['base_model'] = model_name
                prior = self.get_prior(file=f)
                prior['av'] = 0.5
                prior['t0'] = 58555
                if model_name in supernova_models:
                    function = 't0_supernova_extinction'
                elif model_name in kilonova_models:
                    function = 't0_kilonova_extinction'
                elif model_name in tde_models:
                    function = 't0_tde_extinction'
                elif model_name in magnetardriven_models:
                    function = 't0_magnetar_driven_extinction'
                if model_name == 'shock_cooling':
                    function = 't0_shock_powered_extinction'
                function = redback.model_library.all_models_dict[function]
                ys = function(times, **prior.sample(), **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestHomologousExpansion(unittest.TestCase):
    def setUp(self):
        self.time = np.array([1, 2, 3])
        self.base_model = 'arnett_bolometric'

    def tearDown(self) -> None:
        pass

    def get_prior(self):
        prior_dict = redback.priors.get_priors(self.base_model)
        return prior_dict

    def test_base_model(self):
        with mock.patch('redback.transient_models.supernova_models.homologous_expansion_supernova') as m:
            kwargs = dict(frequency=2e14, bands='ztfg')
            prior = self.get_prior()
            kwargs['output_format'] = 'flux_density'
            prior.pop('vej')
            prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
            prior['redshift'] = 0.075
            prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
            function = redback.transient_models.supernova_models.homologous_expansion_supernova
            m.base_model = self.base_model
            sample = prior.sample()
            actual = function(self.time, base_model=self.base_model, **sample, **kwargs)
            m.assert_called_with(self.time, base_model=self.base_model, **sample, **kwargs)

    def test_fluxdensity_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'flux_density'
        function = redback.model_library.all_models_dict['homologous_expansion_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

    def test_magnitude_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'magnitude'
        function = redback.model_library.all_models_dict['homologous_expansion_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

class TestThinShellExpansion(unittest.TestCase):
    def setUp(self):
        self.time = np.array([1, 2, 3])
        self.base_model = 'arnett_bolometric'

    def tearDown(self) -> None:
        pass

    def get_prior(self):
        prior_dict = redback.priors.get_priors(self.base_model)
        return prior_dict

    def test_base_model(self):
        with mock.patch('redback.transient_models.supernova_models.thin_shell_supernova') as m:
            kwargs = dict(frequency=2e14, bands='ztfg')
            prior = self.get_prior()
            kwargs['output_format'] = 'flux_density'
            prior.pop('vej')
            prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
            prior['redshift'] = 0.075
            prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
            function = redback.transient_models.supernova_models.thin_shell_supernova
            m.base_model = self.base_model
            sample = prior.sample()
            actual = function(self.time, base_model=self.base_model, **sample, **kwargs)
            m.assert_called_with(self.time, base_model=self.base_model, **sample, **kwargs)

    def test_fluxdensity_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'flux_density'
        function = redback.model_library.all_models_dict['thin_shell_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

    def test_magnitude_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'magnitude'
        function = redback.model_library.all_models_dict['thin_shell_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

