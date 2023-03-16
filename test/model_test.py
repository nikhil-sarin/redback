import unittest
from os import listdir
from os.path import dirname

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
            function = redback.model_library.all_models_dict[model_name]
            ys = function(times, **prior.sample(), **kwargs)
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
            if model_name == 'trapped_magnetar':
                kwargs['output_format'] = 'luminosity'
            else:
                kwargs['output_format'] = 'flux_density'
            prior = self.get_prior(file=f)
            function = redback.model_library.all_models_dict['t0_base_model']
            kwargs['base_model'] = model_name
            kwargs['t0'] = 55855
            ys = function(times, **prior.sample(), **kwargs)
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
            else:
                kwargs['output_format'] = 'magnitude'
            prior = self.get_prior(file=f)
            function = redback.model_library.all_models_dict[model_name]
            ys = function(times, **prior.sample(), **kwargs)
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
                        'tde_analytical', 'basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'flux'
                prior = self.get_prior(file=f)
                function = redback.model_library.all_models_dict[model_name]
                ys = function(times, **prior.sample(), **kwargs)
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
                function = redback.model_library.all_models_dict['integrated_flux_afterglowpy_base_model']
                ys = function(times, **prior.sample(), **kwargs)
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