import os
import unittest
import numpy as np
import bilby
from os import listdir
from os.path import dirname
import pandas as pd
from pathlib import Path
from shutil import rmtree
import redback

_dirname = dirname(__file__)

class TestPlotModels(unittest.TestCase):
    outdir = "testing_plotting"

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
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(2))

    def test_plotting(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3])
        yobs = np.array([1e-3, 1e-3, 1e-3])
        yerr = np.ones_like(yobs) * 1e-4
        bands = np.array(['r', 'r', 'r'])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                posterior = self.get_posterior(file=f)
                transient = redback.supernova.Supernova(time=times, flux_density=yobs,
                                                        flux_density_err=yerr, bands=bands,
                                                        name='test',data_mode='flux_density',
                                                        use_phase_model=False)

                kwargs['output_format'] = 'flux_density'
                redback.analysis.plot_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)
                redback.analysis.plot_multiband_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)

class TestPlotDifferentBands(unittest.TestCase):
    outdir = "testing_plotting"

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
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(2))

    def test_plotting(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3])
        yobs = np.array([1e-3, 1e-3, 1e-3])
        yerr = np.ones_like(yobs) * 1e-4
        bands = np.array(['r', 'z', 'u'])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                posterior = self.get_posterior(file=f)
                transient = redback.supernova.Supernova(time=times, magnitude=yobs,
                                                        magnitude_err=yerr, bands=bands,
                                                        name='test',data_mode='magnitude',
                                                        use_phase_model=False)

                kwargs['output_format'] = 'magnitude'
                redback.analysis.plot_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)
                redback.analysis.plot_multiband_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)

class TestMagnitudePlot(unittest.TestCase):
    outdir = "testing_plotting"

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
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(2))

    def test_plotting(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3])
        yobs = np.array([1e-3, 1e-3, 1e-3])
        yerr = np.ones_like(yobs) * 1e-4
        bands = np.array(['r', 'r', 'r'])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                posterior = self.get_posterior(file=f)
                transient = redback.supernova.Supernova(time=times, magnitude=yobs,
                                                        magnitude_err=yerr, bands=bands,
                                                        name='test',data_mode='magnitude',
                                                        use_phase_model=False)

                kwargs['output_format'] = 'magnitude'
                redback.analysis.plot_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)
                redback.analysis.plot_multiband_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)

class TestFluxPlot(unittest.TestCase):
    outdir = "testing_plotting"

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
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(2))

    def test_plotting(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3])
        yobs = np.array([1e-3, 1e-3, 1e-3])
        yerr = np.ones_like(yobs) * 1e-4
        bands = np.array(['r', 'r', 'r'])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                posterior = self.get_posterior(file=f)
                transient = redback.supernova.Supernova(time=times, flux=yobs,
                                                        flux_err=yerr, bands=bands,
                                                        name='test',data_mode='flux',
                                                        use_phase_model=False)

                kwargs['output_format'] = 'flux'
                redback.analysis.plot_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)
                redback.analysis.plot_multiband_lightcurve(transient=transient, parameters=posterior,
                                                 model=model_name, model_kwargs=kwargs)

class TestPlotPhaseModels(unittest.TestCase):
    outdir = "testing_plotting"

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
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(2))

    def test_plotting(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3]) + 55855
        yobs = np.array([1e-3, 1e-3, 1e-3])
        yerr = np.ones_like(yobs) * 1e-4
        bands = np.array(['r', 'r', 'r'])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                posterior = self.get_posterior(file=f)
                transient = redback.supernova.Supernova(time_mjd=times, flux_density=yobs,
                                                        flux_density_err=yerr, bands=bands,
                                                        name='test',data_mode='flux_density',
                                                        use_phase_model=True)
                model = 't0_base_model'
                kwargs['t0'] = 55855
                kwargs['base_model'] = model_name
                kwargs['output_format'] = 'flux_density'
                redback.analysis.plot_lightcurve(transient=transient, parameters=posterior,
                                                 model=model, model_kwargs=kwargs)
                redback.analysis.plot_multiband_lightcurve(transient=transient, parameters=posterior,
                                                 model=model, model_kwargs=kwargs)