import os
import unittest
from unittest.mock import MagicMock, patch
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import bilby
from os import listdir
from os.path import dirname
import pandas as pd
from pathlib import Path
from shutil import rmtree
import redback
from redback.analysis import (plot_evolution_parameters, plot_spectrum, plot_gp_lightcurves,
                              fit_temperature_and_radius_gp, generate_new_transient_data_from_gp)

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
        bands = np.array(['sdssr', 'sdssz', 'sdssu'])
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
        bands = np.array(['sdssr', 'sdssr', 'sdssr'])
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
        bands = np.array(['sdssr', 'sdssr', 'sdssr'])
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

# Dummy “evolving magnetar” model – used by plot_evolution_parameters:
DummyEvolvingMagnetarOutput = namedtuple("DummyEvolvingMagnetarOutput", ["nn", "mu", "alpha"])
def dummy_evolving_magnetar_only(time, **kwargs):
    # Return dummy arrays (one value per time point)
    return DummyEvolvingMagnetarOutput(
        nn = np.ones_like(time)*3.0,
        mu = np.ones_like(time)*1e30,
        alpha = np.ones_like(time)*2.5
    )

# Dummy spectrum model – used by plot_spectrum:
DummySpectrumOutput = namedtuple("DummySpectrumOutput", ["lambdas", "time", "spectra"])
def dummy_spectrum_model(time_to_plot, **kwargs):
    # Create an array of wavelengths (in Angstroms)
    lambdas = np.linspace(4000, 7000, 100)
    # Assume a dummy time array in seconds; for simplicity, create one with the same number of elements as time_to_plot
    time_array = np.linspace(0, (len(time_to_plot)-1)*86400, len(time_to_plot))
    # Dummy spectra: for each time, create a spectrum (here a simple linear ramp)
    spectra = np.tile(np.linspace(1, 100, 100), (len(time_to_plot), 1))
    return DummySpectrumOutput(lambdas=lambdas, time=time_array, spectra=spectra)

# Dummy “find_nearest” helper (used by plot_spectrum)
def dummy_find_nearest(arr, target):
    idx = (np.abs(arr - target)).argmin()
    return arr[idx], idx

# Dummy GP classes used as stand‑ins by the GP‐plotting functions.
class DummyGP:
    def predict(self, scaled_y, X_new, return_var=True, return_cov=True):
        n = X_new.shape[0] if isinstance(X_new, np.ndarray) and X_new.ndim == 2 else len(X_new)
        prediction = np.ones(n) * 2.0
        cov = np.eye(n) * 0.25
        return prediction, cov

class Dummy1DGP:
    def predict(self, scaled_y, t_new, return_var=True, return_cov=True):
        n = len(t_new)
        prediction = np.ones(n) * 2.0
        cov = np.eye(n) * 0.25
        return prediction, cov

# Dummy GP output container; for the 2D (with frequency) branch
class DummyGPOutput:
    def __init__(self, use_frequency=True, unique_bands=None):
        """
        Initialize a dummy Gaussian Process output for testing.

        :param use_frequency: If True, use frequency-based GP; otherwise, use band-specific GP.
        :param unique_bands: List of unique bands for which GP output is created.
        """
        self.use_frequency = use_frequency
        self.y_scaler = 1.0
        if use_frequency:
            # Single GP for frequency-based mode
            self.gp = DummyGP()
            self.scaled_y = np.ones(10)  # Dummy scaled data
        else:
            # Dictionary of GPs for each band (band-specific mode)
            if unique_bands is None:
                unique_bands = []  # Avoid issues with missing bands
            self.gp = {band: Dummy1DGP() for band in unique_bands}
            self.scaled_y = {band: np.ones(10) for band in unique_bands}


# Dummy transient for GP plotting.
class DummyTransientForGP:
    def __init__(self, x, unique_bands, data_mode):
        self.x = np.array(x)
        self.use_phase_model = False
        self.data_mode = data_mode
        self.unique_bands = unique_bands  # Expect a list of band names
    @property
    def unique_frequencies(self):
        # Return a dummy array of frequencies associated with unique_bands.
        return np.array([8.43500e+14 for band in self.unique_bands])

# Dummy transient for generating new GP‐data. (A minimal dummy OpticalTransient)
class DummyOpticalTransient:
    def __init__(self, name, data_mode, unique_frequencies=None, unique_bands=None, redshift=0.1):
        self.name = name
        self.data_mode = data_mode
        self.redshift = redshift
        self.unique_frequencies = unique_frequencies if unique_frequencies is not None else np.array([8.43500e+14])
        self.unique_bands = unique_bands if unique_bands is not None else ["dummy"]

# Dummy GP output container for generate_new_transient_data_from_gp.
class DummyGPOutputForGenerate:
    def __init__(self, use_frequency=True):
        self.use_frequency = use_frequency
        self.y_scaler = 1.0
        self.scaled_y = np.ones(10) if use_frequency else {"dummy": np.ones(10)}
        self.gp = DummyGP() if use_frequency else {"dummy": Dummy1DGP()}

# Dummy bands_to_frequency function for use in plot_gp_lightcurves.
def dummy_bands_to_frequency(band_list):
    # For testing, simply return a constant frequency.
    return 8.43500e+14

# === Test classes below ===

class TestPlotEvolutionParameters(unittest.TestCase):
    def setUp(self):
        # Create a dummy result object with the required metadata and posterior DataFrame.
        self.dummy_metadata = {"time": np.array([1, 10, 100])}
        # Create a posterior DataFrame with a few dummy rows (the content is not critical)
        df = pd.DataFrame({"param1": [0.1, 0.2], "param2": [1, 2]})
        self.dummy_result = type("DummyResult", (), {"metadata": self.dummy_metadata, "posterior": df})
        # Patch the evolving magnetar model in redback.model_library.all_models_dict.
        self.orig_evolving = redback.model_library.all_models_dict.get("evolving_magnetar_only")
        redback.model_library.all_models_dict["evolving_magnetar_only"] = dummy_evolving_magnetar_only

    def tearDown(self):
        if self.orig_evolving is not None:
            redback.model_library.all_models_dict["evolving_magnetar_only"] = self.orig_evolving

    def test_plot_evolution_parameters_returns_fig_and_axes(self):
        # Call with a small number of random models
        fig, ax = plot_evolution_parameters(self.dummy_result, random_models=3)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(ax), 3)
        # Check that each axis has a ylabel (e.g. 'braking index', etc.)
        for a in ax:
            self.assertNotEqual(a.get_ylabel(), "")
        plt.close(fig)

class TestPlotSpectrum(unittest.TestCase):
    def setUp(self):
        # Patch the spectrum model to our dummy.
        self.orig_spec_model = redback.model_library.all_models_dict.get("dummy_spectrum_model")
        redback.model_library.all_models_dict["dummy_spectrum_model"] = dummy_spectrum_model
        # Patch the find_nearest function that is used inside plot_spectrum.
        self.find_nearest_patcher = patch('redback.utils.find_nearest', dummy_find_nearest)
        self.find_nearest_patcher.start()
        # In case day_to_s is expected (for converting time), define it if necessary.
        self.day_to_s = 86400

    def tearDown(self):
        if self.orig_spec_model is not None:
            redback.model_library.all_models_dict["dummy_spectrum_model"] = self.orig_spec_model
        self.find_nearest_patcher.stop()

    def test_plot_spectrum_returns_axes(self):
        parameters = {"some_parameter": 1}
        time_to_plot = np.array([1, 2])  # in days
        fig, tmp_ax = plt.subplots()
        ax = plot_spectrum("dummy_spectrum_model", parameters, time_to_plot, axes=tmp_ax)
        self.assertIsNotNone(ax)
        self.assertIn("Wavelength", ax.get_xlabel())
        plt.close(fig)

class TestPlotGPLightcurves(unittest.TestCase):
    @patch('redback.utils.bands_to_frequency', side_effect=dummy_bands_to_frequency)
    def test_plot_gp_lightcurves_with_frequency(self, mock_btf):
        dummy_trans = DummyTransientForGP(x=np.linspace(0, 10, 50), unique_bands=["g", "r"], data_mode="flux_density")
        dummy_gp_output = DummyGPOutput(use_frequency=True)
        fig, ax = plt.subplots()
        ax_out = plot_gp_lightcurves(dummy_trans, dummy_gp_output, axes=ax)
        self.assertIsNotNone(ax_out)
        self.assertGreater(len(ax_out.get_lines()), 0)
        plt.close(fig)

    @patch('redback.utils.bands_to_frequency', side_effect=dummy_bands_to_frequency)
    def test_plot_gp_lightcurves_without_frequency(self, mock_btf):
        dummy_trans = DummyTransientForGP(x=np.linspace(0, 10, 50), unique_bands=["g", "r"], data_mode="flux_density")
        dummy_gp_output = DummyGPOutput(use_frequency=False, unique_bands=["g", "r"])
        fig, ax = plt.subplots()
        ax_out = plot_gp_lightcurves(dummy_trans, dummy_gp_output, axes=ax)
        self.assertIsNotNone(ax_out)
        self.assertGreater(len(ax_out.get_lines()), 0)
        plt.close(fig)

class TestFitTemperatureAndRadiusGP(unittest.TestCase):
    def setUp(self):
        # Build a simple DataFrame with the required columns.
        self.data = pd.DataFrame({
            "epoch_times": np.linspace(1, 100, 20),
            "temperature": np.linspace(10000, 5000, 20),
            "radius": np.linspace(1e14, 5e14, 20),
            "temp_err": np.full(20, 500),
            "radius_err": np.full(20, 1e13)
        })
        # Use a simple george exponential-squared kernel.
        from george.kernels import ExpSquaredKernel
        self.kernelT = ExpSquaredKernel(metric=1.0)
        self.kernelR = ExpSquaredKernel(metric=1.0)

    def test_fit_temperature_and_radius_gp_without_plot(self):
        gp_T, gp_R = fit_temperature_and_radius_gp(self.data, self.kernelT, self.kernelR, plot=False)
        # Import the GP type from george (may be george.GP)
        from george.gp import GP
        self.assertIsInstance(gp_T, GP)
        self.assertIsInstance(gp_R, GP)

    def test_fit_temperature_and_radius_gp_with_plot(self):
        output = fit_temperature_and_radius_gp(self.data, self.kernelT, self.kernelR, plot=True, fit_in_log=True)
        self.assertEqual(len(output), 4)
        gp_T, gp_R, fig, axes = output
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 2)
        plt.close(fig)

class TestGenerateNewTransientDataFromGP(unittest.TestCase):
    def setUp(self):
        # Create a dummy gp_output for the “use_frequency” branch.
        self.gp_out = DummyGPOutputForGenerate(use_frequency=True)
        # Create a dummy new time array.
        self.t_new = np.linspace(0, 100, 10)
        # Create a dummy transient (OpticalTransient) with minimal required attributes.
        self.transient = DummyOpticalTransient(name="TestTransient", data_mode="flux_density", redshift=0.1)

    def test_generate_new_transient_data_from_gp_flux_density(self):
        new_transient = generate_new_transient_data_from_gp(self.gp_out, self.t_new, self.transient)
        # Check that the returned transient has a name ending with '_gp'
        self.assertTrue(new_transient.name.endswith("_gp"))
        # For flux_density mode, these attributes should be present.
        self.assertTrue(hasattr(new_transient, "flux_density"))
        self.assertTrue(hasattr(new_transient, "flux_density_err"))

    def test_generate_new_transient_data_from_gp_flux(self):
        self.transient.data_mode = "flux"
        new_transient = generate_new_transient_data_from_gp(self.gp_out, self.t_new, self.transient)
        self.assertTrue(hasattr(new_transient, "flux"))
        self.assertTrue(hasattr(new_transient, "flux_err"))

    def test_generate_new_transient_data_from_gp_magnitude(self):
        self.transient.data_mode = "magnitude"
        new_transient = generate_new_transient_data_from_gp(self.gp_out, self.t_new, self.transient)
        self.assertTrue(hasattr(new_transient, "magnitude"))
        self.assertTrue(hasattr(new_transient, "magnitude_err"))

    def test_generate_new_transient_data_from_gp_luminosity(self):
        self.transient.data_mode = "luminosity"
        new_transient = generate_new_transient_data_from_gp(self.gp_out, self.t_new, self.transient)
        # In luminosity mode, check for attributes such as 'Lum50' and 'Lum50_err'
        self.assertTrue(hasattr(new_transient, "Lum50"))
        self.assertTrue(hasattr(new_transient, "Lum50_err"))