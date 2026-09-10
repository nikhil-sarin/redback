import unittest
import tempfile
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import bilby

from redback.model_library import all_models_dict
from redback.result import RedbackResult
from redback.transient.afterglow import Afterglow
from redback.transient.prompt import PromptTimeSeries
from redback.transient.transient import OpticalTransient, Transient, Spectrum
from redback.sampler import fit_model, _get_filtered_upper_limit_sigma


# --- Dummy Model and Result --- #
def dummy_model(x, **kwargs):
    """A dummy model function that returns an array of ones."""
    return np.ones_like(x)


def constant_model(x, amplitude, **kwargs):
    return np.full_like(x, amplitude, dtype=float)


# Allow the model lookup via the standard dictionary.
all_models_dict["dummy_model"] = dummy_model


class DummyResult(RedbackResult):
    """A minimal dummy result class mimic."""

    def __init__(self):
        self.data = "dummy_result"

    def plot_spectrum(self, model):
        pass

    def plot_lightcurve(self, model):
        pass


# --- Revised Dummy Transient Classes --- #
class DummySpectrum(Spectrum):
    def __init__(self, outdir):
        # Set required attributes.
        self.data_mode = "flux_density"
        self.directory_structure = SimpleNamespace(directory_path=outdir)
        self.name = "DummySpectrum"
        self.use_phase_model = False  # Required by base transient.
        self.angstroms = np.linspace(4000, 7000, 100)
        self.flux_density = np.ones(100) * 1e-16
        self.flux_density_err = np.ones(100) * 1e-18
        # Define _bands: for spectrum, use dummy filter name for each wavelength.
        self._bands = np.array(["dummy"] * len(self.angstroms))
        # (Spectrum may not require _active_bands)


class DummyAfterglow(Afterglow):
    def __init__(self, outdir):
        self.data_mode = "flux_density"
        self.directory_structure = SimpleNamespace(directory_path=outdir)
        self.name = "DummyAfterglow"
        self.use_phase_model = False  # Required by base transient.
        self.x = np.linspace(0, 10, 50)
        self.x_err = np.zeros((2, 50))
        self.y = np.ones(50) * 10.0
        self.y_err = np.ones(50)
        self.photon_index = 1.0
        # Define _bands first so that self.bands is available.
        self._bands = np.array(["dummy"] * len(self.x))
        # Also provide _active_bands so that filtered_indices works.
        self._active_bands = self._bands.copy()
        # Now setting frequency calls the setter which uses self.bands.
        self.frequency = np.ones(len(self.x))


class DummyPromptTimeSeries(PromptTimeSeries):
    def __init__(self, outdir):
        self.data_mode = "counts"  # Acceptable for prompt data.
        self.directory_structure = SimpleNamespace(directory_path=outdir)
        self.name = "DummyPrompt"
        self.use_phase_model = False  # Required.
        self.x = np.linspace(0, 10, 50)
        self.bin_size = 1.0
        self.y = np.ones(50) * 5.0
        self.y_err = np.ones(50)
        # Provide dummy _bands (if required downstream).
        self._bands = np.array(["dummy"] * len(self.x))


class DummyOpticalTransient(OpticalTransient):
    def __init__(self, outdir):
        self.data_mode = "flux_density"
        self.directory_structure = SimpleNamespace(directory_path=outdir)
        self.name = "DummyOptical"
        self.use_phase_model = False  # Prevent errors in base transient.
        self.x = np.linspace(0, 10, 50)
        self.x_err = np.zeros((2, 50))
        self.y = np.ones(50) * 15.0
        self.y_err = np.ones(50)
        # Set _bands so that the bands property works.
        self._bands = np.array(["dummy"] * len(self.x))
        # Also provide _active_bands so that filtered_indices works.
        self._active_bands = self._bands.copy()
        # Upper limit support (none by default)
        self._detections = None
        self._upper_limit_sigma = 3.0


class DummyTransient(Transient):
    def __init__(self, outdir):
        self.data_mode = "flux_density"
        self.directory_structure = SimpleNamespace(directory_path=outdir)
        self.name = "DummyTransient"
        self.use_phase_model = False  # Required by base transient.
        self.x = np.linspace(0, 10, 50)
        self.x_err = np.zeros((2, 50))
        self.y = np.ones(50) * 20.0
        self.y_err = np.ones(50)
        # Define _bands and _active_bands.
        self._bands = np.array(["dummy"] * len(self.x))
        self._active_bands = self._bands.copy()
        # Upper limit support (none by default)
        self._detections = None
        self._upper_limit_sigma = 3.0


# Dummy object that is not a recognized transient type.
class DummyNotTransient:
    data_mode = "flux_density"  # Provide a dummy attribute.


# --- Tests for the fit_model function --- #
class TestFitModel(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for outdir.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = self.temp_dir.name
        # Default model_kwargs.
        self.model_kwargs = {"output_format": "flux_density"}
        self.sampler = "dynesty"
        self.nlive = 100
        self.walks = 50
        self.prior = bilby.prior.PriorDict()  # Empty PriorDict for testing.
        # Create a dummy RedbackResult to be returned by the sampler.
        self.dummy_result = DummyResult()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_spectrum(self, mock_run_sampler, mock_read_result):
        trans = DummySpectrum(self.outdir)
        # For spectrum, add a frequency array to model_kwargs.
        model_kwargs = self.model_kwargs.copy()
        model_kwargs["frequency"] = np.linspace(1e14, 1e15, len(trans.angstroms))
        mock_run_sampler.return_value = self.dummy_result

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir, label="TestSpectrum",
            sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
            model_kwargs=model_kwargs, plot=False
        )
        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_called_once()

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_afterglow(self, mock_run_sampler, mock_read_result):
        trans = DummyAfterglow(self.outdir)
        # Supply a frequency key (if needed) for consistency.
        model_kwargs = self.model_kwargs.copy()
        model_kwargs["frequency"] = np.linspace(1e14, 1e15, len(trans.x))
        mock_run_sampler.return_value = self.dummy_result

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir, label="TestAfterglow",
            sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
            model_kwargs=model_kwargs, plot=False
        )
        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_called_once()

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_prompt(self, mock_run_sampler, mock_read_result):
        trans = DummyPromptTimeSeries(self.outdir)
        # For prompt objects, add a dummy frequency array to model_kwargs.
        model_kwargs = self.model_kwargs.copy()
        model_kwargs["frequency"] = np.linspace(1e14, 1e15, len(trans.x))
        mock_run_sampler.return_value = self.dummy_result

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir, label="TestPrompt",
            sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
            model_kwargs=model_kwargs, plot=False
        )
        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_called_once()

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_optical_transient(self, mock_run_sampler, mock_read_result):
        trans = DummyOpticalTransient(self.outdir)
        # For optical transients, supply a frequency key in model_kwargs.
        model_kwargs = self.model_kwargs.copy()
        model_kwargs["frequency"] = np.linspace(1e14, 1e15, len(trans.x))
        mock_run_sampler.return_value = self.dummy_result

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir, label="TestOptical",
            sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
            model_kwargs=model_kwargs, plot=False
        )
        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_called_once()

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_transient_base(self, mock_run_sampler, mock_read_result):
        trans = DummyTransient(self.outdir)
        # For base transient objects, supply a frequency key as well.
        model_kwargs = self.model_kwargs.copy()
        model_kwargs["frequency"] = np.linspace(1e14, 1e15, len(trans.x))
        mock_run_sampler.return_value = self.dummy_result

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir, label="TestTransient",
            sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
            model_kwargs=model_kwargs, plot=False
        )
        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_called_once()

    def test_inconsistent_data_mode(self):
        # Test that if the transient's output_format does not match its data_mode, a ValueError is raised.
        trans = DummyTransient(self.outdir)
        trans.data_mode = "flux_density"
        inconsistent_kwargs = {"output_format": "magnitude"}
        with self.assertRaises(ValueError) as context:
            fit_model(
                transient=trans, model="dummy_model", outdir=self.outdir, label="TestInconsistency",
                sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
                model_kwargs=inconsistent_kwargs, plot=False
            )
        self.assertIn("inconsistent", str(context.exception))

    def test_unknown_transient_type(self):
        # Test that passing an object that is not a recognized transient type causes a ValueError.
        trans = DummyNotTransient()
        model_kwargs = self.model_kwargs.copy()
        model_kwargs['frequency'] = 1e15
        with self.assertRaises(ValueError) as context:
            fit_model(
                transient=trans, model="dummy_model", outdir=self.outdir, label="TestUnknown",
                sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
                model_kwargs=model_kwargs, plot=False
            )
        self.assertIn("not known", str(context.exception))

class TestFitModelAdditional(unittest.TestCase):
    """Additional fit_model tests covering previously untested branches."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = self.temp_dir.name
        self.prior = bilby.prior.PriorDict()
        self.dummy_result = DummyResult()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_filtered_upper_limit_sigma_matches_active_band_data(self):
        trans = DummyOpticalTransient(self.outdir)
        trans.x = np.arange(4.0)
        trans.x_err = None
        trans.y = np.array([20.0, 21.0, 22.0, 23.0])
        trans.y_err = np.ones(4) * 0.1
        trans._bands = np.array(["g", "r", "g", "r"])
        trans._active_bands = ["g"]
        trans._detections = np.array([True, False, False, True])
        trans._upper_limit_sigma = np.array([3.0, 4.0, 5.0, 6.0])

        sigma = _get_filtered_upper_limit_sigma(trans)

        np.testing.assert_array_equal(sigma, np.array([3.0, 5.0]))

    def test_filtered_upper_limit_sigma_accepts_upper_limit_only_array(self):
        trans = DummyOpticalTransient(self.outdir)
        trans.x = np.arange(4.0)
        trans.x_err = None
        trans.y = np.array([20.0, 21.0, 22.0, 23.0])
        trans.y_err = np.ones(4) * 0.1
        trans._bands = np.array(["g", "r", "g", "r"])
        trans._active_bands = ["g"]
        trans._detections = np.array([True, False, False, True])
        trans._upper_limit_sigma = np.array([10.0, 20.0])

        sigma = _get_filtered_upper_limit_sigma(trans)

        np.testing.assert_array_equal(sigma, np.array([20.0]))

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_no_model_kwargs_warning_flux_density(self, mock_run_sampler, mock_read):
        """fit_model warns (not raises) when model_kwargs is None for flux_density mode."""
        trans = DummyAfterglow(self.outdir)
        trans.data_mode = "flux_density"
        mock_run_sampler.return_value = self.dummy_result
        # Should not raise even without model_kwargs
        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir,
            label="TestNoKwargs", sampler="dynesty", nlive=10, prior=self.prior,
            walks=10, model_kwargs=None, plot=False)
        self.assertEqual(result, self.dummy_result)

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_model_returns_cached_result(self, mock_run_sampler, mock_read_result):
        """fit_model returns the cached result when read_in_result succeeds."""
        mock_read_result.side_effect = None
        mock_read_result.return_value = self.dummy_result
        trans = DummyAfterglow(self.outdir)
        model_kwargs = {"output_format": "flux_density",
                        "frequency": np.linspace(1e14, 1e15, len(trans.x))}
        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir,
            label="CachedLabel", sampler="dynesty", nlive=10, prior=self.prior,
            walks=10, model_kwargs=model_kwargs, plot=False)
        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_not_called()

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_model_clean_skips_cache(self, mock_run_sampler, mock_read_result):
        """clean=True forces re-run even when a cached result exists."""
        mock_read_result.return_value = self.dummy_result
        mock_run_sampler.return_value = self.dummy_result
        trans = DummyAfterglow(self.outdir)
        model_kwargs = {"output_format": "flux_density",
                        "frequency": np.linspace(1e14, 1e15, len(trans.x))}
        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir,
            label="CleanLabel", sampler="dynesty", nlive=10, prior=self.prior,
            walks=10, model_kwargs=model_kwargs, plot=False, clean=True)
        mock_run_sampler.assert_called_once()
        self.assertEqual(result, self.dummy_result)

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    def test_fit_model_prior_none_loads_from_non_default_priors(self, mock_run_sampler, mock_read_result):
        """fit_model with prior=None must resolve priors from non_default_priors/ without raising."""
        mock_run_sampler.return_value = self.dummy_result
        trans = DummyAfterglow(self.outdir)
        model_kwargs = {"output_format": "flux_density",
                        "frequency": np.linspace(1e14, 1e15, len(trans.x))}
        # vegas_tophat lives only in non_default_priors/ — previously caused FileNotFoundError
        import redback.priors as rp
        with patch.object(rp, "get_priors", wraps=rp.get_priors) as mock_get_priors:
            result = fit_model(
                transient=trans, model="vegas_tophat", outdir=self.outdir,
                label="NonDefaultPrior", sampler="dynesty", nlive=10, prior=None,
                walks=10, model_kwargs=model_kwargs, plot=False)
        mock_get_priors.assert_called_once_with("vegas_tophat")
        self.assertEqual(result, self.dummy_result)

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    @patch("redback.sampler.run_point_estimation", autospec=True)
    def test_fit_model_routes_map_to_point_estimation(
            self, mock_point_estimation, mock_run_sampler, mock_read_result):
        mock_point_estimation.return_value = self.dummy_result
        trans = DummyOpticalTransient(self.outdir)
        model_kwargs = {
            "output_format": "flux_density",
            "frequency": np.linspace(1e14, 1e15, len(trans.x)),
        }

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir,
            label="PointFit", prior=self.prior, model_kwargs=model_kwargs,
            fit_method="map", plot=False)

        self.assertEqual(result, self.dummy_result)
        mock_run_sampler.assert_not_called()
        self.assertEqual(mock_point_estimation.call_args.kwargs["fit_method"], "map")
        self.assertEqual(mock_point_estimation.call_args.kwargs["label"], "PointFit_map")

    def test_fit_model_rejects_sampler_with_point_estimation(self):
        trans = DummyOpticalTransient(self.outdir)
        with self.assertRaisesRegex(ValueError, "sampler cannot be combined"):
            fit_model(
                transient=trans, model="dummy_model", prior=self.prior,
                sampler="laplace", fit_method="mle")

    @patch("bilby.core.sampler.get_implemented_samplers", return_value=[])
    def test_laplace_sampler_has_actionable_missing_plugin_error(self, mock_samplers):
        trans = DummyOpticalTransient(self.outdir)
        with self.assertRaisesRegex(ImportError, "GregoryAshton/bilby-laplace"):
            fit_model(
                transient=trans, model="dummy_model", prior=self.prior,
                sampler="laplace")

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("bilby.run_sampler", autospec=True)
    @patch("bilby.core.sampler.get_implemented_samplers", return_value=["laplace"])
    def test_laplace_sampler_uses_existing_sampling_path(
            self, mock_samplers, mock_run_sampler, mock_read_result):
        mock_run_sampler.return_value = self.dummy_result
        trans = DummyOpticalTransient(self.outdir)
        model_kwargs = {
            "output_format": "flux_density",
            "frequency": np.linspace(1e14, 1e15, len(trans.x)),
        }

        result = fit_model(
            transient=trans, model="dummy_model", outdir=self.outdir,
            label="LaplaceFit", sampler="laplace", prior=self.prior,
            model_kwargs=model_kwargs, plot=False)

        self.assertEqual(result, self.dummy_result)
        self.assertEqual(mock_run_sampler.call_args.kwargs["sampler"], "laplace")

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    def test_fit_model_mle_end_to_end(self, mock_read_result):
        trans = DummyOpticalTransient(self.outdir)
        model_kwargs = {
            "output_format": "flux_density",
            "frequency": np.linspace(1e14, 1e15, len(trans.x)),
        }
        prior = bilby.prior.PriorDict({
            "amplitude": bilby.prior.Uniform(0, 30),
        })

        result = fit_model(
            transient=trans, model=constant_model, outdir=self.outdir,
            label="ConstantFit", prior=prior, model_kwargs=model_kwargs,
            fit_method="mle", plot=False,
            optimizer_kwargs={
                "seed": 12, "maxiter": 30, "popsize": 6,
                "local_options": {"xtol": 1e-8, "ftol": 1e-8},
            })

        self.assertEqual(result.label, "ConstantFit_mle")
        self.assertAlmostEqual(result.point_estimate["amplitude"], 15.0, places=5)

    @patch("redback.result.read_in_result", side_effect=Exception("No result"))
    @patch("redback.sampler.run_point_estimation", autospec=True)
    def test_map_routes_all_lightcurve_and_spectrum_types(
            self, mock_point_estimation, mock_read_result):
        mock_point_estimation.return_value = self.dummy_result
        transients = [
            DummySpectrum(self.outdir),
            DummyAfterglow(self.outdir),
            DummyPromptTimeSeries(self.outdir),
            DummyOpticalTransient(self.outdir),
            DummyTransient(self.outdir),
        ]

        for index, trans in enumerate(transients):
            size = len(trans.angstroms) if isinstance(trans, Spectrum) else len(trans.x)
            model_kwargs = {
                "output_format": "flux_density",
                "frequency": np.linspace(1e14, 1e15, size),
            }
            with self.subTest(transient=type(trans).__name__):
                fit_model(
                    transient=trans, model="dummy_model", outdir=self.outdir,
                    label=f"Route{index}", prior=self.prior,
                    model_kwargs=model_kwargs, fit_method="map", plot=False)

        self.assertEqual(mock_point_estimation.call_count, len(transients))
        self.assertTrue(all(
            call.kwargs["fit_method"] == "map"
            for call in mock_point_estimation.call_args_list
        ))


class TestFitSpectralDatasetBranches(unittest.TestCase):
    """Tests for _fit_spectral_dataset statistic selection and validation."""

    def setUp(self):
        from redback.spectral.dataset import SpectralDataset
        import numpy as np
        self.n = 5
        edges = np.linspace(0.5, 3.0, self.n + 1)
        self.dataset = SpectralDataset(
            counts=np.full(self.n, 20.0),
            exposure=1000.0,
            energy_edges_keV=edges,
        )
        self.dataset_with_bkg = SpectralDataset(
            counts=np.full(self.n, 20.0),
            exposure=1000.0,
            energy_edges_keV=edges,
            counts_bkg=np.full(self.n, 5.0),
            bkg_exposure=2000.0,
            bkg_backscale=1.0,
            bkg_areascal=1.0,
        )

    def test_invalid_model_signature_raises(self):
        """Model without energies_keV/energy_keV and no **kwargs raises ValueError."""
        from redback.sampler import _fit_spectral_dataset
        import bilby

        def bad_model(time, amplitude):
            return np.ones_like(time) * amplitude

        with self.assertRaises(ValueError):
            _fit_spectral_dataset(
                transient=self.dataset, model=bad_model,
                outdir=self.setUp.__module__, label="test",
                prior=None, plot=False)

    def test_unknown_statistic_raises(self):
        """Passing an unknown statistic string raises ValueError."""
        from redback.sampler import _fit_spectral_dataset

        def good_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        with self.assertRaises(ValueError):
            _fit_spectral_dataset(
                transient=self.dataset, model=good_model,
                outdir="/tmp", label="test",
                prior=None, plot=False, statistic="bogus_stat")

    def test_auto_statistic_selects_cstat_without_background(self):
        """statistic=auto selects cstat when no background is present."""
        from redback.sampler import _fit_spectral_dataset
        from unittest.mock import patch, MagicMock

        def good_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        dummy_result = MagicMock()
        with patch("bilby.run_sampler", return_value=dummy_result), \
             patch("redback.result.read_in_result", side_effect=Exception):
            result = _fit_spectral_dataset(
                transient=self.dataset, model=good_model,
                outdir="/tmp", label="cstat_test",
                prior=None, plot=False, statistic="auto")
        self.assertEqual(result, dummy_result)

    def test_auto_statistic_selects_wstat_with_background(self):
        """statistic=auto selects wstat when background is present."""
        from redback.sampler import _fit_spectral_dataset
        from unittest.mock import patch, MagicMock

        def good_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        dummy_result = MagicMock()
        with patch("bilby.run_sampler", return_value=dummy_result), \
             patch("redback.result.read_in_result", side_effect=Exception):
            result = _fit_spectral_dataset(
                transient=self.dataset_with_bkg, model=good_model,
                outdir="/tmp", label="wstat_test",
                prior=None, plot=False, statistic="auto")
        self.assertEqual(result, dummy_result)

    def test_explicit_chi2_statistic(self):
        """statistic='chi2' selects ChiSquareSpectralLikelihood."""
        from redback.sampler import _fit_spectral_dataset
        from unittest.mock import patch, MagicMock

        def good_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        dummy_result = MagicMock()
        with patch("bilby.run_sampler", return_value=dummy_result), \
             patch("redback.result.read_in_result", side_effect=Exception):
            result = _fit_spectral_dataset(
                transient=self.dataset, model=good_model,
                outdir="/tmp", label="chi2_test",
                prior=None, plot=False, statistic="chi2")
        self.assertEqual(result, dummy_result)

    def test_json_save_format_converted_to_pkl(self):
        """JSON save format is silently converted to pkl for spectral datasets."""
        from redback.sampler import _fit_spectral_dataset
        from unittest.mock import patch, MagicMock

        def good_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        dummy_result = MagicMock()
        captured = {}
        original_run = __import__("bilby").run_sampler

        def capture_run(**kw):
            captured["save"] = kw.get("save")
            return dummy_result

        with patch("bilby.run_sampler", side_effect=capture_run), \
             patch("redback.result.read_in_result", side_effect=Exception):
            _fit_spectral_dataset(
                transient=self.dataset, model=good_model,
                outdir="/tmp", label="pkl_test",
                prior=None, plot=False, save_format="json")
        self.assertEqual(captured.get("save"), "pkl")

    def test_map_uses_point_estimation_for_spectral_dataset(self):
        from redback.sampler import _fit_spectral_dataset
        from unittest.mock import patch, MagicMock

        def good_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        prior = bilby.prior.PriorDict({
            "amplitude": bilby.prior.Uniform(0.1, 10),
        })
        dummy_result = MagicMock()
        with patch("redback.sampler.run_point_estimation", return_value=dummy_result) as point, \
             patch("redback.result.read_in_result", side_effect=Exception):
            result = _fit_spectral_dataset(
                transient=self.dataset, model=good_model,
                outdir="/tmp", label="spectral_map", prior=prior,
                plot=False, fit_method="map", save_format="json")

        self.assertEqual(result, dummy_result)
        self.assertEqual(point.call_args.kwargs["fit_method"], "map")
        self.assertEqual(point.call_args.kwargs["save_format"], "pkl")


class TestSamplerValidationAndFallbacks(unittest.TestCase):

    @patch("redback.sampler._run_fit_backend")
    def test_spectrum_custom_likelihood_and_plot(self, backend):
        from redback.sampler import _fit_spectrum

        result = MagicMock()
        backend.return_value = result
        transient = DummySpectrum(".")
        likelihood = MagicMock()
        _fit_spectrum(
            transient, dummy_model, ".", "spectrum", likelihood=likelihood,
            prior={}, model_kwargs=None, clean=True, plot=True)
        self.assertIs(backend.call_args.kwargs["likelihood"], likelihood)
        result.plot_spectrum.assert_called_once_with(model=dummy_model)

    @patch("redback.sampler._run_fit_backend")
    def test_grb_photon_prior_custom_likelihood_and_plot(self, backend):
        from redback.sampler import _fit_grb

        result = MagicMock()
        backend.return_value = result
        likelihood = MagicMock()
        prior = {}
        transient = SimpleNamespace(
            photon_index=-1.0, name="GRB", flux_data=False,
            magnitude_data=False, flux_density_data=False,
            x=np.arange(2.0), x_err=None, y=np.ones(2), y_err=np.ones(2))
        _fit_grb(
            transient, dummy_model, ".", "grb", likelihood=likelihood,
            prior=prior, model_kwargs=None, clean=True, plot=True,
            use_photon_index_prior=True)
        self.assertIsInstance(prior["alpha_1"], bilby.prior.Uniform)
        self.assertIs(backend.call_args.kwargs["likelihood"], likelihood)
        result.plot_lightcurve.assert_called_once_with(model=dummy_model)

    @patch("redback.sampler._run_fit_backend")
    def test_grb_positive_photon_index_uses_gaussian_prior(self, backend):
        from redback.sampler import _fit_grb

        backend.return_value = MagicMock()
        prior = {}
        transient = DummyAfterglow(".")
        _fit_grb(
            transient, dummy_model, ".", "grb", prior=prior,
            model_kwargs={"output_format": "flux_density", "frequency": 1e14},
            clean=True, plot=False,
            use_photon_index_prior=True)
        self.assertIsInstance(prior["alpha_1"], bilby.prior.Gaussian)

    @patch("redback.sampler._run_fit_backend")
    def test_optical_non_photometry_custom_likelihood_and_plot(self, backend):
        from redback.sampler import _fit_optical_transient

        result = MagicMock()
        backend.return_value = result
        transient = SimpleNamespace(
            name="generic", flux_data=False, magnitude_data=False,
            flux_density_data=False, has_upper_limits=False,
            x=np.arange(2.0), x_err=None, y=np.ones(2), y_err=np.ones(2))
        likelihood = MagicMock()
        _fit_optical_transient(
            transient, dummy_model, ".", "generic", likelihood=likelihood,
            prior={}, model_kwargs=None, clean=True, plot=True)
        self.assertIs(backend.call_args.kwargs["likelihood"], likelihood)
        result.plot_lightcurve.assert_called_once_with(model=dummy_model)

    @patch("redback.result.read_in_result")
    def test_prompt_cached_result_is_plotted(self, read_result):
        from redback.sampler import _fit_prompt

        result = MagicMock()
        read_result.return_value = result
        returned = _fit_prompt(
            DummyPromptTimeSeries("."), dummy_model, ".", "prompt",
            prior={}, model_kwargs={"output_format": "counts", "frequency": 1e14},
            plot=True)
        self.assertIs(returned, result)

    @patch("redback.sampler._run_fit_backend")
    @patch("inspect.signature", side_effect=ValueError("uninspectable"))
    def test_spectral_uninspectable_model_accepts_custom_likelihood(
            self, signature, backend):
        from redback.sampler import _fit_spectral_dataset
        from redback.spectral.dataset import SpectralDataset

        dataset = SpectralDataset(
            counts=np.ones(2), exposure=1.0,
            energy_edges_keV=np.array([1.0, 2.0, 3.0]))
        likelihood = MagicMock()
        backend.return_value = MagicMock()
        returned = _fit_spectral_dataset(
            dataset, dummy_model, ".", "spectral", likelihood=likelihood,
            prior=None, model_kwargs=None, save_format="pkl",
            clean=True, plot=False)
        self.assertIs(returned, backend.return_value)
        self.assertIs(backend.call_args.kwargs["likelihood"], likelihood)

    def test_fit_model_validates_method_and_photometry_kwargs(self):
        transient = SimpleNamespace(data_mode="magnitude")
        with self.assertRaisesRegex(ValueError, "fit_method"):
            fit_model(transient, dummy_model, fit_method="invalid", prior={})
        with self.assertRaisesRegex(ValueError, "bands"):
            fit_model(
                transient, dummy_model, prior={},
                model_kwargs={"output_format": "magnitude", "bands": None})
        transient.data_mode = "flux_density"
        with self.assertRaisesRegex(ValueError, "frequency"):
            fit_model(
                transient, dummy_model, prior={},
                model_kwargs={"output_format": "flux_density", "frequency": None})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            fit_model(
                transient, dummy_model, prior={},
                model_kwargs={"output_format": "flux", "bands": ["g"]})

    def test_fit_model_requires_available_default_prior(self):
        transient = SimpleNamespace(data_mode="counts")
        with patch("redback.priors.get_priors", return_value=None):
            with self.assertRaisesRegex(ValueError, "No prior found"):
                fit_model(transient, dummy_model, prior=None)

    def test_fit_model_rejects_unknown_transient_type(self):
        transient = SimpleNamespace(
            data_mode="counts", name="unknown",
            directory_structure=SimpleNamespace(directory_path="unused"))
        with tempfile.TemporaryDirectory() as outdir:
            with self.assertRaisesRegex(ValueError, "Source type"):
                fit_model(
                    transient, dummy_model, outdir=outdir, prior={"amplitude": 1.0},
                    plot=False)

    @patch("redback.sampler.bilby.run_sampler")
    def test_pymultinest_nested_sample_failure_falls_back_to_dynesty(self, run_sampler):
        from redback.sampler import _run_fit_backend

        fallback_result = MagicMock()
        run_sampler.side_effect = [
            ValueError("dead_points and live_points mismatch"), fallback_result]
        result = _run_fit_backend(
            likelihood=MagicMock(), prior={}, label="fallback", sampler="pymultinest",
            nlive=10, outdir=".", walks=5, resume=False, save_format="json",
            meta_data={}, sampler_plot=False)
        self.assertIs(result, fallback_result)
        self.assertEqual(2, run_sampler.call_count)
        self.assertEqual("dynesty", run_sampler.call_args_list[1].kwargs["sampler"])

    @patch("redback.sampler.bilby.run_sampler", side_effect=ValueError("other failure"))
    def test_sampling_backend_propagates_unrelated_value_error(self, run_sampler):
        from redback.sampler import _run_fit_backend

        with self.assertRaisesRegex(ValueError, "other failure"):
            _run_fit_backend(
                likelihood=MagicMock(), prior={}, label="failure", sampler="dynesty",
                nlive=10, outdir=".", walks=5, resume=False, save_format="json",
                meta_data={}, sampler_plot=False)

    def test_filtered_upper_limit_sigma_supports_all_storage_shapes(self):
        transient = SimpleNamespace(
            upper_limit_sigma=5.0, filtered_indices=np.array([True, False, True]),
            x=np.arange(3), detections=None)
        self.assertEqual(5.0, _get_filtered_upper_limit_sigma(transient))

        transient.upper_limit_sigma = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(_get_filtered_upper_limit_sigma(transient), [2.0, 4.0])

        transient.detections = np.array([True, False, False])
        transient.upper_limits = ~transient.detections
        transient.filtered_indices = np.array([False, True, True])
        transient.upper_limit_sigma = np.array([3.0, 4.0])
        np.testing.assert_array_equal(_get_filtered_upper_limit_sigma(transient), [3.0, 4.0])

        transient.upper_limit_sigma = np.array([1.0])
        np.testing.assert_array_equal(_get_filtered_upper_limit_sigma(transient), [1.0])

    @patch("redback.sampler._run_fit_backend")
    @patch("redback.sampler.GaussianLikelihood")
    def test_optical_fit_drops_nonfinite_upper_limits(self, gaussian, backend):
        from redback.sampler import _fit_optical_transient

        backend.return_value = MagicMock()
        transient = MagicMock(
            flux_data=True, magnitude_data=False, flux_density_data=False,
            has_upper_limits=True, name="test")
        transient.get_filtered_data_with_limits.return_value = (
            np.array([1.0, 2.0]), None, np.array([3.0, np.nan]),
            np.array([0.1, np.nan]), np.array([True, False]))
        model = MagicMock(__name__="model")
        _fit_optical_transient(
            transient, model, ".", "test", prior={},
            model_kwargs={"output_format": "flux_density", "frequency": np.array([1e14, 1e14])},
            plot=False, clean=True)
        np.testing.assert_array_equal(gaussian.call_args.kwargs["x"], [1.0])
        np.testing.assert_array_equal(gaussian.call_args.kwargs["y"], [3.0])

    @patch("redback.sampler._run_fit_backend")
    @patch("redback.sampler.GaussianLikelihoodWithUpperLimits")
    def test_optical_fit_uses_upper_limit_likelihood_for_finite_limits(self, upper_likelihood, backend):
        from redback.sampler import _fit_optical_transient

        backend.return_value = MagicMock()
        transient = MagicMock(
            flux_data=False, magnitude_data=True, flux_density_data=False,
            has_upper_limits=True, name="test", upper_limit_sigma=5.0)
        transient.get_filtered_data_with_limits.return_value = (
            np.array([1.0, 2.0]), None, np.array([20.0, 22.0]),
            np.array([0.1, np.nan]), np.array([True, False]))
        model = MagicMock(__name__="model")
        _fit_optical_transient(
            transient, model, ".", "test", prior={},
            model_kwargs={"output_format": "magnitude", "bands": np.array(["g", "g"])},
            plot=False, clean=True)
        self.assertEqual("magnitude", upper_likelihood.call_args.kwargs["data_mode"])
        self.assertEqual(5.0, upper_likelihood.call_args.kwargs["upper_limit_sigma"])


if __name__ == '__main__':
    unittest.main()
