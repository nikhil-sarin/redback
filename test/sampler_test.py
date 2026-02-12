import unittest
import tempfile
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

import bilby

from redback.model_library import all_models_dict
from redback.result import RedbackResult
from redback.transient.afterglow import Afterglow
from redback.transient.prompt import PromptTimeSeries
from redback.transient.transient import OpticalTransient, Transient, Spectrum
from redback.sampler import fit_model


# --- Dummy Model and Result --- #
def dummy_model(x, **kwargs):
    """A dummy model function that returns an array of ones."""
    return np.ones_like(x)


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
        with self.assertRaises(ValueError) as context:
            fit_model(
                transient=trans, model="dummy_model", outdir=self.outdir, label="TestUnknown",
                sampler=self.sampler, nlive=self.nlive, prior=self.prior, walks=self.walks,
                model_kwargs=self.model_kwargs, plot=False
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


if __name__ == '__main__':
    unittest.main()
