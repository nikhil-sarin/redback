import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np

from redback import result


class TestGRBResult(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_plot_directory_structure(self):
        pass

    def test_save_to_file(self):
        pass

    def test_plot_corner(self):
        pass

    def test_plot_lightcurve(self):
        pass


class TestReadInGRBResult(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.test_dir)

    def test_read_in_result_file_not_found(self):
        """Test that read_in_result logs error for missing file."""
        with self.assertRaises(FileNotFoundError):
            result.read_in_result(filename="/nonexistent/file.json")

    def test_read_in_result_invalid_extension(self):
        """Test that read_in_result logs error for unknown file type."""
        # Create a dummy file with unknown extension
        test_file = os.path.join(self.test_dir, "test.xyz")
        with open(test_file, 'w') as f:
            f.write("dummy")
        with self.assertRaises(ValueError):
            result.read_in_result(filename=test_file)

    def test_read_in_result_no_extension(self):
        """Test that read_in_result handles file without extension."""
        test_file = os.path.join(self.test_dir, "test")
        with open(test_file, 'w') as f:
            f.write("dummy")
        # This should fail because we can't determine the type
        with self.assertRaises(ValueError):
            result.read_in_result(filename=test_file)

    @patch('redback.result.RedbackResult.from_json')
    def test_read_in_result_json_success(self, mock_from_json):
        """Test successful JSON file loading."""
        test_file = os.path.join(self.test_dir, "test_result.json")
        with open(test_file, 'w') as f:
            json.dump({}, f)

        mock_result = MagicMock()
        mock_result.label = "test"
        mock_result.model = "test_model"
        mock_from_json.return_value = mock_result

        loaded = result.read_in_result(filename=test_file)
        self.assertEqual(loaded.label, "test")
        mock_from_json.assert_called_once()

    @patch('redback.result.RedbackResult.from_hdf5')
    def test_read_in_result_hdf5_success(self, mock_from_hdf5):
        """Test successful HDF5 file loading."""
        test_file = os.path.join(self.test_dir, "test_result.hdf5")
        with open(test_file, 'w') as f:
            f.write("dummy")

        mock_result = MagicMock()
        mock_result.label = "test"
        mock_result.model = "test_model"
        mock_from_hdf5.return_value = mock_result

        loaded = result.read_in_result(filename=test_file)
        self.assertEqual(loaded.label, "test")

    @patch('redback.result.RedbackResult.from_pickle')
    def test_read_in_result_pickle_success(self, mock_from_pickle):
        """Test successful pickle file loading."""
        test_file = os.path.join(self.test_dir, "test_result.pkl")
        with open(test_file, 'w') as f:
            f.write("dummy")

        mock_result = MagicMock()
        mock_result.label = "test"
        mock_result.model = "test_model"
        mock_from_pickle.return_value = mock_result

        loaded = result.read_in_result(filename=test_file)
        self.assertEqual(loaded.label, "test")

    @patch('redback.result.RedbackResult.from_json')
    def test_read_in_result_gzip_json(self, mock_from_json):
        """Test gzipped JSON file loading."""
        test_file = os.path.join(self.test_dir, "test_result.json.gz")
        with open(test_file, 'w') as f:
            f.write("dummy")

        mock_result = MagicMock()
        mock_result.label = "test"
        mock_result.model = "test_model"
        mock_from_json.return_value = mock_result

        loaded = result.read_in_result(filename=test_file)
        self.assertEqual(loaded.label, "test")


class TestRedbackResultTransient(unittest.TestCase):

    def setUp(self):
        self.result = result.RedbackResult()
        self.result.meta_data = {
            'name': 'test_transient',
            'time': np.array([1, 2, 3]),
            'time_err': np.array([0.1, 0.1, 0.1]),
            'flux_density': np.array([1e-3, 2e-3, 3e-3]),
            'flux_density_err': np.array([1e-4, 2e-4, 3e-4]),
            'bands': np.array(['r', 'r', 'r']),
            'system': np.array(['AB', 'AB', 'AB']),
            'redshift': 0.01,
            'data_mode': 'flux_density'
        }

    @patch('redback.result.TRANSIENT_DICT')
    def test_transient_reconstruction_success(self, mock_dict):
        """Test successful transient reconstruction."""
        mock_transient = MagicMock()
        mock_dict.__getitem__ = MagicMock(return_value=MagicMock(return_value=mock_transient))
        self.result._transient_type = 'supernova'
        self.result._name = 'test'

        transient = self.result.transient
        self.assertIsNotNone(transient)

    @patch('redback.result.TRANSIENT_DICT')
    def test_transient_reconstruction_unknown_type(self, mock_dict):
        """Test transient reconstruction with unknown type logs error."""
        mock_dict.__getitem__ = MagicMock(side_effect=KeyError("unknown"))
        self.result._transient_type = 'unknown_type'

        with self.assertRaises(KeyError):
            _ = self.result.transient

    @patch('redback.result.TRANSIENT_DICT')
    def test_transient_reconstruction_failure(self, mock_dict):
        """Test transient reconstruction failure logs error."""
        mock_dict.__getitem__ = MagicMock(return_value=MagicMock(side_effect=Exception("Failed")))
        self.result._transient_type = 'supernova'

        with self.assertRaises(Exception):
            _ = self.result.transient


class TestRedbackResultPlotting(unittest.TestCase):

    def setUp(self):
        self.result = result.RedbackResult()
        self.result._model = 'test_model'
        self.result._posterior = pd.DataFrame({'a': [1, 2, 3]})
        self.result._model_kwargs = {}

    @patch('redback.result.model_library.all_models_dict')
    @patch.object(result.RedbackResult, 'transient', new_callable=lambda: property(lambda self: MagicMock()))
    def test_plot_lightcurve_uses_stored_model(self, mock_transient_prop, mock_models):
        """Test that plot_lightcurve uses stored model when none provided."""
        mock_model = MagicMock()
        mock_models.__getitem__ = MagicMock(return_value=mock_model)

        mock_transient = MagicMock()
        mock_transient.plot_lightcurve = MagicMock(return_value=None)

        with patch.object(result.RedbackResult, 'transient', mock_transient):
            self.result.plot_lightcurve()
            mock_transient.plot_lightcurve.assert_called_once()

    @patch('redback.result.model_library.all_models_dict')
    def test_plot_spectrum_uses_stored_model(self, mock_models):
        """Test that plot_spectrum uses stored model when none provided."""
        mock_model = MagicMock()
        mock_models.__getitem__ = MagicMock(return_value=mock_model)

        mock_transient = MagicMock()
        mock_transient.plot_spectrum = MagicMock(return_value=None)

        with patch.object(result.RedbackResult, 'transient', mock_transient):
            self.result.plot_spectrum()
            mock_transient.plot_spectrum.assert_called_once()

    @patch('redback.result.model_library.all_models_dict')
    def test_plot_residual_uses_stored_model(self, mock_models):
        """Test that plot_residual uses stored model when none provided."""
        mock_model = MagicMock()
        mock_models.__getitem__ = MagicMock(return_value=mock_model)

        mock_transient = MagicMock()
        mock_transient.plot_residual = MagicMock(return_value=None)

        with patch.object(result.RedbackResult, 'transient', mock_transient):
            self.result.plot_residual()
            mock_transient.plot_residual.assert_called_once()

    @patch('redback.result.model_library.all_models_dict')
    def test_plot_multiband_lightcurve_uses_stored_model(self, mock_models):
        """Test that plot_multiband_lightcurve uses stored model when none provided."""
        mock_model = MagicMock()
        mock_models.__getitem__ = MagicMock(return_value=mock_model)

        mock_transient = MagicMock()
        mock_transient.plot_multiband_lightcurve = MagicMock(return_value=None)

        with patch.object(result.RedbackResult, 'transient', mock_transient):
            self.result.plot_multiband_lightcurve()
            mock_transient.plot_multiband_lightcurve.assert_called_once()

    def test_plot_data_delegates_to_transient(self):
        """Test that plot_data delegates to transient."""
        mock_transient = MagicMock()
        mock_transient.plot_data = MagicMock(return_value=None)

        with patch.object(result.RedbackResult, 'transient', mock_transient):
            self.result.plot_data()
            mock_transient.plot_data.assert_called_once()

    def test_plot_multiband_delegates_to_transient(self):
        """Test that plot_multiband delegates to transient."""
        mock_transient = MagicMock()
        mock_transient.plot_multiband = MagicMock(return_value=None)

        with patch.object(result.RedbackResult, 'transient', mock_transient):
            self.result.plot_multiband()
            mock_transient.plot_multiband.assert_called_once()


class TestReadInResultEdgeCases(unittest.TestCase):

    def test_read_in_result_no_extension_logs_error(self):
        """Test that read_in_result with no extension logs error and raises ValueError."""
        with patch('redback.result._determine_file_name') as mock_determine:
            # Mock a filename with no extension
            mock_determine.return_value = 'test_file'
            with patch('redback.result.os.path.exists') as mock_exists:
                mock_exists.return_value = True

                with self.assertRaises(ValueError) as context:
                    result.read_in_result(filename='test_file')
                # Empty extension goes to unsupported filetype branch
                self.assertIn('Filetype', str(context.exception))

    def test_read_in_result_unsupported_extension_logs_error(self):
        """Test that read_in_result with unsupported extension logs error."""
        with patch('redback.result._determine_file_name') as mock_determine:
            mock_determine.return_value = 'test_file.xyz'
            with patch('redback.result.os.path.exists') as mock_exists:
                mock_exists.return_value = True

                with self.assertRaises(ValueError) as context:
                    result.read_in_result(filename='test_file.xyz')
                self.assertIn('Filetype', str(context.exception))


class TestPlotModels(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class TestCalculateBF(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass
