import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
import bilby.core.prior

from redback import result
from redback.result import RedbackResult, read_in_result


class TestRedbackResult(unittest.TestCase):
    """Test RedbackResult class"""

    def setUp(self):
        """Set up test fixtures"""
        self.tempdir = tempfile.mkdtemp()
        self.label = 'test_result'

        # Create minimal posterior DataFrame
        self.posterior = pd.DataFrame({
            'param1': np.random.randn(100),
            'param2': np.random.randn(100),
            'log_likelihood': np.random.randn(100),
            'log_prior': np.random.randn(100)
        })

        # Create metadata
        self.meta_data = {
            'name': 'test_transient',
            'data_mode': 'flux',
            'time': np.array([1.0, 2.0, 3.0]),
            'flux': np.array([1e-12, 2e-12, 1.5e-12]),
            'flux_err': np.array([1e-13, 2e-13, 1.5e-13]),
            'model': 'test_model',
            'transient_type': 'afterglow',
            'model_kwargs': {}
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_initialization(self):
        """Test basic RedbackResult initialization"""
        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        self.assertEqual(res.label, self.label)
        self.assertEqual(res.outdir, self.tempdir)
        self.assertIsNotNone(res.posterior)

    def test_meta_data_accessors(self):
        """Test metadata accessor properties"""
        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        self.assertEqual(res.model, 'test_model')
        self.assertEqual(res.transient_type, 'afterglow')
        self.assertEqual(res.name, 'test_transient')
        self.assertIsInstance(res.model_kwargs, dict)

    @patch('redback.result.TRANSIENT_DICT')
    def test_transient_property(self, mock_transient_dict):
        """Test transient property reconstruction"""
        mock_transient_class = MagicMock()
        mock_transient_instance = MagicMock()
        mock_transient_class.return_value = mock_transient_instance
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        transient = res.transient
        mock_transient_dict.__getitem__.assert_called_once_with('afterglow')
        mock_transient_class.assert_called_once()
        self.assertEqual(transient, mock_transient_instance)

    @patch('redback.result.TRANSIENT_DICT')
    @patch('redback.model_library.all_models_dict')
    def test_plot_lightcurve_with_model(self, mock_models, mock_transient_dict):
        """Test plot_lightcurve method with provided model"""
        mock_transient = MagicMock()
        mock_transient.plot_lightcurve.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        mock_model = MagicMock()
        mock_models.__getitem__.return_value = mock_model

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        # Test with explicit model
        result_plot = res.plot_lightcurve(model=mock_model)
        mock_transient.plot_lightcurve.assert_called_once()
        self.assertIsNotNone(result_plot)

    @patch('redback.result.TRANSIENT_DICT')
    @patch('redback.model_library.all_models_dict')
    def test_plot_lightcurve_default_model(self, mock_models, mock_transient_dict):
        """Test plot_lightcurve with default model from metadata"""
        mock_transient = MagicMock()
        mock_transient.plot_lightcurve.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        mock_model = MagicMock()
        mock_models.__getitem__.return_value = mock_model

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        # Test with default model (None)
        result_plot = res.plot_lightcurve()
        mock_models.__getitem__.assert_called_once_with('test_model')
        self.assertIsNotNone(result_plot)

    @patch('redback.result.TRANSIENT_DICT')
    @patch('redback.model_library.all_models_dict')
    def test_plot_spectrum(self, mock_models, mock_transient_dict):
        """Test plot_spectrum method"""
        mock_transient = MagicMock()
        mock_transient.plot_spectrum.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        mock_model = MagicMock()
        mock_models.__getitem__.return_value = mock_model

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        result_plot = res.plot_spectrum()
        mock_transient.plot_spectrum.assert_called_once()
        self.assertIsNotNone(result_plot)

    @patch('redback.result.TRANSIENT_DICT')
    @patch('redback.model_library.all_models_dict')
    def test_plot_residual(self, mock_models, mock_transient_dict):
        """Test plot_residual method"""
        mock_transient = MagicMock()
        mock_transient.plot_residual.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        mock_model = MagicMock()
        mock_models.__getitem__.return_value = mock_model

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        result_plot = res.plot_residual()
        mock_transient.plot_residual.assert_called_once()
        self.assertIsNotNone(result_plot)

    @patch('redback.result.TRANSIENT_DICT')
    @patch('redback.model_library.all_models_dict')
    def test_plot_multiband_lightcurve(self, mock_models, mock_transient_dict):
        """Test plot_multiband_lightcurve method"""
        mock_transient = MagicMock()
        mock_transient.plot_multiband_lightcurve.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        mock_model = MagicMock()
        mock_models.__getitem__.return_value = mock_model

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        result_plot = res.plot_multiband_lightcurve()
        mock_transient.plot_multiband_lightcurve.assert_called_once()
        self.assertIsNotNone(result_plot)

    @patch('redback.result.TRANSIENT_DICT')
    def test_plot_data(self, mock_transient_dict):
        """Test plot_data method"""
        mock_transient = MagicMock()
        mock_transient.plot_data.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        result_plot = res.plot_data()
        mock_transient.plot_data.assert_called_once()
        self.assertIsNotNone(result_plot)

    @patch('redback.result.TRANSIENT_DICT')
    def test_plot_multiband(self, mock_transient_dict):
        """Test plot_multiband method"""
        mock_transient = MagicMock()
        mock_transient.plot_multiband.return_value = MagicMock()
        mock_transient_class = MagicMock(return_value=mock_transient)
        mock_transient_dict.__getitem__.return_value = mock_transient_class

        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            posterior=self.posterior,
            meta_data=self.meta_data
        )

        result_plot = res.plot_multiband()
        mock_transient.plot_multiband.assert_called_once()
        self.assertIsNotNone(result_plot)

    def test_with_sampling_results(self):
        """Test with sampling-specific parameters"""
        res = RedbackResult(
            label=self.label,
            outdir=self.tempdir,
            sampler='dynesty',
            posterior=self.posterior,
            meta_data=self.meta_data,
            log_evidence=100.5,
            log_evidence_err=0.1,
            sampling_time=123.45,
            num_likelihood_evaluations=10000
        )

        self.assertEqual(res.sampler, 'dynesty')
        self.assertAlmostEqual(res.log_evidence, 100.5)
        self.assertAlmostEqual(res.log_evidence_err, 0.1)
        # sampling_time gets converted to timedelta by bilby
        self.assertIsNotNone(res.sampling_time)
        self.assertEqual(res.num_likelihood_evaluations, 10000)


class TestReadInResult(unittest.TestCase):
    """Test read_in_result function"""

    def setUp(self):
        """Set up test fixtures"""
        self.tempdir = tempfile.mkdtemp()
        self.label = 'test_read'

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    @patch('redback.result.RedbackResult.from_json')
    @patch('redback.result._determine_file_name')
    def test_read_json_result(self, mock_determine_filename, mock_from_json):
        """Test reading JSON result"""
        test_filename = os.path.join(self.tempdir, 'test_result.json')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_json.return_value = mock_result

        result = read_in_result(filename=test_filename)

        mock_from_json.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result.RedbackResult.from_hdf5')
    @patch('redback.result._determine_file_name')
    def test_read_hdf5_result(self, mock_determine_filename, mock_from_hdf5):
        """Test reading HDF5 result"""
        test_filename = os.path.join(self.tempdir, 'test_result.hdf5')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_hdf5.return_value = mock_result

        result = read_in_result(filename=test_filename)

        mock_from_hdf5.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result.RedbackResult.from_hdf5')
    @patch('redback.result._determine_file_name')
    def test_read_h5_result(self, mock_determine_filename, mock_from_hdf5):
        """Test reading .h5 result"""
        test_filename = os.path.join(self.tempdir, 'test_result.h5')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_hdf5.return_value = mock_result

        result = read_in_result(filename=test_filename)

        mock_from_hdf5.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result.RedbackResult.from_pickle')
    @patch('redback.result._determine_file_name')
    def test_read_pickle_result(self, mock_determine_filename, mock_from_pickle):
        """Test reading pickle result"""
        test_filename = os.path.join(self.tempdir, 'test_result.pkl')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_pickle.return_value = mock_result

        result = read_in_result(filename=test_filename)

        mock_from_pickle.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result.RedbackResult.from_pickle')
    @patch('redback.result._determine_file_name')
    def test_read_pickle_alt_extension(self, mock_determine_filename, mock_from_pickle):
        """Test reading .pickle result"""
        test_filename = os.path.join(self.tempdir, 'test_result.pickle')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_pickle.return_value = mock_result

        result = read_in_result(filename=test_filename)

        mock_from_pickle.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result.RedbackResult.from_json')
    @patch('redback.result._determine_file_name')
    def test_read_gzipped_json(self, mock_determine_filename, mock_from_json):
        """Test reading gzipped JSON result"""
        test_filename = os.path.join(self.tempdir, 'test_result.json.gz')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_json.return_value = mock_result

        result = read_in_result(filename=test_filename)

        mock_from_json.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result._determine_file_name')
    def test_read_invalid_extension(self, mock_determine_filename):
        """Test that invalid extension raises ValueError"""
        test_filename = os.path.join(self.tempdir, 'test_result.invalid')
        mock_determine_filename.return_value = test_filename

        with self.assertRaises(ValueError) as context:
            read_in_result(filename=test_filename)

        self.assertIn('not understood', str(context.exception))

    @patch('redback.result.RedbackResult.from_json')
    @patch('redback.result._determine_file_name')
    def test_read_with_label_outdir(self, mock_determine_filename, mock_from_json):
        """Test reading with outdir and label instead of filename"""
        test_filename = os.path.join(self.tempdir, 'test_label_result.json')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_json.return_value = mock_result

        result = read_in_result(outdir=self.tempdir, label='test_label', extension='json')

        mock_determine_filename.assert_called_once_with(
            None, self.tempdir, 'test_label', 'json', False)
        mock_from_json.assert_called_once_with(filename=test_filename)
        self.assertEqual(result, mock_result)

    @patch('redback.result.RedbackResult.from_json')
    @patch('redback.result._determine_file_name')
    def test_read_with_gzip_flag(self, mock_determine_filename, mock_from_json):
        """Test reading with gzip flag"""
        test_filename = os.path.join(self.tempdir, 'test_result.json.gz')
        mock_determine_filename.return_value = test_filename
        mock_result = MagicMock(spec=RedbackResult)
        mock_from_json.return_value = mock_result

        result = read_in_result(
            outdir=self.tempdir, label='test_result', extension='json', gzip=True)

        mock_determine_filename.assert_called_once_with(
            None, self.tempdir, 'test_result', 'json', True)
        self.assertEqual(result, mock_result)


class TestRedbackResultEdgeCases(unittest.TestCase):
    """Test edge cases for RedbackResult"""

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_minimal_initialization(self):
        """Test with minimal parameters"""
        res = RedbackResult()
        self.assertEqual(res.label, 'no_label')
        # outdir defaults to current directory, not necessarily '.'
        self.assertIsNotNone(res.outdir)
        self.assertIsInstance(res.outdir, str)

    def test_with_priors(self):
        """Test initialization with priors"""
        priors = bilby.core.prior.PriorDict()
        priors['param1'] = bilby.core.prior.Uniform(0, 10)
        priors['param2'] = bilby.core.prior.Gaussian(5, 1)

        # Need minimal posterior for Result initialization
        posterior = pd.DataFrame({
            'param1': np.random.randn(10),
            'param2': np.random.randn(10)
        })

        res = RedbackResult(
            label='test',
            outdir=self.tempdir,
            priors=priors,
            posterior=posterior,
            search_parameter_keys=['param1', 'param2']  # Required for bilby Result
        )

        self.assertIsNotNone(res.priors)
        self.assertIn('param1', res.priors)
        self.assertIn('param2', res.priors)

    def test_with_nested_samples(self):
        """Test with nested sampling results"""
        nested_samples = pd.DataFrame({
            'param1': np.random.randn(1000),
            'param2': np.random.randn(1000),
            'log_likelihood': np.random.randn(1000)
        })

        posterior = pd.DataFrame({
            'param1': np.random.randn(100),
            'param2': np.random.randn(100)
        })

        res = RedbackResult(
            label='test',
            outdir=self.tempdir,
            nested_samples=nested_samples,
            posterior=posterior,
            sampler='dynesty'
        )

        self.assertIsNotNone(res.nested_samples)
        self.assertEqual(len(res.nested_samples), 1000)


if __name__ == '__main__':
    unittest.main()
