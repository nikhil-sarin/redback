import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import redback.priors as priors
from bilby.core.prior import PriorDict


class TestGetGaussianPriors(unittest.TestCase):
    """Test get_gaussian_priors function"""

    def test_get_gaussian_priors_basic(self):
        """Test basic Gaussian priors generation"""
        times = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50, 60])
        yerr = np.array([1, 2, 3, 4, 5, 6])

        result = priors.get_gaussian_priors(times, y, yerr)

        # Check that it returns a PriorDict
        self.assertIsInstance(result, PriorDict)

        # Check that required parameters are present
        self.assertIn('amplitude', result)
        self.assertIn('sigma', result)
        self.assertIn('t_0', result)

    def test_get_gaussian_priors_parameter_ranges(self):
        """Test that parameter ranges are set correctly"""
        times = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50, 60])
        yerr = np.array([1, 2, 3, 4, 5, 6])

        result = priors.get_gaussian_priors(times, y, yerr)

        # Amplitude should be between min(yerr) and max(y)
        self.assertEqual(result['amplitude'].minimum, np.min(yerr))
        self.assertEqual(result['amplitude'].maximum, np.max(y))

        # t_0 should span the time range
        self.assertEqual(result['t_0'].minimum, times[0])
        self.assertEqual(result['t_0'].maximum, times[-1])


class TestGetSkewGaussianPriors(unittest.TestCase):
    """Test get_skew_gaussian_priors function"""

    def test_get_skew_gaussian_priors_basic(self):
        """Test basic skew Gaussian priors"""
        times = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50, 60])
        yerr = np.array([1, 2, 3, 4, 5, 6])

        result = priors.get_skew_gaussian_priors(times, y, yerr)

        # Check that it returns a PriorDict
        self.assertIsInstance(result, PriorDict)

        # Check that required parameters are present
        self.assertIn('amplitude', result)
        self.assertIn('sigma_rise', result)
        self.assertIn('sigma_fall', result)
        self.assertIn('t_0', result)

        # sigma should be removed (replaced by sigma_rise)
        self.assertNotIn('sigma', result)


class TestGetSkewExponentialPriors(unittest.TestCase):
    """Test get_skew_exponential_priors function"""

    def test_get_skew_exponential_priors_basic(self):
        """Test basic skew exponential priors"""
        times = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50, 60])
        yerr = np.array([1, 2, 3, 4, 5, 6])

        result = priors.get_skew_exponential_priors(times, y, yerr)

        # Check that it returns a PriorDict
        self.assertIsInstance(result, PriorDict)

        # Check that required parameters are present
        self.assertIn('amplitude', result)
        self.assertIn('tau_rise', result)
        self.assertIn('tau_fall', result)
        self.assertIn('t_0', result)

        # sigma should be removed (replaced by tau_rise)
        self.assertNotIn('sigma', result)


class TestGetFredPriors(unittest.TestCase):
    """Test get_fred_priors function"""

    def test_get_fred_priors_basic(self):
        """Test basic FRED priors"""
        times = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50, 60])
        yerr = np.array([1, 2, 3, 4, 5, 6])

        result = priors.get_fred_priors(times, y, yerr)

        # Check that it returns a PriorDict
        self.assertIsInstance(result, PriorDict)

        # Check that required parameters are present
        self.assertIn('amplitude', result)
        self.assertIn('tau', result)
        self.assertIn('psi', result)
        self.assertIn('delta', result)


class TestGetFredExtendedPriors(unittest.TestCase):
    """Test get_fred_extended_priors function"""

    def test_get_fred_extended_priors_basic(self):
        """Test basic extended FRED priors"""
        times = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50, 60])
        yerr = np.array([1, 2, 3, 4, 5, 6])

        result = priors.get_fred_extended_priors(times, y, yerr)

        self.assertIsInstance(result, PriorDict)
        self.assertIn('amplitude', result)
        self.assertIn('tau', result)
        self.assertIn('psi', result)
        self.assertIn('delta', result)
        self.assertIn('gamma', result)
        self.assertIn('nu', result)
        self.assertEqual(r"$\nu$", result['nu'].latex_label)


class TestGetPriors(unittest.TestCase):
    """Test get_priors main function"""

    @patch('redback.priors.redback.model_library.modules_dict')
    def test_get_priors_prompt_model_gaussian(self, mock_modules_dict):
        """Test get_priors with a prompt model (gaussian_prompt)"""
        # Mock the modules_dict to include gaussian_prompt
        mock_modules_dict.__getitem__.return_value = ['gaussian_prompt', 'skew_gaussian']

        times = np.array([0, 1, 2, 3, 4])
        y = np.array([10, 20, 30, 40, 50])
        yerr = np.array([1, 2, 3, 4, 5])
        dt = np.ones(len(times))

        result = priors.get_priors('gaussian_prompt', times=times, y=y, yerr=yerr, dt=dt)

        # Check that it returns a PriorDict
        self.assertIsInstance(result, PriorDict)

        # Should include background_rate for prompt models
        self.assertIn('background_rate', result)

    @patch('redback.priors.redback.model_library.modules_dict')
    def test_get_priors_prompt_model_with_defaults(self, mock_modules_dict):
        """Test get_priors with prompt model using default values"""
        mock_modules_dict.__getitem__.return_value = ['fred']

        # Call without providing times, y, yerr, dt
        result = priors.get_priors('fred')

        # Check that it returns a PriorDict
        self.assertIsInstance(result, PriorDict)

        # Should include background_rate
        self.assertIn('background_rate', result)

    @patch('redback.priors.redback.model_library.base_models_dict', {'base_model': 'something'})
    @patch('redback.priors.logger')
    def test_get_priors_base_model(self, mock_logger):
        """Test get_priors with a base model"""
        # Mock file reading to raise FileNotFoundError
        with patch.object(PriorDict, 'from_file', side_effect=FileNotFoundError):
            result = priors.get_priors('base_model')

            # Should return empty PriorDict
            self.assertIsInstance(result, PriorDict)
            self.assertEqual(len(result), 0)

            # Should log info messages
            mock_logger.info.assert_called()

    @patch('redback.priors.logger')
    def test_get_priors_file_not_found(self, mock_logger):
        """Test get_priors when prior file is not found"""
        # Mock file reading to raise FileNotFoundError
        with patch.object(PriorDict, 'from_file', side_effect=FileNotFoundError):
            result = priors.get_priors('nonexistent_model')

            # Should return empty PriorDict
            self.assertIsInstance(result, PriorDict)
            self.assertEqual(len(result), 0)

            # Should log warning
            mock_logger.warning.assert_called()

    def test_get_priors_loads_from_file(self):
        """Test that get_priors attempts to load from file"""
        # Mock to ensure from_file is called with correct path
        with patch.object(PriorDict, 'from_file') as mock_from_file:
            mock_from_file.return_value = None  # Simulate successful load

            result = priors.get_priors('test_model')

            # Check that from_file was called (even if it raised FileNotFoundError)
            self.assertTrue(mock_from_file.called)

    def test_get_priors_with_csm_constraints(self):
        """Test that get_priors can attach configured CSM constraints."""
        result = priors.get_priors('csm_nickel', constraint=True)

        self.assertIs(result.conversion_function, priors.redback.constraints.csm_constraints)
        for key in ['shock_time', 'photosphere_constraint_1', 'photosphere_constraint_2']:
            self.assertIn(key, result)

    def test_get_priors_with_nickel_mixing_constraint(self):
        result = priors.get_priors('nickelmixing', constraint=True)

        self.assertIs(
            result.conversion_function,
            priors.redback.constraints.nickel_mixing_constraints)
        self.assertIn('nickel_fraction_in_mixed_region', result)

    def test_get_priors_with_kilonova_ejecta_relation_constraints(self):
        """Test that get_priors can attach derived ejecta-relation constraints."""
        result = priors.get_priors('one_component_nsbh_ejecta_relation', constraint=True)

        self.assertIs(result.conversion_function, priors.redback.constraints.one_component_nsbh_ejecta_relation_constraints)
        for key in ['ejecta_mej_min', 'ejecta_mej_max', 'ejecta_vej_min', 'ejecta_vej_max']:
            self.assertIn(key, result)

    def test_constrained_kilonova_ejecta_relation_samples_barnes_kasen_domain(self):
        """Derived constrained KN relation priors should sample the Barnes-Kasen domain."""
        result = priors.get_priors('one_component_nsbh_ejecta_relation', constraint=True)
        sample = result.sample(5)
        converted = result.conversion_function(sample)

        for key in ['ejecta_mej_min', 'ejecta_mej_max', 'ejecta_vej_min', 'ejecta_vej_max']:
            self.assertTrue(np.all(converted[key] >= 0))
            self.assertTrue(np.all(converted[key] <= 1))

    def test_band_function_high_energy_prior_samples_physical_indices(self):
        """Band high-energy priors should keep the Band-function break physical."""
        result = priors.get_priors('band_function_high_energy')
        sample = result.sample(100)

        self.assertTrue(np.all(sample['alpha'] > -2))
        self.assertTrue(np.all(sample['beta'] < sample['alpha']))

    def test_get_priors_with_cooling_envelope_constraints(self):
        """Test that get_priors can attach cooling-envelope constraints."""
        result = priors.get_priors('gaussianrise_cooling_envelope', constraint=True)

        self.assertIs(result.conversion_function, priors.redback.constraints.cooling_envelope_constraints)
        for key in ['eta_min_ratio', 'beta_max_ratio', 'gaussian_stitching_tail']:
            self.assertIn(key, result)


class TestGetPromptPriors(unittest.TestCase):
    """Test get_prompt_priors function"""

    @patch('redback.priors.get_gaussian_priors')
    def test_get_prompt_priors_gaussian(self, mock_get_gaussian):
        """Test get_prompt_priors with gaussian model"""
        times = np.array([0, 1, 2, 3, 4])
        y = np.array([10, 20, 30, 40, 50])
        yerr = np.array([1, 2, 3, 4, 5])

        priors.get_prompt_priors('gaussian', times, y, yerr)

        # Should call get_gaussian_priors
        mock_get_gaussian.assert_called_once()

    def test_get_prompt_priors_other_model(self):
        """Test get_prompt_priors with non-gaussian model"""
        times = np.array([0, 1, 2, 3, 4])
        y = np.array([10, 20, 30, 40, 50])
        yerr = np.array([1, 2, 3, 4, 5])

        # Should not raise an error, just return None
        result = priors.get_prompt_priors('other_model', times, y, yerr)
        self.assertIsNone(result)
