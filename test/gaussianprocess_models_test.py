import unittest
import numpy as np
from redback.transient_models import gaussianprocess_models


class TestCalculateFluxWithLabels(unittest.TestCase):
    """Test the calculate_flux_with_labels function"""

    def test_calculate_flux_with_single_label(self):
        """Test with a single label"""
        time = np.linspace(0, 10, 50)
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = ['band1']

        kwargs = {
            'a_band1': 1.5,
            'b_band1': 0.5
        }

        result = gaussianprocess_models.calculate_flux_with_labels(
            time, t0, tau_rise, tau_fall, labels, **kwargs
        )

        # Check that result is a dictionary
        self.assertIsInstance(result, dict)

        # Check that the label is in the result
        self.assertIn('band1', result)

        # Check that the flux is an array of the right shape
        self.assertEqual(len(result['band1']), len(time))

        # Check that values are finite
        self.assertTrue(np.all(np.isfinite(result['band1'])))

    def test_calculate_flux_with_multiple_labels(self):
        """Test with multiple labels"""
        time = np.linspace(0, 10, 50)
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = ['r', 'g', 'i']

        kwargs = {
            'a_r': 1.0,
            'b_r': 0.1,
            'a_g': 1.2,
            'b_g': 0.2,
            'a_i': 0.8,
            'b_i': 0.15
        }

        result = gaussianprocess_models.calculate_flux_with_labels(
            time, t0, tau_rise, tau_fall, labels, **kwargs
        )

        # Check that all labels are in the result
        for label in labels:
            self.assertIn(label, result)
            self.assertEqual(len(result[label]), len(time))
            self.assertTrue(np.all(np.isfinite(result[label])))

        # Check that different labels give different flux values
        self.assertFalse(np.allclose(result['r'], result['g']))
        self.assertFalse(np.allclose(result['r'], result['i']))

    def test_missing_a_parameter(self):
        """Test that missing 'a' parameter raises ValueError"""
        time = np.linspace(0, 10, 50)
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = ['band1']

        # Missing a_band1
        kwargs = {
            'b_band1': 0.5
        }

        with self.assertRaises(ValueError) as context:
            gaussianprocess_models.calculate_flux_with_labels(
                time, t0, tau_rise, tau_fall, labels, **kwargs
            )

        self.assertIn("Missing parameters", str(context.exception))
        self.assertIn("band1", str(context.exception))

    def test_missing_b_parameter(self):
        """Test that missing 'b' parameter raises ValueError"""
        time = np.linspace(0, 10, 50)
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = ['band1']

        # Missing b_band1
        kwargs = {
            'a_band1': 1.5
        }

        with self.assertRaises(ValueError) as context:
            gaussianprocess_models.calculate_flux_with_labels(
                time, t0, tau_rise, tau_fall, labels, **kwargs
            )

        self.assertIn("Missing parameters", str(context.exception))
        self.assertIn("band1", str(context.exception))

    def test_missing_parameters_multiple_labels(self):
        """Test that missing parameters for one of multiple labels raises ValueError"""
        time = np.linspace(0, 10, 50)
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = ['band1', 'band2', 'band3']

        # Missing parameters for band2
        kwargs = {
            'a_band1': 1.0,
            'b_band1': 0.1,
            # Missing a_band2 and b_band2
            'a_band3': 0.8,
            'b_band3': 0.15
        }

        with self.assertRaises(ValueError) as context:
            gaussianprocess_models.calculate_flux_with_labels(
                time, t0, tau_rise, tau_fall, labels, **kwargs
            )

        self.assertIn("Missing parameters", str(context.exception))
        self.assertIn("band2", str(context.exception))

    def test_empty_labels_list(self):
        """Test with empty labels list"""
        time = np.linspace(0, 10, 50)
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = []

        kwargs = {}

        result = gaussianprocess_models.calculate_flux_with_labels(
            time, t0, tau_rise, tau_fall, labels, **kwargs
        )

        # Should return empty dictionary
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_flux_behavior_at_different_times(self):
        """Test that flux behaves as expected at different times"""
        time = np.array([0, 1, 5, 10, 20])
        t0 = 1.0
        tau_rise = 2.0
        tau_fall = 3.0
        labels = ['test']

        kwargs = {
            'a_test': 1.0,
            'b_test': 0.0
        }

        result = gaussianprocess_models.calculate_flux_with_labels(
            time, t0, tau_rise, tau_fall, labels, **kwargs
        )

        flux = result['test']

        # Flux should be non-negative for a typical Bazin function
        # (though it could be negative depending on parameters)
        self.assertEqual(len(flux), len(time))

        # All values should be finite
        self.assertTrue(np.all(np.isfinite(flux)))

    def test_different_parameter_values(self):
        """Test with various parameter combinations"""
        time = np.linspace(0, 20, 100)
        labels = ['test']

        # Test with different parameter combinations
        test_cases = [
            {'t0': 0.0, 'tau_rise': 1.0, 'tau_fall': 2.0, 'a_test': 1.0, 'b_test': 0.0},
            {'t0': 5.0, 'tau_rise': 3.0, 'tau_fall': 5.0, 'a_test': 2.0, 'b_test': 0.5},
            {'t0': 10.0, 'tau_rise': 0.5, 'tau_fall': 1.0, 'a_test': 0.5, 'b_test': -0.1},
        ]

        for params in test_cases:
            t0 = params['t0']
            tau_rise = params['tau_rise']
            tau_fall = params['tau_fall']
            kwargs = {'a_test': params['a_test'], 'b_test': params['b_test']}

            result = gaussianprocess_models.calculate_flux_with_labels(
                time, t0, tau_rise, tau_fall, labels, **kwargs
            )

            self.assertIn('test', result)
            self.assertEqual(len(result['test']), len(time))
            self.assertTrue(np.all(np.isfinite(result['test'])))
