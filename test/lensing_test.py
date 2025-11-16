import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from collections import namedtuple
import bilby

import redback.model_library
from redback.transient_models import lensing_models


class TestLensingCoreFunctions(unittest.TestCase):
    """Test core lensing calculation functions"""

    def test_perform_lensing_two_images(self):
        """Test _perform_lensing with two images"""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Simple base model that returns time values as flux
        def base_model_func(t):
            return np.array(t)

        kwargs = {
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 1.0,  # Delay second image by 1 day
            'mu_2': 0.5   # Half magnification
        }

        result = lensing_models._perform_lensing(
            time=time,
            flux_density_or_spectra_function=base_model_func,
            nimages=2,
            **kwargs
        )

        # Expected: mu_1 * flux(t - dt_1) + mu_2 * flux(t - dt_2)
        # = 1.0 * [1,2,3,4,5] + 0.5 * [0,1,2,3,4]
        expected = np.array([1.0, 2.5, 4.0, 5.5, 7.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_perform_lensing_three_images(self):
        """Test _perform_lensing with three images"""
        time = np.array([10.0, 20.0, 30.0])

        def base_model_func(t):
            return np.array(t)

        kwargs = {
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 5.0,
            'mu_2': 0.8,
            'dt_3': 10.0,
            'mu_3': 0.6
        }

        result = lensing_models._perform_lensing(
            time=time,
            flux_density_or_spectra_function=base_model_func,
            nimages=3,
            **kwargs
        )

        # Expected: 1.0*[10,20,30] + 0.8*[5,15,25] + 0.6*[0,10,20]
        expected = np.array([14.0, 38.0, 62.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_perform_lensing_negative_magnification(self):
        """Test _perform_lensing with negative magnification (parity flip)"""
        time = np.array([1.0, 2.0, 3.0])

        def base_model_func(t):
            return np.array(t) * 10.0

        kwargs = {
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 0.0,  # Same time
            'mu_2': -0.5  # Negative magnification
        }

        result = lensing_models._perform_lensing(
            time=time,
            flux_density_or_spectra_function=base_model_func,
            nimages=2,
            **kwargs
        )

        # Expected: 1.0*[10,20,30] + (-0.5)*[10,20,30] = 0.5*[10,20,30]
        expected = np.array([5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_perform_lensing_single_image(self):
        """Test _perform_lensing with single image (no lensing)"""
        time = np.array([1.0, 2.0, 3.0])

        def base_model_func(t):
            return np.array(t) ** 2

        kwargs = {
            'dt_1': 0.0,
            'mu_1': 2.0  # Just magnification
        }

        result = lensing_models._perform_lensing(
            time=time,
            flux_density_or_spectra_function=base_model_func,
            nimages=1,
            **kwargs
        )

        expected = np.array([2.0, 8.0, 18.0])  # 2 * [1, 4, 9]
        np.testing.assert_array_almost_equal(result, expected)


class TestLensingModelLibrary(unittest.TestCase):
    """Test that lensing models are properly registered"""

    def test_lensing_models_in_all_models_dict(self):
        """Test that lensing wrapper functions are in all_models_dict"""
        expected_functions = [
            'lensing_with_function',
            'lensing_with_supernova_base_model',
            'lensing_with_kilonova_base_model',
            'lensing_with_tde_base_model',
            'lensing_with_shock_powered_base_model',
            'lensing_with_magnetar_driven_base_model',
            'lensing_with_stellar_interaction_base_model',
            'lensing_with_general_synchrotron_base_model',
            'lensing_with_afterglow_base_model'
        ]

        for func_name in expected_functions:
            self.assertIn(func_name, redback.model_library.all_models_dict,
                          f"{func_name} not found in all_models_dict")

    def test_lensing_models_in_base_models_dict(self):
        """Test that lensing wrapper functions are in base_models_dict"""
        expected_functions = [
            'lensing_with_function',
            'lensing_with_supernova_base_model',
            'lensing_with_kilonova_base_model',
        ]

        for func_name in expected_functions:
            self.assertIn(func_name, redback.model_library.base_models_dict,
                          f"{func_name} not found in base_models_dict")


class TestLensingGetCorrectFunction(unittest.TestCase):
    """Test the _get_correct_function helper"""

    def test_get_correct_function_supernova(self):
        """Test getting correct function for supernova model"""
        function = lensing_models._get_correct_function('arnett', 'supernova')
        self.assertTrue(callable(function))

    def test_get_correct_function_kilonova(self):
        """Test getting correct function for kilonova model"""
        function = lensing_models._get_correct_function('one_component_kilonova_model', 'kilonova')
        self.assertTrue(callable(function))

    def test_get_correct_function_invalid_model(self):
        """Test that invalid model raises ValueError"""
        with self.assertRaises(ValueError):
            lensing_models._get_correct_function('invalid_model', 'supernova')

    def test_get_correct_function_with_callable(self):
        """Test that passing a callable returns it directly"""
        def custom_func(time, **kwargs):
            return time

        result = lensing_models._get_correct_function(custom_func, 'supernova')
        self.assertEqual(result, custom_func)


class TestLensingWrapperFunctions(unittest.TestCase):
    """Test lensing wrapper functions with actual base models"""

    def setUp(self):
        self.times = np.array([1.0, 2.0, 3.0])
        self.base_kwargs = {
            'redshift': 0.01,
            'frequency': 6e14,
            'output_format': 'flux_density',
        }

    def test_lensing_with_function_custom(self):
        """Test lensing_with_function with custom base model"""
        def custom_base_model(time, **kwargs):
            # Simple model: flux = 10 * time
            return 10.0 * np.array(time)

        kwargs = self.base_kwargs.copy()
        kwargs['base_model'] = custom_base_model
        kwargs['dt_1'] = 0.0
        kwargs['mu_1'] = 1.0
        kwargs['dt_2'] = 0.5
        kwargs['mu_2'] = 0.5

        result = lensing_models.lensing_with_function(
            time=self.times,
            nimages=2,
            **kwargs
        )

        # Check output shape
        self.assertEqual(len(result), len(self.times))
        # Check that magnification is applied
        self.assertTrue(np.all(result > 0))


class TestLensingModelsFluxDensity(unittest.TestCase):
    """Test lensing models with flux_density output"""

    def setUp(self):
        # Use larger times to ensure all lensed images have arrived
        self.times = np.array([10.0, 20.0, 30.0])

    def test_lensing_supernova_arnett(self):
        """Test lensing with Arnett supernova model"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/arnett.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'arnett',
            'dt_1': 0.0,
            'mu_1': 1.5,
            'dt_2': 5.0,
            'mu_2': 1.2,
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_supernova_base_model']
        result = function(self.times, nimages=2, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_lensing_kilonova_one_component(self):
        """Test lensing with one component kilonova model"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/one_component_kilonova_model.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'one_component_kilonova_model',
            'dt_1': 0.0,
            'mu_1': 2.0,
            'dt_2': 5.0,  # Use smaller delay so images overlap
            'mu_2': 1.5,
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_kilonova_base_model']
        result = function(self.times, nimages=2, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_lensing_tde_analytical(self):
        """Test lensing with analytical TDE model"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/tde_analytical.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'tde_analytical',
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 5.0,  # Use smaller delay
            'mu_2': 0.8,
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_tde_base_model']
        result = function(self.times, nimages=2, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_lensing_shock_powered(self):
        """Test lensing with shock powered model"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/shock_cooling.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'shock_cooling',
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 2.0,
            'mu_2': 0.7,
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_shock_powered_base_model']
        result = function(self.times, nimages=2, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_lensing_magnetar_driven(self):
        """Test lensing with magnetar driven model"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/basic_mergernova.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'basic_mergernova',
            'dt_1': 0.0,
            'mu_1': 1.2,
            'dt_2': 5.0,  # Use smaller delay
            'mu_2': 0.9,
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_magnetar_driven_base_model']
        result = function(self.times, nimages=2, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))


class TestLensingModelsMagnitude(unittest.TestCase):
    """Test lensing models with magnitude output"""

    def setUp(self):
        self.times = np.array([10.0, 20.0, 30.0])

    def test_lensing_supernova_magnitude(self):
        """Test lensing supernova with magnitude output"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/arnett.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'magnitude',
            'bands': 'bessellb',  # Use a band that doesn't require remote data
            'base_model': 'arnett',
            'dt_1': 0.0,
            'mu_1': 1.5,
            'dt_2': 5.0,
            'mu_2': 1.0,
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_supernova_base_model']
        result = function(self.times, nimages=2, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))


class TestLensingMultipleImages(unittest.TestCase):
    """Test lensing with different numbers of images"""

    def setUp(self):
        self.times = np.array([20.0, 30.0, 40.0])  # Larger times for delayed images
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/arnett.prior")
        self.sample = prior_dict.sample()

    def test_three_image_lensing(self):
        """Test lensing with 3 images"""
        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'arnett',
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 5.0,
            'mu_2': 0.8,
            'dt_3': 10.0,
            'mu_3': 0.6,
        }
        kwargs.update(self.sample)

        function = redback.model_library.all_models_dict['lensing_with_supernova_base_model']
        result = function(self.times, nimages=3, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_four_image_lensing(self):
        """Test lensing with 4 images (quad lens)"""
        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'arnett',
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 5.0,
            'mu_2': 0.8,
            'dt_3': 10.0,
            'mu_3': 0.6,
            'dt_4': 15.0,
            'mu_4': 0.4,
        }
        kwargs.update(self.sample)

        function = redback.model_library.all_models_dict['lensing_with_supernova_base_model']
        result = function(self.times, nimages=4, **kwargs)

        self.assertEqual(len(result), len(self.times))
        self.assertTrue(np.all(np.isfinite(result)))


class TestLensingPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of lensing models"""

    def test_magnification_increases_flux(self):
        """Test that positive magnification increases flux"""
        times = np.array([10.0, 20.0, 30.0])
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/arnett.prior")
        sample = prior_dict.sample()

        kwargs_base = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'arnett',
        }
        kwargs_base.update(sample)

        # Get unlensed flux (single image with mu=1)
        kwargs_unlensed = kwargs_base.copy()
        kwargs_unlensed['dt_1'] = 0.0
        kwargs_unlensed['mu_1'] = 1.0

        function = redback.model_library.all_models_dict['lensing_with_supernova_base_model']
        unlensed = function(times, nimages=1, **kwargs_unlensed)

        # Get lensed flux (two images, both with positive magnification)
        kwargs_lensed = kwargs_base.copy()
        kwargs_lensed['dt_1'] = 0.0
        kwargs_lensed['mu_1'] = 1.0
        kwargs_lensed['dt_2'] = 0.0  # Same time
        kwargs_lensed['mu_2'] = 1.0  # Same magnification

        lensed = function(times, nimages=2, **kwargs_lensed)

        # Lensed flux should be exactly 2x unlensed
        np.testing.assert_array_almost_equal(lensed, 2.0 * unlensed)

    def test_time_delay_shifts_peak(self):
        """Test that time delay properly shifts the light curve"""
        times = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"redback/priors/arnett.prior")
        sample = prior_dict.sample()

        kwargs = {
            'frequency': 6e14,
            'output_format': 'flux_density',
            'base_model': 'arnett',
            'dt_1': 0.0,
            'mu_1': 1.0,
            'dt_2': 10.0,  # Second image delayed by 10 days
            'mu_2': 1.0,   # Same magnification
        }
        kwargs.update(sample)

        function = redback.model_library.all_models_dict['lensing_with_supernova_base_model']
        result = function(times, nimages=2, **kwargs)

        # Result should be valid
        self.assertEqual(len(result), len(times))
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    unittest.main()
