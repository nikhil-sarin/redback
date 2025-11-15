import unittest
import bilby
import numpy as np

from redback.priors import get_lensing_priors


class TestLensingPriors(unittest.TestCase):
    """Test lensing prior generation functions"""

    def test_two_image_priors(self):
        """Test generating priors for 2 images"""
        priors = get_lensing_priors(nimages=2)

        self.assertIn('dt_1', priors)
        self.assertIn('mu_1', priors)
        self.assertIn('dt_2', priors)
        self.assertIn('mu_2', priors)

        # First image should have fixed dt=0
        self.assertIsInstance(priors['dt_1'], bilby.core.prior.DeltaFunction)
        self.assertEqual(priors['dt_1'].peak, 0.0)

        # Magnifications should be LogUniform
        self.assertIsInstance(priors['mu_1'], bilby.core.prior.LogUniform)
        self.assertIsInstance(priors['mu_2'], bilby.core.prior.LogUniform)

        # Second image should have Uniform time delay
        self.assertIsInstance(priors['dt_2'], bilby.core.prior.Uniform)

    def test_three_image_priors(self):
        """Test generating priors for 3 images"""
        priors = get_lensing_priors(nimages=3)

        self.assertEqual(len(priors), 6)  # 3 images * 2 params each
        self.assertIn('dt_3', priors)
        self.assertIn('mu_3', priors)

    def test_four_image_priors(self):
        """Test generating priors for 4 images (quad lens)"""
        priors = get_lensing_priors(nimages=4)

        self.assertEqual(len(priors), 8)  # 4 images * 2 params each
        self.assertIn('dt_4', priors)
        self.assertIn('mu_4', priors)

    def test_custom_bounds(self):
        """Test custom bounds for priors"""
        priors = get_lensing_priors(
            nimages=2,
            dt_min=10.0,
            dt_max=500.0,
            mu_min=0.5,
            mu_max=50.0
        )

        # Check time delay bounds
        self.assertEqual(priors['dt_2'].minimum, 10.0)
        self.assertEqual(priors['dt_2'].maximum, 500.0)

        # Check magnification bounds
        self.assertEqual(priors['mu_1'].minimum, 0.5)
        self.assertEqual(priors['mu_1'].maximum, 50.0)

    def test_sampling_priors(self):
        """Test that priors can be sampled"""
        priors = get_lensing_priors(nimages=2)
        sample = priors.sample()

        self.assertEqual(sample['dt_1'], 0.0)  # Fixed
        self.assertGreater(sample['dt_2'], 0.0)
        self.assertLess(sample['dt_2'], 1000.0)
        self.assertGreater(sample['mu_1'], 0.1)
        self.assertLess(sample['mu_1'], 100.0)

    def test_load_prior_files(self):
        """Test loading prior files from disk"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file('redback/priors/lensing_two_images.prior')

        self.assertIn('dt_1', prior_dict)
        self.assertIn('mu_1', prior_dict)
        self.assertIn('dt_2', prior_dict)
        self.assertIn('mu_2', prior_dict)

    def test_load_three_image_prior_file(self):
        """Test loading three-image prior file"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file('redback/priors/lensing_three_images.prior')

        self.assertEqual(len(prior_dict), 6)
        sample = prior_dict.sample()
        self.assertIn('dt_3', sample)
        self.assertIn('mu_3', sample)

    def test_load_four_image_prior_file(self):
        """Test loading four-image prior file"""
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file('redback/priors/lensing_four_images.prior')

        self.assertEqual(len(prior_dict), 8)
        sample = prior_dict.sample()
        self.assertIn('dt_4', sample)
        self.assertIn('mu_4', sample)

    def test_combine_with_base_model_priors(self):
        """Test combining lensing priors with base model priors"""
        # Load base model prior
        base_prior = bilby.prior.PriorDict()
        base_prior.from_file('redback/priors/arnett.prior')

        # Get lensing priors
        lensing_prior = get_lensing_priors(nimages=2)

        # Combine
        combined = bilby.prior.PriorDict()
        combined.update(base_prior)
        combined.update(lensing_prior)

        # Check both are present
        self.assertIn('redshift', combined)  # From arnett
        self.assertIn('mej', combined)  # From arnett
        self.assertIn('dt_1', combined)  # From lensing
        self.assertIn('mu_2', combined)  # From lensing

        # Sample combined priors
        sample = combined.sample()
        self.assertIn('redshift', sample)
        self.assertIn('dt_2', sample)


class TestLensingPriorPhysicalConstraints(unittest.TestCase):
    """Test that priors respect physical constraints"""

    def test_time_delay_positive(self):
        """Test that time delays are non-negative"""
        priors = get_lensing_priors(nimages=2)

        for _ in range(100):
            sample = priors.sample()
            self.assertEqual(sample['dt_1'], 0.0)
            self.assertGreaterEqual(sample['dt_2'], 0.0)

    def test_magnification_positive(self):
        """Test that magnifications are positive"""
        priors = get_lensing_priors(nimages=3)

        for _ in range(100):
            sample = priors.sample()
            self.assertGreater(sample['mu_1'], 0)
            self.assertGreater(sample['mu_2'], 0)
            self.assertGreater(sample['mu_3'], 0)


if __name__ == '__main__':
    unittest.main()
