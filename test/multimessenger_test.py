import numpy as np
import unittest
from unittest import mock
import tempfile
import shutil
import os

import bilby
import redback
from redback.multimessenger import MultiMessengerTransient, create_joint_prior
from redback.transient.transient import Transient


class MultiMessengerTransientTest(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for multi-messenger analysis"""
        # Create synthetic optical data
        self.optical_time = np.linspace(0, 10, 20)
        self.optical_flux = 1e-12 * np.exp(-self.optical_time / 5.0)
        self.optical_flux_err = 0.1 * self.optical_flux

        self.optical_transient = Transient(
            time=self.optical_time,
            flux=self.optical_flux,
            flux_err=self.optical_flux_err,
            data_mode='flux',
            name='test_optical'
        )

        # Create synthetic X-ray data
        self.xray_time = np.linspace(1, 15, 15)
        self.xray_flux = 5e-13 * (self.xray_time ** -1.2)
        self.xray_flux_err = 0.1 * self.xray_flux

        self.xray_transient = Transient(
            time=self.xray_time,
            flux=self.xray_flux,
            flux_err=self.xray_flux_err,
            data_mode='flux',
            name='test_xray'
        )

        # Create synthetic radio data
        self.radio_time = np.linspace(5, 20, 10)
        self.radio_flux_density = 1e-3 * (self.radio_time ** 0.5)
        self.radio_flux_density_err = 0.1 * self.radio_flux_density
        self.radio_freq = np.ones_like(self.radio_time) * 5e9  # 5 GHz

        self.radio_transient = Transient(
            time=self.radio_time,
            flux_density=self.radio_flux_density,
            flux_density_err=self.radio_flux_density_err,
            frequency=self.radio_freq,
            data_mode='flux_density',
            name='test_radio'
        )

        # Create a mock GW likelihood
        self.mock_gw_likelihood = mock.Mock(spec=bilby.Likelihood)
        self.mock_gw_likelihood.parameters = {'chirp_mass': None, 'mass_ratio': None}

        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init_single_messenger(self):
        """Test initialization with single messenger"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)
        self.assertEqual(len(mm.messengers), 1)
        self.assertIn('optical', mm.messengers)
        self.assertEqual(mm.messengers['optical'], self.optical_transient)

    def test_init_multiple_messengers(self):
        """Test initialization with multiple messengers"""
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient,
            radio_transient=self.radio_transient
        )
        self.assertEqual(len(mm.messengers), 3)
        self.assertIn('optical', mm.messengers)
        self.assertIn('xray', mm.messengers)
        self.assertIn('radio', mm.messengers)

    def test_init_with_external_likelihood(self):
        """Test initialization with external likelihood (e.g., GW)"""
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            gw_likelihood=self.mock_gw_likelihood
        )
        self.assertEqual(len(mm.messengers), 1)
        self.assertEqual(len(mm.external_likelihoods), 1)
        self.assertIn('gw', mm.external_likelihoods)

    def test_init_name(self):
        """Test custom name initialization"""
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            name='GW170817'
        )
        self.assertEqual(mm.name, 'GW170817')

    def test_add_messenger_transient(self):
        """Test adding a messenger with transient data"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)
        mm.add_messenger('xray', transient=self.xray_transient)
        self.assertEqual(len(mm.messengers), 2)
        self.assertIn('xray', mm.messengers)

    def test_add_messenger_likelihood(self):
        """Test adding a messenger with external likelihood"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)
        mm.add_messenger('gw', likelihood=self.mock_gw_likelihood)
        self.assertEqual(len(mm.external_likelihoods), 1)
        self.assertIn('gw', mm.external_likelihoods)

    def test_add_messenger_error(self):
        """Test that adding messenger with both transient and likelihood raises error"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)
        with self.assertRaises(ValueError):
            mm.add_messenger('test', transient=self.xray_transient,
                           likelihood=self.mock_gw_likelihood)

    def test_add_messenger_no_data_error(self):
        """Test that adding messenger with neither transient nor likelihood raises error"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)
        with self.assertRaises(ValueError):
            mm.add_messenger('test')

    def test_remove_messenger(self):
        """Test removing a messenger"""
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )
        mm.remove_messenger('xray')
        self.assertEqual(len(mm.messengers), 1)
        self.assertNotIn('xray', mm.messengers)

    def test_remove_external_likelihood(self):
        """Test removing an external likelihood"""
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            gw_likelihood=self.mock_gw_likelihood
        )
        mm.remove_messenger('gw')
        self.assertEqual(len(mm.external_likelihoods), 0)

    def test_build_likelihood_for_messenger(self):
        """Test building likelihood for a single messenger"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        # Define a simple model
        def simple_model(time, amplitude, decay_rate, **kwargs):
            return amplitude * np.exp(-time / decay_rate)

        likelihood = mm._build_likelihood_for_messenger(
            messenger='optical',
            transient=self.optical_transient,
            model=simple_model,
            model_kwargs={'output_format': 'flux'}
        )

        self.assertIsInstance(likelihood, bilby.Likelihood)
        self.assertIn('amplitude', likelihood.parameters)
        self.assertIn('decay_rate', likelihood.parameters)

    def test_repr(self):
        """Test string representation"""
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient,
            name='test_event'
        )
        repr_str = repr(mm)
        self.assertIn('test_event', repr_str)
        self.assertIn('optical', repr_str)
        self.assertIn('xray', repr_str)

    def test_fit_joint_no_likelihoods_error(self):
        """Test that fit_joint raises error when no likelihoods are built"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        with self.assertRaises(ValueError):
            mm.fit_joint(
                models={},  # No models provided
                priors=bilby.core.prior.PriorDict()
            )


class CreateJointPriorTest(unittest.TestCase):

    def test_create_joint_prior_shared_params(self):
        """Test creating joint prior with shared parameters"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['viewing_angle'] = bilby.core.prior.Uniform(0, np.pi/2, 'viewing_angle')
        optical_priors['kappa'] = bilby.core.prior.Uniform(0.1, 10, 'kappa')

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['viewing_angle'] = bilby.core.prior.Uniform(0, np.pi/2, 'viewing_angle')
        xray_priors['log_n0'] = bilby.core.prior.Uniform(-5, 2, 'log_n0')

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=['viewing_angle']
        )

        # Check that shared parameter appears once
        self.assertIn('viewing_angle', joint_prior)
        # Check that messenger-specific parameters have prefixes
        self.assertIn('optical_kappa', joint_prior)
        self.assertIn('xray_log_n0', joint_prior)

    def test_create_joint_prior_custom_shared_priors(self):
        """Test creating joint prior with custom shared parameter priors"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['viewing_angle'] = bilby.core.prior.Uniform(0, np.pi/2, 'viewing_angle')

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['viewing_angle'] = bilby.core.prior.Uniform(0, np.pi/2, 'viewing_angle')

        custom_viewing_angle = bilby.core.prior.Uniform(0, np.pi/4, 'viewing_angle')

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=['viewing_angle'],
            shared_param_priors={'viewing_angle': custom_viewing_angle}
        )

        # Check that the custom prior is used
        self.assertEqual(joint_prior['viewing_angle'].maximum, np.pi/4)

    def test_create_joint_prior_no_shared_params(self):
        """Test creating joint prior without shared parameters"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['param_a'] = bilby.core.prior.Uniform(0, 1, 'param_a')

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['param_b'] = bilby.core.prior.Uniform(0, 1, 'param_b')

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=[]
        )

        # All parameters should have messenger prefixes
        self.assertIn('optical_param_a', joint_prior)
        self.assertIn('xray_param_b', joint_prior)
        self.assertNotIn('param_a', joint_prior)
        self.assertNotIn('param_b', joint_prior)


if __name__ == '__main__':
    unittest.main()
