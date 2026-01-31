import numpy as np
import unittest
from unittest import mock
import tempfile
import shutil
import os

import bilby
import redback
from redback.multimessenger import MultiMessengerTransient, create_joint_prior
from redback.transient.transient import Transient, Spectrum
from redback.likelihoods import GaussianLikelihood, GaussianLikelihoodQuadratureNoise


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


class JointGalaxyTransientSpectrumTest(unittest.TestCase):
    """Tests for joint galaxy + transient spectrum fitting"""

    def setUp(self):
        """Set up test fixtures for galaxy + transient spectrum analysis"""
        # Wavelength array
        self.wavelengths = np.linspace(3500, 9000, 100)

        # Simple galaxy model (blackbody-like)
        self.galaxy_temp = 5500  # K
        self.galaxy_lum = 2.0

        # Transient model parameters
        self.transient_temp = 8000  # K
        self.transient_scale = 3.0

        # True redshift (shared)
        self.redshift = 0.05

        # Generate synthetic spectra
        self.galaxy_flux = self._simple_galaxy_model(
            self.wavelengths, self.redshift, self.galaxy_temp, self.galaxy_lum
        )
        self.transient_flux = self._simple_transient_model(
            self.wavelengths, self.redshift, self.transient_temp, self.transient_scale
        )
        self.combined_flux = self.galaxy_flux + self.transient_flux

        # Add noise
        self.flux_err = 0.05 * self.combined_flux
        np.random.seed(42)
        self.observed_flux = np.random.normal(self.combined_flux, self.flux_err)

        # Create spectrum object
        from redback.transient.transient import Spectrum
        self.spectrum = Spectrum(
            angstroms=self.wavelengths,
            flux_density=self.observed_flux,
            flux_density_err=self.flux_err,
            time='test_epoch',
            name='test_galaxy_transient'
        )

    def _simple_galaxy_model(self, wavelength, redshift, temperature, luminosity):
        """Simple galaxy spectrum model for testing"""
        # Simplified blackbody-like spectrum
        wave_rest = wavelength / (1 + redshift)
        flux = luminosity * 1e-16 * (wave_rest / 5000)**(-2) * np.exp(-wave_rest / (50 * temperature))
        return flux

    def _simple_transient_model(self, wavelength, redshift, temperature, scale):
        """Simple transient spectrum model for testing"""
        wave_rest = wavelength / (1 + redshift)
        flux = scale * 1e-16 * (wave_rest / 6000)**(-3) * np.exp(-wave_rest / (40 * temperature))
        return flux

    def _combined_model(self, wavelength, redshift, galaxy_temp, galaxy_lum,
                       transient_temp, transient_scale, **kwargs):
        """Combined galaxy + transient model"""
        galaxy = self._simple_galaxy_model(wavelength, redshift, galaxy_temp, galaxy_lum)
        transient = self._simple_transient_model(wavelength, redshift, transient_temp, transient_scale)
        return galaxy + transient

    def test_combined_model_output(self):
        """Test that combined model produces correct output"""
        combined = self._combined_model(
            self.wavelengths, self.redshift,
            self.galaxy_temp, self.galaxy_lum,
            self.transient_temp, self.transient_scale
        )
        expected = self.galaxy_flux + self.transient_flux
        np.testing.assert_array_almost_equal(combined, expected)

    def test_combined_likelihood_creation(self):
        """Test creating likelihood with combined galaxy + transient model"""
        likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.observed_flux,
            sigma=self.flux_err,
            function=self._combined_model,
            kwargs={}
        )

        self.assertIsInstance(likelihood, bilby.Likelihood)
        # Check parameters are correctly inferred
        self.assertIn('redshift', likelihood.parameters)
        self.assertIn('galaxy_temp', likelihood.parameters)
        self.assertIn('galaxy_lum', likelihood.parameters)
        self.assertIn('transient_temp', likelihood.parameters)
        self.assertIn('transient_scale', likelihood.parameters)

    def test_likelihood_evaluation(self):
        """Test that likelihood can be evaluated with correct parameters"""
        likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.observed_flux,
            sigma=self.flux_err,
            function=self._combined_model,
            kwargs={}
        )

        # Set parameters to true values
        likelihood.parameters['redshift'] = self.redshift
        likelihood.parameters['galaxy_temp'] = self.galaxy_temp
        likelihood.parameters['galaxy_lum'] = self.galaxy_lum
        likelihood.parameters['transient_temp'] = self.transient_temp
        likelihood.parameters['transient_scale'] = self.transient_scale

        # Likelihood should be finite
        log_l = likelihood.log_likelihood()
        self.assertTrue(np.isfinite(log_l))

    def test_transient_only_model_comparison(self):
        """Test that transient-only model gives different results than combined"""
        # Transient-only likelihood
        transient_only_likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.observed_flux,
            sigma=self.flux_err,
            function=self._simple_transient_model,
            kwargs={}
        )

        # Combined likelihood
        combined_likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.observed_flux,
            sigma=self.flux_err,
            function=self._combined_model,
            kwargs={}
        )

        # Set true parameters for combined model
        combined_likelihood.parameters['redshift'] = self.redshift
        combined_likelihood.parameters['galaxy_temp'] = self.galaxy_temp
        combined_likelihood.parameters['galaxy_lum'] = self.galaxy_lum
        combined_likelihood.parameters['transient_temp'] = self.transient_temp
        combined_likelihood.parameters['transient_scale'] = self.transient_scale

        # Set parameters for transient-only (try to match true transient params)
        transient_only_likelihood.parameters['redshift'] = self.redshift
        transient_only_likelihood.parameters['temperature'] = self.transient_temp
        transient_only_likelihood.parameters['scale'] = self.transient_scale

        # Combined model should have higher likelihood since it's the correct model
        combined_log_l = combined_likelihood.log_likelihood()
        transient_only_log_l = transient_only_likelihood.log_likelihood()

        # Combined model should fit better
        self.assertGreater(combined_log_l, transient_only_log_l)

    def test_prior_setup_for_joint_fitting(self):
        """Test setting up priors for galaxy + transient parameters"""
        priors = bilby.core.prior.PriorDict()

        # Shared parameter
        priors['redshift'] = bilby.core.prior.Gaussian(0.05, 0.001, 'redshift')

        # Galaxy parameters
        priors['galaxy_temp'] = bilby.core.prior.Uniform(4000, 7000, 'galaxy_temp')
        priors['galaxy_lum'] = bilby.core.prior.Uniform(0.5, 5.0, 'galaxy_lum')

        # Transient parameters
        priors['transient_temp'] = bilby.core.prior.Uniform(5000, 12000, 'transient_temp')
        priors['transient_scale'] = bilby.core.prior.Uniform(1.0, 10.0, 'transient_scale')

        # Verify all priors are set
        self.assertEqual(len(priors), 5)
        self.assertIn('redshift', priors)
        self.assertIn('galaxy_temp', priors)
        self.assertIn('galaxy_lum', priors)
        self.assertIn('transient_temp', priors)
        self.assertIn('transient_scale', priors)

        # Check prior sampling works
        sample = priors.sample()
        self.assertIn('redshift', sample)
        self.assertIn('galaxy_temp', sample)

    def test_custom_likelihoods_in_multimessenger(self):
        """Test using galaxy + transient as custom likelihoods in MultiMessengerTransient"""
        # Create likelihoods for each component
        galaxy_likelihood = mock.Mock(spec=bilby.Likelihood)
        galaxy_likelihood.parameters = {'redshift': None, 'galaxy_temp': None, 'galaxy_lum': None}

        transient_likelihood = mock.Mock(spec=bilby.Likelihood)
        transient_likelihood.parameters = {'redshift': None, 'transient_temp': None, 'transient_scale': None}

        # Create MultiMessengerTransient with custom likelihoods
        mm_transient = MultiMessengerTransient(
            custom_likelihoods={
                'galaxy': galaxy_likelihood,
                'transient': transient_likelihood
            },
            name='galaxy_transient_decomposition'
        )

        self.assertEqual(len(mm_transient.external_likelihoods), 2)
        self.assertIn('galaxy', mm_transient.external_likelihoods)
        self.assertIn('transient', mm_transient.external_likelihoods)

    def test_gaussian_emission_line(self):
        """Test adding Gaussian emission line to galaxy model"""
        def gaussian_line(wavelength, center, amplitude, width):
            """Gaussian emission line profile"""
            return amplitude * np.exp(-0.5 * ((wavelength - center) / width)**2)

        # H-alpha at 6563 Angstroms
        h_alpha_rest = 6563
        h_alpha_obs = h_alpha_rest * (1 + self.redshift)
        line_flux = 1e-16
        line_width = 3.0  # Angstroms

        line_profile = gaussian_line(self.wavelengths, h_alpha_obs, line_flux, line_width)

        # Line should be peaked at the right position
        peak_idx = np.argmax(line_profile)
        peak_wavelength = self.wavelengths[peak_idx]

        # Check peak is near expected position
        self.assertAlmostEqual(peak_wavelength, h_alpha_obs, delta=50)

        # Check line amplitude is close to input (may not be exact if wavelength grid
        # doesn't have a point exactly at line center)
        # The peak should be at least 50% of the input amplitude (for reasonable grid spacing)
        self.assertGreater(np.max(line_profile), 0.5 * line_flux)
        # And should not exceed the input amplitude (Gaussian peak is at center)
        self.assertLessEqual(np.max(line_profile), line_flux)

    def test_galaxy_model_with_emission_lines(self):
        """Test galaxy model including emission lines"""
        def gaussian_line(wavelength, center, amplitude, width):
            return amplitude * np.exp(-0.5 * ((wavelength - center) / width)**2)

        def galaxy_with_lines(wavelength, redshift, galaxy_temp, galaxy_lum,
                             h_alpha_flux, h_beta_flux, line_width, **kwargs):
            """Galaxy model with continuum + emission lines"""
            continuum = self._simple_galaxy_model(wavelength, redshift, galaxy_temp, galaxy_lum)

            # Add emission lines (redshifted)
            h_alpha = gaussian_line(wavelength, 6563 * (1 + redshift), h_alpha_flux, line_width)
            h_beta = gaussian_line(wavelength, 4861 * (1 + redshift), h_beta_flux, line_width)

            return continuum + h_alpha + h_beta

        # Create likelihood with emission lines
        likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.observed_flux,
            sigma=self.flux_err,
            function=galaxy_with_lines,
            kwargs={}
        )

        # Check that line parameters are included
        self.assertIn('h_alpha_flux', likelihood.parameters)
        self.assertIn('h_beta_flux', likelihood.parameters)
        self.assertIn('line_width', likelihood.parameters)

    def test_shared_redshift_constraint(self):
        """Test that shared redshift properly constrains both components"""
        priors = bilby.core.prior.PriorDict()

        # Tight prior on redshift (known from galaxy)
        priors['redshift'] = bilby.core.prior.Gaussian(0.05, 0.001, 'redshift')

        # Sample should be near the mean
        samples = [priors.sample()['redshift'] for _ in range(100)]
        mean_sample = np.mean(samples)
        std_sample = np.std(samples)

        self.assertAlmostEqual(mean_sample, 0.05, places=2)
        self.assertLess(std_sample, 0.01)

    def test_decomposition_extraction(self):
        """Test extracting galaxy and transient components from fit"""
        # Simulate a fit result (best-fit parameters)
        best_params = {
            'redshift': self.redshift,
            'galaxy_temp': self.galaxy_temp,
            'galaxy_lum': self.galaxy_lum,
            'transient_temp': self.transient_temp,
            'transient_scale': self.transient_scale
        }

        # Extract individual components
        galaxy_component = self._simple_galaxy_model(
            self.wavelengths, best_params['redshift'],
            best_params['galaxy_temp'], best_params['galaxy_lum']
        )
        transient_component = self._simple_transient_model(
            self.wavelengths, best_params['redshift'],
            best_params['transient_temp'], best_params['transient_scale']
        )

        # Components should sum to combined
        combined_from_params = self._combined_model(self.wavelengths, **best_params)
        np.testing.assert_array_almost_equal(
            galaxy_component + transient_component,
            combined_from_params
        )

    def test_flux_ratio_calculation(self):
        """Test calculating flux ratio between transient and galaxy"""
        galaxy_mean_flux = np.mean(self.galaxy_flux)
        transient_mean_flux = np.mean(self.transient_flux)

        flux_ratio = transient_mean_flux / galaxy_mean_flux

        # Should be positive
        self.assertGreater(flux_ratio, 0)

        # In our setup, transient is brighter
        self.assertGreater(flux_ratio, 1.0)

    def test_multiple_spectra_epochs(self):
        """Test handling multiple spectra at different epochs"""
        # Create spectra at different epochs with evolving transient
        epoch_1_transient_scale = 5.0  # Peak brightness
        epoch_2_transient_scale = 3.0  # Declining
        epoch_3_transient_scale = 1.5  # Fainter

        # Simulated observed spectra at each epoch
        flux_1 = self.galaxy_flux + self._simple_transient_model(
            self.wavelengths, self.redshift, self.transient_temp, epoch_1_transient_scale
        )
        flux_2 = self.galaxy_flux + self._simple_transient_model(
            self.wavelengths, self.redshift, self.transient_temp, epoch_2_transient_scale
        )
        flux_3 = self.galaxy_flux + self._simple_transient_model(
            self.wavelengths, self.redshift, self.transient_temp, epoch_3_transient_scale
        )

        # Transient should fade while galaxy stays constant
        transient_contrib_1 = np.mean(flux_1 - self.galaxy_flux)
        transient_contrib_2 = np.mean(flux_2 - self.galaxy_flux)
        transient_contrib_3 = np.mean(flux_3 - self.galaxy_flux)

        self.assertGreater(transient_contrib_1, transient_contrib_2)
        self.assertGreater(transient_contrib_2, transient_contrib_3)

        # Galaxy contribution should be constant
        galaxy_contrib_1 = np.mean(self.galaxy_flux)
        galaxy_contrib_2 = np.mean(self.galaxy_flux)
        galaxy_contrib_3 = np.mean(self.galaxy_flux)

        np.testing.assert_almost_equal(galaxy_contrib_1, galaxy_contrib_2)
        np.testing.assert_almost_equal(galaxy_contrib_2, galaxy_contrib_3)


class SpectrumPhotometryJointFittingTest(unittest.TestCase):
    """Tests for joint spectrum and photometry fitting"""

    def setUp(self):
        """Set up test fixtures for spectrum + photometry joint analysis"""
        # Photometry data
        self.phot_time = np.array([1, 3, 5, 7, 10])
        self.phot_flux = 1e-12 * np.exp(-self.phot_time / 5.0)
        self.phot_flux_err = 0.1 * self.phot_flux

        self.photometry = Transient(
            time=self.phot_time,
            flux=self.phot_flux,
            flux_err=self.phot_flux_err,
            data_mode='flux',
            name='test_photometry'
        )

        # Spectrum data at t = 3 days
        self.wavelengths = np.linspace(4000, 8000, 50)
        self.spec_flux = 1e-16 * (self.wavelengths / 5000)**(-2)
        self.spec_flux_err = 0.05 * self.spec_flux

        self.spectrum = Spectrum(
            angstroms=self.wavelengths,
            flux_density=self.spec_flux,
            flux_density_err=self.spec_flux_err,
            time='3 days',
            name='test_spectrum'
        )

    def test_photometry_and_spectrum_different_data_types(self):
        """Test that photometry and spectrum are different data types"""
        self.assertIsInstance(self.photometry, Transient)
        self.assertIsInstance(self.spectrum, Spectrum)
        self.assertNotIsInstance(self.photometry, Spectrum)

    def test_custom_likelihoods_for_different_data(self):
        """Test creating separate likelihoods for photometry and spectrum"""
        def phot_model(time, amplitude, decay, **kwargs):
            return amplitude * np.exp(-time / decay)

        def spec_model(wavelength, temperature, scale, **kwargs):
            return scale * (wavelength / 5000)**(-2)

        phot_likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.phot_time,
            y=self.phot_flux,
            sigma=self.phot_flux_err,
            function=phot_model,
            kwargs={}
        )

        spec_likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.spec_flux,
            sigma=self.spec_flux_err,
            function=spec_model,
            kwargs={}
        )

        # Both should be valid likelihoods
        self.assertIsInstance(phot_likelihood, bilby.Likelihood)
        self.assertIsInstance(spec_likelihood, bilby.Likelihood)

        # Parameters should be different
        self.assertIn('amplitude', phot_likelihood.parameters)
        self.assertIn('decay', phot_likelihood.parameters)
        self.assertIn('temperature', spec_likelihood.parameters)
        self.assertIn('scale', spec_likelihood.parameters)

    def test_joint_likelihood_combination(self):
        """Test combining photometry and spectrum likelihoods"""
        def phot_model(time, amplitude, decay, **kwargs):
            return amplitude * np.exp(-time / decay)

        def spec_model(wavelength, temperature, scale, **kwargs):
            return scale * (wavelength / 5000)**(-2)

        phot_likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.phot_time,
            y=self.phot_flux,
            sigma=self.phot_flux_err,
            function=phot_model,
            kwargs={}
        )

        spec_likelihood = redback.likelihoods.GaussianLikelihood(
            x=self.wavelengths,
            y=self.spec_flux,
            sigma=self.spec_flux_err,
            function=spec_model,
            kwargs={}
        )

        # Combine using bilby's JointLikelihood
        joint_likelihood = bilby.core.likelihood.JointLikelihood(
            phot_likelihood, spec_likelihood
        )

        self.assertIsInstance(joint_likelihood, bilby.core.likelihood.JointLikelihood)
        # Joint likelihood should have parameters from both
        all_params = joint_likelihood.parameters
        self.assertIn('amplitude', all_params)
        self.assertIn('decay', all_params)
        self.assertIn('temperature', all_params)
        self.assertIn('scale', all_params)

    def test_multimessenger_with_custom_likelihoods(self):
        """Test MultiMessengerTransient with photometry and spectrum as custom likelihoods"""
        mock_phot_likelihood = mock.Mock(spec=bilby.Likelihood)
        mock_phot_likelihood.parameters = {'amplitude': None, 'decay': None}

        mock_spec_likelihood = mock.Mock(spec=bilby.Likelihood)
        mock_spec_likelihood.parameters = {'temperature': None, 'scale': None}

        mm_transient = MultiMessengerTransient(
            custom_likelihoods={
                'photometry': mock_phot_likelihood,
                'spectrum': mock_spec_likelihood
            },
            name='joint_phot_spec'
        )

        self.assertEqual(len(mm_transient.external_likelihoods), 2)
        self.assertIn('photometry', mm_transient.external_likelihoods)
        self.assertIn('spectrum', mm_transient.external_likelihoods)
        self.assertEqual(mm_transient.name, 'joint_phot_spec')


class MultiMessengerCoreFunctionalityTest(unittest.TestCase):
    """Tests for core MultiMessengerTransient methods with high coverage"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

        # Create optical transient
        self.optical_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.optical_flux = np.array([1e-12, 8e-13, 6e-13, 4e-13, 2e-13])
        self.optical_flux_err = 0.1 * self.optical_flux

        self.optical_transient = Transient(
            time=self.optical_time,
            flux=self.optical_flux,
            flux_err=self.optical_flux_err,
            data_mode='flux',
            name='test_optical'
        )

        # Create X-ray transient
        self.xray_time = np.array([2.0, 4.0, 6.0, 8.0])
        self.xray_flux = np.array([5e-13, 3e-13, 2e-13, 1e-13])
        self.xray_flux_err = 0.15 * self.xray_flux

        self.xray_transient = Transient(
            time=self.xray_time,
            flux=self.xray_flux,
            flux_err=self.xray_flux_err,
            data_mode='flux',
            name='test_xray'
        )

        # Simple test model
        def simple_model(time, amplitude, decay_rate, **kwargs):
            return amplitude * np.exp(-time / decay_rate)

        self.simple_model = simple_model

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_build_likelihood_for_messenger_with_callable(self):
        """Test _build_likelihood_for_messenger with callable model"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        likelihood = mm._build_likelihood_for_messenger(
            messenger='optical',
            transient=self.optical_transient,
            model=self.simple_model,
            model_kwargs={'test_kwarg': 'value'}
        )

        self.assertIsInstance(likelihood, GaussianLikelihood)
        self.assertIn('amplitude', likelihood.parameters)
        self.assertIn('decay_rate', likelihood.parameters)
        self.assertEqual(likelihood.kwargs, {'test_kwarg': 'value'})

    def test_build_likelihood_for_messenger_with_string_model_invalid(self):
        """Test _build_likelihood_for_messenger with invalid string model"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        with self.assertRaises(ValueError) as context:
            mm._build_likelihood_for_messenger(
                messenger='optical',
                transient=self.optical_transient,
                model='nonexistent_model_name_12345'
            )

        self.assertIn('not found in redback model library', str(context.exception))

    def test_build_likelihood_for_messenger_unsupported_type(self):
        """Test _build_likelihood_for_messenger with unsupported likelihood type"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        with self.assertRaises(ValueError) as context:
            mm._build_likelihood_for_messenger(
                messenger='optical',
                transient=self.optical_transient,
                model=self.simple_model,
                likelihood_type='UnsupportedLikelihoodType'
            )

        self.assertIn('Unsupported likelihood type', str(context.exception))

    def test_build_likelihood_for_messenger_gaussian_quadrature_noise(self):
        """Test _build_likelihood_for_messenger with GaussianLikelihoodQuadratureNoise"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        likelihood = mm._build_likelihood_for_messenger(
            messenger='optical',
            transient=self.optical_transient,
            model=self.simple_model,
            likelihood_type='GaussianLikelihoodQuadratureNoise'
        )

        self.assertIsInstance(likelihood, GaussianLikelihoodQuadratureNoise)

    def test_build_likelihood_none_model_kwargs(self):
        """Test _build_likelihood_for_messenger with None model_kwargs"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        likelihood = mm._build_likelihood_for_messenger(
            messenger='optical',
            transient=self.optical_transient,
            model=self.simple_model,
            model_kwargs=None
        )

        self.assertEqual(likelihood.kwargs, {})

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_single_likelihood(self, mock_sampler):
        """Test fit_joint with single likelihood (warning case)"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')

        result = mm.fit_joint(
            models={'optical': self.simple_model},
            priors=priors,
            shared_params=['amplitude'],
            model_kwargs={'optical': {}},
            outdir=self.test_dir,
            nlive=100
        )

        self.assertEqual(result, mock_result)
        mock_sampler.assert_called_once()

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_multiple_likelihoods(self, mock_sampler):
        """Test fit_joint with multiple likelihoods"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')

        result = mm.fit_joint(
            models={'optical': self.simple_model, 'xray': self.simple_model},
            priors=priors,
            shared_params=['amplitude'],
            outdir=self.test_dir,
            label='test_joint',
            nlive=100,
            walks=50
        )

        self.assertEqual(result, mock_result)
        mock_sampler.assert_called_once()

        # Check that JointLikelihood was created
        call_kwargs = mock_sampler.call_args[1]
        self.assertIsInstance(call_kwargs['likelihood'], bilby.core.likelihood.JointLikelihood)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_with_external_likelihoods(self, mock_sampler):
        """Test fit_joint with external likelihoods (e.g., GW)"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mock_gw_likelihood = mock.Mock(spec=bilby.Likelihood)
        mock_gw_likelihood.parameters = {'chirp_mass': None}

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            gw_likelihood=mock_gw_likelihood
        )

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')
        priors['chirp_mass'] = bilby.core.prior.Uniform(1, 2, 'chirp_mass')

        result = mm.fit_joint(
            models={'optical': self.simple_model},
            priors=priors,
            outdir=self.test_dir,
            nlive=100
        )

        self.assertEqual(result, mock_result)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_with_dict_priors(self, mock_sampler):
        """Test fit_joint with dict (not PriorDict) priors"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        # Use regular dict instead of PriorDict
        priors = {
            'amplitude': bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude'),
            'decay_rate': bilby.core.prior.Uniform(1, 10, 'decay_rate')
        }

        result = mm.fit_joint(
            models={'optical': self.simple_model},
            priors=priors,
            outdir=self.test_dir,
            nlive=100
        )

        self.assertEqual(result, mock_result)

    def test_fit_joint_no_likelihoods(self):
        """Test fit_joint raises error when no likelihoods"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')

        with self.assertRaises(ValueError) as context:
            mm.fit_joint(
                models={},  # No models
                priors=priors,
                outdir=self.test_dir
            )

        self.assertIn('No likelihoods were constructed', str(context.exception))

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_default_outdir_and_label(self, mock_sampler):
        """Test fit_joint uses default outdir and label"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            name='custom_name'
        )

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')

        mm.fit_joint(
            models={'optical': self.simple_model},
            priors=priors,
            nlive=100
        )

        call_kwargs = mock_sampler.call_args[1]
        self.assertEqual(call_kwargs['label'], 'custom_name')
        self.assertIn('outdir_multimessenger', call_kwargs['outdir'])

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_with_different_likelihood_types(self, mock_sampler):
        """Test fit_joint with different likelihood types per messenger"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')

        result = mm.fit_joint(
            models={'optical': self.simple_model, 'xray': self.simple_model},
            priors=priors,
            likelihood_types={
                'optical': 'GaussianLikelihood',
                'xray': 'GaussianLikelihoodQuadratureNoise'
            },
            outdir=self.test_dir,
            nlive=100
        )

        self.assertEqual(result, mock_result)

    @mock.patch('redback.fit_model')
    def test_fit_individual(self, mock_fit_model):
        """Test fit_individual method"""
        mock_result_optical = mock.Mock()
        mock_result_xray = mock.Mock()
        mock_fit_model.side_effect = [mock_result_optical, mock_result_xray]

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        optical_priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['amplitude'] = bilby.core.prior.Uniform(1e-14, 1e-12, 'amplitude')
        xray_priors['decay_rate'] = bilby.core.prior.Uniform(2, 15, 'decay_rate')

        results = mm.fit_individual(
            models={'optical': self.simple_model, 'xray': self.simple_model},
            priors={'optical': optical_priors, 'xray': xray_priors},
            model_kwargs={'optical': {}, 'xray': {}},
            outdir=self.test_dir,
            nlive=100
        )

        self.assertEqual(results['optical'], mock_result_optical)
        self.assertEqual(results['xray'], mock_result_xray)
        self.assertEqual(mock_fit_model.call_count, 2)

    @mock.patch('redback.fit_model')
    def test_fit_individual_missing_model(self, mock_fit_model):
        """Test fit_individual skips messengers without models"""
        mock_result = mock.Mock()
        mock_fit_model.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')

        results = mm.fit_individual(
            models={'optical': self.simple_model},  # No xray model
            priors={'optical': optical_priors},
            outdir=self.test_dir
        )

        # Only optical should be fitted
        self.assertIn('optical', results)
        self.assertNotIn('xray', results)

    @mock.patch('redback.fit_model')
    def test_fit_individual_missing_prior(self, mock_fit_model):
        """Test fit_individual skips messengers without priors"""
        mock_result = mock.Mock()
        mock_fit_model.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')

        results = mm.fit_individual(
            models={'optical': self.simple_model, 'xray': self.simple_model},
            priors={'optical': optical_priors},  # No xray priors
            outdir=self.test_dir
        )

        # Only optical should be fitted
        self.assertIn('optical', results)
        self.assertNotIn('xray', results)

    @mock.patch('redback.fit_model')
    def test_fit_individual_default_outdir(self, mock_fit_model):
        """Test fit_individual uses default outdir"""
        mock_result = mock.Mock()
        mock_fit_model.return_value = mock_result

        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')

        mm.fit_individual(
            models={'optical': self.simple_model},
            priors={'optical': optical_priors}
        )

        # Check that outdir was set
        call_kwargs = mock_fit_model.call_args[1]
        self.assertIn('outdir_individual', call_kwargs['outdir'])

    def test_init_with_uv_and_infrared(self):
        """Test initialization with UV and infrared transients"""
        uv_transient = Transient(
            time=self.optical_time,
            flux=self.optical_flux * 2,
            flux_err=self.optical_flux_err,
            data_mode='flux',
            name='test_uv'
        )

        ir_transient = Transient(
            time=self.optical_time,
            flux=self.optical_flux * 0.5,
            flux_err=self.optical_flux_err,
            data_mode='flux',
            name='test_ir'
        )

        mm = MultiMessengerTransient(
            uv_transient=uv_transient,
            infrared_transient=ir_transient
        )

        self.assertIn('uv', mm.messengers)
        self.assertIn('infrared', mm.messengers)
        self.assertEqual(len(mm.messengers), 2)

    def test_init_with_neutrino_likelihood(self):
        """Test initialization with neutrino likelihood"""
        mock_neutrino_likelihood = mock.Mock(spec=bilby.Likelihood)
        mock_neutrino_likelihood.parameters = {'neutrino_energy': None}

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            neutrino_likelihood=mock_neutrino_likelihood
        )

        self.assertIn('neutrino', mm.external_likelihoods)
        self.assertEqual(len(mm.external_likelihoods), 1)

    def test_remove_nonexistent_messenger(self):
        """Test removing a messenger that doesn't exist"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        # Should not raise, just log warning
        mm.remove_messenger('nonexistent')

        # Original messenger should still be there
        self.assertIn('optical', mm.messengers)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_metadata(self, mock_sampler):
        """Test that fit_joint sets correct metadata"""
        mock_result = mock.Mock()
        mock_sampler.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient,
            name='test_metadata'
        )

        priors = bilby.core.prior.PriorDict()
        priors['amplitude'] = bilby.core.prior.Uniform(1e-13, 1e-11, 'amplitude')
        priors['decay_rate'] = bilby.core.prior.Uniform(1, 10, 'decay_rate')

        mm.fit_joint(
            models={'optical': self.simple_model, 'xray': self.simple_model},
            priors=priors,
            shared_params=['amplitude'],
            outdir=self.test_dir,
            nlive=100
        )

        call_kwargs = mock_sampler.call_args[1]
        meta_data = call_kwargs['meta_data']

        self.assertTrue(meta_data['multimessenger'])
        self.assertIn('optical', meta_data['messengers'])
        self.assertIn('xray', meta_data['messengers'])
        self.assertIn('amplitude', meta_data['shared_params'])
        self.assertEqual(meta_data['name'], 'test_metadata')


class CreateJointPriorAdvancedTest(unittest.TestCase):
    """Additional tests for create_joint_prior utility"""

    def test_create_joint_prior_multiple_shared_params(self):
        """Test with multiple shared parameters"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['viewing_angle'] = bilby.core.prior.Uniform(0, 1.57)
        optical_priors['distance'] = bilby.core.prior.Uniform(10, 100)
        optical_priors['mej'] = bilby.core.prior.Uniform(0.01, 0.1)

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['viewing_angle'] = bilby.core.prior.Uniform(0, 1.57)
        xray_priors['distance'] = bilby.core.prior.Uniform(10, 100)
        xray_priors['logn0'] = bilby.core.prior.Uniform(-3, 2)

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=['viewing_angle', 'distance']
        )

        # Shared params appear once
        self.assertIn('viewing_angle', joint_prior)
        self.assertIn('distance', joint_prior)

        # Non-shared params have prefixes
        self.assertIn('optical_mej', joint_prior)
        self.assertIn('xray_logn0', joint_prior)

        # No duplicates
        self.assertNotIn('optical_viewing_angle', joint_prior)
        self.assertNotIn('xray_distance', joint_prior)

    def test_create_joint_prior_empty_individual(self):
        """Test with empty individual priors"""
        optical_priors = bilby.core.prior.PriorDict()
        xray_priors = bilby.core.prior.PriorDict()

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=[]
        )

        self.assertEqual(len(joint_prior), 0)

    def test_create_joint_prior_shared_param_not_in_any_messenger(self):
        """Test when shared param is not found in any messenger"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['mej'] = bilby.core.prior.Uniform(0.01, 0.1)

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['logn0'] = bilby.core.prior.Uniform(-3, 2)

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=['nonexistent_param']
        )

        # Non-existent shared param should not be in result
        self.assertNotIn('nonexistent_param', joint_prior)

        # Other params should have prefixes
        self.assertIn('optical_mej', joint_prior)
        self.assertIn('xray_logn0', joint_prior)


class RealCodePathsTest(unittest.TestCase):
    """Tests that execute real code paths without excessive mocking for coverage"""

    def setUp(self):
        """Set up real transients"""
        # Create real transient with actual data
        self.time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.flux = np.array([1e-12, 8e-13, 6e-13, 4e-13, 2e-13])
        self.flux_err = self.flux * 0.1

        self.optical_transient = Transient(
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
            data_mode='flux',
            name='real_optical'
        )

        self.xray_transient = Transient(
            time=self.time,
            flux=self.flux * 0.5,
            flux_err=self.flux_err * 0.5,
            data_mode='flux',
            name='real_xray'
        )

    def test_init_with_custom_likelihoods_dict(self):
        """Test custom_likelihoods dict is properly updated (line 119-120)"""
        mock_custom = mock.MagicMock(spec=bilby.Likelihood)
        mock_custom.parameters = {}

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            custom_likelihoods={'gamma': mock_custom, 'submm': mock_custom}
        )

        self.assertIn('gamma', mm.external_likelihoods)
        self.assertIn('submm', mm.external_likelihoods)
        self.assertEqual(len(mm.external_likelihoods), 2)

    def test_build_likelihood_with_string_model_valid(self):
        """Test building likelihood with valid string model name from library (line 159-162)"""
        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        # Use a real model from redback library - exponential_powerlaw exists
        from redback.model_library import all_models_dict

        # Find a simple model that exists
        if 'exponential_powerlaw' in all_models_dict:
            model_name = 'exponential_powerlaw'
        elif 'arnett_bolometric' in all_models_dict:
            model_name = 'arnett_bolometric'
        else:
            # Use first available model
            model_name = list(all_models_dict.keys())[0]

        likelihood = mm._build_likelihood_for_messenger(
            'optical',
            self.optical_transient,
            model_name,
            model_kwargs={}
        )

        self.assertIsInstance(likelihood, GaussianLikelihood)
        # The model function should be resolved from the string
        self.assertIsNotNone(likelihood.function)

    def test_none_entries_removed_from_messengers(self):
        """Test that None entries are filtered out (line 111)"""
        # Create with only some messengers, others should be None
        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=None,  # Explicitly None
            radio_transient=None
        )

        # Only optical should be present
        self.assertEqual(len(mm.messengers), 1)
        self.assertIn('optical', mm.messengers)
        self.assertNotIn('xray', mm.messengers)
        self.assertNotIn('radio', mm.messengers)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_with_real_transient_data(self, mock_sampler):
        """Test fit_joint with real transient.get_filtered_data() call"""
        mock_sampler.return_value = mock.MagicMock()

        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        def simple_model(time, amplitude=1e-12, decay_time=2.0):
            return amplitude * np.exp(-time / decay_time)

        priors = bilby.core.prior.PriorDict({
            'amplitude': bilby.core.prior.LogUniform(1e-14, 1e-10),
            'decay_time': bilby.core.prior.Uniform(0.1, 10)
        })

        mm.fit_joint(models={'optical': simple_model}, priors=priors)

        # The actual transient's get_filtered_data should have been called
        self.assertTrue(mock_sampler.called)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_constructs_joint_likelihood(self, mock_sampler):
        """Test that JointLikelihood is actually constructed (line 321)"""
        mock_sampler.return_value = mock.MagicMock()

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        def opt_model(time, a=1):
            return a * np.exp(-time)

        def xray_model(time, b=1):
            return b * time**(-1)

        priors = bilby.core.prior.PriorDict({
            'a': bilby.core.prior.Uniform(0, 10),
            'b': bilby.core.prior.Uniform(0, 10)
        })

        mm.fit_joint(
            models={'optical': opt_model, 'xray': xray_model},
            priors=priors
        )

        # Check that JointLikelihood was constructed and passed to sampler
        call_kwargs = mock_sampler.call_args[1]
        self.assertIn('likelihood', call_kwargs)
        # With 2 messengers, should be JointLikelihood
        self.assertIsInstance(call_kwargs['likelihood'], bilby.core.likelihood.JointLikelihood)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_metadata_with_string_model(self, mock_sampler):
        """Test metadata captures string model names correctly (line 335)"""
        mock_sampler.return_value = mock.MagicMock()

        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        # Use string that would fail (to be caught in try/except or error)
        # Instead use callable with specific name
        def named_model(time, param=1):
            return time * param

        priors = bilby.core.prior.PriorDict({'param': bilby.core.prior.Uniform(0, 10)})

        mm.fit_joint(models={'optical': named_model}, priors=priors)

        call_kwargs = mock_sampler.call_args[1]
        meta_data = call_kwargs['meta_data']
        # Model should be recorded as function name
        self.assertEqual(meta_data['models']['optical'], 'named_model')

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_all_sampler_params_passed(self, mock_sampler):
        """Test all parameters are passed to bilby.run_sampler (lines 342-357)"""
        mock_sampler.return_value = mock.MagicMock()

        mm = MultiMessengerTransient(optical_transient=self.optical_transient)

        def model(time, a=1):
            return time * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        mm.fit_joint(
            models={'optical': model},
            priors=priors,
            sampler='nestle',
            nlive=500,
            walks=100,
            outdir='./test_out',
            label='test_label',
            resume=False,
            plot=False,
            save_format='hdf5',
            extra_param='value'  # Additional kwarg
        )

        call_kwargs = mock_sampler.call_args[1]
        self.assertEqual(call_kwargs['sampler'], 'nestle')
        self.assertEqual(call_kwargs['nlive'], 500)
        self.assertEqual(call_kwargs['walks'], 100)
        self.assertEqual(call_kwargs['outdir'], './test_out')
        self.assertEqual(call_kwargs['label'], 'test_label')
        self.assertEqual(call_kwargs['resume'], False)
        self.assertEqual(call_kwargs['plot'], False)
        self.assertEqual(call_kwargs['save'], 'hdf5')
        self.assertEqual(call_kwargs['extra_param'], 'value')

    @mock.patch('redback.fit_model')
    def test_fit_individual_with_real_transients(self, mock_fit_model):
        """Test fit_individual calls redback.fit_model correctly"""
        mock_result = mock.MagicMock()
        mock_fit_model.return_value = mock_result

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient
        )

        optical_priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})
        xray_priors = bilby.core.prior.PriorDict({'b': bilby.core.prior.Uniform(0, 10)})

        results = mm.fit_individual(
            models={'optical': 'model1', 'xray': 'model2'},
            priors={'optical': optical_priors, 'xray': xray_priors},
            model_kwargs={'optical': {'kwarg1': 'val1'}},
            sampler='emcee',
            nlive=1000,
            walks=150,
            outdir='./indiv_out',
            resume=False,
            plot=False
        )

        # Should be called twice, once per messenger
        self.assertEqual(mock_fit_model.call_count, 2)
        self.assertIn('optical', results)
        self.assertIn('xray', results)

        # Check that parameters were passed correctly
        calls = mock_fit_model.call_args_list
        # Find optical call
        for call in calls:
            kwargs = call[1]
            if kwargs.get('label', '').endswith('_optical'):
                self.assertEqual(kwargs['model'], 'model1')
                self.assertEqual(kwargs['transient'], self.optical_transient)
                self.assertEqual(kwargs['model_kwargs'], {'kwarg1': 'val1'})

    def test_init_all_messenger_types(self):
        """Test initialization with all EM messenger types"""
        uv_transient = Transient(
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
            data_mode='flux',
            name='uv'
        )
        ir_transient = Transient(
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
            data_mode='flux',
            name='ir'
        )
        radio_transient = Transient(
            time=self.time,
            flux_density=self.flux,
            flux_density_err=self.flux_err,
            frequency=np.ones_like(self.time) * 1e9,
            data_mode='flux_density',
            name='radio'
        )

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient,
            radio_transient=radio_transient,
            uv_transient=uv_transient,
            infrared_transient=ir_transient
        )

        self.assertEqual(len(mm.messengers), 5)
        self.assertIn('optical', mm.messengers)
        self.assertIn('xray', mm.messengers)
        self.assertIn('radio', mm.messengers)
        self.assertIn('uv', mm.messengers)
        self.assertIn('infrared', mm.messengers)

    def test_repr_with_all_types(self):
        """Test __repr__ includes all messenger types (lines 507-512)"""
        mock_gw = mock.MagicMock(spec=bilby.Likelihood)

        mm = MultiMessengerTransient(
            optical_transient=self.optical_transient,
            xray_transient=self.xray_transient,
            gw_likelihood=mock_gw,
            name='full_mm'
        )

        repr_str = repr(mm)
        self.assertIn('full_mm', repr_str)
        self.assertIn('optical', repr_str)
        self.assertIn('xray', repr_str)
        self.assertIn('gw', repr_str)


class EdgeCasesAndWarningsTest(unittest.TestCase):
    """Test edge cases and warning paths for full coverage"""

    def setUp(self):
        """Set up mock transients"""
        self.mock_transient = mock.MagicMock(spec=Transient)
        self.mock_transient.get_filtered_data.return_value = (
            np.array([1.0, 2.0, 3.0]),  # x
            None,  # x_err
            np.array([10.0, 20.0, 30.0]),  # y
            np.array([1.0, 2.0, 3.0])  # y_err
        )

    def test_build_likelihood_with_time_errors(self):
        """Test likelihood building when time errors are present (line 178-180)"""
        mock_transient_with_xerr = mock.MagicMock(spec=Transient)
        mock_transient_with_xerr.get_filtered_data.return_value = (
            np.array([1.0, 2.0, 3.0]),  # x
            np.array([0.1, 0.2, 0.3]),  # x_err - non-zero time errors
            np.array([10.0, 20.0, 30.0]),  # y
            np.array([1.0, 2.0, 3.0])  # y_err
        )

        mm = MultiMessengerTransient(optical_transient=mock_transient_with_xerr)

        def dummy_model(x, param1=1.0):
            return x * param1

        likelihood = mm._build_likelihood_for_messenger(
            'optical', mock_transient_with_xerr, dummy_model
        )

        self.assertIsInstance(likelihood, GaussianLikelihood)
        self.assertEqual(len(likelihood.x), 3)

    def test_build_likelihood_with_zero_time_errors(self):
        """Test likelihood building when time errors are all zeros"""
        mock_transient_zero_xerr = mock.MagicMock(spec=Transient)
        mock_transient_zero_xerr.get_filtered_data.return_value = (
            np.array([1.0, 2.0, 3.0]),  # x
            np.array([0.0, 0.0, 0.0]),  # x_err - all zeros
            np.array([10.0, 20.0, 30.0]),  # y
            np.array([1.0, 2.0, 3.0])  # y_err
        )

        mm = MultiMessengerTransient(optical_transient=mock_transient_zero_xerr)

        def dummy_model(x, param1=1.0):
            return x * param1

        likelihood = mm._build_likelihood_for_messenger(
            'optical', mock_transient_zero_xerr, dummy_model
        )

        self.assertIsInstance(likelihood, GaussianLikelihood)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_missing_model_for_messenger(self, mock_sampler):
        """Test warning when no model is specified for a messenger (line 305)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )
        mock_xray = mock.MagicMock(spec=Transient)
        mock_xray.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(
            optical_transient=mock_optical,
            xray_transient=mock_xray
        )

        def opt_model(x, a=1):
            return x * a

        # Only provide model for optical, not xray
        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(models={'optical': opt_model}, priors=priors)
            # Check warning was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if 'No model specified' in str(call)]
            self.assertTrue(len(warning_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_single_likelihood_warning(self, mock_sampler):
        """Test warning when only single likelihood is present (line 317)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def opt_model(x, a=1):
            return x * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(models={'optical': opt_model}, priors=priors)
            # Check warning about single likelihood was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if 'single' in str(call).lower()]
            self.assertTrue(len(warning_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_with_shared_params_logging(self, mock_sampler):
        """Test that shared parameters are logged (line 328-329)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def opt_model(x, a=1):
            return x * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(
                models={'optical': opt_model},
                priors=priors,
                shared_params=['viewing_angle', 'distance']
            )
            # Check info about shared params was logged
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Shared parameters' in str(call) or 'shared' in str(call).lower()]
            self.assertTrue(len(info_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_callable_model_in_metadata(self, mock_sampler):
        """Test that callable models use __name__ in metadata (line 335)"""
        mock_result = mock.MagicMock()
        mock_sampler.return_value = mock_result

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def my_custom_model(x, a=1):
            return x * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        mm.fit_joint(models={'optical': my_custom_model}, priors=priors)

        # Check that metadata was passed correctly with callable's __name__
        call_kwargs = mock_sampler.call_args[1]
        self.assertIn('meta_data', call_kwargs)
        self.assertEqual(call_kwargs['meta_data']['models']['optical'], 'my_custom_model')

    @mock.patch('redback.fit_model')
    def test_fit_individual_missing_prior_warning(self, mock_fit_model):
        """Test warning when no prior specified for messenger (line 427-429)"""
        mock_fit_model.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_xray = mock.MagicMock(spec=Transient)

        mm = MultiMessengerTransient(
            optical_transient=mock_optical,
            xray_transient=mock_xray
        )

        optical_priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            results = mm.fit_individual(
                models={'optical': 'model1', 'xray': 'model2'},
                priors={'optical': optical_priors}  # No prior for xray
            )
            # Check warning was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if 'No prior specified' in str(call)]
            self.assertTrue(len(warning_calls) > 0)
            # Only optical should be fitted
            self.assertIn('optical', results)
            self.assertNotIn('xray', results)

    @mock.patch('redback.fit_model')
    def test_fit_individual_missing_model_warning(self, mock_fit_model):
        """Test warning when no model specified for messenger (line 424)"""
        mock_fit_model.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_xray = mock.MagicMock(spec=Transient)

        mm = MultiMessengerTransient(
            optical_transient=mock_optical,
            xray_transient=mock_xray
        )

        optical_priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})
        xray_priors = bilby.core.prior.PriorDict({'b': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            results = mm.fit_individual(
                models={'optical': 'model1'},  # No model for xray
                priors={'optical': optical_priors, 'xray': xray_priors}
            )
            # Check warning was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if 'No model specified' in str(call)]
            self.assertTrue(len(warning_calls) > 0)
            # Only optical should be fitted
            self.assertIn('optical', results)
            self.assertNotIn('xray', results)

    def test_remove_messenger_not_found_warning(self):
        """Test warning when trying to remove non-existent messenger (line 504-505)"""
        mm = MultiMessengerTransient(optical_transient=self.mock_transient)

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.remove_messenger('nonexistent')
            # Check warning was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if 'not found' in str(call)]
            self.assertTrue(len(warning_calls) > 0)

    def test_remove_transient_messenger(self):
        """Test removing a transient (not external likelihood) messenger (line 498-500)"""
        mock_optical = mock.MagicMock(spec=Transient)
        mock_xray = mock.MagicMock(spec=Transient)

        mm = MultiMessengerTransient(
            optical_transient=mock_optical,
            xray_transient=mock_xray
        )

        self.assertIn('optical', mm.messengers)
        mm.remove_messenger('optical')
        self.assertNotIn('optical', mm.messengers)
        self.assertIn('xray', mm.messengers)

    def test_create_joint_prior_uses_first_messenger_prior(self):
        """Test that first messenger's prior is used for shared param (line 564-568)"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['viewing_angle'] = bilby.core.prior.Uniform(0, 1.57, name='viewing_angle')
        optical_priors['mej'] = bilby.core.prior.Uniform(0.01, 0.1)

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['viewing_angle'] = bilby.core.prior.Uniform(0, 3.14, name='viewing_angle')  # Different range
        xray_priors['logn0'] = bilby.core.prior.Uniform(-3, 2)

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            joint_prior = create_joint_prior(
                individual_priors={'optical': optical_priors, 'xray': xray_priors},
                shared_params=['viewing_angle']
            )
            # Should use optical's prior (first messenger)
            self.assertEqual(joint_prior['viewing_angle'].maximum, 1.57)
            # Check logger info was called about using first messenger's prior
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'optical' in str(call) and 'viewing_angle' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    def test_create_joint_prior_messenger_specific_prefixes(self):
        """Test that non-shared params get messenger prefixes (line 573-576)"""
        optical_priors = bilby.core.prior.PriorDict()
        optical_priors['mej'] = bilby.core.prior.Uniform(0.01, 0.1)
        optical_priors['vej'] = bilby.core.prior.Uniform(0.1, 0.3)

        xray_priors = bilby.core.prior.PriorDict()
        xray_priors['logn0'] = bilby.core.prior.Uniform(-3, 2)
        xray_priors['p'] = bilby.core.prior.Uniform(2.0, 3.0)

        joint_prior = create_joint_prior(
            individual_priors={'optical': optical_priors, 'xray': xray_priors},
            shared_params=[]
        )

        # All params should have messenger prefixes
        self.assertIn('optical_mej', joint_prior)
        self.assertIn('optical_vej', joint_prior)
        self.assertIn('xray_logn0', joint_prior)
        self.assertIn('xray_p', joint_prior)
        # Original names should not be in joint prior
        self.assertNotIn('mej', joint_prior)
        self.assertNotIn('logn0', joint_prior)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_with_dict_priors_conversion(self, mock_sampler):
        """Test that dict priors are converted to PriorDict (line 324-325)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def opt_model(x, a=1):
            return x * a

        # Provide priors as plain dict
        priors = {'a': bilby.core.prior.Uniform(0, 10)}

        mm.fit_joint(models={'optical': opt_model}, priors=priors)

        # Should succeed without error
        self.assertTrue(mock_sampler.called)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_creates_output_directory(self, mock_sampler):
        """Test that output directory is created (line 288)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def opt_model(x, a=1):
            return x * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            test_outdir = os.path.join(tmpdir, 'new_output_dir')
            self.assertFalse(os.path.exists(test_outdir))

            mm.fit_joint(
                models={'optical': opt_model},
                priors=priors,
                outdir=test_outdir
            )

            # Directory should have been created
            self.assertTrue(os.path.exists(test_outdir))

    @mock.patch('redback.fit_model')
    def test_fit_individual_creates_output_directory(self, mock_fit_model):
        """Test that fit_individual creates output directory (line 418)"""
        mock_fit_model.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        optical_priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            test_outdir = os.path.join(tmpdir, 'individual_output')
            self.assertFalse(os.path.exists(test_outdir))

            mm.fit_individual(
                models={'optical': 'model1'},
                priors={'optical': optical_priors},
                outdir=test_outdir
            )

            # Directory should have been created
            self.assertTrue(os.path.exists(test_outdir))

    def test_init_logging(self):
        """Test that initialization logs info (line 122-123)"""
        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm = MultiMessengerTransient(
                optical_transient=self.mock_transient,
                name='test_mm'
            )
            # Check info was logged about initialization
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Initialized' in str(call) or 'test_mm' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    def test_add_messenger_logging(self):
        """Test that adding messenger logs info (line 484, 487)"""
        mm = MultiMessengerTransient()

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.add_messenger('gamma', transient=self.mock_transient)
            # Check info was logged
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Added' in str(call) and 'gamma' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    def test_add_external_likelihood_logging(self):
        """Test that adding external likelihood logs info"""
        mm = MultiMessengerTransient()
        mock_likelihood = mock.MagicMock(spec=bilby.Likelihood)

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.add_messenger('custom', likelihood=mock_likelihood)
            # Check info was logged
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Added' in str(call) and 'external' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    def test_remove_messenger_logging(self):
        """Test that removing messenger logs info (line 500, 503)"""
        mm = MultiMessengerTransient(optical_transient=self.mock_transient)

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.remove_messenger('optical')
            # Check info was logged
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Removed' in str(call) and 'optical' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_external_likelihood_logging(self, mock_sampler):
        """Test that adding external likelihoods logs info (line 309)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_gw_likelihood = mock.MagicMock(spec=bilby.Likelihood)
        mock_gw_likelihood.parameters = {}

        mm = MultiMessengerTransient(gw_likelihood=mock_gw_likelihood)

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(models={}, priors=priors)
            # Check info was logged about adding external likelihood
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Adding external likelihood' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_combining_likelihoods_logging(self, mock_sampler):
        """Test that combining likelihoods logs info (line 320)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )
        mock_xray = mock.MagicMock(spec=Transient)
        mock_xray.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(
            optical_transient=mock_optical,
            xray_transient=mock_xray
        )

        def opt_model(x, a=1):
            return x * a

        def xray_model(x, b=1):
            return x * b

        priors = bilby.core.prior.PriorDict({
            'a': bilby.core.prior.Uniform(0, 10),
            'b': bilby.core.prior.Uniform(0, 10)
        })

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(
                models={'optical': opt_model, 'xray': xray_model},
                priors=priors
            )
            # Check info was logged about combining likelihoods
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Combining' in str(call) and 'likelihood' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_sampler_start_logging(self, mock_sampler):
        """Test that starting sampler logs info (line 341)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def opt_model(x, a=1):
            return x * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(models={'optical': opt_model}, priors=priors)
            # Check info was logged about starting sampler
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Starting' in str(call) and 'sampler' in str(call)]
            self.assertTrue(len(info_calls) > 0)

    @mock.patch('bilby.run_sampler')
    def test_fit_joint_complete_logging(self, mock_sampler):
        """Test that joint analysis completion logs info (line 359)"""
        mock_sampler.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mock_optical.get_filtered_data.return_value = (
            np.array([1.0]), None, np.array([10.0]), np.array([1.0])
        )

        mm = MultiMessengerTransient(optical_transient=mock_optical)

        def opt_model(x, a=1):
            return x * a

        priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_joint(models={'optical': opt_model}, priors=priors)
            # Check info was logged about completion
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'complete' in str(call).lower()]
            self.assertTrue(len(info_calls) > 0)

    @mock.patch('redback.fit_model')
    def test_fit_individual_per_messenger_logging(self, mock_fit_model):
        """Test that fitting each messenger logs info (line 435, 455)"""
        mock_fit_model.return_value = mock.MagicMock()

        mock_optical = mock.MagicMock(spec=Transient)
        mm = MultiMessengerTransient(optical_transient=mock_optical)

        optical_priors = bilby.core.prior.PriorDict({'a': bilby.core.prior.Uniform(0, 10)})

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm.fit_individual(
                models={'optical': 'model1'},
                priors={'optical': optical_priors}
            )
            # Check info was logged about fitting and completion
            fitting_calls = [call for call in mock_logger.info.call_args_list
                            if 'Fitting' in str(call) or 'Completed' in str(call)]
            self.assertTrue(len(fitting_calls) >= 2)

    def test_build_likelihood_logging(self):
        """Test that building likelihood logs info (line 180, 189)"""
        mm = MultiMessengerTransient(optical_transient=self.mock_transient)

        def dummy_model(x, param1=1.0):
            return x * param1

        with mock.patch('redback.multimessenger.logger') as mock_logger:
            mm._build_likelihood_for_messenger('optical', self.mock_transient, dummy_model)
            # Check info was logged
            info_calls = [call for call in mock_logger.info.call_args_list
                         if 'Built likelihood' in str(call)]
            self.assertTrue(len(info_calls) > 0)


if __name__ == '__main__':
    unittest.main()
