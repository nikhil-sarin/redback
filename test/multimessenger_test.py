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

        # Check line amplitude
        self.assertAlmostEqual(np.max(line_profile), line_flux, places=20)

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

        from redback.transient.transient import Spectrum
        self.spectrum = Spectrum(
            angstroms=self.wavelengths,
            flux_density=self.spec_flux,
            flux_density_err=self.spec_flux_err,
            time='3 days',
            name='test_spectrum'
        )

    def test_photometry_and_spectrum_different_data_types(self):
        """Test that photometry and spectrum are different data types"""
        from redback.transient.transient import Spectrum

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


if __name__ == '__main__':
    unittest.main()
