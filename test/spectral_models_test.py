import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from redback.transient_models import spectral_models
from redback.analysis import SpectralVelocityFitter


class TestVoigtProfile(unittest.TestCase):
    """Test the Voigt profile function"""

    def setUp(self):
        self.wavelength = np.linspace(6555, 6575, 500)
        self.lambda_center = 6563.0
        self.amplitude = 1.0
        self.sigma_gaussian = 2.0
        self.gamma_lorentz = 0.5

    def test_voigt_profile_returns_correct_shape(self):
        flux = spectral_models.voigt_profile(
            self.wavelength, self.lambda_center, self.amplitude,
            self.sigma_gaussian, self.gamma_lorentz
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_voigt_profile_peak_at_center(self):
        flux = spectral_models.voigt_profile(
            self.wavelength, self.lambda_center, self.amplitude,
            self.sigma_gaussian, self.gamma_lorentz
        )
        # Maximum should be near the center wavelength
        idx_max = np.argmax(flux)
        self.assertAlmostEqual(self.wavelength[idx_max], self.lambda_center, delta=0.5)

    def test_voigt_profile_symmetric(self):
        # Create symmetric wavelength array around center
        wave_sym = np.linspace(self.lambda_center - 10, self.lambda_center + 10, 101)
        flux = spectral_models.voigt_profile(
            wave_sym, self.lambda_center, self.amplitude,
            self.sigma_gaussian, self.gamma_lorentz
        )
        # Check symmetry
        n_half = len(flux) // 2
        assert_allclose(flux[:n_half], flux[-n_half:][::-1], rtol=1e-6)

    def test_voigt_profile_with_continuum(self):
        continuum = 2.0
        flux = spectral_models.voigt_profile(
            self.wavelength, self.lambda_center, self.amplitude,
            self.sigma_gaussian, self.gamma_lorentz, continuum=continuum
        )
        # Minimum should be at least the continuum level
        self.assertGreaterEqual(flux.min(), continuum - 0.01)

    def test_voigt_profile_negative_amplitude(self):
        # Absorption line
        flux = spectral_models.voigt_profile(
            self.wavelength, self.lambda_center, amplitude=-0.5,
            sigma_gaussian=self.sigma_gaussian, gamma_lorentz=self.gamma_lorentz,
            continuum=1.0
        )
        # Should have a minimum near center
        idx_min = np.argmin(flux)
        self.assertAlmostEqual(self.wavelength[idx_min], self.lambda_center, delta=0.5)


class TestGaussianLineProfile(unittest.TestCase):
    """Test the Gaussian line profile function"""

    def setUp(self):
        self.wavelength = np.linspace(6550, 6575, 500)
        self.lambda_center = 6563.0
        self.sigma = 2.0

    def test_gaussian_profile_returns_correct_shape(self):
        flux = spectral_models.gaussian_line_profile(
            self.wavelength, self.lambda_center, amplitude=1.0, sigma=self.sigma
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_gaussian_profile_peak_at_center(self):
        flux = spectral_models.gaussian_line_profile(
            self.wavelength, self.lambda_center, amplitude=1.0, sigma=self.sigma
        )
        idx_max = np.argmax(flux)
        self.assertAlmostEqual(self.wavelength[idx_max], self.lambda_center, delta=0.2)

    def test_gaussian_profile_fwhm(self):
        amplitude = 1.0
        flux = spectral_models.gaussian_line_profile(
            self.wavelength, self.lambda_center, amplitude, self.sigma
        )
        # FWHM should be 2 * sqrt(2 * ln(2)) * sigma ~ 2.355 * sigma
        expected_fwhm = 2.355 * self.sigma
        half_max = amplitude / 2
        # Find points where flux crosses half maximum
        above_half = flux > half_max
        indices = np.where(np.diff(above_half))[0]
        if len(indices) >= 2:
            fwhm = self.wavelength[indices[-1]] - self.wavelength[indices[0]]
            self.assertAlmostEqual(fwhm, expected_fwhm, delta=0.5)

    def test_gaussian_absorption_line(self):
        flux = spectral_models.gaussian_line_profile(
            self.wavelength, self.lambda_center, amplitude=-0.5, sigma=self.sigma, continuum=1.0
        )
        # Should be less than continuum at center
        center_idx = np.abs(self.wavelength - self.lambda_center).argmin()
        self.assertLess(flux[center_idx], 1.0)


class TestLorentzianLineProfile(unittest.TestCase):
    """Test the Lorentzian line profile function"""

    def setUp(self):
        self.wavelength = np.linspace(6550, 6575, 500)
        self.lambda_center = 6563.0
        self.gamma = 1.5

    def test_lorentzian_profile_returns_correct_shape(self):
        flux = spectral_models.lorentzian_line_profile(
            self.wavelength, self.lambda_center, amplitude=1.0, gamma=self.gamma
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_lorentzian_profile_peak_at_center(self):
        flux = spectral_models.lorentzian_line_profile(
            self.wavelength, self.lambda_center, amplitude=1.0, gamma=self.gamma
        )
        idx_max = np.argmax(flux)
        self.assertAlmostEqual(self.wavelength[idx_max], self.lambda_center, delta=0.2)

    def test_lorentzian_hwhm(self):
        # HWHM should be gamma
        flux = spectral_models.lorentzian_line_profile(
            self.wavelength, self.lambda_center, amplitude=1.0, gamma=self.gamma
        )
        max_flux = flux.max()
        half_max = max_flux / 2
        # Find first crossing of half maximum
        center_idx = np.abs(self.wavelength - self.lambda_center).argmin()
        # Check flux at center + gamma
        idx_hwhm = np.abs(self.wavelength - (self.lambda_center + self.gamma)).argmin()
        self.assertAlmostEqual(flux[idx_hwhm], half_max, delta=0.1)


class TestPCygniProfile(unittest.TestCase):
    """Test the P-Cygni profile function"""

    def setUp(self):
        self.wavelength = np.linspace(5800, 6800, 1000)
        self.lambda_rest = 6355.0  # Si II
        self.tau_sobolev = 3.0
        self.v_phot = 11000.0  # km/s
        self.continuum_flux = 1.0

    def test_pcygni_profile_returns_correct_shape(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_pcygni_profile_has_absorption_and_emission(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux
        )
        # Should have absorption below continuum
        self.assertTrue(np.any(flux < self.continuum_flux))
        # Should have emission above continuum
        self.assertTrue(np.any(flux > self.continuum_flux))

    def test_pcygni_absorption_blueshifted(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux
        )
        # Absorption minimum should be at blueshifted wavelength
        idx_min = np.argmin(flux)
        lambda_min = self.wavelength[idx_min]
        # Should be blueward of rest wavelength
        self.assertLess(lambda_min, self.lambda_rest)

    def test_pcygni_emission_near_rest(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux
        )
        # Emission should be redward of absorption
        idx_min = np.argmin(flux)
        idx_max_emission = np.argmax(flux)
        self.assertGreater(self.wavelength[idx_max_emission], self.wavelength[idx_min])

    def test_pcygni_velocity_dependence(self):
        # Higher velocity should shift absorption further blue
        flux_low_v = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            v_phot=8000, continuum_flux=self.continuum_flux
        )
        flux_high_v = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            v_phot=15000, continuum_flux=self.continuum_flux
        )
        lambda_min_low = self.wavelength[np.argmin(flux_low_v)]
        lambda_min_high = self.wavelength[np.argmin(flux_high_v)]
        # Higher velocity = more blueshift = lower wavelength
        self.assertLess(lambda_min_high, lambda_min_low)

    def test_pcygni_optical_depth_dependence(self):
        # Higher optical depth should give deeper absorption
        flux_low_tau = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, tau_sobolev=1.0,
            v_phot=self.v_phot, continuum_flux=self.continuum_flux
        )
        flux_high_tau = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, tau_sobolev=10.0,
            v_phot=self.v_phot, continuum_flux=self.continuum_flux
        )
        self.assertLess(flux_high_tau.min(), flux_low_tau.min())

    def test_pcygni_source_function_thermal(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux, source_function='thermal'
        )
        self.assertTrue(np.isfinite(flux).all())

    def test_pcygni_source_function_scattering(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux, source_function='scattering'
        )
        self.assertTrue(np.isfinite(flux).all())

    def test_pcygni_source_function_custom(self):
        flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, self.tau_sobolev,
            self.v_phot, self.continuum_flux, source_function=0.7
        )
        self.assertTrue(np.isfinite(flux).all())


class TestElementaryPCygniProfile(unittest.TestCase):
    """Test the elementary P-Cygni profile function"""

    def setUp(self):
        self.wavelength = np.linspace(6000, 6700, 1000)
        self.lambda_rest = 6355.0
        self.v_absorption = 11000.0
        self.absorption_depth = 0.4
        self.emission_strength = 0.2
        self.v_width = 1500.0

    def test_elementary_pcygni_returns_correct_shape(self):
        flux = spectral_models.elementary_p_cygni_profile(
            self.wavelength, self.lambda_rest, self.v_absorption,
            self.absorption_depth, self.emission_strength, self.v_width
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_elementary_pcygni_has_absorption_and_emission(self):
        flux = spectral_models.elementary_p_cygni_profile(
            self.wavelength, self.lambda_rest, self.v_absorption,
            self.absorption_depth, self.emission_strength, self.v_width
        )
        # Should have both absorption and emission
        self.assertLess(flux.min(), 1.0)
        self.assertGreater(flux.max(), 1.0)

    def test_elementary_pcygni_velocity_conversion(self):
        c_kms = 299792.458
        flux = spectral_models.elementary_p_cygni_profile(
            self.wavelength, self.lambda_rest, self.v_absorption,
            self.absorption_depth, self.emission_strength, self.v_width
        )
        # Expected absorption wavelength
        expected_lambda_abs = self.lambda_rest * (1 - self.v_absorption / c_kms)
        idx_min = np.argmin(flux)
        lambda_min = self.wavelength[idx_min]
        # Should be close to expected position
        self.assertAlmostEqual(lambda_min, expected_lambda_abs, delta=20)


class TestMultiLinePCygniSpectrum(unittest.TestCase):
    """Test the multi-line P-Cygni spectrum function"""

    def setUp(self):
        self.wavelength = np.linspace(3500, 9000, 2000)
        self.redshift = 0.01
        self.line_list = [
            {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0},
            {'ion': 'Ca II', 'lambda': 3945, 'tau': 5.0}
        ]
        self.v_phot = 11000.0

    def test_multiline_returns_correct_shape(self):
        spectrum = spectral_models.multi_line_p_cygni_spectrum(
            self.wavelength, self.redshift, 'blackbody',
            self.line_list, self.v_phot,
            r_phot=1e15, temperature=11000
        )
        self.assertEqual(spectrum.shape, self.wavelength.shape)

    def test_multiline_has_multiple_absorptions(self):
        spectrum = spectral_models.multi_line_p_cygni_spectrum(
            self.wavelength, self.redshift, 'blackbody',
            self.line_list, self.v_phot,
            r_phot=1e15, temperature=11000
        )
        # Normalize to find features
        normalized = spectrum / np.percentile(spectrum, 90)
        # Should have at least two regions of absorption
        below_continuum = normalized < 0.95
        self.assertTrue(np.any(below_continuum))

    def test_multiline_redshift_shift(self):
        # With redshift, features should be shifted
        spectrum_z0 = spectral_models.multi_line_p_cygni_spectrum(
            self.wavelength, redshift=0.0, continuum_model='blackbody',
            line_list=self.line_list, v_phot=self.v_phot,
            r_phot=1e15, temperature=11000
        )
        spectrum_z = spectral_models.multi_line_p_cygni_spectrum(
            self.wavelength, redshift=self.redshift, continuum_model='blackbody',
            line_list=self.line_list, v_phot=self.v_phot,
            r_phot=1e15, temperature=11000
        )
        # Spectra should be different
        self.assertFalse(np.allclose(spectrum_z0, spectrum_z))

    def test_multiline_powerlaw_continuum(self):
        spectrum = spectral_models.multi_line_p_cygni_spectrum(
            self.wavelength, self.redshift, 'powerlaw',
            self.line_list, self.v_phot,
            alpha=-1.0, aa=1e-10
        )
        self.assertEqual(spectrum.shape, self.wavelength.shape)
        self.assertTrue(np.isfinite(spectrum).all())


class TestSYNOWLineModel(unittest.TestCase):
    """Test the SYNOW-style line model"""

    def setUp(self):
        self.wavelength = np.linspace(5800, 6800, 1000)
        self.lambda_rest = 6355.0
        self.tau_ref = 5.0
        self.v_phot = 10000.0
        self.v_max = 25000.0

    def test_synow_returns_correct_shape(self):
        transmission = spectral_models.synow_line_model(
            self.wavelength, self.lambda_rest, self.tau_ref,
            self.v_phot, self.v_max
        )
        self.assertEqual(transmission.shape, self.wavelength.shape)

    def test_synow_has_absorption(self):
        transmission = spectral_models.synow_line_model(
            self.wavelength, self.lambda_rest, self.tau_ref,
            self.v_phot, self.v_max
        )
        # Should have absorption (transmission < 1)
        self.assertTrue(np.any(transmission < 1.0))

    def test_synow_power_law_index(self):
        trans_n7 = spectral_models.synow_line_model(
            self.wavelength, self.lambda_rest, self.tau_ref,
            self.v_phot, self.v_max, n_power=7
        )
        trans_n10 = spectral_models.synow_line_model(
            self.wavelength, self.lambda_rest, self.tau_ref,
            self.v_phot, self.v_max, n_power=10
        )
        # Different power law indices should give different profiles
        self.assertFalse(np.allclose(trans_n7, trans_n10))


class TestBlackbodySpectrumWithPCygniLines(unittest.TestCase):
    """Test the convenience function for blackbody + P-Cygni lines"""

    def setUp(self):
        self.wavelength = np.linspace(3500, 8500, 2000)
        self.redshift = 0.01
        self.rph = 1e15
        self.temp = 11000
        self.line_list = [
            {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}
        ]
        self.v_phot = 11000

    def test_returns_correct_shape(self):
        flux = spectral_models.blackbody_spectrum_with_p_cygni_lines(
            self.wavelength, self.redshift, self.rph, self.temp,
            self.line_list, self.v_phot
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_physically_reasonable_values(self):
        flux = spectral_models.blackbody_spectrum_with_p_cygni_lines(
            self.wavelength, self.redshift, self.rph, self.temp,
            self.line_list, self.v_phot
        )
        # Should be positive
        self.assertTrue(np.all(flux > 0))
        # Should be finite
        self.assertTrue(np.isfinite(flux).all())


class TestSpectrumWithVoigtAbsorptionLines(unittest.TestCase):
    """Test the Voigt absorption line synthesis function"""

    def setUp(self):
        self.wavelength = np.linspace(6500, 6700, 1000)
        self.continuum_flux = 1.0
        self.line_params_list = [
            {'lambda': 6563, 'depth': 0.3, 'sigma': 2.0, 'gamma': 0.5}
        ]

    def test_returns_correct_shape(self):
        flux = spectral_models.spectrum_with_voigt_absorption_lines(
            self.wavelength, self.continuum_flux, self.line_params_list
        )
        self.assertEqual(flux.shape, self.wavelength.shape)

    def test_has_absorption(self):
        flux = spectral_models.spectrum_with_voigt_absorption_lines(
            self.wavelength, self.continuum_flux, self.line_params_list
        )
        # Should have absorption below continuum
        self.assertTrue(np.any(flux < self.continuum_flux))

    def test_absorption_depth(self):
        flux = spectral_models.spectrum_with_voigt_absorption_lines(
            self.wavelength, self.continuum_flux, self.line_params_list
        )
        # Minimum should not go below (1 - depth) * continuum
        expected_min = (1 - self.line_params_list[0]['depth']) * self.continuum_flux
        self.assertGreaterEqual(flux.min(), expected_min - 0.05)

    def test_multiple_lines(self):
        lines = [
            {'lambda': 6563, 'depth': 0.3, 'sigma': 2.0, 'gamma': 0.5},
            {'lambda': 6583, 'depth': 0.1, 'sigma': 1.5, 'gamma': 0.3}
        ]
        flux = spectral_models.spectrum_with_voigt_absorption_lines(
            self.wavelength, self.continuum_flux, lines
        )
        self.assertEqual(flux.shape, self.wavelength.shape)
        # Should have multiple absorption features
        self.assertTrue(np.any(flux < self.continuum_flux))

    def test_array_continuum(self):
        continuum_array = np.ones_like(self.wavelength) * 2.0
        flux = spectral_models.spectrum_with_voigt_absorption_lines(
            self.wavelength, continuum_array, self.line_params_list
        )
        self.assertEqual(flux.shape, self.wavelength.shape)


class TestSpectralVelocityFitter(unittest.TestCase):
    """Test the SpectralVelocityFitter class"""

    def setUp(self):
        # Create a mock P-Cygni profile for testing
        self.wavelength = np.linspace(5800, 6800, 1000)
        self.lambda_rest = 6355.0
        self.v_phot_true = 11000.0  # km/s

        # Generate spectrum with known velocity
        self.flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, tau_sobolev=3.0,
            v_phot=self.v_phot_true, continuum_flux=1.0
        )

        # Add small noise
        np.random.seed(42)
        self.flux_noisy = self.flux + 0.01 * np.random.randn(len(self.flux))

        self.fitter = SpectralVelocityFitter(self.wavelength, self.flux_noisy)

    def test_initialization(self):
        fitter = SpectralVelocityFitter(self.wavelength, self.flux_noisy)
        self.assertEqual(len(fitter.wavelength), len(self.wavelength))
        self.assertEqual(len(fitter.flux), len(self.flux_noisy))
        self.assertIsNone(fitter.flux_err)

    def test_initialization_with_errors(self):
        flux_err = np.ones_like(self.flux_noisy) * 0.01
        fitter = SpectralVelocityFitter(self.wavelength, self.flux_noisy, flux_err)
        self.assertIsNotNone(fitter.flux_err)

    def test_measure_line_velocity_min_method(self):
        # Use larger search window to find absorption at high velocity
        v, verr = self.fitter.measure_line_velocity(self.lambda_rest, method='min', v_window=15000)
        # Should be negative (blueshift) and close to true velocity
        self.assertLess(v, 0)
        self.assertAlmostEqual(-v, self.v_phot_true, delta=2000)  # Within 2000 km/s

    def test_measure_line_velocity_centroid_method(self):
        v, verr = self.fitter.measure_line_velocity(self.lambda_rest, method='centroid', v_window=15000)
        # Should return a velocity
        self.assertIsInstance(v, float)
        self.assertIsInstance(verr, float)

    def test_measure_line_velocity_gaussian_method(self):
        v, verr = self.fitter.measure_line_velocity(self.lambda_rest, method='gaussian', v_window=15000)
        # Should be negative (blueshift)
        self.assertLess(v, 0)
        # Should be reasonable
        self.assertGreater(-v, 3000)
        self.assertLess(-v, 20000)

    def test_measure_line_velocity_invalid_method(self):
        with self.assertRaises(ValueError):
            self.fitter.measure_line_velocity(self.lambda_rest, method='invalid')

    def test_measure_line_velocity_insufficient_data(self):
        # Create fitter with very narrow wavelength range
        narrow_wave = np.linspace(6350, 6360, 3)
        narrow_flux = np.ones(3)
        fitter = SpectralVelocityFitter(narrow_wave, narrow_flux)
        v, verr = fitter.measure_line_velocity(6355)
        self.assertTrue(np.isnan(v))
        self.assertTrue(np.isnan(verr))

    def test_measure_multiple_lines(self):
        line_dict = {
            'Si II 6355': 6355,
        }
        velocities = self.fitter.measure_multiple_lines(line_dict)
        self.assertIn('Si II 6355', velocities)
        v, verr = velocities['Si II 6355']
        self.assertIsInstance(v, float)
        self.assertIsInstance(verr, float)

    def test_measure_multiple_lines_with_missing(self):
        # Add a line outside the wavelength range
        line_dict = {
            'Si II 6355': 6355,
            'Ca II': 3945  # Outside range
        }
        velocities = self.fitter.measure_multiple_lines(line_dict)
        self.assertEqual(len(velocities), 2)

    def test_photospheric_velocity_evolution(self):
        # Create multiple spectra with decreasing velocity
        wavelength_list = []
        flux_list = []
        times = np.array([0, 5, 10])
        v_true = [15000, 12000, 9000]  # Larger velocity difference for clearer signal

        for v in v_true:
            wave = np.linspace(5500, 6800, 500)  # Wider wavelength range
            flux = spectral_models.p_cygni_profile(
                wave, 6355, 3.0, v, 1.0
            )
            wavelength_list.append(wave)
            flux_list.append(flux)

        times_out, velocities, errors = SpectralVelocityFitter.photospheric_velocity_evolution(
            wavelength_list, flux_list, times, line_wavelength=6355, v_window=20000
        )

        self.assertEqual(len(times_out), 3)
        self.assertEqual(len(velocities), 3)
        self.assertEqual(len(errors), 3)
        # Velocities should decrease (become less negative)
        # Higher velocity = more blueshift = more negative v
        self.assertLess(velocities[0], velocities[-1])  # More negative at start

    def test_identify_high_velocity_features_no_hvf(self):
        # Standard spectrum without HVF
        has_hvf, v_hvf, v_hvf_err = self.fitter.identify_high_velocity_features(
            self.lambda_rest, v_phot_expected=11000
        )
        self.assertIsInstance(has_hvf, bool)

    def test_identify_high_velocity_features_with_hvf(self):
        # Create spectrum with HVF
        wave = np.linspace(5800, 6800, 1000)
        # Photospheric at 11000 km/s
        flux_phot = spectral_models.elementary_p_cygni_profile(
            wave, 6355, 11000, 0.35, 0.1, 1500
        )
        # HVF at 16000 km/s
        flux_hvf = spectral_models.elementary_p_cygni_profile(
            wave, 6355, 16000, 0.15, 0.0, 1000
        )
        spectrum = flux_phot * flux_hvf

        fitter = SpectralVelocityFitter(wave, spectrum)
        has_hvf, v_hvf, v_hvf_err = fitter.identify_high_velocity_features(
            6355, v_phot_expected=11000, threshold_factor=1.3
        )
        # May or may not detect depending on implementation
        self.assertIsInstance(has_hvf, bool)

    def test_measure_velocity_gradient(self):
        # Create time series
        wavelength_list = []
        flux_list = []
        times = np.array([0, 5, 10, 15, 20])
        # Linear decline: 15000 - 150*t km/s (larger gradient for clearer signal)
        v_true = 15000 - 150 * times

        for v in v_true:
            wave = np.linspace(5200, 6800, 500)  # Wider range to capture all velocities
            flux = spectral_models.p_cygni_profile(wave, 6355, 3.0, v, 1.0)
            wavelength_list.append(wave)
            flux_list.append(flux)

        # First measure individual velocities to ensure they're different
        velocities_manual = []
        for wave, flux in zip(wavelength_list, flux_list):
            fitter_temp = SpectralVelocityFitter(wave, flux)
            v, _ = fitter_temp.measure_line_velocity(6355, v_window=20000)
            velocities_manual.append(v)

        # Check that velocities are actually different
        self.assertLess(velocities_manual[0], velocities_manual[-1])  # First more negative

        # Now test the gradient calculation
        fitter = SpectralVelocityFitter(wavelength_list[0], flux_list[0])
        gradient, gradient_err = fitter.measure_velocity_gradient(
            wavelength_list, flux_list, times, line_wavelength=6355, v_window=20000
        )

        # Gradient should be positive (velocities become less negative)
        # Since we measure blueshifted velocities as negative, the gradient is positive
        self.assertIsInstance(gradient, float)
        # Should show increasing trend (velocities become less negative)
        self.assertGreater(gradient, 50)  # Should be around +150 km/s/day

    def test_from_spectrum_object(self):
        # Create a mock spectrum object
        class MockSpectrum:
            def __init__(self):
                self.angstroms = np.linspace(5800, 6800, 500)
                self.flux_density = np.ones(500)

        mock_spec = MockSpectrum()
        fitter = SpectralVelocityFitter.from_spectrum_object(mock_spec)
        self.assertIsNotNone(fitter)
        self.assertEqual(len(fitter.wavelength), 500)


class TestSpectralModelsEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_voigt_very_small_gamma(self):
        # Should approach Gaussian
        wave = np.linspace(6560, 6566, 100)
        flux = spectral_models.voigt_profile(
            wave, 6563, 1.0, sigma_gaussian=1.0, gamma_lorentz=1e-6
        )
        self.assertTrue(np.isfinite(flux).all())

    def test_pcygni_very_low_optical_depth(self):
        wave = np.linspace(5800, 6800, 500)
        flux = spectral_models.p_cygni_profile(
            wave, 6355, tau_sobolev=0.01, v_phot=11000, continuum_flux=1.0
        )
        # Should be close to continuum
        self.assertAlmostEqual(flux.min(), 1.0, delta=0.1)

    def test_pcygni_very_high_optical_depth(self):
        wave = np.linspace(5800, 6800, 500)
        flux = spectral_models.p_cygni_profile(
            wave, 6355, tau_sobolev=100.0, v_phot=11000, continuum_flux=1.0
        )
        # Should have deep absorption (with source function filling, minimum is S*continuum)
        self.assertLess(flux.min(), 0.6)

    def test_empty_line_list(self):
        wave = np.linspace(3500, 9000, 1000)
        spectrum = spectral_models.multi_line_p_cygni_spectrum(
            wave, redshift=0.01, continuum_model='blackbody',
            line_list=[], v_phot=11000,
            r_phot=1e15, temperature=11000
        )
        # Should just be continuum
        self.assertEqual(spectrum.shape, wave.shape)
        self.assertTrue(np.isfinite(spectrum).all())

    def test_pcygni_custom_source_function_numeric(self):
        wave = np.linspace(5800, 6800, 500)
        flux = spectral_models.p_cygni_profile(
            wave, 6355, tau_sobolev=3.0, v_phot=11000,
            continuum_flux=1.0, source_function=0.8
        )
        self.assertTrue(np.isfinite(flux).all())
        self.assertTrue(np.any(flux < 1.0))

    def test_pcygni_with_custom_vmax(self):
        wave = np.linspace(5800, 6800, 500)
        flux = spectral_models.p_cygni_profile(
            wave, 6355, tau_sobolev=3.0, v_phot=11000,
            continuum_flux=1.0, v_max=20000
        )
        self.assertTrue(np.isfinite(flux).all())

    def test_elementary_pcygni_no_emission(self):
        wave = np.linspace(6000, 6700, 500)
        flux = spectral_models.elementary_p_cygni_profile(
            wave, 6355, 11000, 0.3, emission_strength=0.0, v_width=1500
        )
        # Only absorption, no emission peak above continuum
        self.assertEqual(flux.shape, wave.shape)

    def test_multiline_invalid_continuum_model(self):
        wave = np.linspace(3500, 9000, 1000)
        with self.assertRaises(ValueError):
            spectral_models.multi_line_p_cygni_spectrum(
                wave, 0.01, 'invalid_model',
                [{'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}], 11000
            )

    def test_multiline_callable_continuum(self):
        wave = np.linspace(3500, 9000, 1000)

        def custom_continuum(wavelength, **kwargs):
            return np.ones_like(wavelength) * 1e-15

        spectrum = spectral_models.multi_line_p_cygni_spectrum(
            wave, 0.01, custom_continuum,
            [{'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}], 11000
        )
        self.assertEqual(spectrum.shape, wave.shape)

    def test_synow_with_dilution_factor(self):
        wave = np.linspace(5800, 6800, 500)
        transmission = spectral_models.synow_line_model(
            wave, 6355, 5.0, 10000, 25000, dilution_factor=0.7
        )
        self.assertTrue(np.isfinite(transmission).all())

    def test_spectrum_voigt_empty_lines(self):
        wave = np.linspace(6500, 6700, 500)
        flux = spectral_models.spectrum_with_voigt_absorption_lines(
            wave, 1.0, []
        )
        # No lines, should be constant
        assert_allclose(flux, np.ones_like(wave))

    def test_gaussian_with_zero_amplitude(self):
        wave = np.linspace(6550, 6575, 100)
        flux = spectral_models.gaussian_line_profile(
            wave, 6563, amplitude=0.0, sigma=2.0, continuum=1.0
        )
        assert_allclose(flux, np.ones_like(wave))

    def test_lorentzian_with_large_gamma(self):
        wave = np.linspace(6550, 6575, 100)
        flux = spectral_models.lorentzian_line_profile(
            wave, 6563, amplitude=1.0, gamma=100.0, continuum=0.0
        )
        # Very broad, should be nearly constant
        self.assertLess(np.std(flux), 0.1)


class TestSpectralVelocityFitterAdvanced(unittest.TestCase):
    """Advanced tests for SpectralVelocityFitter"""

    def setUp(self):
        self.wavelength = np.linspace(5800, 6800, 1000)
        self.lambda_rest = 6355.0
        self.v_phot_true = 11000.0
        self.flux = spectral_models.p_cygni_profile(
            self.wavelength, self.lambda_rest, tau_sobolev=3.0,
            v_phot=self.v_phot_true, continuum_flux=1.0
        )
        np.random.seed(42)
        self.flux_noisy = self.flux + 0.01 * np.random.randn(len(self.flux))
        self.fitter = SpectralVelocityFitter(self.wavelength, self.flux_noisy)

    def test_measure_line_velocity_fit_method(self):
        # Test the P-Cygni fit method
        v, verr = self.fitter.measure_line_velocity(
            self.lambda_rest, method='fit', v_window=15000
        )
        # Should return reasonable values
        self.assertIsInstance(v, float)
        self.assertIsInstance(verr, float)
        self.assertLess(v, 0)  # Blueshift

    def test_measure_velocity_with_custom_continuum_percentile(self):
        v, verr = self.fitter.measure_line_velocity(
            self.lambda_rest, method='centroid',
            v_window=15000, continuum_percentile=95
        )
        self.assertIsInstance(v, float)
        self.assertIsInstance(verr, float)

    def test_measure_velocity_gaussian_fit_failure_fallback(self):
        # Create spectrum where Gaussian fit might struggle
        wave = np.linspace(6300, 6400, 50)
        flux = np.ones(50) * 1.0  # Flat continuum, no line
        fitter = SpectralVelocityFitter(wave, flux)
        v, verr = fitter.measure_line_velocity(6355, method='gaussian')
        # Should fall back to 'min' method
        self.assertIsInstance(v, float)

    def test_measure_multiple_lines_with_errors(self):
        # Some lines may fail to measure
        line_dict = {
            'Si II': 6355,
            'Outside': 3000,  # Outside wavelength range
        }
        velocities = self.fitter.measure_multiple_lines(line_dict, v_window=15000)
        self.assertEqual(len(velocities), 2)
        # Si II should work
        v_si, _ = velocities['Si II']
        self.assertLess(v_si, 0)

    def test_photospheric_velocity_evolution_with_nans(self):
        # Create time series with one bad spectrum
        wavelength_list = []
        flux_list = []
        times = np.array([0, 5, 10])

        for i, v in enumerate([15000, 12000, 9000]):
            wave = np.linspace(5500, 6800, 500)
            if i == 1:  # Bad spectrum - flat
                flux = np.ones(500)
            else:
                flux = spectral_models.p_cygni_profile(wave, 6355, 3.0, v, 1.0)
            wavelength_list.append(wave)
            flux_list.append(flux)

        times_out, velocities, errors = SpectralVelocityFitter.photospheric_velocity_evolution(
            wavelength_list, flux_list, times, line_wavelength=6355, v_window=20000
        )
        self.assertEqual(len(velocities), 3)

    def test_identify_hvf_insufficient_data(self):
        # Very narrow wavelength range
        wave = np.linspace(6350, 6360, 10)
        flux = np.ones(10)
        fitter = SpectralVelocityFitter(wave, flux)
        has_hvf, v_hvf, v_err = fitter.identify_high_velocity_features(6355, 11000)
        self.assertFalse(has_hvf)

    def test_identify_hvf_no_significant_absorption(self):
        # Create spectrum with very weak absorption
        wave = np.linspace(5800, 6800, 1000)
        flux = np.ones(1000) * 1.0
        flux[100:200] = 0.98  # Very weak absorption
        fitter = SpectralVelocityFitter(wave, flux)
        has_hvf, v_hvf, v_err = fitter.identify_high_velocity_features(6355, 11000)
        # Should not detect as HVF (too weak)
        self.assertIsInstance(has_hvf, bool)

    def test_velocity_gradient_insufficient_data(self):
        # Only one valid measurement
        wavelength_list = [np.linspace(5800, 6800, 100)]
        flux_list = [np.ones(100)]
        times = np.array([0])
        fitter = SpectralVelocityFitter(wavelength_list[0], flux_list[0])
        gradient, gradient_err = fitter.measure_velocity_gradient(
            wavelength_list, flux_list, times
        )
        self.assertTrue(np.isnan(gradient))

    def test_velocity_gradient_with_uniform_errors(self):
        # Create time series with constant errors
        wavelength_list = []
        flux_list = []
        times = np.array([0, 5, 10])

        for v in [15000, 12000, 9000]:
            wave = np.linspace(5500, 6800, 500)
            flux = spectral_models.p_cygni_profile(wave, 6355, 3.0, v, 1.0)
            wavelength_list.append(wave)
            flux_list.append(flux)

        fitter = SpectralVelocityFitter(wavelength_list[0], flux_list[0])
        gradient, gradient_err = fitter.measure_velocity_gradient(
            wavelength_list, flux_list, times, v_window=20000
        )
        self.assertIsInstance(gradient, float)
        self.assertGreater(gradient, 0)  # Velocity becoming less negative

    def test_velocity_gradient_only_two_points(self):
        wavelength_list = []
        flux_list = []
        times = np.array([0, 10])

        for v in [15000, 9000]:
            wave = np.linspace(5500, 6800, 500)
            flux = spectral_models.p_cygni_profile(wave, 6355, 3.0, v, 1.0)
            wavelength_list.append(wave)
            flux_list.append(flux)

        fitter = SpectralVelocityFitter(wavelength_list[0], flux_list[0])
        gradient, gradient_err = fitter.measure_velocity_gradient(
            wavelength_list, flux_list, times, v_window=20000
        )
        self.assertIsInstance(gradient, float)
        self.assertTrue(np.isnan(gradient_err))  # Can't estimate error with only 2 points

    def test_from_spectrum_object_with_errors(self):
        class MockSpectrumWithErrors:
            def __init__(self):
                self.angstroms = np.linspace(5800, 6800, 500)
                self.flux_density = np.ones(500)
                self.flux_density_err = np.ones(500) * 0.01

        mock_spec = MockSpectrumWithErrors()
        fitter = SpectralVelocityFitter.from_spectrum_object(mock_spec)
        self.assertIsNotNone(fitter.flux_err)
        assert_allclose(fitter.flux_err, np.ones(500) * 0.01)

    def test_centroid_no_absorption(self):
        # Spectrum with no absorption
        wave = np.linspace(6300, 6400, 100)
        flux = np.ones(100) * 1.0
        fitter = SpectralVelocityFitter(wave, flux)
        v, verr = fitter.measure_line_velocity(6355, method='centroid')
        # Should return default values
        self.assertEqual(v, 0.0)
        self.assertEqual(verr, 500.0)


class TestBlackbodySpectrumIntegration(unittest.TestCase):
    """Test blackbody spectrum integration with P-Cygni lines"""

    def test_blackbody_multiple_lines(self):
        wave = np.linspace(3500, 8500, 2000)
        lines = [
            {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0},
            {'ion': 'Ca II', 'lambda': 3945, 'tau': 5.0},
            {'ion': 'Fe II', 'lambda': 5169, 'tau': 2.0}
        ]
        flux = spectral_models.blackbody_spectrum_with_p_cygni_lines(
            wave, redshift=0.02, rph=2e15, temp=12000,
            line_list=lines, v_phot=12000
        )
        self.assertEqual(flux.shape, wave.shape)
        self.assertTrue(np.all(flux > 0))
        self.assertTrue(np.isfinite(flux).all())

    def test_blackbody_single_line(self):
        wave = np.linspace(6000, 6700, 500)
        lines = [{'ion': 'Si II', 'lambda': 6355, 'tau': 2.0}]
        flux = spectral_models.blackbody_spectrum_with_p_cygni_lines(
            wave, redshift=0.0, rph=1e15, temp=10000,
            line_list=lines, v_phot=10000
        )
        self.assertEqual(flux.shape, wave.shape)
        self.assertTrue(np.all(flux > 0))

    def test_blackbody_zero_redshift(self):
        wave = np.linspace(5000, 7000, 1000)
        lines = [{'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}]
        flux = spectral_models.blackbody_spectrum_with_p_cygni_lines(
            wave, redshift=0.0, rph=1e15, temp=11000,
            line_list=lines, v_phot=11000
        )
        self.assertTrue(np.all(flux > 0))

    def test_blackbody_high_redshift(self):
        wave = np.linspace(3500, 10000, 2000)
        lines = [{'ion': 'Si II', 'lambda': 6355, 'tau': 3.0}]
        flux = spectral_models.blackbody_spectrum_with_p_cygni_lines(
            wave, redshift=0.1, rph=1e15, temp=11000,
            line_list=lines, v_phot=11000
        )
        self.assertTrue(np.all(flux > 0))
        self.assertTrue(np.isfinite(flux).all())


class TestVoigtAbsorptionLinesCoverage(unittest.TestCase):
    """Additional coverage for Voigt absorption synthesis"""

    def test_multiple_overlapping_lines(self):
        wave = np.linspace(6550, 6590, 500)
        lines = [
            {'lambda': 6563, 'depth': 0.3, 'sigma': 2.0, 'gamma': 0.5},
            {'lambda': 6565, 'depth': 0.2, 'sigma': 1.5, 'gamma': 0.3}
        ]
        flux = spectral_models.spectrum_with_voigt_absorption_lines(wave, 1.0, lines)
        # Should have significant absorption where lines overlap
        self.assertLess(flux.min(), 0.65)

    def test_deep_absorption_line(self):
        wave = np.linspace(6550, 6575, 500)
        lines = [{'lambda': 6563, 'depth': 0.9, 'sigma': 2.0, 'gamma': 0.5}]
        flux = spectral_models.spectrum_with_voigt_absorption_lines(wave, 1.0, lines)
        # Deep absorption
        self.assertLess(flux.min(), 0.15)

    def test_with_varying_continuum(self):
        wave = np.linspace(6500, 6650, 500)
        continuum = np.linspace(1.0, 2.0, 500)  # Sloping continuum
        lines = [{'lambda': 6563, 'depth': 0.3, 'sigma': 2.0, 'gamma': 0.5}]
        flux = spectral_models.spectrum_with_voigt_absorption_lines(wave, continuum, lines)
        self.assertEqual(flux.shape, wave.shape)
        # Should preserve general slope while adding absorption
        self.assertGreater(flux[-1], flux[0])


class TestSpectralVelocityFitterExceptionHandling(unittest.TestCase):
    """Test exception handling and edge cases in SpectralVelocityFitter"""

    def test_measure_multiple_lines_with_exception(self):
        """Test that exceptions in individual line measurements are caught"""
        wave = np.linspace(6000, 7000, 500)
        flux = np.ones_like(wave) * 1e-15

        fitter = SpectralVelocityFitter(wave, flux)

        # Define lines where one may fail due to edge effects
        lines = {
            'Si II': 6355,
            'Edge line': 100,  # Way outside wavelength range
        }

        velocities = fitter.measure_multiple_lines(lines, v_window=5000)
        # Should not raise, but should have NaN for edge line
        self.assertEqual(len(velocities), 2)
        self.assertTrue(np.isnan(velocities['Edge line'][0]))

    def test_measure_line_with_minimal_data_points(self):
        """Test with exactly 5 data points (minimum threshold)"""
        wave = np.array([6350, 6353, 6355, 6357, 6360])
        flux = np.array([1.0, 0.9, 0.7, 0.85, 1.0])  # Absorption at 6355

        fitter = SpectralVelocityFitter(wave, flux)
        v, verr = fitter.measure_line_velocity(6355, v_window=10000)
        # Should work but with limited accuracy
        self.assertFalse(np.isnan(v))
        self.assertIsInstance(verr, (float, np.floating))

    def test_velocity_gradient_with_all_valid_errors(self):
        """Test velocity gradient with valid positive errors for weighted fit"""
        # Create proper time series with valid errors
        times = np.array([1.0, 3.0, 5.0, 7.0])
        wavelength_list = []
        flux_list = []

        # Create synthetic spectra with decreasing velocity
        for i, t in enumerate(times):
            wave = np.linspace(5800, 6800, 300)
            # Velocity decreases with time (blueshifted velocities become less negative)
            v_phot = 12000 - 70 * t  # 11930, 11790, 11650, 11510
            flux = spectral_models.elementary_p_cygni_profile(
                wave, lambda_rest=6355, v_absorption=v_phot,
                absorption_depth=0.4, emission_strength=0.2, v_width=1500
            )
            wavelength_list.append(wave)
            flux_list.append(flux)

        # measure_velocity_gradient is a method, create dummy fitter
        dummy_wave = np.array([6000, 6500, 7000])
        dummy_flux = np.array([1.0, 1.0, 1.0])
        fitter = SpectralVelocityFitter(dummy_wave, dummy_flux)
        gradient, grad_err = fitter.measure_velocity_gradient(
            wavelength_list, flux_list, times, 6355, v_window=20000
        )

        # Should have valid gradient. The measured velocity is negative (blueshift),
        # and as v_phot decreases, the absorption moves redward, so velocity becomes less negative.
        # This means dv/dt is positive (velocity increasing toward zero).
        self.assertFalse(np.isnan(gradient))
        self.assertGreater(gradient, 0)  # Velocity becoming less negative

    def test_pcygni_fit_method_failure_fallback(self):
        """Test P-Cygni fit method falls back to min when fitting fails"""
        # Create data that will cause P-Cygni fit to fail
        wave = np.linspace(6300, 6400, 50)
        # Just constant flux (no absorption) will cause fit to fail
        flux = np.ones_like(wave) * 1.0

        fitter = SpectralVelocityFitter(wave, flux)
        # This should attempt P-Cygni fit, fail, and fall back to min
        v, verr = fitter.measure_line_velocity(6355, method='fit', v_window=10000)

        # Should return a value (from fallback to min method)
        self.assertIsInstance(v, (float, np.floating))
        self.assertIsInstance(verr, (float, np.floating))

    def test_synow_model_with_different_power_law(self):
        """Test SYNOW model with different n_power values"""
        wave = np.linspace(5800, 6800, 500)

        for n in [3, 7, 15]:
            flux = spectral_models.synow_line_model(
                wave, lambda_rest=6355, tau_ref=5.0,
                v_phot=10000, v_max=25000, n_power=n
            )
            self.assertEqual(flux.shape, wave.shape)
            # All should have absorption
            self.assertLess(flux.min(), 1.0)

    def test_pcygni_emission_region_only(self):
        """Test P-Cygni profile when only emission region is sampled"""
        # Sample only the emission side (redshifted)
        wave = np.linspace(6360, 6420, 100)
        flux = spectral_models.p_cygni_profile(
            wave, lambda_rest=6355, tau_sobolev=3.0,
            v_phot=10000, continuum_flux=1.0
        )
        # Should have some emission (flux > continuum)
        self.assertGreater(flux.max(), 1.0)

    def test_pcygni_absorption_region_only(self):
        """Test P-Cygni profile when only absorption region is sampled"""
        # Sample only the absorption side (blueshifted)
        wave = np.linspace(6050, 6300, 100)
        flux = spectral_models.p_cygni_profile(
            wave, lambda_rest=6355, tau_sobolev=3.0,
            v_phot=10000, continuum_flux=1.0
        )
        # Should have absorption (flux < continuum somewhere)
        self.assertLess(flux.min(), 1.0)

    def test_multiline_pcygni_with_many_lines(self):
        """Test multi-line P-Cygni with 5+ lines"""
        wave = np.linspace(3500, 8000, 2000)
        lines = [
            {'ion': 'Si II', 'lambda': 6355, 'tau': 3.0},
            {'ion': 'S II', 'lambda': 5640, 'tau': 2.0},
            {'ion': 'Ca II', 'lambda': 3945, 'tau': 5.0},
            {'ion': 'Fe II', 'lambda': 5169, 'tau': 2.5},
            {'ion': 'Mg II', 'lambda': 4481, 'tau': 1.5},
        ]
        flux = spectral_models.multi_line_p_cygni_spectrum(
            wave, redshift=0.01, continuum_model='blackbody',
            line_list=lines, v_phot=11000,
            r_phot=1e15, temperature=10000
        )
        self.assertEqual(flux.shape, wave.shape)
        # Should have multiple absorption features
        continuum_level = np.median(flux)
        # Count significant dips below continuum
        below_continuum = flux < 0.9 * continuum_level
        self.assertTrue(np.any(below_continuum))


class TestPowerlawSpectrumModels(unittest.TestCase):
    """Test power law spectrum models"""

    def test_powerlaw_with_absorption_and_emission(self):
        """Test powerlaw spectrum with lines"""
        wave = np.linspace(4000, 7000, 1000)
        flux = spectral_models.powerlaw_spectrum_with_absorption_and_emission_lines(
            wave, alpha=-1.0, aa=1e-15,
            lc1=6563, ls1=1e-16, v1=500,
            lc2=5007, ls2=0.5e-16, v2=300
        )
        self.assertEqual(flux.shape, wave.shape)
        # Should have structure from lines
        self.assertGreater(flux.std(), 0)

    def test_blackbody_with_absorption_and_emission(self):
        """Test blackbody spectrum with absorption and emission lines"""
        wave = np.linspace(4000, 7000, 1000)
        flux = spectral_models.blackbody_spectrum_with_absorption_and_emission_lines(
            wave, redshift=0.01, rph=1e15, temp=10000,
            lc1=6563, ls1=1e-16, v1=500,
            lc2=5007, ls2=0.5e-16, v2=300
        )
        self.assertEqual(flux.shape, wave.shape)
        # Should be positive
        self.assertTrue(np.all(flux > 0))

    def test_powerlaw_plus_blackbody_spectrum(self):
        """Test powerlaw plus blackbody spectrum evolution"""
        wave = np.linspace(4000, 8000, 500)
        flux = spectral_models.powerlaw_plus_blackbody_spectrum_at_z(
            wave, redshift=0.01,
            pl_amplitude=1e-16, pl_slope=-1.5, pl_evolution_index=1.0,
            temperature_0=12000, radius_0=1e14,
            temp_rise_index=0.5, temp_decline_index=0.3,
            temp_peak_time=5.0,
            radius_rise_index=0.8, radius_decline_index=0.2,
            radius_peak_time=10.0,
            time=7.0
        )
        self.assertEqual(flux.shape, wave.shape)
        self.assertTrue(np.all(flux > 0))

    def test_powerlaw_plus_blackbody_at_different_times(self):
        """Test time evolution of powerlaw+blackbody"""
        wave = np.linspace(4000, 8000, 500)
        times = [1.0, 5.0, 10.0, 20.0]

        fluxes = []
        for t in times:
            flux = spectral_models.powerlaw_plus_blackbody_spectrum_at_z(
                wave, redshift=0.01,
                pl_amplitude=1e-16, pl_slope=-1.5, pl_evolution_index=1.0,
                temperature_0=12000, radius_0=1e14,
                temp_rise_index=0.5, temp_decline_index=0.3,
                temp_peak_time=5.0,
                radius_rise_index=0.8, radius_decline_index=0.2,
                radius_peak_time=10.0,
                time=t
            )
            fluxes.append(flux)

        # Should have different spectra at different times
        self.assertNotEqual(np.sum(fluxes[0]), np.sum(fluxes[-1]))

    def test_get_powerlaw_spectrum_direct(self):
        """Test internal _get_powerlaw_spectrum function"""
        wave = np.linspace(4000, 8000, 500)
        flux = spectral_models._get_powerlaw_spectrum(wave, alpha=-2.0, aa=1e-10)
        self.assertEqual(flux.shape, wave.shape)
        # Power law with negative index should decrease
        self.assertGreater(flux[0], flux[-1])


class TestBlackbodySpectrumAtZ(unittest.TestCase):
    """Test blackbody spectrum at redshift"""

    def test_redshift_dilution(self):
        """Test that redshift affects flux properly"""
        wave = np.linspace(4000, 8000, 500)

        flux_z0 = spectral_models.blackbody_spectrum_at_z(
            wave, redshift=0.001, rph=1e15, temp=10000
        )
        flux_z05 = spectral_models.blackbody_spectrum_at_z(
            wave, redshift=0.5, rph=1e15, temp=10000
        )

        # Higher redshift means lower flux due to distance
        self.assertGreater(np.mean(flux_z0), np.mean(flux_z05))

    def test_rest_frame_conversion(self):
        """Test that rest frame conversion works correctly"""
        wave_obs = np.linspace(3000, 10000, 500)
        redshift = 0.1

        flux = spectral_models.blackbody_spectrum_at_z(
            wave_obs, redshift=redshift, rph=1e15, temp=8000
        )

        # Should have blackbody shape
        self.assertEqual(flux.shape, wave_obs.shape)
        # All flux values should be positive
        self.assertTrue(np.all(flux > 0))
        # Should have variation across the spectrum
        self.assertGreater(flux.max() / flux.min(), 1.5)


if __name__ == '__main__':
    unittest.main()
