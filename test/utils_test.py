import unittest
from collections import namedtuple
from unittest.mock import patch, MagicMock

import redback
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import redback.utils as utils
from redback.utils import UserCosmology, DataModeSwitch


class TestTimeConversion(unittest.TestCase):

    def setUp(self) -> None:
        self.mjd = 0
        self.jd = 2400000.5

        self.year = 1858
        self.month = 11
        self.day = 17

    def tearDown(self) -> None:
        del self.mjd
        del self.jd

        del self.year
        del self.month
        del self.day

    def test_mjd_to_jd(self):
        jd = redback.utils.mjd_to_jd(mjd=self.mjd)
        self.assertEqual(self.jd, jd)

    def test_jd_to_mjd(self):
        mjd = redback.utils.jd_to_mjd(jd=self.jd)
        self.assertEqual(self.mjd, mjd)

    def test_jd_to_date(self):
        year, month, day = redback.utils.jd_to_date(jd=self.jd)
        self.assertEqual(self.year, year)
        self.assertEqual(self.month, month)
        self.assertEqual(self.day, day)

    def test_mjd_to_date(self):
        year, month, day = redback.utils.mjd_to_date(mjd=self.mjd)
        self.assertEqual(self.year, year)
        self.assertEqual(self.month, month)
        self.assertEqual(self.day, day)

    def test_date_to_jd(self):
        jd = redback.utils.date_to_jd(year=self.year, month=self.month, day=self.day)
        self.assertEqual(self.jd, jd)

    def test_date_to_mjd(self):
        mjd = redback.utils.date_to_mjd(year=self.year, month=self.month, day=self.day)
        self.assertEqual(self.mjd, mjd)

class TestUserCosmology(unittest.TestCase):
    def setUp(self) -> None:
        self.default_dl = 100 * u.Mpc
        self.default_cosmology = UserCosmology(dl=self.default_dl, H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

    def tearDown(self) -> None:
        self.default_cosmology = None

    def test_initial_luminosity_distance(self):
        self.assertEqual(self.default_cosmology.luminosity_distance(None), self.default_dl)

    def test_update_luminosity_distance(self):
        new_dl = 200 * u.Mpc
        self.default_cosmology.set_luminosity_distance(new_dl)
        self.assertEqual(self.default_cosmology.luminosity_distance(None), new_dl)

    def test_luminosity_distance_ignores_redshift(self):
        redshift = 1.0
        self.assertEqual(self.default_cosmology.luminosity_distance(redshift), self.default_dl)

    def test_cosmology_inherits_from_flatlambda(self):
        self.assertIsInstance(self.default_cosmology, FlatLambdaCDM)

    def test_repr_overridden_as_flatlambda(self):
        output_repr = repr(self.default_cosmology)
        self.assertTrue("FlatLambdaCDM" in output_repr)
        self.assertFalse("UserCosmology" in output_repr)

class TestDataModeSwitch(unittest.TestCase):
    class DummyInstance:
        def __init__(self):
            self.data_mode = None

    def setUp(self) -> None:
        self.instance = self.DummyInstance()
        self.switch = DataModeSwitch(data_mode="mode1")

    def test_get_correct_data_mode(self):
        self.instance.data_mode = "mode1"
        self.assertTrue(self.switch.__get__(self.instance, None))

    def test_get_incorrect_data_mode(self):
        self.instance.data_mode = "mode2"
        self.assertFalse(self.switch.__get__(self.instance, None))

    def test_set_data_mode_to_true(self):
        self.switch.__set__(self.instance, True)
        self.assertEqual(self.instance.data_mode, "mode1")

    def test_set_data_mode_to_false(self):
        self.instance.data_mode = "mode1"
        self.switch.__set__(self.instance, False)
        self.assertIsNone(self.instance.data_mode)

class TestMetaDataAccessor(unittest.TestCase):
    def setUp(self) -> None:
        class MockClass:
            def __init__(self):
                self.meta_data = {}

        self.mock_instance = MockClass()

    def test_metadata_accessor_get_existing_property(self):
        self.mock_instance.meta_data = {'test_property': 'test_value'}
        accessor = utils.MetaDataAccessor(property_name='test_property')

        self.assertEqual(accessor.__get__(self.mock_instance, None), 'test_value')

    def test_metadata_accessor_get_non_existent_property_with_default(self):
        accessor = utils.MetaDataAccessor(property_name='non_existent_property', default='default_value')

        self.assertEqual(accessor.__get__(self.mock_instance, None), 'default_value')

    def test_metadata_accessor_get_non_existent_property_without_default(self):
        accessor = utils.MetaDataAccessor(property_name='non_existent_property')

        self.assertIsNone(accessor.__get__(self.mock_instance, None))

    def test_metadata_accessor_set_property(self):
        accessor = utils.MetaDataAccessor(property_name='new_property')
        accessor.__set__(self.mock_instance, 'new_value')

        self.assertEqual(self.mock_instance.meta_data['new_property'], 'new_value')

class TestCheckKwargsValidity(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_check_kwargs_validity_no_kwargs(self):
        kwargs = None
        result = utils.check_kwargs_validity(kwargs)
        self.assertIsNone(result)

    def test_check_kwargs_validity_missing_output_format(self):
        kwargs = {'frequency': [100, 200]}
        with self.assertRaises(ValueError) as context:
            utils.check_kwargs_validity(kwargs)
        self.assertIn("output_format must be specified", str(context.exception))

    def test_check_kwargs_validity_missing_frequency_and_bands(self):
        kwargs = {'output_format': 'flux_density'}
        with self.assertRaises(ValueError) as context:
            utils.check_kwargs_validity(kwargs)
        self.assertIn("frequency or bands must be specified in model_kwargs", str(context.exception))

    @patch('redback.utils.bands_to_frequency')
    def test_check_kwargs_validity_flux_density_adds_frequency(self, mock_bands_to_frequency):
        mock_bands_to_frequency.return_value = [150]
        kwargs = {'output_format': 'flux_density', 'bands': ['B']}
        result = utils.check_kwargs_validity(kwargs)
        self.assertIn('frequency', result)
        mock_bands_to_frequency.assert_called_with(['B'])

    @patch('redback.utils.frequency_to_bandname')
    def test_check_kwargs_validity_flux_adds_bands(self, mock_frequency_to_bandname):
        mock_frequency_to_bandname.return_value = ['B']
        kwargs = {'output_format': 'flux', 'frequency': [150]}
        result = utils.check_kwargs_validity(kwargs)
        self.assertIn('bands', result)
        mock_frequency_to_bandname.assert_called_with([150])

    def test_check_kwargs_validity_spectra_adds_frequency_array(self):
        kwargs = {'output_format': 'spectra', 'frequency': [150]}
        result = utils.check_kwargs_validity(kwargs)
        self.assertIn('frequency_array', result)
        self.assertTrue(len(result['frequency_array']) > 0)

    def test_check_kwargs_validity_no_changes_needed(self):
        kwargs = {'output_format': 'flux_density', 'frequency': [150]}
        result = utils.check_kwargs_validity(kwargs)
        self.assertEqual(result, kwargs)

class TestSomeUtility(unittest.TestCase):

    def test_calc_credible_intervals_valid_input(self):
        samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper, median = utils.calc_credible_intervals(samples, interval=0.8)
        self.assertAlmostEqual(lower, np.quantile(samples, 0.1))
        self.assertAlmostEqual(upper, np.quantile(samples, 0.9))
        self.assertEqual(median, np.median(samples))

    def test_calc_credible_intervals_raises_error_for_invalid_interval(self):
        samples = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            utils.calc_credible_intervals(samples, interval=-0.1)
        with self.assertRaises(ValueError):
            utils.calc_credible_intervals(samples, interval=1.5)

    def test_calc_credible_intervals_with_multi_dimensional_samples(self):
        samples = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        lower, upper, median = utils.calc_credible_intervals(samples, interval=0.9)
        np.testing.assert_array_almost_equal(lower, [1.45, 2.45, 3.45], decimal=2)
        np.testing.assert_array_almost_equal(upper, [9.55, 10.55, 11.55], decimal=2)
        np.testing.assert_array_equal(median, [5.5, 6.5, 7.5])

    def test_calc_credible_intervals_empty_samples(self):
        samples = np.array([])
        with self.assertRaises(IndexError):
            utils.calc_credible_intervals(samples, interval=0.9)

    def test_calc_credible_intervals_with_non_default_interval(self):
        samples = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        lower, upper, median = utils.calc_credible_intervals(samples, interval=0.5)
        self.assertAlmostEqual(lower, 5.5)
        self.assertAlmostEqual(upper, 14.5)
        self.assertEqual(median, 10)

    def test_valid_samples_with_default_quantiles(self):
        samples = np.array([1, 2, 3, 4, 5])
        result = utils.calc_one_dimensional_median_and_error_bar(samples)
        self.assertAlmostEqual(result.median, 3)
        self.assertAlmostEqual(result.lower, 1.36, places=2)
        self.assertAlmostEqual(result.upper, 1.36, places=2)
        self.assertEqual(result.string, r"$3.00_{-1.36}^{+1.36}$")

    def test_valid_samples_with_custom_quantiles(self):
        samples = np.array([1, 2, 3, 4, 5])
        quantiles = (0.1, 0.9)
        result = utils.calc_one_dimensional_median_and_error_bar(samples, quantiles=quantiles)
        self.assertAlmostEqual(result.median, 3)
        self.assertAlmostEqual(result.lower, 1.6, places=1)
        self.assertAlmostEqual(result.upper, 1.6, places=1)
        self.assertEqual(result.string, r"$3.00_{-1.60}^{+1.60}$")

    def test_invalid_quantiles_length(self):
        samples = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            utils.calc_one_dimensional_median_and_error_bar(samples, quantiles=(0.16, 0.5, 0.84))

    def test_empty_samples_array(self):
        samples = np.array([])
        with self.assertRaises(ValueError):
            utils.calc_one_dimensional_median_and_error_bar(samples)

    def test_different_formatting(self):
        samples = np.array([1, 2, 3, 4, 5])
        fmt = '.3f'
        result = utils.calc_one_dimensional_median_and_error_bar(samples, fmt=fmt)
        self.assertEqual(result.string, r"$3.000_{-1.360}^{+1.360}$")

    def test_find_nearest(self):
        array = np.array([1, 2, 3, 4, 5])
        value = 3.2
        expected_val, expected_idx = 3, 2
        val, idx = utils.find_nearest(array, value)
        self.assertEqual(val, expected_val)
        self.assertEqual(idx, expected_idx)

        value = 0.5
        expected_val, expected_idx = 1, 0
        val, idx = utils.find_nearest(array, value)
        self.assertEqual(val, expected_val)
        self.assertEqual(idx, expected_idx)

        value = 5.5
        expected_val, expected_idx = 5, 4
        val, idx = utils.find_nearest(array, value)
        self.assertEqual(val, expected_val)
        self.assertEqual(idx, expected_idx)

        array = [10, 20, 30]
        value = 21
        expected_val, expected_idx = 20, 1
        val, idx = utils.find_nearest(array, value) # Test with list input
        self.assertEqual(val, expected_val)
        self.assertEqual(idx, expected_idx)

    @patch('pandas.read_csv')
    def test_sncosmo_bandname_from_band(self, mock_read_csv):
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'bands': ['g', 'r', 'i', 'z'],
            'sncosmo_name': ['sdssg', 'sdssr', 'sdssi', 'sdssz']
        })
        mock_read_csv.return_value = mock_df

        # Test single band
        self.assertTrue(np.array_equal(utils.sncosmo_bandname_from_band('r'), np.array(['sdssr'])))
        # Test list of bands
        self.assertTrue(np.array_equal(utils.sncosmo_bandname_from_band(['g', 'i']), np.array(['sdssg', 'sdssi'])))
        # Test empty list
        self.assertTrue(np.array_equal(utils.sncosmo_bandname_from_band([]), np.array([])))
        # Test None input
        self.assertTrue(np.array_equal(utils.sncosmo_bandname_from_band(None), np.array([])))

        # Test unknown band with default 'softest' warning (appends 'r')
        with patch.object(utils.logger, 'info') as mock_log:
            res = utils.sncosmo_bandname_from_band(['g', 'unknown', 'i'], warning_style='softest')
            self.assertTrue(np.array_equal(res, np.array(['sdssg', 'r', 'sdssi'])))
            # No logging expected for 'softest'

        # Test unknown band with 'soft' warning (appends 'r', logs info)
        with patch.object(utils.logger, 'info') as mock_log:
            res = utils.sncosmo_bandname_from_band(['g', 'unknown', 'i'], warning_style='soft')
            self.assertTrue(np.array_equal(res, np.array(['sdssg', 'r', 'sdssi'])))
            self.assertGreaterEqual(mock_log.call_count, 1)  # Check if logger was called
            # Check if the specific error message was logged (optional, depends on exact logging format)
            # self.assertTrue(any("Band unknown is not defined" in call_args[0][0] for call_args in mock_log.call_args_list))

        # Test unknown band with 'hard' warning (raises KeyError)
        with self.assertRaises(KeyError) as cm:
            utils.sncosmo_bandname_from_band(['g', 'unknown', 'i'], warning_style='hard')
        self.assertIn("Band unknown is not defined", str(cm.exception))

    def test_calc_tfb(self):
        # Example values (replace with realistic ones if known)
        binding_energy_const = 0.8
        mbh_6 = 1.0  # 1 million solar masses
        stellar_mass = 1.0  # 1 solar mass
        expected_tfb = 58. * (3600. * 24.) * (1.0 ** 0.5) * (1.0 ** 0.2) * ((0.8 / 0.8) ** (-1.5))
        tfb = utils.calc_tfb(binding_energy_const, mbh_6, stellar_mass)
        self.assertAlmostEqual(tfb, expected_tfb)

    def test_abmag_to_flambda(self):
        mag = 20.0
        lam_eff = 5500.0  # Angstroms
        # Expected calculation:
        # lam_eff_cm = 5500e-8
        # f_nu = 10**(-0.4 * (20.0 + 48.6)) = 10**(-0.4 * 68.6) = 10**(-27.44)
        # f_lambda = f_nu * (c / lam_eff_cm^2) / 1e8
        # f_lambda = 10**(-27.44) * (2.99792458e10 / (5500e-8)**2) / 1e8
        expected_flambda = 10 ** (-0.4 * (mag + 48.6)) * (utils.speed_of_light / (lam_eff * 1e-8) ** 2) / 1e8
        flambda = utils.abmag_to_flambda(mag, lam_eff)
        self.assertAlmostEqual(flambda, expected_flambda)

    def test_flambda_err_from_mag_err(self):
        flux = 1e-15
        mag_err = 0.1
        expected_err = 0.4 * np.log(10) * flux * mag_err
        err = utils.flambda_err_from_mag_err(flux, mag_err)
        self.assertAlmostEqual(err, expected_err)

    def test_fnu_to_flambda(self):
        f_nu = 3.631e-20  # erg/s/cm^2/Hz (corresponds to AB mag 0 at 1 Hz, unrealistic freq but tests formula)
        wavelength_A = 5500.0  # Angstrom
        # expected = f_nu * c * 1e8 / wavelength_A**2
        expected = f_nu * utils.speed_of_light * 1e8 / wavelength_A ** 2
        flambda = utils.fnu_to_flambda(f_nu, wavelength_A)
        self.assertAlmostEqual(flambda, expected)

    def test_lambda_to_nu(self):
        wavelength = 5500.0  # Angstrom
        expected_nu = utils.speed_of_light_si / (wavelength * 1e-10)
        nu = utils.lambda_to_nu(wavelength)
        self.assertAlmostEqual(nu, expected_nu)

    def test_nu_to_lambda(self):
        frequency = 5.45e14  # Hz (approx for 5500 A)
        expected_lambda = 1e10 * (utils.speed_of_light_si / frequency)
        lam = utils.nu_to_lambda(frequency)
        self.assertAlmostEqual(lam, expected_lambda)

    def test_calc_kcorrected_properties(self):
        frequency = 1e15  # Hz (observer frame)
        redshift = 1.0
        time = 10.0  # days (observer frame)

        expected_freq_source = frequency * (1 + redshift)
        expected_time_source = time / (1 + redshift)

        k_freq, k_time = utils.calc_kcorrected_properties(frequency, redshift, time)
        self.assertAlmostEqual(k_freq, expected_freq_source)
        self.assertAlmostEqual(k_time, expected_time_source)

    @patch("redback.model_library.all_models_dict")
    def test_calculate_normalisation_single_frequency(self, mock_all_models_dict):
        """Test calculate_normalisation when unique_frequency is None."""

        # Mock models
        mock_model_1 = MagicMock(return_value=10)
        mock_model_2 = MagicMock(return_value=20)
        mock_all_models_dict.__getitem__.side_effect = lambda \
            model: mock_model_1 if model == "model_1" else mock_model_2

        model_1_dict = {"param_1": 1}
        model_2_dict = {"param_2": 2}
        tref = 1.0
        unique_frequency = None

        result = utils.calculate_normalisation(unique_frequency, "model_1", "model_2", tref, model_1_dict, model_2_dict)

        # Check results
        self.assertIsInstance(result, tuple)
        self.assertEqual(result.bolometric_luminosity, 2.0)

    @patch("redback.model_library.all_models_dict")
    def test_calculate_normalisation_multiple_frequencies(self, mock_all_models_dict):
        """
        Test calculate_normalisation with unique_frequency as an array of valid identifiers.
        """
        mock_model_1 = MagicMock(return_value=10)
        mock_model_2 = MagicMock(return_value=np.array([20, 30]))
        mock_all_models_dict.__getitem__.side_effect = lambda \
            model: mock_model_1 if model == "model_1" else mock_model_2

        # Use valid Python identifiers for the frequency array
        unique_frequency = np.array(["freq_100", "freq_200"])
        model_1_dict = {"param_1": 1}
        model_2_dict = {"param_2": 2}
        tref = 1.0

        result = utils.calculate_normalisation(unique_frequency, "model_1", "model_2", tref, model_1_dict, model_2_dict)

        self.assertIsInstance(result, tuple)
        self.assertEqual(result._fields, ("freq_100", "freq_200"))
        self.assertTrue(np.allclose(result, np.array([2.0, 3.0])))

    @patch("redback.model_library.all_models_dict")
    def test_calculate_normalisation_empty_parameters(self, mock_all_models_dict):
        """Test calculate_normalisation with empty model dictionaries."""

        # Mock models
        mock_model_1 = MagicMock(return_value=5)
        mock_model_2 = MagicMock(return_value=15)
        mock_all_models_dict.__getitem__.side_effect = lambda \
            model: mock_model_1 if model == "model_1" else mock_model_2

        result = utils.calculate_normalisation(None, "model_1", "model_2", 1.0, {}, {})

        # Check results
        self.assertIsInstance(result, tuple)
        self.assertEqual(result.bolometric_luminosity, 3.0)


class TestBuildSpectralFeatureList(unittest.TestCase):
    """Test suite for build_spectral_feature_list function"""

    def test_empty_kwargs_returns_defaults(self):
        """Test that empty kwargs returns default features when use_default_features=True"""
        result = utils.build_spectral_feature_list()
        expected = utils._get_default_sn_ia_features()
        assert result == expected

    def test_empty_kwargs_with_no_defaults(self):
        """Test that empty kwargs returns empty list when use_default_features=False"""
        result = utils.build_spectral_feature_list(use_default_features=False)
        assert result == []

    def test_single_feature_complete(self):
        """Test building a single complete feature"""
        kwargs = {
            'rest_wavelength_feature_1': 6355.0,
            'sigma_feature_1': 400.0,
            'amplitude_feature_1': -0.4,
            't_start_feature_1': 0,
            't_end_feature_1': 30
        }

        result = utils.build_spectral_feature_list(**kwargs)

        assert len(result) == 1
        feature = result[0]
        assert feature['rest_wavelength'] == 6355.0
        assert feature['sigma'] == 400.0
        assert feature['amplitude'] == -0.4
        assert feature['t_start'] == 0  # 0 days = 0 seconds
        assert feature['t_end'] == 30 * 24 * 3600  # 30 days to seconds
        # Check defaults for smooth mode
        assert feature['t_rise'] == 2.0 * 24 * 3600  # 2 days default
        assert feature['t_fall'] == 5.0 * 24 * 3600  # 5 days default

    def test_single_feature_with_custom_rise_fall(self):
        """Test single feature with custom rise/fall times"""
        kwargs = {
            'rest_wavelength_feature_1': 6355.0,
            'sigma_feature_1': 400.0,
            'amplitude_feature_1': -0.4,
            't_start_feature_1': 0,
            't_end_feature_1': 30,
            't_rise_feature_1': 3.0,
            't_fall_feature_1': 7.0
        }

        result = utils.build_spectral_feature_list(**kwargs)

        assert len(result) == 1
        feature = result[0]
        assert feature['t_rise'] == 3.0 * 24 * 3600
        assert feature['t_fall'] == 7.0 * 24 * 3600

    def test_multiple_features_sequential(self):
        """Test multiple features with sequential numbering"""
        kwargs = {
            'rest_wavelength_feature_1': 6355.0,
            'sigma_feature_1': 400.0,
            'amplitude_feature_1': -0.4,
            't_start_feature_1': 0,
            't_end_feature_1': 30,
            'rest_wavelength_feature_2': 3934.0,
            'sigma_feature_2': 300.0,
            'amplitude_feature_2': -0.5,
            't_start_feature_2': 5,
            't_end_feature_2': 60,
            'rest_wavelength_feature_3': 8600.0,
            'sigma_feature_3': 500.0,
            'amplitude_feature_3': -0.3,
            't_start_feature_3': 10,
            't_end_feature_3': 50
        }

        result = utils.build_spectral_feature_list(**kwargs)

        assert len(result) == 3

        # Check feature 1
        assert result[0]['rest_wavelength'] == 6355.0
        assert result[0]['t_start'] == 0

        # Check feature 2
        assert result[1]['rest_wavelength'] == 3934.0
        assert result[1]['t_start'] == 5 * 24 * 3600

        # Check feature 3
        assert result[2]['rest_wavelength'] == 8600.0
        assert result[2]['t_start'] == 10 * 24 * 3600

    def test_multiple_features_non_sequential(self):
        """Test multiple features with non-sequential numbering"""
        kwargs = {
            'rest_wavelength_feature_5': 6355.0,
            'sigma_feature_5': 400.0,
            'amplitude_feature_5': -0.4,
            't_start_feature_5': 0,
            't_end_feature_5': 30,
            'rest_wavelength_feature_2': 3934.0,
            'sigma_feature_2': 300.0,
            'amplitude_feature_2': -0.5,
            't_start_feature_2': 5,
            't_end_feature_2': 60,
            'rest_wavelength_feature_10': 8600.0,
            'sigma_feature_10': 500.0,
            'amplitude_feature_10': -0.3,
            't_start_feature_10': 10,
            't_end_feature_10': 50
        }

        result = utils.build_spectral_feature_list(**kwargs)

        assert len(result) == 3
        # Should be sorted by feature number: 2, 5, 10
        assert result[0]['rest_wavelength'] == 3934.0  # feature_2
        assert result[1]['rest_wavelength'] == 6355.0  # feature_5
        assert result[2]['rest_wavelength'] == 8600.0  # feature_10

    def test_sharp_evolution_mode_no_rise_fall(self):
        """Test that sharp mode doesn't add t_rise/t_fall parameters"""
        kwargs = {
            'rest_wavelength_feature_1': 6355.0,
            'sigma_feature_1': 400.0,
            'amplitude_feature_1': -0.4,
            't_start_feature_1': 0,
            't_end_feature_1': 30,
            'evolution_mode': 'sharp'
        }

        result = utils.build_spectral_feature_list(**kwargs)

        assert len(result) == 1
        feature = result[0]
        assert 't_rise' not in feature
        assert 't_fall' not in feature

    def test_time_conversion_days_to_seconds(self):
        """Test that time parameters are correctly converted from days to seconds"""
        kwargs = {
            'rest_wavelength_feature_1': 6355.0,
            'sigma_feature_1': 400.0,
            'amplitude_feature_1': -0.4,
            't_start_feature_1': 1.5,  # 1.5 days
            't_end_feature_1': 30.25,  # 30.25 days
            't_rise_feature_1': 2.5,  # 2.5 days
            't_fall_feature_1': 5.75  # 5.75 days
        }

        result = utils.build_spectral_feature_list(**kwargs)

        feature = result[0]
        assert feature['t_start'] == 1.5 * 24 * 3600
        assert feature['t_end'] == 30.25 * 24 * 3600
        assert feature['t_rise'] == 2.5 * 24 * 3600
        assert feature['t_fall'] == 5.75 * 24 * 3600

    def test_malformed_parameter_names_ignored(self):
        """Test that malformed parameter names are ignored"""
        kwargs = {
            'rest_wavelength_feature_1': 6355.0,
            'sigma_feature_1': 400.0,
            'amplitude_feature_1': -0.4,
            't_start_feature_1': 0,
            't_end_feature_1': 30,
            'bad_feature_name': 123,  # No number after _feature_
            'another_feature_bad_number': 456,  # Not following pattern
            'rest_wavelength_feature_abc': 789,  # Non-numeric feature number
        }

        # Should work fine and only create 1 feature
        result = utils.build_spectral_feature_list(**kwargs)
        assert len(result) == 1

    def test_zero_feature_number(self):
        """Test feature with number 0"""
        kwargs = {
            'rest_wavelength_feature_0': 6355.0,
            'sigma_feature_0': 400.0,
            'amplitude_feature_0': -0.4,
            't_start_feature_0': 0,
            't_end_feature_0': 30
        }

        result = utils.build_spectral_feature_list(**kwargs)
        assert len(result) == 1
        assert result[0]['rest_wavelength'] == 6355.0

    def test_negative_feature_number(self):
        """Test feature with negative number"""
        kwargs = {
            'rest_wavelength_feature_-1': 6355.0,
            'sigma_feature_-1': 400.0,
            'amplitude_feature_-1': -0.4,
            't_start_feature_-1': 0,
            't_end_feature_-1': 30
        }

        result = utils.build_spectral_feature_list(**kwargs)
        assert len(result) == 1
        assert result[0]['rest_wavelength'] == 6355.0

    def test_large_feature_numbers(self):
        """Test with large feature numbers"""
        kwargs = {
            'rest_wavelength_feature_1000': 6355.0,
            'sigma_feature_1000': 400.0,
            'amplitude_feature_1000': -0.4,
            't_start_feature_1000': 0,
            't_end_feature_1000': 30
        }

        result = utils.build_spectral_feature_list(**kwargs)
        assert len(result) == 1
        assert result[0]['rest_wavelength'] == 6355.0

    def test_partial_parameters_with_defaults_fallback(self):
        """Test that partial parameters don't prevent fallback to defaults"""
        kwargs = {
            'some_other_parameter': 123,
            'not_a_feature_param': 456,
            'use_default_features': True
        }

        result = utils.build_spectral_feature_list(**kwargs)
        expected = utils._get_default_sn_ia_features()
        assert result == expected


class TestGetDefaultSnIaFeatures(unittest.TestCase):
    """Test suite for _get_default_sn_ia_features function"""

    def test_returns_list(self):
        """Test that function returns a list"""
        result = utils._get_default_sn_ia_features()
        assert isinstance(result, list)

    def test_has_expected_number_of_features(self):
        """Test that default features has expected number of items"""
        result = utils._get_default_sn_ia_features()
        assert len(result) == 3  # Based on your implementation

    def test_features_have_required_keys(self):
        """Test that each feature has all required keys"""
        result = utils._get_default_sn_ia_features()
        required_keys = {'t_start', 't_end', 't_rise', 't_fall', 'rest_wavelength', 'sigma', 'amplitude'}

        for feature in result:
            assert isinstance(feature, dict)
            assert required_keys.issubset(feature.keys())

    def test_time_values_are_in_seconds(self):
        """Test that time values are in seconds (not days)"""
        result = utils._get_default_sn_ia_features()

        for feature in result:
            # Time values should be large (seconds, not days)
            assert feature['t_start'] >= 0
            assert feature['t_end'] > feature['t_start']
            assert feature['t_rise'] > 0
            assert feature['t_fall'] > 0
            # Should be much larger than typical day values
            assert feature['t_end'] > 1000  # More than 1000 seconds

    def test_wavelengths_are_reasonable(self):
        """Test that wavelengths are in reasonable ranges"""
        result = utils._get_default_sn_ia_features()

        for feature in result:
            # Wavelengths should be in Angstroms, optical range
            assert 1000 < feature['rest_wavelength'] < 50000
            assert feature['sigma'] > 0

    def test_amplitudes_are_reasonable(self):
        """Test that amplitudes are reasonable"""
        result = utils._get_default_sn_ia_features()

        for feature in result:
            # Amplitudes should be reasonable fractions
            assert -1.0 <= feature['amplitude'] <= 1.0
            # Most SN Ia features are absorption (negative)
            assert feature['amplitude'] != 0


class TestSEDErrorHandling(unittest.TestCase):
    """Tests for SED module error handling and logging."""

    def test_bandflux_zp_without_zpsys_logs_error(self):
        """Test that providing zp without zpsys logs error and raises ValueError."""
        from redback import sed
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_band = 'bessellb'
        time_or_phase = np.array([1.0])
        zp = 25.0
        zpsys = None

        with self.assertRaises(ValueError) as context:
            sed._bandflux_redback(mock_model, mock_band, time_or_phase, zp, zpsys)
        self.assertIn('zpsys', str(context.exception))

    def test_bandflux_single_wavelength_range_error_logs(self):
        """Test that wavelength range mismatch logs error."""
        from redback import sed
        from unittest.mock import MagicMock

        # Create mock model with narrow wavelength range
        mock_model = MagicMock()
        mock_model.minwave.return_value = 5000.0
        mock_model.maxwave.return_value = 6000.0

        # Create mock band that extends beyond model range
        mock_band = MagicMock()
        mock_band.minwave.return_value = 4000.0  # Outside model range
        mock_band.maxwave.return_value = 6000.0
        mock_band.name = 'test_band'

        time_or_phase = np.array([1.0])

        with self.assertRaises(ValueError) as context:
            sed._bandflux_single_redback(mock_model, mock_band, time_or_phase)
        self.assertIn('outside spectral range', str(context.exception))