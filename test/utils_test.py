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
class TestCitationWrapper(unittest.TestCase):
    """Test citation_wrapper decorator function"""

    def test_citation_wrapper_adds_citation_attribute(self):
        """Test that citation_wrapper adds citation attribute to function"""
        @utils.citation_wrapper("Test Citation 2023")
        def test_function():
            return "result"

        self.assertTrue(hasattr(test_function, 'citation'))
        self.assertEqual(test_function.citation, "Test Citation 2023")

    def test_citation_wrapper_preserves_function_behavior(self):
        """Test that decorated function still works normally"""
        @utils.citation_wrapper("Another Citation")
        def add_numbers(a, b):
            return a + b

        self.assertEqual(add_numbers(2, 3), 5)
        self.assertEqual(add_numbers.citation, "Another Citation")


class TestWavelengthFrequencyConversions(unittest.TestCase):
    """Test wavelength/frequency conversion functions"""

    def test_calc_effective_width_hz_from_angstrom(self):
        """Test effective width conversion from Angstrom to Hz"""
        effective_width = 100.0  # Angstrom
        effective_wavelength = 5500.0  # Angstrom

        wavelength_m = effective_wavelength * 1.0e-10
        effective_width_m = effective_width * 1.0e-10
        expected = (3.0e8 / (wavelength_m ** 2)) * effective_width_m

        result = utils.calc_effective_width_hz_from_angstrom(effective_width, effective_wavelength)
        self.assertAlmostEqual(result, expected)

    def test_calc_effective_width_hz_from_angstrom_different_values(self):
        """Test with different wavelength values"""
        for eff_width in [50.0, 100.0, 200.0]:
            for eff_wavelength in [4000.0, 5500.0, 7000.0]:
                result = utils.calc_effective_width_hz_from_angstrom(eff_width, eff_wavelength)
                self.assertGreater(result, 0)


class TestDecelerationTimescale(unittest.TestCase):
    """Test deceleration_timescale function"""

    def test_deceleration_timescale_basic(self):
        """Test basic deceleration timescale calculation"""
        e0 = 1e52  # erg
        g0 = 100  # Lorentz factor
        n0 = 1.0  # cm^-3

        result = utils.deceleration_timescale(e0, g0, n0)

        # Result should be positive
        self.assertGreater(result, 0)

    def test_deceleration_timescale_different_parameters(self):
        """Test with different parameter values"""
        for e0 in [1e51, 1e52, 1e53]:
            for g0 in [10, 100, 1000]:
                for n0 in [0.1, 1.0, 10.0]:
                    result = utils.deceleration_timescale(e0, g0, n0)
                    self.assertGreater(result, 0)


class TestFluxDensityConversions(unittest.TestCase):
    """Test flux density conversion functions"""

    def test_calc_flux_density_from_ABmag(self):
        """Test AB magnitude to flux density conversion"""
        magnitudes = np.array([20.0, 21.0, 22.0])

        result = utils.calc_flux_density_from_ABmag(magnitudes)

        # Check that result is an astropy quantity
        self.assertTrue(hasattr(result, 'unit'))
        # Flux density should be positive
        self.assertTrue(np.all(result.value > 0))
        # Fainter magnitudes should have lower flux densities
        self.assertGreater(result[0].value, result[1].value)
        self.assertGreater(result[1].value, result[2].value)

    def test_calc_ABmag_from_flux_density(self):
        """Test flux density to AB magnitude conversion"""
        fluxdensity = np.array([1.0, 0.1, 0.01])

        result = utils.calc_ABmag_from_flux_density(fluxdensity)

        # Check that result is an astropy quantity
        self.assertTrue(hasattr(result, 'unit'))
        # Higher flux density should give brighter (lower) magnitude
        self.assertLess(result[0].value, result[1].value)
        self.assertLess(result[1].value, result[2].value)

    def test_calc_flux_density_from_vegamag(self):
        """Test Vega magnitude to flux density conversion"""
        magnitudes = 20.0
        zeropoint = 3631.0  # Jy

        result = utils.calc_flux_density_from_vegamag(magnitudes, zeropoint)

        # Result should be positive
        self.assertGreater(result, 0)

    def test_calc_flux_density_from_vegamag_array(self):
        """Test with array of magnitudes"""
        magnitudes = np.array([18.0, 20.0, 22.0])
        zeropoint = 3631.0

        result = utils.calc_flux_density_from_vegamag(magnitudes, zeropoint)

        # Brighter magnitudes should have higher flux densities
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])

    def test_calc_vegamag_from_flux_density(self):
        """Test flux density to Vega magnitude conversion"""
        fluxdensity = 100.0  # mJy
        zeropoint = 3631.0  # Jy

        result = utils.calc_vegamag_from_flux_density(fluxdensity, zeropoint)

        # Result should be a reasonable magnitude
        self.assertGreater(result, 0)
        self.assertLess(result, 30)

    def test_vegamag_roundtrip(self):
        """Test roundtrip conversion Vega mag -> flux -> mag"""
        original_mag = 18.5
        zeropoint = 3631.0

        flux = utils.calc_flux_density_from_vegamag(original_mag, zeropoint)
        recovered_mag = utils.calc_vegamag_from_flux_density(flux, zeropoint)

        self.assertAlmostEqual(original_mag, recovered_mag, places=5)


class TestBandfluxFunctions(unittest.TestCase):
    """Test bandflux-related functions"""

    def test_bandflux_error_from_limiting_mag(self):
        """Test bandflux error calculation from limiting magnitude"""
        fiveSigmaDepth = 24.5
        bandflux_ref = 3631.0  # Reference flux

        result = utils.bandflux_error_from_limiting_mag(fiveSigmaDepth, bandflux_ref)

        # Error should be positive
        self.assertGreater(result, 0)

    def test_bandflux_error_different_depths(self):
        """Test with different 5-sigma depths"""
        bandflux_ref = 3631.0

        depths = [22.0, 24.0, 26.0]
        results = [utils.bandflux_error_from_limiting_mag(d, bandflux_ref) for d in depths]

        # Deeper limiting magnitudes should give smaller errors
        self.assertGreater(results[0], results[1])
        self.assertGreater(results[1], results[2])

    def test_bandpass_flux_to_flux_density(self):
        """Test integrated flux to flux density conversion"""
        flux = 1e-12  # erg/s/cm^2
        flux_err = 1e-13
        delta_nu = 1e14  # Hz

        f_nu_mJy, f_nu_err_mJy = utils.bandpass_flux_to_flux_density(flux, flux_err, delta_nu)

        # Results should be positive
        self.assertGreater(f_nu_mJy, 0)
        self.assertGreater(f_nu_err_mJy, 0)

        # Error should be smaller than flux
        self.assertLess(f_nu_err_mJy, f_nu_mJy)

    def test_bandpass_flux_to_flux_density_array(self):
        """Test with arrays of flux values"""
        flux = np.array([1e-12, 2e-12, 3e-12])
        flux_err = np.array([1e-13, 2e-13, 3e-13])
        delta_nu = 1e14

        f_nu_mJy, f_nu_err_mJy = utils.bandpass_flux_to_flux_density(flux, flux_err, delta_nu)

        # Check shapes match
        self.assertEqual(len(f_nu_mJy), len(flux))
        self.assertEqual(len(f_nu_err_mJy), len(flux_err))

        # All results should be positive
        self.assertTrue(np.all(f_nu_mJy > 0))
        self.assertTrue(np.all(f_nu_err_mJy > 0))


class TestMagnitudeConversions(unittest.TestCase):
    """Test magnitude conversion functions"""

    def test_convert_apparent_mag_to_absolute_default_cosmology(self):
        """Test apparent to absolute magnitude conversion with default cosmology"""
        app_magnitude = 20.0
        redshift = 0.1

        result = utils.convert_apparent_mag_to_absolute(app_magnitude, redshift)

        # Absolute magnitude should be brighter (more negative) than apparent
        self.assertLess(result, app_magnitude)

    def test_convert_apparent_mag_to_absolute_higher_redshift(self):
        """Test with higher redshift"""
        app_magnitude = 22.0
        redshift = 1.0

        result = utils.convert_apparent_mag_to_absolute(app_magnitude, redshift)

        # At higher redshift, absolute magnitude should be much brighter
        self.assertLess(result, app_magnitude - 10)

    def test_convert_absolute_mag_to_apparent(self):
        """Test absolute to apparent magnitude conversion"""
        magnitude = -18.0  # Typical supernova absolute magnitude
        distance = 1e7  # parsecs

        result = utils.convert_absolute_mag_to_apparent(magnitude, distance)

        # Apparent magnitude should be fainter (more positive)
        self.assertGreater(result, magnitude)

    def test_convert_absolute_mag_to_apparent_closer_distance(self):
        """Test with closer distance"""
        magnitude = -18.0
        distance_far = 1e8
        distance_near = 1e6

        app_mag_far = utils.convert_absolute_mag_to_apparent(magnitude, distance_far)
        app_mag_near = utils.convert_absolute_mag_to_apparent(magnitude, distance_near)

        # Closer distance should give brighter (lower) apparent magnitude
        self.assertLess(app_mag_near, app_mag_far)

    def test_abmag_to_flux_density_and_error_inmjy(self):
        """Test AB magnitude to flux density and error in mJy"""
        m_AB = 20.0
        sigma_m = 0.1

        f_nu_mjy, sigma_f_mjy = utils.abmag_to_flux_density_and_error_inmjy(m_AB, sigma_m)

        # Both should be positive
        self.assertGreater(f_nu_mjy, 0)
        self.assertGreater(sigma_f_mjy, 0)

        # Error should be smaller than flux for reasonable magnitude errors
        self.assertLess(sigma_f_mjy, f_nu_mjy)

    def test_abmag_to_flux_density_and_error_inmjy_array(self):
        """Test with arrays"""
        m_AB = np.array([18.0, 20.0, 22.0])
        sigma_m = np.array([0.05, 0.1, 0.15])

        f_nu_mjy, sigma_f_mjy = utils.abmag_to_flux_density_and_error_inmjy(m_AB, sigma_m)

        # Check shapes
        self.assertEqual(len(f_nu_mjy), len(m_AB))
        self.assertEqual(len(sigma_f_mjy), len(sigma_m))

        # Brighter magnitudes should have higher flux densities
        self.assertGreater(f_nu_mjy[0], f_nu_mjy[1])
        self.assertGreater(f_nu_mjy[1], f_nu_mjy[2])


class TestFluxErrorFunctions(unittest.TestCase):
    """Test flux error calculation functions"""

    def test_calc_flux_density_error_from_monochromatic_magnitude_AB(self):
        """Test flux density error from magnitude error (AB system)"""
        magnitude = 20.0
        magnitude_error = 0.1
        reference_flux = 3631.0  # Will be overridden for AB system

        result = utils.calc_flux_density_error_from_monochromatic_magnitude(
            magnitude, magnitude_error, reference_flux, magnitude_system='AB')

        # Error should be positive
        self.assertGreater(result, 0)

    def test_calc_flux_density_error_from_monochromatic_magnitude_vega(self):
        """Test with Vega system"""
        magnitude = 20.0
        magnitude_error = 0.1
        reference_flux = 3000.0

        result = utils.calc_flux_density_error_from_monochromatic_magnitude(
            magnitude, magnitude_error, reference_flux, magnitude_system='Vega')

        # Error should be positive
        self.assertGreater(result, 0)

    def test_calc_flux_error_from_magnitude(self):
        """Test flux error from magnitude error"""
        magnitude = 18.0
        magnitude_error = 0.05
        reference_flux = 3631.0

        result = utils.calc_flux_error_from_magnitude(magnitude, magnitude_error, reference_flux)

        # Error should be positive
        self.assertGreater(result, 0)

    def test_magnitude_error_from_flux_error(self):
        """Test magnitude error from flux error"""
        bandflux = np.array([1000.0, 500.0, 100.0, 0.0, np.nan])
        bandflux_error = np.array([10.0, 5.0, 1.0, 1.0, 1.0])

        result = utils.magnitude_error_from_flux_error(bandflux, bandflux_error)

        # First three should be valid
        self.assertGreater(result[0], 0)
        self.assertGreater(result[1], 0)
        self.assertGreater(result[2], 0)

        # Zero and NaN flux should give NaN magnitude error
        self.assertTrue(np.isnan(result[3]))
        self.assertTrue(np.isnan(result[4]))


class TestBandFunctions(unittest.TestCase):
    """Test band-related lookup functions"""

    @patch('pandas.read_csv')
    def test_bands_to_zeropoint(self, mock_read_csv):
        """Test bands to zeropoint conversion"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r', 'i'],
            'reference_flux': [3631.0, 3000.0, 2500.0]
        })
        mock_read_csv.return_value = mock_df

        result = utils.bands_to_zeropoint(['g'])

        # Zeropoint is calculated as 10^(reference_flux / -2.5)
        # For large reference_flux values, this will be very small (near 0)
        # Just check that the function runs without error
        self.assertIsNotNone(result)

    @patch('pandas.read_csv')
    def test_bandpass_magnitude_to_flux(self, mock_read_csv):
        """Test bandpass magnitude to flux conversion"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r'],
            'reference_flux': [3631.0, 3000.0]
        })
        mock_read_csv.return_value = mock_df

        magnitude = 20.0
        bands = 'g'

        result = utils.bandpass_magnitude_to_flux(magnitude, bands)

        # Flux should be positive
        self.assertGreater(result, 0)

    @patch('pandas.read_csv')
    def test_bandpass_flux_to_magnitude(self, mock_read_csv):
        """Test bandpass flux to magnitude conversion"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r'],
            'reference_flux': [3631.0, 3000.0]
        })
        mock_read_csv.return_value = mock_df

        flux = 100.0
        bands = 'g'

        result = utils.bandpass_flux_to_magnitude(flux, bands)

        # Magnitude should be reasonable
        self.assertGreater(result, 0)
        self.assertLess(result, 30)

    @patch('pandas.read_csv')
    def test_bands_to_reference_flux(self, mock_read_csv):
        """Test bands to reference flux lookup"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r', 'i'],
            'reference_flux': [3631.0, 3000.0, 2500.0]
        })
        mock_read_csv.return_value = mock_df

        # Test single band
        result = utils.bands_to_reference_flux('g')
        np.testing.assert_array_equal(result, np.array([3631.0]))

        # Test multiple bands
        result = utils.bands_to_reference_flux(['g', 'r'])
        np.testing.assert_array_equal(result, np.array([3631.0, 3000.0]))

    @patch('pandas.read_csv')
    def test_bands_to_reference_flux_invalid_band(self, mock_read_csv):
        """Test with invalid band raises KeyError"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r'],
            'reference_flux': [3631.0, 3000.0]
        })
        mock_read_csv.return_value = mock_df

        with self.assertRaises(KeyError):
            utils.bands_to_reference_flux(['unknown_band'])

    @patch('pandas.read_csv')
    def test_bands_to_frequency(self, mock_read_csv):
        """Test bands to frequency conversion"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r', 'i'],
            'wavelength [Hz]': [6.0e14, 5.0e14, 4.5e14]
        })
        mock_read_csv.return_value = mock_df

        result = utils.bands_to_frequency(['g', 'r'])
        np.testing.assert_array_equal(result, np.array([6.0e14, 5.0e14]))

    @patch('pandas.read_csv')
    def test_bands_to_effective_width(self, mock_read_csv):
        """Test bands to effective width conversion"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r', 'i'],
            'effective_width [Hz]': [1.0e14, 1.2e14, 1.5e14]
        })
        mock_read_csv.return_value = mock_df

        result = utils.bands_to_effective_width(['g', 'i'])
        np.testing.assert_array_equal(result, np.array([1.0e14, 1.5e14]))

    @patch('pandas.read_csv')
    def test_frequency_to_bandname(self, mock_read_csv):
        """Test frequency to bandname conversion"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r', 'i'],
            'wavelength [Hz]': [6.0e14, 5.0e14, 4.5e14]
        })
        mock_read_csv.return_value = mock_df

        result = utils.frequency_to_bandname([6.0e14, 4.5e14])
        np.testing.assert_array_equal(result, np.array(['g', 'i']))

    @patch('pandas.read_csv')
    def test_frequency_to_bandname_invalid_frequency(self, mock_read_csv):
        """Test with invalid frequency raises KeyError"""
        mock_df = pd.DataFrame({
            'bands': ['g', 'r'],
            'wavelength [Hz]': [6.0e14, 5.0e14]
        })
        mock_read_csv.return_value = mock_df

        with self.assertRaises(KeyError):
            utils.frequency_to_bandname([1.0e15])


class TestStatisticalFunctions(unittest.TestCase):
    """Test statistical functions"""

    def test_kde_scipy(self):
        """Test KDE calculation"""
        x = np.random.randn(100)
        bandwidth = 0.1

        kde = utils.kde_scipy(x, bandwidth=bandwidth)

        # KDE object should be callable
        self.assertTrue(callable(kde))

        # Should be able to evaluate at points
        result = kde(0.0)
        self.assertGreater(result, 0)

    def test_cdf_no_plot(self):
        """Test CDF without plotting"""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        x_sorted, y_cdf = utils.cdf(x, plot=False)

        # Check that output is sorted
        self.assertTrue(np.all(x_sorted[:-1] <= x_sorted[1:]))

        # CDF should be between 0 and 1
        self.assertTrue(np.all(y_cdf >= 0))
        self.assertTrue(np.all(y_cdf <= 1))

        # CDF should be monotonically increasing
        self.assertTrue(np.all(y_cdf[:-1] <= y_cdf[1:]))

    @patch('matplotlib.pyplot.plot')
    def test_cdf_with_plot(self, mock_plot):
        """Test CDF with plotting"""
        x = np.array([1, 2, 3, 4, 5])

        utils.cdf(x, plot=True)

        # Check that plt.plot was called
        mock_plot.assert_called_once()


class TestTimeBinning(unittest.TestCase):
    """Test time binning functions"""

    def test_bin_ttes(self):
        """Test binning of time-tagged events"""
        ttes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        bin_size = 1.0

        times, counts = utils.bin_ttes(ttes, bin_size)

        # Each bin should have approximately 2 events
        self.assertTrue(np.all(counts > 0))

        # Number of bins should match
        self.assertEqual(len(times), len(counts))

    def test_bin_ttes_different_bin_sizes(self):
        """Test with different bin sizes"""
        ttes = np.linspace(0, 10, 100)

        for bin_size in [0.5, 1.0, 2.0]:
            times, counts = utils.bin_ttes(ttes, bin_size)

            # Check that we get reasonable output
            self.assertGreater(len(times), 0)
            self.assertEqual(len(times), len(counts))


class TestPathFunctions(unittest.TestCase):
    """Test path-related functions"""

    def test_find_path_default(self):
        """Test find_path with 'default' argument"""
        result = utils.find_path('default')

        # Should return a path containing 'GRBData'
        self.assertIn('GRBData', result)

    def test_find_path_custom(self):
        """Test find_path with custom path"""
        custom_path = '/custom/path/to/data'
        result = utils.find_path(custom_path)

        # Should return the same path
        self.assertEqual(result, custom_path)

    def test_download_pointing_tables(self):
        """Test download_pointing_tables function"""
        # This function just returns a log message
        result = utils.download_pointing_tables()

        # Result should be None (logger.info returns None)
        self.assertIsNone(result)


class TestKwargsAccessorWithDefault(unittest.TestCase):
    """Test KwargsAccessorWithDefault descriptor"""

    def test_get_existing_kwarg(self):
        """Test getting an existing kwarg"""
        class MockClass:
            def __init__(self):
                self.kwargs = {'test_key': 'test_value'}

        mock_instance = MockClass()
        accessor = utils.KwargsAccessorWithDefault(kwarg='test_key')

        result = accessor.__get__(mock_instance, None)
        self.assertEqual(result, 'test_value')

    def test_get_nonexistent_kwarg_with_default(self):
        """Test getting a non-existent kwarg with default"""
        class MockClass:
            def __init__(self):
                self.kwargs = {}

        mock_instance = MockClass()
        accessor = utils.KwargsAccessorWithDefault(kwarg='missing_key', default='default_value')

        result = accessor.__get__(mock_instance, None)
        self.assertEqual(result, 'default_value')

    def test_set_kwarg(self):
        """Test setting a kwarg"""
        class MockClass:
            def __init__(self):
                self.kwargs = {}

        mock_instance = MockClass()
        accessor = utils.KwargsAccessorWithDefault(kwarg='new_key')

        accessor.__set__(mock_instance, 'new_value')
        self.assertEqual(mock_instance.kwargs['new_key'], 'new_value')


class TestGetFunctionsDict(unittest.TestCase):
    """Test get_functions_dict function"""

    def test_get_functions_dict(self):
        """Test extracting functions from a module"""
        # Create a mock module
        import types
        mock_module = types.ModuleType('test_module')

        def test_func1():
            pass

        def test_func2():
            pass

        mock_module.test_func1 = test_func1
        mock_module.test_func2 = test_func2
        mock_module.__name__ = 'parent.test_module'

        result = utils.get_functions_dict(mock_module)

        # Check that it returns a dict
        self.assertIsInstance(result, dict)

        # Check that module name is in the result
        self.assertIn('test_module', result)


class TestThermalisation(unittest.TestCase):
    """Test thermalisation and heating functions"""

    def test_interpolated_barnes_and_kasen_thermalisation_efficiency(self):
        """Test Barnes & Kasen thermalisation efficiency interpolation"""
        mej = 0.01  # solar masses
        vej = 0.2  # fraction of c

        av, bv, dv = utils.interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)

        # All values should be positive and reasonable
        self.assertGreater(av, 0)
        self.assertGreater(bv, 0)
        self.assertGreater(dv, 0)

    def test_interpolated_barnes_and_kasen_different_values(self):
        """Test with different ejecta parameters"""
        for mej in [0.001, 0.01, 0.05]:
            for vej in [0.1, 0.2, 0.3]:
                av, bv, dv = utils.interpolated_barnes_and_kasen_thermalisation_efficiency(mej, vej)

                # Check all results are positive
                self.assertGreater(av, 0)
                self.assertGreater(bv, 0)
                self.assertGreater(dv, 0)

    def test_heatinggrids(self):
        """Test heatinggrids function returns interpolators"""
        result = utils.heatinggrids()

        # Check that result has all required attributes
        required_attrs = ['E0', 'ALP', 'T0', 'SIG', 'ALP1', 'T1', 'SIG1',
                          'C1', 'TAU1', 'C2', 'TAU2', 'C3', 'TAU3']

        for attr in required_attrs:
            self.assertTrue(hasattr(result, attr))

        # Each interpolator should be callable
        test_point = [0.2, 0.3]  # velocity, electron fraction
        self.assertTrue(callable(result.E0))
        # Try evaluating one
        value = result.E0(test_point)
        self.assertIsInstance(value, (np.ndarray, float))

    def test_get_heating_terms(self):
        """Test get_heating_terms function"""
        ye = 0.3  # electron fraction
        vel = 0.2  # velocity

        result = utils.get_heating_terms(ye, vel)

        # Check that result has all required attributes
        required_attrs = ['e0', 'alp', 't0', 'sig', 'alp1', 't1', 'sig1',
                          'c1', 'tau1', 'c2', 'tau2', 'c3', 'tau3']

        for attr in required_attrs:
            self.assertTrue(hasattr(result, attr))

    def test_get_heating_terms_with_fudge_factor(self):
        """Test get_heating_terms with heating_rate_fudge"""
        ye = 0.3
        vel = 0.2
        fudge = 2.0

        result = utils.get_heating_terms(ye, vel, heating_rate_fudge=fudge)

        # All values should be affected by fudge factor
        # Just check that we got reasonable values
        self.assertIsNotNone(result.e0)
        self.assertIsNotNone(result.alp)


class TestElectronFractionKappa(unittest.TestCase):
    """Test electron fraction and kappa conversion functions"""

    def test_electron_fraction_from_kappa(self):
        """Test electron fraction calculation from kappa"""
        kappa = 10.0

        result = utils.electron_fraction_from_kappa(kappa)

        # Electron fraction should be between 0 and 1 (roughly)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_kappa_from_electron_fraction(self):
        """Test kappa calculation from electron fraction"""
        ye = 0.3

        result = utils.kappa_from_electron_fraction(ye)

        # Kappa should be positive
        self.assertGreater(result, 0)

    def test_roundtrip_electron_fraction_kappa(self):
        """Test roundtrip conversion"""
        original_ye = 0.25

        kappa = utils.kappa_from_electron_fraction(original_ye)
        recovered_ye = utils.electron_fraction_from_kappa(kappa)

        # Should recover original value (within interpolation error)
        self.assertAlmostEqual(original_ye, recovered_ye, places=3)


class TestLorentzFactorVelocity(unittest.TestCase):
    """Test Lorentz factor and velocity conversion functions"""

    def test_lorentz_factor_from_velocity_low(self):
        """Test Lorentz factor from low velocity"""
        velocity = 0.1 * utils.speed_of_light  # 0.1c

        result = utils.lorentz_factor_from_velocity(velocity)

        # For 0.1c, gamma should be close to 1.005
        self.assertGreater(result, 1.0)
        self.assertLess(result, 1.1)

    def test_lorentz_factor_from_velocity_high(self):
        """Test Lorentz factor from high velocity"""
        velocity = 0.9 * utils.speed_of_light  # 0.9c

        result = utils.lorentz_factor_from_velocity(velocity)

        # For 0.9c, gamma should be around 2.29
        self.assertGreater(result, 2.0)
        self.assertLess(result, 3.0)

    def test_velocity_from_lorentz_factor_low(self):
        """Test velocity from low Lorentz factor"""
        lorentz_factor = 1.1

        result = utils.velocity_from_lorentz_factor(lorentz_factor)

        # Velocity should be less than c
        self.assertGreater(result, 0)
        self.assertLess(result, utils.speed_of_light)

    def test_velocity_from_lorentz_factor_high(self):
        """Test velocity from high Lorentz factor"""
        lorentz_factor = 10.0

        result = utils.velocity_from_lorentz_factor(lorentz_factor)

        # Velocity should be close to c but less than c
        self.assertGreater(result, 0.9 * utils.speed_of_light)
        self.assertLess(result, utils.speed_of_light)

    def test_roundtrip_lorentz_velocity(self):
        """Test roundtrip conversion"""
        original_velocity = 0.5 * utils.speed_of_light

        gamma = utils.lorentz_factor_from_velocity(original_velocity)
        recovered_velocity = utils.velocity_from_lorentz_factor(gamma)

        # Should recover original velocity (use delta for large numbers)
        self.assertAlmostEqual(original_velocity, recovered_velocity, delta=1.0)


class TestCSMProperties(unittest.TestCase):
    """Test CSM properties function"""

    def test_get_csm_properties(self):
        """Test CSM properties calculation"""
        nn = 8.0  # CSM norm
        eta = 1.5  # CSM density profile exponent

        result = utils.get_csm_properties(nn, eta)

        # Check that result has required attributes
        self.assertTrue(hasattr(result, 'AA'))
        self.assertTrue(hasattr(result, 'Bf'))
        self.assertTrue(hasattr(result, 'Br'))

        # All values should be reasonable
        self.assertIsNotNone(result.AA)
        self.assertIsNotNone(result.Bf)
        self.assertIsNotNone(result.Br)

    def test_get_csm_properties_different_values(self):
        """Test with different parameter values"""
        for nn in [7.0, 8.0, 9.0]:
            for eta in [1.0, 1.5, 2.0]:
                result = utils.get_csm_properties(nn, eta)

                # Should return valid results
                self.assertIsNotNone(result.AA)
                self.assertIsNotNone(result.Bf)
                self.assertIsNotNone(result.Br)

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