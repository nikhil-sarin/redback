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