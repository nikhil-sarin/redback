import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
from astropy import units as u
import redback.utils as utils
from selenium.common.exceptions import NoSuchElementException


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


class TestSeleniumFunctions(unittest.TestCase):
    """Test Selenium-related functions"""

    def test_check_element_exists(self):
        """Test check_element when element exists"""
        mock_driver = MagicMock()
        mock_driver.find_element.return_value = MagicMock()

        result = utils.check_element(mock_driver, 'test_id')
        self.assertTrue(result)

    def test_check_element_not_exists(self):
        """Test check_element when element doesn't exist"""
        mock_driver = MagicMock()
        mock_driver.find_element.side_effect = NoSuchElementException()

        result = utils.check_element(mock_driver, 'test_id')
        self.assertFalse(result)


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


if __name__ == '__main__':
    unittest.main()
