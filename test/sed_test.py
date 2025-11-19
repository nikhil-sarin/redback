import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import astropy.units as uu
import redback.sed as sed


class TestBlackbodyToFluxDensity(unittest.TestCase):
    """Test blackbody_to_flux_density function"""

    def test_basic_blackbody(self):
        """Test basic blackbody flux density calculation"""
        temperature = 10000.0  # K
        r_photosphere = 1e14  # cm
        dl = 1e27  # cm
        frequency = 1e15  # Hz

        flux_density = sed.blackbody_to_flux_density(temperature, r_photosphere, dl, frequency)
        self.assertIsInstance(flux_density, uu.Quantity)
        self.assertTrue(flux_density.value > 0)

    def test_blackbody_array_frequency(self):
        """Test with array of frequencies"""
        temperature = 10000.0
        r_photosphere = 1e14
        dl = 1e27
        frequency = np.array([1e14, 1e15, 1e16])

        flux_density = sed.blackbody_to_flux_density(temperature, r_photosphere, dl, frequency)
        self.assertEqual(len(flux_density), 3)

    def test_blackbody_temperature_dependence(self):
        """Test that higher temperature gives higher flux"""
        r_photosphere = 1e14
        dl = 1e27
        frequency = 1e15

        flux_low = sed.blackbody_to_flux_density(5000.0, r_photosphere, dl, frequency)
        flux_high = sed.blackbody_to_flux_density(20000.0, r_photosphere, dl, frequency)

        self.assertGreater(flux_high.value, flux_low.value)


class TestBlackbodyClass(unittest.TestCase):
    """Test Blackbody SED class"""

    def test_initialization(self):
        """Test Blackbody class initialization"""
        temperature = np.array([10000.0, 12000.0])
        r_photosphere = np.array([1e14, 1.2e14])
        frequency = np.array([1e15, 1.2e15])
        luminosity_distance = 1e27

        bb = sed.Blackbody(temperature, r_photosphere, frequency, luminosity_distance)

        self.assertIsNotNone(bb.flux_density)
        self.assertTrue(hasattr(bb, 'reference'))

    def test_calculate_flux_density(self):
        """Test flux density calculation"""
        temperature = np.array([10000.0])
        r_photosphere = np.array([1e14])
        frequency = np.array([1e15])
        luminosity_distance = 1e27

        bb = sed.Blackbody(temperature, r_photosphere, frequency, luminosity_distance)
        flux = bb.calculate_flux_density()

        self.assertIsInstance(flux, uu.Quantity)
        self.assertGreater(flux.value[0], 0)


class TestCutoffBlackbody(unittest.TestCase):
    """Test CutoffBlackbody SED class"""

    def test_initialization(self):
        """Test CutoffBlackbody initialization"""
        time = np.array([1.0, 2.0, 3.0])
        temperature = np.array([10000.0, 12000.0, 11000.0])
        luminosity = np.array([1e43, 1.2e43, 1.1e43])
        r_photosphere = np.array([1e14, 1.2e14, 1.1e14])
        frequency = 1e15
        luminosity_distance = 1e27
        cutoff_wavelength = 3000.0  # Angstrom

        cb = sed.CutoffBlackbody(
            time, temperature, luminosity, r_photosphere,
            frequency, luminosity_distance, cutoff_wavelength)

        self.assertIsNotNone(cb.sed)
        self.assertIsNotNone(cb.flux_density)
        self.assertTrue(hasattr(cb, 'reference'))

    def test_wavelength_property(self):
        """Test wavelength property"""
        time = np.array([1.0, 2.0])
        temperature = np.array([10000.0, 12000.0])
        luminosity = np.array([1e43, 1.2e43])
        r_photosphere = np.array([1e14, 1.2e14])
        frequency = 1e15
        luminosity_distance = 1e27
        cutoff_wavelength = 3000.0

        cb = sed.CutoffBlackbody(
            time, temperature, luminosity, r_photosphere,
            frequency, luminosity_distance, cutoff_wavelength)

        wavelength = cb.wavelength
        self.assertIsNotNone(wavelength)

    def test_mask_property(self):
        """Test mask property for cutoff"""
        time = np.array([1.0, 2.0])
        temperature = np.array([10000.0, 12000.0])
        luminosity = np.array([1e43, 1.2e43])
        r_photosphere = np.array([1e14, 1.2e14])
        frequency = 1e15
        luminosity_distance = 1e27
        cutoff_wavelength = 3000.0

        cb = sed.CutoffBlackbody(
            time, temperature, luminosity, r_photosphere,
            frequency, luminosity_distance, cutoff_wavelength)

        mask = cb.mask
        self.assertIsInstance(mask, (np.ndarray, bool, np.bool_))


class TestPowerlawPlusBlackbody(unittest.TestCase):
    """Test PowerlawPlusBlackbody class"""

    def test_initialization(self):
        """Test PowerlawPlusBlackbody initialization"""
        temperature = np.array([10000.0])
        r_photosphere = np.array([1e14])
        pl_amplitude = 1e-10
        pl_slope = -2.0
        pl_evolution_index = 0.5
        time = np.array([1.0])
        reference_wavelength = 5000.0
        frequency = np.array([1e15])
        luminosity_distance = 1e27

        plbb = sed.PowerlawPlusBlackbody(
            temperature, r_photosphere, pl_amplitude, pl_slope,
            pl_evolution_index, time, reference_wavelength,
            frequency, luminosity_distance)

        self.assertIsNotNone(plbb.flux_density)
        self.assertGreater(plbb.flux_density.value, 0)

    def test_time_evolution(self):
        """Test power law evolution with time"""
        temperature = np.array([10000.0, 10000.0])
        r_photosphere = np.array([1e14, 1e14])
        pl_amplitude = 1e-10
        pl_slope = -2.0
        pl_evolution_index = 0.5
        time = np.array([1.0, 10.0])
        reference_wavelength = 5000.0
        frequency = np.array([1e15, 1e15])
        luminosity_distance = 1e27

        plbb = sed.PowerlawPlusBlackbody(
            temperature, r_photosphere, pl_amplitude, pl_slope,
            pl_evolution_index, time, reference_wavelength,
            frequency, luminosity_distance)

        self.assertIsNotNone(plbb.flux_density)


class TestBlackbodyWithSpectralFeatures(unittest.TestCase):
    """Test BlackbodyWithSpectralFeatures class"""

    def test_initialization_no_features(self):
        """Test initialization without features"""
        temperature = np.array([10000.0])
        r_photosphere = np.array([1e14])
        frequency = np.array([1e15])
        luminosity_distance = 1e27
        time = np.array([1.0])

        bb = sed.BlackbodyWithSpectralFeatures(
            temperature, r_photosphere, frequency,
            luminosity_distance, time)

        self.assertIsNotNone(bb.flux_density)

    def test_initialization_with_smooth_features(self):
        """Test initialization with smooth features"""
        temperature = np.array([10000.0, 11000.0])
        r_photosphere = np.array([1e14, 1.1e14])
        frequency = np.array([1e15])
        luminosity_distance = 1e27
        time = np.array([0.0, 86400.0 * 5])  # 0 and 5 days

        feature = {
            't_start': 0,
            't_end': 86400.0 * 10,
            'rest_wavelength': 6355.0,
            'sigma': 400.0,
            'amplitude': -0.3
        }

        bb = sed.BlackbodyWithSpectralFeatures(
            temperature, r_photosphere, frequency,
            luminosity_distance, time, feature_list=[feature],
            evolution_mode='smooth')

        self.assertIsNotNone(bb.flux_density)

    def test_initialization_with_sharp_features(self):
        """Test initialization with sharp features"""
        temperature = np.array([10000.0])
        r_photosphere = np.array([1e14])
        frequency = np.array([1e15])
        luminosity_distance = 1e27
        time = np.array([86400.0])  # 1 day

        feature = {
            't_start': 0,
            't_end': 86400.0 * 10,
            'rest_wavelength': 6355.0,
            'sigma': 400.0,
            'amplitude': -0.3
        }

        bb = sed.BlackbodyWithSpectralFeatures(
            temperature, r_photosphere, frequency,
            luminosity_distance, time, feature_list=[feature],
            evolution_mode='sharp')

        self.assertIsNotNone(bb.flux_density)

    def test_invalid_evolution_mode(self):
        """Test that invalid evolution mode raises error"""
        temperature = np.array([10000.0])
        r_photosphere = np.array([1e14])
        frequency = np.array([1e15])
        luminosity_distance = 1e27
        time = np.array([1.0])

        with self.assertRaises(ValueError):
            sed.BlackbodyWithSpectralFeatures(
                temperature, r_photosphere, frequency,
                luminosity_distance, time, evolution_mode='invalid')


class TestSynchrotron(unittest.TestCase):
    """Test Synchrotron SED class"""

    def test_initialization(self):
        """Test Synchrotron initialization"""
        frequency = np.array([1e9, 1e10, 1e11])
        luminosity_distance = 1e27
        pp = 2.5
        nu_max = 5e10
        source_radius = 1e15
        f0 = 1e-26

        syn = sed.Synchrotron(
            frequency, luminosity_distance, pp, nu_max, source_radius, f0)

        self.assertIsNotNone(syn.sed)
        self.assertIsNotNone(syn.flux_density)
        self.assertTrue(hasattr(syn, 'reference'))

    def test_f_max_property(self):
        """Test f_max property"""
        frequency = np.array([1e9])
        luminosity_distance = 1e27
        pp = 2.5
        nu_max = 5e10

        syn = sed.Synchrotron(frequency, luminosity_distance, pp, nu_max)
        f_max = syn.f_max
        self.assertGreater(f_max, 0)

    def test_mask_property(self):
        """Test frequency mask"""
        frequency = np.array([1e9, 1e10, 1e11])
        luminosity_distance = 1e27
        pp = 2.5
        nu_max = 5e10

        syn = sed.Synchrotron(frequency, luminosity_distance, pp, nu_max)
        mask = syn.mask
        self.assertEqual(len(mask), 3)


class TestLine(unittest.TestCase):
    """Test Line SED class"""

    def test_initialization(self):
        """Test Line initialization"""
        time = np.array([1.0, 2.0, 3.0])
        luminosity = np.array([1e43, 1.2e43, 1.1e43])
        frequency = 1e15
        luminosity_distance = 1e27

        # Create a mock SED
        mock_sed = MagicMock()
        mock_sed.flux_density = np.ones((1, 3)) * 1e-10 * uu.erg / uu.s / uu.cm**2 / uu.Hz

        line = sed.Line(
            time, luminosity, frequency, mock_sed, luminosity_distance,
            line_wavelength=7500.0, line_width=500.0,
            line_time=2.0, line_duration=1.0, line_amplitude=0.3)

        self.assertIsNotNone(line.flux_density)

    def test_wavelength_property(self):
        """Test wavelength property of Line"""
        time = np.array([1.0])
        luminosity = np.array([1e43])
        frequency = 1e15
        luminosity_distance = 1e27

        mock_sed = MagicMock()
        mock_sed.flux_density = np.ones((1, 1)) * 1e-10 * uu.erg / uu.s / uu.cm**2 / uu.Hz

        line = sed.Line(time, luminosity, frequency, mock_sed, luminosity_distance)
        wavelength = line.wavelength
        self.assertIsNotNone(wavelength)


class TestFluxDensityToSpectrum(unittest.TestCase):
    """Test flux_density_to_spectrum function"""

    def test_conversion(self):
        """Test flux density to spectrum conversion"""
        flux_density = np.array([[1e-10, 2e-10], [1.5e-10, 2.5e-10]]) * uu.erg / uu.s / uu.Hz / uu.cm**2
        redshift = 0.1
        lambda_observer_frame = np.array([4000.0, 5000.0])

        spectra = sed.flux_density_to_spectrum(flux_density, redshift, lambda_observer_frame)

        self.assertIsInstance(spectra, uu.Quantity)
        self.assertEqual(spectra.unit, uu.erg / uu.cm**2 / uu.s / uu.Angstrom)

    def test_conversion_without_units(self):
        """Test conversion when flux_density has no units"""
        flux_density = np.array([[1e-10, 2e-10]])
        redshift = 0.1
        lambda_observer_frame = np.array([4000.0, 5000.0])

        spectra = sed.flux_density_to_spectrum(flux_density, redshift, lambda_observer_frame)

        self.assertIsInstance(spectra, uu.Quantity)


class TestBlackbodyToSpectrum(unittest.TestCase):
    """Test blackbody_to_spectrum function"""

    def test_blackbody_spectrum(self):
        """Test blackbody to spectrum conversion"""
        temperature = np.array([10000.0, 12000.0])
        r_photosphere = np.array([1e14, 1.2e14])
        frequency = np.array([[1e15], [1.2e15]])
        dl = 1e27
        redshift = 0.1
        lambda_observer_frame = np.array([4000.0])

        spectra = sed.blackbody_to_spectrum(
            temperature, r_photosphere, frequency, dl, redshift, lambda_observer_frame)

        self.assertIsInstance(spectra, uu.Quantity)
        self.assertEqual(spectra.unit, uu.erg / uu.cm**2 / uu.s / uu.Angstrom)


class TestBoostedBolometricLuminosity(unittest.TestCase):
    """Test boosted_bolometric_luminosity function"""

    def test_boosted_luminosity(self):
        """Test boosted bolometric luminosity calculation"""
        temperature = 10000.0  # K
        radius = 1e14  # cm
        lambda_cut = 3000.0 * 1e-8  # Convert Angstrom to cm

        L_boosted, L_bb = sed.boosted_bolometric_luminosity(temperature, radius, lambda_cut)

        self.assertGreater(L_boosted, 0)
        self.assertGreater(L_bb, 0)
        self.assertGreater(L_boosted, L_bb)  # Boosted should be larger

    def test_boosted_luminosity_high_temperature(self):
        """Test with high temperature"""
        temperature = 20000.0
        radius = 1e14
        lambda_cut = 3000.0 * 1e-8

        L_boosted, L_bb = sed.boosted_bolometric_luminosity(temperature, radius, lambda_cut)

        self.assertGreater(L_boosted, 0)
        self.assertGreater(L_bb, 0)


class TestRedbackTimeSeriesSource(unittest.TestCase):
    """Test RedbackTimeSeriesSource class"""

    def test_initialization(self):
        """Test RedbackTimeSeriesSource initialization"""
        # Need at least 4 phase points for degree 3 spline
        phase = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        wave = np.array([4000.0, 5000.0, 6000.0])
        flux = np.array([[1e-10, 2e-10, 1.5e-10],
                        [1.2e-10, 2.2e-10, 1.7e-10],
                        [1.1e-10, 2.1e-10, 1.6e-10],
                        [1.3e-10, 2.3e-10, 1.8e-10],
                        [1.0e-10, 2.0e-10, 1.5e-10]])

        source = sed.RedbackTimeSeriesSource(phase=phase, wave=wave, flux=flux)

        self.assertIsNotNone(source)
        self.assertTrue(hasattr(source, 'get_flux_density'))

    @patch('sncosmo.TimeSeriesSource.__init__')
    def test_get_flux_density(self, mock_init):
        """Test get_flux_density method"""
        mock_init.return_value = None

        phase = np.array([0.0, 1.0, 2.0])
        wave = np.array([4000.0, 5000.0])
        flux = np.array([[1e-10, 2e-10], [1.2e-10, 2.2e-10], [1.1e-10, 2.1e-10]])

        source = sed.RedbackTimeSeriesSource(phase=phase, wave=wave, flux=flux)
        source._flux = MagicMock(return_value=1e-10)

        result = source.get_flux_density(1.0, 5000.0)
        source._flux.assert_called_once_with(1.0, 5000.0)


class TestGetCorrectOutputFormatFromSpectra(unittest.TestCase):
    """Test get_correct_output_format_from_spectra function"""

    @patch('redback.sed.RedbackTimeSeriesSource')
    def test_flux_output(self, mock_source_class):
        """Test flux output format"""
        mock_source = MagicMock()
        mock_source.bandmag.return_value = np.array([20.0])
        mock_source_class.return_value = mock_source

        time = np.array([1.0, 2.0])
        time_eval = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])  # Need more rows for spectra[5]
        spectra = np.ones((7, 100)) * 1e-15 * uu.erg / uu.cm**2 / uu.s / uu.Angstrom
        lambda_array = np.linspace(3000, 9000, 100)

        with patch('redback.utils.bandpass_magnitude_to_flux', return_value=1e-10):
            result = sed.get_correct_output_format_from_spectra(
                time, time_eval, spectra, lambda_array,
                output_format='flux', bands='r')

            self.assertIsNotNone(result)

    @patch('redback.sed.RedbackTimeSeriesSource')
    def test_magnitude_output(self, mock_source_class):
        """Test magnitude output format"""
        mock_source = MagicMock()
        mock_source.bandmag.return_value = np.array([20.0])
        mock_source_class.return_value = mock_source

        time = np.array([1.0, 2.0])
        time_eval = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])  # Need more rows for spectra[5]
        spectra = np.ones((7, 100)) * 1e-15 * uu.erg / uu.cm**2 / uu.s / uu.Angstrom
        lambda_array = np.linspace(3000, 9000, 100)

        result = sed.get_correct_output_format_from_spectra(
            time, time_eval, spectra, lambda_array,
            output_format='magnitude', bands='r')

        self.assertIsNotNone(result)

    @patch('redback.sed.RedbackTimeSeriesSource')
    def test_sncosmo_source_output(self, mock_source_class):
        """Test sncosmo_source output format"""
        mock_source = MagicMock()
        mock_source_class.return_value = mock_source

        time = np.array([1.0, 2.0])
        time_eval = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])  # Need more rows for spectra[5]
        spectra = np.ones((7, 100)) * 1e-15 * uu.erg / uu.cm**2 / uu.s / uu.Angstrom
        lambda_array = np.linspace(3000, 9000, 100)

        result = sed.get_correct_output_format_from_spectra(
            time, time_eval, spectra, lambda_array,
            output_format='sncosmo_source')

        self.assertEqual(result, mock_source)

    @patch('redback.sed.RedbackTimeSeriesSource')
    def test_invalid_output_format(self, mock_source_class):
        """Test invalid output format raises error"""
        mock_source = MagicMock()
        mock_source_class.return_value = mock_source

        time = np.array([1.0, 2.0])
        time_eval = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])  # Need more rows for spectra[5]
        spectra = np.ones((7, 100)) * 1e-15 * uu.erg / uu.cm**2 / uu.s / uu.Angstrom
        lambda_array = np.linspace(3000, 9000, 100)

        with self.assertRaises(ValueError):
            sed.get_correct_output_format_from_spectra(
                time, time_eval, spectra, lambda_array,
                output_format='invalid')


if __name__ == '__main__':
    unittest.main()
