import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

import redback

dirname = os.path.dirname(__file__)

import redback.get_data.directory as directory

_original_spec_dir_struct = directory.spectrum_directory_structure
directory.spectrum_directory_structure = lambda transient: "dummy_directory_structure"

class TestSpectrum(unittest.TestCase):

    def setUp(self):
        # Create dummy spectral data
        # Use three wavelengths (in Angstroms) that might cover the optical
        self.angstroms = np.array([4000, 5000, 6000])
        # Fake flux density in erg / s / cm^2 / Angstrom (typical values are small)
        self.flux_density = np.array([1e-17, 2e-17, 3e-17])
        # Assume small errors
        self.flux_density_err = np.array([1e-18, 1e-18, 1e-18])
        # A dummy observation time and a name for the spectrum
        self.time_str = "10d"  # could be a phase string
        self.name = "TestSpec"

    def tearDown(self):
        # Restore the patched directory function if needed.
        directory.spectrum_directory_structure = _original_spec_dir_struct

    def test_initialization_with_time(self):
        spec = redback.transient.Spectrum(self.angstroms, self.flux_density, self.flux_density_err,
                        time=self.time_str, name=self.name)
        self.assertTrue(spec.plot_with_time_label,
                        "When a time is provided, plot_with_time_label should be True.")

    def test_initialization_without_time(self):
        spec = redback.transient.Spectrum(self.angstroms, self.flux_density, self.flux_density_err,
                        name=self.name)  # time defaults to None
        self.assertFalse(spec.plot_with_time_label,
                         "When no time is provided, plot_with_time_label should be False.")

    def test_directory_structure(self):
        spec = redback.transient.Spectrum(self.angstroms, self.flux_density, self.flux_density_err, name=self.name)
        # The __init__ should call redback.get_data.directory.spectrum_directory_structure(name)
        self.assertEqual(spec.directory_structure, "dummy_directory_structure",
                         "Directory structure should be patched to a dummy value.")

    def test_xlabel_property(self):
        spec = redback.transient.Spectrum(self.angstroms, self.flux_density, self.flux_density_err)
        expected_xlabel = r'Wavelength [$\mathrm{\AA}$]'
        self.assertEqual(spec.xlabel, expected_xlabel,
                         "The xlabel property did not match the expected value.")

    def test_ylabel_property(self):
        spec = redback.transient.Spectrum(self.angstroms, self.flux_density, self.flux_density_err)
        expected_ylabel = r'Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}$)'
        self.assertEqual(spec.ylabel, expected_ylabel,
                         "The ylabel property did not match the expected value.")

class TestTransient(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.time_err = np.array([0.2, 0.3, 0.4])
        self.y = np.array([3, 4, 2])
        self.y_err = np.sqrt(self.y)
        self.redshift = 0.75
        self.data_mode = 'counts'
        self.name = "GRB123456"
        self.photon_index = 2
        self.use_phase_model = False
        self.transient = redback.transient.transient.Transient(
            time=self.time, time_err=self.time_err, counts=self.y,
            redshift=self.redshift, data_mode=self.data_mode, name=self.name,
            photon_index=self.photon_index, use_phase_model=self.use_phase_model)

    def tearDown(self) -> None:
        del self.time
        del self.time_err
        del self.y
        del self.y_err
        del self.redshift
        del self.data_mode
        del self.name
        del self.photon_index
        del self.use_phase_model
        del self.transient

    def test_ttes_data_mode_setting(self):
        bin_ttes = MagicMock(return_value=(self.time, self.y))
        ttes = np.arange(0, 1, 1000)
        self.data_mode = 'ttes'
        self.bin_size = 0.1
        self.transient = redback.transient.transient.Transient(
            ttes=ttes, redshift=self.redshift, data_mode=self.data_mode, name=self.name,
            photon_index=self.photon_index, bin_ttes=bin_ttes)
        bin_ttes.assert_called_once()

    def test_data_mode_switches(self):
        self.assertTrue(self.transient.counts_data)
        self.assertFalse(self.transient.luminosity_data)
        self.assertFalse(self.transient.flux_data)
        self.assertFalse(self.transient.flux_density_data)
        self.assertFalse(self.transient.magnitude_data)
        self.assertFalse(self.transient.tte_data)

    def test_set_data_mode_switch(self):
        self.transient.flux_data = True
        self.assertTrue(self.transient.flux_data)
        self.assertFalse(self.transient.counts_data)

    def test_get_time_via_x(self):
        self.assertTrue(np.array_equal(self.time, self.transient.x))
        self.assertTrue(np.array_equal(self.time_err, self.transient.x_err))

    def test_get_time_via_x_luminosity_data(self):
        new_times = np.array([1, 2, 3])
        new_time_errs = np.array([0.1, 0.2, 0.3])
        self.transient.time_rest_frame = new_times
        self.transient.time_rest_frame_err = new_time_errs
        self.transient.data_mode = "luminosity"
        self.assertTrue(np.array_equal(new_times, self.transient.x))
        self.assertTrue(np.array_equal(new_time_errs, self.transient.x_err))

    def test_x_same_as_time(self):
        self.assertTrue(np.array_equal(self.transient.x, self.transient.time))

    def test_xerr_same_as_time_err(self):
        self.assertTrue(np.array_equal(self.transient.x_err, self.transient.time_err))

    def test_set_use_phase_model(self):
        self.assertFalse(self.transient.use_phase_model)

    def test_xlabel(self):
        self.assertEqual(r"Time since explosion [days]", self.transient.xlabel)
        self.transient.use_phase_model = True
        self.assertEqual(r"Time [MJD]", self.transient.xlabel)

    def test_ylabel(self):
        self.assertEqual(r'Counts', self.transient.ylabel)
        self.transient.luminosity_data = True
        self.assertEqual(r'Luminosity [$10^{50}$ erg s$^{-1}$]', self.transient.ylabel)
        self.transient.magnitude_data = True
        self.assertEqual(r'Magnitude', self.transient.ylabel)
        self.transient.flux_data = True
        self.assertEqual(r'Flux [erg cm$^{-2}$ s$^{-1}$]', self.transient.ylabel)
        self.transient.flux_density_data = True
        self.assertEqual(r'Flux density [mJy]', self.transient.ylabel)
        self.transient.flux_density_data = False
        with self.assertRaises(ValueError):
            _ = self.transient.ylabel

    def test_use_phase_model_time_attribute(self):
        self.transient = redback.transient.transient.Transient(
            time_mjd=self.time, time_mjd_err=self.time_err, counts=self.y, redshift=self.redshift,
            data_mode=self.data_mode, name=self.name, photon_index=self.photon_index,
            use_phase_model=True)
        self.assertTrue(np.array_equal(self.transient.time_mjd, self.transient.x))
        self.assertTrue(np.array_equal(self.transient.time_mjd_err, self.transient.x_err))

    def test_set_x(self):
        new_x = np.array([2, 3, 4])
        self.transient.x = new_x
        self.assertTrue(np.array_equal(new_x, self.transient.x))
        self.assertTrue(np.array_equal(new_x, self.transient.time))

    def test_set_x_err(self):
        new_x_err = np.array([3, 4, 5])
        self.transient.x_err = new_x_err
        self.assertTrue(np.array_equal(new_x_err, self.transient.x_err))
        self.assertTrue(np.array_equal(new_x_err, self.transient.time_err))

    def test_set_y(self):
        new_y = np.array([7, 8, 9])
        self.transient.y = new_y
        self.assertTrue(np.array_equal(new_y, self.transient.y))
        self.assertTrue(np.array_equal(new_y, self.transient.counts))

    def test_set_y_err(self):
        new_y_err = np.array([7, 8, 9])
        self.transient.y_err = new_y_err
        self.assertTrue(np.array_equal(new_y_err, self.transient.y_err))
        self.assertTrue(np.array_equal(new_y_err, self.transient.counts_err))

    def test_y_same_as_counts(self):
        self.assertTrue(np.array_equal(self.transient.y, self.transient.counts))

    def test_yerr_same_as_counts(self):
        self.assertTrue(np.array_equal(self.transient.y_err, self.transient.counts_err))

    def test_redshift(self):
        self.assertEqual(self.redshift, self.transient.redshift)

    def test_get_data_mode(self):
        self.assertEqual(self.data_mode, self.transient.data_mode)

    def test_set_data_mode(self):
        new_data_mode = "luminosity"
        self.transient.data_mode = new_data_mode
        self.assertEqual(new_data_mode, self.transient.data_mode)

    def test_set_illegal_data_mode(self):
        with self.assertRaises(ValueError):
            self.transient.data_mode = "abc"

    def test_plot_lightcurve(self):
        pass
        # self.transient.plot_lightcurve(model=None)

    def test_plot_data(self):
        pass
        # self.transient.plot_data()


class TestOpticalTransient(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.time_err = np.array([0.2, 0.3, 0.4])
        self.y = np.array([3, 4, 2])
        self.y_err = np.sqrt(self.y)
        self.redshift = 0.75
        self.data_mode = 'flux_density'
        self.name = "SN2000A"
        self.photon_index = 2
        self.use_phase_model = False
        self.bands = np.array(['i', 'g', 'g'])
        self.active_bands = np.array(['g'])
        self.transient = redback.transient.transient.OpticalTransient(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            redshift=self.redshift, data_mode=self.data_mode, name=self.name,
            photon_index=self.photon_index, use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands)

    def tearDown(self) -> None:
        del self.time
        del self.time_err
        del self.y
        del self.y_err
        del self.redshift
        del self.data_mode
        del self.name
        del self.photon_index
        del self.use_phase_model
        del self.bands
        del self.active_bands
        del self.transient

    def test_load_data_magnitude(self):
        name = "optical_transient_test_data"
        transient_dir = f"{dirname}/data"
        processed_file_path = f"{transient_dir}/{name}.csv"
        data_mode = "magnitude"
        time_days, time_mjd, magnitude, magnitude_err, bands, system = \
            self.transient.load_data(processed_file_path=processed_file_path, data_mode=data_mode)
        expected_time_days = np.array([0.4813999999969383, 0.49020000000018626])
        expected_time_mjd = np.array([57982.9814, 57982.9902])
        expected_magnitude = np.array([17.48, 18.26])
        expected_magnitude_err = np.array([0.02, 0.15])
        expected_bands = np.array(["i", "H"])
        expected_system = np.array(["AB", "AB"])
        self.assertTrue(np.allclose(expected_time_days, time_days))
        self.assertTrue(np.allclose(expected_time_mjd, time_mjd))
        self.assertTrue(np.allclose(expected_magnitude, magnitude))
        self.assertTrue(np.allclose(expected_magnitude_err, magnitude_err))
        self.assertTrue(np.array_equal(expected_bands, bands))
        self.assertTrue(np.array_equal(expected_system, system))

    def test_load_data_flux_density(self):
        name = "optical_transient_test_data"
        transient_dir = f"{dirname}/data"
        data_mode = "flux_density"
        processed_file_path = f"{transient_dir}/{name}.csv"

        time_days, time_mjd, flux_density, flux_density_err, bands, system = \
            self.transient.load_data(processed_file_path=processed_file_path, data_mode=data_mode)
        expected_time_days = np.array([0.4813999999969383, 0.49020000000018626])
        expected_time_mjd = np.array([57982.9814, 57982.9902])
        expected_flux_density = np.array([0.36982817978026444, 0.1803017740859559])
        expected_flux_density_err = np.array([0.006812898591418732, 0.024911116226263914])
        expected_bands = np.array(["i", "H"])
        expected_system = np.array(["AB", "AB"])
        self.assertTrue(np.allclose(expected_time_days, time_days))
        self.assertTrue(np.allclose(expected_time_mjd, time_mjd))
        self.assertTrue(np.allclose(expected_flux_density, flux_density))
        self.assertTrue(np.allclose(expected_flux_density_err, flux_density_err))
        self.assertTrue(np.array_equal(expected_bands, bands))
        self.assertTrue(np.array_equal(expected_system, system))

    def test_load_data_all(self):
        name = "optical_transient_test_data"
        transient_dir = f"{dirname}/data"
        processed_file_path = f"{transient_dir}/{name}.csv"
        data_mode = "all"
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, flux, flux_err, bands, system = \
            self.transient.load_data(processed_file_path=processed_file_path, data_mode=data_mode)
        expected_time_days = np.array([0.4813999999969383, 0.49020000000018626])
        expected_time_mjd = np.array([57982.9814, 57982.9902])
        expected_flux_density = np.array([0.36982817978026444, 0.1803017740859559])
        expected_flux_density_err = np.array([0.006812898591418732, 0.024911116226263914])
        expected_flux = np.array([0.36982817978026444, 0.1803017740859559])
        expected_flux_err = np.array([0.006812898591418732, 0.024911116226263914])
        expected_magnitude = np.array([17.48, 18.26])
        expected_magnitude_err = np.array([0.02, 0.15])
        expected_bands = np.array(["i", "H"])
        expected_system = np.array(["AB", "AB"])
        self.assertTrue(np.allclose(expected_time_days, time_days))
        self.assertTrue(np.allclose(expected_time_mjd, time_mjd))
        self.assertTrue(np.allclose(expected_flux_density, flux_density))
        self.assertTrue(np.allclose(expected_flux_density_err, flux_density_err))
        self.assertTrue(np.allclose(expected_magnitude, magnitude))
        self.assertTrue(np.allclose(expected_magnitude_err, magnitude_err))
        self.assertTrue(np.allclose(expected_flux, flux))
        self.assertTrue(np.allclose(expected_flux_err, flux_err))
        self.assertTrue(np.array_equal(expected_bands, bands))
        self.assertTrue(np.array_equal(expected_system, system))

    def test_get_from_open_access_catalogue(self):
        with mock.patch("redback.transient.transient.OpticalTransient.load_data") as m:
            expected_time_days = np.array([0.4813999999969383, 0.49020000000018626])
            expected_time_mjd = np.array([57982.9814, 57982.9902])
            expected_flux_density = np.array([0.36982817978026444, 0.1803017740859559])
            expected_flux_density_err = np.array([0.006812898591418732, 0.024911116226263914])
            expected_flux = np.array([0.36982817978026444, 0.1803017740859559])
            expected_flux_err = np.array([0.006812898591418732, 0.024911116226263914])
            expected_magnitude = np.array([17.48, 18.26])
            expected_magnitude_err = np.array([0.02, 0.15])
            expected_bands = np.array(["i", "H"])
            expected_system = np.array(["AB", "AB"])
            m.return_value = \
                expected_time_days, expected_time_mjd, expected_flux_density, expected_flux_density_err, \
                expected_magnitude, expected_magnitude_err, expected_flux, expected_flux_err, \
                expected_bands, expected_system
            name = "test"
            transient = redback.transient.transient.OpticalTransient.from_open_access_catalogue(name=name)
            self.assertTrue(transient.magnitude_data)
            self.assertEqual(name, transient.name)
            self.assertTrue(np.allclose(expected_time_days, transient.time))
            self.assertTrue(np.allclose(expected_time_mjd, transient.time_mjd))
            self.assertTrue(np.allclose(expected_flux_density, transient.flux_density))
            self.assertTrue(np.allclose(expected_flux_density_err, transient.flux_density_err))
            self.assertTrue(np.allclose(expected_flux, transient.flux))
            self.assertTrue(np.allclose(expected_flux_err, transient.flux_err))
            self.assertTrue(np.allclose(expected_magnitude, transient.magnitude))
            self.assertTrue(np.allclose(expected_magnitude_err, transient.magnitude_err))
            self.assertTrue(np.array_equal(expected_bands, transient.bands))
            self.assertTrue(np.array_equal(expected_system, transient.system))

    def test_set_active_bands(self):
        self.assertTrue(np.array_equal(np.array(self.active_bands), self.transient.active_bands))

    def test_set_active_bands_all(self):
        self.transient = redback.transient.transient.OpticalTransient(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            redshift=self.redshift, data_mode=self.data_mode, name=self.name,
            photon_index=self.photon_index, use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands='all')
        self.assertTrue(np.array_equal(np.array(['g', 'i']), self.transient.active_bands))

    def test_set_frequencies_from_bands(self):
        expected = [1, 2, 2]
        bands_to_frequency = MagicMock(return_value=expected)
        self.transient = redback.transient.transient.OpticalTransient(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            redshift=self.redshift, data_mode=self.data_mode, name=self.name,
            photon_index=self.photon_index, use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, bands_to_frequency=bands_to_frequency)
        self.assertTrue(np.array_equal(expected, self.transient.frequency))
        bands_to_frequency.assert_called_once()

    def test_set_frequencies_default(self):
        frequency = np.array([1, 2, 2])
        self.transient = redback.transient.transient.OpticalTransient(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            redshift=self.redshift, data_mode=self.data_mode, name=self.name,
            photon_index=self.photon_index, use_phase_model=self.use_phase_model, bands=self.bands,
            frequency=frequency, active_bands=self.active_bands)
        self.assertTrue(np.array_equal(frequency, self.transient.frequency))

    def test_get_filtered_data(self):
        filtered_x, filtered_x_err, filtered_y, filtered_y_err = self.transient.get_filtered_data()
        expected_x = self.time[1:]
        expected_x_err = self.time_err[1:]
        expected_y = self.y[1:]
        expected_y_err = self.y_err[1:]
        self.assertTrue(np.array_equal(expected_x, filtered_x))
        self.assertTrue(np.array_equal(expected_x_err, filtered_x_err))
        self.assertTrue(np.array_equal(expected_y, filtered_y))
        self.assertTrue(np.array_equal(expected_y_err, filtered_y_err))

    def test_get_filtered_data_no_x_err(self):
        self.transient.x_err = None
        _, filtered_x_err, _, _ = self.transient.get_filtered_data()
        self.assertIsNone(filtered_x_err)

    def test_get_filtered_data_illegal_data_mode(self):
        with self.assertRaises(ValueError):
            self.transient.luminosity_data = True
            self.transient.get_filtered_data()

    def test_meta_data_not_available(self):
        self.assertIsNone(self.transient.meta_data)

    @mock.patch("pandas.read_csv")
    def test_meta_data_from_csv(self, read_csv):
        self.transient.directory_structure = redback.get_data.directory.DirectoryStructure(
            directory_path='data', raw_file_path=None, processed_file_path=None)
        expected = dict(a=1)
        read_csv.return_value = expected
        self.transient._set_data()
        self.assertDictEqual(expected, self.transient.meta_data)

    def test_transient_dir(self):
        with mock.patch('redback.get_data.directory.open_access_directory_structure') as m:
            expected = 'expected'
            m.return_value = expected, '_', '_'
            self.assertEqual(expected, self.transient.transient_dir)

    def test_unique_bands(self):
        expected = np.array(['g', 'i'])
        self.assertTrue(np.array_equal(expected, self.transient.unique_bands))

    def test_list_of_band_indices(self):
        expected = [np.array([1, 2]), np.array([0])]
        self.assertTrue(np.array_equal(expected[0], self.transient.list_of_band_indices[0]))
        self.assertTrue(np.array_equal(expected[1], self.transient.list_of_band_indices[1]))

    def test_default_colors(self):
        expected = ["g", "r", "i", "z", "y", "J", "H", "K"]
        self.assertListEqual(expected, self.transient.default_filters)

    def test_get_colors(self):
        with mock.patch('matplotlib.cm.rainbow') as m:
            expected = 'rainbow'
            m.return_value = expected
            self.assertEqual(expected, self.transient.get_colors(filters=['a', 'b']))

    def test_estimate_bb_params_effective(self):
        """Test that estimate_bb_params returns a DataFrame with expected columns and positive values
        in effective flux density mode."""
        # Create a transient instance with flux_density data.
        new_time = np.array([10, 10.1, 10.2, 10.3, 10.4])
        new_flux = np.array([1e4, 1.05e4, 1.1e4, 1.05e4, 1e4])
        new_flux_err = np.array([100, 100, 100, 100, 100])
        new_freq = np.array([5e14, 6e14, 7e14, 8e14, 9e14])
        transient_bb = redback.transient.OpticalTransient(
            time=new_time, flux_density=new_flux,
            redshift=0.1, data_mode="flux_density", name="TestBB",
            frequency=new_freq, use_phase_model=False)
        # Monkey-patch get_filtered_data to return our simulated data.
        transient_bb.get_filtered_data = lambda: (new_time, np.zeros(5), new_flux, new_flux_err)
        df_bb = transient_bb.estimate_bb_params(distance=1e27, bin_width=1.0, min_filters=3)
        self.assertIsNotNone(df_bb, "Expected a DataFrame when sufficient data are provided.")
        self.assertIsInstance(df_bb, pd.DataFrame, "The output must be a DataFrame.")
        for col in ['epoch_times', 'temperature', 'radius', 'temp_err', 'radius_err']:
            self.assertIn(col, df_bb.columns, f"Column '{col}' is missing in the DataFrame.")
        # Check that the fitted temperature and radius are positive.
        self.assertGreater(df_bb['temperature'].iloc[0], 0, "Temperature should be positive.")
        self.assertGreater(df_bb['radius'].iloc[0], 0, "Radius should be positive.")

    def test_estimate_bolometric_luminosity_no_corrections(self):
        """Test that a bolometric luminosity is computed from the BB parameters (without boost or extinction)."""
        # Create a fake DataFrame of blackbody parameters.
        fake_df = pd.DataFrame({
            "epoch_times": [10.5],
            "temperature": [1e4],  # Kelvin
            "radius": [1e15],  # cm
            "temp_err": [500],
            "radius_err": [1e14]
        })
        transient_bb = redback.transient.OpticalTransient(
            time=np.array([10, 10.1, 10.2, 10.3, 10.4]),
            time_err=np.array([0.1] * 5),
            flux_density=np.array([1e4, 1.05e4, 1.1e4, 1.05e4, 1e4]),
            redshift=0.1, data_mode="flux_density", name="TestBB",
            photon_index=2, use_phase_model=False)
        # Monkey-patch estimate_bb_params to return our fake blackbody parameters.
        transient_bb.estimate_bb_params = lambda **kwargs: fake_df
        df_bol = transient_bb.estimate_bolometric_luminosity(distance=1e27, bin_width=1.0, min_filters=3)
        self.assertIsNotNone(df_bol, "A valid bolometric luminosity DataFrame is expected.")
        for col in ['lum_bol', 'lum_bol_err', 'lum_bol_bb', 'time_rest_frame']:
            self.assertIn(col, df_bol.columns, f"Column '{col}' is missing in the bolometric DataFrame.")
        self.assertGreater(df_bol['lum_bol'].iloc[0], 0, "Bolometric luminosity should be positive.")

    def test_estimate_bolometric_luminosity_with_boost_extinction(self):
        """Test that providing lambda_cut and A_ext yields a DataFrame with the boost/extinction corrections applied."""
        fake_df = pd.DataFrame({
            "epoch_times": [10.5],
            "temperature": [1e4],
            "radius": [1e15],
            "temp_err": [500],
            "radius_err": [1e14]
        })
        transient_bb = redback.transient.OpticalTransient(
            time=np.array([10, 10.1, 10.2, 10.3, 10.4]),
            time_err=np.array([0.1] * 5),
            flux_density=np.array([1e4, 1.05e4, 1.1e4, 1.05e4, 1e4]),
            redshift=0.1, data_mode="flux_density", name="TestBB",
            photon_index=2, use_phase_model=False)
        transient_bb.estimate_bb_params = lambda **kwargs: fake_df
        # Set lambda_cut (in angstroms) and an extinction A_ext in magnitudes.
        df_bol = transient_bb.estimate_bolometric_luminosity(
            distance=1e27, bin_width=1.0, min_filters=3, lambda_cut=3000, A_ext=0.5)
        self.assertIsNotNone(df_bol, "A DataFrame with boost and extinction corrections should be returned.")
        self.assertIn('lum_bol', df_bol.columns)
        # Check that luminosity is positive.
        self.assertGreater(df_bol['lum_bol'].iloc[0], 0, "Corrected bolometric luminosity should be positive.")


class TestAfterglow(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.time_err = np.array([0.2, 0.3, 0.4])
        self.y = np.array([3, 4, 2])
        self.y_err = np.sqrt(self.y)
        self.data_mode = 'flux'
        self.name = "GRB070809"
        self.use_phase_model = False
        self.bands = np.array(['i', 'g', 'g'])
        self.active_bands = np.array(['g'])
        self.FluxToLuminosityConverter = MagicMock()
        self.Truncator = MagicMock()
        self.sgrb = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode=self.data_mode, name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, FluxToLuminosityConverter=self.FluxToLuminosityConverter,
            Truncator=self.Truncator)
        self.sgrb_luminosity = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode="luminosity", name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, FluxToLuminosityConverter=self.FluxToLuminosityConverter,
            Truncator=self.Truncator)
        self.sgrb_flux_density = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode="flux_density", name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, FluxToLuminosityConverter=self.FluxToLuminosityConverter,
            Truncator=self.Truncator)
        self.sgrb_not_existing = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode=self.data_mode, name="123456",
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, FluxToLuminosityConverter=self.FluxToLuminosityConverter,
            Truncator=self.Truncator)
        self.sgrb_magnitude = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, magnitude=self.y, magnitude_err=self.y_err,
            data_mode="magnitude", name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, FluxToLuminosityConverter=self.FluxToLuminosityConverter,
            Truncator=self.Truncator)
        self.sgrb_all_active_bands = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode=self.data_mode, name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands='all', FluxToLuminosityConverter=self.FluxToLuminosityConverter, Truncator=self.Truncator)

    def tearDown(self) -> None:
        del self.time
        del self.time_err
        del self.y
        del self.y_err
        del self.data_mode
        del self.name
        del self.use_phase_model
        del self.bands
        del self.active_bands
        del self.sgrb
        del self.sgrb_not_existing
        del self.sgrb_magnitude
        del self.sgrb_all_active_bands
        del self.FluxToLuminosityConverter

    def test_stripped_name(self):
        expected = "070809"
        self.assertEqual(expected, self.sgrb._stripped_name)

    def test_truncate(self):
        expected_x = 0
        expected_x_err = 1
        expected_y = 2
        expected_yerr = 3
        return_value = expected_x, expected_x_err, expected_y, expected_yerr
        truncator = MagicMock(return_value=MagicMock(truncate=MagicMock(return_value=return_value)))
        self.sgrb.Truncator = truncator
        self.sgrb.truncate()
        self.assertListEqual(
            [expected_x, expected_x_err, expected_y, expected_yerr],
            [self.sgrb.x, self.sgrb.x_err, self.sgrb.y, self.sgrb.y_err])

    def test_set_active_bands(self):
        self.assertTrue(np.array_equal(np.array(self.active_bands), self.sgrb.active_bands))

    def test_set_active_bands_all(self):
        self.assertTrue(np.array_equal(np.array(['g', 'i']), self.sgrb_all_active_bands.active_bands))

    def test_set_frequencies_from_bands(self):
        expected = [1, 2, 2]
        bands_to_frequency = MagicMock(return_value=expected)
        self.sgrb = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode=self.data_mode, name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands, bands_to_frequency=bands_to_frequency)
        self.assertTrue(np.array_equal(expected, self.sgrb.frequency))
        bands_to_frequency.assert_called_once()

    def test_set_frequencies_default(self):
        frequency = np.array([1, 2, 2])
        self.sgrb = redback.transient.afterglow.SGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode=self.data_mode, name=self.name,
            use_phase_model=self.use_phase_model, bands=self.bands,
            frequency=frequency, active_bands=self.active_bands)
        self.assertTrue(np.array_equal(frequency, self.sgrb.frequency))

    def test_get_filtered_data(self):
        filtered_x, filtered_x_err, filtered_y, filtered_y_err = self.sgrb_magnitude.get_filtered_data()
        expected_x = self.time[1:]
        expected_x_err = self.time_err[1:]
        expected_y = self.y[1:]
        expected_y_err = self.y_err[1:]
        self.assertTrue(np.array_equal(expected_x, filtered_x))
        self.assertTrue(np.array_equal(expected_x_err, filtered_x_err))
        self.assertTrue(np.array_equal(expected_y, filtered_y))
        self.assertTrue(np.array_equal(expected_y_err, filtered_y_err))

    def test_get_filtered_data_no_x_err(self):
        self.sgrb_magnitude.x_err = None
        _, filtered_x_err, _, _ = self.sgrb_magnitude.get_filtered_data()
        self.assertIsNone(filtered_x_err)

    def test_get_filtered_data_illegal_data_mode(self):
        self.sgrb.data_mode = "luminosity"
        with self.assertRaises(ValueError):
            self.sgrb.get_filtered_data()

    def test_event_table(self):
        expected = "/tables/SGRB_table.txt"
        self.assertIn(expected, self.sgrb.event_table)

    def test_meta_data_from_csv(self):
        with mock.patch("pandas.read_csv") as m:
            field_name = 'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'
            data_frame = pd.DataFrame.from_dict({field_name: [0, 1, np.nan]})
            m.return_value = data_frame
            expected = np.array([0, 1, 0])
            self.sgrb._set_data()
            self.assertTrue(np.array_equal(expected, np.array(self.sgrb.meta_data[field_name])))

    def test_photon_index(self):
        self.assertEqual(1.69, self.sgrb.photon_index)

    def test_photon_index_missing(self):
        self.assertTrue(np.isnan(self.sgrb_not_existing.photon_index))

    def test_redshift(self):
        self.assertTrue(np.isnan(self.sgrb.redshift))

    def test_redshift_missing(self):
        self.assertTrue(np.isnan(self.sgrb_not_existing.redshift))

    def test_t90(self):
        lgrb = redback.transient.afterglow.LGRB(
            time=self.time, time_err=self.time_err, flux_density=self.y, flux_density_err=self.y_err,
            data_mode=self.data_mode, name="210318B",
            use_phase_model=self.use_phase_model, bands=self.bands,
            active_bands=self.active_bands)
        self.assertEqual(14.95, lgrb.t90)

    def test_t90_missing(self):
        self.assertTrue(np.isnan(self.sgrb_not_existing.t90))

    def test_flux_to_luminosity_luminosity(self):
        self.sgrb_luminosity.FluxToLuminosityConverter.convert_flux_to_luminosity = MagicMock()
        self.sgrb_luminosity.analytical_flux_to_luminosity()
        self.sgrb_luminosity.FluxToLuminosityConverter.convert_flux_to_luminosity.assert_not_called()

    def test_flux_to_luminosity_flux_density(self):
        self.sgrb_flux_density.FluxToLuminosityConverter.convert_flux_to_luminosity = MagicMock()
        self.sgrb_flux_density.analytical_flux_to_luminosity()
        self.sgrb_flux_density.FluxToLuminosityConverter.convert_flux_to_luminosity.assert_not_called()
        self.assertTrue(self.sgrb_flux_density.flux_density_data)

    def test_flux_to_luminosity_nan_redshift(self):
        return_value = np.array([0, 1, 2, 3]), np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), \
                       np.array([2, 3, 4, 5]), np.array([[3, 4, 5, 6], [3, 4, 5, 6]])
        converter = MagicMock(return_value=MagicMock(convert_flux_to_luminosity=MagicMock(return_value=return_value)))
        self.sgrb.FluxToLuminosityConverter = converter
        self.sgrb.redshift = np.nan
        self.sgrb.analytical_flux_to_luminosity()
        converter.assert_called_once()
        self.assertTrue(self.sgrb.luminosity_data)
        self.assertTrue(np.array_equal(return_value[0], self.sgrb.x))
        self.assertTrue(np.array_equal(return_value[1], self.sgrb.x_err))
        self.assertTrue(np.array_equal(return_value[2], self.sgrb.y))
        self.assertTrue(np.array_equal(return_value[3], self.sgrb.y_err))

    def test_flux_to_luminosity_none_redshift(self):
        return_value = np.array([0, 1, 2, 3]), np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), \
                       np.array([2, 3, 4, 5]), np.array([[3, 4, 5, 6], [3, 4, 5, 6]])
        converter = MagicMock(return_value=MagicMock(convert_flux_to_luminosity=MagicMock(return_value=return_value)))
        self.sgrb.FluxToLuminosityConverter = converter
        self.sgrb.redshift = None
        self.sgrb.analytical_flux_to_luminosity()

    def test_flux_to_luminosity_real_redshift(self):
        return_value = np.array([0, 1, 2, 3]), np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), \
                       np.array([2, 3, 4, 5]), np.array([[3, 4, 5, 6], [3, 4, 5, 6]])
        converter = MagicMock(return_value=MagicMock(convert_flux_to_luminosity=MagicMock(return_value=return_value)))
        self.sgrb.FluxToLuminosityConverter = converter
        self.sgrb.redshift = 0.5
        self.sgrb.analytical_flux_to_luminosity()
        self.assertTrue(self.sgrb.luminosity_data)
        self.assertTrue(np.array_equal(return_value[0], self.sgrb.x))
        self.assertTrue(np.array_equal(return_value[1], self.sgrb.x_err))
        self.assertTrue(np.array_equal(return_value[2], self.sgrb.y))
        self.assertTrue(np.array_equal(return_value[3], self.sgrb.y_err))
        self.assertEqual(0.5, self.sgrb.redshift)

    def test_numerical_flux_to_luminosity(self):
        return_value = np.array([0, 1, 2, 3]), np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), \
                       np.array([2, 3, 4, 5]), np.array([[3, 4, 5, 6], [3, 4, 5, 6]])
        converter = MagicMock(return_value=MagicMock(convert_flux_to_luminosity=MagicMock(return_value=return_value)))
        self.sgrb.FluxToLuminosityConverter = converter
        self.sgrb.redshift = 0.5
        self.sgrb.numerical_flux_to_luminosity(counts_to_flux_absorbed=1, counts_to_flux_unabsorbed=1)
        self.assertTrue(self.sgrb.luminosity_data)
        self.assertTrue(np.array_equal(return_value[0], self.sgrb.x))
        self.assertTrue(np.array_equal(return_value[1], self.sgrb.x_err))
        self.assertTrue(np.array_equal(return_value[2], self.sgrb.y))
        self.assertTrue(np.array_equal(return_value[3], self.sgrb.y_err))
        self.assertEqual(0.5, self.sgrb.redshift)


    def test_analytical_flux_to_luminosity(self):
        pass


class TestTrunctator(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-1, 0., 0.2, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5])
        self.x_err = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4],
                               [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]).T
        self.y = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
        self.y_err = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4],
                               [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]).T
        self.truncate_method = 'default'
        self.truncator = redback.transient.afterglow.Truncator(
            x=self.x, x_err=self.x_err, y=self.y, y_err=self.y_err,
            time=self.x, time_err=self.x_err, truncate_method=self.truncate_method)

    def tearDown(self):
        del self.x
        del self.x_err
        del self.y
        del self.y_err
        del self.truncate_method
        del self.truncator

    def test_truncate_left_of_max(self):
        x, x_err, y, y_err = self.truncator.truncate_left_of_max()
        expected_x = np.array([0.8, 1.2, 1.5, 2.0, 2.5])
        expected_x_err = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]).T
        expected_y = np.array([4, 3, 2, 1, 0])
        expected_y_err = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]).T
        self.assertTrue(np.array_equal(expected_x, x))
        self.assertTrue(np.array_equal(expected_x_err, x_err))
        self.assertTrue(np.array_equal(expected_y, y))
        self.assertTrue(np.array_equal(expected_y_err, y_err))

    def test_truncate_default(self):
        x, x_err, y, y_err = self.truncator.truncate_default()
        expected_x = np.array([2.5])
        expected_x_err = np.array([[0.9, 0.9]]).T
        expected_y = np.array([0])
        expected_y_err = np.array([[0.9, 0.9]]).T
        self.assertTrue(np.array_equal(expected_x, x))
        self.assertTrue(np.array_equal(expected_x_err, x_err))
        self.assertTrue(np.array_equal(expected_y, y))
        self.assertTrue(np.array_equal(expected_y_err, y_err))

    def test_truncate_prompt_time_error(self):
        x, x_err, y, y_err = self.truncator.truncate_prompt_time_error()
        expected_x = np.array([2.0, 2.5])
        expected_x_err = np.array([[0.8, 0.8], [0.9, 0.9]]).T
        expected_y = np.array([1, 0])
        expected_y_err = np.array([[0.8, 0.8], [0.9, 0.9]]).T
        self.assertTrue(np.array_equal(expected_x, x))
        self.assertTrue(np.array_equal(expected_x_err, x_err))
        self.assertTrue(np.array_equal(expected_y, y))
        self.assertTrue(np.array_equal(expected_y_err, y_err))

    def test_truncate_with_prompt_time_error(self):
        with mock.patch.object(redback.transient.afterglow.Truncator, 'truncate_prompt_time_error') as m:
            self.truncator.truncate_method = 'prompt_time_error'
            self.truncator.truncate()
            m.assert_called_once()

    def test_truncate_with_left_of_max(self):
        with mock.patch.object(redback.transient.afterglow.Truncator, 'truncate_left_of_max') as m:
            self.truncator.truncate_method = 'left_of_max'
            self.truncator.truncate()
            m.assert_called_once()

    def test_truncate_with_default(self):
        with mock.patch.object(redback.transient.afterglow.Truncator, 'truncate_default') as m:
            self.truncator.truncate_method = 'default'
            self.truncator.truncate()
            m.assert_called_once()

    def test_truncate_with_default_with_other_string(self):
        with mock.patch.object(redback.transient.afterglow.Truncator, 'truncate_default') as m:
            self.truncator.truncate_method = 'other_string'
            self.truncator.truncate()
            m.assert_called_once()


class TestFluxToLuminosityConversion(unittest.TestCase):

    def setUp(self):
        self.redshift = 0.5
        self.photon_index = 2
        self.time = np.array([1, 2, 3])
        self.time_err = np.array([[0.1, 0.1], [0.2, 0.2], [0.1, 0.1]]).T
        self.flux = np.array([3.3, 4.4, 5.5])
        self.flux_err = np.array([0.33, 0.44, 0.55])
        self.counts_to_flux_absorbed = 2
        self.counts_to_flux_unabsorbed = 0.5
        self.conversion_method = "analytical"
        self.converter = redback.transient.afterglow.FluxToLuminosityConverter(
            redshift=self.redshift, photon_index=self.photon_index, time=self.time, time_err=self.time_err,
            flux=self.flux, flux_err=self.flux_err, counts_to_flux_absorbed=self.counts_to_flux_absorbed,
            counts_to_flux_unabsorbed=self.counts_to_flux_unabsorbed, conversion_method=self.conversion_method)

    def tearDown(self):
        del self.redshift
        del self.photon_index
        del self.time
        del self.time_err
        del self.flux
        del self.flux_err
        del self.counts_to_flux_absorbed
        del self.counts_to_flux_unabsorbed
        del self.conversion_method
        del self.converter

    def test_counts_to_flux_fraction(self):
        expected = 0.25
        self.assertEqual(expected, self.converter.counts_to_flux_fraction)

    def test_luminosity_distance(self):
        with mock.patch('astropy.cosmology.Planck18') as m:
            planck18_return_object = MagicMock()
            m.luminosity_distance.return_value = planck18_return_object
            # Assuming self.converter has a method that gets the luminosity distance
            result = self.converter.luminosity_distance  # Call the method
            self.assertEqual(result, 9.009021261306723e+27)  # Check the value

    def test_get_isotropic_bolometric_flux(self):
        expected = self.converter.luminosity_distance ** 2 * 4 * np.pi
        isotropic_bolometric_flux = self.converter.get_isotropic_bolometric_flux(k_corr=1)
        self.assertEqual(expected, isotropic_bolometric_flux)

    def test_get_analytical_k_correction(self):
        expected = (1 + self.redshift) ** (self.photon_index - 2)
        actual = self.converter.get_k_correction()
        self.assertEqual(expected, actual)

    def test_convert_flux_to_luminosity(self):
        x, x_err, y, y_err = self.converter.convert_flux_to_luminosity()
        self.assertEqual(len(self.time), len(x))
        self.assertEqual(len(self.time_err), len(x_err))
        self.assertEqual(len(self.flux), len(y))
        self.assertEqual(len(self.flux_err), len(y_err))
        self.assertTrue(np.array_equal(self.converter.time_rest_frame, x))
        self.assertTrue(np.array_equal(self.converter.time_rest_frame_err, x_err))
        self.assertTrue(np.array_equal(self.converter.Lum50, y))
        self.assertTrue(np.array_equal(self.converter.Lum50_err, y_err))

class TestLoadTransient(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_file_path = "test_data.csv"
        self.mock_data = {
            "time (days)": [1.0, 2.0, 3.0],
            "time": [2450000.5, 2450001.5, 2450002.5],
            "magnitude": [21.0, 22.0, 23.0],
            "e_magnitude": [0.1, 0.1, 0.2],
            "band": ["g", "r", "i"],
            "flux_density(mjy)": [1.0, 2.0, 3.0],
            "flux_density_error": [0.1, 0.2, 0.3],
        }
        self.mock_df = pd.DataFrame(self.mock_data)

    def tearDown(self) -> None:
        if os.path.exists(self.mock_file_path):
            os.remove(self.mock_file_path)

    @patch("redback.transient.transient.pd.read_csv")
    def test_load_data_generic_with_magnitude_mode(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_df
        result = redback.transient.Transient.load_data_generic(self.mock_file_path, data_mode="magnitude")
        expected_result = (
            np.array(self.mock_data["time (days)"]),
            np.array(self.mock_data["time"]),
            np.array(self.mock_data["magnitude"]),
            np.array(self.mock_data["e_magnitude"]),
            np.array(self.mock_data["band"]),
        )
        for res, exp in zip(result, expected_result):
            np.testing.assert_array_equal(res, exp)

    @patch("redback.transient.transient.pd.read_csv")
    def test_load_data_generic_with_flux_density_mode(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_df
        result = redback.transient.Transient.load_data_generic(self.mock_file_path, data_mode="flux_density")
        expected_result = (
            np.array(self.mock_data["time (days)"]),
            np.array(self.mock_data["time"]),
            np.array(self.mock_data["flux_density(mjy)"]),
            np.array(self.mock_data["flux_density_error"]),
            np.array(self.mock_data["band"]),
        )
        for res, exp in zip(result, expected_result):
            np.testing.assert_array_equal(res, exp)

    @patch("redback.transient.transient.pd.read_csv")
    def test_load_data_generic_with_all_mode(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_df
        result = redback.transient.Transient.load_data_generic(self.mock_file_path, data_mode="all")
        expected_result = (
            np.array(self.mock_data["time (days)"]),
            np.array(self.mock_data["time"]),
            np.array(self.mock_data["flux_density(mjy)"]),
            np.array(self.mock_data["flux_density_error"]),
            np.array(self.mock_data["magnitude"]),
            np.array(self.mock_data["e_magnitude"]),
            np.array(self.mock_data["band"]),
        )
        for res, exp in zip(result, expected_result):
            np.testing.assert_array_equal(res, exp)

    def test_load_data_generic_invalid_file_path(self):
        invalid_file_path = "invalid_path.csv"
        with self.assertRaises(FileNotFoundError):
            redback.transient.Transient.load_data_generic(invalid_file_path, data_mode="magnitude")

    @patch("redback.transient.transient.pd.read_csv")
    def test_load_data_generic_invalid_data_mode(self, mock_read_csv):
        with self.assertRaises(ValueError):
            redback.transient.Transient.load_data_generic(self.mock_file_path, data_mode="invalid_mode")

class TestFitGP(unittest.TestCase):
    def setUp(self) -> None:
        self.transient = redback.transient.Transient()
        self.transient.data_mode = "luminosity"
        self.transient.time_rest_frame = np.array([0.0, 1.0, 2.0, 3.0])
        self.transient.y = np.array([1.0, 2.0, 1.5, 3.0])
        self.transient.y_err = np.array([0.1, 0.2, 0.15, 0.3])
        self.transient.frequency = np.array([100.0, 200.0, 300.0, 400.0])
        self.transient._filtered_indices = np.array([0, 1, 2, 3])
        self.transient.active_bands = 'all'

    def tearDown(self) -> None:
        del self.transient

    def test_fit_gp_without_mean_model(self):
        """Test fitting a GP without a mean model."""
        kernel = MagicMock()

        with patch("george.GP") as mock_gp, \
                patch("scipy.optimize.minimize") as mock_minimize:
            mock_gp_instance = mock_gp.return_value
            mock_minimize.return_value.x = [1.0]

            result = self.transient.fit_gp(mean_model=None, kernel=kernel, prior=None, use_frequency=False)

            self.assertIsNotNone(result.gp)
            self.assertTrue(mock_gp_instance.compute.called)
            self.assertTrue(mock_minimize.called)
            self.assertEqual(result.mean_model, None)
            self.assertFalse(result.use_frequency)

    def test_fit_gp_with_mean_model(self):
        """Test fitting a GP with a specified mean model."""
        kernel = MagicMock()
        mean_model = MagicMock()
        prior = MagicMock()
        prior.sample.return_value = {"param1": 1.0, "param2": 2.0}

        with patch("george.GP") as mock_gp, \
                patch("bilby.core.likelihood.function_to_george_mean_model") as mock_mean_model, \
                patch("scipy.optimize.minimize") as mock_minimize:
            mock_gp_instance = mock_gp.return_value
            mock_minimize.return_value.x = [2.0]
            mock_mean_model.return_value = MagicMock()

            result = self.transient.fit_gp(mean_model=mean_model, kernel=kernel, prior=prior, use_frequency=False)

            self.assertIsNotNone(result.gp)
            self.assertEqual(result.mean_model, mean_model)
            self.assertTrue(mock_mean_model.called)
            self.assertTrue(mock_gp_instance.compute.called)
            self.assertTrue(mock_minimize.called)

    def test_fit_gp_without_prior_for_mean_model(self):
        """Test fitting a GP with a mean model but without providing priors."""
        kernel = MagicMock()
        mean_model = MagicMock()

        with self.assertRaises(ValueError):
            self.transient.fit_gp(mean_model=mean_model, kernel=kernel, prior=None, use_frequency=False)

    def test_fit_gp_with_invalid_data_mode(self):
        """Test fitting a GP when an invalid/unsupported data mode is set."""
        kernel = MagicMock()

        with self.assertRaises(ValueError) as context:
            self.transient.data_mode = "invalid_mode"  # Invalid data mode

        self.assertEqual(str(context.exception), "Unknown data mode.")

    def test_fit_gp_with_use_frequency(self):
        """Test fitting GP while using frequency as an input (2D GP)."""
        kernel = MagicMock()

        with patch("george.GP") as mock_gp, \
                patch("scipy.optimize.minimize") as mock_minimize:
            mock_gp_instance = mock_gp.return_value
            mock_minimize.return_value.x = [3.0]

            result = self.transient.fit_gp(mean_model=None, kernel=kernel, prior=None, use_frequency=True)

            self.assertIsNotNone(result.gp)
            self.assertTrue(mock_gp_instance.compute.called)
            self.assertTrue(mock_minimize.called)
            self.assertTrue(result.use_frequency)

    def test_fit_gp_scaling_behavior(self):
        """Test that GP fitting scales the y values correctly."""
        kernel = MagicMock()

        with patch("george.GP") as mock_gp, \
                patch("scipy.optimize.minimize") as mock_minimize:
            mock_gp_instance = mock_gp.return_value
            mock_minimize.return_value.x = [1.0]

            result = self.transient.fit_gp(mean_model=None, kernel=kernel, prior=None, use_frequency=False)

            self.assertTrue(mock_gp_instance.compute.called)
            self.assertTrue(mock_minimize.called)
            self.assertAlmostEqual(result.y_scaler, np.max(self.transient.y))
            self.assertTrue(np.allclose(result.scaled_y, self.transient.y / np.max(self.transient.y)))

class TestFromSimulatedOpticalData(unittest.TestCase):
    @mock.patch("pandas.read_csv")
    def test_from_simulated_optical_data_success(self, mock_read_csv):
        mock_data = {
            "time (days)": [1.0, 2.0, 3.0],
            "time": [10.0, 20.0, 30.0],
            "magnitude": [22.1, 23.2, 24.3],
            "e_magnitude": [0.1, 0.2, 0.3],
            "band": ["g", "r", "i"],
            "bands": ["g", "r", "i"],
            "wavelength [Hz]": [1e14, 2e14, 3e14],
            "sncosmo_name": ["ztfg", "ztfr", "ztfi"],
            "flux(erg/cm2/s)": [1e-15, 2e-15, 3e-15],
            "flux_error": [1e-16, 2e-16, 3e-16],
            "flux_density(mjy)": [1.1, 1.2, 1.3],
            "flux_density_error": [0.1, 0.2, 0.3],
            "detected": [1, 1, 1],
        }
        mock_df = pd.DataFrame(mock_data)
        mock_read_csv.return_value = mock_df

        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test_transient",
            data_mode="magnitude",
            active_bands="all",
            plotting_order=None,
            use_phase_model=False,
        )

        self.assertEqual(instance.name, "test_transient")
        np.testing.assert_array_equal(instance.time, np.array(mock_data["time (days)"]))
        np.testing.assert_array_equal(instance.time_mjd, np.array(mock_data["time"]))
        np.testing.assert_array_equal(instance.magnitude, np.array(mock_data["magnitude"]))
        np.testing.assert_array_equal(instance.magnitude_err, np.array(mock_data["e_magnitude"]))
        np.testing.assert_array_equal(instance.bands, np.array(mock_data["band"]))
        np.testing.assert_array_equal(instance.flux, np.array(mock_data["flux(erg/cm2/s)"]))
        np.testing.assert_array_equal(instance.flux_err, np.array(mock_data["flux_error"]))
        np.testing.assert_array_equal(instance.flux_density, np.array(mock_data["flux_density(mjy)"]))
        np.testing.assert_array_equal(instance.flux_density_err, np.array(mock_data["flux_density_error"]))

    @mock.patch("pandas.read_csv")
    def test_from_simulated_optical_data_no_detected_entries(self, mock_read_csv):
        mock_data = {
            "time (days)": [1.0, 2.0, 3.0],
            "time": [10.0, 20.0, 30.0],
            "magnitude": [22.1, 23.2, 24.3],
            "e_magnitude": [0.1, 0.2, 0.3],
            "band": ["g", "r", "i"],
            "bands": ["g", "r", "i"],
            "wavelength [Hz]": [1e14, 2e14, 3e14],
            "sncosmo_name": ["ztfg", "ztfr", "ztfi"],
            "flux(erg/cm2/s)": [1e-15, 2e-15, 3e-15],
            "flux_error": [1e-16, 2e-16, 3e-16],
            "flux_density(mjy)": [1.1, 1.2, 1.3],
            "flux_density_error": [0.1, 0.2, 0.3],
            "detected": [0, 0, 0],
        }
        mock_df = pd.DataFrame(mock_data)
        mock_read_csv.return_value = mock_df

        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test_transient",
            data_mode="magnitude",
            active_bands="all",
            plotting_order=None,
            use_phase_model=False,
        )

        # safer assertions for either empty arrays or None to ensure test robustness
        attributes = [
            instance.time,
            instance.time_mjd,
            instance.magnitude,
            instance.magnitude_err,
            instance.bands,
            instance.flux,
            instance.flux_err,
            instance.flux_density,
            instance.flux_density_err
        ]

        for attr in attributes:
            if attr is not None:
                self.assertEqual(len(attr), 0)
            else:
                self.assertIsNone(attr)

    @mock.patch("pandas.read_csv")
    def test_from_simulated_optical_data_file_not_found(self, mock_read_csv):
        mock_read_csv.side_effect = FileNotFoundError

        with self.assertRaises(FileNotFoundError):
            redback.transient.Transient.from_simulated_optical_data(
                name="non_existent_transient",
                data_mode="magnitude",
                active_bands="all",
                plotting_order=None,
                use_phase_model=False,
            )

class TestFromLasairTransient(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_data = {
            "time (days)": [0.1, 0.2, 0.3],
            "time": [59000.1, 59000.2, 59000.3],
            "magnitude": [20.5, 20.6, 20.7],
            "e_magnitude": [0.1, 0.1, 0.1],
            "band": ["g", "r", "i"],
            "bands": ["g", "r", "i"],
            "wavelength [Hz]": [1e14, 2e14, 3e14],
            "sncosmo_name": ["ztfg", "ztfr", "ztfi"],
            "flux(erg/cm2/s)": [1e-15, 1.1e-15, 1.2e-15],
            "flux_error": [1e-16, 1.1e-16, 1.2e-16],
            "flux_density(mjy)": [0.35, 0.36, 0.37],
            "flux_density_error": [0.05, 0.05, 0.05]
        }
        self.mock_df = pd.DataFrame(self.mock_data)

        self.mock_directory = mock.MagicMock()
        self.mock_directory.processed_file_path = "mock/path/to/file.csv"

    def tearDown(self) -> None:
        pass

    @mock.patch("redback.get_data.directory.lasair_directory_structure")
    @mock.patch("pandas.read_csv")
    def test_from_lasair_data_basic_functionality(self, mock_read_csv, mock_directory_structure):
        mock_directory_structure.return_value = self.mock_directory
        mock_read_csv.return_value = self.mock_df

        transient = redback.transient.Transient.from_lasair_data(
            name="test_transient",
            data_mode="magnitude",
            active_bands="all",
            use_phase_model=False
        )

        self.assertEqual(transient.name, "test_transient")
        self.assertEqual(transient.data_mode, "magnitude")
        np.testing.assert_array_equal(transient.time, self.mock_data["time (days)"])
        np.testing.assert_array_equal(transient.time_mjd, self.mock_data["time"])
        np.testing.assert_array_equal(transient.magnitude, self.mock_data["magnitude"])
        np.testing.assert_array_equal(transient.magnitude_err, self.mock_data["e_magnitude"])
        np.testing.assert_array_equal(transient.bands, self.mock_data["band"])
        np.testing.assert_array_equal(transient.flux, self.mock_data["flux(erg/cm2/s)"])
        np.testing.assert_array_equal(transient.flux_err, self.mock_data["flux_error"])
        np.testing.assert_array_equal(transient.flux_density, self.mock_data["flux_density(mjy)"])
        np.testing.assert_array_equal(transient.flux_density_err, self.mock_data["flux_density_error"])

    @mock.patch("redback.get_data.directory.lasair_directory_structure")
    @mock.patch("pandas.read_csv")
    def test_from_lasair_data_with_plotting_order(self, mock_read_csv, mock_directory_structure):
        mock_directory_structure.return_value = self.mock_directory
        mock_read_csv.return_value = self.mock_df

        plotting_order = np.array(["r", "g", "i"])
        transient = redback.transient.Transient.from_lasair_data(
            name="test_transient",
            data_mode="magnitude",
            active_bands="all",
            use_phase_model=False,
            plotting_order=plotting_order
        )

        self.assertEqual(transient.plotting_order.tolist(), plotting_order.tolist())

    @mock.patch("redback.get_data.directory.lasair_directory_structure")
    @mock.patch("pandas.read_csv")
    def test_from_lasair_data_invalid_data_mode(self, mock_read_csv, mock_directory_structure):
        mock_directory_structure.return_value = self.mock_directory
        mock_read_csv.return_value = self.mock_df

        with self.assertRaises(ValueError):
            redback.transient.Transient.from_lasair_data(
                name="test_transient",
                data_mode="invalid_mode",
                active_bands="all",
                use_phase_model=False
            )