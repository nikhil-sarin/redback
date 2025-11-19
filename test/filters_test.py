import unittest
import os
import tempfile
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from unittest.mock import patch, MagicMock, Mock
import redback.filters as filters
import sncosmo


class TestFilterFunctions(unittest.TestCase):
    """Test basic filter functions"""

    def test_show_all_filters(self):
        """Test that show_all_filters returns a table"""
        result = filters.show_all_filters()
        self.assertIsNotNone(result)
        # Check that required columns exist
        self.assertIn('bands', result.colnames)
        self.assertIn('sncosmo_name', result.colnames)

    def test_add_to_database(self):
        """Test adding a filter to the database"""
        # Create a mock database table
        database = Table(names=['bands', 'wavelength [Hz]', 'wavelength [Angstrom]',
                                'color', 'reference_flux', 'sncosmo_name',
                                'plot_label', 'effective_width [Hz]'],
                        dtype=[str, float, float, str, float, str, str, float])

        label = 'test_filter'
        wavelength = 5.5e-7  # 550 nm in meters
        zeroflux = 1e-10
        plot_label = 'Test'
        effective_width = 100.0  # Angstrom

        # Add to database
        filters.add_to_database(label, wavelength, zeroflux, database, plot_label, effective_width)

        # Check that the row was added
        self.assertEqual(len(database), 1)
        self.assertEqual(database['bands'][0], label)
        self.assertEqual(database['sncosmo_name'][0], label)
        self.assertEqual(database['plot_label'][0], plot_label)
        self.assertGreater(database['wavelength [Hz]'][0], 0)
        self.assertGreater(database['wavelength [Angstrom]'][0], 0)

    def test_add_to_sncosmo(self):
        """Test adding a filter to SNCosmo"""
        # Create mock transmission data
        wavelength = np.linspace(4000, 6000, 100)  # Angstroms
        transmission = np.exp(-0.5 * ((wavelength - 5000) / 200) ** 2)  # Gaussian

        transmission_table = Table()
        transmission_table['Wavelength'] = wavelength
        transmission_table['Transmission'] = transmission

        label = 'test_band'

        # Add to SNCosmo
        filters.add_to_sncosmo(label, transmission_table)

        # Check that it was registered
        band = sncosmo.get_bandpass(label)
        self.assertIsNotNone(band)
        self.assertEqual(band.name, label)


class TestAddFilterUser(unittest.TestCase):
    """Test user filter addition"""

    def test_add_filter_user_new_filter(self):
        """Test adding a new user filter"""
        # Create a temporary filter file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("4000 0.0\n")
            f.write("4500 0.5\n")
            f.write("5000 1.0\n")
            f.write("5500 0.5\n")
            f.write("6000 0.0\n")
            temp_file = f.name

        try:
            # Use a unique label that shouldn't exist
            label = 'test_user_filter_unique_xyz'

            # Save original read function
            original_read = ascii.read

            # Mock the database file operations
            def read_side_effect(filename):
                # When reading the filter file, use the actual file
                if filename == temp_file:
                    return original_read(filename)
                # For the database file, return mock
                else:
                    mock_db = Table(names=['bands', 'wavelength [Hz]', 'wavelength [Angstrom]',
                                          'color', 'reference_flux', 'sncosmo_name',
                                          'plot_label', 'effective_width [Hz]'],
                                   dtype=[str, float, float, str, float, str, str, float])
                    return mock_db

            with patch('astropy.io.ascii.read', side_effect=read_side_effect), \
                 patch.object(Table, 'write'):
                # Add the filter
                filters.add_filter_user(temp_file, label, plot_label='Test User', overwrite=False)

                # Check that the filter was registered in SNCosmo
                band = sncosmo.get_bandpass(label)
                self.assertIsNotNone(band)
                self.assertEqual(band.name, label)

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_add_filter_user_existing_no_overwrite(self):
        """Test adding a filter that already exists without overwrite"""
        # Create a temporary filter file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("4000 0.0\n")
            f.write("5000 1.0\n")
            f.write("6000 0.0\n")
            temp_file = f.name

        try:
            label = 'existing_filter'

            # Save original read function
            original_read = ascii.read

            # Create a mock database with existing entry
            def read_side_effect(filename):
                # When reading the filter file, use the actual file
                if filename == temp_file:
                    return original_read(filename)
                # For the database file, return mock with existing entry
                else:
                    mock_db = Table(names=['bands', 'wavelength [Hz]', 'wavelength [Angstrom]',
                                          'color', 'reference_flux', 'sncosmo_name',
                                          'plot_label', 'effective_width [Hz]'],
                                   dtype=[str, float, float, str, float, str, str, float])
                    mock_db.add_row([label, 5.45e14, 5500.0, 'black', 1e-10, label, 'Test', 1e13])
                    return mock_db

            with patch('astropy.io.ascii.read', side_effect=read_side_effect), \
                 patch('builtins.print') as mock_print, \
                 patch.object(Table, 'write'):

                # Try to add without overwrite
                filters.add_filter_user(temp_file, label, overwrite=False)

                # Check that appropriate message was printed
                mock_print.assert_called()
                print_args = ' '.join([str(arg) for call in mock_print.call_args_list for arg in call[0]])
                self.assertIn('already exists', print_args)

        finally:
            os.unlink(temp_file)


class TestAddFilterSVO(unittest.TestCase):
    """Test SVO filter addition"""

    @patch('astroquery.svo_fps.SvoFps.get_transmission_data')
    @patch('astropy.io.ascii.read')
    def test_add_filter_svo_new_filter(self, mock_read, mock_get_transmission):
        """Test adding a new filter from SVO"""
        # Create mock filter data
        mock_filter = {
            'filterID': 'Generic/Bessell.V',
            'WavelengthRef': 5500.0,  # Angstroms
            'WidthEff': 800.0
        }

        # Create mock transmission data
        mock_transmission = Table()
        mock_transmission['Wavelength'] = np.linspace(4500, 6500, 100)
        mock_transmission['Transmission'] = np.exp(-0.5 * ((mock_transmission['Wavelength'] - 5500) / 300) ** 2)
        mock_get_transmission.return_value = mock_transmission

        # Create mock database
        mock_db = Table(names=['bands', 'wavelength [Hz]', 'wavelength [Angstrom]',
                              'color', 'reference_flux', 'sncosmo_name',
                              'plot_label', 'effective_width [Hz]'],
                       dtype=[str, float, float, str, float, str, str, float])
        mock_read.return_value = mock_db

        label = 'test_svo_filter'

        with patch.object(Table, 'write') as mock_write:
            # Add the filter
            filters.add_filter_svo(mock_filter, label, plot_label='SVO Test', overwrite=False)

            # Check that transmission data was fetched
            mock_get_transmission.assert_called_once_with(mock_filter['filterID'])

            # Check that database was written
            mock_write.assert_called_once()

    @patch('astroquery.svo_fps.SvoFps.get_transmission_data')
    @patch('astropy.io.ascii.read')
    def test_add_filter_svo_overwrite(self, mock_read, mock_get_transmission):
        """Test overwriting an existing SVO filter"""
        # Create mock filter data
        mock_filter = {
            'filterID': 'Generic/Bessell.V',
            'WavelengthRef': 5500.0,
            'WidthEff': 800.0
        }

        # Create mock transmission data
        mock_transmission = Table()
        mock_transmission['Wavelength'] = np.linspace(4500, 6500, 100)
        mock_transmission['Transmission'] = np.ones(100)
        mock_get_transmission.return_value = mock_transmission

        label = 'existing_svo_filter'

        # Create mock database with existing entry
        mock_db = Table(names=['bands', 'wavelength [Hz]', 'wavelength [Angstrom]',
                              'color', 'reference_flux', 'sncosmo_name',
                              'plot_label', 'effective_width [Hz]'],
                       dtype=[str, float, float, str, float, str, str, float])
        mock_db.add_row([label, 5.45e14, 5500.0, 'black', 1e-10, label, 'Old', 1e13])
        mock_read.return_value = mock_db

        with patch.object(Table, 'write') as mock_write:
            # Add with overwrite
            filters.add_filter_svo(mock_filter, label, plot_label='New', overwrite=True)

            # Check that the row was removed and re-added
            mock_write.assert_called_once()


class TestAddCommonFilters(unittest.TestCase):
    """Test adding common filters from SVO"""

    @patch('astroquery.svo_fps.SvoFps.get_filter_list')
    @patch('redback.filters.add_filter_svo')
    @patch('builtins.print')
    def test_add_common_filters_mock(self, mock_print, mock_add_filter, mock_get_filter_list):
        """Test adding common filters with mocked SVO calls"""
        # Create mock filter lists
        mock_grond = Table()
        mock_grond['filterID'] = ['La_Silla/GROND.g', 'La_Silla/GROND.r']
        mock_grond['Band'] = ['g', 'r']

        mock_efosc = Table()
        mock_efosc['filterID'] = ['La_Silla/EFOSC.Gunn_g', 'La_Silla/EFOSC.Gunn_r']
        mock_efosc['Band'] = ['g', 'r']
        mock_efosc['Description'] = ['Gunn g', 'Gunn r']

        mock_euclid_vis = Table()
        mock_euclid_vis['filterID'] = ['Euclid/VIS.vis']

        mock_euclid_nisp = Table()
        mock_euclid_nisp['filterID'] = ['Euclid/NISP.Y', 'Euclid/NISP.J', 'Euclid/other.X']

        mock_irac = Table()
        mock_irac['filterID'] = ['Spitzer/IRAC.I1', 'Spitzer/IRAC.I2']

        mock_wise = Table()
        mock_wise['filterID'] = ['WISE/WISE.W1', 'WISE/WISE.W2']

        def get_filter_list_side_effect(facility=None, instrument=None):
            if facility == 'La Silla' and instrument == 'GROND':
                return mock_grond
            elif facility == 'La Silla' and instrument == 'EFOSC':
                return mock_efosc
            elif facility == 'Euclid' and instrument == 'VIS':
                return mock_euclid_vis
            elif facility == 'Euclid' and instrument == 'NISP':
                return mock_euclid_nisp
            elif facility == 'Spitzer' and instrument == 'IRAC':
                return mock_irac
            elif facility == 'WISE':
                return mock_wise
            return Table()

        mock_get_filter_list.side_effect = get_filter_list_side_effect

        # Call the function
        filters.add_common_filters(overwrite=True)

        # Check that filters were added (at least some calls were made)
        self.assertGreater(mock_add_filter.call_count, 0)

        # Check that progress messages were printed
        self.assertGreater(mock_print.call_count, 0)


class TestAddEffectiveWidths(unittest.TestCase):
    """Test adding effective widths to filter database"""

    @patch('pandas.read_csv')
    @patch('sncosmo.get_bandpass')
    def test_add_effective_widths(self, mock_get_bandpass, mock_read_csv):
        """Test adding effective widths to database"""
        import pandas as pd

        # Create mock dataframe
        mock_df = pd.DataFrame({
            'sncosmo_name': ['bessellv', 'bessellr'],
            'wavelength [Hz]': [5.45e14, 4.61e14]
        })
        mock_read_csv.return_value = mock_df

        # Create mock bandpass
        mock_band = MagicMock()
        mock_band.wave = np.linspace(4000, 6000, 100)
        mock_band.trans = np.ones(100)
        mock_band.wave_eff = 5500.0
        mock_get_bandpass.return_value = mock_band

        # Mock to_csv
        mock_df.to_csv = MagicMock()

        # Call the function
        filters.add_effective_widths()

        # Check that effective widths were added
        self.assertIn('effective_width [Hz]', mock_df.columns)

        # Check that CSV was written
        mock_df.to_csv.assert_called_once()

    @patch('pandas.read_csv')
    @patch('sncosmo.get_bandpass')
    @patch('redback.utils.logger')
    def test_add_effective_widths_with_failure(self, mock_logger, mock_get_bandpass, mock_read_csv):
        """Test handling of failed bandpass retrieval"""
        import pandas as pd

        # Create mock dataframe
        mock_df = pd.DataFrame({
            'sncosmo_name': ['bad_band', 'good_band'],
            'wavelength [Hz]': [5.45e14, 4.61e14]
        })
        mock_read_csv.return_value = mock_df

        # Mock bandpass - first fails, second succeeds
        mock_band = MagicMock()
        mock_band.wave = np.linspace(4000, 6000, 100)
        mock_band.trans = np.ones(100)
        mock_band.wave_eff = 5500.0

        def get_bandpass_side_effect(name):
            if name == 'bad_band':
                raise Exception("Band not found")
            return mock_band

        mock_get_bandpass.side_effect = get_bandpass_side_effect
        mock_df.to_csv = MagicMock()

        # Call the function
        filters.add_effective_widths()

        # Check that warning was logged for failed band
        mock_logger.warning.assert_called_once()

        # Check that effective widths were still added
        self.assertIn('effective_width [Hz]', mock_df.columns)
