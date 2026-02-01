import io
import os.path
import shutil
import unittest
from unittest import mock
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import requests

import redback


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_trigger_number(self):
        trigger = redback.get_data.utils.get_trigger_number("041223")
        self.assertEqual("100585", trigger)

    def test_get_grb_table(self):
        expected_keys = ['GRB', 'Time [UT]', 'Trigger Number', 'BAT RA (J2000)',
                         'BAT Dec (J2000)', 'BAT T90 [sec]',
                         'BAT Fluence (15-150 keV) [10^-7 erg/cm^2]',
                         'BAT Fluence 90% Error (15-150 keV) [10^-7 erg/cm^2]',
                         'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)',
                         'XRT RA (J2000)', 'XRT Dec (J2000)',
                         'XRT Time to First Observation [sec]',
                         'XRT Early Flux (0.3-10 keV) [10^-11 erg/cm^2/s]', 'UVOT RA (J2000)',
                         'UVOT Dec (J2000)', 'UVOT Time to First Observation [sec]',
                         'UVOT Magnitude', 'Other Observatory Detections', 'Redshift',
                         'Host Galaxy', 'Comments', 'References']
        table = redback.get_data.utils.get_grb_table()
        self.assertListEqual(expected_keys, list(table.keys()))


class TestDirectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        _delete_downloaded_files()
        self.grb = "GRB123456"
        self.data_mode = "flux"
        self.instrument = "BAT+XRT"
        self.bin_size = "1s"

    def tearDown(self) -> None:
        _delete_downloaded_files()
        del self.grb
        del self.data_mode
        del self.instrument
        del self.bin_size

    def test_swift_afterglow_directory_structure_bat_xrt(self):
        structure = redback.get_data.directory.afterglow_directory_structure(
            grb=self.grb, data_mode=self.data_mode, instrument=self.instrument)
        self.assertEqual(f"GRBData/afterglow/{self.data_mode}/", structure.directory_path)
        self.assertEqual(f"GRBData/afterglow/{self.data_mode}/{self.grb}_rawSwiftData.csv", structure.raw_file_path)
        self.assertEqual(f"GRBData/afterglow/{self.data_mode}/{self.grb}.csv", structure.processed_file_path)

    def test_swift_afterglow_directory_structure_xrt(self):
        self.instrument = "XRT"
        structure = redback.get_data.directory.afterglow_directory_structure(
            grb=self.grb, data_mode=self.data_mode, instrument=self.instrument)
        self.assertEqual(f"GRBData/afterglow/{self.data_mode}/", structure.directory_path)
        self.assertEqual(f"GRBData/afterglow/{self.data_mode}/{self.grb}_xrt_rawSwiftData.csv", structure.raw_file_path)
        self.assertEqual(f"GRBData/afterglow/{self.data_mode}/{self.grb}_xrt.csv", structure.processed_file_path)

    def test_swift_prompt_directory_structure(self):
        self.data_mode = "prompt"
        structure = redback.get_data.directory.swift_prompt_directory_structure(grb=self.grb, bin_size=self.bin_size)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/", structure.directory_path)
        self.assertEqual(
            f"GRBData/{self.data_mode}/flux/{self.grb}_{self.bin_size}_lc_ascii.dat", structure.raw_file_path)
        self.assertEqual(
            f"GRBData/{self.data_mode}/flux/{self.grb}_{self.bin_size}_lc.csv", structure.processed_file_path)

    def test_swift_prompt_directory_structure_wrong_binning(self):
        with self.assertRaises(ValueError):
            redback.get_data.directory.swift_prompt_directory_structure(grb=self.grb, bin_size='3 dollars')

    def test_batse_prompt_directory_structure_no_trigger(self):
        trigger = "1234"
        self.data_mode = "prompt"
        structure = redback.get_data.directory.batse_prompt_directory_structure(
            grb=self.grb, trigger=None, get_batse_trigger_from_grb=lambda grb: trigger)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/", structure.directory_path)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/tte_bfits_{trigger}.fits.gz", structure.raw_file_path)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/{self.grb}_BATSE_lc.csv", structure.processed_file_path)

    def test_batse_prompt_directory_structure_with_trigger(self):
        trigger = "1234"
        self.data_mode = "prompt"
        structure = redback.get_data.directory.batse_prompt_directory_structure(grb=self.grb, trigger=trigger)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/", structure.directory_path)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/tte_bfits_{trigger}.fits.gz", structure.raw_file_path)
        self.assertEqual(f"GRBData/{self.data_mode}/flux/{self.grb}_BATSE_lc.csv", structure.processed_file_path)

    def test_open_access_directory_structure(self):
        transient = "abc"
        transient_type = "tde"
        self.data_mode = "magnitude"
        structure = redback.get_data.directory.open_access_directory_structure(
            transient=transient, transient_type=transient_type)
        self.assertEqual(f"{transient_type}/", structure.directory_path)
        self.assertEqual(f"{transient_type}/{transient}_rawdata.csv", structure.raw_file_path)
        self.assertEqual(f"{transient_type}/{transient}.csv", structure.processed_file_path)


def _delete_downloaded_files():
    for folder in ["GRBData", "kilonova", "supernova", "tidal_disruption_event"]:
        shutil.rmtree(folder, ignore_errors=True)


class TestDataGetterMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.transient = "GRB000526"
        self.transient_type = "afterglow"

        class SimpleDataGetter(redback.get_data.getter.DataGetter):

            VALID_TRANSIENT_TYPES = ["afterglow"]

        self.getter = SimpleDataGetter(transient=self.transient, transient_type=self.transient_type)

    def tearDown(self) -> None:
        del self.transient
        del self.transient_type
        del self.getter

    def test_get_data(self):
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock()
        self.getter.get_data()
        self.getter.collect_data.assert_called_once()
        self.getter.convert_raw_data_to_csv.assert_called_once()

    def test_set_invalid_transient_type(self):
        with self.assertRaises(ValueError):
            self.getter.transient_type = "invalid"


class TestGRBDataGetterMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.grb = "GRB000526"
        self.transient_type = "afterglow"

        class SimpleGRBDataGetter(redback.get_data.getter.GRBDataGetter):
            VALID_TRANSIENT_TYPES = ["afterglow"]

        self.getter = SimpleGRBDataGetter(grb=self.grb, transient_type=self.transient_type)

    def tearDown(self) -> None:
        del self.grb
        del self.transient_type
        del self.getter

    def test_grb_name(self):
        self.assertEqual(self.grb, self.getter.grb)

        self.getter.grb = self.grb.lstrip("GRB")
        self.assertEqual(self.grb, self.getter.grb)

    def test_set_grb_name_without_prefix(self):
        self.getter.grb = "000526"
        self.assertEqual(self.grb, self.getter.grb)

    def test_stripped_grb(self):
        stripped_grb = "000526"
        self.assertEqual(stripped_grb, self.getter.stripped_grb)


class TestBATSEDataGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        self.grb = "GRB000526"
        self.getter = redback.get_data.BATSEDataGetter(grb=self.grb)

    def tearDown(self) -> None:
        del self.grb
        del self.getter
        _delete_downloaded_files()

    def test_trigger(self):
        expected = 8121
        self.assertEqual(expected, self.getter.trigger)

    def test_trigger_filled(self):
        expected = "08121"
        self.assertEqual(expected, self.getter.trigger_filled)

    def test_url(self):
        expected = f"https://heasarc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/08001_08200/" \
                   f"08121_burst/tte_bfits_8121.fits.gz"
        self.assertEqual(expected, self.getter.url)

    @mock.patch("urllib.request.urlretrieve")
    def collect_data(self, urlretrieve):
        self.getter.collect_data()
        urlretrieve.assert_called_once()

    @mock.patch("astropy.io.fits.open")
    @mock.patch("pandas.DataFrame")
    def test_convert_raw_data_to_csv(self, DataFrame, fits_open):
        pass  # Add unittests, maybe. This is also covered by the reference file tests.


class TestOpenDataGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        _delete_downloaded_files()
        self.transient = "at2017gfo"
        self.transient_type = "kilonova"
        self.getter = redback.get_data.OpenDataGetter(transient=self.transient, transient_type=self.transient_type)

    def tearDown(self) -> None:
        del self.transient
        del self.transient_type
        del self.getter
        _delete_downloaded_files()

    def test_set_invalid_transient_type(self):
        with self.assertRaises(ValueError):
            self.getter.transient_type = "patrick"

    def test_url(self):
        expected = f"https://api.astrocats.space/{self.transient}/photometry/time+magnitude+e_" \
                   f"magnitude+band+system?e_magnitude&band&time&format=csv"
        self.assertEqual(expected, self.getter.url)

    def test_metadata_url(self):
        expected = f"https://api.astrocats.space/{self.transient}/" \
                   f"timeofmerger+discoverdate+redshift+ra+dec+host+alias?format=CSV"
        self.assertEqual(expected, self.getter.metadata_url)

    def test_metadata_path(self):
        expected = f"{self.getter.directory_path}{self.transient}_metadata.csv"
        self.assertEqual(expected, self.getter.metadata_path)

    @mock.patch("os.path.isfile")
    @mock.patch("requests.get")
    def test_collect_data_file_exists(self, get, isfile):
        isfile.return_value = True
        self.getter.collect_data()
        isfile.assert_called_once_with(self.getter.raw_file_path)
        get.assert_not_called()

    @mock.patch("os.path.isfile")
    @mock.patch("requests.get")
    @mock.patch("urllib.request.urlretrieve")
    def test_collect_data_not_found(self, urlretrieve, get, isfile):
        isfile.return_value = False
        type(get.return_value).text = PropertyMock(return_value='not found')
        with self.assertRaises(ValueError):
            self.getter.collect_data()
        get.assert_called_once_with(self.getter.url)
        urlretrieve.assert_not_called()

    @mock.patch("os.path.isfile")
    @mock.patch("requests.get")
    @mock.patch("urllib.request.urlretrieve")
    def test_collect_data(self, urlretrieve, get, isfile):
        isfile.return_value = False
        type(get.return_value).text = PropertyMock(return_value='')
        self.getter.collect_data()
        urlretrieve.assert_called_with(url=self.getter.metadata_url, filename=self.getter.metadata_path)

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.isna")
    @mock.patch("os.path.isfile")
    def test_convert_raw_data_to_csv_file_exists(self, isfile, isna, read_csv):
        isfile.return_value = True
        expected = 5
        read_csv.return_value = expected
        data = self.getter.convert_raw_data_to_csv()
        isfile.assert_called_once_with(self.getter.processed_file_path)
        isna.assert_not_called()
        self.assertEqual(expected, data)

    def test_convert_raw_data_to_csv(self):
        pass
        # This is covered by the reference file tests. More detailed tests can be implemented later.

    def test_get_time_of_event(self):
        pass

    @mock.patch("pandas.read_csv")
    @mock.patch("re.search")
    def test_get_grb_alias(self, search, read_csv):
        expected = "ret"
        alias = "alias"
        data = dict(event=[self.transient], alias=[alias])
        read_csv.return_value = pd.DataFrame.from_dict(data)
        ret = MagicMock()
        ret.group = MagicMock(return_value=expected)
        search.return_value = ret
        self.assertEqual(expected, self.getter.get_grb_alias())
        search.assert_called_once_with("GRB (.+?),", alias)

    @mock.patch("pandas.read_csv")
    @mock.patch("re.search")
    def test_get_grb_alias_fail(self, search, read_csv):
        expected = "ret"
        alias = "alias"
        data = dict(event=[self.transient], alias=[alias])
        read_csv.return_value = pd.DataFrame.from_dict(data)
        ret = MagicMock()
        ret.group = MagicMock(return_value=expected)
        search.return_value = ret
        search.side_effect = AttributeError
        self.assertIsNone(self.getter.get_grb_alias())

    @mock.patch("sqlite3.connect")
    @mock.patch("pandas.read_sql_query")
    def test_get_t0_from_grb_isnan(self, read_sql_query, connect):
        self.getter.get_grb_alias = MagicMock(return_value="alias")
        connect.return_value = "connection"
        read_sql_query.return_value = pd.DataFrame.from_dict(dict(GRB_name=["alias"], mjd=[np.nan]))
        t0 = self.getter.get_t0_from_grb()
        self.assertTrue(np.isnan(t0))

        read_sql_query.assert_called_once_with("SELECT * from Summary", "connection")
        connect.assert_called_once_with('tables/GRBcatalog.sqlite')


class TestSwiftDataGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        self.grb = "050202"
        self.transient_type = "afterglow"
        self.instrument = "BAT+XRT"
        self.data_mode = "flux"
        self.bin_size = None
        self.getter = redback.get_data.swift.SwiftDataGetter(
            grb=self.grb, transient_type=self.transient_type, data_mode=self.data_mode,
            instrument=self.instrument, bin_size=self.bin_size)

    def tearDown(self) -> None:
        shutil.rmtree("GRBData", ignore_errors=True)
        del self.grb
        del self.transient_type
        del self.data_mode
        del self.bin_size
        del self.getter

    def test_set_valid_data_mode(self):
        for valid_data_mode in ['flux', 'flux_density']:
            self.getter.data_mode = valid_data_mode
            self.assertEqual(valid_data_mode, self.getter.data_mode)

    def test_set_invalid_data_mode(self):
        with self.assertRaises(ValueError):
            self.getter.data_mode = 'magnitude'

    def test_set_valid_instrument(self):
        for valid_instrument in ['BAT+XRT', 'XRT']:
            self.getter.instrument = valid_instrument
            self.assertEqual(valid_instrument, self.getter.instrument)

    def test_set_invalid_instrument(self):
        with self.assertRaises(ValueError):
            self.getter.instrument = "potato"

    @mock.patch("redback.get_data.utils.get_trigger_number")
    def test_trigger(self, get_trigger_number):
        expected = "0"
        get_trigger_number.return_value = expected
        trigger = self.getter.trigger
        self.assertEqual(expected, trigger)
        get_trigger_number.assert_called_once()

    @mock.patch("astropy.io.ascii.read")
    def test_get_swift_id_from_grb(self, ascii_read):
        swift_id_stump = "123456"
        swift_id_expected = f"{swift_id_stump}000"
        swift_id_expected = swift_id_expected.zfill(11)
        ascii_read.return_value = dict(col1=[f"GRB{self.grb}"], col2=[swift_id_stump])
        swift_id_actual = self.getter.get_swift_id_from_grb()
        self.assertEqual(swift_id_expected, swift_id_actual)
        ascii_read.assert_called_once()

    @mock.patch("astropy.io.ascii.read")
    def test_grb_website_prompt(self, ascii_read):
        swift_id_stump = "123456"
        self.getter.transient_type = "prompt"
        ascii_read.return_value = dict(col1=[f"GRB{self.grb}"], col2=[swift_id_stump])
        expected = f"https://swift.gsfc.nasa.gov/results/batgrbcat/GRB{self.grb}/data_product/" \
                   f"{self.getter.get_swift_id_from_grb()}-results/lc/{self.bin_size}_lc_ascii.dat"
        self.assertEqual(expected, self.getter.grb_website)

    @mock.patch("redback.get_data.utils.get_trigger_number")
    def test_grb_website_bat_xrt(self, get_trigger_number):
        expected_trigger = "0"
        get_trigger_number.return_value = expected_trigger
        expected = f'http://www.swift.ac.uk/burst_analyser/00{expected_trigger}/'
        self.assertEqual(expected, self.getter.grb_website)

    @mock.patch("redback.get_data.utils.get_trigger_number")
    def test_grb_website_xrt(self, get_trigger_number):
        self.getter.instrument = 'XRT'
        expected_trigger = "0"
        get_trigger_number.return_value = expected_trigger
        expected = f'https://www.swift.ac.uk/xrt_curves/00{expected_trigger}/flux.qdp'
        self.assertEqual(expected, self.getter.grb_website)

    @mock.patch("redback.get_data.directory.afterglow_directory_structure")
    def test_create_directory_structure_afterglow(self, afterglow_directory_structure):
        expected = "0", "1", "2"
        afterglow_directory_structure.return_value = expected
        self.getter = redback.get_data.swift.SwiftDataGetter(
            grb=self.grb, transient_type=self.transient_type, data_mode=self.data_mode,
            instrument=self.instrument, bin_size=self.bin_size)  # method is called in constructor
        self.assertListEqual(
            list(expected),
            list([self.getter.directory_path, self.getter.raw_file_path, self.getter.processed_file_path]))
        afterglow_directory_structure.assert_called_with(
            grb=f"GRB{self.grb}", data_mode=self.data_mode, instrument=self.instrument)

    @mock.patch("redback.get_data.directory.swift_prompt_directory_structure")
    def test_create_directory_structure_prompt(self, prompt_directory_structure):
        expected = "0", "1", "2"
        prompt_directory_structure.return_value = expected
        self.getter = redback.get_data.swift.SwiftDataGetter(
            grb=self.grb, transient_type="prompt", data_mode=self.data_mode,
            instrument=self.instrument, bin_size=self.bin_size)  # method is called in constructor
        self.assertListEqual(
            list(expected),
            list([self.getter.directory_path, self.getter.raw_file_path, self.getter.processed_file_path]))
        prompt_directory_structure.assert_called_with(grb=f"GRB{self.grb}", bin_size=self.bin_size)

    @mock.patch("os.path.isfile")
    def test_collect_data_rawfile_exists(self, isfile):
        isfile.return_value = True
        redback.utils.logger.warning = MagicMock()
        self.getter.collect_data()
        isfile.assert_called_once()
        redback.utils.logger.warning.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch('requests.get')
    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    def test_collect_data_no_lightcurve_available(self, get, isfile):
        isfile.return_value = False
        get.return_value = MagicMock()
        get.return_value.__setattr__('text', 'No Light curve available')
        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.collect_data()

    @mock.patch("os.path.isfile")
    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    def test_collect_data_xrt(self, isfile):
        isfile.return_value = False
        self.getter.instrument = "XRT"
        self.getter.download_directly = MagicMock()
        self.getter.download_integrated_flux_data = MagicMock()
        self.getter.download_flux_density_data = MagicMock()
        self.getter.collect_data()
        self.getter.download_directly.assert_called_once()
        self.getter.download_integrated_flux_data.assert_not_called()
        self.getter.download_flux_density_data.assert_not_called()

    @mock.patch("os.path.isfile")
    def test_collect_data_xrt_via_api(self, isfile):
        isfile.return_value = False
        self.getter.instrument = "XRT"
        self.getter.download_xrt_data_via_api = MagicMock(return_value=MagicMock())
        self.getter.download_directly = MagicMock()
        self.getter.collect_data()
        self.getter.download_xrt_data_via_api.assert_called_once()
        self.getter.download_directly.assert_not_called()

    @mock.patch("os.path.isfile")
    def test_collect_data_prompt(self, isfile):
        isfile.return_value = False
        self.getter.transient_type = 'prompt'
        self.getter.download_directly = MagicMock()
        self.getter.download_integrated_flux_data = MagicMock()
        self.getter.download_flux_density_data = MagicMock()
        self.getter.collect_data()
        self.getter.download_directly.assert_called_once()
        self.getter.download_integrated_flux_data.assert_not_called()
        self.getter.download_flux_density_data.assert_not_called()

    @mock.patch("os.path.isfile")
    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    def test_collect_data_afterglow_flux(self, isfile):
        isfile.return_value = False
        self.getter.instrument = 'BAT+XRT'
        self.getter.transient_type = 'afterglow'
        self.getter.data_mode = 'flux'
        self.getter.download_directly = MagicMock()
        self.getter.download_integrated_flux_data = MagicMock()
        self.getter.download_flux_density_data = MagicMock()
        self.getter.collect_data()
        self.getter.download_directly.assert_not_called()
        self.getter.download_integrated_flux_data.assert_called_once()
        self.getter.download_flux_density_data.assert_not_called()

    @mock.patch("os.path.isfile")
    def test_collect_data_afterglow_flux_via_api(self, isfile):
        isfile.return_value = False
        self.getter.instrument = 'BAT+XRT'
        self.getter.transient_type = 'afterglow'
        self.getter.data_mode = 'flux'
        self.getter.download_burst_analyser_data_via_api = MagicMock(return_value=MagicMock())
        self.getter.download_integrated_flux_data = MagicMock()
        self.getter.collect_data()
        self.getter.download_burst_analyser_data_via_api.assert_called_once()
        self.getter.download_integrated_flux_data.assert_not_called()

    @mock.patch("os.path.isfile")
    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    def test_collect_data_afterglow_flux_density(self, isfile):
        isfile.return_value = False
        self.getter.instrument = 'BAT+XRT'
        self.getter.transient_type = 'afterglow'
        self.getter.data_mode = 'flux_density'
        self.getter.download_directly = MagicMock()
        self.getter.download_integrated_flux_data = MagicMock()
        self.getter.download_flux_density_data = MagicMock()
        self.getter.collect_data()
        self.getter.download_directly.assert_not_called()
        self.getter.download_integrated_flux_data.assert_not_called()
        self.getter.download_flux_density_data.assert_called_once()

    @mock.patch("os.path.isfile")
    def test_collect_data_afterglow_flux_density_via_api(self, isfile):
        isfile.return_value = False
        self.getter.instrument = 'BAT+XRT'
        self.getter.transient_type = 'afterglow'
        self.getter.data_mode = 'flux_density'
        self.getter.download_burst_analyser_data_via_api = MagicMock(return_value=MagicMock())
        self.getter.download_flux_density_data = MagicMock()
        self.getter.collect_data()
        self.getter.download_burst_analyser_data_via_api.assert_called_once()
        self.getter.download_flux_density_data.assert_not_called()

    def _mock_converter_functions(self):
        self.getter.convert_xrt_data_to_csv = MagicMock()
        self.getter.convert_raw_afterglow_data_to_csv = MagicMock()
        self.getter.convert_raw_prompt_data_to_csv = MagicMock()

    @mock.patch("pandas.read_csv")
    def test_convert_raw_data_to_csv_file_exists(self, read_csv):
        self._mock_converter_functions()
        expected = 5
        read_csv.return_value = expected
        with open(self.getter.processed_file_path, "w"):  # create empty file
            pass
        data = self.getter.convert_raw_data_to_csv()
        self.assertEqual(expected, data)
        self.getter.convert_xrt_data_to_csv.assert_not_called()
        self.getter.convert_raw_afterglow_data_to_csv.assert_not_called()
        self.getter.convert_raw_prompt_data_to_csv.assert_not_called()

    def test_convert_raw_data_to_csv_instrument_xrt(self):
        self.getter.instrument = "XRT"
        self._mock_converter_functions()
        self.getter.convert_raw_data_to_csv()
        self.getter.convert_xrt_data_to_csv.assert_called_once()
        self.getter.convert_raw_afterglow_data_to_csv.assert_not_called()
        self.getter.convert_raw_prompt_data_to_csv.assert_not_called()

    def test_convert_raw_data_to_csv_afterglow_data(self):
        self.getter.transient_type = "afterglow"
        self._mock_converter_functions()
        self.getter.convert_raw_data_to_csv()
        self.getter.convert_xrt_data_to_csv.assert_not_called()
        self.getter.convert_raw_afterglow_data_to_csv.assert_called_once()
        self.getter.convert_raw_prompt_data_to_csv.assert_not_called()

    def test_convert_raw_data_to_csv_prompt_data(self):
        self.getter.transient_type = "prompt"
        self._mock_converter_functions()
        self.getter.convert_raw_data_to_csv()
        self.getter.convert_xrt_data_to_csv.assert_not_called()
        self.getter.convert_raw_afterglow_data_to_csv.assert_not_called()
        self.getter.convert_raw_prompt_data_to_csv.assert_called_once()

    def test_convert_raw_data_to_csv_uses_api_data_xrt(self):
        self.getter.instrument = "XRT"
        self.getter._api_data = pd.DataFrame({'Time': [1, 2, 3]})
        self.getter.convert_xrt_api_data_to_csv = MagicMock(return_value=pd.DataFrame())
        self.getter.convert_raw_data_to_csv()
        self.getter.convert_xrt_api_data_to_csv.assert_called_once()

    def test_convert_raw_data_to_csv_uses_api_data_bat_xrt(self):
        self.getter.instrument = "BAT+XRT"
        self.getter._api_data = {'XRT': {'binning': {'band': pd.DataFrame()}}}
        self.getter.convert_burst_analyser_api_data_to_csv = MagicMock(return_value=pd.DataFrame())
        self.getter.convert_raw_data_to_csv()
        self.getter.convert_burst_analyser_api_data_to_csv.assert_called_once()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_xrt_data_via_api_success(self, mock_udg):
        """Test successful XRT data download via API."""
        mock_df = pd.DataFrame({
            'Time': [100, 200, 300],
            'TimePos': [10, 20, 30],
            'TimeNeg': [10, 20, 30],
            'Flux': [1e-11, 2e-11, 3e-11],
            'FluxPos': [1e-12, 2e-12, 3e-12],
            'FluxNeg': [1e-12, 2e-12, 3e-12]
        })
        mock_udg.getLightCurves.return_value = {
            'Datasets': ['PC_incbad_CURVE'],
            'PC_incbad_CURVE': mock_df
        }

        result = self.getter.download_xrt_data_via_api()
        mock_udg.getLightCurves.assert_called_once_with(
            GRBName=self.getter.grb,
            saveData=False,
            returnData=True,
            silent=True
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_xrt_data_via_api_wt_fallback(self, mock_udg):
        """Test XRT API falls back to WT curve when PC not available."""
        mock_df = pd.DataFrame({'Time': [100, 200]})
        mock_udg.getLightCurves.return_value = {
            'Datasets': ['WT_incbad_CURVE'],
            'WT_incbad_CURVE': mock_df
        }

        result = self.getter.download_xrt_data_via_api()
        self.assertIsInstance(result, pd.DataFrame)

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_xrt_data_via_api_no_datasets(self, mock_udg):
        """Test XRT API raises error when no datasets available."""
        mock_udg.getLightCurves.return_value = {'Datasets': []}

        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.download_xrt_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_xrt_data_via_api_no_suitable_curve(self, mock_udg):
        """Test XRT API raises error when no suitable curve found."""
        mock_udg.getLightCurves.return_value = {
            'Datasets': ['OTHER_DATA'],
            'OTHER_DATA': pd.DataFrame()
        }

        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.download_xrt_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    def test_download_xrt_data_via_api_not_available(self):
        """Test XRT API raises ImportError when swifttools not available."""
        with self.assertRaises(ImportError):
            self.getter.download_xrt_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_xrt_data_via_api_exception_handling(self, mock_udg):
        """Test XRT API properly re-raises exceptions."""
        mock_udg.getLightCurves.side_effect = Exception("API Error")

        with self.assertRaises(Exception):
            self.getter.download_xrt_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_burst_analyser_data_via_api_success(self, mock_udg):
        """Test successful Burst Analyser data download via API."""
        mock_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame({'Time': [100, 200]})
                }
            },
            'BAT': {
                'binning1': {
                    'band1': pd.DataFrame({'Time': [10, 20]})
                }
            }
        }
        mock_udg.getBurstAnalyser.return_value = mock_data

        result = self.getter.download_burst_analyser_data_via_api()
        mock_udg.getBurstAnalyser.assert_called_once_with(
            GRBName=self.getter.grb,
            saveData=False,
            returnData=True,
            silent=True
        )
        self.assertIsInstance(result, dict)
        self.assertIn('XRT', result)
        self.assertIn('BAT', result)

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_burst_analyser_data_via_api_empty(self, mock_udg):
        """Test Burst Analyser API raises error when data is empty."""
        mock_udg.getBurstAnalyser.return_value = {}

        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.download_burst_analyser_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_burst_analyser_data_via_api_none(self, mock_udg):
        """Test Burst Analyser API raises error when data is None."""
        mock_udg.getBurstAnalyser.return_value = None

        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.download_burst_analyser_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    def test_download_burst_analyser_data_via_api_not_available(self):
        """Test Burst Analyser API raises ImportError when swifttools not available."""
        with self.assertRaises(ImportError):
            self.getter.download_burst_analyser_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_burst_analyser_data_via_api_exception_handling(self, mock_udg):
        """Test Burst Analyser API properly re-raises exceptions."""
        mock_udg.getBurstAnalyser.side_effect = Exception("API Error")

        with self.assertRaises(Exception):
            self.getter.download_burst_analyser_data_via_api()

    def test_convert_xrt_api_data_to_csv_standard_columns(self):
        """Test XRT API data conversion with standard column names."""
        self.getter.instrument = "XRT"
        self.getter._api_data = pd.DataFrame({
            'Time': [100.0, 200.0, 300.0],
            'TimePos': [10.0, 20.0, 30.0],
            'TimeNeg': [10.0, 20.0, 30.0],
            'Flux': [1e-11, 2e-11, 3e-11],
            'FluxPos': [1e-12, 2e-12, 3e-12],
            'FluxNeg': [1e-12, 2e-12, 3e-12]
        })

        result = self.getter.convert_xrt_api_data_to_csv()

        self.assertIn('Time [s]', result.columns)
        self.assertIn('Flux [erg cm^{-2} s^{-1}]', result.columns)
        self.assertIn('Pos. flux err [erg cm^{-2} s^{-1}]', result.columns)
        self.assertEqual(len(result), 3)
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    def test_convert_xrt_api_data_to_csv_alternative_time_column(self):
        """Test XRT API data conversion with alternative time column name."""
        self.getter.instrument = "XRT"
        self.getter._api_data = pd.DataFrame({
            'T': [100.0, 200.0],
            'TimePos': [10.0, 20.0],
            'TimeNeg': [10.0, 20.0],
            'Flux': [1e-11, 2e-11],
            'FluxPos': [1e-12, 2e-12],
            'FluxNeg': [1e-12, 2e-12]
        })

        result = self.getter.convert_xrt_api_data_to_csv()

        self.assertIn('Time [s]', result.columns)
        self.assertEqual(result['Time [s]'].iloc[0], 100.0)

    def test_convert_xrt_api_data_to_csv_met_time_column(self):
        """Test XRT API data conversion with MET time column name."""
        self.getter.instrument = "XRT"
        self.getter._api_data = pd.DataFrame({
            'MET': [100.0, 200.0],
            'TimePos': [10.0, 20.0],
            'TimeNeg': [10.0, 20.0],
            'Flux': [1e-11, 2e-11],
            'FluxPos': [1e-12, 2e-12],
            'FluxNeg': [1e-12, 2e-12]
        })

        result = self.getter.convert_xrt_api_data_to_csv()

        self.assertIn('Time [s]', result.columns)

    def test_convert_xrt_api_data_to_csv_filters_zero_errors(self):
        """Test XRT API data conversion filters out zero flux errors."""
        self.getter.instrument = "XRT"
        self.getter._api_data = pd.DataFrame({
            'Time': [100.0, 200.0, 300.0],
            'TimePos': [10.0, 20.0, 30.0],
            'TimeNeg': [10.0, 20.0, 30.0],
            'Flux': [1e-11, 2e-11, 3e-11],
            'FluxPos': [1e-12, 0.0, 3e-12],  # Second row has zero error
            'FluxNeg': [1e-12, 2e-12, 3e-12]
        })

        result = self.getter.convert_xrt_api_data_to_csv()

        self.assertEqual(len(result), 2)  # Should filter out row with zero error

    def test_convert_xrt_api_data_to_csv_no_data(self):
        """Test XRT API data conversion raises error when no API data available."""
        self.getter.instrument = "XRT"

        with self.assertRaises(ValueError):
            self.getter.convert_xrt_api_data_to_csv()

    def test_convert_xrt_api_data_to_csv_none_data(self):
        """Test XRT API data conversion raises error when API data is None."""
        self.getter.instrument = "XRT"
        self.getter._api_data = None

        with self.assertRaises(ValueError):
            self.getter.convert_xrt_api_data_to_csv()

    def test_convert_burst_analyser_api_data_to_csv_flux_mode(self):
        """Test Burst Analyser API data conversion in flux mode."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'Time': [1000.0, 2000.0],
                        'TimePos': [100.0, 200.0],
                        'TimeNeg': [100.0, 200.0],
                        'Flux': [1e-11, 2e-11],
                        'FluxPos': [1e-12, 2e-12],
                        'FluxNeg': [1e-12, 2e-12]
                    })
                }
            },
            'BAT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'Time': [10.0, 20.0],
                        'TimePos': [1.0, 2.0],
                        'TimeNeg': [1.0, 2.0],
                        'Flux': [1e-10, 2e-10],
                        'FluxPos': [1e-11, 2e-11],
                        'FluxNeg': [1e-11, 2e-11]
                    })
                }
            }
        }

        result = self.getter.convert_burst_analyser_api_data_to_csv()

        self.assertIn('Time [s]', result.columns)
        self.assertIn('Flux [erg cm^{-2} s^{-1}]', result.columns)
        self.assertIn('Instrument', result.columns)
        self.assertEqual(len(result), 4)  # 2 XRT + 2 BAT
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    def test_convert_burst_analyser_api_data_to_csv_xrt_only(self):
        """Test Burst Analyser API data conversion with only XRT data."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'Time': [1000.0],
                        'TimePos': [100.0],
                        'TimeNeg': [100.0],
                        'Flux': [1e-11],
                        'FluxPos': [1e-12],
                        'FluxNeg': [1e-12]
                    })
                }
            }
        }

        result = self.getter.convert_burst_analyser_api_data_to_csv()

        self.assertEqual(len(result), 1)
        self.assertEqual(result['Instrument'].iloc[0], 'XRT')

    def test_convert_burst_analyser_api_data_to_csv_flux_density_mode(self):
        """Test Burst Analyser API data conversion in flux density mode."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux_density"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'density_band': pd.DataFrame({
                        'Time': [1000.0, 2000.0],
                        'TimePos': [100.0, 200.0],
                        'TimeNeg': [100.0, 200.0],
                        'Flux': [0.001, 0.002],  # In Jy, should be converted
                        'FluxPos': [0.0001, 0.0002],
                        'FluxNeg': [0.0001, 0.0002]
                    })
                }
            }
        }

        result = self.getter.convert_burst_analyser_api_data_to_csv()

        self.assertIn('Time [s]', result.columns)
        self.assertIn('Flux [mJy]', result.columns)
        # Check unit conversion (should multiply by 1000)
        self.assertGreater(result['Flux [mJy]'].iloc[0], 0.1)

    def test_convert_burst_analyser_api_data_to_csv_flux_density_no_conversion(self):
        """Test Burst Analyser API data conversion without unit conversion."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux_density"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'density_band': pd.DataFrame({
                        'Time': [1000.0],
                        'TimePos': [100.0],
                        'TimeNeg': [100.0],
                        'Flux': [1.0],  # Already in mJy
                        'FluxPos': [0.1],
                        'FluxNeg': [0.1]
                    })
                }
            }
        }

        result = self.getter.convert_burst_analyser_api_data_to_csv()

        # No conversion should happen if median > 0.1
        self.assertEqual(result['Flux [mJy]'].iloc[0], 1.0)

    def test_convert_burst_analyser_api_data_to_csv_no_data(self):
        """Test Burst Analyser API data conversion raises error when no API data."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"

        with self.assertRaises(ValueError):
            self.getter.convert_burst_analyser_api_data_to_csv()

    def test_convert_burst_analyser_api_data_to_csv_empty_instruments(self):
        """Test Burst Analyser API data conversion raises error when instruments empty."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        self.getter._api_data = {}

        with self.assertRaises(ValueError):
            self.getter.convert_burst_analyser_api_data_to_csv()

    def test_convert_burst_analyser_api_data_to_csv_no_flux_density_data(self):
        """Test Burst Analyser API data conversion raises error when no flux density data."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux_density"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame()  # No density in name
                }
            }
        }

        with self.assertRaises(ValueError):
            self.getter.convert_burst_analyser_api_data_to_csv()

    def test_convert_burst_analyser_api_data_to_csv_unsupported_mode(self):
        """Test Burst Analyser API data conversion raises error for unsupported mode."""
        self.getter.instrument = "BAT+XRT"
        self.getter._data_mode = "unsupported"
        self.getter._api_data = {'XRT': {}}

        with self.assertRaises(ValueError):
            self.getter.convert_burst_analyser_api_data_to_csv()

    def test_convert_burst_analyser_api_data_to_csv_with_t_column(self):
        """Test Burst Analyser API data conversion with 'T' column name."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'T': [1000.0],  # Alternative name
                        'TimePos': [100.0],
                        'TimeNeg': [100.0],
                        'Flux': [1e-11],
                        'FluxPos': [1e-12],
                        'FluxNeg': [1e-12]
                    })
                }
            }
        }

        result = self.getter.convert_burst_analyser_api_data_to_csv()

        self.assertIn('Time [s]', result.columns)

    @mock.patch("os.path.isfile")
    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    def test_collect_data_api_fallback_on_exception(self, isfile):
        """Test that collect_data falls back to legacy when API raises exception."""
        isfile.return_value = False
        self.getter.instrument = "XRT"

        # Make API method fail
        self.getter.download_xrt_data_via_api = MagicMock(side_effect=Exception("API failed"))
        self.getter.download_directly = MagicMock()

        self.getter.collect_data()

        # Should fall back to legacy method
        self.getter.download_xrt_data_via_api.assert_called_once()
        self.getter.download_directly.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    def test_collect_data_bat_xrt_api_fallback_on_exception(self, isfile):
        """Test BAT+XRT collect_data falls back to legacy when API fails."""
        isfile.return_value = False
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"

        # Make API method fail
        self.getter.download_burst_analyser_data_via_api = MagicMock(side_effect=Exception("API failed"))
        self.getter.download_integrated_flux_data = MagicMock()

        self.getter.collect_data()

        # Should fall back to legacy method
        self.getter.download_burst_analyser_data_via_api.assert_called_once()
        self.getter.download_integrated_flux_data.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch('requests.get')
    def test_collect_data_prompt_no_api(self, get, isfile):
        """Test prompt data collection doesn't use API."""
        isfile.return_value = False
        self.getter.transient_type = "prompt"
        get.return_value = MagicMock()
        get.return_value.text = "Some valid response"
        self.getter.download_directly = MagicMock()

        self.getter.collect_data()

        self.getter.download_directly.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch('requests.get')
    def test_collect_data_prompt_no_lightcurve(self, get, isfile):
        """Test prompt data collection raises error when no lightcurve."""
        isfile.return_value = False
        self.getter.transient_type = "prompt"
        get.return_value = MagicMock()
        get.return_value.text = "No Light curve available"

        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.collect_data()

    def test_get_data_bat_xrt_warning(self):
        """Test get_data logs warning for BAT+XRT instrument."""
        self.getter.instrument = "BAT+XRT"
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock(return_value=pd.DataFrame())

        with mock.patch.object(redback.utils.logger, 'warning') as mock_warning:
            self.getter.get_data()
            mock_warning.assert_called()
            args = mock_warning.call_args[0][0]
            self.assertIn("BAT and XRT", args)

    def test_get_data_xrt_warning(self):
        """Test get_data logs warning for XRT-only instrument."""
        self.getter.instrument = "XRT"
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock(return_value=pd.DataFrame())

        with mock.patch.object(redback.utils.logger, 'warning') as mock_warning:
            self.getter.get_data()
            mock_warning.assert_called()
            args = mock_warning.call_args[0][0]
            self.assertIn("XRT data", args)

    @mock.patch("numpy.loadtxt")
    def test_convert_xrt_data_to_csv(self, mock_loadtxt):
        """Test legacy XRT data conversion from raw file."""
        self.getter.instrument = "XRT"
        # Simulate raw XRT data (6 columns)
        mock_data = np.array([
            [100.0, 10.0, 10.0, 1e-11, 1e-12, 1e-12],
            [200.0, 20.0, 20.0, 2e-11, 0.0, 2e-12],  # Zero error row should be filtered
            [300.0, 30.0, 30.0, 3e-11, 3e-12, 3e-12]
        ])
        mock_loadtxt.return_value = mock_data

        result = self.getter.convert_xrt_data_to_csv()

        self.assertEqual(len(result), 2)  # One row filtered out
        self.assertIn('Time [s]', result.columns)
        self.assertIn('Flux [erg cm^{-2} s^{-1}]', result.columns)
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    @mock.patch("numpy.loadtxt")
    def test_convert_raw_prompt_data_to_csv(self, mock_loadtxt):
        """Test legacy prompt data conversion from raw file."""
        self.getter.transient_type = "prompt"
        # Simulate raw prompt data (11 columns)
        mock_data = np.array([
            [100.0, 1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4, 5.0, 0.5],
            [200.0, 1.1, 0.11, 2.1, 0.21, 3.1, 0.31, 4.1, 0.41, 5.1, 0.51]
        ])
        mock_loadtxt.return_value = mock_data

        result = self.getter.convert_raw_prompt_data_to_csv()

        self.assertEqual(len(result), 2)
        self.assertIn('Time [s]', result.columns)
        self.assertIn('flux_15_25 [counts/s/det]', result.columns)
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    def test_convert_integrated_flux_data_to_csv(self):
        """Test legacy integrated flux data conversion from raw file."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"

        # Create mock raw data file
        raw_data = """Header lines to skip
More header
NO NO NO
! XRT
100.0\t10.0\t10.0\t1e-11\t1e-12\t1e-12
200.0\t20.0\t20.0\t2e-11\t2e-12\t2e-12
! BAT
-50.0\t5.0\t5.0\t1e-10\t1e-11\t1e-11
"""
        with open(self.getter.raw_file_path, 'w') as f:
            f.write(raw_data)

        result = self.getter.convert_integrated_flux_data_to_csv()

        self.assertEqual(len(result), 3)
        self.assertIn('Time [s]', result.columns)
        self.assertIn('Instrument', result.columns)
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    def test_convert_flux_density_data_to_csv(self):
        """Test legacy flux density data conversion from raw file."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux_density"

        # Create mock raw data file (flux density in Jy, will be converted to mJy)
        raw_data = """Header lines to skip
More header
NO NO NO
100.0\t10.0\t10.0\t0.001\t0.0001\t0.0001
200.0\t20.0\t20.0\t0.002\t0.0002\t0.0002
"""
        with open(self.getter.raw_file_path, 'w') as f:
            f.write(raw_data)

        result = self.getter.convert_flux_density_data_to_csv()

        self.assertEqual(len(result), 2)
        self.assertIn('Flux [mJy]', result.columns)
        # Check unit conversion (multiplied by 1000)
        self.assertEqual(result['Flux [mJy]'].iloc[0], 1.0)  # 0.001 * 1000
        self.assertEqual(result['Flux [mJy]'].iloc[1], 2.0)  # 0.002 * 1000
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    def test_convert_raw_afterglow_data_to_csv_flux(self):
        """Test convert_raw_afterglow_data_to_csv routes to flux conversion."""
        self.getter.data_mode = "flux"
        self.getter.convert_integrated_flux_data_to_csv = MagicMock(return_value=pd.DataFrame())
        self.getter.convert_flux_density_data_to_csv = MagicMock()

        self.getter.convert_raw_afterglow_data_to_csv()

        self.getter.convert_integrated_flux_data_to_csv.assert_called_once()
        self.getter.convert_flux_density_data_to_csv.assert_not_called()

    def test_convert_raw_afterglow_data_to_csv_flux_density(self):
        """Test convert_raw_afterglow_data_to_csv routes to flux density conversion."""
        self.getter.data_mode = "flux_density"
        self.getter.convert_integrated_flux_data_to_csv = MagicMock()
        self.getter.convert_flux_density_data_to_csv = MagicMock(return_value=pd.DataFrame())

        self.getter.convert_raw_afterglow_data_to_csv()

        self.getter.convert_integrated_flux_data_to_csv.assert_not_called()
        self.getter.convert_flux_density_data_to_csv.assert_called_once()

    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("urllib.request.urlcleanup")
    def test_download_directly_success(self, mock_cleanup, mock_urlretrieve):
        """Test direct download succeeds."""
        self.getter.download_directly()

        mock_urlretrieve.assert_called_once()
        mock_cleanup.assert_called_once()

    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("urllib.request.urlcleanup")
    def test_download_directly_failure(self, mock_cleanup, mock_urlretrieve):
        """Test direct download handles failure gracefully."""
        mock_urlretrieve.side_effect = Exception("Network error")

        # Should not raise, just log warning
        self.getter.download_directly()

        mock_cleanup.assert_called_once()

    @mock.patch("redback.get_data.swift.fetch_driver")
    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("urllib.request.urlcleanup")
    def test_download_flux_density_data_success(self, mock_cleanup, mock_urlretrieve, mock_driver):
        """Test flux density download with Selenium driver."""
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.current_url = "http://test.url/data.csv"

        self.getter.download_flux_density_data()

        mock_driver_instance.get.assert_called_once()
        mock_driver_instance.quit.assert_called()
        mock_cleanup.assert_called_once()

    @mock.patch("redback.get_data.swift.fetch_driver")
    @mock.patch("urllib.request.urlcleanup")
    def test_download_flux_density_data_failure(self, mock_cleanup, mock_driver):
        """Test flux density download handles driver failure."""
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.get.side_effect = Exception("Driver error")

        # Should not raise, just log warning
        self.getter.download_flux_density_data()

        mock_driver_instance.quit.assert_called()
        mock_cleanup.assert_called_once()

    @mock.patch("redback.get_data.swift.fetch_driver")
    @mock.patch("redback.get_data.swift.check_element")
    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("urllib.request.urlcleanup")
    def test_download_integrated_flux_data_success(self, mock_cleanup, mock_urlretrieve, mock_check, mock_driver):
        """Test integrated flux download with Selenium driver."""
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.current_url = "http://test.url/data.csv"
        mock_check.return_value = True

        self.getter.download_integrated_flux_data()

        mock_driver_instance.get.assert_called_once()
        mock_driver_instance.quit.assert_called()
        mock_cleanup.assert_called_once()

    @mock.patch("redback.get_data.swift.fetch_driver")
    @mock.patch("urllib.request.urlcleanup")
    def test_download_integrated_flux_data_failure(self, mock_cleanup, mock_driver):
        """Test integrated flux download handles driver failure."""
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.get.side_effect = Exception("Driver error")

        # Should not raise, just log warning
        self.getter.download_integrated_flux_data()

        mock_driver_instance.quit.assert_called()
        mock_cleanup.assert_called_once()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    def test_collect_data_writes_marker_file_xrt(self):
        """Test that API data collection writes marker file for XRT."""
        # Remove the file if it exists so we can test the write
        if os.path.isfile(self.getter.raw_file_path):
            os.remove(self.getter.raw_file_path)

        self.getter.instrument = "XRT"
        mock_df = pd.DataFrame({'Time': [100]})
        self.getter.download_xrt_data_via_api = MagicMock(return_value=mock_df)

        self.getter.collect_data()

        self.assertTrue(hasattr(self.getter, '_api_data'))
        self.assertTrue(os.path.isfile(self.getter.raw_file_path))
        with open(self.getter.raw_file_path, 'r') as f:
            content = f.read()
            self.assertIn('swifttools API', content)

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    def test_collect_data_writes_marker_file_bat_xrt(self):
        """Test that API data collection writes marker file for BAT+XRT."""
        # Remove the file if it exists so we can test the write
        if os.path.isfile(self.getter.raw_file_path):
            os.remove(self.getter.raw_file_path)

        self.getter.instrument = "BAT+XRT"
        mock_data = {'XRT': {}, 'BAT': {}}
        self.getter.download_burst_analyser_data_via_api = MagicMock(return_value=mock_data)

        self.getter.collect_data()

        self.assertTrue(hasattr(self.getter, '_api_data'))
        self.assertTrue(os.path.isfile(self.getter.raw_file_path))
        with open(self.getter.raw_file_path, 'r') as f:
            content = f.read()
            self.assertIn('swifttools API', content)

    def test_stripped_grb_property(self):
        """Test stripped_grb removes GRB prefix."""
        self.assertEqual(self.getter.stripped_grb, "050202")

    def test_grb_setter_adds_prefix(self):
        """Test setting grb without prefix adds GRB prefix."""
        self.getter.grb = "123456"
        self.assertEqual(self.getter.grb, "GRB123456")

    def test_grb_setter_keeps_prefix(self):
        """Test setting grb with prefix keeps it."""
        self.getter.grb = "GRB999999"
        self.assertEqual(self.getter.grb, "GRB999999")

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_download_xrt_data_via_api_missing_datasets_key(self, mock_udg):
        """Test XRT API handles missing 'Datasets' key."""
        mock_udg.getLightCurves.return_value = {}

        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.download_xrt_data_via_api()

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_full_xrt_api_flow(self, mock_udg):
        """Integration test: Full XRT API data flow from download to CSV."""
        # Remove existing files
        if os.path.isfile(self.getter.raw_file_path):
            os.remove(self.getter.raw_file_path)
        if os.path.isfile(self.getter.processed_file_path):
            os.remove(self.getter.processed_file_path)

        self.getter.instrument = "XRT"
        mock_df = pd.DataFrame({
            'Time': [100.0, 200.0, 300.0],
            'TimePos': [10.0, 20.0, 30.0],
            'TimeNeg': [10.0, 20.0, 30.0],
            'Flux': [1e-11, 2e-11, 3e-11],
            'FluxPos': [1e-12, 2e-12, 3e-12],
            'FluxNeg': [1e-12, 2e-12, 3e-12]
        })
        mock_udg.getLightCurves.return_value = {
            'Datasets': ['PC_incbad_CURVE'],
            'PC_incbad_CURVE': mock_df
        }

        # Run full flow
        result = self.getter.get_data()

        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn('Time [s]', result.columns)
        self.assertIn('Flux [erg cm^{-2} s^{-1}]', result.columns)
        self.assertTrue(os.path.isfile(self.getter.raw_file_path))
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_full_bat_xrt_api_flow_flux(self, mock_udg):
        """Integration test: Full BAT+XRT API data flow for flux mode."""
        # Remove existing files
        if os.path.isfile(self.getter.raw_file_path):
            os.remove(self.getter.raw_file_path)
        if os.path.isfile(self.getter.processed_file_path):
            os.remove(self.getter.processed_file_path)

        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        mock_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'Time': [1000.0, 2000.0],
                        'TimePos': [100.0, 200.0],
                        'TimeNeg': [100.0, 200.0],
                        'Flux': [1e-11, 2e-11],
                        'FluxPos': [1e-12, 2e-12],
                        'FluxNeg': [1e-12, 2e-12]
                    })
                }
            },
            'BAT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'Time': [10.0, 20.0],
                        'TimePos': [1.0, 2.0],
                        'TimeNeg': [1.0, 2.0],
                        'Flux': [1e-10, 2e-10],
                        'FluxPos': [1e-11, 2e-11],
                        'FluxNeg': [1e-11, 2e-11]
                    })
                }
            }
        }
        mock_udg.getBurstAnalyser.return_value = mock_data

        # Run full flow
        result = self.getter.get_data()

        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 2 XRT + 2 BAT
        self.assertIn('Time [s]', result.columns)
        self.assertIn('Instrument', result.columns)

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    def test_full_bat_xrt_api_flow_flux_density(self, mock_udg):
        """Integration test: Full BAT+XRT API data flow for flux density mode."""
        # Change to flux_density mode
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux_density"

        # Update directory structure for new data_mode
        self.getter.directory_path, self.getter.raw_file_path, self.getter.processed_file_path = \
            self.getter.create_directory_structure()

        # Remove existing files
        if os.path.isfile(self.getter.raw_file_path):
            os.remove(self.getter.raw_file_path)
        if os.path.isfile(self.getter.processed_file_path):
            os.remove(self.getter.processed_file_path)

        mock_data = {
            'XRT': {
                'binning1': {
                    'density_band': pd.DataFrame({
                        'Time': [1000.0, 2000.0],
                        'TimePos': [100.0, 200.0],
                        'TimeNeg': [100.0, 200.0],
                        'Flux': [0.001, 0.002],  # In Jy
                        'FluxPos': [0.0001, 0.0002],
                        'FluxNeg': [0.0001, 0.0002]
                    })
                }
            }
        }
        mock_udg.getBurstAnalyser.return_value = mock_data

        # Run full flow
        result = self.getter.get_data()

        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Flux [mJy]', result.columns)
        # Check unit conversion happened
        self.assertGreater(result['Flux [mJy]'].iloc[0], 0.1)

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', True)
    @mock.patch('redback.get_data.swift.udg')
    @mock.patch("os.path.isfile")
    @mock.patch('redback.get_data.swift.fetch_driver')
    @mock.patch('requests.get')
    def test_api_fallback_to_legacy_complete_flow(self, mock_get, mock_driver, mock_isfile, mock_udg):
        """Integration test: API failure triggers complete legacy fallback."""
        mock_isfile.return_value = False
        self.getter.instrument = "XRT"

        # API fails
        mock_udg.getLightCurves.side_effect = Exception("API unavailable")

        # Legacy method works
        mock_get.return_value = MagicMock()
        mock_get.return_value.text = "Valid data"

        # Mock the urlretrieve to create a valid raw file
        with mock.patch("urllib.request.urlretrieve") as mock_retrieve:
            def create_raw_file(url, path):
                # Create a valid XRT data file
                data = "! Comment\n! READ\n100.0 10.0 10.0 1e-11 1e-12 1e-12\n200.0 20.0 20.0 2e-11 2e-12 2e-12"
                with open(path, 'w') as f:
                    f.write(data)
            mock_retrieve.side_effect = create_raw_file

            with mock.patch("urllib.request.urlcleanup"):
                self.getter.collect_data()

        # Verify API was tried and failed
        mock_udg.getLightCurves.assert_called_once()

    def test_convert_xrt_api_data_missing_flux_columns(self):
        """Test XRT API conversion handles missing optional columns gracefully."""
        self.getter.instrument = "XRT"
        # Only time columns, missing flux
        self.getter._api_data = pd.DataFrame({
            'Time': [100.0, 200.0],
            'TimePos': [10.0, 20.0],
            'TimeNeg': [10.0, 20.0]
        })

        result = self.getter.convert_xrt_api_data_to_csv()

        # Should still create file with available columns
        self.assertIn('Time [s]', result.columns)
        self.assertTrue(os.path.isfile(self.getter.processed_file_path))

    def test_convert_burst_analyser_empty_dataframes(self):
        """Test Burst Analyser conversion handles empty dataframes."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        self.getter._api_data = {
            'XRT': {
                'binning1': {
                    'band1': pd.DataFrame()  # Empty dataframe
                }
            }
        }

        with self.assertRaises(ValueError):
            self.getter.convert_burst_analyser_api_data_to_csv()

    def test_convert_burst_analyser_bat_only(self):
        """Test Burst Analyser API data conversion with only BAT data."""
        self.getter.instrument = "BAT+XRT"
        self.getter.data_mode = "flux"
        self.getter._api_data = {
            'BAT': {
                'binning1': {
                    'band1': pd.DataFrame({
                        'Time': [10.0, 20.0],
                        'TimePos': [1.0, 2.0],
                        'TimeNeg': [1.0, 2.0],
                        'Flux': [1e-10, 2e-10],
                        'FluxPos': [1e-11, 2e-11],
                        'FluxNeg': [1e-11, 2e-11]
                    })
                }
            }
        }

        result = self.getter.convert_burst_analyser_api_data_to_csv()

        self.assertEqual(len(result), 2)
        self.assertEqual(result['Instrument'].iloc[0], 'BAT')

    @mock.patch('redback.get_data.swift.SWIFTTOOLS_AVAILABLE', False)
    @mock.patch('requests.get')
    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("urllib.request.urlcleanup")
    def test_legacy_xrt_complete_flow(self, mock_cleanup, mock_retrieve, mock_get):
        """Integration test: Complete legacy XRT flow when API unavailable."""
        # Remove file if exists so collect_data will try to download
        if os.path.isfile(self.getter.raw_file_path):
            os.remove(self.getter.raw_file_path)

        self.getter.instrument = "XRT"
        mock_get.return_value = MagicMock()
        mock_get.return_value.text = "Valid data"

        def create_raw_file(url, path):
            data = "! Comment\n! READ\n100.0 10.0 10.0 1e-11 1e-12 1e-12\n200.0 20.0 20.0 2e-11 2e-12 2e-12"
            with open(path, 'w') as f:
                f.write(data)
        mock_retrieve.side_effect = create_raw_file

        self.getter.collect_data()

        mock_retrieve.assert_called_once()
        self.assertTrue(os.path.isfile(self.getter.raw_file_path))

    def test_get_swift_id_from_grb_long_id(self):
        """Test get_swift_id_from_grb with already long ID."""
        with mock.patch("astropy.io.ascii.read") as ascii_read:
            swift_id = "12345678901"  # Already 11 digits
            ascii_read.return_value = dict(col1=[f"GRB{self.grb}"], col2=[swift_id])
            result = self.getter.get_swift_id_from_grb()
            self.assertEqual(swift_id, result)

    @mock.patch("redback.get_data.utils.get_trigger_number")
    def test_trigger_property(self, mock_trigger):
        """Test trigger property caches correctly."""
        mock_trigger.return_value = "123456"
        trigger1 = self.getter.trigger
        trigger2 = self.getter.trigger
        self.assertEqual(trigger1, trigger2)
        # Called each time since it's a property
        self.assertEqual(mock_trigger.call_count, 2)


class TestLasairDataGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        self.transient = "ZTF19aagqkrq"
        self.transient_type = "afterglow"
        self.getter = redback.get_data.LasairDataGetter(
            transient=self.transient, transient_type=self.transient_type)

    def tearDown(self) -> None:
        shutil.rmtree("GRBData", ignore_errors=True)
        del self.transient
        del self.transient_type
        del self.getter

    def test_directory_path(self):
        expected_directory_path, expected_raw_file_path, expected_processed_file_path = \
            redback.get_data.directory.lasair_directory_structure(
                transient_type=self.transient_type, transient=self.transient)
        self.assertEqual(expected_directory_path, self.getter.directory_path)
        self.assertEqual(expected_raw_file_path, self.getter.raw_file_path)
        self.assertEqual(expected_processed_file_path, self.getter.processed_file_path)

    def test_url(self):
        expected = f"https://lasair-ztf.lsst.ac.uk/objects/{self.transient}"
        self.assertEqual(expected, self.getter.url)

    @mock.patch("os.path.isfile")
    def test_collect_data_rawfile_exists(self, isfile):
        isfile.return_value = True
        redback.utils.logger.warning = MagicMock()
        self.getter.collect_data()
        isfile.assert_called_once()
        redback.utils.logger.warning.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch("requests.get")
    def test_collect_data_no_lightcurve_available(self, get, isfile):
        isfile.return_value = False
        get.return_value = MagicMock()
        get.return_value.__setattr__('text', 'not in database')
        with self.assertRaises(ValueError):
            self.getter.collect_data()

    @mock.patch("os.path.isfile")
    @mock.patch("requests.get")
    @mock.patch("pandas.read_html")
    def test_collect_data(self, read, get, isfile):
        isfile.return_value = False
        get.return_value = MagicMock()
        get.return_value.__setattr__('text', '')
        self.getter.collect_data()
        read.assert_called_with(self.getter.url)

class TestFinkDataGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        self.transient = "ZTF22abdjqlm"
        self.transient_type = "supernova"
        self.getter = redback.get_data.FinkDataGetter(
            transient=self.transient, transient_type=self.transient_type)

    def tearDown(self) -> None:
        shutil.rmtree("supernova", ignore_errors=True)
        del self.transient
        del self.transient_type
        del self.getter

    def test_directory_path(self):
        expected_directory_path, expected_raw_file_path, expected_processed_file_path = \
            redback.get_data.directory.fink_directory_structure(
                transient_type=self.transient_type, transient=self.transient)
        self.assertEqual(expected_directory_path, self.getter.directory_path)
        self.assertEqual(expected_raw_file_path, self.getter.raw_file_path)
        self.assertEqual(expected_processed_file_path, self.getter.processed_file_path)

    def test_url(self):
        expected = "https://api.fink-portal.org/api/v1/objects"
        self.assertEqual(expected, self.getter.url)

    def test_object_id(self):
        expected = self.transient
        self.assertEqual(expected, self.getter.objectId)

    @mock.patch("os.path.isfile")
    def test_collect_data_rawfile_exists(self, isfile):
        isfile.return_value = True
        redback.utils.logger.warning = MagicMock()
        self.getter.collect_data()
        isfile.assert_called_once()
        redback.utils.logger.warning.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch("pandas.read_csv")
    def test_collect_data_no_lightcurve_available(self, get, isfile):
        isfile.return_value = False
        get.return_value = MagicMock()
        get.return_value.__setattr__('len', 0)
        with self.assertRaises(ValueError):
            self.getter.collect_data()

    @mock.patch("os.path.isfile")
    @mock.patch("requests.post")
    def test_collect_data(self, post, isfile):
        isfile.return_value = False
        post.return_value = MagicMock()
        post.return_value.__setattr__('content', bytearray(b'0    0.3193\nName: i:sigmagap, dtype: float64'))
        json = {'objectId': self.getter.objectId, 'output-format': 'csv', 'withupperlim': 'True'}
        self.getter.collect_data()
        post.assert_called_with(url=self.getter.url, json=json)


class TestUtilsLogging(unittest.TestCase):
    """Tests for logging in get_data/utils.py module."""

    def test_get_trigger_number_not_found_logs_error(self):
        """Test that trigger not found logs error and raises exception."""
        with self.assertRaises(redback.get_data.utils.TriggerNotFoundError):
            redback.get_data.utils.get_trigger_number("NONEXISTENT999999")

    def test_get_batse_trigger_not_found_logs_error(self):
        """Test that BATSE trigger not found logs error and raises exception."""
        with self.assertRaises(ValueError):
            redback.get_data.utils.get_batse_trigger_from_grb("NONEXISTENT999999")

    def test_get_batse_trigger_success(self):
        """Test successful BATSE trigger lookup."""
        # GRB910425A should be in the BATSE table
        trigger = redback.get_data.utils.get_batse_trigger_from_grb("910425A")
        self.assertIsInstance(trigger, int)


class TestBATSEDataGetterLogging(unittest.TestCase):
    """Tests for logging in BATSE data getter."""

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self):
        self.grb = "GRB000526"
        self.getter = redback.get_data.BATSEDataGetter(grb=self.grb)

    def tearDown(self):
        _delete_downloaded_files()

    @mock.patch("urllib.request.urlretrieve")
    def test_collect_data_success_logs_info(self, mock_urlretrieve):
        """Test that successful data collection logs info messages."""
        mock_urlretrieve.return_value = None
        self.getter.collect_data()
        mock_urlretrieve.assert_called_once_with(self.getter.url, self.getter.raw_file_path)

    @mock.patch("urllib.request.urlretrieve")
    def test_collect_data_failure_logs_error(self, mock_urlretrieve):
        """Test that failed data collection logs error."""
        mock_urlretrieve.side_effect = Exception("Network error")
        with self.assertRaises(Exception):
            self.getter.collect_data()

    @mock.patch("astropy.io.fits.open")
    @mock.patch.object(redback.get_data.BATSEDataGetter, "_get_columns")
    def test_convert_raw_data_success_logs_info(self, mock_get_cols, mock_fits_open):
        """Test that successful conversion logs info."""
        mock_get_cols.return_value = np.zeros((10, 10))
        mock_fits_open.return_value.__enter__ = MagicMock()
        mock_fits_open.return_value.__exit__ = MagicMock()

        # Create dummy raw file
        import os
        os.makedirs(os.path.dirname(self.getter.raw_file_path), exist_ok=True)
        with open(self.getter.raw_file_path, 'w') as f:
            f.write("dummy")

        df = self.getter.convert_raw_data_to_csv()
        self.assertIsInstance(df, pd.DataFrame)

    @mock.patch("astropy.io.fits.open")
    def test_convert_raw_data_failure_logs_error(self, mock_fits_open):
        """Test that failed conversion logs error."""
        mock_fits_open.side_effect = Exception("Invalid FITS file")
        with self.assertRaises(Exception):
            self.getter.convert_raw_data_to_csv()


class TestDirectoryLogging(unittest.TestCase):
    """Tests for logging in directory structure creation."""

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def test_swift_prompt_invalid_bin_size_logs_error(self):
        """Test that invalid bin size logs error before raising ValueError."""
        with self.assertRaises(ValueError) as context:
            redback.get_data.directory.swift_prompt_directory_structure(
                grb='GRB123456', bin_size='invalid_bin_size'
            )
        self.assertIn('invalid_bin_size', str(context.exception))

    def test_spectrum_directory_structure_creates_correctly(self):
        """Test spectrum directory structure is created with logging."""
        structure = redback.get_data.directory.spectrum_directory_structure(transient='test_spectrum')
        self.assertEqual(structure.directory_path, 'spectrum/')
        self.assertIn('test_spectrum', structure.raw_file_path)
        self.assertIn('test_spectrum', structure.processed_file_path)
        # Clean up
        shutil.rmtree('spectrum', ignore_errors=True)
class TestUtilsLogging(unittest.TestCase):
    """Tests for logging in get_data/utils.py module."""

    def test_get_trigger_number_not_found_logs_error(self):
        """Test that trigger not found logs error and raises exception."""
        with self.assertRaises(redback.get_data.utils.TriggerNotFoundError):
            redback.get_data.utils.get_trigger_number("NONEXISTENT999999")

    def test_get_batse_trigger_not_found_logs_error(self):
        """Test that BATSE trigger not found logs error and raises exception."""
        with self.assertRaises(ValueError):
            redback.get_data.utils.get_batse_trigger_from_grb("NONEXISTENT999999")

    def test_get_batse_trigger_success(self):
        """Test successful BATSE trigger lookup."""
        # GRB910425A should be in the BATSE table
        trigger = redback.get_data.utils.get_batse_trigger_from_grb("910425A")
        self.assertIsInstance(trigger, int)


class TestBATSEDataGetterLogging(unittest.TestCase):
    """Tests for logging in BATSE data getter."""

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self):
        self.grb = "GRB000526"
        self.getter = redback.get_data.BATSEDataGetter(grb=self.grb)

    def tearDown(self):
        _delete_downloaded_files()

    @mock.patch("urllib.request.urlretrieve")
    def test_collect_data_success_logs_info(self, mock_urlretrieve):
        """Test that successful data collection logs info messages."""
        mock_urlretrieve.return_value = None
        self.getter.collect_data()
        mock_urlretrieve.assert_called_once_with(self.getter.url, self.getter.raw_file_path)

    @mock.patch("urllib.request.urlretrieve")
    def test_collect_data_failure_logs_error(self, mock_urlretrieve):
        """Test that failed data collection logs error."""
        mock_urlretrieve.side_effect = Exception("Network error")
        with self.assertRaises(Exception):
            self.getter.collect_data()

    @mock.patch("astropy.io.fits.open")
    @mock.patch.object(redback.get_data.BATSEDataGetter, "_get_columns")
    def test_convert_raw_data_success_logs_info(self, mock_get_cols, mock_fits_open):
        """Test that successful conversion logs info."""
        mock_get_cols.return_value = np.zeros((10, 10))
        mock_fits_open.return_value.__enter__ = MagicMock()
        mock_fits_open.return_value.__exit__ = MagicMock()

        # Create dummy raw file
        import os
        os.makedirs(os.path.dirname(self.getter.raw_file_path), exist_ok=True)
        with open(self.getter.raw_file_path, 'w') as f:
            f.write("dummy")

        df = self.getter.convert_raw_data_to_csv()
        self.assertIsInstance(df, pd.DataFrame)

    @mock.patch("astropy.io.fits.open")
    def test_convert_raw_data_failure_logs_error(self, mock_fits_open):
        """Test that failed conversion logs error."""
        mock_fits_open.side_effect = Exception("Invalid FITS file")
        with self.assertRaises(Exception):
            self.getter.convert_raw_data_to_csv()


class TestDirectoryLogging(unittest.TestCase):
    """Tests for logging in directory structure creation."""

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def test_swift_prompt_invalid_bin_size_logs_error(self):
        """Test that invalid bin size logs error before raising ValueError."""
        with self.assertRaises(ValueError) as context:
            redback.get_data.directory.swift_prompt_directory_structure(
                grb='GRB123456', bin_size='invalid_bin_size'
            )
        self.assertIn('invalid_bin_size', str(context.exception))

    def test_spectrum_directory_structure_creates_correctly(self):
        """Test spectrum directory structure is created with logging."""
        structure = redback.get_data.directory.spectrum_directory_structure(transient='test_spectrum')
        self.assertEqual(structure.directory_path, 'spectrum/')
        self.assertIn('test_spectrum', structure.raw_file_path)
        self.assertIn('test_spectrum', structure.processed_file_path)
        # Clean up
        shutil.rmtree('spectrum', ignore_errors=True)
