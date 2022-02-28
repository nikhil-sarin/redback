import filecmp
import os.path
import shutil
import unittest
from unittest import mock
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd

import redback

_dirname = os.path.dirname(__file__)


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

    def test_transient_directory_structure(self):
        transient = "abc"
        transient_type = "tde"
        self.data_mode = "photometry"
        structure = redback.get_data.directory.transient_directory_structure(
            transient=transient, transient_type=transient_type, data_mode=self.data_mode)
        self.assertEqual(f"{transient_type}/{self.data_mode}/", structure.directory_path)
        self.assertEqual(f"{transient_type}/{self.data_mode}/{transient}_rawdata.csv", structure.raw_file_path)
        self.assertEqual(f"{transient_type}/{self.data_mode}/{transient}.csv", structure.processed_file_path)


def _delete_downloaded_files():
    for folder in ["GRBData", "kilonova", "supernova", "tidal_disruption_event"]:
        shutil.rmtree(folder, ignore_errors=True)


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

    def test_grb(self):
        self.assertEqual(self.grb, self.getter.grb)

        self.getter.grb = self.grb.lstrip("GRB")
        self.assertEqual(self.grb, self.getter.grb)

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

    def test_get_data(self):
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock()
        self.getter.get_data()
        self.getter.collect_data.assert_called_once()
        self.getter.convert_raw_data_to_csv.assert_called_once()

    @mock.patch("urllib.request.urlretrieve")
    def collect_data(self, urlretrieve):
        self.getter.collect_data()
        urlretrieve.assert_called_once()

    @mock.patch("astropy.io.fits.open")
    @mock.patch("pandas.DataFrame")
    def test_convert_raw_data_to_csv(self, DataFrame, fits_open):
        pass  # Add unittests maybe. This is also covered by the reference file tests.


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

    def test_get_data(self):
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock()
        self.getter.get_data()
        self.getter.collect_data.assert_called_once()
        self.getter.convert_raw_data_to_csv.assert_called_once()

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

    @mock.patch("pandas.isna")
    @mock.patch("os.path.isfile")
    def test_convert_raw_data_to_csv_file_exists(self, isfile, isna):
        isfile.return_value = True
        self.getter.convert_raw_data_to_csv()
        isfile.assert_called_once_with(self.getter.processed_file_path)
        isna.assert_not_called()

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
        data = dict(event=[self.transient], alias=alias)
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
        data = dict(event=[self.transient], alias=alias)
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
            self.getter.data_mode = 'photometry'

    def test_set_valid_instrument(self):
        for valid_instrument in ['BAT+XRT', 'XRT']:
            self.getter.instrument = valid_instrument
            self.assertEqual(valid_instrument, self.getter.instrument)

    def test_set_invalid_instrument(self):
        with self.assertRaises(ValueError):
            self.getter.instrument = "potato"

    def test_grb(self):
        expected = f"GRB{self.grb}"
        self.assertEqual(expected, self.getter.grb)

        self.getter.grb = expected
        self.assertEqual(expected, self.getter.grb)

    def test_stripped_grb(self):
        self.assertEqual(self.grb, self.getter.stripped_grb)
        grb_with_prepended_grb = f"GRB{self.grb}"
        self.getter.grb = grb_with_prepended_grb
        self.assertEqual(self.grb, self.getter.stripped_grb)

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
        self.assertListEqual(list(expected), list([self.getter.directory_path, self.getter.raw_file_path, self.getter.processed_file_path]))
        afterglow_directory_structure.assert_called_with(grb=f"GRB{self.grb}", data_mode=self.data_mode, instrument=self.instrument)

    @mock.patch("redback.get_data.directory.swift_prompt_directory_structure")
    def test_create_directory_structure_prompt(self, prompt_directory_structure):
        expected = "0", "1", "2"
        prompt_directory_structure.return_value = expected
        self.getter = redback.get_data.swift.SwiftDataGetter(
            grb=self.grb, transient_type="prompt", data_mode=self.data_mode,
            instrument=self.instrument, bin_size=self.bin_size)  # method is called in constructor
        self.assertListEqual(list(expected), list([self.getter.directory_path, self.getter.raw_file_path, self.getter.processed_file_path]))
        prompt_directory_structure.assert_called_with(grb=f"GRB{self.grb}", bin_size=self.bin_size)

    def test_get_data(self):
        self.getter.create_directory_structure = MagicMock()
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock()
        self.getter.get_data()
        self.getter.create_directory_structure.assert_not_called()
        self.getter.collect_data.assert_called_once()
        self.getter.convert_raw_data_to_csv.assert_called_once()

    @mock.patch("os.path.isfile")
    def test_collect_data_rawfile_exists(self, isfile):
        isfile.return_value = True
        redback.utils.logger.warning = MagicMock()
        self.getter.collect_data()
        isfile.assert_called_once()
        redback.utils.logger.warning.assert_called_once()

    @mock.patch("os.path.isfile")
    @mock.patch('requests.get')
    def test_collect_data_no_lightcurve_available(self, get, isfile):
        isfile.return_value = False
        get.return_value = MagicMock()
        get.return_value.__setattr__('text', 'No Light curve available')
        with self.assertRaises(redback.redback_errors.WebsiteExist):
            self.getter.collect_data()

    @mock.patch("os.path.isfile")
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

    def _mock_converter_functions(self):
        self.getter.convert_xrt_data_to_csv = MagicMock()
        self.getter.convert_raw_afterglow_data_to_csv = MagicMock()
        self.getter.convert_raw_prompt_data_to_csv = MagicMock()

    def test_convert_raw_data_to_csv_file_exists(self):
        self._mock_converter_functions()
        with open(self.getter.processed_file_path, "w"):  # create empty file
            pass
        self.getter.convert_raw_data_to_csv()

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


class TestReferenceFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _delete_downloaded_files()

    @classmethod
    def tearDownClass(cls) -> None:
        _delete_downloaded_files()

    def setUp(self) -> None:
        self.downloaded_file = ""

    def tearDown(self) -> None:
        _delete_downloaded_files()
        del self._downloaded_file

    @property
    def reference_file(self):
        return f"{_dirname}/reference_data/{self._downloaded_file}"

    @property
    def downloaded_file(self):
        return f"{_dirname}/{self._downloaded_file}"

    @downloaded_file.setter
    def downloaded_file(self, downloaded_file):
        self._downloaded_file = downloaded_file

    def _compare_files_line_by_line(self):
        with open(self.reference_file, 'r') as rf:
            with open(self.downloaded_file, 'r') as df:
                for l1, l2 in zip(rf.readlines(), df.readlines()):
                    self.assertEqual(l1, l2)

    def test_swift_afterglow_flux_data(self):
        redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb='GRB070809', data_mode='flux')
        self.downloaded_file = "GRBData/afterglow/flux/GRB070809_rawSwiftData.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/afterglow/flux/GRB070809.csv"
        self._compare_files_line_by_line()

    def test_swift_xrt_flux_data(self):
        redback.get_data.get_xrt_afterglow_data_from_swift(grb='GRB070809', data_mode='flux')
        self.downloaded_file = "GRBData/afterglow/flux/GRB070809_xrt_rawSwiftData.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/afterglow/flux/GRB070809_xrt.csv"
        self._compare_files_line_by_line()

    def test_swift_afterglow_flux_density_data(self):
        redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb='GRB070809', data_mode='flux_density')
        self.downloaded_file = "GRBData/afterglow/flux_density/GRB070809_rawSwiftData.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/afterglow/flux_density/GRB070809.csv"
        self._compare_files_line_by_line()

    def test_swift_xrt_flux_density_data(self):
        redback.get_data.get_xrt_afterglow_data_from_swift(grb='GRB070809', data_mode='flux_density')
        self.downloaded_file = "GRBData/afterglow/flux_density/GRB070809_xrt_rawSwiftData.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/afterglow/flux_density/GRB070809_xrt.csv"
        self._compare_files_line_by_line()

    def test_swift_prompt_data(self):
        bin_size = "1s"
        redback.get_data.get_prompt_data_from_swift('GRB070809', bin_size=bin_size)
        self.downloaded_file = f"GRBData/prompt/flux/GRB070809_{bin_size}_lc.csv"
        self._compare_files_line_by_line()

    def test_open_catalog_kilonova_data(self):
        redback.get_data.get_open_transient_catalog_data(transient="at2017gfo", transient_type="kilonova")

        self.downloaded_file = f"kilonova/flux_density/at2017gfo.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"kilonova/flux_density/at2017gfo_metadata.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"kilonova/flux_density/at2017gfo_rawdata.csv"
        self._compare_files_line_by_line()

    def test_open_catalog_supernova_data(self):
        redback.get_data.get_open_transient_catalog_data(transient="SN2011kl", transient_type="supernova")

        self.downloaded_file = f"supernova/flux_density/SN2011kl.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"supernova/flux_density/SN2011kl_metadata.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"supernova/flux_density/SN2011kl_rawdata.csv"
        self._compare_files_line_by_line()

    def test_open_catalog_tde_data(self):
        redback.get_data.get_open_transient_catalog_data(transient="PS18kh", transient_type="tidal_disruption_event")

        self.downloaded_file = f"tidal_disruption_event/flux_density/PS18kh.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"tidal_disruption_event/flux_density/PS18kh_metadata.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"tidal_disruption_event/flux_density/PS18kh_rawdata.csv"
        self._compare_files_line_by_line()

    def test_batse_prompt_data(self):
        redback.get_data.get_prompt_data_from_batse(grb="000526")
        self.downloaded_file = "GRBData/prompt/flux/GRB000526_BATSE_lc.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/prompt/flux/tte_bfits_8121.fits.gz"
        self.assertTrue(filecmp.cmp(self.reference_file, self.downloaded_file))
