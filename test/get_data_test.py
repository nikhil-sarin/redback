import mock
import unittest
from unittest.mock import MagicMock
import redback
import os
import shutil


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


class TestSwiftDataGetter(unittest.TestCase):

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
        os.remove("GRBData")
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

    def set_valid_instrument(self):
        for valid_instrument in ['BAT+XRT', 'XRT']:
            self.getter.instrument = valid_instrument
            self.assertEqual(valid_instrument, self.getter.instrument)

    def text_set_invalid_instrument(self):
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
        swift_id_expected = swift_id_stump + "000"
        swift_id_expected = swift_id_expected.zfill(11)
        ascii_read.return_value = dict(col1=[self.grb], col2=[swift_id_stump])
        swift_id_actual = self.getter.get_swift_id_from_grb()
        self.assertEqual(swift_id_expected, swift_id_actual)
        ascii_read.assert_called_once()

    @mock.patch("redback.get_data.utils.get_trigger_number")
    def test_grb_website_bat_xrt(self, get_trigger_number):
        expected_trigger = "0"
        get_trigger_number.return_value = expected_trigger
        expected = f'http://www.swift.ac.uk/burst_analyser/00{expected_trigger}/'
        self.assertEqual(expected, self.getter.grb_website)

    @mock.patch("redback.get_data.utils.get_trigger_number")
    def test_grb_website_xrt(self, get_trigger_number):
        self.getter.instrument = 'xrt'
        expected_trigger = "0"
        get_trigger_number.return_value = expected_trigger
        expected = f'https://www.swift.ac.uk/xrt_curves/00{expected_trigger}/flux.qdp'
        self.assertEqual(expected, self.getter.grb_website)

    @mock.patch("astropy.io.ascii.read")
    def test_grb_website_prompt(self, ascii_read):
        swift_id_stump = "123456"
        self.getter.transient_type = "prompt"
        ascii_read.return_value = dict(col1=[self.grb], col2=[swift_id_stump])
        expected = f"https://swift.gsfc.nasa.gov/results/batgrbcat/{self.grb}/data_product/" \
                   f"{self.getter.get_swift_id_from_grb()}-results/lc/{self.bin_size}_lc_ascii.dat"
        self.assertEqual(expected, self.getter.grb_website)

    @mock.patch("redback.get_data.directory.afterglow_directory_structure")
    def test_create_directory_structure_afterglow(self, afterglow_directory_structure):
        expected = "0", "1", "2"
        afterglow_directory_structure.return_value = expected
        self.getter.create_directory_structure()
        self.assertListEqual(list(expected), list([self.getter.grbdir, self.getter.rawfile, self.getter.fullfile]))
        afterglow_directory_structure.assert_called_with(grb=self.grb, data_mode=self.data_mode, instrument=self.instrument)

    @mock.patch("redback.get_data.directory.prompt_directory_structure")
    def test_create_directory_structure_prompt(self, prompt_directory_structure):
        self.getter.transient_type = "prompt"
        expected = "0", "1", "2"
        prompt_directory_structure.return_value = expected
        self.getter.create_directory_structure()
        self.assertListEqual(list(expected), list([self.getter.grbdir, self.getter.rawfile, self.getter.fullfile]))
        prompt_directory_structure.assert_called_with(grb=self.grb, bin_size=self.bin_size)

    def test_get_data(self):
        self.getter.create_directory_structure = MagicMock()
        self.getter.collect_data = MagicMock()
        self.getter.convert_raw_data_to_csv = MagicMock()
        self.getter.get_data()
        self.getter.create_directory_structure.assert_called_once()
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
    def test_collect_data_no_lightcurve_available(self, isfile, get):
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
        self.getter.data_mode = 'prompt'
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

    # @mock.patch("redback.get_data.utils.fetch_driver")
    # @mock.patch("urllib.request.urlretrieve")
    # @mock.patch("urllib.request.urlcleanup")
    # def test_download_flux_density_data_mocked(self, fetch_driver, urlretrieve, urlcleanup):
    #     mock_driver = MagicMock()
    #     mock_driver.get = MagicMock()
    #     mock_driver.find_element_by_xpath = MagicMock()
    #     mock_driver.find_element_by_xpath.click = MagicMock()
    #     mock_driver.find_element_by_id = MagicMock()
    #     mock_driver.find_element_by_id.click = MagicMock()
    #     mock_driver.quit = MagicMock()
    #     mock_url = 'mock_url'
    #     mock_driver.__setattr__('current_url', mock_url)
    #     fetch_driver.return_value = MagicMock()
    #
    #     self.getter.data_mode = 'flux_density'
    #     self.getter.transient_type = 'afterglow'
    #     self.getter.instrument = "BAT+XRT"
    #     self.getter.download_flux_density_data()
    #
    #     mock_driver.get.assert_called_once_with(self.getter.grb_website)
    #     mock_driver.find_element_by_xpath.assert_called_once_with("//select[@name='xrtsub']/option[text()='no']")
    #     mock_driver.find_element_by_xpath.click.assert_called_once()
    #     mock_driver.find_element_by_id.assert_called_once_with("xrt_DENSITY_makeDownload")
    #     mock_driver.find_element_by_id.click.assert_called_once()
    #     urlretrieve.assert_called_once_with(url=mock_url, filename=self.getter.rawfile)
    #     urlcleanup.assert_called_once()

    def _download_afterglow(self):
        self.getter.transient_type = 'afterglow'
        self.getter.instrument = "BAT+XRT"
        self.getter.create_directory_structure()
        self.getter.collect_data()

    def _test_raw_afterglow(self):
        with open(f"reference_data/afterglow/{self.getter.data_mode}/GRB{self.grb}_rawSwiftData.csv") as file:
            reference_data = file.read()
        with open(self.getter.rawfile) as file:
            downloaded_data = file.read()
        self.assertEqual(reference_data, downloaded_data)

    def _test_converted_afterglow(self):
        with open(f"reference_data/afterglow/{self.getter.data_mode}/GRB{self.grb}.csv") as file:
            reference_data = file.read()
        with open(self.getter.fullfile) as file:
            downloaded_data = file.read()
        self.assertEqual(reference_data, downloaded_data)

    def test_download_flux_density_data(self):
        self.getter.data_mode = 'flux_density'
        self._download_afterglow()
        self._test_raw_afterglow()

    def test_download_flux_data(self):
        self.getter.data_mode = 'flux'
        self._download_afterglow()
        self._test_raw_afterglow()

    def test_convert_flux_density_data_to_csv(self):
        self.getter.data_mode = 'flux_density'
        self._download_afterglow()
        self.getter.convert_integrated_flux_data_to_csv()
        self._test_converted_afterglow()

    def test_convert_integrated_flux_data_to_csv(self):
        self.getter.data_mode = 'flux'
        self._download_afterglow()
        self.getter.convert_integrated_flux_data_to_csv()
        self._test_converted_afterglow()

    def _download_xrt(self):
        self.getter.instrument = 'XRT'
        self.getter.create_directory_structure()
        self.getter.download_directly()

    def test_download_xrt_data(self):
        self._download_xrt()
        with open(f"reference_data/afterglow/{self.getter.data_mode}/GRB{self.grb}_xrt_rawSwiftData.csv") as file:
            reference_data = file.read()
        with open(self.getter.rawfile) as file:
            downloaded_data = file.read()
        self.assertEqual(reference_data, downloaded_data)

    def test_convert_xrt_data_to_csv(self):
        self._download_xrt()
        self.getter.convert_xrt_data_to_csv()
        with open(f"reference_data/afterglow/{self.getter.data_mode}/GRB{self.grb}_xrt.csv") as file:
            reference_data = file.read()
        with open(self.getter.fullfile) as file:
            downloaded_data = file.read()
        self.assertEqual(reference_data, downloaded_data)

    def _download_prompt(self):
        self.data_mode = 'prompt'
        self.getter.bin_size = '2ms'
        self.getter.data_mode = 'counts'
        self.getter.create_directory_structure()
        self.getter.download_directly()

    def test_download_prompt(self):
        self._download_prompt()
        with open(f"reference_data/prompt/{self.getter.data_mode}/GRB{self.grb}_{self.bin_size}_lc_ascii.dat") as file:
            reference_data = file.read()
        with open(self.getter.rawfile) as file:
            downloaded_data = file.read()
        self.assertEqual(reference_data, downloaded_data)

    def test_convert_raw_prompt_data_to_csv(self):
        self._download_prompt()
        self.getter.convert_raw_prompt_data_to_csv()
        with open(f"reference_data/prompt/{self.getter.data_mode}/GRB{self.grb}_{self.bin_size}_lc.csv") as file:
            reference_data = file.read()
        with open(self.getter.fullfile) as file:
            downloaded_data = file.read()
        self.assertEqual(reference_data, downloaded_data)

    def _mock_converter_functions(self):
        self.getter.convert_xrt_data_to_csv = MagicMock()
        self.getter.convert_raw_afterglow_data_to_csv = MagicMock()
        self.getter.convert_raw_prompt_data_to_csv = MagicMock()

    def test_convert_raw_data_to_csv_file_exists(self):
        self._mock_converter_functions()
        with open(self.getter.fullfile, "w"):  # create empty file
            pass
        self.getter.convert_raw_data_to_csv()

        self.getter.convert_xrt_data_to_csv.assert_not_called()
        self.getter.convert_raw_afterglow_data_to_csv.assert_not_called()
        self.getter.convert_raw_prompt_data_to_csv.assert_not_called()

    def test_convert_raw_data_to_csv_instrument_xrt(self):
        self._mock_converter_functions()
        self.getter.convert_xrt_data_to_csv.assert_called_once()
        self.getter.convert_raw_afterglow_data_to_csv.assert_not_called()
        self.getter.convert_raw_prompt_data_to_csv.assert_not_called()

    def test_convert_raw_data_to_csv_afterglow_data(self):
        self._mock_converter_functions()
        self.getter.convert_xrt_data_to_csv.assert_not_called()
        self.getter.convert_raw_afterglow_data_to_csv.assert_called_once()
        self.getter.convert_raw_prompt_data_to_csv.assert_not_called()

    def test_convert_raw_data_to_csv_prompt_data(self):
        self._mock_converter_functions()
        self.getter.convert_xrt_data_to_csv.assert_not_called()
        self.getter.convert_raw_afterglow_data_to_csv.assert_not_called()
        self.getter.convert_raw_prompt_data_to_csv.assert_called_once()


class TestGRBReferenceFiles(unittest.TestCase):

    def setUp(self) -> None:
        self.downloaded_file = ""

    def tearDown(self) -> None:
        shutil.rmtree('GRBData')
        del self.downloaded_file

    def _compare_files(self):
        with open(self.reference_file, 'r') as rf:
            with open(self.downloaded_file, 'r') as df:
                for l1, l2 in zip(rf.readlines(), df.readlines()):
                    self.assertEqual(l1, l2)

    @property
    def reference_file(self):
        return f"reference_data/{self.downloaded_file}"

    def test_swift_afterglow_flux_data(self):
        # Raw File
        self.downloaded_file = "GRBData/afterglow/flux/GRB070809_rawSwiftData.csv"
        redback.get_data.get_afterglow_data_from_swift(grb='GRB070809', data_mode='flux')
        self._compare_files()

        # Processed File
        self.downloaded_file = "GRBData/afterglow/flux/GRB070809.csv"
        redback.get_data.get_afterglow_data_from_swift(grb='GRB070809', data_mode='flux')
        self._compare_files()

    def test_swift_xrt_flux_data(self):
        # Raw File
        self.downloaded_file = "GRBData/afterglow/flux/GRB070809_xrt_rawSwiftData.csv"
        redback.get_data.get_xrt_data_from_swift(grb='GRB070809', data_mode='flux')
        self._compare_files()

        # Processed File
        self.downloaded_file = "GRBData/afterglow/flux/GRB070809_xrt.csv"
        redback.get_data.get_xrt_data_from_swift(grb='GRB070809', data_mode='flux')
        self._compare_files()

    def test_swift_prompt_data(self):
        bin_size = "1s"
        self.downloaded_file = f"GRBData/prompt/flux/GRB070809_{bin_size}_lc.csv"
        redback.get_data.get_prompt_data_from_swift('GRB070809', bin_size=bin_size)
        self._compare_files()

    # def test_raw_swift_afterglow_flux_data(self):
    #     reference_file = "reference_data/GRBData/GRB070809/afterglow/flux/GRB070809_rawSwiftData.csv"
    #     downloaded_file = "GRBData/GRB070809/afterglow/flux/GRB070809_rawSwiftData.csv"
    #     redback.get_data.get_afterglow_data_from_swift(grb='GRB070809', data_mode='flux')
    #     with open(reference_file, 'r') as rf:
    #         with open(downloaded_file, 'r') as df:
    #             for l1, l2 in zip(rf.readlines(), df.readlines()):
    #                 self.assertEqual(l1, l2)

# class TestGetGRBTable(unittest.TestCase):
#
#     def setUp(self) -> None:
#         pass
#
#     def tearDown(self) -> None:
#         pass
#
#
# class TestCheckElement(unittest.TestCase):
#
#     def setUp(self) -> None:
#         pass
#
#     def tearDown(self) -> None:
#         pass
#
#
# class TestGetGRBFile(unittest.TestCase):
#
#     def setUp(self) -> None:
#         pass
#
#     def tearDown(self) -> None:
#         pass
#
#     def test_get_grb_file(self):
#         grb_file = getdata.get_grb_file(grb="GRB041223")
#         self.assertIsNotNone(grb_file)
#
#
# class TestSortData(unittest.TestCase):
#
#     def setUp(self) -> None:
#         pass
#
#     def tearDown(self) -> None:
#         pass
#
#
# class TestRetrieveAndProcessData(unittest.TestCase):
#
#     def setUp(self) -> None:
#         pass
#
#     def tearDown(self) -> None:
#         pass
