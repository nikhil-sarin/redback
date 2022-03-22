import filecmp
import os
import shutil
import unittest

import redback

_dirname = os.path.dirname(__file__)


def _delete_downloaded_files():
    for folder in ["GRBData", "kilonova", "supernova", "tidal_disruption_event"]:
        shutil.rmtree(folder, ignore_errors=True)


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

        self.downloaded_file = f"kilonova/at2017gfo.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"kilonova/at2017gfo_metadata.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"kilonova/at2017gfo_rawdata.csv"
        self._compare_files_line_by_line()

    def test_open_catalog_supernova_data(self):
        redback.get_data.get_open_transient_catalog_data(transient="SN2011kl", transient_type="supernova")

        self.downloaded_file = f"supernova/SN2011kl.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"supernova/SN2011kl_metadata.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"supernova/SN2011kl_rawdata.csv"
        self._compare_files_line_by_line()

    def test_open_catalog_tde_data(self):
        redback.get_data.get_open_transient_catalog_data(transient="PS18kh", transient_type="tidal_disruption_event")

        self.downloaded_file = f"tidal_disruption_event/PS18kh.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"tidal_disruption_event/PS18kh_metadata.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = f"tidal_disruption_event/PS18kh_rawdata.csv"
        self._compare_files_line_by_line()

    def test_batse_prompt_data(self):
        redback.get_data.get_prompt_data_from_batse(grb="000526")
        self.downloaded_file = "GRBData/prompt/flux/GRB000526_BATSE_lc.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/prompt/flux/tte_bfits_8121.fits.gz"
        self.assertTrue(filecmp.cmp(self.reference_file, self.downloaded_file))

    def test_lasair_data(self):
        redback.get_data.get_lasair_data(transient="ZTF19aagqkrq", transient_type="afterglow")
        self.downloaded_file = "GRBData/afterglow/ZTF19aagqkrq.csv"
        self._compare_files_line_by_line()

        self.downloaded_file = "GRBData/afterglow/ZTF19aagqkrq_rawdata.json"
        self._compare_files_line_by_line()
