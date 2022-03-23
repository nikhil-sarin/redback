from __future__ import annotations

import os
import time
from typing import Union
import urllib
import urllib.request

import astropy.io.ascii
import numpy as np
import pandas as pd
import requests

import redback.get_data.directory
import redback.get_data.utils
import redback.redback_errors
from redback.get_data.getter import GRBDataGetter
from redback.utils import fetch_driver, check_element
from redback.utils import logger

dirname = os.path.dirname(__file__)


class SwiftDataGetter(GRBDataGetter):

    VALID_TRANSIENT_TYPES = ["afterglow", "prompt"]
    VALID_DATA_MODES = ['flux', 'flux_density', 'prompt']
    VALID_INSTRUMENTS = ['BAT+XRT', 'XRT']

    XRT_DATA_KEYS = ['Time [s]', "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
                     "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]"]
    INTEGRATED_FLUX_KEYS = ["Time [s]", "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
                            "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]", "Instrument"]
    FLUX_DENSITY_KEYS = ['Time [s]', "Pos. time err [s]", "Neg. time err [s]",
                         'Flux [mJy]', 'Pos. flux err [mJy]', 'Neg. flux err [mJy]']
    PROMPT_DATA_KEYS = ["Time [s]", "flux_15_25 [counts/s/det]", "flux_15_25_err [counts/s/det]",
                        "flux_25_50 [counts/s/det]",
                        "flux_25_50_err [counts/s/det]", "flux_50_100 [counts/s/det]", "flux_50_100_err [counts/s/det]",
                        "flux_100_350 [counts/s/det]", "flux_100_350_err [counts/s/det]", "flux_15_350 [counts/s/det]",
                        "flux_15_350_err [counts/s/det]"]
    SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

    def __init__(
            self, grb: str, transient_type: str, data_mode: str,
            instrument: str = 'BAT+XRT', bin_size: str = None) -> None:
        """Constructor class for a data getter. The instance will be able to download the specified Swift data.

        :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        :type grb: str
        :param transient_type: Type of the transient. Should be 'prompt' or 'afterglow'.
        :type transient_type: str
        :param data_mode: Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
        :type data_mode: str
        :param instrument: Instrument(s) to use.
                           Must be from `redback.get_data.swift.SwiftDataGetter.VALID_INSTRUMENTS`.
        :type instrument: str
        :param bin_size: Bin size. Must be from `redback.get_data.swift.SwiftDataGetter.SWIFT_PROMPT_BIN_SIZES`.
        :type bin_size: str
        """
        super().__init__(grb=grb, transient_type=transient_type)
        self.grb = grb
        self.instrument = instrument
        self.data_mode = data_mode
        self.bin_size = bin_size
        self.directory_path, self.raw_file_path, self.processed_file_path = self.create_directory_structure()

    @property
    def data_mode(self) -> str:
        """Ensures the data mode to be from `SwiftDataGetter.VALID_DATA_MODES`.

        :return: The data mode
        :rtype: str
        """
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        """
        :param data_mode: The data mode.
        :type data_mode: str
        """
        if data_mode not in self.VALID_DATA_MODES:
            raise ValueError("Swift does not have {} data".format(self.data_mode))
        self._data_mode = data_mode

    @property
    def instrument(self) -> str:
        """
        Ensures the data mode to be from `SwiftDataGetter.VALID_INSTRUMENTS`.

        :return: The instrument
        :rtype: str
        """
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: str) -> None:
        """
        :param instrument: The instrument
        :type: str
        """
        if instrument not in self.VALID_INSTRUMENTS:
            raise ValueError("Swift does not have {} instrument mode".format(self.instrument))
        self._instrument = instrument

    @property
    def trigger(self) -> str:
        """Gets the trigger number based on the GRB name.

        :return: The trigger number.
        :rtype: str
        """
        logger.info('Getting trigger number')
        return redback.get_data.utils.get_trigger_number(self.stripped_grb)

    def get_swift_id_from_grb(self) -> str:
        """
        Gets the Swift ID from the GRB number.

        :return: The Swift ID
        :rtype: str
        """
        data = astropy.io.ascii.read(f'{dirname.rstrip("get_data/")}/tables/summary_general_swift_bat.txt')
        triggers = list(data['col2'])
        event_names = list(data['col1'])
        swift_id = triggers[event_names.index(self.grb)]
        if len(swift_id) == 6:
            swift_id += "000"
            swift_id = swift_id.zfill(11)
        return swift_id

    @property
    def grb_website(self) -> str:
        """
        :return: The GRB website depending on the data mode and instrument.
        :rtype: str
        """
        if self.transient_type == 'prompt':
            return f"https://swift.gsfc.nasa.gov/results/batgrbcat/{self.grb}/data_product/" \
                   f"{self.get_swift_id_from_grb()}-results/lc/{self.bin_size}_lc_ascii.dat"
        if self.instrument == 'BAT+XRT':
            return f'http://www.swift.ac.uk/burst_analyser/00{self.trigger}/'
        elif self.instrument == 'XRT':
            return f'https://www.swift.ac.uk/xrt_curves/00{self.trigger}/flux.qdp'

    def get_data(self) -> pd.DataFrame:
        """
        Downloads the raw data and produces a processed .csv file.

        :return: The processed data
        :rtype: pandas.DataFrame
        """
        if self.instrument == "BAT+XRT":
            logger.warning(
                "You are downloading BAT and XRT data, "
                "you will need to truncate the data for some models.")
        elif self.instrument == "XRT":
            logger.warning(
                "You are only downloading XRT data, you may not capture"
                " the tail of the prompt emission.")
        return super(SwiftDataGetter, self).get_data()

    def create_directory_structure(self) -> redback.get_data.directory.DirectoryStructure:
        """
        :return: A namedtuple with the directory path, raw file path, and processed file path.
        :rtype: redback.get_data.directory.DirectoyStructure
        """
        if self.transient_type == 'afterglow':
            return redback.get_data.directory.afterglow_directory_structure(
                    grb=self.grb, data_mode=self.data_mode, instrument=self.instrument)
        elif self.transient_type == 'prompt':
            return redback.get_data.directory.swift_prompt_directory_structure(
                    grb=self.grb, bin_size=self.bin_size)

    def collect_data(self) -> None:
        """Downloads the data from the Swift website and saves it into the raw file path."""
        if os.path.isfile(self.raw_file_path):
            logger.warning('The raw data file already exists. Returning.')
            return

        response = requests.get(self.grb_website)
        if 'No Light curve available' in response.text:
            raise redback.redback_errors.WebsiteExist(
                f'Problem loading the website for GRB{self.stripped_grb}. '
                f'Are you sure GRB {self.stripped_grb} has Swift data?')
        if self.instrument == 'XRT' or self.transient_type == "prompt":
            self.download_directly()
        elif self.transient_type == 'afterglow':
            if self.data_mode == 'flux':
                self.download_integrated_flux_data()
            elif self.data_mode == 'flux_density':
                self.download_flux_density_data()

    def download_flux_density_data(self) -> None:
        """Downloads flux density data from the Swift website.
        Uses the PhantomJS headless browser to click through the website.
        Properly quits the driver.
        """
        driver = fetch_driver()
        try:
            driver.get(self.grb_website)
            driver.find_element_by_xpath("//select[@name='xrtsub']/option[text()='no']").click()
            time.sleep(20)
            driver.find_element_by_id("xrt_DENSITY_makeDownload").click()
            time.sleep(20)
            grb_url = driver.current_url
            # scrape the data
            urllib.request.urlretrieve(url=grb_url, filename=self.raw_file_path)
            logger.info(f'Congratulations, you now have raw data for {self.grb}')
        except Exception as e:
            logger.warning(f'Cannot load the website for {self.grb} \n'
                           f'Failed with exception: \n'
                           f'{e}')
        finally:
            # Close the driver and all opened windows
            driver.quit()
            urllib.request.urlcleanup()

    def download_integrated_flux_data(self) -> None:
        """Downloads integrated flux density data from the Swift website.
        Uses the PhantomJS headless browser to click through the website.
        Properly quits the driver.
        """
        driver = fetch_driver()
        try:
            driver.get(self.grb_website)
            # select option for BAT bin_size
            bat_binning = 'batxrtbin'
            if check_element(driver, bat_binning):
                driver.find_element_by_xpath("//select[@name='batxrtbin']/option[text()='SNR 4']").click()
            # select option for subplot
            subplot = "batxrtsub"
            if check_element(driver, subplot):
                driver.find_element_by_xpath("//select[@name='batxrtsub']/option[text()='no']").click()
            # Select option for flux density
            flux_density1 = "batxrtband1"
            flux_density0 = "batxrtband0"
            if (check_element(driver, flux_density1)) and (check_element(driver, flux_density0)):
                driver.find_element_by_xpath(".//*[@id='batxrtband1']").click()
                driver.find_element_by_xpath(".//*[@id='batxrtband0']").click()
            # Generate data file
            driver.find_element_by_xpath(".//*[@id='batxrt_XRTBAND_makeDownload']").click()
            time.sleep(20)
            grb_url = driver.current_url
            driver.quit()
            urllib.request.urlretrieve(grb_url, self.raw_file_path)
            logger.info(f'Congratulations, you now have raw data for {self.grb}')
        except Exception as e:
            logger.warning(f'Cannot load the website for {self.grb} \n'
                           f'Failed with exception: \n'
                           f'{e}')
        finally:
            # Close the driver and all opened windows
            driver.quit()
            urllib.request.urlcleanup()

    def download_directly(self) -> None:
        """Downloads prompt or XRT data directly without using PhantomJS if possible."""
        try:
            urllib.request.urlretrieve(self.grb_website, self.raw_file_path)
            logger.info(f'Congratulations, you now have raw {self.instrument} {self.transient_type} '
                        f'data for {self.grb}')
        except Exception as e:
            logger.warning(f'Cannot load the website for {self.grb} \n'
                           f'Failed with exception: \n'
                           f'{e}')
        finally:
            urllib.request.urlcleanup()

    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """Converts the raw data into processed data and saves it into the processed file path.

        :return: The processed data
        :rtype: pandas.DataFrame
        """

        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)
        if self.instrument == 'XRT':
            return self.convert_xrt_data_to_csv()
        elif self.transient_type == 'afterglow':
            return self.convert_raw_afterglow_data_to_csv()
        elif self.transient_type == 'prompt':
            return self.convert_raw_prompt_data_to_csv()

    def convert_xrt_data_to_csv(self) -> pd.DataFrame:
        """Converts the raw XRT data into processed data and saves it into the processed file path.
        The column names are in `SwiftDataGetter.XRT_DATA_KEYS`

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        data = np.loadtxt(self.raw_file_path, comments=['!', 'READ', 'NO'])
        data = {key: data[:, i] for i, key in enumerate(self.XRT_DATA_KEYS)}
        data = pd.DataFrame(data)
        data = data[data["Pos. flux err [erg cm^{-2} s^{-1}]"] != 0.]
        data.to_csv(self.processed_file_path, index=False, sep=',')
        return data

    def convert_raw_afterglow_data_to_csv(self) -> pd.DataFrame:
        """Converts the raw afterglow data into processed data and saves it into the processed file path.

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        if self.data_mode == 'flux':
            return self.convert_integrated_flux_data_to_csv()
        if self.data_mode == 'flux_density':
            return self.convert_flux_density_data_to_csv()

    def convert_raw_prompt_data_to_csv(self) -> pd.DataFrame:
        """Converts the raw prompt data into processed data and saves it into the processed file path.
        The column names are in `SwiftDataGetter.PROMPT_DATA_KEYS`

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        data = np.loadtxt(self.raw_file_path)
        df = pd.DataFrame(data=data, columns=self.PROMPT_DATA_KEYS)
        df.to_csv(self.processed_file_path, index=False, sep=',')
        return df

    def convert_integrated_flux_data_to_csv(self) -> pd.DataFrame:
        """Converts the flux data into processed data and saves it into the processed file path.
        The column names are in `SwiftDataGetter.INTEGRATED_FLUX_KEYS`

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        data = {key: [] for key in self.INTEGRATED_FLUX_KEYS}
        with open(self.raw_file_path) as f:
            started = False
            for num, line in enumerate(f.readlines()):
                if line.startswith('NO NO NO'):
                    started = True
                if not started:
                    continue
                if line.startswith('!'):
                    instrument = line[2:].replace('\n', '')
                if line[0].isnumeric() or line[0] == '-':
                    line_items = line.split('\t')
                    data['Instrument'] = instrument
                    for key, item in zip(self.INTEGRATED_FLUX_KEYS, line_items):
                        data[key].append(item.replace('\n', ''))
        df = pd.DataFrame(data=data)
        df.to_csv(self.processed_file_path, index=False, sep=',')
        return df

    def convert_flux_density_data_to_csv(self) -> pd.DataFrame:
        """Converts the flux data into processed data and saves it into the processed file path.
        The column names are in `SwiftDataGetter.FLUX_DENSITY_KEYS`

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        data = {key: [] for key in self.FLUX_DENSITY_KEYS}
        with open(self.raw_file_path) as f:
            started = False
            for num, line in enumerate(f.readlines()):
                if line.startswith('NO NO NO'):
                    started = True
                if not started:
                    continue
                if line[0].isnumeric() or line[0] == '-':
                    line_items = line.split('\t')
                    for key, item in zip(self.FLUX_DENSITY_KEYS, line_items):
                        data[key].append(item.replace('\n', ''))
        data['Flux [mJy]'] = [float(x) * 1000 for x in data['Flux [mJy]']]
        data['Pos. flux err [mJy]'] = [float(x) * 1000 for x in data['Pos. flux err [mJy]']]
        data['Neg. flux err [mJy]'] = [float(x) * 1000 for x in data['Neg. flux err [mJy]']]
        df = pd.DataFrame(data=data)
        df.to_csv(self.processed_file_path, index=False, sep=',')
        return df
