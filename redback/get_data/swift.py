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
        """
        Initialize a Swift data getter to download Swift BAT/XRT data.

        Parameters
        ----------
        grb : str
            GRB identifier, e.g., 'GRB140903A' or '140903A' are valid inputs
        transient_type : str
            Type of the transient. Should be 'prompt' or 'afterglow'
        data_mode : str
            Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
            Options are 'flux', 'flux_density', or 'prompt'
        instrument : str, optional
            Instrument(s) to use. Must be from `redback.get_data.swift.SwiftDataGetter.VALID_INSTRUMENTS`.
            Options are 'BAT+XRT' or 'XRT' (default is 'BAT+XRT')
        bin_size : str, optional
            Bin size for prompt data. Must be from `redback.get_data.swift.SwiftDataGetter.SWIFT_PROMPT_BIN_SIZES`.
            Options are '1s', '2ms', '8ms', '16ms', '64ms', '256ms' (default is None)

        Examples
        --------
        Get Swift afterglow flux data for GRB 170817A:

        >>> from redback.get_data.swift import SwiftDataGetter
        >>> getter = SwiftDataGetter('GRB170817A', 'afterglow', 'flux')
        >>> data = getter.get_data()

        Get Swift prompt emission data with 64ms binning:

        >>> getter = SwiftDataGetter('GRB140903A', 'prompt', 'prompt', bin_size='64ms')
        >>> data = getter.get_data()

        Get XRT-only afterglow flux density data:

        >>> getter = SwiftDataGetter('GRB170817A', 'afterglow', 'flux_density', instrument='XRT')
        >>> data = getter.get_data()
        """
        super().__init__(grb=grb, transient_type=transient_type)
        self.grb = grb
        self.instrument = instrument
        self.data_mode = data_mode
        self.bin_size = bin_size
        self.directory_path, self.raw_file_path, self.processed_file_path = self.create_directory_structure()

    @property
    def data_mode(self) -> str:
        """
        Get the data mode.

        Ensures the data mode is from `SwiftDataGetter.VALID_DATA_MODES`.

        Returns
        -------
        str
            The data mode ('flux', 'flux_density', or 'prompt')
        """
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        """
        Set the data mode.

        Parameters
        ----------
        data_mode : str
            The data mode. Must be from VALID_DATA_MODES
        """
        if data_mode not in self.VALID_DATA_MODES:
            raise ValueError("Swift does not have {} data".format(self.data_mode))
        self._data_mode = data_mode

    @property
    def instrument(self) -> str:
        """
        Get the instrument.

        Ensures the instrument is from `SwiftDataGetter.VALID_INSTRUMENTS`.

        Returns
        -------
        str
            The instrument ('BAT+XRT' or 'XRT')
        """
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: str) -> None:
        """
        Set the instrument.

        Parameters
        ----------
        instrument : str
            The instrument. Must be from VALID_INSTRUMENTS
        """
        if instrument not in self.VALID_INSTRUMENTS:
            raise ValueError("Swift does not have {} instrument mode".format(self.instrument))
        self._instrument = instrument

    @property
    def trigger(self) -> str:
        """
        Get the Swift trigger number based on the GRB name.

        Returns
        -------
        str
            The Swift trigger number
        """
        logger.info('Getting trigger number')
        return redback.get_data.utils.get_trigger_number(self.stripped_grb)

    def get_swift_id_from_grb(self) -> str:
        """
        Get the Swift ID from the GRB number.

        Returns
        -------
        str
            The Swift observation ID
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
        Get the Swift GRB data URL.

        Returns
        -------
        str
            The GRB website URL depending on the data mode and instrument
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
        Download the raw Swift data and produce a processed .csv file.

        Returns
        -------
        pandas.DataFrame
            The processed data with time, flux/flux density, and error columns

        Examples
        --------
        >>> from redback.get_data.swift import SwiftDataGetter
        >>> getter = SwiftDataGetter('GRB170817A', 'afterglow', 'flux')
        >>> data = getter.get_data()
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
        Create the directory structure for storing data.

        Returns
        -------
        redback.get_data.directory.DirectoryStructure
            A namedtuple with the directory path, raw file path, and processed file path
        """
        if self.transient_type == 'afterglow':
            return redback.get_data.directory.afterglow_directory_structure(
                    grb=self.grb, data_mode=self.data_mode, instrument=self.instrument)
        elif self.transient_type == 'prompt':
            return redback.get_data.directory.swift_prompt_directory_structure(
                    grb=self.grb, bin_size=self.bin_size)

    def collect_data(self) -> None:
        """
        Download data from the Swift website and save to raw file path.

        Raises
        ------
        redback.redback_errors.WebsiteExist
            If the GRB does not have Swift data available
        """
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
        """
        Download flux density data from the Swift website.

        Uses a headless browser to navigate the Swift burst analyzer website
        and download flux density data. Properly quits the driver after use.
        """
        driver = fetch_driver()
        try:
            driver.get(self.grb_website)
            driver.find_element("xpath", "//select[@name='xrtsub']/option[text()='no']").click()
            time.sleep(20)
            driver.find_element("id","xrt_DENSITY_makeDownload").click()
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
        """
        Download integrated flux data from the Swift website.

        Uses a headless browser to navigate the Swift burst analyzer website
        and download integrated flux data. Properly quits the driver after use.
        """
        driver = fetch_driver()
        try:
            driver.get(self.grb_website)
            # select option for BAT bin_size
            bat_binning = 'batxrtbin'
            if check_element(driver, bat_binning):
                driver.find_element("xpath", "//select[@name='batxrtbin']/option[text()='SNR 4']").click()
            # select option for subplot
            subplot = "batxrtsub"
            if check_element(driver, subplot):
                driver.find_element("xpath","//select[@name='batxrtsub']/option[text()='no']").click()
            # Select option for flux density
            flux_density1 = "batxrtband1"
            flux_density0 = "batxrtband0"
            if (check_element(driver, flux_density1)) and (check_element(driver, flux_density0)):
                driver.find_element("xpath",".//*[@id='batxrtband1']").click()
                driver.find_element("xpath",".//*[@id='batxrtband0']").click()
            # Generate data file
            driver.find_element("xpath",".//*[@id='batxrt_XRTBAND_makeDownload']").click()
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
        """
        Download prompt or XRT data directly without using a headless browser.

        Used for data that can be downloaded directly via URL without
        interactive web page navigation.
        """
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
        """
        Convert raw data to processed CSV format.

        Parses the raw Swift data and converts it into a standardized
        CSV format with proper column names and units.

        Returns
        -------
        pandas.DataFrame or None
            The processed data with standardized columns
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
        """
        Convert raw XRT data to processed CSV format.

        The column names are defined in `SwiftDataGetter.XRT_DATA_KEYS`.

        Returns
        -------
        pandas.DataFrame
            The processed XRT data with columns for time, flux, and errors
        """
        data = np.loadtxt(self.raw_file_path, comments=['!', 'READ', 'NO'])
        data = {key: data[:, i] for i, key in enumerate(self.XRT_DATA_KEYS)}
        data = pd.DataFrame(data)
        data = data[data["Pos. flux err [erg cm^{-2} s^{-1}]"] != 0.]
        data.to_csv(self.processed_file_path, index=False, sep=',')
        return data

    def convert_raw_afterglow_data_to_csv(self) -> pd.DataFrame:
        """
        Convert raw afterglow data to processed CSV format.

        Routes to the appropriate conversion method based on data_mode.

        Returns
        -------
        pandas.DataFrame
            The processed afterglow data
        """
        if self.data_mode == 'flux':
            return self.convert_integrated_flux_data_to_csv()
        if self.data_mode == 'flux_density':
            return self.convert_flux_density_data_to_csv()

    def convert_raw_prompt_data_to_csv(self) -> pd.DataFrame:
        """
        Convert raw prompt emission data to processed CSV format.

        The column names are defined in `SwiftDataGetter.PROMPT_DATA_KEYS`.

        Returns
        -------
        pandas.DataFrame
            The processed prompt emission data with multiple energy bands
        """
        data = np.loadtxt(self.raw_file_path)
        df = pd.DataFrame(data=data, columns=self.PROMPT_DATA_KEYS)
        df.to_csv(self.processed_file_path, index=False, sep=',')
        return df

    def convert_integrated_flux_data_to_csv(self) -> pd.DataFrame:
        """
        Convert integrated flux data to processed CSV format.

        The column names are defined in `SwiftDataGetter.INTEGRATED_FLUX_KEYS`.

        Returns
        -------
        pandas.DataFrame
            The processed integrated flux data from BAT and XRT instruments
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
        """
        Convert flux density data to processed CSV format.

        The column names are defined in `SwiftDataGetter.FLUX_DENSITY_KEYS`.

        Returns
        -------
        pandas.DataFrame
            The processed flux density data in mJy units
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
