import os
import time
import urllib
import urllib.request

from astropy.io import ascii
import numpy as np
import pandas as pd
import requests

from redback.get_data.directory import afterglow_directory_structure, prompt_directory_structure
from redback.get_data.utils import get_trigger_number
from redback.redback_errors import DataExists, WebsiteExist
from redback.utils import fetch_driver, check_element
from redback.utils import logger


dirname = os.path.dirname(__file__)


class SwiftDataGetter(object):

    VALID_DATA_MODES = {'flux', 'flux_density'}

    def __init__(self, grb, transient_type, data_mode, bin_size=None):
        self.grb = grb
        self.transient_type = transient_type
        self.instrument = 'BAT+XRT'
        self.data_mode = data_mode
        self.bin_size = bin_size
        self.grbdir = None
        self.rawfile = None
        self.fullfile = None
        self.create_directory_structure()

    @property
    def data_mode(self):
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode):
        if self.data_mode not in self.VALID_DATA_MODES:
            raise ValueError("Swift does not have {} data".format(self.data_mode))
        self._data_mode = data_mode

    @property
    def trigger(self):
        logger.info('Getting trigger number')
        return get_trigger_number(self.grb)

    def get_swift_id_from_grb(self):
        data = ascii.read(f'{dirname}/tables/summary_general_swift_bat.txt')
        triggers = list(data['col2'])
        event_names = list(data['col1'])
        swift_id = triggers[event_names.index(self.grb)]
        if len(swift_id) == 6:
            swift_id += "000"
            swift_id = swift_id.zfill(11)
        return swift_id

    @property
    def grb_website(self):
        if self.transient_type == 'prompt':
            return f"https://swift.gsfc.nasa.gov/results/batgrbcat/{self.grb}/data_product/" \
                   f"{self.get_swift_id_from_grb()}-results/lc/{self.bin_size}_lc_ascii.dat"
        if self.instrument == 'BAT+XRT':
            return f'http://www.swift.ac.uk/burst_analyser/00{self.trigger}/'
        elif self.instrument == 'xrt':
            return f'https://www.swift.ac.uk/xrt_curves/00{self.trigger}/flux.qdp'

    def get_data(self):
        self.create_directory_structure()
        logger.info(f'opening Swift website for GRB {self.grb}')
        self.collect_data()
        self.convert_raw_data_to_csv()
        logger.info(f'Congratulations, you now have a nice data file: {self.fullfile}')

    def create_directory_structure(self):
        if self.transient_type == 'afterglow':
            self.grbdir, self.rawfile, self.fullfile = \
                afterglow_directory_structure(grb=self.grb, data_mode=self.data_mode, instrument=self.instrument)
        elif self.transient_type == 'prompt':
            self.grbdir, self.rawfile, self.fullfile = \
                prompt_directory_structure(grb=self.grb, bin_size=self.bin_size)

    def collect_data(self):
        if os.path.isfile(self.rawfile):
            logger.warning('The raw data file already exists. Returning.')

        response = requests.get(self.grb_website)
        if 'No Light curve available' in response.text:
            raise WebsiteExist(f'Problem loading the website for GRB {self.grb}' 
                               f'Are you sure GRB {self.grb} has Swift data?')
        if self.instrument == 'xrt' or self.transient_type == "prompt":
            self.download_directly()
        elif self.transient_type == 'afterglow':
            if self.data_mode == 'flux':
                self.download_integrated_flux_data()
            elif self.data_mode == 'flux_density':
                self.download_flux_density_data()
        elif self.transient_type == 'prompt':
            self.download_prompt_data()

    def download_flux_density_data(self):
        driver = fetch_driver()
        try:
            driver.get(self.grb_website)
            driver.find_element_by_xpath("//select[@name='xrtsub']/option[text()='no']").click()
            time.sleep(20)
            driver.find_element_by_id("xrt_DENSITY_makeDownload").click()
            time.sleep(20)
            grb_url = driver.current_url
            # scrape the data
            urllib.request.urlretrieve(grb_url, self.rawfile)
            logger.info(f'Congratulations, you now have raw data for GRB {self.grb}')
        except Exception as e:
            logger.warning(f'Cannot load the website for GRB {self.grb} \n'
                           f'Failed with exception: \n'
                           f'{e}')
        finally:
            # Close the driver and all opened windows
            driver.quit()
            urllib.request.urlcleanup()

    def download_integrated_flux_data(self):
        driver = fetch_driver()
        try:
            driver.get(self.grb_website)
            # celect option for BAT bin_size
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
            urllib.request.urlretrieve(grb_url, self.rawfile)
            logger.info(f'Congratulations, you now have raw data for GRB {self.grb}')
        except Exception as e:
            logger.warning(f'Cannot load the website for GRB {self.grb} \n'
                           f'Failed with exception: \n'
                           f'{e}')
        finally:
            # Close the driver and all opened windows
            driver.quit()
            urllib.request.urlcleanup()

    def download_directly(self):
        try:
            urllib.request.urlretrieve(self.grb_website, self.rawfile)
            logger.info(f'Congratulations, you now have raw {self.instrument} {self.transient_type} '
                        f'data for GRB {self.grb}')
        except Exception as e:
            logger.warning(f'Cannot load the website for GRB {self.grb} \n'
                           f'Failed with exception: \n'
                           f'{e}')
        finally:
            urllib.request.urlcleanup()

    def convert_raw_data_to_csv(self):
        if os.path.isfile(self.fullfile):
            logger.warning('The processed data file already exists. Returning.')
        if self.instrument == 'xrt':
            self.convert_xrt_data_to_csv()
        if self.transient_type == 'afterglow':
            self.convert_raw_afterglow_data_to_csv()
        elif self.transient_type == 'prompt':
            self.convert_raw_prompt_data_to_csv()

    def convert_xrt_data_to_csv(self):
        keys = ['Time [s]', "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
                "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]"]

        data = np.loadtxt(self.rawfile, comments=['!', 'READ', 'NO'])
        data = {key: data[:, i] for i, key in enumerate(keys)}
        data = pd.DataFrame(data)
        data = data[data["Pos. flux err [erg cm^{-2} s^{-1}]"] != 0.]
        data.to_csv(self.fullfile, index=False, sep=',')

    def convert_raw_afterglow_data_to_csv(self):
        if self.data_mode == 'flux':
            self.convert_integrated_flux_data_to_csv()
        if self.data_mode == 'flux_density':
            self.convert_flux_density_data_to_csv()

    def convert_raw_prompt_data_to_csv(self):
        keys = ["Time [s]", "flux_15_25 [counts/s/det]", "flux_15_25_err [counts/s/det]", "flux_25_50 [counts/s/det]",
                "flux_25_50_err [counts/s/det]", "flux_50_100 [counts/s/det]", "flux_50_100_err [counts/s/det]",
                "flux_100_350 [counts/s/det]", "flux_100_350_err [counts/s/det]", "flux_15_350 [counts/s/det]",
                "flux_15_350_err [counts/s/det]"]
        data = np.loadtxt(self.rawfile)
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(self.fullfile, index=False, sep=',')

    def convert_integrated_flux_data_to_csv(self):
        keys = ["Time [s]", "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
                "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]", "Instrument"]

        data = {key: [] for key in keys}
        with open(self.rawfile) as f:
            instrument = None
            for num, line in enumerate(f.readlines()):
                if line == "!\n":
                    continue
                elif "READ TERR 1 2" in line or "NO NO NO NO NO NO" in line:
                    continue
                elif 'batSNR4flux' in line:
                    instrument = 'batSNR4flux'
                elif 'xrtpcflux' in line:
                    instrument = 'xrtpcflux'
                elif 'xrtwtflux' in line:
                    instrument = 'xrtwtflux'
                else:
                    line_items = line.split('\t')
                    line_items.append(instrument)
                    for key, item in zip(keys, line_items):
                        data[key].append(item.replace('\n', ''))

        df = pd.DataFrame(data=data)
        df.to_csv(self.fullfile, index=False, sep=',')

    def convert_flux_density_data_to_csv(self):
        data = np.loadtxt(self.rawfile, skiprows=2, delimiter='\t')
        df = pd.DataFrame(data=data, columns=['Time [s]', "Pos. time err [s]", "Neg. time err [s]",
                                              'Flux [mJy]', 'Pos. flux err [mJy]', 'Neg. flux err [mJy]'])
        df.to_csv(self.fullfile, index=False, sep=',')
