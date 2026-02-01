from __future__ import annotations

import os
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
from redback.utils import logger

try:
    import swifttools.ukssdc.data.GRB as udg
    SWIFTTOOLS_AVAILABLE = True
except ImportError:
    SWIFTTOOLS_AVAILABLE = False
    logger.warning("swifttools not available. Falling back to legacy data retrieval methods.")

dirname = os.path.dirname(__file__)


class SwiftDataGetter(GRBDataGetter):

    VALID_TRANSIENT_TYPES = ["afterglow", "prompt"]
    VALID_DATA_MODES = ['flux', 'flux_density', 'prompt']
    VALID_INSTRUMENTS = ['BAT+XRT', 'XRT']
    VALID_BAT_SNR = ['SNR4', 'SNR5', 'SNR6', 'SNR7']

    XRT_DATA_KEYS = ['Time [s]', "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
                     "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]"]
    INTEGRATED_FLUX_KEYS = ["Time [s]", "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
                            "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]", "Instrument"]
    FLUX_DENSITY_KEYS = ['Time [s]', "Pos. time err [s]", "Neg. time err [s]",
                         'Flux [mJy]', 'Pos. flux err [mJy]', 'Neg. flux err [mJy]', 'Frequency [Hz]']
    PROMPT_DATA_KEYS = ["Time [s]", "flux_15_25 [counts/s/det]", "flux_15_25_err [counts/s/det]",
                        "flux_25_50 [counts/s/det]",
                        "flux_25_50_err [counts/s/det]", "flux_50_100 [counts/s/det]", "flux_50_100_err [counts/s/det]",
                        "flux_100_350 [counts/s/det]", "flux_100_350_err [counts/s/det]", "flux_15_350 [counts/s/det]",
                        "flux_15_350_err [counts/s/det]"]
    SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

    def __init__(
            self, grb: str, transient_type: str, data_mode: str,
            instrument: str = 'BAT+XRT', bin_size: str = None,
            snr: Union[int, str] = 4, force_download: bool = False) -> None:
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
        :param snr: BAT Burst Analyser SNR choice (e.g., 4, 5, 6, 7).
        :type snr: int or str
        :param force_download: If True, re-download data from API even if cached files exist.
        :type force_download: bool
        """
        super().__init__(grb=grb, transient_type=transient_type)
        self.grb = grb
        self.instrument = instrument
        self.data_mode = data_mode
        self.bin_size = bin_size
        self.snr = snr
        self.force_download = force_download
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
    def snr(self) -> str:
        """Returns BAT Burst Analyser SNR selection."""
        return self._snr

    @snr.setter
    def snr(self, snr: Union[int, str]) -> None:
        """Normalizes SNR to 'SNR4' style and validates."""
        snr_str = f"SNR{snr}".upper() if isinstance(snr, (int, float)) else str(snr).upper()
        if not snr_str.startswith("SNR"):
            snr_str = f"SNR{snr_str}"
        if snr_str not in self.VALID_BAT_SNR:
            raise ValueError(f"Unsupported BAT SNR selection: {snr}. Choose from {self.VALID_BAT_SNR}.")
        self._snr = snr_str

    @property
    def trigger(self) -> str:
        """Gets the trigger number based on the GRB name.

        :return: The trigger number.
        :rtype: str
        """
        logger.info('Getting trigger number')
        return redback.get_data.utils.get_trigger_number(self.stripped_grb)

    @property
    def swifttools_grb_name(self) -> str:
        """Formats the GRB name for swifttools (expects 'GRB 060729' style)."""
        if self.grb.startswith("GRB ") or not self.grb.startswith("GRB"):
            return self.grb
        return f"GRB {self.grb[len('GRB'):]}"

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
                    grb=self.grb, data_mode=self.data_mode, instrument=self.instrument, snr=self.snr)
        elif self.transient_type == 'prompt':
            return redback.get_data.directory.swift_prompt_directory_structure(
                    grb=self.grb, bin_size=self.bin_size)

    def collect_data(self) -> None:
        """Downloads the data from the Swift website and saves it into the raw file path."""
        # For prompt emission, continue using direct download (no API available yet)
        if self.transient_type == "prompt":
            if os.path.isfile(self.raw_file_path) and not self.force_download:
                logger.warning('The raw data file already exists. Returning.')
                return
            response = requests.get(self.grb_website)
            if 'No Light curve available' in response.text:
                raise redback.redback_errors.WebsiteExist(
                    f'Problem loading the website for GRB{self.stripped_grb}. '
                    f'Are you sure GRB {self.stripped_grb} has Swift data?')
            self.download_directly()
            return

        # For afterglow data, use swifttools API
        # Check both raw and processed files - if force_download, we need to regenerate both
        if not self.force_download:
            if os.path.isfile(self.processed_file_path):
                logger.warning('The processed data file already exists. Returning.')
                return

        if not SWIFTTOOLS_AVAILABLE:
            raise ImportError(
                "swifttools is required for Swift afterglow data retrieval. "
                "Please install it with: pip install swifttools"
            )
        
        # Check if raw data already exists and can be loaded (unless force_download is True)
        if os.path.isfile(self.raw_file_path) and not self.force_download:
            try:
                logger.info(f'Raw data file exists, loading from {self.raw_file_path}')
                self.load_raw_api_data()
                return
            except Exception as e:
                logger.warning(f'Could not load raw data file: {e}. Re-downloading from API.')
        
        # Download from API if raw data doesn't exist, couldn't be loaded, or force_download is True
        if self.force_download:
            logger.info('Force download requested, re-downloading from API')
        
        # For BAT+XRT mode, get both instruments
        if self.instrument == 'BAT+XRT':
            # Get XRT data from getLightCurves (correct flux units)
            xrt_data = self.download_xrt_data_via_api()
            # Get BAT data from getBurstAnalyser
            ba_data = self.download_burst_analyser_data_via_api()
            self._api_data = {'xrt': xrt_data, 'bat': ba_data}
        elif self.instrument == 'XRT':
            # XRT only: use getLightCurves
            xrt_data = self.download_xrt_data_via_api()
            self._api_data = {'xrt': xrt_data, 'bat': None}
        else:
            # Shouldn't happen but handle gracefully
            xrt_data = self.download_xrt_data_via_api()
            self._api_data = {'xrt': xrt_data, 'bat': None}
        
        # Save raw API data for debugging/reprocessing
        self.save_raw_api_data()


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

    def download_xrt_data_via_api(self) -> pd.DataFrame:
        """Downloads XRT data using the swifttools API.

        :return: The XRT lightcurve data
        :rtype: pandas.DataFrame
        """
        if not SWIFTTOOLS_AVAILABLE:
            raise ImportError("swifttools is required for API-based data retrieval. "
                            "Please install it with: pip install swifttools")

        try:
            logger.info(f'Downloading XRT data for {self.grb} using swifttools API')
            
            # Get the lightcurve data using swifttools
            lc_data = None
            last_error = None
            grb_names = [self.swifttools_grb_name, self.grb, self.stripped_grb]
            for grb_name in grb_names:
                if not grb_name:
                    continue
                try:
                    lc_data = udg.getLightCurves(
                        GRBName=grb_name,
                        incbad="both",
                        nosys="both",
                        saveData=False,
                        returnData=True,
                        silent=True
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f'XRT API failed for GRBName={grb_name}: {e}')

            if lc_data is None:
                raise last_error if last_error else RuntimeError("Failed to retrieve XRT data via API")

            # The API returns datasets at the top level (not nested under GRB name)
            # We want to combine WT and PC modes with priority: PC_nosys_incbad, WT_nosys_incbad
            def select_best_curve(lc_data: dict, mode: str) -> pd.DataFrame | None:
                """Select the best curve for a given mode (PC or WT)."""
                preferred_keys = [
                    f"{mode}_nosys_incbad",
                    f"{mode}_incbad", 
                    f"{mode}_nosys",
                    f"{mode}",
                ]
                for key in preferred_keys:
                    if key in lc_data:
                        df = lc_data[key]
                        if isinstance(df, pd.DataFrame) and len(df) > 0:
                            # Filter out upper limits if present
                            if 'UL' in df.columns and df['UL'].dtype == bool:
                                df = df[~df['UL']]
                            return df
                return None

            # Collect all XRT data (both WT and PC modes)
            all_dfs = []
            
            # Get WT mode data (typically earlier times)
            wt_df = select_best_curve(lc_data, "WT")
            if wt_df is not None:
                all_dfs.append(wt_df)
                logger.info(f'Found {len(wt_df)} WT mode data points')
            
            # Get PC mode data (typically later times)
            pc_df = select_best_curve(lc_data, "PC")
            if pc_df is not None:
                all_dfs.append(pc_df)
                logger.info(f'Found {len(pc_df)} PC mode data points')

            if not all_dfs:
                raise redback.redback_errors.WebsiteExist(
                    f'No suitable XRT lightcurve data found for {self.grb}')

            # Combine all dataframes
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Sort by time
            combined_df = combined_df.sort_values('Time').reset_index(drop=True)
            
            logger.info(f'Successfully downloaded {len(combined_df)} XRT data points for {self.grb}')

            return combined_df

        except Exception as e:
            logger.warning(f'Failed to download XRT data via API for {self.grb}: {e}')
            raise

    def download_burst_analyser_data_via_api(self) -> dict:
        """Downloads BAT+XRT Burst Analyser data using the swifttools API.

        :return: The Burst Analyser data dictionary
        :rtype: dict
        """
        if not SWIFTTOOLS_AVAILABLE:
            raise ImportError("swifttools is required for API-based data retrieval. "
                            "Please install it with: pip install swifttools")

        try:
            logger.info(f'Downloading Burst Analyser data for {self.grb} using swifttools API')

            # Get the Burst Analyser data using swifttools
            ba_data = None
            last_error = None
            grb_names = [self.swifttools_grb_name, self.grb, self.stripped_grb]
            for grb_name in grb_names:
                if not grb_name:
                    continue
                try:
                    ba_data = udg.getBurstAnalyser(
                        GRBName=grb_name,
                        saveData=False,
                        returnData=True,
                        silent=True
                    )
                    if not ba_data or len(ba_data) == 0:
                        raise redback.redback_errors.WebsiteExist(
                            f'No Burst Analyser data available for {self.grb}')
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f'Burst Analyser API failed for GRBName={grb_name}: {e}')

            if ba_data is None:
                raise last_error if last_error else RuntimeError("Failed to retrieve Burst Analyser data via API")

            logger.info(f'Successfully downloaded Burst Analyser data for {self.grb}')

            return ba_data

        except Exception as e:
            logger.warning(f'Failed to download Burst Analyser data via API for {self.grb}: {e}')
            raise

    def save_raw_api_data(self) -> None:
        """Saves the raw API data to the raw file path for debugging and reprocessing."""
        if not hasattr(self, '_api_data') or self._api_data is None:
            # Fallback to marker file
            with open(self.raw_file_path, 'w') as f:
                f.write('# Data retrieved via swifttools API\n')
            return

        try:
            # Create a comprehensive raw data file
            with open(self.raw_file_path, 'w') as f:
                f.write('# Swift data retrieved via swifttools API\n')
                f.write(f'# GRB: {self.grb}\n')
                f.write(f'# Instrument: {self.instrument}\n')
                f.write(f'# Data mode: {self.data_mode}\n')
                f.write('#\n')
                
                if isinstance(self._api_data, dict):
                    # New format with separate XRT and BAT data
                    if 'xrt' in self._api_data and self._api_data['xrt'] is not None:
                        f.write('# XRT DATA (from getLightCurves)\n')
                        f.write('#\n')
                        xrt_df = self._api_data['xrt']
                        xrt_df.to_csv(f, index=False)
                        f.write('\n')
                    
                    if 'bat' in self._api_data and self._api_data['bat'] is not None:
                        f.write('# BAT DATA (from getBurstAnalyser XRTBand)\n')
                        f.write('#\n')
                        ba_data = self._api_data['bat']
                        if 'BAT' in ba_data:
                            # Save the BAT XRTBand data
                            for snr_key in [self.snr, 'SNR4', 'SNR5', 'SNR6', 'SNR7']:
                                if snr_key in ba_data['BAT']:
                                    bat_entry = ba_data['BAT'][snr_key]
                                    if isinstance(bat_entry, dict) and 'XRTBand' in bat_entry:
                                        bat_df = bat_entry['XRTBand']
                                        f.write(f'# BAT {snr_key} XRTBand\n')
                                        bat_df.to_csv(f, index=False)
                                        break
                else:
                    # Legacy format: single dataframe
                    self._api_data.to_csv(f, index=False)
                    
            logger.info(f'Saved raw API data to {self.raw_file_path}')
        except Exception as e:
            logger.warning(f'Could not save raw API data: {e}')
            # Fallback to marker file
            with open(self.raw_file_path, 'w') as f:
                f.write('# Data retrieved via swifttools API\n')

    def load_raw_api_data(self) -> None:
        """Loads previously saved raw API data from the raw file path."""
        if not os.path.isfile(self.raw_file_path):
            raise FileNotFoundError(f'Raw data file not found: {self.raw_file_path}')
        
        logger.info(f'Loading raw API data from {self.raw_file_path}')
        
        # Read the file and parse sections
        with open(self.raw_file_path, 'r') as f:
            lines = f.readlines()
        
        # Check if it's just a marker file
        if len(lines) == 1 and lines[0].strip() == '# Data retrieved via swifttools API':
            raise ValueError('Raw file is just a marker, no actual data saved')
        
        # Find section boundaries
        xrt_start = None
        bat_start = None
        
        for i, line in enumerate(lines):
            if '# XRT DATA' in line:
                xrt_start = i
            elif '# BAT DATA' in line:
                bat_start = i
        
        xrt_data = None
        bat_data = None
        
        # Parse XRT section
        if xrt_start is not None:
            # Find where CSV data starts (skip comment lines after section header)
            csv_start = xrt_start + 1
            while csv_start < len(lines) and (lines[csv_start].startswith('#') or lines[csv_start].strip() == ''):
                csv_start += 1
            
            # Find where CSV data ends (next section or end of file)
            csv_end = bat_start if bat_start is not None else len(lines)
            
            # Extract CSV data
            if csv_start < csv_end:
                from io import StringIO
                csv_content = ''.join(lines[csv_start:csv_end])
                xrt_data = pd.read_csv(StringIO(csv_content))
        
        # Parse BAT section  
        if bat_start is not None:
            # Find where CSV data starts
            csv_start = bat_start + 1
            while csv_start < len(lines) and (lines[csv_start].startswith('#') or lines[csv_start].strip() == ''):
                csv_start += 1
            
            # Extract CSV data (to end of file)
            if csv_start < len(lines):
                from io import StringIO
                csv_content = ''.join(lines[csv_start:])
                bat_df = pd.read_csv(StringIO(csv_content))
                # Create minimal BA structure
                bat_data = {
                    'BAT': {
                        self.snr: {
                            'XRTBand': bat_df
                        }
                    }
                }
        
        # Store in the expected format
        if xrt_data is not None or bat_data is not None:
            self._api_data = {'xrt': xrt_data, 'bat': bat_data}
            xrt_len = len(xrt_data) if xrt_data is not None else 0
            bat_len = len(bat_df) if bat_data is not None else 0
            logger.info(f'Successfully loaded raw data: XRT={xrt_len} points, BAT={bat_len} points')
        else:
            raise ValueError('Could not parse any data from raw file')

    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """Converts the raw data into processed data and saves it into the processed file path.

        :return: The processed data
        :rtype: pandas.DataFrame
        """

        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)

        # Check if we have API data stored
        if hasattr(self, '_api_data') and self._api_data is not None:
            if isinstance(self._api_data, dict) and 'xrt' in self._api_data:
                # New API format with separate XRT and BAT data
                return self.convert_combined_api_data_to_csv()
            else:
                # Legacy: single dataframe (XRT only from old code)
                return self.convert_xrt_api_data_to_csv()

        # Fall back to legacy conversion methods
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

    def convert_xrt_api_data_to_csv(self) -> pd.DataFrame:
        """Converts XRT data from the swifttools API into the expected CSV format.

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        if not hasattr(self, '_api_data') or self._api_data is None:
            raise ValueError("No API data available to convert")

        df = self._api_data

        # The API returns data with column names like 'Time', 'TimePos', 'TimeNeg', 'Flux', 'FluxPos', 'FluxNeg'
        # We need to convert this to match the expected format
        # Expected columns: 'Time [s]', "Pos. time err [s]", "Neg. time err [s]",
        #                   "Flux [erg cm^{-2} s^{-1}]", "Pos. flux err [erg cm^{-2} s^{-1}]",
        #                   "Neg. flux err [erg cm^{-2} s^{-1}]"

        # Create mapping from API column names to expected column names
        column_mapping = {
            'Time': 'Time [s]',
            'TimePos': 'Pos. time err [s]',
            'TimeNeg': 'Neg. time err [s]',
            'Flux': 'Flux [erg cm^{-2} s^{-1}]',
            'FluxPos': 'Pos. flux err [erg cm^{-2} s^{-1}]',
            'FluxNeg': 'Neg. flux err [erg cm^{-2} s^{-1}]',
        }

        # Rename columns if they exist in the dataframe
        data = {}
        for api_col, expected_col in column_mapping.items():
            if api_col in df.columns:
                data[expected_col] = df[api_col].values

        # If we didn't find the expected columns, try alternative names
        if 'Time [s]' not in data:
            if 'T' in df.columns:
                data['Time [s]'] = df['T'].values
            elif 'MET' in df.columns:
                data['Time [s]'] = df['MET'].values

        # Create a new dataframe with the expected format
        processed_df = pd.DataFrame(data)

        # Filter out rows with zero or invalid flux errors (matching legacy behavior)
        if 'Pos. flux err [erg cm^{-2} s^{-1}]' in processed_df.columns:
            processed_df = processed_df[processed_df['Pos. flux err [erg cm^{-2} s^{-1}]'] != 0.]

        processed_df.to_csv(self.processed_file_path, index=False, sep=',')
        logger.info(f'Converted XRT API data to CSV format for {self.grb}')

        return processed_df

    def convert_combined_api_data_to_csv(self) -> pd.DataFrame:
        """Converts combined XRT (from getLightCurves) and BAT (from getBurstAnalyser) data to CSV.

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        if not hasattr(self, '_api_data') or self._api_data is None:
            raise ValueError("No API data available to convert")

        xrt_data = self._api_data.get('xrt')
        bat_ba_data = self._api_data.get('bat')

        # Handle different data modes
        if self.data_mode == 'flux':
            return self._convert_flux_mode(xrt_data, bat_ba_data)
        elif self.data_mode == 'flux_density':
            return self._convert_flux_density_mode(xrt_data, bat_ba_data)
        else:
            raise ValueError(f"Unsupported data mode: {self.data_mode}")

    def _convert_flux_mode(self, xrt_data, bat_ba_data) -> pd.DataFrame:
        """Convert flux mode data (integrated flux in erg/cm²/s)."""
        all_data = []

        # Process XRT data from getLightCurves (already in correct units)
        if xrt_data is not None and len(xrt_data) > 0:
            xrt_mapping = {
                'Time': 'Time [s]',
                'TimePos': 'Pos. time err [s]',
                'TimeNeg': 'Neg. time err [s]',
                'Flux': 'Flux [erg cm^{-2} s^{-1}]',
                'FluxPos': 'Pos. flux err [erg cm^{-2} s^{-1}]',
                'FluxNeg': 'Neg. flux err [erg cm^{-2} s^{-1}]',
            }
            
            xrt_df = xrt_data.rename(columns={k: v for k, v in xrt_mapping.items() if k in xrt_data.columns}).copy()
            required_cols = list(xrt_mapping.values())
            
            if all(col in xrt_df.columns for col in required_cols):
                xrt_df = xrt_df[required_cols].copy()
                xrt_df['Instrument'] = 'XRT'
                all_data.append(xrt_df)
                logger.info(f'Processed {len(xrt_df)} XRT data points')

        # Process BAT data from getBurstAnalyser
        if bat_ba_data is not None and 'BAT' in bat_ba_data:
            # Get the requested SNR level
            snr_keys = [self.snr, 'SNR4', 'SNR5', 'SNR6', 'SNR7']
            bat_df = None
            
            for snr_key in snr_keys:
                if snr_key in bat_ba_data['BAT']:
                    bat_entry = bat_ba_data['BAT'][snr_key]
                    # Use XRTBand which has BAT data in XRT-band flux (erg/cm^2/s)
                    if isinstance(bat_entry, dict) and 'XRTBand' in bat_entry:
                        bat_df = bat_entry['XRTBand']
                        logger.info(f'Using BAT {snr_key} XRTBand data')
                        break
            
            if bat_df is not None and len(bat_df) > 0:
                # Filter out bad bins
                if 'BadBin' in bat_df.columns:
                    bat_df = bat_df[bat_df['BadBin'] == False].copy()
                
                bat_mapping = {
                    'Time': 'Time [s]',
                    'TimePos': 'Pos. time err [s]',
                    'TimeNeg': 'Neg. time err [s]',
                    'Flux': 'Flux [erg cm^{-2} s^{-1}]',
                    'FluxPos': 'Pos. flux err [erg cm^{-2} s^{-1}]',
                    'FluxNeg': 'Neg. flux err [erg cm^{-2} s^{-1}]',
                }
                
                bat_processed = bat_df.rename(columns={k: v for k, v in bat_mapping.items() if k in bat_df.columns}).copy()
                required_cols = list(bat_mapping.values())
                
                if all(col in bat_processed.columns for col in required_cols):
                    # Filter for valid data
                    mask = (
                        np.isfinite(bat_processed['Time [s]']) &
                        np.isfinite(bat_processed['Flux [erg cm^{-2} s^{-1}]']) &
                        (bat_processed['Flux [erg cm^{-2} s^{-1}]'] > 0)
                    )
                    bat_processed = bat_processed[mask].copy()
                    bat_processed = bat_processed[required_cols].copy()
                    bat_processed['Instrument'] = 'BAT'
                    all_data.append(bat_processed)
                    logger.info(f'Processed {len(bat_processed)} BAT data points')

        if not all_data:
            raise ValueError(f"No valid XRT or BAT data found for {self.grb}")

        # Combine and sort by time
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('Time [s]').reset_index(drop=True)

        # Save to CSV
        expected_columns = self.INTEGRATED_FLUX_KEYS
        final_df = combined_df[expected_columns]
        final_df.to_csv(self.processed_file_path, index=False, sep=',')
        logger.info(f'Saved combined data: {len(final_df)} total points for {self.grb}')

        return final_df

    def _convert_flux_density_mode(self, xrt_data, bat_ba_data) -> pd.DataFrame:
        """Convert flux_density mode data (flux density in mJy).
        
        Returns XRT Density datasets (PC and WT modes) only - matches old web scraping behavior.
        """
        if bat_ba_data is None:
            raise ValueError(f"No Burst Analyser data available for flux_density mode for {self.grb}")
        
        all_data = []
        
        # Process XRT Density data ONLY (no BAT for backward compatibility)
        if 'XRT' in bat_ba_data:
            # Try PC mode first
            for key in ['Density_PC_incbad', 'Density_PC']:
                if key in bat_ba_data['XRT']:
                    pc_df = bat_ba_data['XRT'][key]
                    if isinstance(pc_df, pd.DataFrame) and len(pc_df) > 0:
                        pc_df = pc_df.copy()
                        pc_df['Instrument'] = 'XRT'
                        all_data.append(pc_df)
                        logger.info(f'Found {len(pc_df)} XRT PC mode flux density points')
                        break
            
            # Then try WT mode
            for key in ['Density_WT_incbad', 'Density_WT']:
                if key in bat_ba_data['XRT']:
                    wt_df = bat_ba_data['XRT'][key]
                    if isinstance(wt_df, pd.DataFrame) and len(wt_df) > 0:
                        wt_df = wt_df.copy()
                        wt_df['Instrument'] = 'XRT'
                        all_data.append(wt_df)
                        logger.info(f'Found {len(wt_df)} XRT WT mode flux density points')
                        break
        
        if not all_data:
            raise ValueError(f"No flux density data found for {self.grb}")
        
        # Combine datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('Time').reset_index(drop=True)
        
        # Map to expected column names
        column_mapping = {
            'Time': 'Time [s]',
            'TimePos': 'Pos. time err [s]',
            'TimeNeg': 'Neg. time err [s]',
            'Flux': 'Flux [mJy]',
            'FluxPos': 'Pos. flux err [mJy]',
            'FluxNeg': 'Neg. flux err [mJy]'
        }
        
        final_df = combined_df.rename(columns={k: v for k, v in column_mapping.items() if k in combined_df.columns}).copy()
        
        # Add frequency column (Swift flux density is at 10 keV = 2.418e18 Hz)
        final_df['Frequency [Hz]'] = 2.418e18
        
        # Select only the expected columns
        expected_columns = self.FLUX_DENSITY_KEYS
        missing = [col for col in expected_columns if col not in final_df.columns]
        if missing:
            raise ValueError(f"Missing required flux density columns: {missing}")
        
        final_df = final_df[expected_columns]
        final_df.to_csv(self.processed_file_path, index=False, sep=',')
        logger.info(f'Saved flux density data: {len(final_df)} total points for {self.grb}')
        
        return final_df

    def convert_burst_analyser_api_data_to_csv(self) -> pd.DataFrame:
        """Converts Burst Analyser data from the swifttools API into the expected CSV format.

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        if not hasattr(self, '_api_data') or self._api_data is None:
            raise ValueError("No API data available to convert")

        ba_data = self._api_data

        def normalize_ba_flux_dataframe(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame | None:
            """Map Burst Analyser columns to integrated flux columns."""
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return None
            
            mapping = {
                'Time': 'Time [s]',
                'TimePos': 'Pos. time err [s]',
                'TimeNeg': 'Neg. time err [s]',
                'Flux': 'Flux [erg cm^{-2} s^{-1}]',
                'FluxPos': 'Pos. flux err [erg cm^{-2} s^{-1}]',
                'FluxNeg': 'Neg. flux err [erg cm^{-2} s^{-1}]',
            }
            renamed = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}).copy()
            
            required = [
                'Time [s]',
                'Pos. time err [s]',
                'Neg. time err [s]',
                'Flux [erg cm^{-2} s^{-1}]',
                'Pos. flux err [erg cm^{-2} s^{-1}]',
                'Neg. flux err [erg cm^{-2} s^{-1}]',
            ]
            if not all(col in renamed.columns for col in required):
                return None
            
            cleaned = renamed[required].copy()
            
            # Filter bad bins if provided by BA (BadBin column in the source df)
            if 'BadBin' in source_df.columns:
                mask = (source_df['BadBin'] == False) | (source_df['BadBin'] == 0)
                cleaned = cleaned[mask].reset_index(drop=True)
            
            # Require finite, positive flux and finite time
            mask = (
                np.isfinite(cleaned['Time [s]']) &
                np.isfinite(cleaned['Flux [erg cm^{-2} s^{-1}]']) &
                (cleaned['Flux [erg cm^{-2} s^{-1}]'] > 0)
            )
            cleaned = cleaned[mask].reset_index(drop=True)
            
            return cleaned if len(cleaned) > 0 else None

        # For flux mode, we want integrated flux data
        if self.data_mode == 'flux':
            all_data = []

            # Extract XRT data - need to handle both WT and PC modes separately
            if 'XRT' in ba_data:
                # Try PC mode first
                for key in ['ObservedFlux_PC_incbad', 'ObservedFlux_PC']:
                    if key in ba_data['XRT']:
                        pc_df = ba_data['XRT'][key]
                        if isinstance(pc_df, pd.DataFrame) and len(pc_df) > 0:
                            df_norm = normalize_ba_flux_dataframe(pc_df, pc_df)
                            if df_norm is not None:
                                df_norm['Instrument'] = 'XRT'
                                all_data.append(df_norm)
                                logger.info(f'Found {len(df_norm)} XRT PC mode points')
                            break
                
                # Then try WT mode
                for key in ['ObservedFlux_WT_incbad', 'ObservedFlux_WT']:
                    if key in ba_data['XRT']:
                        wt_df = ba_data['XRT'][key]
                        if isinstance(wt_df, pd.DataFrame) and len(wt_df) > 0:
                            df_norm = normalize_ba_flux_dataframe(wt_df, wt_df)
                            if df_norm is not None:
                                df_norm['Instrument'] = 'XRT'
                                all_data.append(df_norm)
                                logger.info(f'Found {len(df_norm)} XRT WT mode points')
                            break

            # Extract BAT data - it's nested as ba_data['BAT'][SNR_key]['ObservedFlux']
            if 'BAT' in ba_data:
                snr_keys = [self.snr, 'SNR4', 'SNR5', 'SNR6', 'SNR7']
                for snr_key in snr_keys:
                    if snr_key in ba_data['BAT']:
                        bat_entry = ba_data['BAT'][snr_key]
                        if isinstance(bat_entry, dict) and 'ObservedFlux' in bat_entry:
                            bat_df = bat_entry['ObservedFlux']
                            if isinstance(bat_df, pd.DataFrame) and len(bat_df) > 0:
                                df_norm = normalize_ba_flux_dataframe(bat_df, bat_df)
                                if df_norm is not None:
                                    df_norm['Instrument'] = 'BAT'
                                    all_data.append(df_norm)
                                    logger.info(f'Found {len(df_norm)} BAT {snr_key} points')
                                break

            if not all_data:
                raise ValueError(
                    f"No suitable Burst Analyser flux data found for {self.grb}. "
                    f"Check that the selected BAT SNR ({self.snr}) has Flux columns."
                )

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by time
            combined_df = combined_df.sort_values('Time [s]').reset_index(drop=True)

            # Select only the expected columns
            expected_columns = self.INTEGRATED_FLUX_KEYS
            final_df = combined_df[expected_columns]

        elif self.data_mode == 'flux_density':
            # For flux density mode, extract the appropriate data
            # Similar logic but for flux density instead of flux
            all_data = []

            if 'XRT' in ba_data:
                df = select_burst_analyser_dataframe(
                    ba_data['XRT'],
                    key_order=[
                        'Density_PC_incbad',
                        'Density_WT_incbad',
                        'Density_PC',
                        'Density_WT',
                    ]
                )
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    all_data.append(df)

            if not all_data:
                raise ValueError(f"No flux density data found for {self.grb}")

            combined_df = all_data[0]

            # Map to expected column names for flux density
            column_mapping = {
                'Time': 'Time [s]',
                'T': 'Time [s]',
                'TimePos': 'Pos. time err [s]',
                'TimeNeg': 'Neg. time err [s]',
                'Flux': 'Flux [mJy]',
                'FluxPos': 'Pos. flux err [mJy]',
                'FluxNeg': 'Neg. flux err [mJy]'
            }

            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in combined_df.columns and new_name not in combined_df.columns:
                    combined_df.rename(columns={old_name: new_name}, inplace=True)

            # Convert flux density units if needed (API might return different units)
            if 'Flux [mJy]' in combined_df.columns:
                # Check if values are very small (indicating they might be in Jy instead of mJy)
                if combined_df['Flux [mJy]'].median() < 0.1:
                    combined_df['Flux [mJy]'] = combined_df['Flux [mJy]'] * 1000
                    combined_df['Pos. flux err [mJy]'] = combined_df['Pos. flux err [mJy]'] * 1000
                    combined_df['Neg. flux err [mJy]'] = combined_df['Neg. flux err [mJy]'] * 1000

            expected_columns = self.FLUX_DENSITY_KEYS
            final_df = combined_df[[col for col in expected_columns if col in combined_df.columns]]

        else:
            raise ValueError(f"Unsupported data mode: {self.data_mode}")

        final_df.to_csv(self.processed_file_path, index=False, sep=',')
        logger.info(f'Converted Burst Analyser API data to CSV format for {self.grb}')

        return final_df
