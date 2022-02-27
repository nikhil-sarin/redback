import os
import re
import sqlite3
from typing import Union
import urllib
import urllib.request
import urllib.request

import astropy.units as uu
import numpy as np
import pandas as pd
import requests
from astropy.time import Time

import redback
import redback.get_data.directory
import redback.get_data.utils
import redback.redback_errors
from redback.utils import logger, calc_flux_density_from_ABmag, calc_flux_density_error

dirname = os.path.dirname(__file__)


class OpenDataGetter(object):
    VALID_TRANSIENT_TYPES = ['kilonova', 'supernova', 'tidal_disruption_event']

    DATA_MODE = 'flux_density'

    def __init__(
            self, transient: str, transient_type: str) -> None:
        """
        Constructor class for a data getter. The instance will be able to downloaded the specified Swift data.

        Parameters
        ----------
        transient: str
            Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        transient_type:
            Type of the transient. Should be 'prompt' or 'afterglow'
        """
        self.transient = transient
        self.transient_type = transient_type
        self.open_transient_dir, self.rawfile, self.fullfile = \
            redback.get_data.directory.transient_directory_structure(
                transient_type=self.transient_type, transient=self.transient, data_mode=self.DATA_MODE)

    def get_data(self) -> None:
        logger.info(f'opening Swift website for {self.transient}')
        self.collect_data()
        self.convert_raw_data_to_csv()
        logger.info(f'Congratulations, you now have a nice data file: {self.fullfile}')

    @property
    def transient_type(self) -> str:
        return self._transient_type

    @transient_type.setter
    def transient_type(self, transient_type: str) -> str:
        if transient_type not in self.VALID_TRANSIENT_TYPES:
            raise ValueError("Transient type does not have open access data.")
        self._transient_type = transient_type

    def collect_data(self) -> None:
        if os.path.isfile(self.rawfile):
            logger.warning('The raw data file already exists')
            return None

        url = f"https://api.astrocats.space/{self.transient}/photometry/time+magnitude+e_" \
              f"magnitude+band+system?e_magnitude&band&time&format=csv"
        response = requests.get(url)

        if 'not found' in response.text:
            raise ValueError(
                f"Transient {self.transient} does not exist in the catalog. "
                f"Are you sure you are using the right alias?")
        else:
            if os.path.isfile(self.fullfile):
                logger.warning('The processed data file already exists')
            else:
                metadata = f"{self.open_transient_dir}{self.transient}_metadata.csv"
                urllib.request.urlretrieve(url, self.rawfile)
                logger.info(f"Retrieved data for {self.transient}")
                metadata_url = f"https://api.astrocats.space/{self.transient}/" \
                               f"timeofmerger+discoverdate+redshift+ra+dec+host+alias?format=CSV"
                urllib.request.urlretrieve(metadata_url, metadata)
                logger.info(f"Metdata for transient {self.transient} added.")

    def convert_raw_data_to_csv(self) -> None:
        if os.path.isfile(self.fullfile):
            logger.warning('The processed data file already exists. Returning.')
            return

        rawdata = pd.read_csv(self.rawfile, sep=',')
        if pd.isna(rawdata['system']).any():
            logger.warning("Some data points do not have system information. Assuming AB magnitude")
            rawdata['system'].fillna('AB', inplace=True)
        logger.info('Processing data for transient {}.'.format(self.transient))

        data = rawdata.copy()
        data = data[data['band'] != 'C']
        data = data[data['band'] != 'W']
        data = data[data['system'] == 'AB']
        logger.info('Keeping only AB magnitude data')
        data['flux_density(mjy)'] = calc_flux_density_from_ABmag(data['magnitude'].values)
        data['flux_density_error'] = calc_flux_density_error(magnitude=data['magnitude'].values,
                                                             magnitude_error=data['e_magnitude'].values,
                                                             reference_flux=3631,
                                                             magnitude_system='AB')
        metadata = pd.read_csv(f"{self.open_transient_dir}{self.transient}_metadata.csv")
        metadata.replace(r'^\s+$', np.nan, regex=True)
        time_of_event = self.get_time_of_event(data=data, metadata=metadata)

        tt = Time(np.asarray(data['time'], dtype=float), format='mjd')
        data['time (days)'] = (tt - time_of_event).to(uu.day)
        data.to_csv(self.fullfile, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.fullfile}')
        return data

    def get_time_of_event(self, data: pd.DataFrame, metadata: pd.DataFrame) -> Time:
        time_of_event = metadata['timeofmerger'].iloc[0]
        if np.isnan(time_of_event):
            if self.transient_type == 'kilonova':
                logger.warning('No time_of_event in metadata. Looking through associated GRBs')
                time_of_event = self.get_t0_from_grb()
            else:
                logger.warning('No time of event in metadata.')
                logger.warning('Temporarily using the first data point as a start time')
                time_of_event = data['time'].iloc[0]
        return Time(time_of_event, format='mjd')

    def get_t0_from_grb(self) -> float:
        grb_alias = self.get_grb_alias()
        catalog = sqlite3.connect('tables/GRBcatalog.sqlite')
        summary_table = pd.read_sql_query("SELECT * from Summary", catalog)
        time_of_event = summary_table[summary_table['GRB_name'] == grb_alias]['mjd'].iloc[0]
        if np.isnan(time_of_event):
            logger.warning('Not found an associated GRB. Temporarily using the first data point as a start time')
        return time_of_event

    def get_grb_alias(self) -> Union[re.Match, None]:
        metadata = pd.read_csv('tables/OAC_metadata.csv')
        transient = metadata[metadata['event'] == self.transient]
        alias = transient['alias'].iloc[0]
        try:
            return re.search('GRB (.+?),', alias).group(1)
        except AttributeError as e:
            logger.warning(e)
            logger.warning("Did not find a valid alias, returning None.")
            return None
