import os
import re
import sqlite3
from typing import Union
import urllib
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
        transient_type: str
            Type of the transient. Must be from `redback.get_data.open_data.OpenDataGetter.VALID_TRANSIENT_TYPES`.
        """
        self.transient = transient
        self.transient_type = transient_type
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.transient_directory_structure(
                transient_type=self.transient_type, transient=self.transient, data_mode=self.DATA_MODE)

    def get_data(self) -> None:
        """
        Downloads the raw data and produces a processed .csv file.
        """
        logger.info(f'Opening Swift website for {self.transient}.')
        self.collect_data()
        self.convert_raw_data_to_csv()
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')

    @property
    def transient_type(self) -> str:
        """
        Checks if the transient type is valid when setting.

        Returns
        -------
        str: The transient type.
        """
        return self._transient_type

    @transient_type.setter
    def transient_type(self, transient_type: str) -> None:
        if transient_type not in self.VALID_TRANSIENT_TYPES:
            raise ValueError("Transient type does not have open access data.")
        self._transient_type = transient_type

    @property
    def url(self) -> str:
        """

        Returns
        -------
        str: The astrocats raw data url.
        """
        return f"https://api.astrocats.space/{self.transient}/photometry/time+magnitude+e_" \
               f"magnitude+band+system?e_magnitude&band&time&format=csv"

    @property
    def metadata_url(self) -> str:
        """

        Returns
        -------
        str: The astrocats metadata url.
        """
        return f"https://api.astrocats.space/{self.transient}/" \
               f"timeofmerger+discoverdate+redshift+ra+dec+host+alias?format=CSV"

    @property
    def metadata_path(self):
        return f"{self.directory_path}{self.transient}_metadata.csv"

    def collect_data(self) -> None:
        """
        Downloads the data from astrocats and saves it into the raw file path.
        """
        if os.path.isfile(self.raw_file_path):
            logger.warning('The raw data file already exists.')
            return None

        if 'not found' in requests.get(self.url).text:
            raise ValueError(
                f"Transient {self.transient} does not exist in the catalog. "
                f"Are you sure you are using the right alias?")
        urllib.request.urlretrieve(url=self.url, filename=self.raw_file_path)
        logger.info(f"Retrieved data for {self.transient}.")
        urllib.request.urlretrieve(url=self.metadata_url, filename=self.metadata_path)
        logger.info(f"Metadata for {self.transient} added.")

    def convert_raw_data_to_csv(self) -> None:
        """
        Converts the raw data into processed data and saves it into the processed file path.
        The data columns are in `OpenDataGetter.PROCESSED_FILE_COLUMNS`.
        """
        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return

        raw_data = pd.read_csv(self.raw_file_path, sep=',')
        if pd.isna(raw_data['system']).any():
            logger.warning("Some data points do not have system information. Assuming AB magnitude")
            raw_data['system'].fillna('AB', inplace=True)
        logger.info('Processing data for transient {}.'.format(self.transient))

        data = raw_data.copy()
        data = data[data['band'] != 'C']
        data = data[data['band'] != 'W']
        data = data[data['system'] == 'AB']
        logger.info('Keeping only AB magnitude data')
        data['flux_density(mjy)'] = calc_flux_density_from_ABmag(data['magnitude'].values).value
        data['flux_density_error'] = calc_flux_density_error(magnitude=data['magnitude'].values,
                                                             magnitude_error=data['e_magnitude'].values,
                                                             reference_flux=3631,
                                                             magnitude_system='AB')
        metadata = pd.read_csv(f"{self.directory_path}{self.transient}_metadata.csv")
        metadata.replace(r'^\s+$', np.nan, regex=True)
        time_of_event = self.get_time_of_event(data=data, metadata=metadata)

        tt = Time(np.asarray(data['time'], dtype=float), format='mjd')
        data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')

    def get_time_of_event(self, data: pd.DataFrame, metadata: pd.DataFrame) -> Time:
        """
        Infers the time of the event from the given data.

        Parameters
        ----------
        data: pandas.DataFrame
            The half-processed data.
        metadata: pandas.DataFrame
            The metadata.

        Returns
        -------
        astropy.time.Time: The time of the event in the astropy format.

        """
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
        """
        Tries to infer the event time from the GRB catalog.

        Returns
        -------
        float: The event time.
        """
        grb_alias = self.get_grb_alias()
        catalog = sqlite3.connect('tables/GRBcatalog.sqlite')
        summary_table = pd.read_sql_query("SELECT * from Summary", catalog)
        time_of_event = summary_table[summary_table['GRB_name'] == grb_alias]['mjd'].iloc[0]
        if np.isnan(time_of_event):
            logger.warning('Not found an associated GRB. Temporarily using the first data point as a start time')
        return time_of_event

    def get_grb_alias(self) -> Union[re.Match, None]:
        """
        Tries to get the GRB alias from the Open Access Catalog metadata table.

        Returns
        -------
        Union[re.Match, None]: The grb alias if found, else None.
        """
        metadata = pd.read_csv('tables/OAC_metadata.csv')
        transient = metadata[metadata['event'] == self.transient]
        alias = transient['alias'].iloc[0]
        try:
            return re.search('GRB (.+?),', alias).group(1)
        except AttributeError as e:
            logger.warning(e)
            logger.warning("Did not find a valid alias, returning None.")
            return None
