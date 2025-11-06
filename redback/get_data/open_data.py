import os
import re
import sqlite3
from typing import Union
import urllib
import urllib.request
import ssl

import astropy.units as uu
import numpy as np
import pandas as pd
import requests
from astropy.time import Time

import redback
import redback.get_data.directory
import redback.get_data.utils
from redback.get_data.getter import DataGetter
import redback.redback_errors
from redback.utils import logger, calc_flux_density_from_ABmag, \
    calc_flux_density_error_from_monochromatic_magnitude, bandpass_magnitude_to_flux, bands_to_reference_flux, \
    calc_flux_error_from_magnitude

dirname = os.path.dirname(__file__)


class OpenDataGetter(DataGetter):
    VALID_TRANSIENT_TYPES = ['kilonova', 'supernova', 'tidal_disruption_event']

    DATA_MODE = 'flux_density'

    def __init__(self, transient: str, transient_type: str) -> None:
        """
        Initialize an Open Access Catalog data getter to download photometric data.

        Parameters
        ----------
        transient : str
            Transient name, e.g., 'SN2011kl', 'AT2017gfo', 'PS1-10jh'
        transient_type : str
            Type of the transient. Must be from `redback.get_data.open_data.OpenDataGetter.VALID_TRANSIENT_TYPES`.
            Options are 'kilonova', 'supernova', or 'tidal_disruption_event'

        Examples
        --------
        Get kilonova data for AT2017gfo:

        >>> from redback.get_data.open_data import OpenDataGetter
        >>> getter = OpenDataGetter('AT2017gfo', 'kilonova')
        >>> data = getter.get_data()

        Get supernova data for SN2011kl:

        >>> getter = OpenDataGetter('SN2011kl', 'supernova')
        >>> data = getter.get_data()

        Get tidal disruption event data:

        >>> getter = OpenDataGetter('PS1-10jh', 'tidal_disruption_event')
        >>> data = getter.get_data()
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.open_access_directory_structure(transient=self.transient,
                                                                       transient_type=self.transient_type)

    @property
    def url(self) -> str:
        """
        Get the astrocats API URL for photometric data.

        Returns
        -------
        str
            The astrocats raw data API URL
        """
        return f"https://api.astrocats.space/{self.transient}/photometry/time+magnitude+e_" \
               f"magnitude+band+system?e_magnitude&band&time&format=csv"

    @property
    def metadata_url(self) -> str:
        """
        Get the astrocats API URL for transient metadata.

        Returns
        -------
        str
            The astrocats metadata API URL
        """
        return f"https://api.astrocats.space/{self.transient}/" \
               f"timeofmerger+discoverdate+redshift+ra+dec+host+alias?format=CSV"

    @property
    def metadata_path(self):
        """
        Get the local path to the metadata file.

        Returns
        -------
        str
            The path to the metadata CSV file
        """
        return f"{self.directory_path}{self.transient}_metadata.csv"

    def collect_data(self) -> None:
        """
        Download data from astrocats and save to raw file path.

        Raises
        ------
        ValueError
            If the transient does not exist in the astrocats catalog
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


    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """
        Convert raw astrocats data to processed CSV format.

        Converts magnitudes to flux and flux density, and calculates
        time relative to the event.

        Returns
        -------
        pandas.DataFrame or None
            The processed data with time, magnitude, flux, flux density,
            and associated errors in multiple bands
        """
        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)

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
        data['flux_density_error'] = calc_flux_density_error_from_monochromatic_magnitude(
            magnitude=data['magnitude'].values, magnitude_error=data['e_magnitude'].values, reference_flux=3631,
            magnitude_system='AB')
        data['flux(erg/cm2/s)'] = bandpass_magnitude_to_flux(data['magnitude'].values, data['band'].values)
        data['flux_error'] = calc_flux_error_from_magnitude(magnitude=data['magnitude'].values,
                                        magnitude_error=data['e_magnitude'].values,
                                        reference_flux=bands_to_reference_flux(data['band'].values))
        data['band'] = [b.replace("'", "") for b in data["band"]]
        metadata = pd.read_csv(f"{self.directory_path}{self.transient}_metadata.csv")
        metadata.replace(r'^\s+$', np.nan, regex=True)
        time_of_event = self.get_time_of_event(data=data, metadata=metadata)

        tt = Time(np.asarray(data['time'], dtype=float), format='mjd')
        data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        return data

    def get_time_of_event(self, data: pd.DataFrame, metadata: pd.DataFrame) -> Time:
        """
        Infer the time of the event from the given data and metadata.

        Parameters
        ----------
        data : pandas.DataFrame
            The partially-processed photometric data
        metadata : pandas.DataFrame
            The transient metadata from astrocats

        Returns
        -------
        astropy.time.Time
            The time of the event in MJD format
        """
        time_of_event = metadata['timeofmerger'].iloc[0]
        if np.isnan(time_of_event):
            if self.transient_type == 'kilonova':
                logger.warning('No time_of_event in metadata. Looking through associated GRBs')
                time_of_event = self.get_t0_from_grb()
            else:
                logger.warning('No time of event in metadata.')
                logger.warning('Temporarily using 0.1d before the first data point as a start time')
                time_of_event = data['time'].iloc[0] - 0.1
        return Time(time_of_event, format='mjd')

    def get_t0_from_grb(self) -> float:
        """
        Infer the event time from the GRB catalog.

        Searches the GRB catalog for an associated GRB to find
        the merger/event time.

        Returns
        -------
        float
            The event time in MJD
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
        Get the GRB alias from the Open Access Catalog metadata table.

        Searches the metadata for an associated GRB name.

        Returns
        -------
        str or None
            The GRB alias if found, else None
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
