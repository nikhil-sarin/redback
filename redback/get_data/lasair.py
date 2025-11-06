import os
from typing import Union

import astropy.units as uu
import numpy as np
import pandas as pd
import requests
from astropy.time import Time

import redback
import redback.get_data.directory
import redback.get_data.utils
import redback.redback_errors
from redback.get_data.getter import DataGetter
from redback.utils import logger, calc_flux_density_from_ABmag, \
    calc_flux_density_error_from_monochromatic_magnitude, bandpass_magnitude_to_flux, bands_to_reference_flux, \
    calc_flux_error_from_magnitude

dirname = os.path.dirname(__file__)


class LasairDataGetter(DataGetter):

    VALID_TRANSIENT_TYPES = ["afterglow", "kilonova", "supernova", "tidal_disruption_event", "unknown"]

    def __init__(self, transient: str, transient_type: str) -> None:
        """
        Initialize a Lasair data getter to download ZTF photometric data.

        Parameters
        ----------
        transient : str
            ZTF object identifier, e.g., 'ZTF21aaeyldq', 'ZTF18abokyfk'
        transient_type : str
            Type of the transient. Must be from
            `redback.get_data.lasair.LasairDataGetter.VALID_TRANSIENT_TYPES`.
            Options are 'afterglow', 'kilonova', 'supernova', 'tidal_disruption_event', or 'unknown'

        Examples
        --------
        Get ZTF data for a kilonova from Lasair:

        >>> from redback.get_data.lasair import LasairDataGetter
        >>> getter = LasairDataGetter('ZTF21aaeyldq', 'kilonova')
        >>> data = getter.get_data()

        Get ZTF data for a supernova:

        >>> getter = LasairDataGetter('ZTF18abokyfk', 'supernova')
        >>> data = getter.get_data()

        Get data for an unknown transient type:

        >>> getter = LasairDataGetter('ZTF20abcdefg', 'unknown')
        >>> data = getter.get_data()
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.lasair_directory_structure(transient=self.transient,
                                                                  transient_type=self.transient_type)

    @property
    def url(self) -> str:
        """
        Get the Lasair URL for the transient.

        Returns
        -------
        str
            The Lasair transient page URL
        """
        return f"https://lasair-ztf.lsst.ac.uk/objects/{self.transient}"

    def collect_data(self) -> None:
        """
        Download data from Lasair website and save to raw file path.

        Scrapes the HTML table from the Lasair object page and extracts
        difference magnitude photometry.

        Raises
        ------
        ValueError
            If the transient does not exist in the Lasair database
        """
        if os.path.isfile(self.raw_file_path):
            logger.warning('The raw data file already exists.')
            return None

        if 'not in database' in requests.get(self.url).text:
            raise ValueError(
                f"Transient {self.transient} does not exist in the catalog. "
                f"Are you sure you are using the right alias?")
        data = pd.read_html(self.url)
        data = data[1]
        data['diff_magnitude'] = [data['unforced mag'].iloc[x].split(" ")[0] for x in range(len(data))]
        data['diff_magnitude_error'] = [data['unforced mag'].iloc[x].split(" ")[-1] for x in range(len(data))]

        logger.warning('Using the difference magnitude to calculate quantities. '
                       'Reduce the data yourself if you would like to use a reference magnitude')

        # Change the dataframe to the correct raw dataframe format
        del data['UTC']
        del data['images']
        data.to_csv(self.raw_file_path, index=False)
        logger.info(f"Retrieved data for {self.transient}.")

    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """
        Convert raw Lasair data to processed CSV format.

        Converts ZTF difference magnitudes to flux and flux density,
        and calculates time relative to the first detection.

        Returns
        -------
        pandas.DataFrame or None
            The processed data with time, magnitude, flux, flux density,
            and associated errors in ZTF bands (g, r, i)
        """
        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)

        raw_data = pd.read_csv(self.raw_file_path)
        raw_data = raw_data[raw_data['unforced mag status'] != 'limit']
        lasair_to_general_bands = {"g": "ztfg", "r": "ztfr", "i":'ztfi'}
        processed_data = pd.DataFrame()

        processed_data["time"] = raw_data['MJD']
        processed_data["magnitude"] = raw_data['diff_magnitude'].values
        processed_data["e_magnitude"] = raw_data['diff_magnitude_error'].values
        processed_data['system'] = 'AB'
        bands = [lasair_to_general_bands[x] for x in raw_data['Filter']]
        processed_data["band"] = bands

        processed_data["flux_density(mjy)"] = calc_flux_density_from_ABmag(processed_data["magnitude"].values).value
        processed_data["flux_density_error"] = calc_flux_density_error_from_monochromatic_magnitude(
            magnitude=processed_data["magnitude"].values, magnitude_error=processed_data["e_magnitude"].values,
            reference_flux=3631, magnitude_system="AB")
        processed_data['flux(erg/cm2/s)'] = bandpass_magnitude_to_flux(processed_data['magnitude'].values, processed_data['band'].values)
        processed_data['flux_error'] = calc_flux_error_from_magnitude(magnitude=processed_data['magnitude'].values,
                                        magnitude_error=processed_data['e_magnitude'].values,
                                        reference_flux=bands_to_reference_flux(processed_data['band'].values))
        processed_data = processed_data.sort_values(by="time")

        time_of_event = min(processed_data["time"]) - 0.1
        time_of_event = Time(time_of_event, format='mjd')

        tt = Time(np.asarray(processed_data["time"], dtype=float), format='mjd')
        processed_data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        processed_data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        return processed_data
