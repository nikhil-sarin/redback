import os
import io
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
    jd_to_mjd, calc_flux_error_from_magnitude

dirname = os.path.dirname(__file__)


class FinkDataGetter(DataGetter):

    VALID_TRANSIENT_TYPES = ["afterglow", "kilonova", "supernova", "tidal_disruption_event", "unknown"]

    def __init__(self, transient: str, transient_type: str) -> None:
        """
        Constructor class for a data getter. The instance will be able to downloaded the specified Swift data.

        :param transient: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        :type transient: str
        :param transient_type: Type of the transient. Must be from
                               `redback.get_data.open_data.FinkDataGetter.VALID_TRANSIENT_TYPES`.
        :type transient_type: str
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.lasair_directory_structure(transient=self.transient,
                                                                  transient_type=self.transient_type)

    @property
    def url(self) -> str:
        """
        :return: The fink raw data url.
        :rtype: str
        """
        return "https://fink-portal.org/api/v1/objects"

    @property
    def objectId(self) -> str:
        """
        :return: The object ID i.e., the transient name
        :rtype: str
        """
        return self.transient

    def collect_data(self) -> None:
        """Downloads the data from astrocats and saves it into the raw file path."""
        if os.path.isfile(self.raw_file_path):
            logger.warning('The raw data file already exists.')
            return None

        response = requests.post(url=self.url,
            json={'objectId': self.objectId, 'output-format': 'csv', 'withupperlim': 'True'})

        data = pd.read_csv(io.BytesIO(response.content))

        if len(data) == 0:
            raise ValueError(
                f"Transient {self.transient} does not exist in the catalog. "
                f"Are you sure you are using the right alias?")

        data.to_csv(self.raw_file_path, index=False)
        logger.info(f"Retrieved data for {self.transient}.")

    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """Converts the raw data into processed data and saves it into the processed file path.
        The data columns are in `OpenDataGetter.PROCESSED_FILE_COLUMNS`.

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)

        raw_data = pd.read_csv(self.raw_file_path)
        raw_data = raw_data[raw_data['d:tag']=='valid']
        fink_to_general_bands = {1: "ztfg", 2: "ztfr", 3:'ztfi'}
        processed_data = pd.DataFrame()

        processed_data["time"] = jd_to_mjd(raw_data["i:jd"].values)
        processed_data["magnitude"] = raw_data['i:magap'].values
        processed_data["e_magnitude"] = raw_data['i:sigmagap'].values
        processed_data['system'] = 'AB'
        bands = [fink_to_general_bands[x] for x in raw_data['i:fid']]
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

        time_of_event = min(processed_data["time"])
        time_of_event = Time(time_of_event, format='mjd')

        tt = Time(np.asarray(processed_data["time"], dtype=float), format='mjd')
        processed_data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        processed_data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        return processed_data
