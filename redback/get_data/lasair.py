import json
import os
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
from redback.get_data.getter import DataGetter
from redback.utils import logger, calc_flux_density_from_ABmag, calc_flux_density_error

dirname = os.path.dirname(__file__)


class LasairDataGetter(DataGetter):
    VALID_TRANSIENT_TYPES = ["afterglow", "kilonova", "supernova", "tidal_disruption_event", "unknown"]

    def __init__(self, transient: str, transient_type: str) -> None:
        """
        Constructor class for a data getter. The instance will be able to downloaded the specified Swift data.

        Parameters
        ----------
        transient: str
            Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        transient_type: str
            Type of the transient. Must be from `redback.get_data.open_data.LasairDataGetter.VALID_TRANSIENT_TYPES`.
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.lasair_directory_structure(transient=self.transient,
                                                                  transient_type=self.transient_type)

    @property
    def url(self) -> str:
        """

        Returns
        -------
        str: The lasair raw data url.
        """
        return f"https://lasair.roe.ac.uk/object/{self.transient}/json/"

    def collect_data(self) -> None:
        """
        Downloads the data from astrocats and saves it into the raw file path.
        """
        if os.path.isfile(self.raw_file_path):
            logger.warning('The raw data file already exists.')
            return None

        if 'does not exist' in requests.get(self.url).text:
            raise ValueError(
                f"Transient {self.transient} does not exist in the catalog. "
                f"Are you sure you are using the right alias?")
        urllib.request.urlretrieve(url=self.url, filename=self.raw_file_path)
        logger.info(f"Retrieved data for {self.transient}.")

    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """
        Converts the raw data into processed data and saves it into the processed file path.
        The data columns are in `OpenDataGetter.PROCESSED_FILE_COLUMNS`.
        """
        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)

        with open(self.raw_file_path, "r") as f:
            raw_data = json.load(f)

        lasair_to_general_bands = {1: "g", 2: "r"}
        processed_data = pd.DataFrame()

        processed_data["time"] = [d["mjd"] for d in raw_data["candidates"] if "candid" in d]
        processed_data["magnitude"] = [d["dc_mag"] for d in raw_data["candidates"] if "candid" in d]
        processed_data["e_magnitude"] = [d["dc_sigmag"] for d in raw_data["candidates"] if "candid" in d]
        processed_data["band"] = [lasair_to_general_bands[d["fid"]] for d in raw_data["candidates"] if "candid" in d]

        processed_data["flux_density(mjy)"] = calc_flux_density_from_ABmag(processed_data["magnitude"].values).value
        processed_data["flux_density_error"] = calc_flux_density_error(
            magnitude=processed_data["magnitude"].values,
            magnitude_error=processed_data["e_magnitude"].values,
            reference_flux=3631,
            magnitude_system="AB")
        processed_data = processed_data.sort_values(by="time")

        time_of_event = min(processed_data["time"])
        time_of_event = Time(time_of_event, format='mjd')

        tt = Time(np.asarray(processed_data["time"], dtype=float), format='mjd')
        processed_data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        processed_data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        return processed_data
