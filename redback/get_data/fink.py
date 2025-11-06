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
        Initialize a Fink data getter to download ZTF photometric data.

        Parameters
        ----------
        transient : str
            ZTF object identifier, e.g., 'ZTF21aaeyldq', 'ZTF18abokyfk'
        transient_type : str
            Type of the transient. Must be from
            `redback.get_data.fink.FinkDataGetter.VALID_TRANSIENT_TYPES`.
            Options are 'afterglow', 'kilonova', 'supernova', 'tidal_disruption_event', or 'unknown'

        Examples
        --------
        Get ZTF data for a kilonova from Fink:

        >>> from redback.get_data.fink import FinkDataGetter
        >>> getter = FinkDataGetter('ZTF21aaeyldq', 'kilonova')
        >>> data = getter.get_data()

        Get ZTF data for a supernova:

        >>> getter = FinkDataGetter('ZTF18abokyfk', 'supernova')
        >>> data = getter.get_data()

        Get data for an unknown transient type:

        >>> getter = FinkDataGetter('ZTF20abcdefg', 'unknown')
        >>> data = getter.get_data()
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.lasair_directory_structure(transient=self.transient,
                                                                  transient_type=self.transient_type)

    @property
    def url(self) -> str:
        """
        Get the Fink API URL.

        Returns
        -------
        str
            The Fink API endpoint URL
        """
        return "https://api.fink-portal.org/api/v1/objects"

    @property
    def objectId(self) -> str:
        """
        Get the object ID.

        Returns
        -------
        str
            The ZTF object ID (transient name)
        """
        return self.transient

    def collect_data(self) -> None:
        """
        Download data from Fink API and save to raw file path.

        Queries the Fink API for the object and retrieves all available
        photometry including upper limits.

        Raises
        ------
        ValueError
            If the transient does not exist in the Fink database
        """
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
        """
        Convert raw Fink data to processed CSV format.

        Converts ZTF aperture magnitudes to flux and flux density,
        and calculates time relative to the first detection. Filters
        for valid detections only.

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

        time_of_event = min(processed_data["time"]) - 0.1
        time_of_event = Time(time_of_event, format='mjd')

        tt = Time(np.asarray(processed_data["time"], dtype=float), format='mjd')
        processed_data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        processed_data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        return processed_data
