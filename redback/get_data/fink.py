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

    def __init__(self, transient: str, transient_type: str, source='ztf') -> None:
        """
        Constructor class for a data getter. The instance will be able to downloaded the specified Swift data.

        :param transient: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        :type transient: str
        :param transient_type: Type of the transient. Must be from
                               `redback.get_data.open_data.FinkDataGetter.VALID_TRANSIENT_TYPES`.
        :param source: The source of the data. Must be either 'ztf' or 'lsst'. Default is 'ztf'.
        :type transient_type: str
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.lasair_directory_structure(transient=self.transient,
                                                                  transient_type=self.transient_type)
        self.source = source

    @property
    def url(self) -> str:
        """
        :return: The fink raw data url.
        :rtype: str
        """
        if self.source == 'ztf':
            url = "https://api.ztf.fink-portal.org/api/v1/objects"
        elif self.source == 'lsst':
            url = "https://api.lsst.fink-portal.org//api/v1/sources"
        else:
            raise ValueError(f"Invalid source {self.source}. Valid sources are 'ztf' and 'lsst'.")
        return url

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

        logger.info(f"Collecting data: source={self.source}, transient={self.transient}")
        
        if self.source == 'ztf':
            response = requests.post(url=self.url,
                json={'objectId': self.objectId, 'output-format': 'csv', 'withupperlim': 'True'})
            data = pd.read_csv(io.BytesIO(response.content))
        elif self.source == 'lsst':
            logger.info(f"Fetching LSST data for {self.transient} from {self.url}")
            response = requests.post(url=self.url,
                json={'diaObjectId': self.objectId, 'output-format': 'csv', 'withupperlim': 'True'})
            logger.info(f"Got response: status={response.status_code}, content_length={len(response.content)}")
            data = pd.read_csv(io.BytesIO(response.content))
            logger.info(f"Parsed DataFrame: shape={data.shape}, len={len(data)}")

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
        processed_data = pd.DataFrame()
        if self.source == 'ztf':
            raw_data = raw_data[raw_data['d:tag']=='valid']
            fink_to_ztf_bands = {1: "ztfg", 2: "ztfr", 3:'ztfi'}
            processed_data["time"] = jd_to_mjd(raw_data["i:jd"].values)
            processed_data["magnitude"] = raw_data['i:magap'].values
            processed_data["e_magnitude"] = raw_data['i:sigmagap'].values
            processed_data['system'] = 'AB'
            processed_data['band'] = [fink_to_ztf_bands[x] for x in raw_data['i:fid']]
        elif self.source == 'lsst':
            fink_to_lsst_bands = {'u':"lsstu", 'g':"lsstg", 'r':"lsstr", 'i':"lssti", 'z':"lsstz", 'y':"lssty"}
            processed_data['time'] = raw_data['r:midpointMjdTai']

            # Convert flux to magnitude: m = -2.5*log10(F) + zeropoint
            # LSST uses nanomaggies (31.4 zeropoint)
            processed_data['magnitude'] = 31.4 - 2.5 * np.log10(raw_data['r:scienceFlux'].values)
            processed_data['e_magnitude'] = (2.5 * raw_data['r:scienceFluxErr'])/(raw_data['r:scienceFlux'] * np.log(10))
            processed_data['system'] = 'AB'
            processed_data['band'] = [fink_to_lsst_bands[x] for x in raw_data['r:band']]

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
