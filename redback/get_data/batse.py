import os
import urllib
import urllib.request

import astropy.io.fits.hdu
import numpy as np
import pandas as pd
from astropy.io import fits

import redback
from redback.get_data.getter import GRBDataGetter
from redback.get_data.utils import get_batse_trigger_from_grb

_dirname = os.path.dirname(__file__)


class BATSEDataGetter(GRBDataGetter):
    """ """

    VALID_TRANSIENT_TYPES = ["prompt"]

    PROCESSED_FILE_COLUMNS = [
            "Time bin left [s]",
            "Time bin right [s]",
            "flux_20_50 [counts/s]",
            "flux_20_50_err [counts/s]",
            "flux_50_100 [counts/s]",
            "flux_50_100_err [counts/s]",
            "flux_100_300 [counts/s]",
            "flux_100_300_err [counts/s]",
            "flux_greater_300 [counts/s]",
            "flux_greater_300_err [counts/s]"]

    def __init__(self, grb: str) -> None:
        """
        Constructor class for a data getter. The instance will be able to download the specified BATSE data.

        :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        :type grb: str
        """
        super().__init__(grb=grb, transient_type="prompt")
        self.directory_path, self.raw_file_path, self.processed_file_path = self.create_directory_structure()

    @property
    def grb(self) -> str:
        return self.transient

    @grb.setter
    def grb(self, grb: str) -> None:
        self.transient = "GRB" + grb.lstrip('GRB')

    @property
    def trigger(self) -> int:
        """This method infers the BATSE trigger number from the given GRB name."""
        return get_batse_trigger_from_grb(grb=self.grb)

    @property
    def trigger_filled(self) -> str:
        """Trigger number with prepended zeros."""
        return str(self.trigger).zfill(5)

    def create_directory_structure(self) -> tuple:
        """Creates and returns the directory structure."""
        return redback.get_data.directory.batse_prompt_directory_structure(grb=self.grb, trigger=str(self.trigger))

    @property
    def _s(self) -> int:
        """Helper function to figure out the correct subfolder on HEASARC."""
        return self.trigger - self.trigger % 200 + 1

    @property
    def _start(self) -> str:
        return str(self._s).zfill(5)

    @property
    def _stop(self) -> str:
        return str(self._s + 199).zfill(5)

    @property
    def url(self) -> str:
        return f"https://heasarc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/{self._start}_{self._stop}/" \
               f"{self.trigger_filled}_burst/tte_bfits_{self.trigger}.fits.gz"

    def collect_data(self) -> None:
        """Downloads the data from HEASARC and saves it into the raw file path."""
        urllib.request.urlretrieve(self.url, self.raw_file_path)

    def convert_raw_data_to_csv(self) -> pd.DataFrame:
        """Converts the raw data into processed data and saves it into the processed file path.
        The column names are in `BATSEDataGetter.PROCESSED_FILE_COLUMNS`.
        
        :return: The processed data frame.
        :rtype: pd.DataFrame
        """
        with fits.open(self.raw_file_path) as fits_data:
            data = self._get_columns(fits_data=fits_data)
        df = pd.DataFrame(data=data, columns=self.PROCESSED_FILE_COLUMNS)
        df.to_csv(self.processed_file_path, index=False)
        return df

    @staticmethod
    def _get_columns(fits_data: astropy.io.fits.hdu.PrimaryHDU) -> np.ndarray:
        """
        :param fits_data: The fits formatted data to which extract the columns from.
        :type fits_data: astropy.io.fits.hdu.PrimaryHDU

        :return: The columns.
        :rtype: numpy.ndarray
        """
        data = fits_data[-1].data
        bin_left = np.array(data['TIMES'][:, 0])
        bin_right = np.array(data['TIMES'][:, 1])
        rates = np.array(data['RATES'][:, :])
        errors = np.array(data['ERRORS'][:, :])
        # counts = np.array([np.multiply(rates[:, i],
        #                                bin_right - bin_left) for i in range(4)]).T
        # count_err = np.sqrt(counts)
        # t90_st, end = bin_left[0], bin_right[-1]

        return np.array(
            [bin_left, bin_right, rates[:, 0], errors[:, 0], rates[:, 1], errors[:, 1],
             rates[:, 2], errors[:, 2], rates[:, 3], errors[:, 3]]).T

