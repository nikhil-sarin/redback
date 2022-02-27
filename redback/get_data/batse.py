import os
import urllib
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits

import redback
from redback.get_data.utils import get_batse_trigger_from_grb

_dirname = os.path.dirname(__file__)


class BATSEDataGetter(object):

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

        Parameters
        ----------
        grb: str
            Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        """
        self.grb = grb
        self.directory_path, self.raw_file_path, self.processed_file_path = self.create_directory_structure()

    @property
    def grb(self) -> str:
        """

        Returns
        -------
        str: The GRB number with prepended 'GRB'.
        """
        return self._grb

    @grb.setter
    def grb(self, grb: str) -> None:
        self._grb = "GRB" + grb.lstrip('GRB')

    @property
    def trigger(self) -> int:
        """
        This method infers the BATSE trigger number from the given GRB name.

        Returns
        -------
        int: The trigger number.

        """
        return get_batse_trigger_from_grb(grb=self.grb)

    @property
    def trigger_filled(self) -> str:
        """

        Returns
        -------
        str: The trigger number with prepended zeros so it has 5 characters in total.

        """
        return str(self.trigger).zfill(5)

    def create_directory_structure(self) -> tuple:
        """
        Creates and returns the directory structure.

        Returns
        -------
        tuple: Named tuple with the directory path, raw file path, and processed file path.
        """
        return redback.get_data.directory.batse_prompt_directory_structure(grb=self.grb, trigger=self.trigger)

    @property
    def _s(self) -> int:
        """
        Helper function to figure out the correct subfolder on HEASARC.

        Returns
        -------
        int: A helpful number that I have obtained in a mysterious way.
        """
        return self.trigger - self.trigger % 200 + 1

    @property
    def _start(self) -> str:
        """

        Returns
        -------
        str: First part of the correct subfolder on HEASARC.
        """
        return str(self._s).zfill(5)

    @property
    def _stop(self) -> str:
        """

        Returns
        -------
        str: Second part of the correct subfolder on HEASARC.
        """
        return str(self._s + 199).zfill(5)

    @property
    def url(self) -> str:
        """

        Returns
        -------
        str: The url to the raw data file on HEASARC.
        """
        return f"https://heasarc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/{self._start}_{self._stop}/" \
               f"{self.trigger_filled}_burst/tte_bfits_{self.trigger}.fits.gz"

    def get_data(self) -> None:
        """
        Downloads the raw data and produces a processed .csv file.
        """
        self.collect_data()
        self.convert_raw_data_to_csv()

    def collect_data(self) -> None:
        """
        Downloads the data from HEASARC and saves it into the raw file path.
        """
        urllib.request.urlretrieve(self.url, self.raw_file_path)

    def convert_raw_data_to_csv(self) -> None:
        """
        Converts the raw data into processed data and saves it into the processed file path.
        The column names are in `BATSEDataGetter.PROCESSED_FILE_COLUMNS`.
        """
        with fits.open(self.raw_file_path) as fits_data:
            data = fits_data[-1].data
            bin_left = np.array(data['TIMES'][:, 0])
            bin_right = np.array(data['TIMES'][:, 1])
            rates = np.array(data['RATES'][:, :])
            errors = np.array(data['ERRORS'][:, :])
            # counts = np.array([np.multiply(rates[:, i],
            #                                bin_right - bin_left) for i in range(4)]).T
            # count_err = np.sqrt(counts)
            # t90_st, end = bin_left[0], bin_right[-1]

        data = np.array([bin_left, bin_right, rates[:, 0], errors[:, 0], rates[:, 1], errors[:, 1],
                         rates[:, 2], errors[:, 2], rates[:, 3], errors[:, 3]]).T
        df = pd.DataFrame(data=data, columns=self.PROCESSED_FILE_COLUMNS)
        df.to_csv(self.processed_file_path, index=False)
