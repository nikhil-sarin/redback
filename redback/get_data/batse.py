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

    BATSE_COLUMNS = [
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
        self.grb = grb
        self.grb_dir = None
        self.raw_file = None
        self.processed_file = None
        self.create_directory_structure()

    @property
    def grb(self) -> str:
        return self._grb

    @grb.setter
    def grb(self, grb: str) -> None:
        self._grb = "GRB" + grb.lstrip('GRB')

    @property
    def trigger(self) -> int:
        return get_batse_trigger_from_grb(grb=self.grb)

    @property
    def trigger_filled(self) -> str:
        return str(self.trigger).zfill(5)

    def create_directory_structure(self) -> None:
        self.grb_dir, self.raw_file, self.processed_file = \
            redback.get_data.directory.batse_prompt_directory_structure(grb=self.grb, trigger=self.trigger)

    @property
    def _s(self) -> int:
        return self.trigger - self.trigger % 200 + 1

    @property
    def start(self) -> str:
        return str(self._s).zfill(5)

    @property
    def stop(self) -> str:
        return str(self._s + 199).zfill(5)

    @property
    def url(self) -> str:
        return f"https://heasarc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/{self.start}_{self.stop}/" \
               f"{self.trigger_filled}_burst/tte_bfits_{self.trigger}.fits.gz"

    def get_data(self) -> None:
        self.collect_data()
        self.convert_raw_data_to_csv()

    def collect_data(self) -> None:
        urllib.request.urlretrieve(self.url, self.raw_file)

    def convert_raw_data_to_csv(self) -> None:
        with fits.open(self.raw_file) as fits_data:
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
        df = pd.DataFrame(data=data, columns=self.BATSE_COLUMNS)
        df.to_csv(self.processed_file, index=False)
