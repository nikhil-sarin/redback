"""
Nikhil Sarin
Contains GRB class, with method to load and truncate data for SGRB and in future LGRB
"""
import numpy as np
import os
import pandas as pd

from .analysis import find_path
from .getdata import retrieve_and_process_data
from .getdata import get_grb_table
from .utils import find_path

dirname = os.path.dirname(__file__)


class GRB(object):
    """Class for SGRB"""

    def __init__(self, name, path):
        """
        :param name: Telephone number of SGRB, e.g., GRB 140903A
        :param path: Path to the GRB data
        """
        self.name = name
        if path == 'default':
            self.path = find_path(path)
        else:
            self.path = path
        self.time = []
        self.time_err = []
        self.Lum50 = []
        self.Lum50_err = []
        self.luminosity_data = []

        self.__removeables = ["PL", "CPL", ",", "C", "~"]
        self._set_data()
        self._set_photon_index()
        self._set_t90()

    def load_and_truncate_data(self, truncate=True, truncate_method='prompt_time_error', luminosity_data=False):
        """
        Read data of SGRB from given path and GRB telephone number.
        Truncate the data to get rid of all but the last prompt emission point
        make a cut based on the size of the temporal error; ie if t_error < 1s, the data point is
        part of the prompt emission
        """
        self.load_data(luminosity_data=luminosity_data)
        if truncate:
            self.truncate(truncate_method=truncate_method)

    def load_data(self, luminosity_data=False):
        self.luminosity_data = luminosity_data

        label = ''
        if self.luminosity_data:
            label = '_luminosity'

        data_file = f"{self.path}/GRB{self.name}/GRB{self.name}{label}.dat"
        data = np.loadtxt(data_file)
        self.time = data[:, 0]  # time (secs)
        self.time_err = np.abs(data[:, 1:3].T)  # \Delta time (secs)
        self.Lum50 = data[:, 3]  # Lum (1e50 erg/s)
        self.Lum50_err = np.abs(data[:, 4:].T)

    def truncate(self, truncate_method='prompt_time_error'):
        if truncate_method == 'prompt_time_error':
            mask1 = self.time_err[0, :] > 0.0025
            mask2 = self.time < 0.2  # dont truncate if data point is after 0.2 seconds
            mask = np.logical_and(mask1, mask2)
            self.time = self.time[~mask]
            self.time_err = self.time_err[:, ~mask]
            self.Lum50 = self.Lum50[~mask]
            self.Lum50_err = self.Lum50_err[:, ~mask]
        else:
            truncate = self.time_err[0, :] > 0.1
            to_del = len(self.time) - (len(self.time[truncate]) + 2)
            self.time = self.time[to_del:]
            self.time_err = self.time_err[:, to_del:]
            self.Lum50 = self.Lum50[to_del:]
            self.Lum50_err = self.Lum50_err[:, to_del:]

    @property
    def event_table(self):
        return os.path.join(dirname, f'tables/{self.__class__.__name__}_table.txt')

    def get_flux_density(self):
        pass

    def get_integrated_flux(self):
        pass

    def flux_to_luminosity(self):
        pass

    def get_prompt(self):
        pass

    def get_optical(self):
        pass

    def _set_data(self):
        data = pd.read_csv(self.event_table, header=0, error_bad_lines=False, delimiter='\t', dtype='str')
        data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data[
            'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        self.data = data

    def _process_data(self):
        pass

    def _set_photon_index(self):
        photon_index = self.data.query('GRB == @self.name')[
            'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].values[0]
        if photon_index == 0.:
            return 0.
        self.photon_index = self.__clean_string(photon_index)

    def _set_t90(self):
        # data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data['BAT Photon
        # Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        t90 = self.data.query('GRB == @self.name')['BAT T90 [sec]'].values[0]
        if t90 == 0.:
            return np.nan
        self.t90 = self.__clean_string(t90)

    def __clean_string(self, string):
        for r in self.__removeables:
            string.replace(r, "")
        return float(string)

    def process_grbs(self, use_default_directory=False):
        for GRB in self.data['GRB'].values:
            retrieve_and_process_data(GRB, use_default_directory=use_default_directory)

        return print(f'Flux data for all {self.__class__.__name__}s added')

    @staticmethod
    def process_grbs_w_redshift(use_default_directory=False):
        data = pd.read_csv(dirname + '/tables/GRBs_w_redshift.txt', header=0,
                           error_bad_lines=False, delimiter='\t', dtype='str')
        for GRB in data['GRB'].values:
            retrieve_and_process_data(GRB, use_default_directory=use_default_directory)

        return print('Flux data for all GRBs with redshift added')

    @staticmethod
    def process_grb_list(data, use_default_directory=False):
        """
        :param data: a list containing telephone number of GRB needing to process
        :param use_default_directory:
        :return: saves the flux file in the location specified
        """

        for GRB in data:
            retrieve_and_process_data(GRB, use_default_directory=use_default_directory)

        return print('Flux data for all GRBs in list added')


class SGRB(GRB):
    pass


class LGRB(GRB):
    pass
