"""
Nikhil Sarin
Contains GRB class, with method to load and truncate data for SGRB and in future LGRB
"""
import numpy as np
import os
import pandas as pd
from grb_bilby.analysis.Analysis import find_path
dirname = os.path.dirname(__file__)

class SGRB:
    """Class for SGRB"""
    def __init__(self, name, path):
        """
        :param name: Telephone number of SGRB, e.g., GRB 140903A
        :param path: Path to the GRB data
        :param time: time of GRB data (seconds)
        :param time_err: time error in GRB data
        :param Lum50: Luminosity of GRB in 1e50 erg/s
        :param Lum50_err: error in luminsoty data
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
        self.photon_index = self._get_photon_index()
        self.luminosity_data = []
        self.T90 = self._get_T90()

    def load_and_truncate_data(self, truncate = True, truncate_method = 'prompt_time_error', luminosity_data=False):
        """
        Read data of SGRB from given path and GRB telephone number.
        Truncate the data to get rid of all but the last prompt emission point
        make a cut based on the size of the temporal error; ie if t_error < 1s, the data point is
        part of the prompt emission
        """
        if luminosity_data:
            data_file = self.path + '/GRB' + self.name + '/GRB' + self.name + '_luminosity.dat'
            self.luminosity_data=True
        else:
            data_file = self.path+'/GRB' + self.name + '/GRB' + self.name + '.dat'
            self.luminosity_data=False
        data = np.loadtxt(data_file)
        self.time = data[:, 0]      ## time (secs)
        self.time_err = np.abs(data[:, 1:3].T) ## \Delta time (secs)
        self.Lum50 = data[:, 3]        ## Lum (1e50 erg/s)
        self.Lum50_err = np.abs(data[:, 4:].T)


        if truncate:
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

        return None

    def _get_photon_index(self):
        short_table = os.path.join(dirname, 'SGRB_table.txt')
        sgrb = pd.read_csv(short_table, header=0,
                           error_bad_lines=False, delimiter='\t', dtype='str')
        long_table = os.path.join(dirname, 'LGRB_table.txt')
        lgrb = pd.read_csv(long_table, header=0,
                           error_bad_lines=False, delimiter='\t', dtype='str')
        frames = [lgrb, sgrb]
        data = pd.concat(frames, ignore_index=True)
        data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        photon_index = data.query('GRB == @self.name')['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)']
        photon_index = photon_index.values[0]
        if photon_index == 0.:
            return 0.
        else:
            return float(photon_index.replace("PL","").replace("CPL","").replace(",","").replace("C","").replace("~",""))

    def _get_T90(self):
        short_table = os.path.join(dirname, 'SGRB_table.txt')
        sgrb = pd.read_csv(short_table, header=0,
                           error_bad_lines=False, delimiter='\t', dtype='str')
        long_table = os.path.join(dirname, 'LGRB_table.txt')
        lgrb = pd.read_csv(long_table, header=0,
                           error_bad_lines=False, delimiter='\t', dtype='str')
        frames = [lgrb, sgrb]
        data = pd.concat(frames, ignore_index=True)
        #data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        T90 = data.query('GRB == @self.name')['BAT T90 [sec]']
        T90 = T90.values[0]
        if T90 == 0.:
            return np.nan
        else:
            return float(T90.replace("PL","").replace("CPL","").replace(",","").replace("C","").replace("~",""))