"""
Nikhil Sarin
Contains GRB class, with method to load and truncate data for SGRB and in future LGRB
"""
import numpy as np

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
        self.path = path
        self.time = []
        self.time_err = []
        self.Lum50 = []
        self.Lum50_err = []


    def load_and_truncate_data(self, truncate = True):
        """
        Read data of SGRB from given path and GRB telephone number.
        Truncate the data to get rid of all but the last prompt emission point
        make a cut based on the size of the temporal error; ie if t_error < 1s, the data point is
        part of the prompt emission
        """
        data_file = self.path+'/GRB' + self.name + '/GRB' + self.name + '.dat'
        data = np.loadtxt(data_file)

        self.time      = data[:, 0]      ## time (secs)
        self.time_err  = np.abs(data[:, 1:3].T) ## \Delta time (secs)
        self.Lum50 = data[:, 3]        ## Lum (1e50 erg/s)
        self.Lum50_err = np.abs(data[:, 4:].T)

        if truncate:
            truncate = self.time_err[0, :] > 0.1
            to_del = len(self.time) - (len(self.time[truncate]) + 2)
            self.time = self.time[to_del:]
            self.time_err = self.time_err[:, to_del:]
            self.Lum50 = self.Lum50[to_del:]
            self.Lum50_err = self.Lum50_err[:, to_del:]

        return None
