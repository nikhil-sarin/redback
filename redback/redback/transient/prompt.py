import numpy as np
from os.path import join

from .transient import Transient
from ..utils import bin_ttes
from ..getdata import prompt_directory_structure, get_prompt_data_from_batse




class PromptTimeSeries(Transient):

    DATA_MODES = ['counts', 'tte']

    def __init__(self, name, bin_size, time_tagged_events=None, times=None, counts=None,
                 channel_tags=None, data_mode='tte'):
        if data_mode == 'tte':
            times, counts = bin_ttes(time_tagged_events, bin_size)
        super().__init__(time=times, time_err=None, y=counts, y_err=np.sqrt(counts), name=name, data_mode=data_mode)
        self.time_tagged_events = time_tagged_events
        self.channel_tags = channel_tags
        self._set_data()
        self._get_redshift()
        self.bin_size = bin_size

    def load_data(self):
        grb_dir, _, _ = prompt_directory_structure(grb=self._stripped_name, use_default_directory=False,
                                                   bin_size=f"{self.bin_size}")
        filename = f"{self.name}.csv"
        data_file = join(grb_dir, filename)
        data = np.genfromtxt(data_file, delimiter=",")[1:]
        self.x = data[:, 0]
        self.x_err = data[:, 1:3].T
        self.y, self.y_err = self._load(data)

    @property
    def _stripped_name(self):
        return self.name.lstrip('GRB')

    def _set_data(self):
        pass

    def plot_data(self):
        pass

    def plot_different_channels(self):
        pass


class PromptSpectra(Transient):
    def __init__(self, name, binning, datamode='tte'):
        self.counts = []
        self.name = name

        self._set_data()
        self._get_redshift()

    def _set_data(self):
        pass

    def plot_data(self):
        pass
