import numpy as np
import os
from pathlib import Path
import pandas as pd

from .transient import Transient
from ..utils import bin_ttes
from ..getdata import prompt_directory_structure, get_prompt_data_from_batse

dirname = os.path.dirname(__file__)


class PromptTimeSeries(Transient):
    DATA_MODES = ['counts', 'tte']

    def __init__(self, name, bin_size, time_tagged_events=None, time=None, counts=None,
                 channel_tags=None, data_mode='tte', trigger_number=None, channels="all", instrument="batse"):
        if data_mode == 'tte':
            time, counts = bin_ttes(time_tagged_events, bin_size)
        super().__init__(time=time, time_err=None, y=counts, y_err=np.sqrt(counts), name=name, data_mode=data_mode)
        self.time_tagged_events = time_tagged_events
        self.channel_tags = channel_tags
        self.bin_size = bin_size
        self.trigger_number = str(trigger_number)
        self.channels = channels
        self.instrument = instrument

        self._set_data()

    @classmethod
    def from_batse_grb_name(cls, name, trigger_number=None, channels="all"):
        time, dt, counts = cls.load_batse_data(name=name, channels=channels)
        return cls(name=name, bin_size=dt, time=time, counts=counts, data_mode="counts",
                   trigger_number=trigger_number, channels=channels, instrument="batse")

    @staticmethod
    def load_batse_data(name, channels):
        grb_dir, _, _ = prompt_directory_structure(grb=name.lstrip("GRB"), use_default_directory=False)
        filename = f"BATSE_lc.csv"
        data_file = os.path.join(grb_dir, filename)
        _time_series_data = np.genfromtxt(data_file, delimiter=",")[1:]

        bin_left = _time_series_data[:, 0]
        bin_right = _time_series_data[:, 1]
        dt = bin_right - bin_left
        time = 0.5 * (bin_left + bin_right)

        counts_by_channel = [np.around(_time_series_data[:, i] * dt) for i in [2, 4, 6, 8]]
        if channels == "all":
            channels = np.array([0, 1, 2, 3])

        counts = np.zeros(len(time))
        for c in channels:
            counts += counts_by_channel[c]

        return time, dt, counts


    @property
    def _stripped_name(self):
        return self.name.lstrip('GRB')

    def plot_data(self):
        pass

    def plot_different_channels(self):
        pass

    @property
    def event_table(self):
        return os.path.join(dirname, f'../tables/BATSE_4B_catalogue.xls')

    def _set_data(self):
        dtypes = dict(trigger_num=np.int32, t90=np.float64, t90_error=np.float64, t90_start=np.float64)
        columns = list(dtypes.keys())
        self.data = pd.read_excel(self.event_table, sheet_name='batsegrb', header=0, usecols=columns, dtype=dtypes)
        self._data_index = self.data.index[self.data['trigger_num'] == int(self.trigger_number)].tolist()[0]

    @property
    def t90(self):
        return self.data['t90'][self._data_index]

    @property
    def t90_error(self):
        return self.data['t90_error'][self._data_index]

    @property
    def t90_start(self):
        return self.data['t90_start'][self._data_index]

    @property
    def t90_end(self):
        return self.t90_start + self.t90
