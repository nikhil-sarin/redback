import numpy as np

from .transient import Transient


data_mode = ['counts', 'tte']

class PromptTimeSeries(Transient):
    def __init__(self, name, binning, time, time_err, y, data_mode='tte'):
        super().__init__(time=time, time_err=time_err, y=y, y_err=np.sqrt(y), name=name, data_mode=data_mode)
        self.time_tagged_events = []

        self._set_data()
        self._get_redshift()
        self.binning = binning

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
