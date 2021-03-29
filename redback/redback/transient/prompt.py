from .transient import Transient


data_mode = ['counts', 'tte']

class PromptTimeSeries(Transient):
    def __init__(self, name, binning, datamode = 'tte'):
        self.time_tagged_events = []
        self.times = []
        self.counts = []

        self.name = name

        self._set_data()
        self._get_redshift()

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
