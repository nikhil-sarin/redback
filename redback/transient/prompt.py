from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Union

from redback.getdata import prompt_directory_structure, get_batse_trigger_from_grb
from redback.transient.transient import Transient

dirname = os.path.dirname(__file__)


class PromptTimeSeries(Transient):
    DATA_MODES = ['counts', 'ttes']

    def __init__(
            self, name: str, bin_size: float = 1, ttes: np.ndarray = None, time: np.ndarray = None, 
            time_err: np.ndarray = None, time_rest_frame: np.ndarray = None, time_rest_frame_err: np.ndarray = None, 
            counts: np.ndarray = None, channel_tags: np.ndarray = None, data_mode: str = 'ttes',
            trigger_number: str = None, channels: Union[np.ndarray, str] = "all", instrument: str = "batse", 
            **kwargs: dict) -> None:
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame,
                         time_rest_frame_err=time_rest_frame_err, counts=counts, ttes=ttes, bin_size=bin_size,
                         name=name, data_mode=data_mode, **kwargs)
        self.channel_tags = channel_tags
        self.trigger_number = trigger_number
        self.channels = channels
        self.instrument = instrument
        self._set_data()

    @classmethod
    def from_batse_grb_name(
            cls, name: str, trigger_number: str = None, channels: Union[np.ndarray, str] = "all") -> PromptTimeSeries:
        time, dt, counts = cls.load_batse_data(name=name, channels=channels)
        return cls(name=name, bin_size=dt, time=time, counts=counts, data_mode="counts",
                   trigger_number=trigger_number, channels=channels, instrument="batse")

    @staticmethod
    def load_batse_data(name: str, channels: Union[np.ndarray, str]) -> tuple:
        grb_dir, _, _ = prompt_directory_structure(grb=name.lstrip("GRB"))
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
    def _stripped_name(self) -> str:
        return self.name.lstrip('GRB')

    @property
    def trigger_number(self) -> str:
        return self._trigger_number

    @trigger_number.setter
    def trigger_number(self, trigger_number: str) -> None:
        if trigger_number is None:
            self._trigger_number = get_batse_trigger_from_grb(self.name)
        else:
            self._trigger_number = str(trigger_number)

    def plot_data(self, **kwargs: dict) -> None:
        plt.step(self.time, self.counts / self.bin_size)
        plt.show()
        plt.clf()

    def plot_lightcurve(
            self, model: str, axes: matplotlib.axes.Axes = None, plot_save: bool = True, plot_show: bool = True,
            random_models: int = 1000, posterior: pd.DataFrame = None, outdir: str = None, **kwargs: dict) -> None:
        plt.clf()
        plt.step(self.time, self.counts / self.bin_size)
        plt.plot(self.time, model(self.time, **dict(posterior.iloc[-1])))
        plt.show()
        plt.clf()

    def plot_different_channels(self) -> None:
        pass

    @property
    def event_table(self) -> str:
        return os.path.join(dirname, f'../tables/BATSE_4B_catalogue.csv')

    def _set_data(self) -> None:
        dtypes = dict(trigger_num=np.int32, t90=np.float64, t90_error=np.float64, t90_start=np.float64)
        columns = list(dtypes.keys())
        self.data = pd.read_csv(self.event_table, header=0, usecols=columns, dtype=dtypes)
        self._data_index = self.data.index[self.data['trigger_num'] == int(self.trigger_number)].tolist()[0]

    @property
    def t90(self) -> float:
        return self.data['t90'][self._data_index]

    @property
    def t90_error(self) -> float:
        return self.data['t90_error'][self._data_index]

    @property
    def t90_start(self) -> float:
        return self.data['t90_start'][self._data_index]

    @property
    def t90_end(self) -> float:
        return self.t90_start + self.t90
