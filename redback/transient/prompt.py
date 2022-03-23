from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Union

from redback.get_data.utils import get_batse_trigger_from_grb
from redback.get_data.directory import swift_prompt_directory_structure
from redback.transient.transient import Transient

dirname = os.path.dirname(__file__)


class PromptTimeSeries(Transient):
    DATA_MODES = ['counts', 'ttes']

    def __init__(
            self, name: str, bin_size: float = 1, ttes: np.ndarray = None, time: np.ndarray = None, 
            time_err: np.ndarray = None, time_rest_frame: np.ndarray = None, time_rest_frame_err: np.ndarray = None, 
            counts: np.ndarray = None, channel_tags: np.ndarray = None, data_mode: str = 'ttes',
            trigger_number: str = None, channels: Union[np.ndarray, str] = "all", instrument: str = "batse", 
            **kwargs: None) -> None:
        """

        Parameters
        ----------
        name: str
            Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs
        bin_size: float
            Bin size for binning time-tagged event data.
        ttes: np.ndarray, optional
            Time-tagged events data for unbinned prompt data.
        time: np.ndarray, optional
            Times in the observer frame.
        time_err: np.ndarray, optional
            Time errors in the observer frame.
        time_rest_frame: np.ndarray, optional
            Times in the rest frame. Used for luminosity data.
        time_rest_frame_err: np.ndarray, optional
            Time errors in the rest frame. Used for luminosity data.
        counts: np.ndarray, optional
            The number of counts at each given time.
        channel_tags: np.ndarray, optional
            The channel tag associated with each time.
        data_mode: str
            Data mode. Must be one from `PromptTimeSeries.DATA_MODES`.
        trigger_number: str
            BATSE trigger number.
        channels: Union[np.ndarray, float]
            Array of channels to use. Use all channels if 'all' is given.
        instrument: str, optional
            Instrument we use. Default is 'batse'.
        kwargs:
            Any other kwargs.
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame,
                         time_rest_frame_err=time_rest_frame_err, counts=counts, ttes=ttes, bin_size=bin_size,
                         name=name, data_mode=data_mode, **kwargs)
        self.channel_tags = channel_tags
        self.trigger_number = trigger_number
        self.channels = channels
        self.instrument = instrument
        self._set_data()
        self.directory_structure = swift_prompt_directory_structure(grb=self.name, bin_size=self.bin_size)

    @classmethod
    def from_batse_grb_name(
            cls, name: str, trigger_number: str = None, channels: Union[np.ndarray, str] = "all") -> PromptTimeSeries:
        """
        Constructor that loads batse data given a trigger number.

        Parameters
        ----------
        name: str
            Name of the transient.
        trigger_number: str
            BATSE trigger number.
        channels: Union[np.ndarray, float]
            Array of channels to use. Use all channels if 'all' is given.

        Returns
        -------
        PromptTimeSeries: An instance of `PromptTimeSeries`.
        """
        time, dt, counts = cls.load_batse_data(name=name, channels=channels)
        return cls(name=name, bin_size=dt, time=time, counts=counts, data_mode="counts",
                   trigger_number=trigger_number, channels=channels, instrument="batse")

    @staticmethod
    def load_batse_data(name: str, channels: Union[np.ndarray, str]) -> tuple:
        """
        Load batse data given a transient name.

        Parameters
        ----------
        name: str
            Name of the GRB, e.g. GRB123456.
        channels: Union[np.ndarray, float]
            Array of channels to use. Use all channels if 'all' is given.

        Returns
        -------
        tuple: Time, time step size, and counts in the format (time, dt, counts)

        """
        name = f"GRB{name.lstrip('GRB')}"
        directory_structure = swift_prompt_directory_structure(grb=name)
        _time_series_data = np.genfromtxt(directory_structure.processed_file_path, delimiter=",")[1:]

        bin_left = _time_series_data[:, 0]
        bin_right = _time_series_data[:, 1]
        dt = bin_right - bin_left
        time = 0.5 * (bin_left + bin_right)

        counts_by_channel = [np.around(_time_series_data[:, i] * dt) for i in [2, 4, 6, 8]]
        if str(channels) == "all":
            channels = np.array([0, 1, 2, 3])

        counts = np.zeros(len(time))
        for c in channels:
            counts += counts_by_channel[c]

        return time, dt, counts

    @property
    def _stripped_name(self) -> str:
        """
        Strips 'GRB' from the transient name.

        Returns
        -------
        str: The stripped transient name.
        """
        return self.name.lstrip('GRB')

    @property
    def trigger_number(self) -> str:
        """
        Trigger number getter.

        Returns
        -------
        str: The trigger number.
        """
        return self._trigger_number

    @trigger_number.setter
    def trigger_number(self, trigger_number: str) -> None:
        """
        Trigger number setter. If no trigger number is given, get trigger number from GRB using the conversion table.

        Parameters
        ----------
        trigger_number: str
            The trigger number.
        """
        if trigger_number is None:
            self._trigger_number = str(get_batse_trigger_from_grb(self.name))
        else:
            self._trigger_number = str(trigger_number)

    def plot_data(self, **kwargs: None) -> None:
        """
        Simple plot of the data.

        Parameters
        ----------
        kwargs:
            Placeholder.
        """
        plt.step(self.time, self.counts / self.bin_size)
        plt.show()
        plt.clf()

    def plot_lightcurve(
            self, model: callable, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True,
            random_models: int = 1000, posterior: pd.DataFrame = None, outdir: str = None, **kwargs: None) -> None:
        """

        Parameters
        ----------
        model: callable
            The model we are using
        axes: matplotlib.axes.Axes, optional
            Axes to plot into. Currently a placeholder.
        save: bool, option
            Whether to save the plot. Default is `True`. Currently, a placeholder.
        show: bool, optional
            Whether to show the plot. Default is `True`. Currently, a placeholder.
        random_models: int, optional
            Number of random posterior samples to use for plots. Default is 1000.
        posterior: pd.DataFrame, optional
            Posterior from which to draw samples from.
        outdir:
            Out directory to save the plot in. Currently, a placeholder.
        kwargs: dict
            All other plotting kwargs. Currently, a placeholder.
        """
        plt.clf()
        plt.step(self.time, self.counts / self.bin_size)
        plt.plot(self.time, model(self.time, **dict(posterior.iloc[-1])))
        plt.show()
        plt.clf()

    def plot_different_channels(self) -> None:
        """

        """
        pass

    @property
    def event_table(self) -> str:
        """
        Gets the event table using a relative path.

        Returns
        -------
        str: The event table.
        """
        return os.path.join(dirname, f'../tables/BATSE_4B_catalogue.csv')

    def _set_data(self) -> None:
        """
        Sets the data from the event table.
        """
        dtypes = dict(trigger_num=np.int32, t90=np.float64, t90_error=np.float64, t90_start=np.float64)
        columns = list(dtypes.keys())
        self.data = pd.read_csv(self.event_table, header=0, usecols=columns, dtype=dtypes)
        self._data_index = self.data.index[self.data['trigger_num'] == int(self.trigger_number)].tolist()[0]

    @property
    def t90(self) -> float:
        """
        Returns
        -------
        float: The t90 data.
        """
        return self.data['t90'][self._data_index]

    @property
    def t90_error(self) -> float:
        """
        Returns
        -------
        float: The t90 error value.
        """
        return self.data['t90_error'][self._data_index]

    @property
    def t90_start(self) -> float:
        """
        Returns
        -------
        float: The t90 start value.
        """
        return self.data['t90_start'][self._data_index]

    @property
    def t90_end(self) -> float:
        """
        Returns
        -------
        float: The t90 end value.
        """
        return self.t90_start + self.t90
