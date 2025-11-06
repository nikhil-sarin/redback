from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Union

from redback.get_data.utils import get_batse_trigger_from_grb
from redback.get_data.directory import swift_prompt_directory_structure, batse_prompt_directory_structure
from redback.transient.transient import Transient

dirname = os.path.dirname(__file__)


class PromptTimeSeries(Transient):
    """
    Class for GRB prompt emission time series data.

    Inherits from Transient and provides prompt emission-specific data loading and processing
    for BATSE and Swift data.

    Attributes
    ----------
    DATA_MODES : list
        Valid data modes: ['counts', 'ttes'].
    """
    DATA_MODES = ['counts', 'ttes']

    def __init__(
            self, name: str, bin_size: float = 1, ttes: np.ndarray = None, time: np.ndarray = None,
            time_err: np.ndarray = None, time_rest_frame: np.ndarray = None, time_rest_frame_err: np.ndarray = None,
            counts: np.ndarray = None, channel_tags: np.ndarray = None, data_mode: str = 'ttes',
            trigger_number: str = None, channels: Union[np.ndarray, str] = "all", instrument: str = "batse",
            **kwargs: None) -> None:
        """
        General constructor for the PromptTimeSeries class.

        Parameters
        ----------
        name : str
            Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
        bin_size : float, optional
            Bin size for binning time-tagged event data (default is 1).
        ttes : np.ndarray, optional
            Time-tagged events data for unbinned prompt data.
        time : np.ndarray, optional
            Times in the observer frame.
        time_err : np.ndarray, optional
            Time errors in the observer frame.
        time_rest_frame : np.ndarray, optional
            Times in the rest frame. Used for luminosity data.
        time_rest_frame_err : np.ndarray, optional
            Time errors in the rest frame. Used for luminosity data.
        counts : np.ndarray, optional
            The number of counts at each given time.
        channel_tags : np.ndarray, optional
            The channel tag associated with each time.
        data_mode : str, optional
            Data mode. Must be one from `PromptTimeSeries.DATA_MODES` (default is 'ttes').
        trigger_number : str, optional
            BATSE trigger number.
        channels : Union[np.ndarray, str], optional
            Array of channels to use. Use all channels if 'all' is given (default is 'all').
        instrument : str, optional
            Instrument we use. Must be 'batse' or 'swift' (default is 'batse').
        **kwargs : dict, optional
            Any other keyword arguments.

        Examples
        --------
        >>> import redback
        >>> prompt = PromptTimeSeries.from_batse_grb_name('GRB910425', channels=[0, 1, 2, 3])
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame,
                         time_rest_frame_err=time_rest_frame_err, counts=counts, ttes=ttes, bin_size=bin_size,
                         name=name, data_mode=data_mode, **kwargs)
        self.channel_tags = channel_tags
        self.trigger_number = trigger_number
        self.channels = channels
        self.instrument = instrument
        self._set_data()
        if self.instrument == "batse":
            self.directory_structure = batse_prompt_directory_structure(grb=self.name)
        elif self.instrument == "swift":
            self.directory_structure = swift_prompt_directory_structure(grb=self.name, bin_size=self.bin_size)
        else:
            raise ValueError("Instrument must be either 'batse' or 'swift'.")

    @classmethod
    def from_batse_grb_name(
            cls, name: str, trigger_number: str = None, channels: Union[np.ndarray, str] = "all") -> PromptTimeSeries:
        """
        Constructor that loads BATSE data given a GRB name.

        Parameters
        ----------
        name : str
            Name of the transient (e.g., 'GRB910425').
        trigger_number : str, optional
            BATSE trigger number. If None, will be inferred from the name.
        channels : Union[np.ndarray, str], optional
            Array of channels to use. Use all channels if 'all' is given (default is 'all').

        Returns
        -------
        PromptTimeSeries
            An instance with BATSE data loaded.

        Examples
        --------
        >>> prompt = PromptTimeSeries.from_batse_grb_name('GRB910425')
        >>> # Use specific channels
        >>> prompt = PromptTimeSeries.from_batse_grb_name('GRB910425', channels=[0, 1])
        """
        time, dt, counts = cls.load_batse_data(name=name, channels=channels)
        return cls(name=name, bin_size=dt, time=time, counts=counts, data_mode="counts",
                   trigger_number=trigger_number, channels=channels, instrument="batse")

    @staticmethod
    def load_batse_data(name: str, channels: Union[np.ndarray, str]) -> tuple:
        """
        Load BATSE data given a transient name.

        Parameters
        ----------
        name : str
            Name of the GRB (e.g., 'GRB910425').
        channels : Union[np.ndarray, str]
            Array of channels to use. Use all channels if 'all' is given.

        Returns
        -------
        tuple
            Time, time step size, and counts in format (time, dt, counts).
        """
        name = f"GRB{name.lstrip('GRB')}"
        directory_structure = batse_prompt_directory_structure(grb=name)
        path = directory_structure.directory_path + name.lstrip('GRB') + '_BATSE_lc.csv'
        _time_series_data = np.genfromtxt(path, delimiter=",")[1:]

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
        """Strips 'GRB' from the transient name.

        :return: The stripped transient name.
        :rtype: str
        """
        return self.name.lstrip('GRB')

    @property
    def trigger_number(self) -> str:
        """Trigger number getter.

        :return: The trigger number.
        :rtype: str
        """
        return self._trigger_number

    @trigger_number.setter
    def trigger_number(self, trigger_number: str) -> None:
        """Trigger number setter. If no trigger number is given, get trigger number from GRB using the conversion table.

        :param trigger_number: The trigger number.
        :type trigger_number: str
        """
        if trigger_number is None:
            self._trigger_number = str(get_batse_trigger_from_grb(self.name))
        else:
            self._trigger_number = str(trigger_number)

    def plot_data(self, **kwargs: None) -> None:
        """
        Simple plot of the prompt emission data.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments (currently unused).

        Examples
        --------
        >>> prompt.plot_data()
        """
        plt.step(self.time, self.counts / self.bin_size)
        plt.ylabel('Counts')
        plt.xlabel('Time')
        plt.show()
        plt.clf()

    def plot_lightcurve(
            self, model: callable, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True,
            random_models: int = 1000, posterior: pd.DataFrame = None, outdir: str = None, **kwargs: None) -> None:
        """
        Plot the prompt emission lightcurve with model predictions.

        Parameters
        ----------
        model : callable
            The model function to use for predictions.
        axes : matplotlib.axes.Axes, optional
            Axes to plot into. Currently a placeholder.
        save : bool, optional
            Whether to save the plot (default is True). Currently a placeholder.
        show : bool, optional
            Whether to show the plot (default is True). Currently a placeholder.
        random_models : int, optional
            Number of random posterior samples to use for plots (default is 1000).
        posterior : pd.DataFrame, optional
            Posterior from which to draw samples.
        outdir : str, optional
            Directory to save the plot in. Currently a placeholder.
        **kwargs : dict, optional
            All other plotting keyword arguments. Currently a placeholder.

        Examples
        --------
        >>> prompt.plot_lightcurve(model=my_model, posterior=result.posterior)
        """
        plt.clf()
        plt.step(self.time, self.counts / self.bin_size)
        plt.plot(self.time, model(self.time, **dict(posterior.iloc[-1])))
        plt.ylabel('Counts')
        plt.xlabel('Time')
        plt.show()
        plt.clf()

    def plot_different_channels(self) -> None:
        """

        """
        pass

    @property
    def event_table(self) -> str:
        """Gets the event table using a relative path.

        :return: The event table.
        :rtype: str
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
        The T90 duration of the GRB.

        Returns
        -------
        float
            The T90 value in seconds.
        """
        return self.data['t90'][self._data_index]

    @property
    def t90_error(self) -> float:
        """
        The T90 error.

        Returns
        -------
        float
            The T90 error value in seconds.
        """
        return self.data['t90_error'][self._data_index]

    @property
    def t90_start(self) -> float:
        """
        The T90 start time.

        Returns
        -------
        float
            The T90 start time in seconds.
        """
        return self.data['t90_start'][self._data_index]

    @property
    def t90_end(self) -> float:
        """
        The T90 end time.

        Returns
        -------
        float
            The T90 end time in seconds (t90_start + t90).
        """
        return self.t90_start + self.t90
