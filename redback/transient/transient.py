from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os.path import join
import pandas as pd
from typing import Union

import redback
from redback.plotting import \
    LuminosityPlotter, FluxDensityPlotter, IntegratedFluxPlotter, MagnitudePlotter


class Transient(object):
    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'magnitude', 'counts', 'ttes']
    _ATTRIBUTE_NAME_DICT = dict(luminosity="Lum50", flux="flux", flux_density="flux_density",
                                counts="counts", magnitude="magnitude")

    ylabel_dict = dict(luminosity=r'Luminosity [$10^{50}$ erg s$^{-1}$]',
                       magnitude=r'Magnitude',
                       flux=r'Flux [erg cm$^{-2}$ s$^{-1}$]',
                       flux_density=r'Flux density [mJy]',
                       counts=r'Counts')

    luminosity_data = redback.utils.DataModeSwitch('luminosity')
    flux_data = redback.utils.DataModeSwitch('flux')
    flux_density_data = redback.utils.DataModeSwitch('flux_density')
    magnitude_data = redback.utils.DataModeSwitch('magnitude')
    counts_data = redback.utils.DataModeSwitch('counts')
    tte_data = redback.utils.DataModeSwitch('ttes')

    def __init__(
            self, time: np.ndarray = None, time_err: np.ndarray = None, time_mjd: np.ndarray = None,
            time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None, time_rest_frame_err: np.ndarray = None,
            Lum50: np.ndarray = None, Lum50_err: np.ndarray = None, flux: np.ndarray = None,
            flux_err: np.ndarray = None, flux_density: np.ndarray = None, flux_density_err: np.ndarray = None,
            magnitude: np.ndarray = None, magnitude_err: np.ndarray = None, counts: np.ndarray = None,
            ttes: np.ndarray = None, bin_size: float = None, redshift: float = np.nan, data_mode: str = None,
            name: str = '', photon_index: float = np.nan, use_phase_model: bool = False,
            frequency: np.ndarray = None, system: np.ndarray = None, bands: np.ndarray = None,
            active_bands: Union[np.ndarray, str] = None, **kwargs: dict) -> None:
        """
        This is a general constructor for the Transient class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).


        Parameters
        ----------
        time: np.ndarray, optional
            Times in the observer frame.
        time_err: np.ndarray, optional
            Time errors in the observer frame.
        time_mjd: np.ndarray, optional
            Times in MJD. Used if using phase model.
        time_mjd_err: np.ndarray, optional
            Time errors in MJD. Used if using phase model.
        time_rest_frame: np.ndarray, optional
            Times in the rest frame. Used for luminosity data.
        time_rest_frame_err: np.ndarray, optional
            Time errors in the rest frame. Used for luminosity data.
        Lum50: np.ndarray, optional
            Luminosity values.
        Lum50_err: np.ndarray, optional
            Luminosity error values.
        flux: np.ndarray, optional
            Flux values.
        flux_err: np.ndarray, optional
            Flux error values.
        flux_density: np.ndarray, optional
            Flux density values.
        flux_density_err: np.ndarray, optional
            Flux density error values.
        magnitude: np.ndarray, optional
            Magnitude values for photometry data.
        magnitude_err: np.ndarray, optional
            Magnitude error values for photometry data.
        counts: np.ndarray, optional
            Counts for prompt data.
        ttes: np.ndarray, optional
            Time-tagged events data for unbinned prompt data.
        bin_size: float, optional
            Bin size for binning time-tagged event data.
        redshift: float, optional
            Redshift value.
        data_mode: str, optional
            Data mode. Must be one from `Transient.DATA_MODES`.
        name: str, optional
            Name of the transient.
        photon_index: float, optional
            Photon index value.
        use_phase_model: bool, optional
            Whether we are using a phase model.
        frequency: np.ndarray, optional
            Array of band frequencies in photometry data.
        system: np.ndarray, optional
            System values.
        bands: np.ndarray, optional
            Band values.
        active_bands: Union[list, np.ndarray], optional
            List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
        kwargs: dict, optional
            Additional callables:
            bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use
                                  redback.utils.bands_to_frequency if not given.
            bin_ttes: Binning function for time-tagged event data. Use redback.utils.bands_to_frequency if not given.
        """
        self.bin_size = bin_size
        self.bin_ttes = kwargs.get("bin_ttes", redback.utils.bin_ttes)
        self.bands_to_frequency = kwargs.get("bands_to_frequency", redback.utils.bands_to_frequency)

        if data_mode == 'ttes':
            time, counts = self.bin_ttes(ttes, self.bin_size)

        self.time = time
        self.time_err = time_err
        self.time_mjd = time_mjd
        self.time_mjd_err = time_mjd_err
        self.time_rest_frame = time_rest_frame
        self.time_rest_frame_err = time_rest_frame_err

        self.Lum50 = Lum50
        self.Lum50_err = Lum50_err
        self.flux = flux
        self.flux_err = flux_err
        self.flux_density = flux_density
        self.flux_density_err = flux_density_err
        self.magnitude = magnitude
        self.magnitude_err = magnitude_err
        self.counts = counts
        self.counts_err = np.sqrt(counts) if counts is not None else None
        self.ttes = ttes

        self._frequency = None
        self._bands = None
        self.set_bands_and_frequency(bands=bands, frequency=frequency)
        self.system = system
        self.active_bands = active_bands
        self.data_mode = data_mode
        self.redshift = redshift
        self.name = name
        self.use_phase_model = use_phase_model

        self.meta_data = None

        self.photon_index = photon_index

    @property
    def _time_attribute_name(self) -> str:
        if self.luminosity_data:
            return "time_rest_frame"
        elif self.use_phase_model:
            return "time_mjd"
        return "time"

    @property
    def _time_err_attribute_name(self) -> str:
        return self._time_attribute_name + "_err"

    @property
    def _y_attribute_name(self) -> str:
        return self._ATTRIBUTE_NAME_DICT[self.data_mode]

    @property
    def _y_err_attribute_name(self) -> str:
        return self._ATTRIBUTE_NAME_DICT[self.data_mode] + "_err"

    @property
    def x(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray: The time values given the active data mode.
        """
        return getattr(self, self._time_attribute_name)

    @x.setter
    def x(self, x: np.ndarray) -> None:
        """
        Sets the time values for the active data mode.

        Parameters
        -------
        x: np.ndarray
            The desired time values.
        """
        setattr(self, self._time_attribute_name, x)

    @property
    def x_err(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray: The time error values given the active data mode.
        """
        return getattr(self, self._time_err_attribute_name)

    @x_err.setter
    def x_err(self, x_err: np.ndarray) -> None:
        """
        Sets the time error values for the active data mode.

        Parameters
        -------
        x_err: np.ndarray
            The desired time error values.
        """
        setattr(self, self._time_err_attribute_name, x_err)

    @property
    def y(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray: The y values given the active data mode.
        """

        return getattr(self, self._y_attribute_name)

    @y.setter
    def y(self, y: np.ndarray) -> None:
        """
        Sets the y values for the active data mode.

        Parameters
        -------
        y: np.ndarray
            The desired y values.
        """
        setattr(self, self._y_attribute_name, y)

    @property
    def y_err(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray: The y error values given the active data mode.
        """
        return getattr(self, self._y_err_attribute_name)

    @y_err.setter
    def y_err(self, y_err: np.ndarray) -> None:
        """
        Sets the y error values for the active data mode.

        Parameters
        -------
        y_err: np.ndarray
            The desired y error values.
        """
        setattr(self, self._y_err_attribute_name, y_err)

    @property
    def data_mode(self) -> str:
        """

        Returns
        -------
        str: The currently active data mode (one in `Transient.DATA_MODES`)
        """
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        """

        Parameters
        -------
        data_mode: str
            One of the data modes in `Transient.DATA_MODES`
        """
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    @property
    def xlabel(self) -> str:
        """

        Returns
        -------
        str: xlabel used in plotting functions
        """
        if self.use_phase_model:
            return r"Time [MJD]"
        else:
            return r"Time since burst [days]"

    @property
    def ylabel(self) -> str:
        """

        Returns
        -------
        str: ylabel used in plotting functions
        """
        try:
            return self.ylabel_dict[self.data_mode]
        except KeyError:
            raise ValueError("No data mode specified")

    def set_bands_and_frequency(
            self, bands: Union[None, list, np.ndarray], frequency: Union[None, list, np.ndarray]):
        if (bands is None and frequency is None) or (bands is not None and frequency is not None):
            self._bands = bands
            self._frequency = bands
        elif bands is None and frequency is not None:
            self._frequency = frequency
            self._bands = self.frequency
        elif bands is not None and frequency is None:
            self._bands = bands
            self._frequency = self.bands_to_frequency(self.bands)

    @property
    def frequency(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray: Used band frequencies
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: np.ndarray) -> None:
        """

        Parameters
        ----------
        frequency: np.ndarray
            Set band frequencies if an array is given. Otherwise, convert bands to frequencies.
        """
        self.set_bands_and_frequency(bands=self.bands, frequency=frequency)

    @property
    def bands(self) -> Union[list, None, np.ndarray]:
        return self._bands

    @bands.setter
    def bands(self, bands: Union[list, None, np.ndarray]):
        self.set_bands_and_frequency(bands=bands, frequency=self.frequency)

    @property
    def filtered_frequencies(self) -> np.array:
        return self.frequency[self.filtered_indices]

    @property
    def active_bands(self) -> list:
        """

        Returns
        -------
        list: Array of active bands used.
        """
        return self._active_bands

    @active_bands.setter
    def active_bands(self, active_bands: Union[list, str, None]) -> None:
        """

        Parameters
        ----------
        active_bands: Union[list, str]
            Sets active bands based on list given.
            If argument is 'all', all unique bands in `self.bands` will be used.
        """
        if str(active_bands) == 'all':
            self._active_bands = list(np.unique(self.bands))
        else:
            self._active_bands = active_bands

    @property
    def filtered_indices(self) -> Union[list, None]:
        if self.bands is None:
            return list(np.arange(len(self.x)))
        return [b in self.active_bands for b in self.bands]

    def get_filtered_data(self) -> tuple:
        """
        Used to filter flux density and photometry data, so we only use data that is using the active bands.

        Returns
        -------
        tuple: A tuple with the filtered data. Format is (x, x_err, y, y_err)
        """
        if self.flux_density_data or self.magnitude_data:
            filtered_x = self.x[self.filtered_indices]
            try:
                filtered_x_err = self.x_err[self.filtered_indices]
            except (IndexError, TypeError):
                filtered_x_err = None
            filtered_y = self.y[self.filtered_indices]
            filtered_y_err = self.y_err[self.filtered_indices]
            return filtered_x, filtered_x_err, filtered_y, filtered_y_err
        else:
            raise ValueError(f"Transient needs to be in flux density or magnitude data mode, "
                             f"but is in {self.data_mode} instead.")

    @property
    def unique_bands(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray: All bands that we get from the data, eliminating all duplicates.
        """
        return np.unique(self.bands)

    @property
    def unique_frequencies(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray: All frequencies that we get from the data, eliminating all duplicates.
        """
        try:
            if isinstance(self.unique_bands[0], (float, int)):
                return self.unique_bands
        except (TypeError, IndexError):
            pass
        return self.bands_to_frequency(self.unique_bands)

    @property
    def list_of_band_indices(self) -> list:
        """

        Returns
        -------
        list: Indices that map between bands in the data and the unique bands we obtain.
        """
        return [np.where(self.bands == np.array(b))[0] for b in self.unique_bands]

    @property
    def default_filters(self) -> list:
        """

        Returns
        -------
        list: Default list of filters to use
        """
        return ["g", "r", "i", "z", "y", "J", "H", "K"]

    @staticmethod
    def get_colors(filters: Union[np.ndarray, list]) -> matplotlib.colors.Colormap:
        """

        Parameters
        ----------
        filters: list
            Array of list of filters to use in the plot.

        Returns
        -------
        matplotlib.colors.Colormap: Colormap with one color for each filter
        """
        return matplotlib.cm.rainbow(np.linspace(0, 1, len(filters)))

    def plot_data(self, axes: matplotlib.axes.Axes = None, filename: str = None, outdir: str = None, save: bool = True,
            show: bool = True, plot_others: bool = True, color: str = 'k', **kwargs: dict) -> matplotlib.axes.Axes:
        """
        Plots the Afterglow lightcurve and returns Axes.

        Parameters
        ----------
        axes : Union[matplotlib.axes.Axes, None], optional
            Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        color: str, optional
            color of the data.
        kwargs: dict
            Additional keyword arguments to pass in the Plotter methods.

        Returns
        ----------
        matplotlib.axes.Axes: The axes with the plot.
        """

        if self.flux_data:
            plotter = IntegratedFluxPlotter(transient=self, color=color, filename=filename, outdir=outdir, **kwargs)
        elif self.luminosity_data:
            plotter = LuminosityPlotter(transient=self, color=color, filename=filename, outdir=outdir, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, color=color, filename=filename, outdir=outdir, plot_others=plot_others, **kwargs)
        elif self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, color=color, filename=filename, outdir=outdir, plot_others=plot_others, **kwargs)
        else:
            return axes
        return plotter.plot_data(axes=axes, save=save, show=show)

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, filename: str = None,
            outdir: str = None, ncols: int = 2, save: bool = True, show: bool = True,
            nrows: int = None, figsize: tuple = None, filters: list = None, **kwargs: dict) \
            -> matplotlib.axes.Axes:
        """

        Parameters
        ----------
        figure: matplotlib.figure.Figure, optional
            Figure can be given if defaults are not satisfying
        axes: matplotlib.axes.Axes, optional
            Axes can be given if defaults are not satisfying
        ncols: int, optional
            Number of columns to use on the plot. Default is 2.
        nrows: int, optional
            Number of rows to use on the plot. If None are given this will
            be inferred from ncols and the number of filters.
        figsize: tuple, optional
            Size of the figure. A default based on ncols and nrows will be used if None is given.
        filters: list, optional
            Which bands to plot. Will use default filters if None is given.
        kwargs:
            Additional optional plotting kwargs:
            wspace: Extra argument for matplotlib.pyplot.subplots_adjust
            hspace: Extra argument for matplotlib.pyplot.subplots_adjust
            fontsize: Label fontsize
            errorbar_fmt: Errorbar format ('fmt' argument in matplotlib.pyplot.errorbar)
            colors: colors to be used for the bands
            xlabel: Plot xlabel
            ylabel: Plot ylabel
            plot_label: Addional filename label appended to the default name

        Returns
        -------

        """
        if self.data_mode not in ['flux_density', 'magnitude']:
            raise ValueError(
                f'You cannot plot multiband data with {self.data_mode} data mode . Why are you doing this?')
        if self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, filters=filters, filename=filename, outdir=outdir, nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, filters=filters, filename=filename, outdir=outdir, nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        else:
            return
        return plotter.plot_multiband(figure=figure, axes=axes, save=save, show=show)

    def plot_lightcurve(
            self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: object) -> None:
        """

        Parameters
        ----------
        model: callable
            The model used to plot the lightcurve.
        filename: str, optional
            The output filename. Otherwise, use default which starts with the name
            attribute and ends with *lightcurve.png.
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        save: bool, optional
            Whether to save the plot.
        show: bool, optional
            Whether to show the plot.
        random_models: int, optional
            Number of random posterior samples plotted faintly. Default is 100.
        posterior: pd.DataFrame, optional
            Posterior distribution to which to draw samples from. Is optional but must be given.
        outdir: str, optional
            Out directory in which to save the plot. Default is the current working directory.
        model_kwargs: dict
            Additional keyword arguments to be passed into the model.
        kwargs: dict
            No current function.
        """
        if self.flux_data:
            plotter = IntegratedFluxPlotter(transient=self, model=model, filename=filename, outdir=outdir, posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        elif self.luminosity_data:
            plotter = LuminosityPlotter(transient=self, model=model, filename=filename, outdir=outdir, posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, model=model, filename=filename, outdir=outdir, posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        elif self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, model=model, filename=filename, outdir=outdir, posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        else:
            return axes
        return plotter.plot_lightcurve(axes=axes, save=save, show=show)

    def plot_multiband_lightcurve(
            self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: object) -> None:
        """

        Parameters
        ----------
        model: callable
            The model used to plot the lightcurve
        filename: str, optional
            The output filename. Otherwise, use default which starts with the name
            attribute and ends with *lightcurve.png.
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        save: bool, optional
            Whether to save the plot.
        show: bool, optional
            Whether to show the plot.
        random_models: int, optional
            Number of random posterior samples plotted faintly. Default is 100.
        posterior: pd.DataFrame, optional
            Posterior distribution to which to draw samples from. Is optional but must be given.
        outdir: str, optional
            Out directory in which to save the plot. Default is the current working directory.
        model_kwargs: dict
            Additional keyword arguments to be passed into the model.
        kwargs: dict
            No current function.
        -------

        """
        if self.data_mode not in ['flux_density', 'magnitude']:
            raise ValueError(
                f'You cannot plot multiband data with {self.data_mode} data mode . Why are you doing this?')
        if self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, model=model, filename=filename, outdir=outdir, posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, model=model, filename=filename, outdir=outdir, posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        else:
            return
        return plotter.plot_multiband_lightcurve(axes=axes, save=save, show=show)


class OpticalTransient(Transient):
    DATA_MODES = ['flux', 'flux_density', 'magnitude', 'luminosity']

    @staticmethod
    def load_data(processed_file_path, data_mode="magnitude"):
        """
        Loads data from specified directory and file, and returns it as a tuple.

        Parameters
        ----------
        processed_file_path: str
            Path to the processed file to load
        data_mode: str, optional
            Name of the data mode. Must be from ['magnitude', 'flux_density', 'all']. Default is magnitude.

        Returns
        -------
        tuple: Six elements when querying magnitude or flux_density data, Eight for 'all'
        """
        df = pd.read_csv(processed_file_path)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        system = np.array(df["system"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        if data_mode == "magnitude":
            return time_days, time_mjd, magnitude, magnitude_err, bands, system
        elif data_mode == "flux_density":
            return time_days, time_mjd, flux_density, flux_density_err, bands, system
        elif data_mode == "all":
            return time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands, system

    def __init__(
            self, name: str, data_mode: str = 'magnitude', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux: np.ndarray = None, flux_err: np.ndarray = None, flux_density: np.ndarray = None,
            flux_density_err: np.ndarray = None, magnitude: np.ndarray = None, magnitude_err: np.ndarray = None,
            redshift: float = np.nan, photon_index: float = np.nan, frequency: np.ndarray = None,
            bands: np.ndarray = None, system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False, **kwargs: dict) -> None:
        """
        This is a general constructor for the Transient class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

        Parameters
        ----------
        name: str
            Name of the transient.
        data_mode: str, optional
            Data mode. Must be one from `OpticalTransient.DATA_MODES`.
        time: np.ndarray, optional
            Times in the observer frame.
        time_err: np.ndarray, optional
            Time errors in the observer frame.
        time_mjd: np.ndarray, optional
            Times in MJD. Used if using phase model.
        time_mjd_err: np.ndarray, optional
            Time errors in MJD. Used if using phase model.
        time_rest_frame: np.ndarray, optional
            Times in the rest frame. Used for luminosity data.
        time_rest_frame_err: np.ndarray, optional
            Time errors in the rest frame. Used for luminosity data.
        Lum50: np.ndarray, optional
            Luminosity values.
        Lum50_err: np.ndarray, optional
            Luminosity error values.
        flux: np.ndarray, optional
            Flux values.
        flux_err: np.ndarray, optional
            Flux error values.
        flux_density: np.ndarray, optional
            Flux density values.
        flux_density_err: np.ndarray, optional
            Flux density error values.
        magnitude: np.ndarray, optional
            Magnitude values for photometry data.
        magnitude_err: np.ndarray, optional
            Magnitude error values for photometry data.
        redshift: float, optional
            Redshift value.
        photon_index: float, optional
            Photon index value.
        frequency: np.ndarray, optional
            Array of band frequencies in photometry data.
        bands: np.ndarray, optional
            Band values.
        system: np.ndarray, optional
            System values.
        active_bands: Union[list, np.ndarray], optional
            List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
        use_phase_model: bool, optional
            Whether we are using a phase model.
        kwargs: dict, optional
            Additional callables:
            bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use
                                  redback.utils.bands_to_frequency if not given.
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, frequency=frequency,
                         time_rest_frame_err=time_rest_frame_err, Lum50=Lum50, Lum50_err=Lum50_err,
                         flux=flux, flux_err=flux_err, redshift=redshift, photon_index=photon_index,
                         flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                         magnitude_err=magnitude_err, data_mode=data_mode, name=name,
                         use_phase_model=use_phase_model, system=system, bands=bands, active_bands=active_bands,
                         **kwargs)
        self.directory_structure = None

    @classmethod
    def from_open_access_catalogue(
            cls, name: str, data_mode: str = "magnitude", active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False) -> OpticalTransient:
        """
        Constructor method to built object from Open Access Catalogue

        Parameters
        ----------
        name: str
            Name of the transient.
        data_mode: str, optional
            Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
        active_bands: Union[np.ndarray, str]
            Sets active bands based on array given.
            If argument is 'all', all unique bands in `self.bands` will be used.
        use_phase_model: bool, optional
            Whether to use a phase model.

        Returns
        -------
        OpticalTransient: A class instance.
        """
        if cls.__name__ == "TDE":
            transient_type = "tidal_disruption_event"
        else:
            transient_type = cls.__name__.lower()
        directory_structure = redback.get_data.directory.open_access_directory_structure(
            transient=name, transient_type=transient_type)
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands, system = \
            cls.load_data(processed_file_path=directory_structure.processed_file_path, data_mode="all")
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, bands=bands, system=system, active_bands=active_bands,
                   use_phase_model=use_phase_model)

    @property
    def event_table(self) -> str:
        """

        Returns
        -------
        str: Path to the metadata table.
        """
        return f"{self.directory_structure.directory_path}/{self.name}_metadata.csv"

    def _set_data(self) -> None:
        """
        Sets the metadata from the event table.
        """
        try:
            meta_data = pd.read_csv(self.event_table, error_bad_lines=False, delimiter=',', dtype='str')
        except FileNotFoundError as e:
            redback.utils.logger.warning(e)
            redback.utils.logger.warning("Setting metadata to None")
            meta_data = None
        self.meta_data = meta_data

    @property
    def transient_dir(self) -> str:
        """

        Returns
        -------
        str: The transient directory given the name of the transient.
        """
        return self._get_transient_dir()

    def _get_transient_dir(self) -> str:
        transient_dir, _, _ = redback.get_data.directory.open_access_directory_structure(transient=self.name,
                                                                                         transient_type=self.__class__.__name__.lower())
        return transient_dir
