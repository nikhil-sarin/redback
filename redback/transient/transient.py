from __future__ import annotations

from typing import Union

import matplotlib
import numpy as np
import pandas as pd

import redback
from redback.plotting import \
    LuminosityPlotter, FluxDensityPlotter, IntegratedFluxPlotter, MagnitudePlotter, IntegratedFluxOpticalPlotter


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
            optical_data: bool = False, frequency: np.ndarray = None, system: np.ndarray = None, bands: np.ndarray = None,
            active_bands: Union[np.ndarray, str] = None, **kwargs: None) -> None:
        """This is a general constructor for the Transient class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

        :param time: Times in the observer frame.
        :type time: np.ndarray, optional
        :param time_err: Time errors in the observer frame.
        :type time_err: np.ndarray, optional
        :param time_mjd: Times in MJD. Used if using phase model.
        :type time_mjd: np.ndarray, optional
        :param time_mjd_err: Time errors in MJD. Used if using phase model.
        :type time_mjd_err: np.ndarray, optional
        :param time_rest_frame: Times in the rest frame. Used for luminosity data.
        :type time_rest_frame: np.ndarray, optional
        :param time_rest_frame_err: Time errors in the rest frame. Used for luminosity data.
        :type time_rest_frame_err: np.ndarray, optional
        :param Lum50: Luminosity values.
        :type Lum50: np.ndarray, optional
        :param Lum50_err: Luminosity error values.
        :type Lum50_err: np.ndarray, optional
        :param flux: Flux values.
        :type flux: np.ndarray, optional
        :param flux_err: Flux error values.
        :type flux_err: np.ndarray, optional
        :param flux_density: Flux density values.
        :type flux_density: np.ndarray, optional
        :param flux_density_err: Flux density error values.
        :type flux_density_err: np.ndarray, optional
        :param magnitude: Magnitude values for photometry data.
        :type magnitude: np.ndarray, optional
        :param magnitude_err: Magnitude error values for photometry data.
        :type magnitude_err: np.ndarray, optional
        :param counts: Counts for prompt data.
        :type counts: np.ndarray, optional
        :param ttes: Time-tagged events data for unbinned prompt data.
        :type ttes: np.ndarray, optional
        :param bin_size: Bin size for binning time-tagged event data.
        :type bin_size: float, optional
        :param redshift: Redshift value.
        :type redshift: float, optional
        :param data_mode: Data mode. Must be one from `Transient.DATA_MODES`.
        :type data_mode: str, optional
        :param name: Name of the transient.
        :type name: str, optional
        :param photon_index: Photon index value.
        :type photon_index: float, optional
        :param use_phase_model: Whether we are using a phase model.
        :type use_phase_model: bool, optional
        :param optical_data: Whether we are fitting optical data, useful for plotting.
        :type optical_data: bool, optional
        :param frequency: Array of band frequencies in photometry data.
        :type frequency: np.ndarray, optional
        :param system: System values.
        :type system: np.ndarray, optional
        :param bands: Band values.
        :type bands: np.ndarray, optional
        :param active_bands: List or array of active bands to be used in the analysis.
                             Use all available bands if 'all' is given.
        :type active_bands: Union[list, np.ndarray], optional
        :param kwargs: Additional callables:
                       bands_to_frequency: Conversion function to convert a list of bands to frequencies.
                                           Use redback.utils.bands_to_frequency if not given.
                       bin_ttes: Binning function for time-tagged event data.
                                 Use redback.utils.bands_to_frequency if not given.
        :type kwargs: None, optional
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
        self.sncosmo_bands = redback.utils.sncosmo_bandname_from_band(self.bands)
        self.data_mode = data_mode
        self.redshift = redshift
        self.name = name
        self.use_phase_model = use_phase_model
        self.optical_data = optical_data

        self.meta_data = None
        self.photon_index = photon_index
        self.directory_structure = redback.get_data.directory.DirectoryStructure(
            directory_path=".", raw_file_path=".", processed_file_path=".")

    @staticmethod
    def load_data_generic(processed_file_path, data_mode="magnitude"):
        """Loads data from specified directory and file, and returns it as a tuple.

        :param processed_file_path: Path to the processed file to load
        :type processed_file_path: str
        :param data_mode: Name of the data mode.
                          Must be from ['magnitude', 'flux_density', 'all']. Default is magnitude.
        :type data_mode: str, optional

        :return: Six elements when querying magnitude or flux_density data, Eight for 'all'.
        :rtype: tuple
        """
        df = pd.read_csv(processed_file_path)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        if data_mode == "magnitude":
            return time_days, time_mjd, magnitude, magnitude_err, bands
        elif data_mode == "flux_density":
            return time_days, time_mjd, flux_density, flux_density_err, bands
        elif data_mode == "all":
            return time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands

    @classmethod
    def from_lasair_data(
            cls, name: str, data_mode: str = "magnitude", active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False) -> Transient:
        """Constructor method to built object from Open Access Catalogue.

        :param name: Name of the transient.
        :type name: str
        :param data_mode: Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
        :type data_mode: str, optional
        :param active_bands: Sets active bands based on array given.
                             If argument is 'all', all unique bands in `self.bands` will be used.
        :type active_bands: Union[np.ndarray, str]
        :param use_phase_model: Whether to use a phase model.
        :type use_phase_model: bool, optional

        :return: A class instance.
        :rtype: OpticalTransient
        """
        if cls.__name__ == "TDE":
            transient_type = "tidal_disruption_event"
        else:
            transient_type = cls.__name__.lower()
        directory_structure = redback.get_data.directory.lasair_directory_structure(
            transient=name, transient_type=transient_type)
        df = pd.read_csv(directory_structure.processed_file_path)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        flux = np.array(df["flux(erg/cm2/s)"])
        flux_err = np.array(df["flux_error"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, flux=flux, flux_err=flux_err, bands=bands, active_bands=active_bands,
                   use_phase_model=use_phase_model, optical_data=True)

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
        :return: The time values given the active data mode.
        :rtype: np.ndarray
        """
        return getattr(self, self._time_attribute_name)

    @x.setter
    def x(self, x: np.ndarray) -> None:
        """Sets the time values for the active data mode.
        :param x: The desired time values.
        :type x: np.ndarray
        """
        setattr(self, self._time_attribute_name, x)

    @property
    def x_err(self) -> np.ndarray:
        """
        :return: The time error values given the active data mode.
        :rtype: np.ndarray
        """
        return getattr(self, self._time_err_attribute_name)

    @x_err.setter
    def x_err(self, x_err: np.ndarray) -> None:
        """Sets the time error values for the active data mode.
        :param x_err: The desired time error values.
        :type x_err: np.ndarray
        """
        setattr(self, self._time_err_attribute_name, x_err)

    @property
    def y(self) -> np.ndarray:
        """
        :return: The y values given the active data mode.
        :rtype: np.ndarray
        """

        return getattr(self, self._y_attribute_name)

    @y.setter
    def y(self, y: np.ndarray) -> None:
        """Sets the y values for the active data mode.
        :param y: The desired y values.
        :type y: np.ndarray
        """
        setattr(self, self._y_attribute_name, y)

    @property
    def y_err(self) -> np.ndarray:
        """
        :return: The y error values given the active data mode.
        :rtype: np.ndarray
        """
        return getattr(self, self._y_err_attribute_name)

    @y_err.setter
    def y_err(self, y_err: np.ndarray) -> None:
        """Sets the y error values for the active data mode.
        :param y_err: The desired y error values.
        :type y_err: np.ndarray
        """
        setattr(self, self._y_err_attribute_name, y_err)

    @property
    def data_mode(self) -> str:
        """
        :return: The currently active data mode (one in `Transient.DATA_MODES`).
        :rtype: str
        """
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        """
        :param data_mode: One of the data modes in `Transient.DATA_MODES`.
        :type data_mode: str
        """
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    @property
    def xlabel(self) -> str:
        """
        :return: xlabel used in plotting functions
        :rtype: str
        """
        if self.use_phase_model:
            return r"Time [MJD]"
        else:
            return r"Time since burst [days]"

    @property
    def ylabel(self) -> str:
        """
        :return: ylabel used in plotting functions
        :rtype: str
        """
        try:
            return self.ylabel_dict[self.data_mode]
        except KeyError:
            raise ValueError("No data mode specified")

    def set_bands_and_frequency(
            self, bands: Union[None, list, np.ndarray], frequency: Union[None, list, np.ndarray]):
        """Sets bands and frequencies at the same time to keep the logic consistent. If both are given use those values.
        If only frequencies are given, use them also as band names.
        If only bands are given, try to convert them to frequencies.

        :param bands: The bands, e.g. ['g', 'i'].
        :type bands: Union[None, list, np.ndarray]
        :param frequency: The frequencies associated with the bands.
        :type frequency: Union[None, list, np.ndarray]
        """
        if (bands is None and frequency is None) or (bands is not None and frequency is not None):
            self._bands = bands
            self._frequency = frequency
        elif bands is None and frequency is not None:
            self._frequency = frequency
            self._bands = self.frequency
        elif bands is not None and frequency is None:
            self._bands = bands
            self._frequency = self.bands_to_frequency(self.bands)

    @property
    def frequency(self) -> np.ndarray:
        """
        :return: Used band frequencies
        :rtype: np.ndarray
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: np.ndarray) -> None:
        """
        :param frequency: Set band frequencies if an array is given. Otherwise, convert bands to frequencies.
        :type frequency: np.ndarray
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
        """
        :return: The frequencies only associated with the active bands.
        :rtype: np.ndarray
        """
        return self.frequency[self.filtered_indices]

    @property
    def filtered_sncosmo_bands(self) -> np.array:
        """
        :return: The sncosmo bands only associated with the active bands.
        :rtype: np.ndarray
        """
        return self.sncosmo_bands[self.filtered_indices]

    @property
    def filtered_bands(self) -> np.array:
        """
        :return: The band names only associated with the active bands.
        :rtype: np.ndarray
        """
        return self.bands[self.filtered_indices]

    @property
    def active_bands(self) -> list:
        """
        :return: List of active bands used.
        :rtype list:
        """
        return self._active_bands

    @active_bands.setter
    def active_bands(self, active_bands: Union[list, str, None]) -> None:
        """
        :param active_bands: Sets active bands based on list given.
                             If argument is 'all', all unique bands in `self.bands` will be used.
        :type active_bands: Union[list, str]
        """
        if str(active_bands) == 'all':
            self._active_bands = list(np.unique(self.bands))
        else:
            self._active_bands = active_bands

    @property
    def filtered_indices(self) -> Union[list, None]:
        """
        :return: The list indices in `bands` associated with the active bands.
        :rtype: Union[list, None]
        """
        if self.bands is None:
            return list(np.arange(len(self.x)))
        return [b in self.active_bands for b in self.bands]

    def get_filtered_data(self) -> tuple:
        """Used to filter flux density, photometry or integrated flux data, so we only use data that is using the active bands.
        :return: A tuple with the filtered data. Format is (x, x_err, y, y_err)
        :rtype: tuple
        """
        if any([self.flux_data, self.magnitude_data, self.flux_density_data]):
            filtered_x = self.x[self.filtered_indices]
            try:
                filtered_x_err = self.x_err[self.filtered_indices]
            except (IndexError, TypeError):
                filtered_x_err = None
            filtered_y = self.y[self.filtered_indices]
            filtered_y_err = self.y_err[self.filtered_indices]
            return filtered_x, filtered_x_err, filtered_y, filtered_y_err
        else:
            raise ValueError(f"Transient needs to be in flux density, magnitude or flux data mode, "
                             f"but is in {self.data_mode} instead.")

    @property
    def unique_bands(self) -> np.ndarray:
        """
        :return: All bands that we get from the data, eliminating all duplicates.
        :rtype: np.ndarray
        """
        return np.unique(self.bands)

    @property
    def unique_frequencies(self) -> np.ndarray:
        """
        :return: All frequencies that we get from the data, eliminating all duplicates.
        :rtype: np.ndarray
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
        :return: Indices that map between bands in the data and the unique bands we obtain.
        :rtype: list
        """
        return [np.where(self.bands == np.array(b))[0] for b in self.unique_bands]

    @property
    def default_filters(self) -> list:
        """
        :return: Default list of filters to use.
        :rtype: list
        """
        return ["g", "r", "i", "z", "y", "J", "H", "K"]

    @staticmethod
    def get_colors(filters: Union[np.ndarray, list]) -> matplotlib.colors.Colormap:
        """
        :param filters: Array of list of filters to use in the plot.
        :type filters: Union[np.ndarray, list]
        :return: Colormap with one color for each filter.
        :rtype: matplotlib.colors.Colormap
        """
        return matplotlib.cm.rainbow(np.linspace(0, 1, len(filters)))

    def plot_data(self, axes: matplotlib.axes.Axes = None, filename: str = None, outdir: str = None, save: bool = True,
            show: bool = True, plot_others: bool = True, color: str = 'k', **kwargs) -> matplotlib.axes.Axes:
        """Plots the Transient data and returns Axes.

        :param axes: Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        :param filename: Name of the file to be plotted in.
        :param outdir: The directory in which to save the file in.
        :param save: Whether to save the plot. (Default value = True)
        :param show: Whether to show the plot. (Default value = True)
        :param plot_others: Whether to plot inactive bands. (Default value = True)
        :param color: Color of the data.
        :param kwargs: Additional keyword arguments to pass in the Plotter methods.
        Available in the online documentation under at `redback.plotting.Plotter`.
        `print(Transient.plot_data.__doc__)` to see all options!
        :return: The axes with the plot.
        """

        if self.flux_data:
            if self.optical_data:
                plotter = IntegratedFluxOpticalPlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                       plot_others=plot_others, **kwargs)
            else:
                plotter = IntegratedFluxPlotter(transient=self, color=color, filename=filename, outdir=outdir, **kwargs)
        elif self.luminosity_data:
            plotter = LuminosityPlotter(transient=self, color=color, filename=filename, outdir=outdir, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                         plot_others=plot_others, **kwargs)
        elif self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                       plot_others=plot_others, **kwargs)
        else:
            return axes
        return plotter.plot_data(axes=axes, save=save, show=show)

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, filename: str = None,
            outdir: str = None, ncols: int = 2, save: bool = True, show: bool = True,
            nrows: int = None, figsize: tuple = None, filters: list = None, **kwargs: None) \
            -> matplotlib.axes.Axes:
        """
        :param figure: Figure can be given if defaults are not satisfying
        :param axes: Axes can be given if defaults are not satisfying
        :param filename: Name of the file to be plotted in.
        :param outdir: The directory in which to save the file in.
        :param save: Whether to save the plot. (Default value = True)
        :param show: Whether to show the plot. (Default value = True)
        :param ncols: Number of columns to use on the plot. Default is 2.
        :param nrows: Number of rows to use on the plot. If None are given this will
                      be inferred from ncols and the number of filters.
        :param figsize: Size of the figure. A default based on ncols and nrows will be used if None is given.
        :param filters: Which bands to plot. Will use default filters if None is given.
        :param kwargs: Additional keyword arguments to pass in the Plotter methods.
        Available in the online documentation under at `redback.plotting.Plotter`.
        `print(Transient.plot_multiband.__doc__)` to see all options!
        :return: The axes.
        """
        if self.data_mode not in ['flux_density', 'magnitude', 'flux']:
            raise ValueError(
                f'You cannot plot multiband data with {self.data_mode} data mode . Why are you doing this?')
        if self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, filters=filters, filename=filename, outdir=outdir, nrows=nrows,
                                       ncols=ncols, figsize=figsize, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, filters=filters, filename=filename, outdir=outdir, nrows=nrows,
                                         ncols=ncols, figsize=figsize, **kwargs)
        elif self.flux_data:
            plotter = IntegratedFluxOpticalPlotter(transient=self, filters=filters, filename=filename, outdir=outdir,
                                                   nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        else:
            return
        return plotter.plot_multiband(figure=figure, axes=axes, save=save, show=show)

    def plot_lightcurve(
            self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: None) -> matplotlib.axes.Axes:
        """
        :param model: The model used to plot the lightcurve.
        :param filename: The output filename. Otherwise, use default which starts with the name
                         attribute and ends with *lightcurve.png.
        :param axes: Axes to plot in if given.
        :param save:Whether to save the plot.
        :param show: Whether to show the plot.
        :param random_models: Number of random posterior samples plotted faintly. (Default value = 100)
        :param posterior: Posterior distribution to which to draw samples from. Is optional but must be given.
        :param outdir: Out directory in which to save the plot. Default is the current working directory.
        :param model_kwargs: Additional keyword arguments to be passed into the model.
        :param kwargs: Additional keyword arguments to pass in the Plotter methods.
        Available in the online documentation under at `redback.plotting.Plotter`.
        `print(Transient.plot_lightcurve.__doc__)` to see all options!
        :return: The axes.
        """
        if self.flux_data:
            if self.optical_data:
                plotter = IntegratedFluxOpticalPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
            else:
                plotter = IntegratedFluxPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.luminosity_data:
            plotter = LuminosityPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.magnitude_data:
            plotter = MagnitudePlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        else:
            return axes
        return plotter.plot_lightcurve(axes=axes, save=save, show=show)

    def plot_residual(self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
                      save: bool = True, show: bool = True, posterior: pd.DataFrame = None,
                      model_kwargs: dict = None, **kwargs: None) -> matplotlib.axes.Axes:
        """
        :param model: The model used to plot the lightcurve.
        :param filename: The output filename. Otherwise, use default which starts with the name
                         attribute and ends with *lightcurve.png.
        :param axes: Axes to plot in if given.
        :param save:Whether to save the plot.
        :param show: Whether to show the plot.
        :param posterior: Posterior distribution to which to draw samples from. Is optional but must be given.
        :param outdir: Out directory in which to save the plot. Default is the current working directory.
        :param model_kwargs: Additional keyword arguments to be passed into the model.
        :param kwargs: Additional keyword arguments to pass in the Plotter methods.
        Available in the online documentation under at `redback.plotting.Plotter`.
        `print(Transient.plot_residual.__doc__)` to see all options!
        :return: The axes.
        """
        if self.flux_data:
            plotter = IntegratedFluxPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        elif self.luminosity_data:
            plotter = LuminosityPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        else:
            raise ValueError("Residual plotting not implemented for this data mode")
        return plotter.plot_residuals(axes=axes, save=save, show=show)

    def plot_multiband_lightcurve(
            self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: object) -> matplotlib.axes.Axes:
        """
        :param model: The model used to plot the lightcurve.
        :param filename: The output filename. Otherwise, use default which starts with the name
                         attribute and ends with *lightcurve.png.
        :param axes: Axes to plot in if given.
        :param save:Whether to save the plot.
        :param show: Whether to show the plot.
        :param random_models: Number of random posterior samples plotted faintly. (Default value = 100)
        :param posterior: Posterior distribution to which to draw samples from. Is optional but must be given.
        :param outdir: Out directory in which to save the plot. Default is the current working directory.
        :param model_kwargs: Additional keyword arguments to be passed into the model.
        :param kwargs: Additional keyword arguments to pass in the Plotter methods.
        Available in the online documentation under at `redback.plotting.Plotter`.
        `print(Transient.plot_multiband_lightcurve.__doc__)` to see all options!

        :return: The axes.
        """
        if self.data_mode not in ['flux_density', 'magnitude', 'flux']:
            raise ValueError(
                f'You cannot plot multiband data with {self.data_mode} data mode . Why are you doing this?')
        if self.magnitude_data:
            plotter = MagnitudePlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.flux_data:
            plotter = IntegratedFluxOpticalPlotter(transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        else:
            return
        return plotter.plot_multiband_lightcurve(axes=axes, save=save, show=show)

    _formatted_kwargs_options = redback.plotting.Plotter.keyword_docstring
    plot_data.__doc__ = plot_data.__doc__.replace(
        "`print(Transient.plot_data.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_multiband.__doc__ = plot_multiband.__doc__.replace(
        "`print(Transient.plot_multiband.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_lightcurve.__doc__ = plot_lightcurve.__doc__.replace(
        "`print(Transient.plot_lightcurve.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_multiband_lightcurve.__doc__ = plot_multiband_lightcurve.__doc__.replace(
        "`print(Transient.plot_multiband_lightcurve.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_residual.__doc__ = plot_residual.__doc__.replace(
        "`print(Transient.plot_residual.__doc__)` to see all options!", _formatted_kwargs_options)


class OpticalTransient(Transient):
    DATA_MODES = ['flux', 'flux_density', 'magnitude', 'luminosity']

    @staticmethod
    def load_data(processed_file_path, data_mode="magnitude"):
        """Loads data from specified directory and file, and returns it as a tuple.

        :param processed_file_path: Path to the processed file to load
        :type processed_file_path: str
        :param data_mode: Name of the data mode.
                          Must be from ['magnitude', 'flux_density', 'all']. Default is magnitude.
        :type data_mode: str, optional

        :return: Six elements when querying magnitude or flux_density data, Eight for 'all'
        :rtype: tuple
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
        flux = np.array(df["flux(erg/cm2/s)"])
        flux_err = np.array(df['flux_error'])
        if data_mode == "magnitude":
            return time_days, time_mjd, magnitude, magnitude_err, bands, system
        elif data_mode == "flux_density":
            return time_days, time_mjd, flux_density, flux_density_err, bands, system
        elif data_mode == "flux":
            return time_days, time_mjd, flux, flux_err, bands, system
        elif data_mode == "all":
            return time_days, time_mjd, flux_density, flux_density_err, \
                   magnitude, magnitude_err, flux, flux_err, bands, system

    def __init__(
            self, name: str, data_mode: str = 'magnitude', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux: np.ndarray = None, flux_err: np.ndarray = None, flux_density: np.ndarray = None,
            flux_density_err: np.ndarray = None, magnitude: np.ndarray = None, magnitude_err: np.ndarray = None,
            redshift: float = np.nan, photon_index: float = np.nan, frequency: np.ndarray = None,
            bands: np.ndarray = None, system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False, optical_data:bool = True, **kwargs: None) -> None:
        """This is a general constructor for the Transient class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

        :param name: Name of the transient.
        :type name: str
        :param data_mode: Data mode. Must be one from `OpticalTransient.DATA_MODES`.
        :type data_mode: str, optional
        :param time: Times in the observer frame.
        :type time: np.ndarray, optional
        :param time_err: Time errors in the observer frame.
        :type time_err: np.ndarray, optional
        :param time_mjd: Times in MJD. Used if using phase model.
        :type time_mjd: np.ndarray, optional
        :param time_mjd_err: Time errors in MJD. Used if using phase model.
        :type time_mjd_err: np.ndarray, optional
        :param time_rest_frame: Times in the rest frame. Used for luminosity data.
        :type time_rest_frame: np.ndarray, optional
        :param time_rest_frame_err: Time errors in the rest frame. Used for luminosity data.
        :type time_rest_frame_err: np.ndarray, optional
        :param Lum50: Luminosity values.
        :type Lum50: np.ndarray, optional
        :param Lum50_err: Luminosity error values.
        :type Lum50_err: np.ndarray, optional
        :param flux: Flux values.
        :type flux: np.ndarray, optional
        :param flux_err: Flux error values.
        :type flux_err: np.ndarray, optional
        :param flux_density: Flux density values.
        :type flux_density: np.ndarray, optional
        :param flux_density_err: Flux density error values.
        :type flux_density_err: np.ndarray, optional
        :param magnitude: Magnitude values for photometry data.
        :type magnitude: np.ndarray, optional
        :param magnitude_err: Magnitude error values for photometry data.
        :type magnitude_err: np.ndarray, optional
        :param redshift: Redshift value.
        :type redshift: float, optional
        :param photon_index: Photon index value.
        :type photon_index: float, optional
        :param frequency: Array of band frequencies in photometry data.
        :type frequency: np.ndarray, optional
        :param bands: Band values.
        :type bands: np.ndarray, optional
        :param system: System values.
        :type system: np.ndarray, optional
        :param active_bands: List or array of active bands to be used in the analysis.
                             Use all available bands if 'all' is given.
        :type active_bands: Union[list, np.ndarray], optional
        :param use_phase_model: Whether we are using a phase model.
        :type use_phase_model: bool, optional
        :param optical_data: Whether we are fitting optical data, useful for plotting.
        :type optical_data: bool, optional
        :param kwargs:
            Additional callables:
            bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use
                                  redback.utils.bands_to_frequency if not given.
        :type kwargs: dict, optional
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, frequency=frequency,
                         time_rest_frame_err=time_rest_frame_err, Lum50=Lum50, Lum50_err=Lum50_err,
                         flux=flux, flux_err=flux_err, redshift=redshift, photon_index=photon_index,
                         flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                         magnitude_err=magnitude_err, data_mode=data_mode, name=name,
                         use_phase_model=use_phase_model, optical_data=optical_data, system=system, bands=bands,
                         active_bands=active_bands, **kwargs)
        self.directory_structure = None

    @classmethod
    def from_open_access_catalogue(
            cls, name: str, data_mode: str = "magnitude", active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False) -> OpticalTransient:
        """Constructor method to built object from Open Access Catalogue

        :param name: Name of the transient.
        :type name: str
        :param data_mode: Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
        :type data_mode: str, optional
        :param active_bands:
            Sets active bands based on array given.
            If argument is 'all', all unique bands in `self.bands` will be used.
        :type active_bands: Union[np.ndarray, str]
        :param use_phase_model: Whether to use a phase model.
        :type use_phase_model: bool, optional

        :return: A class instance
        :rtype: OpticalTransient
        """
        if cls.__name__ == "TDE":
            transient_type = "tidal_disruption_event"
        else:
            transient_type = cls.__name__.lower()
        directory_structure = redback.get_data.directory.open_access_directory_structure(
            transient=name, transient_type=transient_type)
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, flux, flux_err, bands, system = \
            cls.load_data(processed_file_path=directory_structure.processed_file_path, data_mode="all")
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, bands=bands, system=system, active_bands=active_bands,
                   use_phase_model=use_phase_model, optical_data=True, flux=flux, flux_err=flux_err)

    @property
    def event_table(self) -> str:
        """
        :return: Path to the metadata table.
        :rtype: str
        """
        return f"{self.directory_structure.directory_path}/{self.name}_metadata.csv"

    def _set_data(self) -> None:
        """Sets the metadata from the event table."""
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
        :return: The transient directory given the name of the transient.
        :rtype: str
        """
        return self._get_transient_dir()

    def _get_transient_dir(self) -> str:
        """

        :return: The transient directory path
        :rtype: str
        """
        transient_dir, _, _ = redback.get_data.directory.open_access_directory_structure(
            transient=self.name, transient_type=self.__class__.__name__.lower())
        return transient_dir
