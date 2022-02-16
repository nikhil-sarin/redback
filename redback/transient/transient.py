from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os.path import join
import pandas as pd
from typing import Union

import redback


class Transient(object):
    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'photometry', 'counts', 'ttes']
    _ATTRIBUTE_NAME_DICT = dict(luminosity="Lum50", flux="flux", flux_density="flux_density",
                                counts="counts", photometry="magnitude")

    ylabel_dict = dict(luminosity=r'Luminosity [$10^{50}$ erg s$^{-1}$]',
                       photometry=r'Magnitude',
                       flux=r'Flux [erg cm$^{-2}$ s$^{-1}$]',
                       flux_density=r'Flux density [mJy]',
                       counts=r'Counts')

    luminosity_data = redback.utils.DataModeSwitch('luminosity')
    flux_data = redback.utils.DataModeSwitch('flux')
    flux_density_data = redback.utils.DataModeSwitch('flux_density')
    photometry_data = redback.utils.DataModeSwitch('photometry')
    counts_data = redback.utils.DataModeSwitch('counts')
    tte_data = redback.utils.DataModeSwitch('ttes')

    def __init__(
            self, time: np.ndarray = None, time_err: np.ndarray = None, time_mjd: np.ndarray = None,
            time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None, time_rest_frame_err: np.ndarray = None,
            Lum50: np.ndarray = None, Lum50_err: np.ndarray = None, flux: np.ndarray = None,
            flux_err: np.ndarray = None, flux_density: np.ndarray = None, flux_density_err: np.ndarray = None,
            magnitude: np.ndarray = None, magnitude_err: np.ndarray = None, counts: np.ndarray = None,
            ttes: np.ndarray = None, bin_size: float = None, redshift: float = np.nan, data_mode: str = None,
            name: str = '', path: str = '.', photon_index: float = np.nan, use_phase_model: bool = False,
            frequency: np.ndarray = None, system: np.ndarray = None, bands: np.ndarray = None,
            active_bands: Union[np.ndarray, str] = None, **kwargs: dict) -> None:
        """

        Parameters
        ----------
        time: array_like
        time_err: array_like
        time_mjd: array_like
        time_mjd_err: array_like
        time_rest_frame: array_like
        time_rest_frame_err: array_like
        Lum50: array_like
        Lum50_err: array_like
        flux: array_like
        flux_err: array_like
        flux_density: array_like
        flux_density_err: array_like
        magnitude: array_like
        magnitude_err: array_like
        counts: array_like
        ttes: array_like
        bin_size: float
        redshift: float
        data_mode: str
        name: str
        path: str
        photon_index: float
        use_phase_model: bool
        frequency: array_like
        system: array_like
        bands: list
        active_bands: list
        kwargs: callable
        """
        self.bin_size = bin_size
        self.bin_ttes = kwargs.get("bin_ttes", redback.utils.bin_ttes)
        self.bands_to_frequencies = kwargs.get("bands_to_frequencies", redback.utils.bands_to_frequencies)

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

        self.bands = bands
        self.frequency = frequency
        self.system = system
        self.active_bands = active_bands
        self.data_mode = data_mode
        self.redshift = redshift
        self.name = name
        self.path = path
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
        return getattr(self, self._time_attribute_name)

    @x.setter
    def x(self, x: np.ndarray) -> None:
        setattr(self, self._time_attribute_name, x)

    @property
    def x_err(self) -> np.ndarray:
        return getattr(self, self._time_err_attribute_name)

    @x_err.setter
    def x_err(self, x_err: np.ndarray) -> None:
        setattr(self, self._time_err_attribute_name, x_err)

    @property
    def y(self) -> np.ndarray:
        return getattr(self, self._y_attribute_name)

    @y.setter
    def y(self, y: np.ndarray) -> None:
        setattr(self, self._y_attribute_name, y)

    @property
    def y_err(self) -> np.ndarray:
        return getattr(self, self._y_err_attribute_name)

    @y_err.setter
    def y_err(self, y_err: np.ndarray) -> None:
        setattr(self, self._y_err_attribute_name, y_err)

    @property
    def data_mode(self) -> str:
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    @property
    def xlabel(self) -> str:
        if self.use_phase_model:
            return r"Time [MJD]"
        else:
            return r"Time since burst [days]"

    @property
    def ylabel(self) -> str:
        try:
            return self.ylabel_dict[self.data_mode]
        except KeyError:
            raise ValueError("No data mode specified")

    @property
    def frequency(self) -> np.ndarray:
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: np.ndarray) -> None:
        if frequency is None:
            self._frequency = self.bands_to_frequencies(self.bands)
        else:
            self._frequency = frequency

    @property
    def active_bands(self) -> np.ndarray:
        return self._active_bands

    @active_bands.setter
    def active_bands(self, active_bands: np.ndarray) -> None:
        if active_bands == 'all':
            self._active_bands = np.unique(self.bands)
        else:
            self._active_bands = active_bands

    def get_filtered_data(self) -> tuple:
        if self.flux_density_data or self.photometry_data:
            idxs = [b in self.active_bands for b in self.bands]
            filtered_x = self.x[idxs]
            try:
                filtered_x_err = self.x_err[idxs]
            except TypeError:
                filtered_x_err = None
            filtered_y = self.y[idxs]
            filtered_y_err = self.y_err[idxs]
            return filtered_x, filtered_x_err, filtered_y, filtered_y_err
        else:
            raise ValueError(f"Transient needs to be in flux density or photometry data mode, "
                             f"but is in {self.data_mode} instead.")

    @property
    def unique_bands(self) -> np.ndarray:
        return np.unique(self.bands)

    @property
    def unique_frequencies(self) -> np.ndarray:
        return self.bands_to_frequencies(self.unique_bands)

    @property
    def list_of_band_indices(self) -> list:
        return [np.where(self.bands == np.array(b))[0] for b in self.unique_bands]

    @property
    def default_filters(self) -> list:
        return ["g", "r", "i", "z", "y", "J", "H", "K"]

    @staticmethod
    def get_colors(filters: Union[np.ndarray, list]) -> matplotlib.colors.Colormap:
        return matplotlib.cm.rainbow(np.linspace(0, 1, len(filters)))

    def plot_data(self, axes: matplotlib.axes.Axes = None, colour: str = 'k') -> matplotlib.axes.Axes:
        fig, axes = plt.subplots()
        return axes

    def plot_multiband(self, axes: matplotlib.axes.Axes = None, colour: str = 'k') -> matplotlib.axes.Axes:
        fig, axes = plt.subplots()
        return axes

    def plot_lightcurve(
            self, model: callable, filename: str = None, axes: matplotlib.axes.Axes = None,  plot_save: bool = True, 
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None, outdir: str = '.', 
            model_kwargs: dict = None, **kwargs: object) -> None:
        if filename is None:
            filename = f"{self.name}_lightcurve.png"
        axes = axes or plt.gca()
        axes = self.plot_data(axes=axes)

        times = self._get_times(axes)

        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        ys = model(times, **max_like_params, **model_kwargs)
        axes.plot(times, ys, color='blue', alpha=0.65, lw=2)

        for _ in range(random_models):
            params = posterior.iloc[np.random.randint(len(posterior))]
            ys = model(times, **params, **model_kwargs)
            axes.plot(times, ys, color='red', alpha=0.05, lw=2, zorder=-1)
        plt.savefig(join(outdir, filename), dpi=300, bbox_inches="tight")
        plt.clf()

    def plot_multiband_lightcurve(
            self, model: callable, filename: str = None, axes: matplotlib.axes.Axes = None, plot_save: bool = True, 
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None, outdir: str = '.', 
            model_kwargs: dict = None, **kwargs: object) -> None:
        if self.luminosity_data or self.flux_data:
            redback.utils.logger.warning(f"Plotting multiband lightcurve not possible for {self.data_mode}. Returning.")
            return

        if filename is None:
            filename = f"{self.name}_multiband_lightcurve.png"
        axes = axes or plt.gca()
        axes = self.plot_multiband(axes=axes)

        times = self._get_times(axes)

        times_mesh, frequency_mesh = np.meshgrid(times, self.unique_frequencies)
        model_kwargs['frequency'] = frequency_mesh
        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        ys = model(times_mesh, **max_like_params, **model_kwargs)

        for i in range(len(self.unique_frequencies)):
            axes[i].plot(times_mesh[i], ys[i], color='blue', alpha=0.65, lw=2)
            params = posterior.iloc[np.random.randint(len(posterior))]
            ys = model(times_mesh, **params, **model_kwargs)
            for _ in range(random_models):
                axes.plot(times, ys[i], color='red', alpha=0.05, lw=2, zorder=-1)
        if plot_save:
            plt.savefig(join(outdir, filename), dpi=300, bbox_inches="tight")
        if plot_show:
            plt.show()
        plt.clf()

    def _get_times(self, axes: matplotlib.axes.Axes) -> np.ndarray:
        if axes.get_yscale == 'linear':
            times = np.linspace(self.x[0], self.x[-1], 200)
        else:
            times = np.exp(np.linspace(np.log(self.x[0]), np.log(self.x[-1]), 200))
        return times


class OpticalTransient(Transient):
    DATA_MODES = ['flux', 'flux_density', 'photometry', 'luminosity']

    @staticmethod
    def load_data(name, data_mode='photometry', transient_dir="."):
        filename = f"{name}_data.csv"

        data_file = join(transient_dir, filename)
        df = pd.read_csv(data_file)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        system = np.array(df["system"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        if data_mode == "photometry":
            return time_days, time_mjd, magnitude, magnitude_err, bands, system
        elif data_mode == "flux_density":
            return time_days, time_mjd, flux_density, flux_density_err, bands, system
        elif data_mode == "all":
            return time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands, system

    def __init__(
            self, name: str, data_mode: str = 'photometry', time: np.ndarray = None, time_err: np.ndarray = None, 
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None, 
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux_density: np.ndarray = None, flux_density_err: np.ndarray = None, magnitude: np.ndarray = None, 
            magnitude_err: np.ndarray = None, frequency: np.ndarray = None, bands: np.ndarray = None, 
            system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all', use_phase_model: bool = False, 
            **kwargs: dict) -> None:
        """

        Parameters
        ----------
        name
        data_mode
        time
        time_err
        time_mjd
        time_mjd_err
        time_rest_frame
        time_rest_frame_err
        Lum50
        Lum50_err
        flux_density
        flux_density_err
        magnitude
        magnitude_err
        frequency
        bands
        system
        active_bands
        use_phase_model
        kwargs
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, frequency=frequency,
                         time_rest_frame_err=time_rest_frame_err, Lum50=Lum50, Lum50_err=Lum50_err,
                         flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                         magnitude_err=magnitude_err, data_mode=data_mode, name=name,
                         use_phase_model=use_phase_model, system=system, bands=bands, active_bands=active_bands,
                         **kwargs)
        self._set_data()

    @classmethod
    def from_open_access_catalogue(
            cls, name: str, data_mode: str = "photometry", active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False) -> OpticalTransient:
        transient_dir = cls._get_transient_dir(name=name)
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands, system = \
            cls.load_data(name=name, transient_dir=transient_dir, data_mode="all")
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, bands=bands, system=system, active_bands=active_bands,
                   use_phase_model=use_phase_model)

    @property
    def event_table(self) -> str:
        return f'{self.__class__.__name__.lower()}/{self.name}/metadata.csv'

    def _set_data(self) -> None:
        try:
            meta_data = pd.read_csv(self.event_table, error_bad_lines=False, delimiter=',', dtype='str')
        except FileNotFoundError as e:
            redback.utils.logger.warning(e)
            redback.utils.logger.warning("Setting metadata to None")
            meta_data = None
        self.meta_data = meta_data

    @property
    def transient_dir(self) -> str:
        return self._get_transient_dir(name=self.name)

    @classmethod
    def _get_transient_dir(cls, name: str) -> str:
        transient_dir, _, _ = redback.getdata.transient_directory_structure(
            transient=name, transient_type=cls.__name__.lower())
        return transient_dir

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, filters: np.ndarray = None, plot_others: bool = True, 
            **plot_kwargs: dict) -> None:
        """
        plots the data
        :param axes:
        :param filters:
        :param plot_others:
        """
        errorbar_fmt = plot_kwargs.get("errorbar_fmt", "x")
        colors = plot_kwargs.get("colors", self.get_colors(filters))
        xlabel = plot_kwargs.get("xlabel", self.xlabel)
        ylabel = plot_kwargs.get("ylabel", self.ylabel)
        plot_label = plot_kwargs.get("plot_label", "lc")

        if filters is None:
            filters = self.default_filters

        ax = axes or plt.gca()
        for idxs, band in zip(self.list_of_band_indices, self.unique_bands):
            x_err = self.x_err[idxs] if self is not None else self.x_err
            if band in filters:
                color = colors[filters.index(band)]
                label = band
            elif plot_others:
                color = "black"
                label = None
            else:
                continue
            ax.errorbar(self.x[idxs], self.y[idxs], xerr=x_err, yerr=self.y_err[idxs],
                        fmt=errorbar_fmt, ms=1, color=color, elinewidth=2, capsize=0., label=label)

        ax.set_xlim(0.5 * self.x[0], 1.2 * self.x[-1])
        if self.photometry_data:
            ax.set_ylim(0.8 * min(self.y), 1.2 * np.max(self.y))
            ax.invert_yaxis()
        else:
            ax.set_ylim(0.5 * min(self.y), 2. * np.max(self.y))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', pad=10)
        ax.legend(ncol=2)

        if axes is None:
            plt.tight_layout()

        filename = f"{self.name}_{self.data_mode}_{plot_label}.png"
        plt.savefig(join(self.transient_dir, filename))
        plt.clf()

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, ncols: int = 2,
            nrows: int = None, figsize: tuple = None, filters: np.ndarray = None, **plot_kwargs: dict) \
            -> matplotlib.axes.Axes:
        if self.luminosity_data or self.flux_data:
            redback.utils.logger.warning(f"Can't plot multiband for {self.data_mode} data.")
            return

        if filters is None:
            filters = self.active_bands
        elif filters == 'default':
            filters = self.default_filters

        wspace = plot_kwargs.get("wspace", 0.15)
        hspace = plot_kwargs.get("hspace", 0.04)
        fontsize = plot_kwargs.get("fontsize", 30)
        errorbar_fmt = plot_kwargs.get("errorbar_fmt", "x")
        colors = plot_kwargs.get("colors", self.get_colors(filters))
        xlabel = plot_kwargs.get("xlabel", self.xlabel)
        ylabel = plot_kwargs.get("ylabel", self.ylabel)
        plot_label = plot_kwargs.get("plot_label", "multiband_lc")

        if figure is None or axes is None:
            if nrows is None:
                nrows = int(np.ceil(len(filters) / 2))
            npanels = ncols * nrows
            if npanels < len(filters):
                raise ValueError(f"Insufficient number of panels. {npanels} panels were given "
                                 f"but {len(filters)} panels are needed.")
            if figsize is None:
                figsize = (4 * ncols, 4 * nrows)
            figure, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', sharey='all', figsize=figsize)

        axes = axes.ravel()

        i = 0
        for idxs, band in zip(self.list_of_band_indices, self.unique_bands):
            if band not in filters:
                continue

            x_err = self.x_err[idxs] if self.x_err is not None else self.x_err

            color = colors[filters.index(band)]

            freq = self.bands_to_frequencies([band])
            if 1e10 < freq < 1e15:
                label = band
            else:
                label = freq
            axes[i].errorbar(self.x[idxs], self.y[idxs], xerr=x_err, yerr=self.y_err[idxs],
                             fmt=errorbar_fmt, ms=1, color=color, elinewidth=2, capsize=0., label=label)

            axes[i].set_xlim(0.5 * self.x[idxs][0], 1.2 * self.x[idxs][-1])
            if self.photometry_data:
                axes[i].set_ylim(0.8 * min(self.y[idxs]), 1.2 * np.max(self.y[idxs]))
                axes[i].invert_yaxis()
            else:
                axes[i].set_ylim(0.5 * min(self.y[idxs]), 2. * np.max(self.y[idxs]))
                axes[i].set_yscale("log")
            axes[i].legend(ncol=2)
            axes[i].tick_params(axis='both', which='major', pad=8)
            i += 1

        figure.supxlabel(xlabel, fontsize=fontsize)
        figure.supylabel(ylabel, fontsize=fontsize)
        filename = f"{self.name}_{self.data_mode}_{plot_label}.png"
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(join(self.transient_dir, filename), bbox_inches="tight")
        return axes
