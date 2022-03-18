from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback
from redback.utils import KwargsAccessorWithDefault


class _FilenameGetter(object):
    def __init__(self, suffix):
        self.suffix = suffix

    def __get__(self, instance, owner):
        return instance._get_filename(default=f"{instance.transient.name}_{self.suffix}.png")

    def __set__(self, instance, value):
        pass


class _FilePathGetter(object):

    def __init__(self, directory_property, filename_property):
        self.directory_property = directory_property
        self.filename_property = filename_property

    def __get__(self, instance, owner):
        return join(getattr(instance, self.directory_property), getattr(instance, self.filename_property))


class Plotter(object):

    capsize = KwargsAccessorWithDefault("capsize", 0.)
    color = KwargsAccessorWithDefault("color", "k")
    dpi = KwargsAccessorWithDefault("dpi", 300)
    elinewidth = KwargsAccessorWithDefault("elinewidth", 2)
    errorbar_fmt = KwargsAccessorWithDefault("errorbar_fmt", "x")
    ms = KwargsAccessorWithDefault("ms", 1)
    x_axis_tick_params_pad = KwargsAccessorWithDefault("x_axis_tick_params_pad", 10)
    model = KwargsAccessorWithDefault("model", None)

    max_likelihood_alpha = KwargsAccessorWithDefault("max_likelihood_alpha", 0.65)
    random_sample_alpha = KwargsAccessorWithDefault("random_sample_alpha", 0.05)
    max_likelihood_color = KwargsAccessorWithDefault("max_likelihood_color", "blue")
    random_sample_color = KwargsAccessorWithDefault("random_sample_color", "red")

    bbox_inches = KwargsAccessorWithDefault("bbox_inches", "tight")
    linewidth = KwargsAccessorWithDefault("linewidth", 2)
    zorder = KwargsAccessorWithDefault("zorder", -1)

    xy = KwargsAccessorWithDefault("xy", (0.95, 0.9))
    xycoords = KwargsAccessorWithDefault("xycoords", "axes fraction")
    horizontalalignment = KwargsAccessorWithDefault("horizontalalignment", "right")
    annotation_size = KwargsAccessorWithDefault("annotation_size", 20)

    wspace = KwargsAccessorWithDefault("wspace", 0.15)
    hspace = KwargsAccessorWithDefault("hspace", 0.04)
    fontsize = KwargsAccessorWithDefault("fontsize", 30)

    random_models = KwargsAccessorWithDefault("random_models", 100)
    plot_others = KwargsAccessorWithDefault("plot_others", True)

    def __init__(self, transient, **kwargs):
        self.transient = transient
        self.kwargs = kwargs
        self._posterior_sorted = False

    def _get_times(self, axes: matplotlib.axes.Axes) -> np.ndarray:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes
            The axes used in the plotting procedure.
        Returns
        -------
        np.ndarray: Linearly or logarithmically scaled time values depending on the y scale used in the plot.

        """
        if isinstance(axes, np.ndarray):
            ax = axes[0]
        else:
            ax = axes

        if ax.get_yscale() == 'linear':
            times = np.linspace(self.xlim_low, self.xlim_high, 200)
        else:
            times = np.exp(np.linspace(np.log(self.xlim_low), np.log(self.xlim_high), 200))
        return times

    @property
    def xlim_low(self):
        default = 0.5 * self.transient.x[0]
        if default == 0:
            default += 1e-3
        return self.kwargs.get("xlim_low", default)

    @property
    def xlim_high(self):
        if self.x_err is None:
            default = 2 * self.transient.x[-1]
        else:
            default = 2 * (self.transient.x[-1] + self.x_err[1][-1])
        return self.kwargs.get("xlim_high", default)

    @property
    def ylim_low(self):
        return 0.5 * min(self.transient.y)

    @property
    def ylim_high(self):
        return 2. * np.max(self.transient.y)

    @property
    def x_err(self):
        if self.transient.x_err is not None:
            return [np.abs(self.transient.x_err[1, :]), self.transient.x_err[0, :]]
        else:
            return None

    @property
    def y_err(self):
        return [np.abs(self.transient.y_err[1, :]), self.transient.y_err[0, :]]

    @property
    def lightcurve_plot_outdir(self):
        return self._get_outdir(join(self.transient.directory_structure.directory_path, self.model.__name__))

    @property
    def data_plot_outdir(self):
        return self._get_outdir(self.transient.directory_structure.directory_path)

    def _get_outdir(self, default):
        return self._get_kwarg_with_default(kwarg="outdir", default=default)

    def _get_filename(self, default):
        return self._get_kwarg_with_default(kwarg="filename", default=default)

    def _get_kwarg_with_default(self, kwarg, default):
        return self.kwargs.get(kwarg, default) or default

    @property
    def model_kwargs(self):
        return self._get_kwarg_with_default("model_kwargs", dict())

    @property
    def posterior(self):
        posterior = self.kwargs.get("posterior")
        if not self._posterior_sorted and posterior is not None:
            posterior.sort_values(by='log_likelihood')
            self._posterior_sorted = True
        return posterior

    @property
    def max_like_params(self):
        return self.posterior.iloc[-1]

    def get_random_parameters(self):
        return [self.posterior.iloc[np.random.randint(len(self.posterior))] for _ in range(self.random_models)]

    data_plot_filename = _FilenameGetter(suffix="data")
    lightcurve_plot_filename = _FilenameGetter(suffix="lightcurve")
    multiband_data_plot_filename = _FilenameGetter(suffix="multiband_data")
    multiband_lightcurve_plot_filename = _FilenameGetter(suffix="multiband_lightcurve")

    data_plot_filepath = _FilePathGetter(
        directory_property="data_plot_outdir", filename_property="data_plot_filename")
    lightcurve_plot_filepath = _FilePathGetter(
        directory_property="lightcurve_plot_outdir", filename_property="lightcurve_plot_filename")
    multiband_data_plot_filepath = _FilePathGetter(
        directory_property="data_plot_outdir", filename_property="multiband_data_plot_filename")
    multiband_lightcurve_plot_filepath = _FilePathGetter(
        directory_property="lightcurve_plot_outdir", filename_property="multiband_lightcurve_plot_filename")

    def _save_and_show(self, filepath, save, show):
        plt.tight_layout()
        if save:
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        if show:
            plt.show()


class IntegratedFluxPlotter(Plotter):

    @property
    def xlabel(self):
        return r"Time since burst [s]"

    @property
    def ylabel(self):
        return self.transient.ylabel

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """
        Plots the Afterglow lightcurve and returns Axes.

        Parameters
        ----------
        axes : Union[matplotlib.axes.Axes, None], optional
            Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        kwargs: dict
            Additional keyword arguments.

        Returns
        ----------
        matplotlib.axes.Axes: The axes with the plot.
        """
        ax = axes or plt.gca()

        ax.errorbar(self.transient.x, self.transient.y, xerr=self.x_err, yerr=self.y_err,
                    fmt=self.errorbar_fmt, c=self.color, ms=self.ms, elinewidth=self.elinewidth, capsize=self.capsize)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(self.xlim_low, self.xlim_high)
        ax.set_ylim(self.ylim_low, self.ylim_high)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.annotate(
            self.transient.name, xy=self.xy, xycoords=self.xycoords,
            horizontalalignment=self.horizontalalignment, size=self.annotation_size)

        ax.tick_params(axis='x', pad=self.x_axis_tick_params_pad)

        self._save_and_show(filepath=self.data_plot_filepath, save=save, show=show)
        return ax

    def plot_lightcurve(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> None:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        save: bool, optional
            Whether to save the plot.
        show: bool, optional
            Whether to show the plot.
        kwargs: dict
            No current function.
        """
        axes = axes or plt.gca()

        axes = self.plot_data(axes=axes, save=False, show=False)
        times = self._get_times(axes)

        self._plot_lightcurves(axes, times)

        self._save_and_show(filepath=self.lightcurve_plot_filepath, save=save, show=show)
        return axes

    def _plot_lightcurves(self, axes, times):
        ys = self.model(times, **self.max_like_params, **self.model_kwargs)
        axes.plot(times, ys, color=self.max_likelihood_color, alpha=self.max_likelihood_alpha, lw=self.linewidth)
        for params in self.get_random_parameters():
            self._plot_single_lightcurve(axes=axes, times=times, params=params)

    def _plot_single_lightcurve(self, axes, times, params):
        ys = self.model(times, **params, **self.model_kwargs)
        axes.plot(times, ys, color=self.random_sample_color, alpha=self.random_sample_alpha, lw=self.linewidth,
                  zorder=self.zorder, **self.kwargs)


class LuminosityPlotter(IntegratedFluxPlotter):
    pass


class MagnitudePlotter(Plotter):

    @property
    def colors(self):
        return self.kwargs.get("colors", self.transient.get_colors(self.filters))

    @property
    def xlabel(self):
        if self.transient.use_phase_model:
            default = f"Time since {self.reference_mjd_date} MJD [days]"
        else:
            default = self.transient.xlabel
        return self.kwargs.get("xlabel", default)

    @property
    def ylabel(self):
        return self.kwargs.get("ylabel", self.transient.ylabel)

    @property
    def xlim_low(self):
        if self.transient.use_phase_model:
            default = (self.transient.x[0] - self.reference_mjd_date) * 0.9
        else:
            default = 0.5 * self.transient.x[0]
        return self.kwargs.get("xlim_low", default)

    @property
    def xlim_high(self):
        if self.transient.use_phase_model:
            default = (self.transient.x[-1] - self.reference_mjd_date) * 1.1
        else:
            default = 1.2 * self.transient.x[-1]
        return self.kwargs.get("xlim_high", default)

    def _get_x_err(self, indices):
        return self.transient.x_err[indices] if self.transient.x_err is not None else self.transient.x_err

    def _set_y_axis_data(self, ax):
        ax.set_ylim(0.8 * min(self.transient.y), 1.2 * np.max(self.transient.y))
        ax.invert_yaxis()

    def _set_y_axis_multiband_data(self, axis, indices):
        if self.transient.magnitude_data:
            axis.set_ylim(0.8 * min(self.transient.y[indices]), 1.2 * np.max(self.transient.y[indices]))
            axis.invert_yaxis()
        else:
            axis.set_ylim(0.5 * min(self.transient.y[indices]), 2. * np.max(self.transient.y[indices]))
            axis.set_yscale("log")

    ncols = KwargsAccessorWithDefault("ncols", 2)

    def _set_xaxis(self, axes):
        if self.transient.use_phase_model:
            axes.set_xscale("log")
        axes.set_xlim(self.xlim_low, self.xlim_high)

    @property
    def nrows(self):
        default = int(np.ceil(len(self.filters) / 2))
        return self._get_kwarg_with_default("nrows", default=default)

    @property
    def npanels(self):
        npanels = self.nrows * self.ncols
        if npanels < len(self.filters):
            raise ValueError(f"Insufficient number of panels. {npanels} panels were given "
                             f"but {len(self.filters)} panels are needed.")
        return npanels

    @property
    def figsize(self):
        default = (4 + 4 * self.ncols, 2 + 2 * self.nrows)
        return self._get_kwarg_with_default("figsize", default=default)

    @property
    def reference_mjd_date(self):
        if self.transient.use_phase_model:
            return self.kwargs.get("reference_mjd_date", int(self.transient.x[0]))
        return 0

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> None:
        """
        Plots the data.

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes can be given if defaults are not satisfying
        plot_kwargs:
            Additional optional plotting kwargs:
            errorbar_fmt: Errorbar format ('fmt' argument in matplotlib.pyplot.errorbar)
            colors: colors to be used for the bands
            xlabel: Plot xlabel
            ylabel: Plot ylabel
            plot_label: Additional filename label appended to the default name
        """
        ax = axes or plt.gca()

        for indices, band in zip(self.transient.list_of_band_indices, self.transient.unique_bands):
            if band in self.filters:
                color = self.colors[list(self.filters).index(band)]
                label = band
            elif self.plot_others:
                color = "black"
                label = None
            else:
                continue
            if isinstance(label, float):
                label = f"{label:.2e}"
            ax.errorbar(
                self.transient.x[indices] - self.reference_mjd_date, self.transient.y[indices],
                xerr=self._get_x_err(indices), yerr=self.transient.y_err[indices],
                fmt=self.errorbar_fmt, ms=self.ms, color=color,
                elinewidth=self.elinewidth, capsize=self.capsize, label=label)

        self._set_xaxis(axes=ax)
        self._set_y_axis_data(ax)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.tick_params(axis='x', pad=self.x_axis_tick_params_pad)
        ax.legend(ncol=2, loc='best')

        self._save_and_show(filepath=self.data_plot_filepath, save=save, show=show)
        return ax

    def plot_lightcurve(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True)\
            -> None:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        save: bool, optional
            Whether to save the plot.
        show: bool, optional
            Whether to show the plot.
        kwargs: dict
            No current function.
        """
        axes = axes or plt.gca()

        axes = self.plot_data(axes=axes, save=False, show=False)
        axes.set_yscale('log')

        times = self._get_times(axes)

        random_params = self.get_random_parameters()

        for band, color in zip(self.transient.active_bands, self.transient.get_colors(self.transient.active_bands)):
            frequency = redback.utils.bands_to_frequency([band])
            self.model_kwargs["frequency"] = np.ones(len(times)) * frequency
            ys = self.model(times, **self.max_like_params, **self.model_kwargs)
            axes.plot(times - self.reference_mjd_date, ys, color=color, alpha=0.65, lw=2)

            for params in random_params:
                ys = self.model(times, **params, **self.model_kwargs)
                axes.plot(times - self.reference_mjd_date, ys, color='red', alpha=0.05, lw=2, zorder=-1)

        self._save_and_show(filepath=self.lightcurve_plot_filepath, save=save, show=show)
        return axes

    def _check_valid_multiband_data_mode(self):
        if self.transient.luminosity_data or self.transient.flux_data:
            redback.utils.logger.warning(
                f"Plotting multiband lightcurve/data not possible for {self.transient.data_mode}. Returning.")
            return False
        return True

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, save: bool = True,
            show: bool = True) -> matplotlib.axes.Axes:
        """

        Parameters
        ----------
        figure: matplotlib.figure.Figure, optional
            Figure can be given if defaults are not satisfying
        axes: matplotlib.axes.Axes, optional
            Axes can be given if defaults are not satisfying
        plot_kwargs:
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
        if not self._check_valid_multiband_data_mode():
            return

        if figure is None or axes is None:
            figure, axes = plt.subplots(ncols=self.ncols, nrows=self.nrows, sharex='all', figsize=self.figsize)

        axes = axes.ravel()

        i = 0
        for indices, band, freq in zip(
                self.transient.list_of_band_indices, self.transient.unique_bands, self.transient.unique_frequencies):
            if band not in self.filters:
                continue

            x_err = self._get_x_err(indices)
            color = self.colors[list(self.filters).index(band)]

            label = self._get_multiband_plot_label(band, freq)
            axes[i].errorbar(
                self.transient.x[indices] - self.reference_mjd_date, self.transient.y[indices], xerr=x_err,
                yerr=self.transient.y_err[indices], fmt=self.errorbar_fmt, ms=self.ms, color=color,
                elinewidth=self.elinewidth, capsize=self.capsize,
                label=label)

            self._set_xaxis(axes[i])
            self._set_y_axis_multiband_data(axes[i], indices)
            axes[i].legend(ncol=2)
            axes[i].tick_params(axis='both', which='major', pad=8)
            i += 1

        figure.supxlabel(self.xlabel, fontsize=self.fontsize)
        figure.supylabel(self.ylabel, fontsize=self.fontsize)
        plt.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

        self._save_and_show(filepath=self.multiband_data_plot_filepath, save=save, show=show)
        return axes

    @staticmethod
    def _get_multiband_plot_label(band, freq):
        if isinstance(band, str):
            if 1e10 < freq < 1e15:
                label = band
            else:
                label = freq
        else:
            label = f"{band:.2e}"
        return label

    @property
    def filters(self):
        filters = self.kwargs.get("filters", self.transient.active_bands)
        if filters is None:
            return self.transient.active_bands
        elif str(filters) == 'default':
            return self.transient.default_filters
        return filters

    def plot_multiband_lightcurve(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> None:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        save: bool, optional
            Whether to save the plot.
        show: bool, optional
            Whether to show the plot.
        -------

        """
        if not self._check_valid_multiband_data_mode():
            return

        axes = axes or plt.gca()
        axes = self.plot_multiband(axes=axes, save=False, show=False)

        times = self._get_times(axes)
        times_mesh, frequency_mesh = np.meshgrid(times, self.transient.bands_to_frequency(self.filters))
        new_model_kwargs = self.model_kwargs.copy()
        new_model_kwargs['frequency'] = frequency_mesh

        ys = self.model(times_mesh, **self.max_like_params, **new_model_kwargs)

        random_ys_list = [self.model(times_mesh, **random_params, **new_model_kwargs)
                          for random_params in self.get_random_parameters()]

        for i in range(len(ys)):
            axes[i].plot(
                times - self.reference_mjd_date, ys[i], color=self.max_likelihood_color, alpha=self.max_likelihood_alpha, lw=self.linewidth)
            for random_ys in random_ys_list:
                axes[i].plot(
                    times - self.reference_mjd_date, random_ys[i], color=self.random_sample_color,
                    alpha=self.random_sample_alpha, lw=self.linewidth, zorder=self.zorder)

        self._save_and_show(filepath=self.multiband_lightcurve_plot_filepath, save=save, show=show)
        return axes


class FluxDensityPlotter(MagnitudePlotter):

    def _set_y_axis_data(self, ax):
        ax.set_ylim(0.5 * min(self.transient.y), 2. * np.max(self.transient.y))
        ax.set_yscale('log')
