from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback
from redback.utils import KwargsAccessorWithDefault


class Plotter(object):

    capsize = KwargsAccessorWithDefault("capsize", 0.)
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

    linewidth = KwargsAccessorWithDefault("linewidth", 2)
    zorder = KwargsAccessorWithDefault("zorder", -1)

    def __init__(self, transient, **kwargs):
        self.transient = transient
        self.kwargs = kwargs

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
        xlim_low = 0.5 * self.transient.x[0]
        if xlim_low == 0:
            xlim_low += 1e-3
        return xlim_low

    @property
    def xlim_high(self):
        if self.x_err is None:
            return 2 * self.transient.x[-1]
        return 2 * (self.transient.x[-1] + self.x_err[1][-1])

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
        val = self.kwargs.get(kwarg, default)
        return val or default

    @property
    def data_plot_filename(self):
        return self._get_filename(default=f"{self.transient.name}_data.png")

    @property
    def lightcurve_plot_filename(self):
        return self._get_filename(default=f"{self.transient.name}_lightcurve.png")

    @property
    def multiband_data_plot_filename(self):
        return self._get_filename(default=f"{self.transient.name}_multiband_data.png")

    @property
    def multiband_lightcurve_plot_filename(self):
        return self._get_filename(default=f"{self.transient.name}_multiband_lightcurve.png")

    @property
    def data_plot_filepath(self):
        return join(self.data_plot_outdir, self.data_plot_filename)

    @property
    def lightcurve_plot_filepath(self):
        return join(self.lightcurve_plot_outdir, self.lightcurve_plot_filename)

    @property
    def multiband_data_plot_filepath(self):
        return join(self.data_plot_outdir, self.multiband_data_plot_filename)

    @property
    def multiband_lightcurve_plot_filepath(self):
        return join(self.lightcurve_plot_outdir, self.multiband_lightcurve_plot_filename)


class IntegratedFluxPlotter(Plotter):

    xy = KwargsAccessorWithDefault("xy", (0.95, 0.9))
    xycoords = KwargsAccessorWithDefault("xycoords", "axes fraction")
    horizontalalignment = KwargsAccessorWithDefault("horizontalalignment", "right")
    annotation_size = KwargsAccessorWithDefault("annotation_size", 20)

    @property
    def xlabel(self):
        return r"Time since burst [s]"

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, color: str = 'k', plot_save: bool = True,
            plot_show: bool = True) -> matplotlib.axes.Axes:
        """
        Plots the Afterglow lightcurve and returns Axes.

        Parameters
        ----------
        axes : Union[matplotlib.axes.Axes, None], optional
            Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        color: str, optional
            color of the data.
        kwargs: dict
            Additional keyword arguments.

        Returns
        ----------
        matplotlib.axes.Axes: The axes with the plot.
        """

        ax = axes or plt.gca()
        ax.errorbar(self.transient.x, self.transient.y, xerr=self.x_err, yerr=self.y_err,
                    fmt='x', c=color, ms=self.ms, elinewidth=self.elinewidth, capsize=self.capsize)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(self.xlim_low, self.xlim_high)
        ax.set_ylim(self.ylim_low, self.ylim_high)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.transient.ylabel)

        ax.annotate(
            self.transient.name, xy=self.xy, xycoords=self.xycoords,
            horizontalalignment=self.horizontalalignment, size=self.annotation_size)

        ax.tick_params(axis='x', pad=self.x_axis_tick_params_pad)

        if plot_save:
            plt.tight_layout()
            plt.savefig(self.data_plot_filepath, dpi=self.dpi, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_lightcurve(
            self, axes: matplotlib.axes.Axes = None, plot_save: bool = True,
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None) -> None:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        plot_save: bool, optional
            Whether to save the plot.
        plot_show: bool, optional
            Whether to show the plot.
        random_models: int, optional
            Number of random posterior samples plotted faintly. Default is 100.
        posterior: pd.DataFrame, optional
            Posterior distribution to which to draw samples from. Is optional but must be given.
        model_kwargs: dict
            Additional keyword arguments to be passed into the model.
        kwargs: dict
            No current function.
        """
        if model_kwargs is None:
            model_kwargs = dict()
        axes = axes or plt.gca()

        axes = self.plot_data(axes=axes, plot_save=False, plot_show=False)
        times = self._get_times(axes)

        self._plot_lightcurves(axes, model_kwargs, posterior, random_models, times)

        if plot_save:
            plt.tight_layout()
            plt.savefig(self.lightcurve_plot_filepath, dpi=self.dpi, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes

    def _plot_lightcurves(self, axes, model_kwargs, posterior, random_models, times):
        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        ys = self.model(times, **max_like_params, **model_kwargs)
        axes.plot(times, ys, color=self.max_likelihood_color, alpha=self.max_likelihood_alpha, lw=self.linewidth)
        for _ in range(random_models):
            params = posterior.iloc[np.random.randint(len(posterior))]
            ys = self.model(times, **params, **model_kwargs)
            axes.plot(times, ys, color=self.random_sample_color, alpha=self.random_sample_alpha, lw=self.linewidth,
                      zorder=self.zorder, **self.kwargs)


class LuminosityPlotter(IntegratedFluxPlotter):
    pass


class MagnitudePlotter(Plotter):

    wspace = KwargsAccessorWithDefault("wspace", 0.15)
    hspace = KwargsAccessorWithDefault("hspace", 0.04)
    fontsize = KwargsAccessorWithDefault("fontsize", 30)

    @property
    def colors(self):
        return self.kwargs.get("colors", self.transient.get_colors(self.filters))

    @property
    def xlabel(self):
        return self.kwargs.get("xlabel", self.transient.xlabel)

    @property
    def ylabel(self):
        return self.kwargs.get("ylabel", self.transient.ylabel)

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, plot_others: bool = False,
            plot_save: bool = True, plot_show: bool = True) -> None:
        """
        Plots the data.

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes can be given if defaults are not satisfying
        plot_others: bool, optional
            Plot all bands outside filters in black without label if True.
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
            elif plot_others:
                color = "black"
                label = None
            else:
                continue
            if isinstance(label, float):
                label = f"{label:.2e}"
            ax.errorbar(
                self.transient.x[indices], self.transient.y[indices], xerr=self._get_x_err(indices),
                yerr=self.transient.y_err[indices], fmt=self.errorbar_fmt, ms=self.ms, color=color,
                elinewidth=self.elinewidth, capsize=self.capsize, label=label)

        ax.set_xlim(0.5 * self.transient.x[0], 1.2 * self.transient.x[-1])
        self._set_y_axis_data(ax)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.tick_params(axis='x', pad=self.x_axis_tick_params_pad)
        ax.legend(ncol=2, loc='best')

        if plot_save:
            plt.tight_layout()
            plt.savefig(self.data_plot_filepath, dpi=self.dpi, bbox_inches='tight')
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes

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

    def plot_lightcurve(
            self, axes: matplotlib.axes.Axes = None,  plot_save: bool = True,
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None) -> None:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        plot_save: bool, optional
            Whether to save the plot.
        plot_show: bool, optional
            Whether to show the plot.
        random_models: int, optional
            Number of random posterior samples plotted faintly. Default is 100.
        posterior: pd.DataFrame, optional
            Posterior distribution to which to draw samples from. Is optional but must be given.
        model_kwargs: dict
            Additional keyword arguments to be passed into the model.
        kwargs: dict
            No current function.
        """
        if model_kwargs is None:
            model_kwargs = dict()
        axes = axes or plt.gca()
        axes = self.plot_data(axes=axes, plot_save=False, plot_show=False)
        axes.set_yscale('log')

        times = self._get_times(axes)

        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        random_params_list = [posterior.iloc[np.random.randint(len(posterior))] for _ in range(random_models)]

        for band, color in zip(self.transient.active_bands, self.transient.get_colors(self.transient.active_bands)):
            frequency = redback.utils.bands_to_frequency([band])
            model_kwargs["frequency"] = np.ones(len(times)) * frequency
            ys = self.model(times, **max_like_params, **model_kwargs)
            axes.plot(times, ys, color=color, alpha=0.65, lw=2)

            for params in random_params_list:
                ys = self.model(times, **params, **model_kwargs)
                axes.plot(times, ys, color='red', alpha=0.05, lw=2, zorder=-1)
        if plot_save:
            plt.tight_layout()
            plt.savefig(self.lightcurve_plot_filepath, dpi=self.dpi, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, ncols: int = 2,
            nrows: int = None, figsize: tuple = None, plot_save: bool=True, plot_show: bool=True) -> \
            matplotlib.axes.Axes:
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
        if self.transient.luminosity_data or self.transient.flux_data:
            redback.utils.logger.warning(f"Can't plot multiband for {self.transient.data_mode} data.")
            return

        if figure is None or axes is None:
            if nrows is None:
                nrows = int(np.ceil(len(self.filters) / 2))
            npanels = ncols * nrows
            if npanels < len(self.filters):
                raise ValueError(f"Insufficient number of panels. {npanels} panels were given "
                                 f"but {len(self.filters)} panels are needed.")
            if figsize is None:
                figsize = (4 + 4 * ncols, 2 + 2 * nrows)
            figure, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', figsize=figsize)

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
                self.transient.x[indices], self.transient.y[indices], xerr=x_err, yerr=self.transient.y_err[indices],
                fmt=self.errorbar_fmt, ms=self.ms, color=color, elinewidth=self.elinewidth, capsize=self.capsize,
                label=label)

            axes[i].set_xlim(0.5 * self.transient.x[indices][0], 1.2 * self.transient.x[indices][-1])
            self._set_y_axis_multiband_data(axes[i], indices)
            axes[i].legend(ncol=2)
            axes[i].tick_params(axis='both', which='major', pad=8)
            i += 1

        figure.supxlabel(self.xlabel, fontsize=self.fontsize)
        figure.supylabel(self.ylabel, fontsize=self.fontsize)
        plt.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        if plot_save:
            plt.tight_layout()
            plt.savefig(self.multiband_data_plot_filepath, dpi=self.dpi, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
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
            self, axes: matplotlib.axes.Axes = None, plot_save: bool = True,
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None) -> None:
        """

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes to plot in if given.
        plot_save: bool, optional
            Whether to save the plot.
        plot_show: bool, optional
            Whether to show the plot.
        random_models: int, optional
            Number of random posterior samples plotted faintly. Default is 100.
        posterior: pd.DataFrame, optional
            Posterior distribution to which to draw samples from. Is optional but must be given.
        model_kwargs: dict
            Additional keyword arguments to be passed into the model.
        -------

        """
        if self.transient.luminosity_data or self.transient.flux_data:
            redback.utils.logger.warning(
                f"Plotting multiband lightcurve not possible for {self.transient.data_mode}. Returning.")
            return

        axes = axes or plt.gca()
        axes = self.plot_multiband(axes=axes, plot_save=False, plot_show=False)

        times = self._get_times(axes)
        times_mesh, frequency_mesh = np.meshgrid(times, self.transient.bands_to_frequency(self.filters))
        new_model_kwargs = model_kwargs.copy()
        new_model_kwargs['frequency'] = frequency_mesh
        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        ys = self.model(times_mesh, **max_like_params, **new_model_kwargs)

        random_yss = []
        for _ in range(random_models):
            params = posterior.iloc[np.random.randint(len(posterior))]
            random_yss.append(self.model(times_mesh, **params, **new_model_kwargs))

        for i in range(len(ys)):
            axes[i].plot(times, ys[i], color=self.max_likelihood_color, alpha=self.max_likelihood_alpha, lw=self.linewidth)
            for random_ys in random_yss:
                axes[i].plot(
                    times, random_ys[i], color=self.random_sample_color,
                    alpha=self.random_sample_alpha, lw=self.linewidth, zorder=self.zorder)
        if plot_save:
            plt.tight_layout()
            plt.savefig(self.multiband_lightcurve_plot_filepath, dpi=self.dpi, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes


class FluxDensityPlotter(MagnitudePlotter):

    def _set_y_axis_data(self, ax):
        ax.set_ylim(0.5 * min(self.transient.y), 2. * np.max(self.transient.y))
        ax.set_yscale('log')
