from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback


class Plotter(object):

    def __init__(self, transient):
        self.transient = transient

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


class IntegratedFluxPlotter(Plotter):

    @property
    def xlabel(self):
        return r"Time since burst [s]"

    def plot_data(self, axes: matplotlib.axes.Axes = None, colour: str = 'k', plot_save: bool = True,
                  plot_show: bool = True, **kwargs) -> matplotlib.axes.Axes:
        """
        Plots the Afterglow lightcurve and returns Axes.

        Parameters
        ----------
        axes : Union[matplotlib.axes.Axes, None], optional
            Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        colour: str, optional
            Colour of the data.
        kwargs: dict
            Additional keyword arguments.

        Returns
        ----------
        matplotlib.axes.Axes: The axes with the plot.
        """
        xy = kwargs.get("xy", (0.95, 0.9))
        xycoords = kwargs.get("xycoords", "axes fraction")
        horizontalalignment = kwargs.get("horizontalalignment", "right")
        annotation_size = kwargs.get("annotation_size", 20)


        ax = axes or plt.gca()
        ax.errorbar(self.transient.x, self.transient.y, xerr=self.x_err, yerr=self.y_err,
                    fmt='x', c=colour, ms=1, elinewidth=2, capsize=0.)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(self.xlim_low, self.xlim_high)
        ax.set_ylim(self.ylim_low, self.ylim_high)

        ax.annotate(
            self.transient.name, xy=xy, xycoords=xycoords,
            horizontalalignment=horizontalalignment, size=annotation_size)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.transient.ylabel)
        ax.tick_params(axis='x', pad=10)

        if axes is None:
            plt.tight_layout()

        filename = f"{self.transient.name}_data.png"
        if plot_save:
            plt.tight_layout()
            plt.savefig(join(self.transient.directory_structure.directory_path, filename))
        if plot_show:
            plt.tight_layout()
            plt.show()
        if axes is None:
            plt.clf()
        return ax

    def plot_lightcurve(
            self, model: callable, filename: str = None, axes: matplotlib.axes.Axes = None, plot_save: bool = True,
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None, outdir: str = '.',
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
        plot_save: bool, optional
            Whether to save the plot.
        plot_show: bool, optional
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
        if filename is None:
            filename = f"{self.transient.name}_lightcurve.png"
        if model_kwargs is None:
            model_kwargs = dict()
        axes = axes or plt.gca()
        axes = self.plot_data(axes=axes, plot_save=False, plot_show=False)
        axes.set_yscale('log')
        plt.semilogy()
        times = self._get_times(axes)

        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        ys = model(times, **max_like_params, **model_kwargs)
        axes.plot(times, ys, color='blue', alpha=0.65, lw=2)

        for _ in range(random_models):
            params = posterior.iloc[np.random.randint(len(posterior))]
            ys = model(times, **params, **model_kwargs)
            axes.plot(times, ys, color='red', alpha=0.05, lw=2, zorder=-1, **kwargs)

        if plot_save:
            plt.savefig(join(outdir, filename), dpi=300, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes


class LuminosityPlotter(IntegratedFluxPlotter):
    pass


class MagnitudePlotter(Plotter):

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, filters: list = None, plot_others: bool = False,
            plot_save: bool = True, plot_show: bool = True, **plot_kwargs: dict) -> None:
        """
        Plots the data.

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            Axes can be given if defaults are not satisfying
        filters: list, optional
            Which bands to plot. Will use default filters if None is given.
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
        if filters is None:
            filters = self.transient.active_bands

        errorbar_fmt = plot_kwargs.get("errorbar_fmt", "x")
        colors = plot_kwargs.get("colors", self.transient.get_colors(filters))
        xlabel = plot_kwargs.get("xlabel", self.transient.xlabel)
        ylabel = plot_kwargs.get("ylabel", self.transient.ylabel)
        plot_label = plot_kwargs.get("plot_label", "data")

        ax = axes or plt.gca()
        for indices, band in zip(self.transient.list_of_band_indices, self.transient.unique_bands):
            x_err = self.transient.x_err[indices] if self.transient.x_err is not None else self.transient.x_err
            if band in filters:
                color = colors[list(filters).index(band)]
                label = band
            elif plot_others:
                color = "black"
                label = None
            else:
                continue
            if isinstance(label, float):
                label = f"{label:.2e}"
            ax.errorbar(
                self.transient.x[indices], self.transient.y[indices], xerr=x_err, yerr=self.transient.y_err[indices],
                fmt=errorbar_fmt, ms=1, color=color, elinewidth=2, capsize=0., label=label)

        ax.set_xlim(0.5 * self.transient.x[0], 1.2 * self.transient.x[-1])
        self._set_y_axis(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', pad=10)
        ax.legend(ncol=2, loc='best')

        if axes is None:
            plt.tight_layout()

        if plot_save:
            filename = f"{self.transient.name}_{plot_label}.png"
            plt.savefig(join(self.transient.directory_structure.directory_path, filename), bbox_inches='tight')
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes

    def _set_y_axis(self, ax):
        ax.set_ylim(0.8 * min(self.transient.y), 1.2 * np.max(self.transient.y))
        ax.invert_yaxis()

    def plot_lightcurve(
            self, model: callable, filename: str = None, axes: matplotlib.axes.Axes = None,  plot_save: bool = True,
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None, outdir: str = '.',
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
        plot_save: bool, optional
            Whether to save the plot.
        plot_show: bool, optional
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
        if filename is None:
            filename = f"{self.transient.name}_lightcurve.png"
        if model_kwargs is None:
            model_kwargs = dict()
        axes = axes or plt.gca()
        axes = self.plot_data(axes=axes, plot_save=False, plot_show=False)
        axes.set_yscale('log')
        # plt.semilogy()
        times = self._get_times(axes)

        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        random_params_list = [posterior.iloc[np.random.randint(len(posterior))] for _ in range(random_models)]

        for band, color in zip(self.transient.active_bands, self.transient.get_colors(self.transient.active_bands)):
            frequency = redback.utils.bands_to_frequency([band])
            model_kwargs["frequency"] = np.ones(len(times)) * frequency
            ys = model(times, **max_like_params, **model_kwargs)
            axes.plot(times, ys, color=color, alpha=0.65, lw=2)

            for params in random_params_list:
                ys = model(times, **params, **model_kwargs)
                axes.plot(times, ys, color='red', alpha=0.05, lw=2, zorder=-1)
        if plot_save:
            plt.savefig(join(outdir, filename), dpi=300, bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, ncols: int = 2,
            nrows: int = None, figsize: tuple = None, filters: list = None, plot_save: bool=True, plot_show: bool=True,
            **plot_kwargs: dict) -> \
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
        filters: list, optional
            Which bands to plot. Will use default filters if None is given.
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

        if filters is None:
            filters = self.transient.active_bands
        elif str(filters) == 'default':
            filters = self.transient.default_filters

        wspace = plot_kwargs.get("wspace", 0.15)
        hspace = plot_kwargs.get("hspace", 0.04)
        fontsize = plot_kwargs.get("fontsize", 30)
        errorbar_fmt = plot_kwargs.get("errorbar_fmt", "x")
        colors = plot_kwargs.get("colors", self.transient.get_colors(filters))
        xlabel = plot_kwargs.get("xlabel", self.transient.xlabel)
        ylabel = plot_kwargs.get("ylabel", self.transient.ylabel)
        plot_label = plot_kwargs.get("plot_label", "multiband_data")

        if figure is None or axes is None:
            if nrows is None:
                nrows = int(np.ceil(len(filters) / 2))
            npanels = ncols * nrows
            if npanels < len(filters):
                raise ValueError(f"Insufficient number of panels. {npanels} panels were given "
                                 f"but {len(filters)} panels are needed.")
            if figsize is None:
                figsize = (4 + 4 * ncols, 2 + 2 * nrows)
            figure, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', figsize=figsize)

        axes = axes.ravel()

        i = 0
        for indices, band in zip(self.transient.list_of_band_indices, self.transient.unique_bands):
            if band not in filters:
                continue

            x_err = self.transient.x_err[indices] if self.transient.x_err is not None else self.transient.x_err

            color = colors[list(filters).index(band)]

            if isinstance(band, str):
                freq = self.transient.bands_to_frequency([band])
                if 1e10 < freq < 1e15:
                    label = band
                else:
                    label = freq
            else:
                label = f"{band:.2e}"
            axes[i].errorbar(self.transient.x[indices], self.transient.y[indices], xerr=x_err, yerr=self.transient.y_err[indices],
                             fmt=errorbar_fmt, ms=1, color=color, elinewidth=2, capsize=0., label=label)

            axes[i].set_xlim(0.5 * self.transient.x[indices][0], 1.2 * self.transient.x[indices][-1])
            if self.transient.magnitude_data:
                axes[i].set_ylim(0.8 * min(self.transient.y[indices]), 1.2 * np.max(self.transient.y[indices]))
                axes[i].invert_yaxis()
            else:
                axes[i].set_ylim(0.5 * min(self.transient.y[indices]), 2. * np.max(self.transient.y[indices]))
                axes[i].set_yscale("log")
            axes[i].legend(ncol=2)
            axes[i].tick_params(axis='both', which='major', pad=8)
            i += 1

        figure.supxlabel(xlabel, fontsize=fontsize)
        figure.supylabel(ylabel, fontsize=fontsize)
        filename = f"{self.transient.name}_{plot_label}.png"
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        if plot_save:
            plt.savefig(join(self.transient.directory_structure.directory_path, filename), bbox_inches="tight")
        if plot_show:
            plt.tight_layout()
            plt.show()
        return axes

    def plot_multiband_lightcurve(
            self, model: callable, filename: str = None, axes: matplotlib.axes.Axes = None, plot_save: bool = True,
            plot_show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None, outdir: str = '.',
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
        plot_save: bool, optional
            Whether to save the plot.
        plot_show: bool, optional
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
        if self.transient.luminosity_data or self.transient.flux_data:
            redback.utils.logger.warning(
                f"Plotting multiband lightcurve not possible for {self.transient.data_mode}. Returning.")
            return

        if filename is None:
            filename = f"{self.transient.name}_multiband_lightcurve.png"
        axes = axes or plt.gca()
        filters = kwargs.get("filters", self.transient.active_bands)
        axes = self.plot_multiband(axes=axes, plot_save=False, plot_show=False, filters=filters)

        times = self._get_times(axes)

        times_mesh, frequency_mesh = np.meshgrid(times, redback.utils.bands_to_frequency(filters))
        new_model_kwargs = model_kwargs.copy()
        new_model_kwargs['frequency'] = frequency_mesh
        posterior.sort_values(by='log_likelihood')
        max_like_params = posterior.iloc[-1]
        ys = model(times_mesh, **max_like_params, **new_model_kwargs)

        for i in range(len(filters)):
            axes[i].plot(times, ys[i], color='blue', alpha=0.65, lw=2)
            for _ in range(random_models):
                params = posterior.iloc[np.random.randint(len(posterior))]
                ys = model(times_mesh, **params, **new_model_kwargs)
                axes[i].plot(times, ys[i], color='red', alpha=0.05, lw=2, zorder=-1)
        if plot_save:
            plt.savefig(join(outdir, filename), dpi=300, bbox_inches="tight")
        if plot_show:
            plt.show()
        return axes


class FluxDensityPlotter(MagnitudePlotter):

    def _set_y_axis(self, ax):
        ax.set_ylim(0.5 * min(self.transient.y), 2. * np.max(self.transient.y))
        ax.set_yscale('log')
