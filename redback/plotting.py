import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

import redback
from redback.get_data.directory import afterglow_directory_structure


class MultiBandPlotter(object):

    def __init__(self, transient):
        self.transient = transient

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, ncols: int = 2,
            nrows: int = None, figsize: tuple = None, filters: list = None, **plot_kwargs: dict) -> \
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
                figsize = (4 * ncols, 4 * nrows)
            figure, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', sharey='all', figsize=figsize)

        axes = axes.ravel()

        i = 0
        for indices, band in zip(self.transient.list_of_band_indices, self.transient.unique_bands):
            if band not in filters:
                continue

            x_err = self.transient.x_err[indices] if self.transient.x_err is not None else self.transient.x_err

            color = colors[filters.index(band)]

            freq = self.transient.bands_to_frequencies([band])
            if 1e10 < freq < 1e15:
                label = band
            else:
                label = freq
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
        filename = f"{self.transient.name}_{self.transient.data_mode}_{plot_label}.png"
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(join(self.transient.transient_dir, filename), bbox_inches="tight")
        return axes


class IntegratedFluxPlotter(object):

    def __init__(self, transient):
        self.transient = transient

    def plot_data(self, axes: matplotlib.axes.Axes = None, colour: str = 'k', **kwargs) -> matplotlib.axes.Axes:
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

        if self.transient.x_err is not None:
            x_err = [np.abs(self.transient.x_err[1, :]), self.transient.x_err[0, :]]
        else:
            x_err = None
        y_err = [np.abs(self.transient.y_err[1, :]), self.transient.y_err[0, :]]

        ax = axes or plt.gca()
        ax.errorbar(self.transient.x, self.transient.y, xerr=x_err, yerr=y_err,
                    fmt='x', c=colour, ms=1, elinewidth=2, capsize=0.)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(0.5 * self.transient.x[0], 2 * (self.transient.x[-1] + x_err[1][-1]))
        ax.set_ylim(0.5 * min(self.transient.y), 2. * np.max(self.transient.y))

        ax.annotate(self.transient.name, xy=(0.95, 0.9), xycoords='axes fraction',
                    horizontalalignment='right', size=20)

        ax.set_xlabel(r'Time since burst [s]')
        ax.set_ylabel(self.transient.ylabel)
        ax.tick_params(axis='x', pad=10)

        if axes is None:
            plt.tight_layout()

        directory_structure = afterglow_directory_structure(grb=self.transient.name, data_mode=self.transient.data_mode)
        filename = f"{self.transient.name}_lc.png"
        plt.savefig(join(directory_structure.directory_path, filename))
        if axes is None:
            plt.clf()
        return ax


class LuminosityPlotter(IntegratedFluxPlotter):
    pass


class MagnitudePlotter(object):

    def __init__(self, transient):
        self.transient = transient

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, filters: list = None, plot_others: bool = True,
            plot_save: bool = True, **plot_kwargs: dict) -> None:
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
                color = colors[filters.index(band)]
                label = band
            elif plot_others:
                color = "black"
                label = None
            else:
                continue
            ax.errorbar(
                self.transient.x[indices], self.transient.y[indices], xerr=x_err, yerr=self.transient.y_err[indices],
                fmt=errorbar_fmt, ms=1, color=color, elinewidth=2, capsize=0., label=label)

        ax.set_xlim(0.5 * self.transient.x[0], 1.2 * self.transient.x[-1])
        self._set_y_axis(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', pad=10)
        ax.legend(ncol=2)

        if axes is None:
            plt.tight_layout()

        if plot_save:
            filename = f"{self.transient.name}_{self.transient.data_mode}_{plot_label}.png"
            plt.savefig(join(self.transient.transient_dir, filename), bbox_inches='tight')
            plt.clf()
        return axes

    def _set_y_axis(self, ax):
        ax.set_ylim(0.8 * min(self.transient.y), 1.2 * np.max(self.transient.y))
        ax.invert_yaxis()


class FluxDensityPlotter(MagnitudePlotter):

    def _set_y_axis(self, ax):
        ax.set_ylim(0.5 * min(self.transient.y), 2. * np.max(self.transient.y))
        ax.set_yscale('log')
