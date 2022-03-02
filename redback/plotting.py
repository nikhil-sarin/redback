import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import redback
from os.path import join


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
