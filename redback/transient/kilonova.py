import matplotlib.pyplot

from .transient import Transient

from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from redback.getdata import transient_directory_structure
from redback.utils import logger

data_mode = ['flux_density', 'photometry', 'luminosity']


class Kilonova(Transient):
    def __init__(self, name, data_mode='photometry', time=None, time_err=None, time_rest_frame=None,
                 time_rest_frame_err=None, Lum50=None, Lum50_err=None, flux_density=None, flux_density_err=None,
                 magnitude=None, magnitude_err=None, bands=None, system=None, **kwargs):

        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame,
                         time_rest_frame_err=time_rest_frame_err, Lum50=Lum50, Lum50_err=Lum50_err,
                         flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                         magnitude_err=magnitude_err, data_mode=data_mode, name=name, **kwargs)
        self.name = name
        self.bands = bands
        self.system = system
        self._set_data()

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

    @classmethod
    def from_open_access_catalogue(cls, name, data_mode="photometry"):
        transient_dir = cls._get_transient_dir(name=name)
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands, system = \
            cls.load_data(name=name, transient_dir=transient_dir, data_mode="all")
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, flux_density=flux_density,
                   flux_density_err=flux_density_err, magnitude=magnitude, magnitude_err=magnitude_err, bands=bands,
                   system=system)

    def _set_data(self):
        pass

    def plot_data(self, axes=None, filters=None, plot_others=True, **plot_kwargs):
        """
        plots the data
        :param axes:
        :param colour:
        """
        errorbar_fmt = plot_kwargs.get("errorbar_fmt", "x")
        colors = plot_kwargs.get("colors", self.get_colors(filters))
        xlabel = plot_kwargs.get("xlabel", r'Time since burst [days]')
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

    @property
    def transient_dir(self):
        return self._get_transient_dir(self.name)

    @staticmethod
    def _get_transient_dir(name):
        transient_dir, _, _ = transient_directory_structure(
            transient=name, use_default_directory=False,
            transient_type="kilonova")
        return transient_dir

    def plot_multiband(self, figure=None, axes=None, ncols=2, nrows=None, figsize=None, filters=None, **plot_kwargs):
        wspace = plot_kwargs.get("wspace", 0.15)
        hspace = plot_kwargs.get("hspace", 0.04)
        fontsize = plot_kwargs.get("fontsize", 30)
        errorbar_fmt = plot_kwargs.get("errorbar_fmt", "x")
        colors = plot_kwargs.get("colors", self.get_colors(filters))
        xlabel = plot_kwargs.get("xlabel", "Time [days]")
        ylabel = plot_kwargs.get("ylabel", self.ylabel)
        plot_label = plot_kwargs.get("plot_label", "multiband_lc")

        if figure is None or axes is None:
            if nrows is None:
                nrows = int(np.ceil(len(filters)/2))
            npanels = ncols * nrows
            if npanels < len(filters):
                raise ValueError(f"Insufficient number of panels. {npanels} panels were given "
                                 f"but {len(filters)} panels are needed.")
            if figsize is None:
                figsize = (4*ncols, 4*nrows)
            figure, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=figsize)

        axes = axes.ravel()

        if filters is None:
            filters = self.default_filters

        i = 0
        for idxs, band in zip(self.list_of_band_indices, self.unique_bands):
            if band not in filters:
                continue

            x_err = self.x_err[idxs] if self is not None else self.x_err

            color = colors[filters.index(band)]
            axes[i].errorbar(self.x[idxs], self.y[idxs], xerr=x_err, yerr=self.y_err[idxs],
                             fmt=errorbar_fmt, ms=1, color=color, elinewidth=2, capsize=0., label=band)

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
        plt.clf()

    @property
    def unique_bands(self):
        return np.unique(self.bands)

    @property
    def list_of_band_indices(self):
        return [np.where(self.bands == b)[0] for b in self.unique_bands]

    @property
    def default_filters(self):
        return ["g", "r", "i", "z", "y", "J", "H", "K"]

    def get_colors(self, filters):
        return cm.rainbow(np.linspace(0, 1, len(filters)))
