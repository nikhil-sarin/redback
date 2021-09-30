import matplotlib.pyplot

from .transient import Transient

from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from redback.getdata import transient_directory_structure

data_mode = ['flux_density', 'photometry', 'luminosity']


class Kilonova(Transient):
    def __init__(self, name, data_mode='photometry', time=None, time_err=None, y=None, y_err=None, bands=None, system=None):

        super().__init__(time=time, time_err=time_err, y=y, y_err=y_err, data_mode=data_mode, name=name)
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
    def from_open_access_catalogue(cls, transient, data_mode="photometry"):
        kilonova = cls(name=transient, data_mode=data_mode)
        transient_dir = cls._get_transient_dir(name=transient)
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands, system = cls.load_data(name=transient, transient_dir=transient_dir, data_mode="all")
        kilonova.time = time_days
        kilonova.flux_density = flux_density
        kilonova.flux_density_err = flux_density_err
        kilonova.magnitude = magnitude
        kilonova.magnitude_err = magnitude_err
        kilonova.bands = bands
        kilonova.system = system
        return kilonova

    def _set_data(self):
        pass

    def plot_data(self, axes=None, filters=None, plot_others=True):
        """
        plots the data
        :param axes:
        :param colour:
        """

        unique_bands = np.unique(self.bands)

        list_of_indices = []

        if filters is None:
            filters = ["g", "r", "i", "z", "y", "J", "H", "K"]

        colors = cm.rainbow(np.linspace(0, 1, len(filters)))

        for b in unique_bands:
            list_of_indices.append(np.where(self.bands == b)[0])

        ylabel = self._get_labels()


        ax = axes or plt.gca()
        for idxs, band in zip(list_of_indices, unique_bands):
            if self.x_err is not None:
                x_err = self.x_err[idxs]
            else:
                x_err = self.x_err
            if band in filters:
                color = colors[filters.index(band)]
                label = band
            elif plot_others:
                color = "black"
                label = None
            else:
                continue
            ax.errorbar(self.x[idxs], self.y[idxs], xerr=x_err, yerr=self.y_err[idxs],
                        fmt='x', ms=1, color=color, elinewidth=2, capsize=0., label=label)


        ax.set_xlim(0.5 * self.x[0], 1.2 * self.x[-1])
        if self.photometry_data:
            ax.set_ylim(0.8 * min(self.y), 1.2 * np.max(self.y))
            ax.invert_yaxis()
        else:
            ax.set_ylim(0.5 * min(self.y), 2. * np.max(self.y))
        ax.set_xlabel(r'Time since burst [days]')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', pad=10)
        ax.legend(ncol=2)

        if axes is None:
            plt.tight_layout()

        filename = f"{self.name}_{self.data_mode}_lc.png"
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

    def _get_labels(self):
        if self.luminosity_data:
            return r'Luminosity [$10^{50}$ erg s$^{-1}$]'
        elif self.photometry_data:
            return r'Magnitude'
        elif self.flux_density_data:
            return r'Flux density [mJy]'
        else:
            raise ValueError

    def plot_multiband(self, figure, axes, filters=None):
        axes = axes.ravel()
        unique_bands = np.unique(self.bands)

        list_of_indices = []

        if filters is None:
            filters = ["g", "r", "i", "z", "y", "J", "H", "K"]

        colors = cm.rainbow(np.linspace(0, 1, len(filters)))

        for b in unique_bands:
            list_of_indices.append(np.where(self.bands == b)[0])

        ylabel = self._get_labels()

        for i, (idxs, band) in enumerate(zip(list_of_indices, filters)):
            if self.x_err is not None:
                x_err = self.x_err[idxs]
            else:
                x_err = self.x_err
            if band in filters:
                color = colors[filters.index(band)]
                label = band
            else:
                continue

            axes[i].errorbar(self.x[idxs], self.y[idxs], xerr=x_err, yerr=self.y_err[idxs],
                             fmt='x', ms=1, color=color, elinewidth=2, capsize=0., label=label)

            axes[i].set_xlim(0.5 * self.x[idxs][0], 1.2 * self.x[idxs][-1])
            if self.photometry_data:
                axes[i].set_ylim(0.8 * min(self.y[idxs]), 1.2 * np.max(self.y[idxs]))
                axes[i].invert_yaxis()
            else:
                axes[i].set_ylim(0.5 * min(self.y[idxs]), 2. * np.max(self.y[idxs]))
                axes[i].set_yscale("log")
            axes[i].legend(ncol=2)
            axes[i].tick_params(axis='both', which='major', pad=8)
        figure.supxlabel("Time [days]", fontsize=30)
        figure.supylabel(ylabel, fontsize=30)
        filename = f"{self.name}_{self.data_mode}_multiband_lc.png"
        plt.subplots_adjust(wspace=0.15, hspace=0.04)
        plt.savefig(join(self.transient_dir, filename), bbox_inches="tight")
        plt.clf()
