"""
Contains GRB class, with method to load and truncate data for SGRB and in future LGRB
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from astropy.cosmology import Planck15 as cosmo


from .. import models as mm
from ..model_library import model_dict
from ..utils import find_path
from .transient import Transient

dirname = os.path.dirname(__file__)

DATA_MODES = ['luminosity', 'flux', 'flux_density']


class afterglow(Transient):
    """Class for afterglows"""
    def __init__(self, name):
        """
        :param name: Telephone number of SGRB, e.g., GRB 140903A
        """
        if not name.startswith('GRB'):
            name = 'GRB' + name
        super().__init__(time=[], time_err=[], y=[], y_err=[], data_mode=None, name=name)
        self.path = find_path(path)

        self.Lum50 = []
        self.Lum50_err = []
        self.flux_density = []
        self.flux_density_err = []
        self.flux = []
        self.flux_err = []

        self._set_data()
        self._set_photon_index()
        self._set_t90()
        self._get_redshift()

    @classmethod
    def from_file(cls, filename,data_mode = 'flux'):
        """
        Instantiate the object from a private datafile.
        You do not need all the attributes just some.
        If a user has their own data for a type of transient, they can input the
        :param data_mode: flux, flux_density, luminosity
        :return: afterglow object with corresponding data loaded into an object
        """
        return afterglow_object


    @property
    def _stripped_name(self):
        return self.name.lstrip('GRB')

    @property
    def luminosity_data(self):
        return self.data_mode == DATA_MODES[0]

    def _get_redshift(self):
        return self.redshift

    @property
    def flux_data(self):
        return self.data_mode == DATA_MODES[1]

    @property
    def fluxdensity_data(self):
        return self.data_mode == DATA_MODES[2]

    @classmethod
    def from_path_and_grb(cls, path, grb):
        data_dir = find_path(path)
        return cls(name=grb, path=data_dir)

    @classmethod
    def from_path_and_grb_with_truncation(
            cls, path, grb, truncate=True, truncate_method='prompt_time_error', data_mode='flux'):
        grb = cls.from_path_and_grb(path=path, grb=grb)
        grb.load_and_truncate_data(truncate=truncate, truncate_method=truncate_method, data_mode=data_mode)
        return grb

    def load_and_truncate_data(self, truncate=True, truncate_method='prompt_time_error', data_mode='flux'):
        """
        Read data of SGRB from given path and GRB telephone number.
        Truncate the data to get rid of all but the last prompt emission point
        make a cut based on the size of the temporal error; ie if t_error < 1s, the data point is
        part of the prompt emission
        """
        self.load_data(data_mode=data_mode)
        if truncate:
            self.truncate(truncate_method=truncate_method)

    def load_data(self, data_mode='luminosity'):
        self.data_mode = data_mode
        if self.flux_data:
            label = ''
        else:
            label = f'_{data_mode}'

        data_file = f"{self.path}/{self.name}/{self.name}{label}.dat"
        data = np.loadtxt(data_file)
        self.time = data[:, 0]  # time (secs)
        self.time_err = np.abs(data[:, 1:3].T)  # \Delta time (secs)

        if self.luminosity_data:
            self.Lum50, self.Lum50_err = self._load(data)  # Lum (1e50 erg/s)
        elif self.fluxdensity_data:
            self.flux_density, self.flux_density_err = self._load(data)  # depending on detector its at a specific mJy
        elif self.flux_data:
            self.flux, self.flux_err = self._load(data)  # depending on detector its over a specific frequency range

    @staticmethod
    def _load(data):
        return data[:, 3], np.abs(data[:, 4:].T)

    def truncate(self, truncate_method='prompt_time_error'):
        if truncate_method == 'prompt_time_error':
            mask1 = self.time_err[0, :] > 0.0025
            mask2 = self.time < 0.2  # dont truncate if data point is after 0.2 seconds
            mask = np.logical_and(mask1, mask2)
            self.time = self.time[~mask]
            self.time_err = self.time_err[:, ~mask]
            if self.luminosity_data:
                self.Lum50 = self.Lum50[~mask]
                self.Lum50_err = self.Lum50_err[:, ~mask]
            elif self.flux_data:
                self.flux = self.flux[~mask]
                self.flux_err = self.flux_err[:, ~mask]
            elif self.fluxdensity_data:
                self.flux_density = self.flux_density[~mask]
                self.flux_density_err = self.flux_density_err[:, ~mask]
        else:
            truncate = self.time_err[0, :] > 0.1
            to_del = len(self.time) - (len(self.time[truncate]) + 2)
            self.time = self.time[to_del:]
            self.time_err = self.time_err[:, to_del:]
            if self.luminosity_data:
                self.Lum50 = self.Lum50[to_del:]
                self.Lum50_err = self.Lum50_err[:, to_del:]
            elif self.flux_data:
                self.flux = self.flux[to_del:]
                self.flux_err = self.flux_err[:, to_del:]
            elif self.fluxdensity_data:
                self.flux_density = self.flux_density[to_del:]
                self.flux_density_err = self.flux_density_err[:, to_del:]

    @property
    def event_table(self):
        return os.path.join(dirname, f'../tables/{self.__class__.__name__}_table.txt')

    def get_flux_density(self):
        pass

    def get_integrated_flux(self):
        pass

    def analytical_flux_to_luminosity(self):
        if self.redshift == np.nan:
            print('This GRB has no measured redshift')
            return None
        dl = cosmo.luminosity_distance(self.redshift).cgs.value
        k_corr = (1 + self.redshift) ** (self.photon_index - 2)
        lum = 4*np.pi * dl**2 * self.flux * k_corr
        rest_time = self.time/(1. + self.redshift)
        self.Lum50 = lum
        self.time = rest_time

    def get_prompt(self):
        pass

    def get_optical(self):
        pass

    def _set_data(self):
        data = pd.read_csv(self.event_table, header=0, error_bad_lines=False, delimiter='\t', dtype='str')
        data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data[
            'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        self.data = data

    def _process_data(self):
        pass

    def _set_photon_index(self):
        photon_index = self.data.query('GRB == @self._stripped_name')[
            'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].values[0]
        if photon_index == 0.:
            return 0.
        self.photon_index = self.__clean_string(photon_index)

    def _get_redshift(self):
        # some GRBs dont have measurements
        redshift = self.data.query('GRB == @self._stripped_name')['Redshift'].values[0]
        print(redshift)
        if redshift == np.nan:
            return None
        else:
            self.redshift = self.__clean_string(redshift)

    def _set_t90(self):
        # data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data['BAT Photon
        # Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        t90 = self.data.query('GRB == @self._stripped_name')['BAT T90 [sec]'].values[0]
        if t90 == 0.:
            return np.nan
        self.t90 = self.__clean_string(t90)

    def __clean_string(self, string):
        for r in ["PL", "CPL", ",", "C", "~", " "]:
            string = string.replace(r, "")
        return float(string)

    def process_grbs(self, use_default_directory=False):
        for GRB in self.data['GRB'].values:
            retrieve_and_process_data(GRB, use_default_directory=use_default_directory)

        return print(f'Flux data for all {self.__class__.__name__}s added')

    @staticmethod
    def process_grbs_w_redshift(use_default_directory=False):
        data = pd.read_csv(dirname + '/tables/GRBs_w_redshift.txt', header=0,
                           error_bad_lines=False, delimiter='\t', dtype='str')
        for GRB in data['GRB'].values:
            retrieve_and_process_data(GRB, use_default_directory=use_default_directory)

        return print('Flux data for all GRBs with redshift added')

    @staticmethod
    def process_grb_list(data, use_default_directory=False):
        """
        :param data: a list containing telephone number of GRB needing to process
        :param use_default_directory:
        :return: saves the flux file in the location specified
        """

        for GRB in data:
            retrieve_and_process_data(GRB, use_default_directory=use_default_directory)

        return print('Flux data for all GRBs in list added')

    def plot_data(self, axes=None, colour='k'):
        """
        plots the data
        GRB is the telephone number of the GRB
        :param axes:
        :param colour:
        """
        ax = axes or plt.gca()
        ax.errorbar(self.time, self.Lum50,
                    xerr=[self.time_err[1, :], self.time_err[0, :]],
                    yerr=[self.Lum50_err[1, :], self.Lum50_err[0, :]],
                    fmt='x', c=colour, ms=1, elinewidth=2, capsize=0.)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(0.5 * self.time[0], 2 * (self.time[-1] + self.time_err[0, -1]))
        ax.set_ylim(0.5 * min(self.Lum50), 2. * np.max(self.Lum50))

        ax.annotate(f'GRB{self.name}', xy=(0.95, 0.9), xycoords='axes fraction',
                    horizontalalignment='right', size=20)

        ax.set_xlabel(r'Time since burst [s]')
        if self.luminosity_data:
            ax.set_ylabel(r'Luminosity [$10^{50}$ erg s$^{-1}$]')
        else:
            ax.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]')
        ax.tick_params(axis='x', pad=10)

        if axes is None:
            plt.tight_layout()
        plt.grid(b=None)

    def plot_lightcurve(self, model, axes=None, plot_save=True, plot_show=True, random_models=1000,
                        posterior=None, use_photon_index_prior=False, outdir='./', plot_magnetar=False):
        max_l = dict(posterior.sort_values(by=['log_likelihood']).iloc[-1])

        for j in range(int(random_models)):
            params = dict(posterior.iloc[np.random.randint(len(posterior))])
            plot_models(parameters=params, axes=axes, alpha=0.05, lw=2, colour='r', model=model,
                        plot_magnetar=plot_magnetar)

        # plot max likelihood
        plot_models(parameters=max_l, axes=axes, alpha=0.65, lw=2, colour='b', model=model, plot_magnetar=plot_magnetar)

        self.plot_data(axes=axes)

        label = 'lightcurve'
        if use_photon_index_prior:
            label = f"_photon_index_{label}"

        if plot_save:
            plt.savefig(f"{outdir}{model}{label}.png")

        if plot_show:
            plt.show()


class SGRB(afterglow):
    pass


class LGRB(afterglow):
    pass


def plot_models(parameters, model, plot_magnetar, axes=None, colour='r', alpha=1.0, ls='-', lw=4):
    """
    plot the models
    parameters: dictionary of parameters - 1 set of Parameters
    model: model name
    """
    time = np.logspace(-4, 7, 100)
    ax = axes or plt.gca()

    lightcurve = model_dict[model]
    magnetar_models = ['evolving_magnetar', 'evolving_magnetar_only', 'piecewise_radiative_losses',
                       'radiative_losses', 'radiative_losses_mdr', 'radiative_losses_smoothness', 'radiative_only']
    if model in magnetar_models and plot_magnetar:
        if model == 'radiative_losses_mdr':
            magnetar = mm.magnetar_only(time, nn=3., **parameters)
        else:
            magnetar = mm.magnetar_only(time, **parameters)
        ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle='--')
    ax.plot(time, lightcurve, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32)

