"""
Contains GRB class, with method to load and truncate data for SGRB and in future LGRB
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from redback.utils import logger

from astropy.cosmology import Planck18 as cosmo
from ..getdata import afterglow_directory_structure
from os.path import join

from .transient import Transient

dirname = os.path.dirname(__file__)


class Afterglow(Transient):

    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'photometry']

    """Class for afterglows"""
    def __init__(self, name, data_mode='flux', time=None, time_err=None, Lum50=None, Lum50_err=None,
                 flux=None, flux_err=None, flux_density=None, flux_density_err=None, magnitude=None,
                 magnitude_err=None):

        """
        :param name: Telephone number of SGRB, e.g., GRB 140903A
        """
        if not name.startswith('GRB'):
            name = 'GRB' + name

        super().__init__(name=name, data_mode=data_mode, time=time, time_err=time_err, Lum50=Lum50, Lum50_err=Lum50_err,
                         flux=flux, flux_err=flux_err, flux_density=flux_density, flux_density_err=flux_density_err,
                         magnitude=magnitude, magnitude_err=magnitude_err)

        self._set_data()
        self._set_photon_index()
        self._set_t90()
        self._get_redshift()

    @classmethod
    def from_swift_grb(cls, name, data_mode='flux', truncate=True, truncate_method='prompt_time_error'):
        if not name.startswith('GRB'):
            name = 'GRB' + name
        afterglow = cls(name=name, data_mode=data_mode)

        afterglow._set_data()
        afterglow._set_photon_index()
        afterglow._set_t90()
        afterglow._get_redshift()

        afterglow.load_and_truncate_data(truncate=truncate, truncate_method=truncate_method, data_mode=data_mode)
        return afterglow

    @property
    def _stripped_name(self):
        return self.name.lstrip('GRB')

    def load_and_truncate_data(self, truncate=True, truncate_method='prompt_time_error', data_mode='flux'):
        """
        Read data of SGRB from given path and GRB telephone number.
        Truncate the data to get rid of all but the last prompt emission point
        make a cut based on the size of the temporal error; ie if t_error < 1s, the data point is
        part of the prompt emission
        """
        self.data_mode = data_mode
        self.x, self.x_err, self.y, self.y_err = self.load_data(name=self.name, data_mode=self.data_mode)
        if truncate:
            self.truncate(truncate_method=truncate_method)

    @staticmethod
    def load_data(name, data_mode=None):
        grb_dir, _, _ = afterglow_directory_structure(grb=name.lstrip('GRB'), use_default_directory=False,
                                                      data_mode=data_mode)
        filename = f"{name}.csv"

        data_file = join(grb_dir, filename)
        data = np.genfromtxt(data_file, delimiter=",")[1:]
        x = data[:, 0]
        x_err = data[:, 1:3].T
        y = np.array(data[:, 3])
        y_err = np.array(np.abs(data[:, 4:6].T))
        return x, x_err, y, y_err

    def truncate(self, truncate_method='prompt_time_error'):
        if truncate_method == 'prompt_time_error':
            self._truncate_prompt_time_error()
        elif truncate_method == 'left_of_max':
            self._truncate_left_of_max()
        else:
            self._truncate_default()

    def _truncate_prompt_time_error(self):
        mask1 = self.x_err[0, :] > 0.0025
        mask2 = self.x < 2.0  # dont truncate if data point is after 2.0 seconds
        mask = np.logical_and(mask1, mask2)
        self.x = self.x[~mask]
        self.x_err = self.x_err[:, ~mask]
        self.y = self.y[~mask]
        self.y_err = self.y_err[:, ~mask]

    def _truncate_left_of_max(self):
        max_index = np.argmax(self.y)
        self.x = self.x[max_index:]
        self.x_err = self.x_err[:, max_index:]
        self.y = self.y[max_index:]
        self.y_err = self.y_err[:, max_index:]

    def _truncate_default(self):
        truncate = self.time_err[0, :] > 0.1
        to_del = len(self.time) - (len(self.time[truncate]) + 2)
        self.x = self.x[to_del:]
        self.x_err = self.x_err[:, to_del:]
        self.y = self.y[to_del:]
        self.y_err = self.y_err[:, to_del:]

    @property
    def event_table(self):
        return os.path.join(dirname, f'../tables/{self.__class__.__name__}_table.txt')

    def _save_luminosity_data(self):
        grb_dir, _, _ = afterglow_directory_structure(grb=self._stripped_name, use_default_directory=False,
                                                      data_mode=self.data_mode)
        filename = f"{self.name}.csv"
        data = {"Time in restframe [s]": self.time_rest_frame,
                "Pos. time err in restframe [s]": self.time_rest_frame_err[0, :],
                "Neg. time err in restframe [s]": self.time_rest_frame_err[1, :],
                "Luminosity [10^50 erg s^{-1}]": self.Lum50,
                "Pos. luminosity err [10^50 erg s^{-1}]": self.Lum50_err[0, :],
                "Neg. luminosity err [10^50 erg s^{-1}]": self.Lum50_err[1, :]}
        df = pd.DataFrame(data=data)
        df.to_csv(join(grb_dir, filename), index=False)

    def _set_data(self):
        data = pd.read_csv(self.event_table, header=0, error_bad_lines=False, delimiter='\t', dtype='str')
        data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data[
            'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        self.data = data

    def _set_photon_index(self):
        photon_index = self.data.query('GRB == @self._stripped_name')[
            'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].values[0]
        if photon_index == 0.:
            return 0.
        self.photon_index = self.__clean_string(photon_index)

    def _get_redshift(self):
        # some GRBs dont have measurements
        redshift = self.data.query('GRB == @self._stripped_name')['Redshift'].values[0]
        if isinstance(redshift, str):
            self.redshift = self.__clean_string(redshift)
        elif np.isnan(redshift):
            return None
        else:
            self.redshift = redshift

    def _set_t90(self):
        # data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = data['BAT Photon
        # Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        t90 = self.data.query('GRB == @self._stripped_name')['BAT T90 [sec]'].values[0]
        if t90 == 0.:
            return np.nan
        self.t90 = self.__clean_string(t90)

    @staticmethod
    def __clean_string(string):
        for r in ["PL", "CPL", ",", "C", "~", " ", 'Gemini:emission', '()']:
            string = string.replace(r, "")
        return float(string)

    def analytical_flux_to_luminosity(self):
        redshift = self._get_redshift_for_luminosity_calculation()
        if redshift is None:
            return

        luminosity_distance = cosmo.luminosity_distance(redshift).cgs.value
        k_corr = (1 + redshift) ** (self.photon_index - 2)
        isotropic_bolometric_flux = (luminosity_distance ** 2.) * 4. * np.pi * k_corr
        counts_to_flux_fraction = 1

        self._calculate_rest_frame_time_and_luminosity(
            counts_to_flux_fraction=counts_to_flux_fraction,
            isotropic_bolometric_flux=isotropic_bolometric_flux,
            redshift=redshift)
        self.data_mode = 'luminosity'
        self._save_luminosity_data()

    def numerical_flux_to_luminosity(self, counts_to_flux_absorbed, counts_to_flux_unabsorbed):
        try:
            from sherpa.astro import ui as sherpa
        except ImportError as e:
            logger.warning(e)
            logger.warning("Can't perform numerical flux to luminosity calculation")

        redshift = self._get_redshift_for_luminosity_calculation()
        if redshift is None:
            return

        Ecut = 1000
        obs_elow = 0.3
        obs_ehigh = 10

        bol_elow = 1.  # bolometric restframe low frequency in keV
        bol_ehigh = 10000.  # bolometric restframe high frequency in keV

        alpha = self.photon_index
        beta = self.photon_index

        sherpa.dataspace1d(obs_elow, bol_ehigh, 0.01)
        sherpa.set_source(sherpa.bpl1d.band)
        band.gamma1 = alpha  # noqa
        band.gamma2 = beta  # noqa
        band.eb = Ecut  # noqa

        luminosity_distance = cosmo.luminosity_distance(redshift).cgs.value
        k_corr = sherpa.calc_kcorr(redshift, obs_elow, obs_ehigh, bol_elow, bol_ehigh, id=1)
        isotropic_bolometric_flux = (luminosity_distance ** 2.) * 4. * np.pi * k_corr
        counts_to_flux_fraction = counts_to_flux_unabsorbed / counts_to_flux_absorbed

        self._calculate_rest_frame_time_and_luminosity(
            counts_to_flux_fraction=counts_to_flux_fraction,
            isotropic_bolometric_flux=isotropic_bolometric_flux,
            redshift=redshift)
        self.data_mode = 'luminosity'
        self._save_luminosity_data()

    def _get_redshift_for_luminosity_calculation(self):
        if self.luminosity_data:
            logger.warning('The data is already in luminosity mode, returning.')
        elif self.flux_data:
            if np.isnan(self.redshift):
                logger.warning('This GRB has no measured redshift, using default z = 0.75')
                return 0.75
            return self.redshift
        else:
            logger.warning(f'The data needs to be in flux mode, but is in {self.data_mode}.')

    def _calculate_rest_frame_time_and_luminosity(self, counts_to_flux_fraction, isotropic_bolometric_flux, redshift):
        self.Lum50 = self.flux * counts_to_flux_fraction * isotropic_bolometric_flux * 1e-50
        self.Lum50_err = self.flux_err * isotropic_bolometric_flux * 1e-50
        self.time_rest_frame = self.time / (1 + redshift)
        self.time_rest_frame_err = self.time_err / (1 + redshift)

    def plot_data(self, axes=None, colour='k'):
        """
        plots the data
        GRB is the telephone number of the GRB
        :param axes:
        :param colour:
        """

        x_err = [self.x_err[1, :], self.x_err[0, :]]
        y_err = [self.y_err[1, :], self.y_err[0, :]]

        ax = axes or plt.gca()
        ax.errorbar(self.x, self.y, xerr=x_err, yerr=y_err,
                    fmt='x', c=colour, ms=1, elinewidth=2, capsize=0.)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(0.5 * self.x[0], 2 * (self.x[-1] + x_err[1][-1]))
        ax.set_ylim(0.5 * min(self.y), 2. * np.max(self.y))

        ax.annotate(f'GRB{self.name}', xy=(0.95, 0.9), xycoords='axes fraction',
                    horizontalalignment='right', size=20)

        ax.set_xlabel(r'Time since burst [s]')
        ax.set_ylabel(self.ylabel)
        ax.tick_params(axis='x', pad=10)

        if axes is None:
            plt.tight_layout()

        grb_dir, _, _ = afterglow_directory_structure(grb=self._stripped_name, use_default_directory=False,
                                                      data_mode=self.data_mode)
        filename = f"{self.name}_lc.png"
        plt.savefig(join(grb_dir, filename))
        plt.clf()

    def plot_multiband(self):
        if self.data_mode != 'flux_density':
            logger.warning('why are you doing this')
        pass


class SGRB(Afterglow):
    pass


class LGRB(Afterglow):
    pass


# def plot_models(parameters, model, plot_magnetar, axes=None, colour='r', alpha=1.0, ls='-', lw=4):
#     """
#     plot the models
#     parameters: dictionary of parameters - 1 set of Parameters
#     model: model name
#     """
#     time = np.logspace(-4, 7, 100)
#     ax = axes or plt.gca()
#
#     lightcurve = all_models_dict[model]
#     magnetar_models = ['evolving_magnetar', 'evolving_magnetar_only', 'piecewise_radiative_losses',
#                        'radiative_losses', 'radiative_losses_mdr', 'radiative_losses_smoothness', 'radiative_only']
#     if model in magnetar_models and plot_magnetar:
#         if model == 'radiative_losses_mdr':
#             magnetar = mm.magnetar_only(time, nn=3., **parameters)
#         else:
#             magnetar = mm.magnetar_only(time, **parameters)
#         ax.plot(time, magnetar, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32, linestyle='--')
#     ax.plot(time, lightcurve, color=colour, ls=ls, lw=lw, alpha=alpha, zorder=-32)


# def plot_lightcurve(self, model, axes=None, plot_save=True, plot_show=True, random_models=1000,
#                     posterior=None, use_photon_index_prior=False, outdir='./', plot_magnetar=False):
#     max_l = dict(posterior.sort_values(by=['log_likelihood']).iloc[-1])
#
#     for j in range(int(random_models)):
#         params = dict(posterior.iloc[np.random.randint(len(posterior))])
#         plot_models(parameters=params, axes=axes, alpha=0.05, lw=2, colour='r', model=model,
#                     plot_magnetar=plot_magnetar)
#
#     # plot max likelihood
#     plot_models(parameters=max_l, axes=axes, alpha=0.65, lw=2, colour='b', model=model, plot_magnetar=plot_magnetar)
#
#     self.plot_data(axes=axes)
#
#     label = 'lightcurve'
#     if use_photon_index_prior:
#         label = f"_photon_index_{label}"
#
#     if plot_save:
#         plt.savefig(f"{outdir}{model}{label}.png")
#
#     if plot_show:
#         plt.show()