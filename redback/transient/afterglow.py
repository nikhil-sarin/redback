"""
Contains GRB class, with method to load and truncate data for SGRB and in future LGRB
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from redback.utils import logger, bands_to_frequencies

from astropy.cosmology import Planck18 as cosmo
from ..getdata import afterglow_directory_structure
from os.path import join

import redback
from .transient import Transient

dirname = os.path.dirname(__file__)


class Afterglow(Transient):

    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'photometry']

    """Class for afterglows"""
    def __init__(self, name, data_mode='flux', time=None, time_err=None, time_mjd=None, time_mjd_err=None,
                 time_rest_frame=None, time_rest_frame_err=None, Lum50=None, Lum50_err=None, flux=None, flux_err=None,
                 flux_density=None, flux_density_err=None, magnitude=None, magnitude_err=None, frequency=None,
                 bands=None, system=None, active_bands='all', use_phase_model=False, **kwargs):

        """
        :param name: Telephone number of SGRB, e.g., GRB 140903A
        """
        if not name.startswith('GRB'):
            name = 'GRB' + name

        super().__init__(name=name, data_mode=data_mode, time=time, time_mjd=time_mjd, time_mjd_err=time_mjd_err,
                         time_err=time_err, time_rest_frame=time_rest_frame, time_rest_frame_err=time_rest_frame_err,
                         Lum50=Lum50, Lum50_err=Lum50_err, flux=flux, flux_err=flux_err, flux_density=flux_density,
                         flux_density_err=flux_density_err, use_phase_model=use_phase_model, bands=bands, system=system,
                         magnitude=magnitude, magnitude_err=magnitude_err, **kwargs)
        self.bands = bands
        self.system = system
        self.frequency = frequency
        self.active_bands = active_bands
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
        truncator = Truncator(x=self.x, x_err=self.x_err, y=self.y, y_err=self.y_err, time=self.time,
                              time_err=self.time_err, truncate_method=truncate_method)
        self.x, self.x_err, self.y, self.y_err = truncator.truncate()

    @property
    def active_bands(self):
        return self._active_bands

    @active_bands.setter
    def active_bands(self, active_bands):
        if active_bands == 'all':
            self._active_bands = np.unique(self.bands)
        else:
            self._active_bands = active_bands

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        if frequency is None:
            self._frequency = redback.utils.bands_to_frequencies(self.bands)
        else:
            self._frequency = frequency

    def get_filtered_data(self):
        if self.flux_density_data or self.photometry_data:
            idxs = [b in self.active_bands for b in self.bands]
            filtered_x = self.x[idxs]
            try:
                filtered_x_err = self.x_err[idxs]
            except Exception:
                filtered_x_err = None
            filtered_y = self.y[idxs]
            filtered_y_err = self.y_err[idxs]
            return filtered_x, filtered_x_err, filtered_y, filtered_y_err
        else:
            raise ValueError(f"Transient needs to be in flux density or photometry data mode, "
                             f"but is in {self.data_mode} instead.")

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
        meta_data = pd.read_csv(self.event_table, header=0, error_bad_lines=False, delimiter='\t', dtype='str')
        meta_data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = meta_data[
                  'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        self.meta_data = meta_data

    def _set_photon_index(self):
        try:
            photon_index = self.meta_data.query('GRB == @self._stripped_name')[
                'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].values[0]
            self.photon_index = self.__clean_string(photon_index)
        except IndexError:
            self.photon_index = np.nan

    def _get_redshift(self):
        # some GRBs dont have measurements
        try:
            redshift = self.meta_data.query('GRB == @self._stripped_name')['Redshift'].values[0]
            if isinstance(redshift, str):
                self.redshift = self.__clean_string(redshift)
            else:
                self.redshift = redshift
        except IndexError:
            self.redshift = np.nan

    def _get_redshift_for_luminosity_calculation(self):
        if self.redshift is None:
            return self.redshift
        if np.isnan(self.redshift):
            logger.warning('This GRB has no measured redshift, using default z = 0.75')
            return 0.75
        return self.redshift

    def _set_t90(self):
        try:
            t90 = self.meta_data.query('GRB == @self._stripped_name')['BAT T90 [sec]'].values[0]
            if t90 == 0.:
                return np.nan
            self.t90 = self.__clean_string(t90)
        except IndexError:
            self.t90 = np.nan

    @staticmethod
    def __clean_string(string):
        for r in ["PL", "CPL", ",", "C", "~", " ", 'Gemini:emission', '()']:
            string = string.replace(r, "")
        return float(string)

    def analytical_flux_to_luminosity(self):
        self._convert_flux_to_luminosity(conversion_method="analytical")

    def numerical_flux_to_luminosity(self, counts_to_flux_absorbed, counts_to_flux_unabsorbed):
        self._convert_flux_to_luminosity(
            counts_to_flux_absorbed=counts_to_flux_absorbed, counts_to_flux_unabsorbed=counts_to_flux_unabsorbed,
            conversion_method="numerical")

    def _convert_flux_to_luminosity(self, conversion_method="analytical",
                                    counts_to_flux_absorbed=1, counts_to_flux_unabsorbed=1):
        if self.luminosity_data:
            logger.warning('The data is already in luminosity mode, returning.')
            return
        elif not self.flux_data:
            logger.warning(f'The data needs to be in flux mode, but is in {self.data_mode}.')
            return
        redshift = self._get_redshift_for_luminosity_calculation()
        if redshift is None:
            return
        converter = FluxToLuminosityConverter(
            redshift=redshift, photon_index=self.photon_index, time=self.time, time_err=self.time_err,
            flux=self.flux, flux_err=self.flux_err, counts_to_flux_absorbed=counts_to_flux_absorbed,
            counts_to_flux_unabsorbed=counts_to_flux_unabsorbed, conversion_method=conversion_method)
        self.data_mode = "luminosity"
        self.x, self.x_err, self.y, self.y_err = converter.convert_flux_to_luminosity()
        self._save_luminosity_data()

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


class Truncator(object):

    TRUNCATE_METHODS = ['prompt_time_error', 'left_of_max', 'default']

    def __init__(self, x, x_err, y, y_err, time, time_err, truncate_method='prompt_time_error'):
        self.x = x
        self.x_err = x_err
        self.y = y
        self.y_err = y_err
        self.time = time
        self.time_err = time_err
        self.truncate_method = truncate_method

    def truncate(self):
        if self.truncate_method == 'prompt_time_error':
            return self.truncate_prompt_time_error()
        elif self.truncate_method == 'left_of_max':
            return self.truncate_left_of_max()
        else:
            return self.truncate_default()

    def truncate_prompt_time_error(self):
        mask1 = self.x_err[0, :] > 0.0025
        mask2 = self.x < 2.0  # dont truncate if data point is after 2.0 seconds
        mask = np.logical_and(mask1, mask2)
        self.x = self.x[~mask]
        self.x_err = self.x_err[:, ~mask]
        self.y = self.y[~mask]
        self.y_err = self.y_err[:, ~mask]
        return self.x, self.x_err, self.y, self.y_err

    def truncate_left_of_max(self):
        max_index = np.argmax(self.y)
        self.x = self.x[max_index:]
        self.x_err = self.x_err[:, max_index:]
        self.y = self.y[max_index:]
        self.y_err = self.y_err[:, max_index:]
        return self.x, self.x_err, self.y, self.y_err

    def truncate_default(self):
        truncate = self.time_err[0, :] > 0.1
        to_del = len(self.time) - (len(self.time[truncate]) + 2)
        self.x = self.x[to_del:]
        self.x_err = self.x_err[:, to_del:]
        self.y = self.y[to_del:]
        self.y_err = self.y_err[:, to_del:]
        return self.x, self.x_err, self.y, self.y_err


class FluxToLuminosityConverter(object):

    CONVERSION_METHODS = ["analytical", "numerical"]

    def __init__(self, redshift, photon_index, time, time_err, flux, flux_err,
                 counts_to_flux_absorbed=1, counts_to_flux_unabsorbed=1, conversion_method="analytical"):
        self.redshift = redshift
        self.photon_index = photon_index
        self.time = time
        self.time_err = time_err
        self.flux = flux
        self.flux_err = flux_err
        self.counts_to_flux_absorbed = counts_to_flux_absorbed
        self.counts_to_flux_unabsorbed = counts_to_flux_unabsorbed
        self.conversion_method = conversion_method

    @property
    def counts_to_flux_fraction(self):
        return self.counts_to_flux_unabsorbed/self.counts_to_flux_absorbed

    @property
    def luminosity_distance(self):
        return cosmo.luminosity_distance(self.redshift).cgs.value

    def get_isotropic_bolometric_flux(self, k_corr):
        return (self.luminosity_distance ** 2.) * 4. * np.pi * k_corr

    def get_k_correction(self):
        if self.conversion_method == "analytical":
            return (1 + self.redshift) ** (self.photon_index - 2)
        elif self.conversion_method == "numerical":
            try:
                from sherpa.astro import ui as sherpa
            except ImportError as e:
                logger.warning(e)
                logger.warning("Can't perform numerical flux to luminosity calculation")
            Ecut = 1000
            obs_elow = 0.3
            obs_ehigh = 10

            bol_elow = 1.  # bolometric restframe low frequency in keV
            bol_ehigh = 10000.  # bolometric restframe high frequency in keV

            sherpa.dataspace1d(obs_elow, bol_ehigh, 0.01)
            sherpa.set_source(sherpa.bpl1d.band)
            band.gamma1 = self.photon_index  # noqa
            band.gamma2 = self.photon_index  # noqa
            band.eb = Ecut  # noqa
            return sherpa.calc_kcorr(self.redshift, obs_elow, obs_ehigh, bol_elow, bol_ehigh, id=1)

    def convert_flux_to_luminosity(self):
        k_corr = self.get_k_correction()
        self._calculate_rest_frame_time_and_luminosity(
            counts_to_flux_fraction=1,
            isotropic_bolometric_flux=self.get_isotropic_bolometric_flux(k_corr=k_corr),
            redshift=self.redshift)
        return self.time_rest_frame, self.time_rest_frame_err, self.Lum50, self.Lum50_err

    def _calculate_rest_frame_time_and_luminosity(self, counts_to_flux_fraction, isotropic_bolometric_flux, redshift):
        self.Lum50 = self.flux * counts_to_flux_fraction * isotropic_bolometric_flux * 1e-50
        self.Lum50_err = self.flux_err * isotropic_bolometric_flux * 1e-50
        self.time_rest_frame = self.time / (1 + redshift)
        self.time_rest_frame_err = self.time_err / (1 + redshift)


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