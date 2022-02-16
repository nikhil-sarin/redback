from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import pandas as pd
from typing import Union

from astropy.cosmology import Planck18 as cosmo  # noqa

from redback.utils import logger
from redback.getdata import afterglow_directory_structure
from redback.transient.transient import Transient

dirname = os.path.dirname(__file__)


class Afterglow(Transient):

    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'photometry']

    def __init__(
            self, name: str, data_mode: str = 'flux', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux: np.ndarray = None, flux_err: np.ndarray = None, flux_density: np.ndarray = None,
            flux_density_err: np.ndarray = None, magnitude: np.ndarray = None, magnitude_err: np.ndarray = None,
            frequency: np.ndarray = None, bands: np.ndarray = None, system: np.ndarray = None,
            active_bands: Union[np.ndarray, str] = 'all', use_phase_model: bool = False, **kwargs: dict) -> None:

        """
        This is a general constructor for the Afterglow class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).


        Parameters
        ----------
        name: str
            Telephone number of SGRB, e.g., 'GRB140903A' or '140903A' are valid inputs
        data_mode: str, optional
            Data mode. Must be one from `Afterglow.DATA_MODES`.
        time: np.ndarray, optional
            Times in the observer frame.
        time_err: np.ndarray, optional
            Time errors in the observer frame.
        time_mjd: np.ndarray, optional
            Times in MJD. Used if using phase model.
        time_mjd_err: np.ndarray, optional
            Time errors in MJD. Used if using phase model.
        time_rest_frame: np.ndarray, optional
            Times in the rest frame. Used for luminosity data.
        time_rest_frame_err: np.ndarray, optional
            Time errors in the rest frame. Used for luminosity data.
        Lum50: np.ndarray, optional
            Luminosity values.
        Lum50_err: np.ndarray, optional
            Luminosity error values.
        flux: np.ndarray, optional
            Flux values.
        flux_err: np.ndarray, optional
            Flux error values.
        flux_density: np.ndarray, optional
            Flux density values.
        flux_density_err: np.ndarray, optional
            Flux density error values.
        magnitude: np.ndarray, optional
            Magnitude values for photometry data.
        magnitude_err: np.ndarray, optional
            Magnitude error values for photometry data.
        counts: np.ndarray, optional
            Counts for prompt data.
        ttes: np.ndarray, optional
            Time-tagged events data for unbinned prompt data.
        bin_size: float
            Bin size for binning time-tagged event data.
        redshift: float
            Redshift value.
        path: str
            Path to data directory.
        photon_index: float
            Photon index value.
        use_phase_model: bool
            Whether we are using a phase model.
        frequency: np.ndarray, optional
            Array of band frequencies in photometry data.
        system: np.ndarray, optional
            System values.
        bands: np.ndarray, optional
            Band values.
        active_bands: Union[list, np.ndarray]
            List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
        kwargs: dict, optional
            Additional classes that can be customised to fulfil the truncation on flux to luminosity conversion:
            FluxToLuminosityConverter: Conversion class to convert fluxes to luminosities.
                                       If not given use `FluxToLuminosityConverter` in this module.

            Truncator: Truncation class that truncates the data. If not given use `Truncator` in this module.
        """

        if not name.startswith('GRB'):
            name = 'GRB' + name

        self.FluxToLuminosityConverter = kwargs.get('FluxToLuminosityConverter', FluxToLuminosityConverter)
        self.Truncator = kwargs.get('Truncator', Truncator)

        super().__init__(name=name, data_mode=data_mode, time=time, time_mjd=time_mjd, time_mjd_err=time_mjd_err,
                         time_err=time_err, time_rest_frame=time_rest_frame, time_rest_frame_err=time_rest_frame_err,
                         Lum50=Lum50, Lum50_err=Lum50_err, flux=flux, flux_err=flux_err, flux_density=flux_density,
                         flux_density_err=flux_density_err, use_phase_model=use_phase_model, magnitude=magnitude,
                         magnitude_err=magnitude_err, frequency=frequency,
                         system=system, bands=bands, active_bands=active_bands, **kwargs)
        self._set_data()
        self._set_photon_index()
        self._set_t90()
        self._get_redshift()

    @classmethod
    def from_swift_grb(
            cls, name: str, data_mode: str = 'flux', truncate: bool = True,
            truncate_method: str = 'prompt_time_error') -> Afterglow:
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
    def _stripped_name(self) -> str:
        return self.name.lstrip('GRB')

    def load_and_truncate_data(
            self, truncate: bool = True, truncate_method: str = 'prompt_time_error', data_mode: str = 'flux') -> None:
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
    def load_data(name: str, data_mode: str = None) -> tuple:
        grb_dir, _, _ = afterglow_directory_structure(grb=name.lstrip('GRB'), data_mode=data_mode)
        filename = f"{name}.csv"

        data_file = join(grb_dir, filename)
        data = np.genfromtxt(data_file, delimiter=",")[1:]
        x = data[:, 0]
        x_err = data[:, 1:3].T
        y = np.array(data[:, 3])
        y_err = np.array(np.abs(data[:, 4:6].T))
        return x, x_err, y, y_err

    def truncate(self, truncate_method: str = 'prompt_time_error') -> None:
        truncator = self.Truncator(x=self.x, x_err=self.x_err, y=self.y, y_err=self.y_err, time=self.time,
                                   time_err=self.time_err, truncate_method=truncate_method)
        self.x, self.x_err, self.y, self.y_err = truncator.truncate()

    @property
    def event_table(self) -> str:
        return os.path.join(dirname, f'../tables/{self.__class__.__name__}_table.txt')

    def _save_luminosity_data(self) -> None:
        grb_dir, _, _ = afterglow_directory_structure(grb=self._stripped_name, data_mode=self.data_mode)
        filename = f"{self.name}.csv"
        data = {"Time in restframe [s]": self.time_rest_frame,
                "Pos. time err in restframe [s]": self.time_rest_frame_err[0, :],
                "Neg. time err in restframe [s]": self.time_rest_frame_err[1, :],
                "Luminosity [10^50 erg s^{-1}]": self.Lum50,
                "Pos. luminosity err [10^50 erg s^{-1}]": self.Lum50_err[0, :],
                "Neg. luminosity err [10^50 erg s^{-1}]": self.Lum50_err[1, :]}
        df = pd.DataFrame(data=data)
        df.to_csv(join(grb_dir, filename), index=False)

    def _set_data(self) -> None:
        meta_data = pd.read_csv(self.event_table, header=0, error_bad_lines=False, delimiter='\t', dtype='str')
        meta_data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = meta_data[
                  'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
        self.meta_data = meta_data

    def _set_photon_index(self) -> None:
        try:
            photon_index = self.meta_data.query('GRB == @self._stripped_name')[
                'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].values[0]
            self.photon_index = self.__clean_string(photon_index)
        except IndexError:
            self.photon_index = np.nan

    def _get_redshift(self) -> None:
        # some GRBs dont have measurements
        try:
            redshift = self.meta_data.query('GRB == @self._stripped_name')['Redshift'].values[0]
            if isinstance(redshift, str):
                self.redshift = self.__clean_string(redshift)
            else:
                self.redshift = redshift
        except IndexError:
            self.redshift = np.nan

    def _get_redshift_for_luminosity_calculation(self) -> Union[float, None]:
        if self.redshift is None:
            return self.redshift
        if np.isnan(self.redshift):
            logger.warning('This GRB has no measured redshift, using default z = 0.75')
            return 0.75
        return self.redshift

    def _set_t90(self) -> None:
        try:
            t90 = self.meta_data.query('GRB == @self._stripped_name')['BAT T90 [sec]'].values[0]
            if t90 == 0.:
                return np.nan
            self.t90 = self.__clean_string(t90)
        except IndexError:
            self.t90 = np.nan

    @staticmethod
    def __clean_string(string: str) -> float:
        for r in ["PL", "CPL", ",", "C", "~", " ", 'Gemini:emission', '()']:
            string = string.replace(r, "")
        return float(string)

    def analytical_flux_to_luminosity(self) -> None:
        self._convert_flux_to_luminosity(conversion_method="analytical")

    def numerical_flux_to_luminosity(self, counts_to_flux_absorbed: float, counts_to_flux_unabsorbed: float) -> None:
        self._convert_flux_to_luminosity(
            counts_to_flux_absorbed=counts_to_flux_absorbed, counts_to_flux_unabsorbed=counts_to_flux_unabsorbed,
            conversion_method="numerical")

    def _convert_flux_to_luminosity(
            self, conversion_method: str = "analytical", counts_to_flux_absorbed: float = 1.,
            counts_to_flux_unabsorbed: float = 1.) -> None:
        if self.luminosity_data:
            logger.warning('The data is already in luminosity mode, returning.')
            return
        elif not self.flux_data:
            logger.warning(f'The data needs to be in flux mode, but is in {self.data_mode}.')
            return
        redshift = self._get_redshift_for_luminosity_calculation()
        if redshift is None:
            return
        converter = self.FluxToLuminosityConverter(
            redshift=redshift, photon_index=self.photon_index, time=self.time, time_err=self.time_err,
            flux=self.flux, flux_err=self.flux_err, counts_to_flux_absorbed=counts_to_flux_absorbed,
            counts_to_flux_unabsorbed=counts_to_flux_unabsorbed, conversion_method=conversion_method)
        self.data_mode = "luminosity"
        self.x, self.x_err, self.y, self.y_err = converter.convert_flux_to_luminosity()
        self._save_luminosity_data()

    def plot_data(self, axes: matplotlib.axes.Axes = None, colour: str = 'k') -> matplotlib.axes.Axes:
        """
        plots the data
        GRB is the telephone number of the GRB
        :param axes:
        :param colour:
        """

        x_err = [np.abs(self.x_err[1, :]), self.x_err[0, :]]
        y_err = [np.abs(self.y_err[1, :]), self.y_err[0, :]]

        ax = axes or plt.gca()
        ax.errorbar(self.x, self.y, xerr=x_err, yerr=y_err,
                    fmt='x', c=colour, ms=1, elinewidth=2, capsize=0.)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(0.5 * self.x[0], 2 * (self.x[-1] + x_err[1][-1]))
        ax.set_ylim(0.5 * min(self.y), 2. * np.max(self.y))

        ax.annotate(self.name, xy=(0.95, 0.9), xycoords='axes fraction',
                    horizontalalignment='right', size=20)

        ax.set_xlabel(r'Time since burst [s]')
        ax.set_ylabel(self.ylabel)
        ax.tick_params(axis='x', pad=10)

        if axes is None:
            plt.tight_layout()

        grb_dir, _, _ = afterglow_directory_structure(grb=self._stripped_name, data_mode=self.data_mode)
        filename = f"{self.name}_lc.png"
        plt.savefig(join(grb_dir, filename))
        if axes is None:
            plt.clf()
        return ax

    def plot_multiband(self, axes: matplotlib.axes.Axes = None, colour: str = 'k') -> None:
        if self.data_mode != 'flux_density':
            logger.warning('why are you doing this')


class SGRB(Afterglow):
    pass


class LGRB(Afterglow):
    pass


class Truncator(object):

    TRUNCATE_METHODS = ['prompt_time_error', 'left_of_max', 'default']

    def __init__(
            self, x: np.ndarray, x_err: np.ndarray, y: np.ndarray, y_err: np.ndarray, time: np.ndarray,
            time_err: np.ndarray, truncate_method: str = 'prompt_time_error') -> None:
        self.x = x
        self.x_err = x_err
        self.y = y
        self.y_err = y_err
        self.time = time
        self.time_err = time_err
        self.truncate_method = truncate_method

    def truncate(self) -> tuple:
        if self.truncate_method == 'prompt_time_error':
            return self.truncate_prompt_time_error()
        elif self.truncate_method == 'left_of_max':
            return self.truncate_left_of_max()
        else:
            return self.truncate_default()

    def truncate_prompt_time_error(self) -> tuple:
        mask1 = self.x_err[0, :] > 0.0025
        mask2 = self.x < 2.0  # dont truncate if data point is after 2.0 seconds
        mask = np.logical_and(mask1, mask2)
        self.x = self.x[~mask]
        self.x_err = self.x_err[:, ~mask]
        self.y = self.y[~mask]
        self.y_err = self.y_err[:, ~mask]
        return self.x, self.x_err, self.y, self.y_err

    def truncate_left_of_max(self) -> tuple:
        return self._truncate_by_index(index=np.argmax(self.y))

    def truncate_default(self) -> tuple:
        truncate = self.time_err[0, :] > 0.1
        index = len(self.time) - (len(self.time[truncate]) + 2)
        return self._truncate_by_index(index=index)

    def _truncate_by_index(self, index: Union[int, np.ndarray]) -> tuple:
        self.x = self.x[index:]
        self.x_err = self.x_err[:, index:]
        self.y = self.y[index:]
        self.y_err = self.y_err[:, index:]
        return self.x, self.x_err, self.y, self.y_err


class FluxToLuminosityConverter(object):

    CONVERSION_METHODS = ["analytical", "numerical"]

    def __init__(
            self, redshift: float, photon_index: float, time: np.ndarray, time_err: np.ndarray, flux: np.ndarray,
            flux_err: np.ndarray, counts_to_flux_absorbed: float = 1., counts_to_flux_unabsorbed: float = 1.,
            conversion_method: str = "analytical") -> None:
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
    def counts_to_flux_fraction(self) -> float:
        return self.counts_to_flux_unabsorbed/self.counts_to_flux_absorbed

    @property
    def luminosity_distance(self) -> float:
        return cosmo.luminosity_distance(self.redshift).cgs.value

    def get_isotropic_bolometric_flux(self, k_corr: float) -> float:
        return (self.luminosity_distance ** 2.) * 4. * np.pi * k_corr

    def get_k_correction(self) -> Union[float, None]:
        if self.conversion_method == "analytical":
            return (1 + self.redshift) ** (self.photon_index - 2)
        elif self.conversion_method == "numerical":
            try:
                from sherpa.astro import ui as sherpa
            except ImportError as e:
                logger.warning(e)
                logger.warning("Can't perform numerical flux to luminosity calculation")
                return
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

    def convert_flux_to_luminosity(self) -> tuple:
        k_corr = self.get_k_correction()
        self._calculate_rest_frame_time_and_luminosity(
            counts_to_flux_fraction=1,
            isotropic_bolometric_flux=self.get_isotropic_bolometric_flux(k_corr=k_corr),
            redshift=self.redshift)
        return self.time_rest_frame, self.time_rest_frame_err, self.Lum50, self.Lum50_err

    def _calculate_rest_frame_time_and_luminosity(
            self, counts_to_flux_fraction: float, isotropic_bolometric_flux: float, redshift: float) -> None:
        self.Lum50 = self.flux * counts_to_flux_fraction * isotropic_bolometric_flux * 1e-50
        self.Lum50_err = self.flux_err * isotropic_bolometric_flux * 1e-50
        self.time_rest_frame = self.time / (1 + redshift)
        self.time_rest_frame_err = self.time_err / (1 + redshift)
