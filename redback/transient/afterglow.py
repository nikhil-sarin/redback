from __future__ import annotations

import os
from os.path import join
from typing import Union
import re

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo  # noqa

from redback.get_data.directory import afterglow_directory_structure
from redback.transient.transient import Transient
from redback.utils import logger

dirname = os.path.dirname(__file__)


class Afterglow(Transient):
    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'magnitude']

    def __init__(
            self, name: str, data_mode: str = 'flux', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux: np.ndarray = None, flux_err: np.ndarray = None, flux_density: np.ndarray = None,
            flux_density_err: np.ndarray = None, magnitude: np.ndarray = None, magnitude_err: np.ndarray = None,
            redshift: float = np.nan, photon_index: float = np.nan, frequency: np.ndarray = None,
            bands: np.ndarray = None, system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all',
            plotting_order: Union[np.ndarray, str] = None, use_phase_model: bool = False,
            optical_data: bool = False, **kwargs: None) -> None:

        """
        This is a general constructor for the Afterglow class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

        :param name: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs
        :type name: str
        :param data_mode: Data mode. Must be one from `Afterglow.DATA_MODES`.
        :type data_mode: str, optional
        :param time: Times in the observer frame.
        :type time: np.ndarray, optional
        :param time_err: Time errors in the observer frame.
        :type time_err: np.ndarray, optional
        :param time_mjd: Times in MJD. Used if using phase model.
        :type time_mjd: np.ndarray, optional
        :param time_mjd_err: Time errors in MJD. Used if using phase model.
        :type time_mjd_err: np.ndarray, optional
        :param time_rest_frame: Times in the rest frame. Used for luminosity data.
        :type time_rest_frame: np.ndarray, optional
        :param time_rest_frame_err: Time errors in the rest frame. Used for luminosity data.
        :type time_rest_frame_err: np.ndarray, optional
        :param Lum50: Luminosity values.
        :type Lum50: np.ndarray, optional
        :param Lum50_err: Luminosity error values.
        :type Lum50_err: np.ndarray, optional
        :param flux: Flux values.
        :type flux: np.ndarray, optional
        :type flux_err: np.ndarray, optional
        :param flux_err: Flux error values.
        :param flux_density: Flux density values.
        :type flux_density: np.ndarray, optional
        :param flux_density_err: Flux density error values.
        :type flux_density_err: np.ndarray, optional
        :param magnitude: Magnitude values for photometry data.
        :type magnitude: np.ndarray, optional
        :param magnitude_err: Magnitude error values for photometry data.
        :type magnitude_err: np.ndarray, optional
        :param redshift: Redshift value. Will be read from the metadata table if not given.
        :type redshift: float
        :param photon_index: Photon index value. Will be read from the metadata table if not given.
        :type photon_index: float
        :param use_phase_model: Whether we are using a phase model.
        :type use_phase_model: bool
        :param optical_data: Whether we are fitting optical data, useful for plotting.
        :type optical_data: bool, optional
        :param frequency: Array of band frequencies in photometry data.
        :type frequency: np.ndarray, optional
        :param system: System values.
        :type system: np.ndarray, optional
        :param bands: Band values.
        :type bands: np.ndarray, optional
        :param active_bands: List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
        :type active_bands: Union[list, np.ndarray]
        :param plotting_order: Order in which to plot the bands/and how unique bands are stored.
        :type plotting_order: Union[np.ndarray, str], optional
        :param kwargs:
            Additional classes that can be customised to fulfil the truncation on flux to luminosity conversion:
            FluxToLuminosityConverter: Conversion class to convert fluxes to luminosities.
                                       If not given use `FluxToLuminosityConverter` in this module.
            Truncator: Truncation class that truncates the data. If not given use `Truncator` in this module.
        :type kwargs: None, optional
        """

        name = f"GRB{name.lstrip('GRB')}"

        self.FluxToLuminosityConverter = kwargs.get('FluxToLuminosityConverter', FluxToLuminosityConverter)
        self.Truncator = kwargs.get('Truncator', Truncator)

        super().__init__(name=name, data_mode=data_mode, time=time, time_mjd=time_mjd, time_mjd_err=time_mjd_err,
                         time_err=time_err, time_rest_frame=time_rest_frame, time_rest_frame_err=time_rest_frame_err,
                         Lum50=Lum50, Lum50_err=Lum50_err, flux=flux, flux_err=flux_err, flux_density=flux_density,
                         flux_density_err=flux_density_err, use_phase_model=use_phase_model, optical_data=optical_data,
                         magnitude=magnitude, magnitude_err=magnitude_err, frequency=frequency, redshift=redshift,
                         photon_index=photon_index, system=system, bands=bands, active_bands=active_bands,
                         plotting_order=plotting_order, **kwargs)
        self._set_data()
        self._set_photon_index()
        self._set_t90()
        self._get_redshift()
        self.directory_structure = afterglow_directory_structure(grb=self.name, data_mode=self.data_mode, instrument="")

    @classmethod
    def from_swift_grb(
            cls, name: str, data_mode: str = 'flux', truncate: bool = True,
            truncate_method: str = 'prompt_time_error', **kwargs) -> Afterglow:
        """

        :param name: Telephone number of SGRB, e.g., 'GRB140903A' or '140903A' are valid inputs
        :type name: str
        :param data_mode: Data mode. Must be one from `Afterglow.DATA_MODES`. (Default value = 'flux')
        :type data_mode: str, optional
        :param truncate: Whether to truncate the data. (Default value = True)
        :type truncate: bool
        :param truncate_method: Must be from `Truncator.TRUNCATE_METHODS`. (Default value = 'prompt_time_error')
        :type truncate_method: str
        :param kwargs: Additional keywords to pass into Afterglow.__init__
        :type kwargs: dict

        :return: The Afterglow object.
        :rtype: Afterglow

        """
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

    @property
    def data_mode(self) -> str:
        """
        :return: The currently active data mode (one in `Transient.DATA_MODES`)
        :rtype: str
        """
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        """
        :return: One of the data modes in `Transient.DATA_MODES`.
        :rtype: str
        """
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
            try:
                self.directory_structure = afterglow_directory_structure(
                    grb=self.name, data_mode=self.data_mode, instrument="")
            except AttributeError:
                pass
        else:
            raise ValueError("Unknown data mode.")

    def load_and_truncate_data(
            self, truncate: bool = True, truncate_method: str = 'prompt_time_error', data_mode: str = 'flux') -> None:
        """Read data of SGRB from given path and GRB telephone number.
        Truncate the data to get rid of all but the last prompt emission point
        make a cut based on the size of the temporal error; ie if t_error < 1s, the data point is
        part of the prompt emission

        :param truncate: Whether to truncate the data.
        :type truncate: bool
        :param truncate_method: Must be from `Truncator.TRUNCATE_METHODS`. (Default value = 'prompt_time_error')
        :type truncate_method: str
        :param data_mode: Data mode. Must be one from `Afterglow.DATA_MODES`. (Default value = 'flux')
        :type data_mode: str, optional
        """
        self.data_mode = data_mode
        self.x, self.x_err, self.y, self.y_err = self.load_data(name=self.name, data_mode=self.data_mode)
        if truncate:
            self.truncate(truncate_method=truncate_method)

    @staticmethod
    def load_data(name: str, data_mode: str = None) -> tuple:
        """Loads and returns data from a csv file

        :param name: Telephone number of SGRB, e.g., 'GRB140903A' or '140903A' are valid inputs
        :type name: str
        :param data_mode: Data mode. Must be one from `Afterglow.DATA_MODES`. (Default value = None)
        :type data_mode: str, optional

        :return: A tuple with x, x_err, y, y_err data
        :rtype: tuple
        """
        directory_structure = afterglow_directory_structure(grb=f"GRB{name.lstrip('GRB')}", data_mode=data_mode)

        data = np.genfromtxt(directory_structure.processed_file_path, delimiter=",")[1:]
        x = data[:, 0]
        x_err = data[:, 1:3].T
        y = np.array(data[:, 3])
        y_err = np.array(np.abs(data[:, 4:6].T))
        return x, x_err, y, y_err

    def truncate(self, truncate_method: str = 'prompt_time_error') -> None:
        """Truncate the data using the specified method. See `redback.transient.afterglow.Truncator` for
        documentation of the truncation methods.

        :param truncate_method: Must be from `Truncator.TRUNCATE_METHODS`. (Default value = 'prompt_time_error')
        :type truncate_method: str
        """
        truncator = self.Truncator(x=self.x, x_err=self.x_err, y=self.y, y_err=self.y_err, time=self.time,
                                   time_err=self.time_err, truncate_method=truncate_method)
        self.x, self.x_err, self.y, self.y_err = truncator.truncate()

    @property
    def event_table(self) -> str:
        """
        :return: Relative path to the event table.
        :rtype: str
        """
        return os.path.join(dirname, f'../tables/{self.__class__.__name__}_table.txt')

    def _save_luminosity_data(self) -> None:
        """Saves luminosity data to a csv file."""
        filename = f"{self.name}.csv"
        data = {"Time in restframe [s]": self.time_rest_frame,
                "Pos. time err in restframe [s]": self.time_rest_frame_err[0, :],
                "Neg. time err in restframe [s]": self.time_rest_frame_err[1, :],
                "Luminosity [10^50 erg s^{-1}]": self.Lum50,
                "Pos. luminosity err [10^50 erg s^{-1}]": self.Lum50_err[0, :],
                "Neg. luminosity err [10^50 erg s^{-1}]": self.Lum50_err[1, :]}
        df = pd.DataFrame(data=data)
        df.to_csv(join(self.directory_structure.directory_path, filename), index=False)

    def _set_data(self) -> None:
        """Loads data from the meta data table and sets it to the respective attribute."""
        try:
            meta_data = pd.read_csv(self.event_table, header=0, on_bad_lines='skip', delimiter='\t', dtype='str')
            meta_data['BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'] = meta_data[
                'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].fillna(0)
            self.meta_data = meta_data
        except FileNotFoundError:
            logger.info("Metadata does not exist for this event.")
            logger.info("Setting metadata to None. This is not an error, but a warning that no metadata could be found online.")
            self.meta_data = None

    def _set_photon_index(self) -> None:
        """Set the photon index attribute from the metadata table."""
        if not np.isnan(self.photon_index):
            return
        if self.magnitude_data or self.flux_density_data:
            self.photon_index = np.nan
        try:
            photon_index = self.meta_data.query('GRB == @self._stripped_name')[
                'BAT Photon Index (15-150 keV) (PL = simple power-law, CPL = cutoff power-law)'].values[0]
            self.photon_index = self.__clean_string(photon_index)
        except (AttributeError, IndexError):
            self.photon_index = np.nan

    def _get_redshift(self) -> None:
        """Set redshift from metadata table. Some GRBs do not have measurements."""
        if not np.isnan(self.redshift):
            return
        try:
            redshift = self.meta_data.query('GRB == @self._stripped_name')['Redshift'].values[0]
            if isinstance(redshift, str):
                self.redshift = self.__clean_string(redshift)
            else:
                self.redshift = redshift
        except (AttributeError, IndexError):
            self.redshift = np.nan

    def _get_redshift_for_luminosity_calculation(self) -> Union[float, None]:
        """Gets redshift or defaults to 0.75.

        :return: The redshift value.
        :rtype: Union[float, None]
        """
        if self.redshift is None:
            return self.redshift
        if np.isnan(self.redshift):
            logger.warning('This GRB has no measured redshift, using default z = 0.75')
            return 0.75
        return self.redshift

    def _set_t90(self) -> None:
        """Sets t90 value from meta data table."""
        try:
            t90 = self.meta_data.query('GRB == @self._stripped_name')['BAT T90 [sec]'].values[0]
            if t90 == 0.:
                return np.nan
            self.t90 = self.__clean_string(t90)
        except (AttributeError, IndexError):
            self.t90 = np.nan

    @staticmethod
    def __clean_string(string: str) -> float:
        """Removes superfluous characters from a string. Relevant for redshift, photon index, and t90 values.

        :param string: String to be cleaned.
        :type string: str
        :return: The cleaned string converted into a float.
        :rtype: float
        """
        try:
            for r in ["PL", "CPL", ",", "C", "~", " ", 'Gemini:emission', '()']:
                string = string.replace(r, "")
                new_float = float(string)
        except ValueError:
            new_float = float(re.findall("\d+\.\d+", string)[0])
        return new_float

    def analytical_flux_to_luminosity(self) -> None:
        """Converts flux to luminosity using the analytical method."""
        self._convert_flux_to_luminosity(conversion_method="analytical")

    def numerical_flux_to_luminosity(self, counts_to_flux_absorbed: float, counts_to_flux_unabsorbed: float) -> None:
        """Converts flux to luminosity using the numerical method.

        :param counts_to_flux_absorbed: Absorbed counts to flux ratio - a conversion of the count rate to flux.
        :type counts_to_flux_absorbed: float
        :param counts_to_flux_unabsorbed: Unabsorbed counts to flux ratio - a conversion of the count rate to flux.
        :type counts_to_flux_unabsorbed: float:
        """
        self._convert_flux_to_luminosity(
            counts_to_flux_absorbed=counts_to_flux_absorbed, counts_to_flux_unabsorbed=counts_to_flux_unabsorbed,
            conversion_method="numerical")

    def _convert_flux_to_luminosity(
            self, conversion_method: str = "analytical", counts_to_flux_absorbed: float = 1.,
            counts_to_flux_unabsorbed: float = 1.) -> None:
        """Converts flux to luminosity data. Redshift needs to be set. Changes data mode to luminosity and
        saves luminosity data.

        :param conversion_method: Either 'analytical' or 'numerical' with the standard `FluxToLuminosityConverter`.
        :type conversion_method: str, optional
        :param counts_to_flux_absorbed: Absorbed counts to flux ratio - a conversion of the count rate to flux.
                                        (Default value = 1.)
        :type counts_to_flux_absorbed: float
        :param counts_to_flux_unabsorbed: Unabsorbed counts to flux ratio - a conversion of the count rate to flux.
                                          (Default value = 1.)
        :type counts_to_flux_unabsorbed: float
        """
        if self.luminosity_data:
            logger.warning('The data is already in luminosity mode, returning.')
            return
        elif not self.flux_data:
            logger.warning(f'The data needs to be in flux mode, but is in {self.data_mode}.')
            return
        redshift = self._get_redshift_for_luminosity_calculation()
        if redshift is None:
            return
        self.data_mode = "luminosity"
        converter = self.FluxToLuminosityConverter(
            redshift=redshift, photon_index=self.photon_index, time=self.time, time_err=self.time_err,
            flux=self.flux, flux_err=self.flux_err, counts_to_flux_absorbed=counts_to_flux_absorbed,
            counts_to_flux_unabsorbed=counts_to_flux_unabsorbed, conversion_method=conversion_method)
        self.x, self.x_err, self.y, self.y_err = converter.convert_flux_to_luminosity()
        self._save_luminosity_data()


class SGRB(Afterglow):
    """ """
    pass


class LGRB(Afterglow):
    """ """
    pass


class Truncator(object):
    """ """
    TRUNCATE_METHODS = ['prompt_time_error', 'left_of_max', 'default']

    def __init__(
            self, x: np.ndarray, x_err: np.ndarray, y: np.ndarray, y_err: np.ndarray, time: np.ndarray,
            time_err: np.ndarray, truncate_method: str = 'prompt_time_error') -> None:
        """Truncation class for the truncation behaviour in `Afterglow`. This class can be subclassed and passed
        into `Afterglow` if user specific truncation is desired.

        :param x: X-axis (time) data.
        :type x: np.ndarray
        :param x_err: X-axis (time)  error data.
        :type x_err: np.ndarray
        :param y: Y-axis (flux/flux density/ counts) data
        :type y: np.ndarray
        :param y_err: Y-axis (flux/flux density/ counts) error data
        :type y_err: np.ndarray
        :param time: Time to be used for default truncation method.
        :type time: np.ndarray
        :param time_err: Time error to be used for default truncation method.
        :type time_err: np.ndarray
        :param truncate_method: Must be from Truncator.TRUNCATE_METHODS.
        :type truncate_method: str, optional
        """
        self.x = x
        self.x_err = x_err
        self.y = y
        self.y_err = y_err
        self.time = time
        self.time_err = time_err
        self.truncate_method = truncate_method

    def truncate(self) -> tuple:
        """Executes the truncation and returns data as a tuple.

        :return: The truncated data (x, x_err, y, y_err)
        :rtype: tuple
        """
        if self.truncate_method == 'prompt_time_error':
            return self.truncate_prompt_time_error()
        elif self.truncate_method == 'left_of_max':
            return self.truncate_left_of_max()
        else:
            return self.truncate_default()

    def truncate_prompt_time_error(self) -> tuple:
        """Truncate using the prompt time error method. Does not data points after 2.0 seconds.

        :return: The truncated data (x, x_err, y, y_err)
        :rtype: tuple
        """
        mask1 = self.x_err[0, :] > 0.0025
        mask2 = self.x < 2.0
        mask = np.logical_and(mask1, mask2)
        self.x = self.x[~mask]
        self.x_err = self.x_err[:, ~mask]
        self.y = self.y[~mask]
        self.y_err = self.y_err[:, ~mask]
        return self.x, self.x_err, self.y, self.y_err

    def truncate_left_of_max(self) -> tuple:
        """Truncate all data left of the maximum.

        :return: The truncated data (x, x_err, y, y_err)
        :rtype: tuple
        """
        return self._truncate_by_index(index=np.argmax(self.y))

    def truncate_default(self) -> tuple:
        """Truncate using the default method.

        :return: The truncated data (x, x_err, y, y_err)
        :rtype: tuple
        """
        truncate = self.time_err[0, :] > 0.1
        index = len(self.time) - (len(self.time[truncate]) + 2)
        return self._truncate_by_index(index=index)

    def _truncate_by_index(self, index: Union[int, np.ndarray]) -> tuple:
        """Truncate data left of a given index.

        :param index: The index at which to truncate.
        :type index: Union[int, np.ndarray]
        :return: The truncated data (x, x_err, y, y_err)
        :rtype: tuple
        """
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
        """
        Flux to luminosity conversion class for the conversion behaviour in `Afterglow`.
        This class can be subclassed and passed into `Afterglow` if user specific conversion is desired.

        :param redshift: The redshift value to use.
        :type redshift: np.ndarray
        :param photon_index: The photon index value to use.
        :type photon_index: np.ndarray
        :param time: Time data.
        :type time: np.ndarray
        :param time_err: Time error data.
        :type time_err: np.ndarray
        :param flux: Flux data.
        :type flux: np.ndarray
        :param flux_err: Flux error data.
        :type flux_err: np.ndarray
        :param counts_to_flux_absorbed: Absorbed counts to flux ratio - a conversion of the count rate to flux.
        :type counts_to_flux_absorbed: float
        :param counts_to_flux_unabsorbed: Unabsorbed counts to flux ratio - a conversion of the count rate to flux.
        :type counts_to_flux_unabsorbed: float
        :param conversion_method: The conversion method to use.
                                  Must be from `FluxToLuminosityConverter.CONVERSION_METHODS`
        :type conversion_method: str, optional
        """
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
        """Fraction of `counts_to_flux_absorbed` to `counts_to_flux_unabsorbed`.

        :return: The counts to flux fraction.
        :rtype: float
        """
        return self.counts_to_flux_unabsorbed / self.counts_to_flux_absorbed

    @property
    def luminosity_distance(self) -> float:
        """Luminosity distance given the redshift value."""
        return cosmo.luminosity_distance(self.redshift).cgs.value

    def get_isotropic_bolometric_flux(self, k_corr: float) -> float:
        """Calculates the isotropic bolometric flux given the k-correction

        :param k_corr: 
        :type k_corr: float
        """
        return (self.luminosity_distance ** 2.) * 4. * np.pi * k_corr

    def get_k_correction(self) -> Union[float, None]:
        """Calculates the k-correction depending on the conversion method.
        analytical: Use the redshift and the photon index.
        numerical: Call to `sherpa` package for the calculation.

        :return: The k-correction.
        :rtype: float
        """
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
        """Calculates k-correction and converts the flux to luminosity.

        :return: The rest frame times and luminosities in the format (x, x_err, y, y_err).
        :rtype: tuple
        """
        k_corr = self.get_k_correction()
        self._calculate_rest_frame_time_and_luminosity(
            counts_to_flux_fraction=self.counts_to_flux_fraction,
            isotropic_bolometric_flux=self.get_isotropic_bolometric_flux(k_corr=k_corr),
            redshift=self.redshift)
        return self.time_rest_frame, self.time_rest_frame_err, self.Lum50, self.Lum50_err

    def _calculate_rest_frame_time_and_luminosity(
            self, counts_to_flux_fraction: float, isotropic_bolometric_flux: float, redshift: float) -> None:
        """Carries out flux to luminosity conversion.

        :param counts_to_flux_fraction: Fraction of `counts_to_flux_absorbed` to `counts_to_flux_unabsorbed`.
        :type counts_to_flux_fraction: float
        :param isotropic_bolometric_flux: Isotropic bolometric flux.
        :type isotropic_bolometric_flux: float
        :param redshift: Redshift.
        :type redshift: float
        """
        self.Lum50 = self.flux * counts_to_flux_fraction * isotropic_bolometric_flux * 1e-50
        self.Lum50_err = self.flux_err * isotropic_bolometric_flux * 1e-50
        self.time_rest_frame = self.time / (1 + redshift)
        self.time_rest_frame_err = self.time_err / (1 + redshift)
