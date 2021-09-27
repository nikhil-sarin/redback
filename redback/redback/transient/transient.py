import numpy as np
from redback.redback.utils import logger
from redback.redback.utils import DataModeSwitch
from astropy.cosmology import Planck18 as cosmo


class Transient(object):

    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'photometry', 'counts', 'tte']
    _ATTRIBUTE_NAME_DICT = dict(luminosity="Lum50", flux="flux", flux_density="flux_density", counts="counts")

    luminosity_data = DataModeSwitch('luminosity')
    flux_data = DataModeSwitch('flux')
    flux_density_data = DataModeSwitch('flux_density')
    photometry_data = DataModeSwitch('photometry')
    counts_data = DataModeSwitch('counts')
    tte_data = DataModeSwitch('tte')

    def __init__(self, time, time_err, y, y_err=None, redshift=np.nan, data_mode=None, name='', path='.',
                 photon_index=np.nan):
        """
        Base class for all transients
        """

        self.time = time
        self.time_err = time_err
        self.time_rest_frame = np.array([])
        self.time_rest_frame_err = np.array([])
        self.tte = np.array([])

        self.Lum50 = np.array([])
        self.Lum50_err = np.array([])
        self.flux_density = np.array([])
        self.flux_density_err = np.array([])
        self.flux = np.array([])
        self.flux_err = np.array([])
        self.counts = np.array([])
        self.counts_err = np.array([])

        self.data_mode = data_mode
        self.redshift = redshift
        self.name = name
        self.path = path

        self.y = y
        self.y_err = y_err

        self.photon_index = photon_index

    @property
    def _time_attribute_name(self):
        if self.luminosity_data:
            return "time_rest_frame"
        return "time"

    @property
    def _time_err_attribute_name(self):
        return self._time_attribute_name + "_err"

    @property
    def _y_attribute_name(self):
        return self._ATTRIBUTE_NAME_DICT[self.data_mode]

    @property
    def _y_err_attribute_name(self):
        return self._ATTRIBUTE_NAME_DICT[self.data_mode] + "_err"

    @property
    def x(self):
        return getattr(self, self._time_attribute_name)

    @x.setter
    def x(self, x):
        setattr(self, self._time_attribute_name, x)

    @property
    def x_err(self):
        return getattr(self, self._time_err_attribute_name)

    @x_err.setter
    def x_err(self, x_err):
        setattr(self, self._time_err_attribute_name, x_err)

    @property
    def y(self):
        return getattr(self, self._y_attribute_name)

    @y.setter
    def y(self, y):
        setattr(self, self._y_attribute_name, y)

    @property
    def y_err(self):
        return getattr(self, self._y_err_attribute_name)

    @y_err.setter
    def y_err(self, y_err):
        setattr(self, self._y_err_attribute_name, y_err)

    @property
    def data_mode(self):
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode):
        if data_mode in self.DATA_MODES:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")
        
    def get_flux_density(self):
        pass

    def get_integrated_flux(self):
        pass

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
        if np.isnan(self.redshift):
            logger.warning('This GRB has no measured redshift, using default z = 0.75')
            return 0.75
        elif self.luminosity_data:
            logger.warning('The data is already in luminosity mode, returning.')
        elif self.flux_data:
            return self.redshift
        else:
            logger.warning(f'The data needs to be in flux mode, but is in {self.data_mode}.')

    def _calculate_rest_frame_time_and_luminosity(self, counts_to_flux_fraction, isotropic_bolometric_flux, redshift):
        self.Lum50 = self.flux * counts_to_flux_fraction * isotropic_bolometric_flux * 1e-50
        self.Lum50_err = self.flux_err * isotropic_bolometric_flux * 1e-50
        self.time_rest_frame = self.time / (1 + redshift)
        self.time_rest_frame_err = self.time_err / (1 + redshift)

    def _save_luminosity_data(self):
        pass

    def get_optical(self):
        pass

    def _process_data(self):
        pass

    def _set_photon_index(self):
        pass

    def _get_redshift(self):
        pass

    def _set_t90(self):
        pass

    def plot_data(self, axes=None, colour='k'):
        pass

    def plot_lightcurve(self, model, axes=None, plot_save=True, plot_show=True, random_models=1000,
                        posterior=None, outdir=None, **kwargs):
        pass
