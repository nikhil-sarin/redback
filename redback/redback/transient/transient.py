import numpy as np
from redback.redback.utils import logger
from astropy.cosmology import Planck18 as cosmo


class Transient(object):

    DATA_MODES = []

    def __init__(self, time, time_err, y, y_err=None, redshift=np.nan, data_mode=None, name='', path='.'):
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

        self.photon_index = None

    @property
    def luminosity_data(self):
        return self.data_mode == 'luminosity'

    @property
    def flux_data(self):
        return self.data_mode == 'flux'

    @property
    def fluxdensity_data(self):
        return self.data_mode == 'flux_density'

    @property
    def photometry_data(self):
        return self.data_mode == 'photometry'

    @property
    def tte_data(self):
        return self.data_mode == 'tte_data'

    @property
    def counts_data(self):
        return self.data_mode == 'counts'

    @property
    def x(self):
        if self.luminosity_data:
            return self.time_rest_frame
        elif self.fluxdensity_data or self.flux_data or self.counts_data:
            return self.time
        else:
            raise ValueError

    @x.setter
    def x(self, x):
        if self.luminosity_data:
            self.time_rest_frame = x
        elif self.fluxdensity_data or self.flux_data or self.counts_data:
            self.time = x

    @property
    def x_err(self):
        if self.luminosity_data:
            return self.time_rest_frame_err
        elif self.fluxdensity_data or self.flux_data or self.counts_data:
            return self.time_err
        else:
            raise ValueError

    @x_err.setter
    def x_err(self, x_err):
        if self.luminosity_data:
            self.time_rest_frame_err = x_err
        elif self.fluxdensity_data or self.flux_data or self.counts_data:
            self.time_err = x_err

    @property
    def y(self):
        if self.luminosity_data:
            return self.Lum50
        elif self.flux_data:
            return self.flux
        elif self.fluxdensity_data:
            return self.flux_density
        elif self.counts_data:
            return self.counts
        else:
            raise ValueError

    @y.setter
    def y(self, y):
        if self.luminosity_data:
            self.Lum50 = y
        elif self.flux_data:
            self.flux = y
        elif self.fluxdensity_data:
            self.flux_density = y
        elif self.counts_data:
            self.counts = y
        else:
            raise ValueError

    @property
    def y_err(self):
        if self.luminosity_data:
            return self.Lum50_err
        elif self.flux_data:
            return self.flux_err
        elif self.fluxdensity_data:
            return self.flux_density_err
        else:
            raise ValueError

    @y_err.setter
    def y_err(self, y_err):
        if self.luminosity_data:
            self.Lum50_err = y_err
        elif self.flux_data:
            self.flux_err = y_err
        elif self.fluxdensity_data:
            self.flux_density_err = y_err
        elif self.counts_data:
            self.counts_err = y_err
        else:
            raise ValueError

    @property
    def _counts_err(self):
        return np.sqrt(self.counts)

    @property
    def data_mode(self):
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode):
        if data_mode in self.DATA_MODES:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    # @classmethod
    # def simulate_transient_object(cls):
    #     return transient_object
        
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
        elif self.data_mode == 'luminosity':
            logger.warning('The data is already in luminosity mode, returning.')
            return None
        elif self.data_mode == 'flux_density':
            logger.warning(f'The data needs to be in flux mode, but is in {self.data_mode}.')
            return None
        else:
            return self.redshift

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
