import numpy as np
from redback.utils import logger
from redback.utils import DataModeSwitch
from ..utils import bin_ttes


class Transient(object):

    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'photometry', 'counts', 'ttes']
    _ATTRIBUTE_NAME_DICT = dict(luminosity="Lum50", flux="flux", flux_density="flux_density",
                                counts="counts", photometry="magnitude")

    luminosity_data = DataModeSwitch('luminosity')
    flux_data = DataModeSwitch('flux')
    flux_density_data = DataModeSwitch('flux_density')
    photometry_data = DataModeSwitch('photometry')
    counts_data = DataModeSwitch('counts')
    tte_data = DataModeSwitch('ttes')

    def __init__(self, time, time_err=None, time_rest_frame=None, time_rest_frame_err=None, Lum50=None, Lum50_err=None,
                 flux=None, flux_err=None, flux_density=None, flux_density_err=None, magnitude=None, magnitude_err=None,
                 counts=None, ttes=None, bin_size=None, redshift=np.nan, data_mode=None, name='', path='.',
                 photon_index=np.nan):
        """
        Base class for all transients
        """
        self.bin_size = bin_size
        if data_mode == 'ttes':
            time, counts = bin_ttes(ttes, self.bin_size)

        self.time = time
        self.time_err = time_err
        self.time_rest_frame = time_rest_frame
        self.time_rest_frame_err = time_rest_frame_err

        self.Lum50 = Lum50
        self.Lum50_err = Lum50_err
        self.flux = flux
        self.flux_err = flux_err
        self.flux_density = flux_density
        self.flux_density_err = flux_density_err
        self.magnitude = magnitude
        self.magnitude_err = magnitude_err
        self.counts = counts
        self.counts_err = np.sqrt(counts) if counts is not None else None
        self.ttes = ttes

        self.data_mode = data_mode
        self.redshift = redshift
        self.name = name
        self.path = path

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
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    @property
    def ylabel(self):
        if self.luminosity_data:
            return r'Luminosity [$10^{50}$ erg s$^{-1}$]'
        elif self.photometry_data:
            return r'Magnitude'
        elif self.flux_data:
            return r'Flux [erg cm$^{-2}$ s$^{-1}$]'
        elif self.flux_density_data:
            return r'Flux density [mJy]'
        elif self.counts_data:
            return r'Counts'
        else:
            raise ValueError

    def plot_data(self, axes=None, colour='k'):
        pass

    def plot_lightcurve(self, model, axes=None, plot_save=True, plot_show=True, random_models=1000,
                        posterior=None, outdir=None, **kwargs):
        pass
