import numpy as np
from redback.utils import logger
from redback.utils import DataModeSwitch


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
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    def plot_data(self, axes=None, colour='k'):
        pass

    def plot_lightcurve(self, model, axes=None, plot_save=True, plot_show=True, random_models=1000,
                        posterior=None, outdir=None, **kwargs):
        pass
