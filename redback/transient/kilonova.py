import os
from .transient import OpticalTransient

dirname = os.path.dirname(__file__)


class Kilonova(OpticalTransient):

    DATA_MODES = ['flux_density', 'photometry', 'luminosity']

    def __init__(self, name, data_mode='photometry', time=None, time_err=None, time_mjd=None, time_mjd_err=None,
                 time_rest_frame=None, time_rest_frame_err=None, Lum50=None, Lum50_err=None, flux_density=None,
                 flux_density_err=None, magnitude=None, magnitude_err=None, bands=None, system=None,
                 use_phase_model=False, **kwargs):

        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, time_rest_frame_err=time_rest_frame_err, Lum50=Lum50,
                         Lum50_err=Lum50_err, flux_density=flux_density, flux_density_err=flux_density_err,
                         magnitude=magnitude, magnitude_err=magnitude_err, data_mode=data_mode, name=name, bands=bands,
                         system=system, use_phase_model=use_phase_model, **kwargs)
        self._set_data()

