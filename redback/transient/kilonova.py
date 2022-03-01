import os
from typing import Union

import numpy as np

import redback.get_data.directory
from redback.transient.transient import OpticalTransient

dirname = os.path.dirname(__file__)


class Kilonova(OpticalTransient):

    DATA_MODES = ['flux_density', 'photometry', 'luminosity']

    def __init__(
            self, name: str, data_mode: str = 'photometry', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux_density: np.ndarray = None, flux_density_err: np.ndarray = None, magnitude: np.ndarray = None,
            magnitude_err: np.ndarray = None, bands: np.ndarray = None, system: np.ndarray = None,
            active_bands: Union[np.ndarray, str] = 'all', use_phase_model: bool = False, **kwargs: dict) -> None:
        """

        This is a general constructor for the Kilonova class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

        Parameters
        ----------
        name: str
            Name of the transient
        data_mode: str
            Data mode. Must be one from `Kilonova.DATA_MODES`.
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
        system: np.ndarray, optional
            System values.
        bands: np.ndarray, optional
            Band values.
        active_bands: Union[list, np.ndarray]
            List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
        use_phase_model: bool, optional
            Whether we are using a phose model. Default is `False`
        kwargs: dict
            Any additional kwargs.
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, time_rest_frame_err=time_rest_frame_err, Lum50=Lum50,
                         Lum50_err=Lum50_err, flux_density=flux_density, flux_density_err=flux_density_err,
                         magnitude=magnitude, magnitude_err=magnitude_err, data_mode=data_mode, name=name, bands=bands,
                         system=system, active_bands=active_bands, use_phase_model=use_phase_model, **kwargs)
        self.directory_structure = redback.get_data.directory.transient_directory_structure(
            transient=name, transient_type="kilonova", data_mode=data_mode)
        self._set_data()
