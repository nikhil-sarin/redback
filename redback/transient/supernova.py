import numpy as np
from typing import Union

import redback.get_data.directory
from redback.transient.transient import OpticalTransient


class Supernova(OpticalTransient):
    DATA_MODES = ['flux', 'flux_density', 'magnitude', 'luminosity']

    def __init__(
            self, name: str, data_mode: str = 'magnitude', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux_density: np.ndarray = None, flux_density_err: np.ndarray = None, magnitude: np.ndarray = None,
            magnitude_err: np.ndarray = None, redshift: float = np.nan, photon_index: float = np.nan,
            bands: np.ndarray = None, system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False, **kwargs: None) -> None:
        """
        This is a general constructor for the Supernova class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

        Parameters
        ----------
        name: str
            Name of the transient.
        data_mode: str, optional
            Data mode. Must be one from `Supernova.DATA_MODES`.
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
        redshift: float, optional
            Redshift value.
        photon_index: float, optional
            Photon index value.
        frequency: np.ndarray, optional
            Array of band frequencies in photometry data.
        bands: np.ndarray, optional
            Band values.
        system: np.ndarray, optional
            System values.
        active_bands: Union[list, np.ndarray], optional
            List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
        use_phase_model: bool, optional
            Whether we are using a phase model.
        kwargs: dict, optional
            Additional callables:
            bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use
                                  redback.utils.bands_to_frequency if not given.
        """

        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, time_rest_frame_err=time_rest_frame_err, Lum50=Lum50,
                         Lum50_err=Lum50_err, flux_density=flux_density, flux_density_err=flux_density_err,
                         magnitude=magnitude, magnitude_err=magnitude_err, data_mode=data_mode, name=name,
                         use_phase_model=use_phase_model, bands=bands, system=system, active_bands=active_bands,
                         redshift=redshift, photon_index=photon_index, **kwargs)
        self.directory_structure = redback.get_data.directory.open_access_directory_structure(
            transient=name, transient_type="supernova")
        self._set_data()
