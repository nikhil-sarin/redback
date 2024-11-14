import numpy as np
from typing import Union

import redback.get_data.directory
from redback.transient.transient import OpticalTransient


class TDE(OpticalTransient):
    DATA_MODES = ['flux', 'flux_density', 'magnitude', 'luminosity']

    def __init__(
            self, name: str, data_mode: str = 'magnitude', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux_density: np.ndarray = None, flux_density_err: np.ndarray = None, magnitude: np.ndarray = None,
            magnitude_err: np.ndarray = None, redshift: float = np.nan, photon_index: float = np.nan,
            bands: np.ndarray = None, system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all',
            plotting_order: Union[np.ndarray, str] = None, use_phase_model: bool = False,
            optical_data:bool = True, **kwargs: None) -> None:
        """

        :param name: Name of the transient.
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
        :param kwargs: Additional callables:
                        bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use
                        redback.utils.bands_to_frequency if not given.
        :type kwargs: None
        """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, time_rest_frame_err=time_rest_frame_err, Lum50=Lum50,
                         Lum50_err=Lum50_err, flux_density=flux_density, flux_density_err=flux_density_err,
                         magnitude=magnitude, magnitude_err=magnitude_err, data_mode=data_mode, name=name,
                         use_phase_model=use_phase_model, optical_data=optical_data, bands=bands,
                         system=system, active_bands=active_bands,redshift=redshift,
                         photon_index=photon_index, plotting_order=plotting_order, **kwargs)
        self.directory_structure = redback.get_data.directory.open_access_directory_structure(
            transient=self.name, transient_type="tidal_disruption_event")
        self._set_data()
