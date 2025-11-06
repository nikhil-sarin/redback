from __future__ import annotations

from typing import Union

import matplotlib
import numpy as np
import pandas as pd

import redback
from redback.plotting import \
    LuminosityPlotter, FluxDensityPlotter, IntegratedFluxPlotter, MagnitudePlotter, \
    IntegratedFluxOpticalPlotter, SpectrumPlotter, LuminosityOpticalPlotter
from redback.model_library import all_models_dict
from collections import namedtuple

class Spectrum(object):
    def __init__(self, angstroms: np.ndarray, flux_density: np.ndarray, flux_density_err: np.ndarray,
                 time: str = None, name: str = '', **kwargs) -> None:
        """
        A class to store spectral data.

    Parameters
    ----------
    angstroms
        Wavelength in angstroms.
    flux_density
        flux density in ergs/s/cm^2/angstrom.
    flux_density_err
        flux density error in ergs/s/cm^2/angstrom.
    time
        Time of the spectrum. Could be a phase or time since burst. Only used for plotting.
    name
        Name of the spectrum.
    """

        self.angstroms = angstroms
        self.flux_density = flux_density
        self.flux_density_err = flux_density_err
        self.time = time
        self.name = name
        if self.time is None:
            self.plot_with_time_label = False
        else:
            self.plot_with_time_label = True
        self.directory_structure = redback.get_data.directory.spectrum_directory_structure(transient=name)
        self.data_mode = 'spectrum'

    @property
    def xlabel(self) -> str:
        """

    Returns
    -------
    str
        xlabel used in plotting functions
    """
        return r'Wavelength [$\mathrm{\AA}$]'

    @property
    def ylabel(self) -> str:
        """

    Returns
    -------
    str
        ylabel used in plotting functions
    """
        return r'Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}$)'

    def plot_data(self, axes: matplotlib.axes.Axes = None, filename: str = None, outdir: str = None, save: bool = True,
            show: bool = True, color: str = 'k', **kwargs) -> matplotlib.axes.Axes:
        """Plots the Transient data and returns Axes.

    Parameters
    ----------
    axes
        Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
    filename
        Name of the file to be plotted in.
    outdir
        The directory in which to save the file in.
    save
        Whether to save the plot. (Default value = True)
    show
        Whether to show the plot. (Default value = True)
    color
        Color of the data.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_data.__doc__)` to see all options!

    Returns
    -------
        The axes with the plot.
    """

        plotter = SpectrumPlotter(spectrum=self, color=color, filename=filename, outdir=outdir, **kwargs)
        return plotter.plot_data(axes=axes, save=save, show=show)

    def plot_spectrum(
            self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: None) -> matplotlib.axes.Axes:
        """

    Parameters
    ----------
    model
        The model used to plot the lightcurve.
    filename
        The output filename. Otherwise, use default which starts with the name attribute and ends with *lightcurve.png.
    axes
        Axes to plot in if given.
    save
        Whether to save the plot.
    show
        Whether to show the plot.
    random_models
        Number of random posterior samples plotted faintly. (Default value = 100)
    posterior
        Posterior distribution to which to draw samples from. Is optional but must be given.
    outdir
        Out directory in which to save the plot. Default is the current working directory.
    model_kwargs
        Additional keyword arguments to be passed into the model.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_lightcurve.__doc__)` to see all options!

    Returns
    -------
        The axes.
    """
        plotter = SpectrumPlotter(
            spectrum=self, model=model, filename=filename, outdir=outdir,
            posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        return plotter.plot_spectrum(axes=axes, save=save, show=show)

    def plot_residual(self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
                      save: bool = True, show: bool = True, posterior: pd.DataFrame = None,
                      model_kwargs: dict = None, **kwargs: None) -> matplotlib.axes.Axes:
        """

    Parameters
    ----------
    model
        The model used to plot the lightcurve.
    filename
        The output filename. Otherwise, use default which starts with the name attribute and ends with *lightcurve.png.
    axes
        Axes to plot in if given.
    save
        Whether to save the plot.
    show
        Whether to show the plot.
    posterior
        Posterior distribution to which to draw samples from. Is optional but must be given.
    outdir
        Out directory in which to save the plot. Default is the current working directory.
    model_kwargs
        Additional keyword arguments to be passed into the model.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_residual.__doc__)` to see all options!

    Returns
    -------
        The axes.
    """
        plotter = SpectrumPlotter(
            spectrum=self, model=model, filename=filename, outdir=outdir,
            posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        return plotter.plot_residuals(axes=axes, save=save, show=show)
    LuminosityPlotter, FluxDensityPlotter, IntegratedFluxPlotter, MagnitudePlotter, IntegratedFluxOpticalPlotter

class Transient(object):
    DATA_MODES = ['luminosity', 'flux', 'flux_density', 'magnitude', 'counts', 'ttes']
    _ATTRIBUTE_NAME_DICT = dict(luminosity="Lum50", flux="flux", flux_density="flux_density",
                                counts="counts", magnitude="magnitude")

    ylabel_dict = dict(luminosity=r'Luminosity [$10^{50}$ erg s$^{-1}$]',
                       magnitude=r'Magnitude',
                       flux=r'Flux [erg cm$^{-2}$ s$^{-1}$]',
                       flux_density=r'Flux density [mJy]',
                       counts=r'Counts')

    luminosity_data = redback.utils.DataModeSwitch('luminosity')
    flux_data = redback.utils.DataModeSwitch('flux')
    flux_density_data = redback.utils.DataModeSwitch('flux_density')
    magnitude_data = redback.utils.DataModeSwitch('magnitude')
    counts_data = redback.utils.DataModeSwitch('counts')
    tte_data = redback.utils.DataModeSwitch('ttes')

    def __init__(
            self, time: np.ndarray = None, time_err: np.ndarray = None, time_mjd: np.ndarray = None,
            time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None, time_rest_frame_err: np.ndarray = None,
            Lum50: np.ndarray = None, Lum50_err: np.ndarray = None, flux: np.ndarray = None,
            flux_err: np.ndarray = None, flux_density: np.ndarray = None, flux_density_err: np.ndarray = None,
            magnitude: np.ndarray = None, magnitude_err: np.ndarray = None, counts: np.ndarray = None,
            ttes: np.ndarray = None, bin_size: float = None, redshift: float = np.nan, data_mode: str = None,
            name: str = '', photon_index: float = np.nan, use_phase_model: bool = False,
            optical_data: bool = False, frequency: np.ndarray = None, system: np.ndarray = None, bands: np.ndarray = None,
            active_bands: Union[np.ndarray, str] = None, plotting_order: Union[np.ndarray, str] = None, **kwargs: None) -> None:
        """This is a general constructor for the Transient class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

    Parameters
    ----------
    time : np.ndarray, optional
        Times in the observer frame.
    time_err : np.ndarray, optional
        Time errors in the observer frame.
    time_mjd : np.ndarray, optional
        Times in MJD. Used if using phase model.
    time_mjd_err : np.ndarray, optional
        Time errors in MJD. Used if using phase model.
    time_rest_frame : np.ndarray, optional
        Times in the rest frame. Used for luminosity data.
    time_rest_frame_err : np.ndarray, optional
        Time errors in the rest frame. Used for luminosity data.
    Lum50 : np.ndarray, optional
        Luminosity values.
    Lum50_err : np.ndarray, optional
        Luminosity error values.
    flux : np.ndarray, optional
        Flux values.
    flux_err : np.ndarray, optional
        Flux error values.
    flux_density : np.ndarray, optional
        Flux density values.
    flux_density_err : np.ndarray, optional
        Flux density error values.
    magnitude : np.ndarray, optional
        Magnitude values for photometry data.
    magnitude_err : np.ndarray, optional
        Magnitude error values for photometry data.
    counts : np.ndarray, optional
        Counts for prompt data.
    ttes : np.ndarray, optional
        Time-tagged events data for unbinned prompt data.
    bin_size : float, optional
        Bin size for binning time-tagged event data.
    redshift : float, optional
        Redshift value.
    data_mode : str, optional
        Data mode. Must be one from `Transient.DATA_MODES`.
    name : str, optional
        Name of the transient.
    photon_index : float, optional
        Photon index value.
    use_phase_model : bool, optional
        Whether we are using a phase model.
    optical_data : bool, optional
        Whether we are fitting optical data, useful for plotting.
    frequency : np.ndarray, optional
        Array of band frequencies in photometry data.
    system : np.ndarray, optional
        System values.
    bands : np.ndarray, optional
        Band values.
    active_bands : Union[list, np.ndarray], optional
        List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
    plotting_order : Union[np.ndarray, str], optional
        Order in which to plot the bands/and how unique bands are stored.
    kwargs : None, optional
        Additional callables: bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use redback.utils.bands_to_frequency if not given. bin_ttes: Binning function for time-tagged event data. Use redback.utils.bands_to_frequency if not given.
    """
        self.bin_size = bin_size
        self.bin_ttes = kwargs.get("bin_ttes", redback.utils.bin_ttes)
        self.bands_to_frequency = kwargs.get("bands_to_frequency", redback.utils.bands_to_frequency)

        if data_mode == 'ttes':
            time, counts = self.bin_ttes(ttes, self.bin_size)

        self.time = time
        self.time_err = time_err
        self.time_mjd = time_mjd
        self.time_mjd_err = time_mjd_err
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

        self._frequency = None
        self._bands = None
        self.set_bands_and_frequency(bands=bands, frequency=frequency)
        self.system = system
        self.data_mode = data_mode
        self.active_bands = active_bands
        self.sncosmo_bands = redback.utils.sncosmo_bandname_from_band(self.bands)
        self.redshift = redshift
        self.name = name
        self.use_phase_model = use_phase_model
        self.optical_data = optical_data
        self.plotting_order = plotting_order

        self.meta_data = None
        self.photon_index = photon_index
        self.directory_structure = redback.get_data.directory.DirectoryStructure(
            directory_path=".", raw_file_path=".", processed_file_path=".")

    @staticmethod
    def load_data_generic(processed_file_path, data_mode="magnitude"):
        """Loads data from specified directory and file, and returns it as a tuple.

    Parameters
    ----------
    processed_file_path : str
        Path to the processed file to load
    data_mode : str, optional
        Name of the data mode. Must be from ['magnitude', 'flux_density', 'all']. Default is magnitude.

    Returns
    -------
    tuple
        Six elements when querying magnitude or flux_density data, Eight for 'all'.
    """
        DATA_MODES = ['luminosity', 'flux', 'flux_density', 'magnitude', 'counts', 'ttes', 'all']
        df = pd.read_csv(processed_file_path)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        if data_mode not in DATA_MODES:
            raise ValueError(f"Data mode {data_mode} not in {DATA_MODES}")
        if data_mode == "magnitude":
            return time_days, time_mjd, magnitude, magnitude_err, bands
        elif data_mode == "flux_density":
            return time_days, time_mjd, flux_density, flux_density_err, bands
        elif data_mode == "all":
            return time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, bands

    @classmethod
    def from_lasair_data(
            cls, name: str, data_mode: str = "magnitude", active_bands: Union[np.ndarray, str] = 'all',
            use_phase_model: bool = False, plotting_order: Union[np.ndarray, str] = None) -> Transient:
        """Constructor method to built object from LASAIR data.

    Parameters
    ----------
    name : str
        Name of the transient.
    data_mode : str, optional
        Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
    active_bands : Union[np.ndarray, str]
        Sets active bands based on array given. If argument is 'all', all unique bands in `self.bands` will be used.
    plotting_order : Union[np.ndarray, str], optional
        Order in which to plot the bands/and how unique bands are stored.
    use_phase_model : bool, optional
        Whether to use a phase model.

    Returns
    -------
    OpticalTransient
        A class instance.
    """
        if cls.__name__ == "TDE":
            transient_type = "tidal_disruption_event"
        else:
            transient_type = cls.__name__.lower()
        directory_structure = redback.get_data.directory.lasair_directory_structure(
            transient=name, transient_type=transient_type)
        df = pd.read_csv(directory_structure.processed_file_path)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        flux = np.array(df["flux(erg/cm2/s)"])
        flux_err = np.array(df["flux_error"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, flux=flux, flux_err=flux_err, bands=bands, active_bands=active_bands,
                   use_phase_model=use_phase_model, optical_data=True, plotting_order=plotting_order)

    @classmethod
    def from_simulated_optical_data(
            cls, name: str, data_mode: str = "magnitude", active_bands: Union[np.ndarray, str] = 'all',
            plotting_order: Union[np.ndarray, str] = None, use_phase_model: bool = False) -> Transient:
        """Constructor method to built object from SimulatedOpticalTransient.

    Parameters
    ----------
    name : str
        Name of the transient.
    data_mode : str, optional
        Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
    active_bands : Union[np.ndarray, str]
        Sets active bands based on array given. If argument is 'all', all unique bands in `self.bands` will be used.
    plotting_order : Union[np.ndarray, str], optional
        Order in which to plot the bands/and how unique bands are stored.
    use_phase_model : bool, optional
        Whether to use a phase model.

    Returns
    -------
    OpticalTransient
        A class instance.
    """
        path = "simulated/" + name + ".csv"
        df = pd.read_csv(path)
        df = df[df.detected != 0]
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        flux = np.array(df["flux(erg/cm2/s)"])
        flux_err = np.array(df["flux_error"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, flux=flux, flux_err=flux_err, bands=bands, active_bands=active_bands,
                   use_phase_model=use_phase_model, optical_data=True, plotting_order=plotting_order)


    @classmethod
    def from_lightcurvelynx(
            cls, name: str, data: pd.DataFrame = None, data_mode: str = "magnitude",
            active_bands: Union[np.ndarray, str] = 'all', plotting_order: Union[np.ndarray, str] = None,
            use_phase_model: bool = False) -> Transient:
        """Constructor method to built object from a LightCurveLynx simulated light curve.
        https://github.com/lincc-frameworks/lightcurvelynx
        Only the time, bands, magnitude and magnitude error columns are used. The rest are computed
        from those.

    Parameters
    ----------
    name : str
        Name of the transient.
    data : pd.DataFrame, optional
        DataFrame containing the light curve data. If None, it will try to load from "simulated/{name}.csv".
    data_mode : str, optional
        Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
    active_bands : Union[np.ndarray, str]
        Sets active bands based on array given. If argument is 'all', all unique bands in `self.bands` will be used.
    plotting_order : Union[np.ndarray, str], optional
        Order in which to plot the bands/and how unique bands are stored.
    use_phase_model : bool, optional
        Whether to use a phase model.

    Returns
    -------
    OpticalTransient
        A class instance.
    """
        if data is None:
            path = "simulated/" + name + ".csv"
            data = pd.read_csv(path)

        # Filter out the non-detections.
        if "detected" in data.columns:
            data = data[data.detected != 0]

        # Process the time and bands data.
        bands = data["filter"].to_numpy()        
        time_mjd = data["mjd"].to_numpy()
        if "time_rel" in data.columns:
            time_days = data["time_rel"].to_numpy()
        elif not use_phase_model:
            raise ValueError("Relative time column 'time_rel' is required if not using phase model.")
        else:
            time_days = None

        # Process the magnitude data. Checking that we have the values if the data mode is magnitude.
        if "mag" not in data.columns:
            raise ValueError("Magnitude values are required for LightCurveLynx input.")
        magnitude = data["mag"].to_numpy()
        magnitude_err = data["magerr"].to_numpy()

        # Compute the other values from the given magnitude and band data. We use the formulation
        # from SimulateOpticalTransient._make_observation_single().
        ref_flux = redback.utils.bands_to_reference_flux(bands)
        flux = redback.utils.bandpass_magnitude_to_flux(magnitude=magnitude, bands=bands)
        flux_err = redback.utils.calc_flux_error_from_magnitude(magnitude, magnitude_err, ref_flux)
        flux_density = redback.utils.calc_flux_density_from_ABmag(magnitude).value
        flux_density_err = redback.utils.calc_flux_density_error_from_monochromatic_magnitude(
            magnitude=magnitude,
            magnitude_error=magnitude_err,
            reference_flux=3631,
            magnitude_system='AB',
        )

        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, flux=flux, flux_err=flux_err, bands=bands, active_bands=active_bands,
                   use_phase_model=use_phase_model, optical_data=True, plotting_order=plotting_order)


    @property
    def _time_attribute_name(self) -> str:
        if self.luminosity_data:
            return "time_rest_frame"
        elif self.use_phase_model:
            return "time_mjd"
        return "time"

    @property
    def _time_err_attribute_name(self) -> str:
        return self._time_attribute_name + "_err"

    @property
    def _y_attribute_name(self) -> str:
        return self._ATTRIBUTE_NAME_DICT[self.data_mode]

    @property
    def _y_err_attribute_name(self) -> str:
        return self._ATTRIBUTE_NAME_DICT[self.data_mode] + "_err"

    @property
    def x(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        The time values given the active data mode.
    """
        return getattr(self, self._time_attribute_name)

    @x.setter
    def x(self, x: np.ndarray) -> None:
        """Sets the time values for the active data mode.

    Parameters
    ----------
    x : np.ndarray
        The desired time values.
    """
        setattr(self, self._time_attribute_name, x)

    @property
    def x_err(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        The time error values given the active data mode.
    """
        return getattr(self, self._time_err_attribute_name)

    @x_err.setter
    def x_err(self, x_err: np.ndarray) -> None:
        """Sets the time error values for the active data mode.

    Parameters
    ----------
    x_err : np.ndarray
        The desired time error values.
    """
        setattr(self, self._time_err_attribute_name, x_err)

    @property
    def y(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        The y values given the active data mode.
    """

        return getattr(self, self._y_attribute_name)

    @y.setter
    def y(self, y: np.ndarray) -> None:
        """Sets the y values for the active data mode.

    Parameters
    ----------
    y : np.ndarray
        The desired y values.
    """
        setattr(self, self._y_attribute_name, y)

    @property
    def y_err(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        The y error values given the active data mode.
    """
        return getattr(self, self._y_err_attribute_name)

    @y_err.setter
    def y_err(self, y_err: np.ndarray) -> None:
        """Sets the y error values for the active data mode.

    Parameters
    ----------
    y_err : np.ndarray
        The desired y error values.
    """
        setattr(self, self._y_err_attribute_name, y_err)

    @property
    def data_mode(self) -> str:
        """

    Returns
    -------
    str
        The currently active data mode (one in `Transient.DATA_MODES`).
    """
        return self._data_mode

    @data_mode.setter
    def data_mode(self, data_mode: str) -> None:
        """

    Parameters
    ----------
    data_mode : str
        One of the data modes in `Transient.DATA_MODES`.
    """
        if data_mode in self.DATA_MODES or data_mode is None:
            self._data_mode = data_mode
        else:
            raise ValueError("Unknown data mode.")

    @property
    def xlabel(self) -> str:
        """

    Returns
    -------
    str
        xlabel used in plotting functions
    """
        if self.use_phase_model:
            return r"Time [MJD]"
        else:
            return r"Time since explosion [days]"

    @property
    def ylabel(self) -> str:
        """

    Returns
    -------
    str
        ylabel used in plotting functions
    """
        try:
            return self.ylabel_dict[self.data_mode]
        except KeyError:
            raise ValueError("No data mode specified")

    def set_bands_and_frequency(
            self, bands: Union[None, list, np.ndarray], frequency: Union[None, list, np.ndarray]):
        """Sets bands and frequencies at the same time to keep the logic consistent. If both are given use those values.
        If only frequencies are given, use them also as band names.
        If only bands are given, try to convert them to frequencies.

    Parameters
    ----------
    bands : Union[None, list, np.ndarray]
        The bands, e.g. ['g', 'i'].
    frequency : Union[None, list, np.ndarray]
        The frequencies associated with the bands i.e., the effective frequency.
    """
        if (bands is None and frequency is None) or (bands is not None and frequency is not None):
            self._bands = bands
            self._frequency = frequency
        elif bands is None and frequency is not None:
            self._frequency = frequency
            self._bands = self.frequency
        elif bands is not None and frequency is None:
            self._bands = bands
            self._frequency = self.bands_to_frequency(self.bands)

    @property
    def frequency(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        Used band frequencies
    """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: np.ndarray) -> None:
        """

    Parameters
    ----------
    frequency : np.ndarray
        Set band frequencies if an array is given. Otherwise, convert bands to frequencies.
    """
        self.set_bands_and_frequency(bands=self.bands, frequency=frequency)

    @property
    def bands(self) -> Union[list, None, np.ndarray]:
        return self._bands

    @bands.setter
    def bands(self, bands: Union[list, None, np.ndarray]):
        self.set_bands_and_frequency(bands=bands, frequency=self.frequency)

    @property
    def filtered_frequencies(self) -> np.array:
        """

    Returns
    -------
    np.ndarray
        The frequencies only associated with the active bands.
    """
        return self.frequency[self.filtered_indices]

    @property
    def filtered_sncosmo_bands(self) -> np.array:
        """

    Returns
    -------
    np.ndarray
        The sncosmo bands only associated with the active bands.
    """
        return self.sncosmo_bands[self.filtered_indices]

    @property
    def filtered_bands(self) -> np.array:
        """

    Returns
    -------
    np.ndarray
        The band names only associated with the active bands.
    """
        return self.bands[self.filtered_indices]

    @property
    def active_bands(self) -> list:
        """

    Returns
    -------
        List of active bands used.
    """
        return self._active_bands

    @active_bands.setter
    def active_bands(self, active_bands: Union[list, str, None]) -> None:
        """

    Parameters
    ----------
    active_bands : Union[list, str]
        Sets active bands based on list given. If argument is 'all', all unique bands in `self.bands` will be used.
    """
        if str(active_bands) == 'all':
            self._active_bands = list(np.unique(self.bands))
        else:
            self._active_bands = active_bands

    @property
    def filtered_indices(self) -> Union[list, None]:
        """

    Returns
    -------
    Union[list, None]
        The list indices in `bands` associated with the active bands.
    """
        if self.bands is None:
            return list(np.arange(len(self.x)))
        return [b in self.active_bands for b in self.bands]

    def get_filtered_data(self) -> tuple:
        """Used to filter flux density, photometry or integrated flux data, so we only use data that is using the active bands.

    Returns
    -------
    tuple
        A tuple with the filtered data. Format is (x, x_err, y, y_err)
    """
        if any([self.flux_data, self.magnitude_data, self.flux_density_data]):
            filtered_x = self.x[self.filtered_indices]
            try:
                filtered_x_err = self.x_err[self.filtered_indices]
            except (IndexError, TypeError):
                filtered_x_err = None
            filtered_y = self.y[self.filtered_indices]
            filtered_y_err = self.y_err[self.filtered_indices]
            return filtered_x, filtered_x_err, filtered_y, filtered_y_err
        else:
            raise ValueError(f"Transient needs to be in flux density, magnitude or flux data mode, "
                             f"but is in {self.data_mode} instead.")

    @property
    def unique_bands(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        All bands that we get from the data, eliminating all duplicates.
    """
        if self.plotting_order is not None:
            return self.plotting_order
        else:
            return np.unique(self.bands)

    @property
    def unique_frequencies(self) -> np.ndarray:
        """

    Returns
    -------
    np.ndarray
        All frequencies that we get from the data, eliminating all duplicates.
    """
        try:
            if isinstance(self.unique_bands[0], (float, int)):
                return self.unique_bands
        except (TypeError, IndexError):
            pass
        return self.bands_to_frequency(self.unique_bands)

    @property
    def list_of_band_indices(self) -> list:
        """

    Returns
    -------
    list
        Indices that map between bands in the data and the unique bands we obtain.
    """
        return [np.where(self.bands == np.array(b))[0] for b in self.unique_bands]

    @property
    def default_filters(self) -> list:
        """

    Returns
    -------
    list
        Default list of filters to use.
    """
        return ["g", "r", "i", "z", "y", "J", "H", "K"]

    @staticmethod
    def get_colors(filters: Union[np.ndarray, list]) -> matplotlib.colors.Colormap:
        """

    Parameters
    ----------
    filters : Union[np.ndarray, list]
        Array of list of filters to use in the plot.

    Returns
    -------
    matplotlib.colors.Colormap
        Colormap with one color for each filter.
    """
        return matplotlib.cm.rainbow(np.linspace(0, 1, len(filters)))

    def plot_data(self, axes: matplotlib.axes.Axes = None, filename: str = None, outdir: str = None, save: bool = True,
            show: bool = True, plot_others: bool = True, color: str = 'k', **kwargs) -> matplotlib.axes.Axes:
        """Plots the Transient data and returns Axes.

    Parameters
    ----------
    axes
        Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
    filename
        Name of the file to be plotted in.
    outdir
        The directory in which to save the file in.
    save
        Whether to save the plot. (Default value = True)
    show
        Whether to show the plot. (Default value = True)
    plot_others
        Whether to plot inactive bands. (Default value = True)
    color
        Color of the data.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_data.__doc__)` to see all options!

    Returns
    -------
        The axes with the plot.
    """

        if self.flux_data:
            if self.optical_data:
                plotter = IntegratedFluxOpticalPlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                       plot_others=plot_others, **kwargs)
            else:
                plotter = IntegratedFluxPlotter(transient=self, color=color, filename=filename, outdir=outdir, **kwargs)
        elif self.luminosity_data:
            if self.optical_data:
                plotter = LuminosityOpticalPlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                                   **kwargs)
            else:
                plotter = LuminosityPlotter(transient=self, color=color, filename=filename, outdir=outdir, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                         plot_others=plot_others, **kwargs)
        elif self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, color=color, filename=filename, outdir=outdir,
                                       plot_others=plot_others, **kwargs)
        else:
            return axes
        return plotter.plot_data(axes=axes, save=save, show=show)

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, filename: str = None,
            outdir: str = None, ncols: int = 2, save: bool = True, show: bool = True,
            nrows: int = None, figsize: tuple = None, filters: list = None, **kwargs: None) \
            -> matplotlib.axes.Axes:
        """

    Parameters
    ----------
    figure
        Figure can be given if defaults are not satisfying.
    axes
        Axes can be given if defaults are not satisfying.
    filename
        Name of the file to be plotted in.
    outdir
        The directory in which to save the file in.
    save
        Whether to save the plot. (Default value = True)
    show
        Whether to show the plot. (Default value = True)
    ncols
        Number of columns to use on the plot. Default is 2.
    nrows
        Number of rows to use on the plot. If None are given this will be inferred from ncols and the number of filters.
    figsize
        Size of the figure. A default based on ncols and nrows will be used if None is given.
    filters
        Which bands to plot. Will use default filters if None is given.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_multiband.__doc__)` to see all options!

    Returns
    -------
        The axes.
    """
        if self.data_mode not in ['flux_density', 'magnitude', 'flux']:
            raise ValueError(
                f'You cannot plot multiband data with {self.data_mode} data mode . Why are you doing this?')
        if self.magnitude_data:
            plotter = MagnitudePlotter(transient=self, filters=filters, filename=filename, outdir=outdir, nrows=nrows,
                                       ncols=ncols, figsize=figsize, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(transient=self, filters=filters, filename=filename, outdir=outdir, nrows=nrows,
                                         ncols=ncols, figsize=figsize, **kwargs)
        elif self.flux_data:
            plotter = IntegratedFluxOpticalPlotter(transient=self, filters=filters, filename=filename, outdir=outdir,
                                                   nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        else:
            return
        return plotter.plot_multiband(figure=figure, axes=axes, save=save, show=show)

    def plot_lightcurve(
            self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: None) -> matplotlib.axes.Axes:
        """

    Parameters
    ----------
    model
        The model used to plot the lightcurve.
    filename
        The output filename. Otherwise, use default which starts with the name attribute and ends with *lightcurve.png.
    axes
        Axes to plot in if given.
    save
        Whether to save the plot.
    show
        Whether to show the plot.
    random_models
        Number of random posterior samples plotted faintly. (Default value = 100)
    posterior
        Posterior distribution to which to draw samples from. Is optional but must be given.
    outdir
        Out directory in which to save the plot. Default is the current working directory.
    model_kwargs
        Additional keyword arguments to be passed into the model.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_lightcurve.__doc__)` to see all options!

    Returns
    -------
        The axes.
    """
        if self.flux_data:
            if self.optical_data:
                plotter = IntegratedFluxOpticalPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
            else:
                plotter = IntegratedFluxPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.luminosity_data:
            if self.optical_data:
                plotter = LuminosityOpticalPlotter(transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
            else:
                plotter = LuminosityPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.magnitude_data:
            plotter = MagnitudePlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        else:
            return axes
        return plotter.plot_lightcurve(axes=axes, save=save, show=show)

    def plot_residual(self, model: callable, filename: str = None, outdir: str = None, axes: matplotlib.axes.Axes = None,
                      save: bool = True, show: bool = True, posterior: pd.DataFrame = None,
                      model_kwargs: dict = None, **kwargs: None) -> matplotlib.axes.Axes:
        """

    Parameters
    ----------
    model
        The model used to plot the lightcurve.
    filename
        The output filename. Otherwise, use default which starts with the name attribute and ends with *lightcurve.png.
    axes
        Axes to plot in if given.
    save
        Whether to save the plot.
    show
        Whether to show the plot.
    posterior
        Posterior distribution to which to draw samples from. Is optional but must be given.
    outdir
        Out directory in which to save the plot. Default is the current working directory.
    model_kwargs
        Additional keyword arguments to be passed into the model.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_residual.__doc__)` to see all options!

    Returns
    -------
        The axes.
    """
        if self.flux_data:
            plotter = IntegratedFluxPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        elif self.luminosity_data:
            if self.optical_data:
                plotter = LuminosityOpticalPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, **kwargs)
            else:
                plotter = LuminosityPlotter(
                    transient=self, model=model, filename=filename, outdir=outdir,
                    posterior=posterior, model_kwargs=model_kwargs, **kwargs)
        else:
            raise ValueError("Residual plotting not implemented for this data mode")
        return plotter.plot_residuals(axes=axes, save=save, show=show)


    def fit_gp(self, mean_model, kernel, prior=None, use_frequency=True):
        """
        Fit a GP to the data using george and scipy minimization.

    Parameters
    ----------
    mean_model
        Mean model to use in the GP fit. Can be a string to refer to a redback model, a callable, or None
    kernel
        George GP to use. User must ensure this is set up correctly.
    prior
        Prior to use when fitting with a mean model.
    use_frequency
        Whether to use the effective frequency in a 2D GP fit. Cannot be used with most mean models.

    Returns
    -------
        Named tuple with George GP object and additional useful data.
    """
        try:
            import george
            import george.kernels as kernels
        except ImportError:
            redback.utils.logger.warning("George must be installed to use GP fitting.")
        import scipy.optimize as op
        from bilby.core.likelihood import function_to_george_mean_model

        output = namedtuple("gp_out", ["gp", "scaled_y", "y_scaler", 'use_frequency', 'mean_model'])
        output.use_frequency = use_frequency
        output.mean_model = mean_model

        if self.data_mode == 'luminosity':
            x = self.time_rest_frame
            y = self.y
            try:
                y_err = np.max(self.y_err, axis=0)
            except IndexError:
                y_err = self.y_err
        else:
            x, x_err, y, y_err = self.get_filtered_data()
        redback.utils.logger.info("Rescaling data for GP fitting.")
        gp_y_err = y_err / np.max(y)
        gp_y = y / np.max(y)
        output.scaled_y = gp_y
        output.y_scaler = np.max(y)

        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(gp_y, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        def grad_nll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(gp_y, quiet=True)

        if use_frequency:
            redback.utils.logger.info("Using frequencies and time in the GP fit.")
            redback.utils.logger.info("Kernel used: " + str(kernel))
            redback.utils.logger.info("Ensure that the kernel is set up correctly for 2D GP.")
            redback.utils.logger.info("You will be returned a single GP object with frequency as a parameter")
            freqs = self.filtered_frequencies
            X = np.column_stack((freqs, x))
        else:
            redback.utils.logger.info("Using time in GP fit.")
            redback.utils.logger.info("Kernel used: " + str(kernel))
            redback.utils.logger.info("Ensure that the kernel is set up correctly for 1D GP.")
            redback.utils.logger.info("You will be returned a GP object unique to a band/frequency"
                                      " in the data if working with multiband data")
            X = x

        if mean_model is None:
            redback.utils.logger.info("Mean model not given, fitting GP with no mean model.")
            gp = george.GP(kernel)
            gp.compute(X, gp_y_err + 1e-8)
            p0 = gp.get_parameter_vector()
            results = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(results.x)
            redback.utils.logger.info(f"GP final loglikelihood: {gp.log_likelihood(gp_y)}")
            redback.utils.logger.info(f"GP final parameters: {gp.get_parameter_dict()}")
            output.gp = gp
        else:
            if isinstance(mean_model, str):
                mean_model_func = all_models_dict[mean_model]
                redback.utils.logger.info("Using inbuilt redback function {} as a mean model.".format(mean_model))
                if prior is None:
                    redback.utils.logger.warning("No prior given for mean model. Using default prior.")
                    prior = redback.priors.get_priors(mean_model)
            else:
                mean_model_func = mean_model
                redback.utils.logger.info("Using user-defined python function as a mean model.")

            if prior is None:
                redback.utils.logger.warning("Prior must be specified for GP fit with a mean model")
                raise ValueError("No prior specified")

            if self.data_mode in ['flux_density', 'magnitude', 'flux']:
                redback.utils.logger.info("Setting up GP version of mean model.")
                gp_dict = {}
                scaled_y_dict = {}
                for ii in range(len(self.unique_bands)):
                    scaled_y_dict[self.unique_bands[ii]] = gp_y[self.list_of_band_indices[ii]]
                    redback.utils.logger.info("Fitting for band {}".format(self.unique_bands[ii]))
                    gp_x = X[self.list_of_band_indices[ii]]

                    def nll(p):
                        gp.set_parameter_vector(p)
                        ll = gp.log_likelihood(gp_y[self.list_of_band_indices[ii]], quiet=True)
                        return -ll if np.isfinite(ll) else 1e25

                    mean_model_class = function_to_george_mean_model(mean_model_func)
                    mm = mean_model_class(**prior.sample())
                    gp = george.GP(kernel, mean=mm, fit_mean=True)
                    gp.compute(gp_x, gp_y_err[self.list_of_band_indices[ii]] + 1e-8)
                    p0 = gp.get_parameter_vector()
                    results = op.minimize(nll, p0)
                    gp.set_parameter_vector(results.x)
                    redback.utils.logger.info(f"GP final loglikelihood: {gp.log_likelihood(gp_y[self.list_of_band_indices[ii]])}")
                    redback.utils.logger.info(f"GP final parameters: {gp.get_parameter_dict()}")
                    gp_dict[self.unique_bands[ii]] = gp
                    del gp
                output.gp = gp_dict
                output.scaled_y = scaled_y_dict
            else:
                mean_model_class = function_to_george_mean_model(mean_model_func)
                mm = mean_model_class(**prior.sample())
                gp = george.GP(kernel, mean=mm, fit_mean=True)
                gp.compute(X, gp_y_err + 1e-8)
                p0 = gp.get_parameter_vector()
                results = op.minimize(nll, p0)
                gp.set_parameter_vector(results.x)
                redback.utils.logger.info(f"GP final loglikelihood: {gp.log_likelihood(gp_y)}")
                redback.utils.logger.info(f"GP final parameters: {gp.get_parameter_dict()}")
                output.gp = gp
        return output

    def plot_multiband_lightcurve(
            self, model: callable, filename: str = None, outdir: str = None,
            figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None,
            save: bool = True, show: bool = True, random_models: int = 100, posterior: pd.DataFrame = None,
            model_kwargs: dict = None, **kwargs: object) -> matplotlib.axes.Axes:
        """

    Parameters
    ----------
    model
        The model used to plot the lightcurve.
    filename
        The output filename. Otherwise, use default which starts with the name attribute and ends with *lightcurve.png.
    figure
        Figure can be given if defaults are not satisfying.
    axes
        Axes to plot in if given.
    save
        Whether to save the plot.
    show
        Whether to show the plot.
    random_models
        Number of random posterior samples plotted faintly. (Default value = 100)
    posterior
        Posterior distribution to which to draw samples from. Is optional but must be given.
    outdir
        Out directory in which to save the plot. Default is the current working directory.
    model_kwargs
        Additional keyword arguments to be passed into the model.
    kwargs
        Additional keyword arguments to pass in the Plotter methods. Available in the online documentation under at `redback.plotting.Plotter`. `print(Transient.plot_multiband_lightcurve.__doc__)` to see all options!

    Returns
    -------
        The axes.
    """
        if self.data_mode not in ['flux_density', 'magnitude', 'flux']:
            raise ValueError(
                f'You cannot plot multiband data with {self.data_mode} data mode . Why are you doing this?')
        if self.magnitude_data:
            plotter = MagnitudePlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.flux_data:
            plotter = IntegratedFluxOpticalPlotter(transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        elif self.flux_density_data:
            plotter = FluxDensityPlotter(
                transient=self, model=model, filename=filename, outdir=outdir,
                posterior=posterior, model_kwargs=model_kwargs, random_models=random_models, **kwargs)
        else:
            return
        return plotter.plot_multiband_lightcurve(figure=figure, axes=axes, save=save, show=show)

    _formatted_kwargs_options = redback.plotting.Plotter.keyword_docstring
    plot_data.__doc__ = plot_data.__doc__.replace(
        "`print(Transient.plot_data.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_multiband.__doc__ = plot_multiband.__doc__.replace(
        "`print(Transient.plot_multiband.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_lightcurve.__doc__ = plot_lightcurve.__doc__.replace(
        "`print(Transient.plot_lightcurve.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_multiband_lightcurve.__doc__ = plot_multiband_lightcurve.__doc__.replace(
        "`print(Transient.plot_multiband_lightcurve.__doc__)` to see all options!", _formatted_kwargs_options)
    plot_residual.__doc__ = plot_residual.__doc__.replace(
        "`print(Transient.plot_residual.__doc__)` to see all options!", _formatted_kwargs_options)


class OpticalTransient(Transient):
    DATA_MODES = ['flux', 'flux_density', 'magnitude', 'luminosity']

    @staticmethod
    def load_data(processed_file_path, data_mode="magnitude"):
        """Loads data from specified directory and file, and returns it as a tuple.

    Parameters
    ----------
    processed_file_path : str
        Path to the processed file to load
    data_mode : str, optional
        Name of the data mode. Must be from ['magnitude', 'flux_density', 'all']. Default is magnitude.

    Returns
    -------
    tuple
        Six elements when querying magnitude or flux_density data, Eight for 'all'
    """
        df = pd.read_csv(processed_file_path)
        time_days = np.array(df["time (days)"])
        time_mjd = np.array(df["time"])
        magnitude = np.array(df["magnitude"])
        magnitude_err = np.array(df["e_magnitude"])
        bands = np.array(df["band"])
        system = np.array(df["system"])
        flux_density = np.array(df["flux_density(mjy)"])
        flux_density_err = np.array(df["flux_density_error"])
        flux = np.array(df["flux(erg/cm2/s)"])
        flux_err = np.array(df['flux_error'])
        if data_mode == "magnitude":
            return time_days, time_mjd, magnitude, magnitude_err, bands, system
        elif data_mode == "flux_density":
            return time_days, time_mjd, flux_density, flux_density_err, bands, system
        elif data_mode == "flux":
            return time_days, time_mjd, flux, flux_err, bands, system
        elif data_mode == "all":
            return time_days, time_mjd, flux_density, flux_density_err, \
                   magnitude, magnitude_err, flux, flux_err, bands, system

    def __init__(
            self, name: str, data_mode: str = 'magnitude', time: np.ndarray = None, time_err: np.ndarray = None,
            time_mjd: np.ndarray = None, time_mjd_err: np.ndarray = None, time_rest_frame: np.ndarray = None,
            time_rest_frame_err: np.ndarray = None, Lum50: np.ndarray = None, Lum50_err: np.ndarray = None,
            flux: np.ndarray = None, flux_err: np.ndarray = None, flux_density: np.ndarray = None,
            flux_density_err: np.ndarray = None, magnitude: np.ndarray = None, magnitude_err: np.ndarray = None,
            redshift: float = np.nan, photon_index: float = np.nan, frequency: np.ndarray = None,
            bands: np.ndarray = None, system: np.ndarray = None, active_bands: Union[np.ndarray, str] = 'all',
            plotting_order: Union[np.ndarray, str] = None, use_phase_model: bool = False,
            optical_data:bool = True, **kwargs: None) -> None:
        """This is a general constructor for the Transient class. Note that you only need to give data corresponding to
        the data mode you are using. For luminosity data provide times in the rest frame, if using a phase model
        provide time in MJD, else use the default time (observer frame).

    Parameters
    ----------
    name : str
        Name of the transient.
    data_mode : str, optional
        Data mode. Must be one from `OpticalTransient.DATA_MODES`.
    time : np.ndarray, optional
        Times in the observer frame.
    time_err : np.ndarray, optional
        Time errors in the observer frame.
    time_mjd : np.ndarray, optional
        Times in MJD. Used if using phase model.
    time_mjd_err : np.ndarray, optional
        Time errors in MJD. Used if using phase model.
    time_rest_frame : np.ndarray, optional
        Times in the rest frame. Used for luminosity data.
    time_rest_frame_err : np.ndarray, optional
        Time errors in the rest frame. Used for luminosity data.
    Lum50 : np.ndarray, optional
        Luminosity values.
    Lum50_err : np.ndarray, optional
        Luminosity error values.
    flux : np.ndarray, optional
        Flux values.
    flux_err : np.ndarray, optional
        Flux error values.
    flux_density : np.ndarray, optional
        Flux density values.
    flux_density_err : np.ndarray, optional
        Flux density error values.
    magnitude : np.ndarray, optional
        Magnitude values for photometry data.
    magnitude_err : np.ndarray, optional
        Magnitude error values for photometry data.
    redshift : float, optional
        Redshift value.
    photon_index : float, optional
        Photon index value.
    frequency : np.ndarray, optional
        Array of band frequencies in photometry data.
    bands : np.ndarray, optional
        Band values.
    system : np.ndarray, optional
        System values.
    active_bands : Union[list, np.ndarray], optional
        List or array of active bands to be used in the analysis. Use all available bands if 'all' is given.
    plotting_order : Union[np.ndarray, str], optional
        Order in which to plot the bands/and how unique bands are stored.
    use_phase_model : bool, optional
        Whether we are using a phase model.
    optical_data : bool, optional
        Whether we are fitting optical data, useful for plotting.
    kwargs : dict, optional
         Additional callables: bands_to_frequency: Conversion function to convert a list of bands to frequencies. Use redback.utils.bands_to_frequency if not given.
    """
        super().__init__(time=time, time_err=time_err, time_rest_frame=time_rest_frame, time_mjd=time_mjd,
                         time_mjd_err=time_mjd_err, frequency=frequency,
                         time_rest_frame_err=time_rest_frame_err, Lum50=Lum50, Lum50_err=Lum50_err,
                         flux=flux, flux_err=flux_err, redshift=redshift, photon_index=photon_index,
                         flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                         magnitude_err=magnitude_err, data_mode=data_mode, name=name,
                         use_phase_model=use_phase_model, optical_data=optical_data,
                         system=system, bands=bands, plotting_order=plotting_order,
                         active_bands=active_bands, **kwargs)
        self.directory_structure = redback.get_data.directory.DirectoryStructure(
            directory_path=".", raw_file_path=".", processed_file_path=".")

    @classmethod
    def from_open_access_catalogue(
            cls, name: str, data_mode: str = "magnitude", active_bands: Union[np.ndarray, str] = 'all',
            plotting_order: Union[np.ndarray, str] = None, use_phase_model: bool = False) -> OpticalTransient:
        """Constructor method to built object from Open Access Catalogue

    Parameters
    ----------
    name : str
        Name of the transient.
    data_mode : str, optional
        Data mode used. Must be from `OpticalTransient.DATA_MODES`. Default is magnitude.
    active_bands : Union[np.ndarray, str]
         Sets active bands based on array given. If argument is 'all', all unique bands in `self.bands` will be used.
    plotting_order : Union[np.ndarray, str], optional
        Order in which to plot the bands/and how unique bands are stored.
    use_phase_model : bool, optional
        Whether to use a phase model.

    Returns
    -------
    OpticalTransient
        A class instance
    """
        if cls.__name__ == "TDE":
            transient_type = "tidal_disruption_event"
        else:
            transient_type = cls.__name__.lower()
        directory_structure = redback.get_data.directory.open_access_directory_structure(
            transient=name, transient_type=transient_type)
        time_days, time_mjd, flux_density, flux_density_err, magnitude, magnitude_err, flux, flux_err, bands, system = \
            cls.load_data(processed_file_path=directory_structure.processed_file_path, data_mode="all")
        return cls(name=name, data_mode=data_mode, time=time_days, time_err=None, time_mjd=time_mjd,
                   flux_density=flux_density, flux_density_err=flux_density_err, magnitude=magnitude,
                   magnitude_err=magnitude_err, bands=bands, system=system, active_bands=active_bands,
                   use_phase_model=use_phase_model, optical_data=True, flux=flux, flux_err=flux_err,
                   plotting_order=plotting_order)

    @property
    def event_table(self) -> str:
        """

    Returns
    -------
    str
        Path to the metadata table.
    """
        return f"{self.directory_structure.directory_path}/{self.name}_metadata.csv"

    def _set_data(self) -> None:
        """Sets the metadata from the event table."""
        try:
            meta_data = pd.read_csv(self.event_table, on_bad_lines='skip', delimiter=',', dtype='str')
        except FileNotFoundError as e:
            redback.utils.logger.warning(e)
            redback.utils.logger.warning("Setting metadata to None. This is not an error, but a warning that no metadata could be found online.")
            meta_data = None
        self.meta_data = meta_data

    @property
    def transient_dir(self) -> str:
        """

    Returns
    -------
    str
        The transient directory given the name of the transient.
    """
        return self._get_transient_dir()

    def _get_transient_dir(self) -> str:
        """

    Returns
    -------
    str
        The transient directory path
    """
        transient_dir, _, _ = redback.get_data.directory.open_access_directory_structure(
            transient=self.name, transient_type=self.__class__.__name__.lower())
        return transient_dir

    def estimate_bb_params(self, distance: float = 1e27, bin_width: float = 1.0, min_filters: int = 3, **kwargs):
        """
        Estimate the blackbody temperature and photospheric radius as functions of time by fitting
        a blackbody SED to the multi‑band photometry.

        The method groups the photometric data into time bins (epochs) of width bin_width (in the
        same units as self.x, typically days). For each epoch with at least min_filters measurements
        (from distinct filters), it fits a blackbody model to the data. When working with photometry
        provided in an effective flux density format (data_mode == "flux_density") the effective–wavelength
        approximation is used. When the data_mode is "flux" (or "magnitude") users have the option
        (via use_eff_wavelength=True) to instead use the effective wavelength approximation by converting AB
        magnitudes to flux density (using redback.utils.calc_flux_density_from_ABmag). If this flag is not
        provided (or is False) then the full bandpass integration is applied.

        Parameters
        ----------
        distance : float, optional
            Distance to the transient in centimeters. Default is 1e27 cm.
        bin_width : float, optional
            Width of the time bins (in days) used to group the photometric data. Default is 1.0.
        min_filters : int, optional
            Minimum number of measurements (from distinct filters) required in a bin to perform the fit.
            Default is 3.
        kwargs : Additional keyword arguments
            maxfev : int, optional, default is 1000
            T_init : float, optional, default is 1e4, used as the initial guess for the fit.
            R_init : float, optional, default is 1e15, used as the initial guess for the fit.
            use_eff_wavelength : bool, optional, default is False.
                If True, then even for photometry provided as magnitudes (or bandpass fluxes),
                the effective wavelength approximation is used. In that case the AB magnitudes are
                converted to flux densities via redback.utils.calc_flux_density_from_ABmag.
                If False, full bandpass integration is used.

        Returns
        -------
        df_bb : pandas.DataFrame or None
            A DataFrame containing columns:
              - epoch_times : binned epoch times,
              - temperature : best-fit blackbody temperatures (Kelvin),
              - radius : best-fit photospheric radii (cm),
              - temp_err : 1σ uncertainties on the temperatures,
              - radius_err : 1σ uncertainties on the radii.
            Returns None if insufficient data are available.
        """
        from scipy.optimize import curve_fit
        import astropy.units as uu
        import numpy as np
        import pandas as pd

        # Get the filtered photometry.
        # Assumes self.get_filtered_data() returns (time, time_err, y, y_err)
        time_data, _, flux_data, flux_err_data = self.get_filtered_data()

        redback.utils.logger.info("Estimating blackbody parameters for {}.".format(self.name))
        redback.utils.logger.info("Using data mode = {}".format(self.data_mode))

        # Determine whether we are in bandpass mode.
        use_bandpass = False
        if hasattr(self, "data_mode") and self.data_mode in ['flux', 'magnitude']:
            use_bandpass = True
            # Assume self.filtered_sncosmo_bands contains the (string) band names.
            band_data = self.filtered_sncosmo_bands
        else:
            # Otherwise the flux data and frequencies are assumed to be given.
            redback.utils.logger.info("Using effective wavelength approximation for {}".format(self.data_mode))
            freq_data = self.filtered_frequencies

        # Option: force effective wavelength approximation even if data_mode is bandpass.
        force_eff = kwargs.get('use_eff_wavelength', False)
        if use_bandpass and force_eff:
            redback.utils.logger.warning("Using effective wavelength approximation for {}".format(self.data_mode))

            if self.data_mode == 'magnitude':
                # Convert the AB magnitudes to flux density using the redback function.
                from redback.utils import abmag_to_flux_density_and_error_inmjy
                flux_data, flux_err_data = abmag_to_flux_density_and_error_inmjy(flux_data, flux_err_data)
                freq_data = redback.utils.bands_to_frequency(band_data)
            else:
                # Convert the bandpass fluxes to flux density using the redback function.
                from redback.utils import bandpass_flux_to_flux_density, bands_to_effective_width
                redback.utils.logger.warning("Ensure filters.csv has the correct bandpass effective widths for your filter.")
                effective_widths = bands_to_effective_width(band_data)
                freq_data = redback.utils.bands_to_frequency(band_data)
                flux_data, flux_err_data = bandpass_flux_to_flux_density(flux_data, flux_err_data, effective_widths)
            # Use the effective frequency approach.
            use_bandpass = False

        # Get initial guesses.
        T_init = kwargs.get('T_init', 1e4)
        R_init = kwargs.get('R_init', 1e15)
        maxfev = kwargs.get('maxfev', 1000)

        # Sort photometric data by time.
        sort_idx = np.argsort(time_data)
        time_data = time_data[sort_idx]
        flux_data = flux_data[sort_idx]
        flux_err_data = flux_err_data[sort_idx]
        if use_bandpass:
            band_data = np.array(band_data)[sort_idx]
        else:
            freq_data = np.array(freq_data)[sort_idx]

        # Retrieve redshift.
        redshift = np.nan_to_num(self.redshift)
        if redshift <= 0.:
            raise ValueError("Redshift must be provided to perform K-correction.")

        # For effective frequency mode, K-correct frequencies.
        if not use_bandpass:
            freq_data, _ = redback.utils.calc_kcorrected_properties(frequency=freq_data,
                                                                    redshift=redshift, time=0.)

        # Define the model functions.
        if not use_bandpass:
            # --- Effective-wavelength model ---
            def bb_model(freq, logT, logR):
                T = 10 ** logT
                R = 10 ** logR
                # Compute the model flux density in erg/s/cm^2/Hz.
                # freq is in rest frame, so we need to apply cosmological dimming factor
                model_flux_cgs = redback.sed.blackbody_to_flux_density(T, R, distance, freq) / (1 + redshift)
                # Convert to mJy. (1 Jy = 1e-23 erg/s/cm^2/Hz; 1 mJy = 1e-3 Jy = 1e-26 erg/s/cm^2/Hz)
                model_flux_mjy = (model_flux_cgs / (1e-26 * uu.erg / uu.s / uu.cm**2 / uu.Hz)).value
                return model_flux_mjy

            model_func = bb_model
        else:
            # --- Full bandpass integration model ---
            # In this branch we do NOT want to pass strings to curve_fit.
            # Instead, we will dummy-encode the independent variable as indices.
            # We also capture the band names in a closure variable.
            def bb_model_bandpass_from_index(x, logT, logR):
                # Ensure x is a numpy array and convert indices to integers.
                i_idx = np.round(x).astype(int)
                # Retrieve all corresponding band names in one step.
                bands = np.array(epoch_bands)[i_idx]
                # Call bb_model_bandpass with the entire array of bands.
                return bb_model_bandpass(bands, logT, logR, redshift, distance, output_format=self.data_mode)

            def bb_model_bandpass(band, logT, logR, redshift, distance, output_format='magnitude'):
                from redback.utils import calc_kcorrected_properties, lambda_to_nu, bandpass_magnitude_to_flux
                # Create a wavelength grid (in Å) from 100 to 80,000 Å.
                lambda_obs = np.geomspace(100, 80000, 300)
                # Convert to frequency (Hz) and apply K-correction.
                frequency, _ = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_obs),
                                                          redshift=redshift, time=0.)
                T = 10 ** logT
                R = 10 ** logR
                # Compute the model SED (flux density in erg/s/cm^2/Hz).
                # Apply cosmological dimming factor for observer-frame flux
                model_flux = redback.sed.blackbody_to_flux_density(T, R, distance, frequency) / (1 + redshift)
                # Convert the SED to per-Å units
                _spectra = model_flux.to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                                         equivalencies=uu.spectral_density(wav=lambda_obs * uu.Angstrom))
                spectra = np.zeros((5, 300))
                spectra[:, :] = _spectra.value
                # Create a source object from the spectrum.
                source = redback.sed.RedbackTimeSeriesSource(phase=np.array([0, 1, 2, 3, 4]),
                                                             wave=lambda_obs, flux=spectra)
                if output_format == 'flux':
                    # Convert bandpass magnitude to flux.
                    mag = source.bandmag(phase=0, band=band, magsys='ab')
                    return bandpass_magnitude_to_flux(magnitude=mag, bands=band)
                elif output_format == 'magnitude':
                    mag = source.bandmag(phase=0, band=band, magsys='ab')
                    return mag
                else:
                    raise ValueError("Unknown output_format in bb_model_bandpass.")

            # Our wrapper for curve_fit uses dummy x-values.
            model_func = bb_model_bandpass_from_index

        # Initialize lists to store fit results.
        epoch_times = []
        temperatures = []
        radii = []
        temp_errs = []
        radius_errs = []

        t_min = np.min(time_data)
        t_max = np.max(time_data)
        bins = np.arange(t_min, t_max + bin_width, bin_width)
        redback.utils.logger.info("Number of bins: {}".format(len(bins)))

        # Ensure at least one bin has enough points.
        bins_with_enough = [i for i in range(len(bins) - 1)
                            if np.sum((time_data >= bins[i]) & (time_data < bins[i + 1])) >= min_filters]
        if len(bins_with_enough) == 0:
            redback.utils.logger.warning("No time bins have at least {} measurements. Fitting cannot proceed.".format(min_filters))
            redback.utils.logger.warning("Try generating more data through GPs, increasing bin widths, or using fewer filters.")
            return None

        # Loop over bins (epochs): for each with enough data perform the fit.
        for i in range(len(bins) - 1):
            mask = (time_data >= bins[i]) & (time_data < bins[i + 1])
            if np.sum(mask) < min_filters:
                continue
            t_epoch = np.mean(time_data[mask])
            try:
                if not use_bandpass:
                    # Use effective frequency array (numeric).
                    xdata = freq_data[mask]
                else:
                    # For full bandpass integration mode, we dummy encode xdata.
                    # We ignore the value and simply use indices [0, 1, 2, ...].
                    epoch_bands = list(band_data[mask])  # capture the list of bands for this epoch
                    xdata = np.arange(len(epoch_bands))
                popt, pcov = curve_fit(
                    model_func,
                    xdata,
                    flux_data[mask],
                    sigma=flux_err_data[mask],
                    p0=[np.log10(T_init), np.log10(R_init)],
                    absolute_sigma=True,
                    maxfev=maxfev
                )
            except Exception as e:
                redback.utils.logger.warning(f"Fit failed for epoch {i}: {e}")
                redback.utils.logger.warning(f"Skipping epoch {i} with time {t_epoch:.2f} days.")
                continue

            logT_fit, logR_fit = popt
            T_fit = 10 ** logT_fit
            R_fit = 10 ** logR_fit
            perr = np.sqrt(np.diag(pcov))
            T_err = np.log(10) * T_fit * perr[0]
            R_err = np.log(10) * R_fit * perr[1]

            epoch_times.append(t_epoch)
            temperatures.append(T_fit)
            radii.append(R_fit)
            temp_errs.append(T_err)
            radius_errs.append(R_err)

        if len(epoch_times) == 0:
            redback.utils.logger.warning("No epochs with sufficient data yielded a successful fit.")
            return None

        df_bb = pd.DataFrame({
            'epoch_times': epoch_times,
            'temperature': temperatures,
            'radius': radii,
            'temp_err': temp_errs,
            'radius_err': radius_errs
        })

        redback.utils.logger.info('Masking epochs with likely wrong extractions')
        df_bb = df_bb[df_bb['temp_err'] / df_bb['temperature'] < 1]
        df_bb = df_bb[df_bb['radius_err'] / df_bb['radius'] < 1]
        return df_bb


    def estimate_bolometric_luminosity(self, distance: float = 1e27, bin_width: float = 1.0,
                                          min_filters: int = 3, **kwargs):
        """
        Estimate the bolometric luminosity as a function of time by fitting the blackbody SED
        to the multi‑band photometry and then integrating that spectrum. For each epoch the bolometric
        luminosity is computed using the Stefan–Boltzmann law evaluated at the source:

            L_bol = 4 π R² σ_SB T⁴

        Uncertainties in T and R are propagated assuming

            (ΔL_bol / L_bol)² = (2 ΔR / R)² + (4 ΔT / T)².

        Optionally, two corrections can be applied:

        1. A boost–factor to “restore” missing blue flux. If a cutoff wavelength is provided via
           the keyword 'lambda_cut' (in angstroms), it is converted to centimeters and a boost factor is
           calculated as:

               Boost = (F_tot / F_red)

           where F_tot = σ_SB T⁴ and F_red is computed by numerically integrating π * B_λ(T)
           from the cutoff wavelength (in cm) to infinity. The final (boosted) luminosity becomes:

               L_boosted = Boost × (4π R² σ_SB T⁴).

        2. An extinction correction. If the bolometric extinction (A_ext, in magnitudes) is supplied via
           the keyword 'A_ext', the luminosity will be reduced by a factor of 10^(–0.4·A_ext) to account
           for dust extinction. (A_ext defaults to 0.)

        Parameters
        ----------
        distance : float, optional
            Distance to the transient in centimeters. (Default is 1e27 cm.)
        bin_width : float, optional
            Width of the time bins (in days) used for grouping photometry. (Default is 1.0.)
        min_filters : int, optional
            Minimum number of independent filters required in a bin to perform a fit. (Default is 3.)
        kwargs : dict, optional
            Additional keyword arguments to pass to `estimate_bb_params` (e.g., maxfev, T_init, R_init,
            use_eff_wavelength, etc.). Additionally:
        - 'lambda_cut': If provided (in angstroms), the bolometric luminosity will be “boosted”
          to account for missing blue flux.
        - 'A_ext': Bolometric extinction in magnitudes. The observed luminosity is increased by a factor
          10^(+0.4·A_ext). (Default is 0.)

        Returns
        -------
        df_bol : pandas.DataFrame or None
            A DataFrame containing columns:
              - epoch_times: Mean time of the bin (days).
              - temperature: Fitted blackbody temperature (K).
              - radius: Fitted photospheric radius (cm).
              - lum_bol: Derived bolometric luminosity (1e50 erg/s) computed as 4π R² σ_SB T⁴
                         (boosted and extinction-corrected if requested).
              - lum_bol_bb: Derived bolometric blackbody luminosity (1e50 erg/s) computed as 4π R² σ_SB T⁴,
                            before applying either the boost or extinction correction.
              - lum_bol_err: 1σ uncertainty on L_bol (1e50 erg/s) from error propagation.
              - time_rest_frame: Epoch time divided by (1+redshift), i.e., the rest-frame time in days.
            Returns None if no valid blackbody fits were obtained.
        """
        from redback.sed import boosted_bolometric_luminosity

        # Retrieve optional lambda_cut (in angstroms) for the boost correction.
        lambda_cut_angstrom = kwargs.pop('lambda_cut', None)
        if lambda_cut_angstrom is not None:
            redback.utils.logger.info("Including effects of missing flux due to line blanketing.")
            redback.utils.logger.info(
                "Using lambda_cut = {} Å for bolometric luminosity boost.".format(lambda_cut_angstrom))
            # Convert lambda_cut from angstroms to centimeters (1 Å = 1e-8 cm)
            lambda_cut = lambda_cut_angstrom * 1e-8
        else:
            redback.utils.logger.info("No lambda_cut provided; no correction applied. Assuming a pure blackbody SED.")
            lambda_cut = None

        # Retrieve optional extinction in magnitudes.
        A_ext = kwargs.pop('A_ext', 0.0)
        if A_ext != 0.0:
            redback.utils.logger.info("Applying extinction correction with A_ext = {} mag.".format(A_ext))
        extinction_factor = 10 ** (0.4 * A_ext)

        # Retrieve blackbody parameters via your existing method.
        df_bb = self.estimate_bb_params(distance=distance, bin_width=bin_width, min_filters=min_filters, **kwargs)
        if df_bb is None or len(df_bb) == 0:
            redback.utils.logger.warning("No valid blackbody fits were obtained; cannot estimate bolometric luminosity.")
            return None

        # Compute L_bol (or L_boosted) for each epoch and propagate uncertainties.
        L_bol = []
        L_bol_err = []
        L_bol_bb = []
        L_bol_bb_err = []
        for index, row in df_bb.iterrows():
            temp = row['temperature']
            radius = row['radius']
            T_err = row['temp_err']
            R_err = row['radius_err']

            # Use boosted luminosity if lambda_cut is provided.
            if lambda_cut is not None:
                lum, lum_bb = boosted_bolometric_luminosity(temp, radius, lambda_cut)
            else:
                lum = 4 * np.pi * (radius ** 2) * redback.constants.sigma_sb * (temp ** 4)
                lum_bb = lum

            # Apply extinction correction to both luminosities.
            lum *= extinction_factor
            lum_bb *= extinction_factor

            # Propagate uncertainties using:
            # (ΔL/L)² = (2 ΔR / R)² + (4 ΔT / T)².
            rel_err = np.sqrt((2 * R_err / radius) ** 2 + (4 * T_err / temp) ** 2)
            L_err = lum * rel_err
            L_err_bb = lum_bb * rel_err

            L_bol.append(lum)
            L_bol_bb.append(lum_bb)
            L_bol_err.append(L_err)
            L_bol_bb_err.append(L_err_bb)

        df_bol = df_bb.copy()
        df_bol['lum_bol'] = np.array(L_bol) / 1e50
        df_bol['lum_bol_err'] = np.array(L_bol_err) / 1e50
        df_bol['lum_bol_bb'] = np.array(L_bol_bb) / 1e50
        df_bol['lum_bol_bb_err'] = np.array(L_bol_bb_err) / 1e50
        df_bol['time_rest_frame'] = df_bol['epoch_times'] / (1 + self.redshift)

        redback.utils.logger.info('Masking bolometric estimates with likely wrong extractions')
        df_bol = df_bol[df_bol['lum_bol_err'] / df_bol['lum_bol'] < 1]
        redback.utils.logger.info(
            "Estimated bolometric luminosity using blackbody integration (with boost and extinction corrections if specified).")
        return df_bol