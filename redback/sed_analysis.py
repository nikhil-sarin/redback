"""Structured results and interfaces for photometric SED estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from redback import constants
from redback.sed import RedbackTimeSeriesSource, blackbody_to_flux_density
from redback.utils import (
    abmag_to_flux_density_and_error_inmjy,
    bandpass_flux_to_flux_density,
    bandpass_magnitude_to_flux,
    bands_to_effective_width,
    bands_to_frequency,
    calc_kcorrected_properties,
    lambda_to_nu,
    logger,
    nu_to_lambda,
)


@runtime_checkable
class SEDModel(Protocol):
    """Protocol implemented by SED models used by :class:`SEDResult`."""

    name: str
    parameter_names: Sequence[str]

    def bolometric_luminosity(self, parameters: Mapping[str, float]) -> float:
        """Return the integrated source luminosity in erg/s."""

    def luminosity_fractions(
            self, parameters: Mapping[str, float], wavelength_range: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Return UV, observed-range, and IR fractions of total luminosity."""

    def evaluate_photometry(
            self, coordinates: np.ndarray, parameters: Mapping[str, float], context: Mapping[str, Any]
    ) -> np.ndarray:
        """Evaluate the SED in the coordinate system stored for an epoch."""

    def fit_setup(self, **kwargs) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Return optimizer starting values and lower/upper bounds."""

    def evaluate_fit(self, coordinates: np.ndarray, *values: float, context: Mapping[str, Any]):
        """Evaluate photometry from parameters in optimizer coordinates."""

    def parameters_from_fit(
            self, values: np.ndarray, covariance: np.ndarray
    ) -> tuple[dict[str, float], np.ndarray]:
        """Convert optimizer coordinates and covariance to physical parameters."""


@dataclass(frozen=True)
class ExtinctionConfig:
    """Host and Milky Way dust applied to forward SED photometry."""

    av_host: float = 0.0
    rv_host: float = 3.1
    av_mw: float = 0.0
    rv_mw: float = 3.1
    host_law: str = "fitzpatrick99"
    mw_law: str = "fitzpatrick99"

    def __post_init__(self):
        values = (self.av_host, self.rv_host, self.av_mw, self.rv_mw)
        if not np.all(np.isfinite(values)):
            raise ValueError("Extinction values must be finite")
        if self.av_host < 0 or self.av_mw < 0:
            raise ValueError("Extinction A_V values must be non-negative")
        if self.rv_host <= 0 or self.rv_mw <= 0:
            raise ValueError("Extinction R_V values must be positive")

    def transmission(self, observer_wavelength, redshift):
        from redback.transient_models.extinction_models import _perform_extinction

        wavelength = np.atleast_1d(np.asarray(observer_wavelength, dtype=float))
        return np.asarray(_perform_extinction(
            flux_density=np.ones_like(wavelength), angstroms=wavelength,
            av_host=self.av_host, rv_host=self.rv_host,
            av_mw=self.av_mw, rv_mw=self.rv_mw, redshift=redshift,
            host_law=self.host_law, mw_law=self.mw_law), dtype=float)


@dataclass
class SEDEpochResult:
    """Fit result and diagnostic information for one photometric epoch."""

    epoch_time: float
    success: bool
    status: str
    message: str = ""
    parameters: dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray | None = None
    chi_square: float = np.nan
    degrees_of_freedom: int = 0
    n_filters: int = 0
    wavelength_min: float = np.nan
    wavelength_max: float = np.nan
    coordinates: np.ndarray | None = None
    observed: np.ndarray | None = None
    observed_error: np.ndarray | None = None
    context: dict[str, Any] = field(default_factory=dict)
    integrated_luminosity: float = np.nan
    integrated_luminosity_error: float = np.nan
    luminosity_components: dict[str, float] = field(default_factory=dict)

    @property
    def reduced_chi_square(self) -> float:
        if self.degrees_of_freedom <= 0:
            return np.nan
        return self.chi_square / self.degrees_of_freedom


@dataclass
class BolometricResult:
    """Bolometric luminosities derived from a structured SED fit."""

    transient_name: str
    method: str
    epochs: pd.DataFrame

    def to_dataframe(self, successful_only: bool = False) -> pd.DataFrame:
        data = self.epochs
        if successful_only:
            data = data[data["success"]]
        return data.copy()

    def plot_evolution(self, ax=None, rest_frame: bool = True, **kwargs):
        """Plot the successful bolometric luminosity estimates."""
        if ax is None:
            _, ax = plt.subplots()
        data = self.epochs[self.epochs["success"]]
        time_column = "time_rest_frame" if rest_frame else "epoch_times"
        ax.errorbar(
            data[time_column], data["lum_bol"], yerr=data["lum_bol_err"],
            fmt=kwargs.pop("fmt", "o"), **kwargs)
        ax.set_xlabel("Rest-frame time" if rest_frame else "Observer-frame time")
        ax.set_ylabel(r"Bolometric luminosity [$10^{50}$ erg s$^{-1}$]")
        return ax


@dataclass
class SEDResult:
    """Structured output from fitting one SED model across photometric epochs."""

    transient_name: str
    method: str
    model: SEDModel
    epoch_results: list[SEDEpochResult]
    redshift: float
    distance: float

    def to_dataframe(self, successful_only: bool = False) -> pd.DataFrame:
        """Return one row per epoch, retaining failures by default."""
        rows = []
        parameter_names = tuple(self.model.parameter_names)
        for epoch in self.epoch_results:
            row = {
                "epoch_times": epoch.epoch_time,
                "success": epoch.success,
                "status": epoch.status,
                "message": epoch.message,
                "chi_square": epoch.chi_square,
                "degrees_of_freedom": epoch.degrees_of_freedom,
                "reduced_chi_square": epoch.reduced_chi_square,
                "n_filters": epoch.n_filters,
                "wavelength_min": epoch.wavelength_min,
                "wavelength_max": epoch.wavelength_max,
                "covariance": epoch.covariance,
                "method": self.method,
            }
            for name in parameter_names:
                row[name] = epoch.parameters.get(name, np.nan)
                row[f"{name}_err"] = self._parameter_error(epoch, name)
            rows.append(row)
        data = pd.DataFrame(rows)
        if successful_only and len(data):
            data = data[data["success"]]
        return data.copy()

    def integrate(self) -> BolometricResult:
        """Integrate every successful epoch and propagate its full covariance."""
        rows = []
        for epoch in self.epoch_results:
            row = {
                "epoch_times": epoch.epoch_time,
                "time_rest_frame": epoch.epoch_time / (1.0 + self.redshift),
                "success": epoch.success,
                "status": epoch.status,
                "message": epoch.message,
                "method": self.method,
                "lum_bol": np.nan,
                "lum_bol_err": np.nan,
                "uv_fraction": np.nan,
                "observed_fraction": np.nan,
                "ir_fraction": np.nan,
            }
            row.update(epoch.parameters)
            if epoch.success:
                if np.isfinite(epoch.integrated_luminosity):
                    luminosity = epoch.integrated_luminosity
                    luminosity_error = epoch.integrated_luminosity_error
                    total = sum(epoch.luminosity_components.values())
                    fractions = tuple(
                        epoch.luminosity_components.get(name, 0.0) / total
                        for name in ("ultraviolet", "observed", "infrared"))
                    row.update({
                        f"lum_{name}": value / 1e50
                        for name, value in epoch.luminosity_components.items()})
                else:
                    luminosity = self.model.bolometric_luminosity(epoch.parameters)
                    variance = self._luminosity_variance(epoch)
                    luminosity_error = np.sqrt(variance)
                    fractions = self.model.luminosity_fractions(
                        epoch.parameters, (epoch.wavelength_min, epoch.wavelength_max))
                row.update({
                    "lum_bol": luminosity / 1e50,
                    "lum_bol_err": luminosity_error / 1e50,
                    "uv_fraction": fractions[0],
                    "observed_fraction": fractions[1],
                    "ir_fraction": fractions[2],
                })
            rows.append(row)
        return BolometricResult(
            transient_name=self.transient_name, method=self.method,
            epochs=pd.DataFrame(rows))

    def plot_epoch(self, epoch: int | float, ax=None, **kwargs):
        """Plot observations and the fitted model for one successful epoch."""
        selected = self._select_epoch(epoch)
        if not selected.success:
            raise ValueError(f"Cannot plot unsuccessful epoch: {selected.status}")
        if selected.coordinates is None or selected.observed is None:
            raise ValueError("Epoch does not contain photometry diagnostics")
        if ax is None:
            _, ax = plt.subplots()
        coordinates = np.asarray(selected.coordinates)
        model_values = self.model.evaluate_photometry(
            coordinates, selected.parameters, selected.context)
        order = np.argsort(coordinates)
        ax.errorbar(
            coordinates, selected.observed, yerr=selected.observed_error,
            fmt=kwargs.pop("fmt", "o"), label=kwargs.pop("data_label", "data"))
        ax.plot(
            coordinates[order], np.asarray(model_values)[order],
            label=kwargs.pop("model_label", self.method), **kwargs)
        ax.legend()
        return ax

    def plot_evolution(self, axes=None):
        """Plot fitted temperature and radius for successful epochs."""
        if axes is None:
            _, axes = plt.subplots(2, 1, sharex=True)
        data = self.to_dataframe(successful_only=True)
        axes[0].errorbar(data["epoch_times"], data["temperature"],
                         yerr=data.get("temperature_err"), fmt="o")
        axes[1].errorbar(data["epoch_times"], data["radius"],
                         yerr=data.get("radius_err"), fmt="o")
        axes[0].set_ylabel("Temperature [K]")
        axes[1].set_ylabel("Radius [cm]")
        axes[1].set_xlabel("Observer-frame time")
        return axes

    def _legacy_parameter_dataframe(self):
        data = self.to_dataframe(successful_only=True)
        if len(data) == 0:
            return None
        data = data[data["status"] == "success"].copy()
        if len(data) == 0:
            return None
        data = data.rename(columns={"temperature_err": "temp_err"})
        columns = ["epoch_times", "temperature", "radius", "temp_err", "radius_err"]
        if self.method == "cutoff_blackbody":
            columns.extend([
                "cutoff_wavelength", "cutoff_wavelength_err",
                "absorption_index", "absorption_index_err"])
        columns.append("method")
        return data[columns].reset_index(drop=True)

    def _legacy_bolometric_dataframe(self, lambda_cut=None, A_ext=0.0):
        if not np.isfinite(A_ext):
            raise ValueError("A_ext must be finite")
        extinction_factor = 10.0 ** (0.4 * A_ext)
        rows = []
        for epoch in self.epoch_results:
            if not epoch.success or epoch.status != "success":
                continue
            parameters = epoch.parameters
            blackbody_model = BlackbodySEDModel(self.distance, self.redshift)
            luminosity_bb = blackbody_model.bolometric_luminosity(parameters)
            luminosity = self.model.bolometric_luminosity(parameters)
            luminosity_function = self.model.bolometric_luminosity
            if self.method == "blackbody" and lambda_cut is not None:
                def luminosity_function(point):
                    redward_fraction = blackbody_model.luminosity_fractions(
                        point, (float(lambda_cut), float(lambda_cut)))[2]
                    return blackbody_model.bolometric_luminosity(point) / redward_fraction

                luminosity = luminosity_function(parameters)
            luminosity_variance = self._function_variance(
                epoch, luminosity_function)
            blackbody_variance = self._function_variance(
                epoch, blackbody_model.bolometric_luminosity)
            row = {
                "epoch_times": epoch.epoch_time,
                "temperature": parameters["temperature"],
                "radius": parameters["radius"],
                "temp_err": self._parameter_error(epoch, "temperature"),
                "radius_err": self._parameter_error(epoch, "radius"),
                "lum_bol": luminosity * extinction_factor / 1e50,
                "lum_bol_err": np.sqrt(luminosity_variance) * extinction_factor / 1e50,
                "lum_bol_bb": luminosity_bb * extinction_factor / 1e50,
                "lum_bol_bb_err": np.sqrt(blackbody_variance) * extinction_factor / 1e50,
                "method": self.method,
                "time_rest_frame": epoch.epoch_time / (1.0 + self.redshift),
            }
            if self.method == "cutoff_blackbody":
                fraction = luminosity / luminosity_bb
                row.update({
                    "cutoff_wavelength": parameters["cutoff_wavelength"],
                    "cutoff_wavelength_err": self._parameter_error(epoch, "cutoff_wavelength"),
                    "absorption_index": parameters["absorption_index"],
                    "absorption_index_err": self._parameter_error(epoch, "absorption_index"),
                    "cutoff_fraction": fraction,
                    "cutoff_boost": 1.0 / fraction,
                })
            rows.append(row)
        if not rows:
            return None
        data = pd.DataFrame(rows)
        return data[data["lum_bol_err"] / data["lum_bol"] < 1].reset_index(drop=True)

    def _select_epoch(self, epoch: int | float) -> SEDEpochResult:
        if isinstance(epoch, (int, np.integer)):
            return self.epoch_results[int(epoch)]
        times = np.asarray([item.epoch_time for item in self.epoch_results])
        return self.epoch_results[int(np.argmin(np.abs(times - float(epoch))))]

    def _parameter_error(self, epoch: SEDEpochResult, name: str) -> float:
        if epoch.covariance is None:
            return np.nan
        index = list(self.model.parameter_names).index(name)
        variance = epoch.covariance[index, index]
        return np.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan

    def _luminosity_variance(self, epoch: SEDEpochResult) -> float:
        if epoch.covariance is None:
            return np.nan
        covariance = np.asarray(epoch.covariance, dtype=float)
        gradient = self._luminosity_gradient(epoch.parameters)
        variance = float(gradient @ covariance @ gradient)
        if not np.isfinite(variance):
            return np.nan
        return max(variance, 0.0)

    def _function_variance(self, epoch, function):
        if epoch.covariance is None:
            return np.nan
        gradient = []
        for name in self.model.parameter_names:
            value = float(epoch.parameters[name])
            step = max(abs(value) * 1e-5, 1e-8)
            lower, upper = dict(epoch.parameters), dict(epoch.parameters)
            lower[name], upper[name] = value - step, value + step
            gradient.append((function(upper) - function(lower)) / (2.0 * step))
        gradient = np.asarray(gradient)
        variance = float(gradient @ epoch.covariance @ gradient)
        return max(variance, 0.0) if np.isfinite(variance) else np.nan

    def _luminosity_gradient(self, parameters: Mapping[str, float]) -> np.ndarray:
        gradient = []
        for name in self.model.parameter_names:
            value = float(parameters[name])
            step = max(abs(value) * 1e-5, 1e-8)
            lower = dict(parameters)
            upper = dict(parameters)
            lower[name] = value - step
            upper[name] = value + step
            lower_value = self.model.bolometric_luminosity(lower)
            upper_value = self.model.bolometric_luminosity(upper)
            gradient.append((upper_value - lower_value) / (2.0 * step))
        return np.asarray(gradient)


class BlackbodySEDModel:
    """Planck SED with temperature and photospheric radius parameters."""

    name = "blackbody"
    parameter_names = ("temperature", "radius")

    def __init__(
            self, distance: float, redshift: float,
            extinction: ExtinctionConfig | None = None):
        self.distance = float(distance)
        self.redshift = float(redshift)
        self.extinction = extinction

    def fit_setup(self, **kwargs):
        initial = np.array([
            np.log10(kwargs.get("T_init", 1e4)),
            np.log10(kwargs.get("R_init", 1e15)),
        ])
        bounds = kwargs.get("parameter_bounds", ((3.0, 8.0), (10.0, 20.0)))
        lower = np.array([bounds[0][0], bounds[1][0]], dtype=float)
        upper = np.array([bounds[0][1], bounds[1][1]], dtype=float)
        return initial, (lower, upper)

    def parameters_from_fit(self, values, covariance):
        temperature, radius = np.power(10.0, values)
        jacobian = np.diag([np.log(10.0) * temperature, np.log(10.0) * radius])
        physical_covariance = jacobian @ covariance @ jacobian.T
        return {"temperature": temperature, "radius": radius}, physical_covariance

    def evaluate_fit(self, coordinates, *values, context):
        parameters, _ = self.parameters_from_fit(values, np.zeros((len(values), len(values))))
        if context["coordinate_type"] == "band":
            indices = np.round(coordinates).astype(int)
            coordinates = np.asarray(context["bands"])[indices]
        return self.evaluate_photometry(coordinates, parameters, context)

    def evaluate_photometry(self, coordinates, parameters, context):
        if context["coordinate_type"] == "frequency":
            return self._flux_density_mjy(np.asarray(coordinates, dtype=float), parameters)
        return self._bandpass_photometry(np.asarray(coordinates), parameters, context["data_mode"])

    def bolometric_luminosity(self, parameters):
        return (
            4.0 * np.pi * parameters["radius"] ** 2 * constants.sigma_sb
            * parameters["temperature"] ** 4
        )

    def luminosity_fractions(self, parameters, wavelength_range):
        return self._dimensionless_fractions(
            parameters["temperature"], wavelength_range, cutoff_wavelength=None,
            absorption_index=0.0)

    def _flux_density_mjy(self, rest_frequency, parameters):
        import astropy.units as uu

        flux = blackbody_to_flux_density(
            parameters["temperature"], parameters["radius"], self.distance,
            rest_frequency) * (1.0 + self.redshift)
        flux_mjy = (flux / (1e-26 * uu.erg / uu.s / uu.cm ** 2 / uu.Hz)).value
        flux_mjy *= self._suppression(nu_to_lambda(rest_frequency), parameters)
        if self.extinction is not None:
            observer_wavelength = nu_to_lambda(rest_frequency) * (1.0 + self.redshift)
            flux_mjy *= self.extinction.transmission(observer_wavelength, self.redshift)
        return flux_mjy

    def _suppression(self, rest_wavelength, parameters):
        return np.ones_like(np.asarray(rest_wavelength, dtype=float))

    def _bandpass_photometry(self, bands, parameters, output_format):
        import astropy.units as uu

        observer_wavelength = np.geomspace(100.0, 80000.0, 500)
        observer_frequency = lambda_to_nu(observer_wavelength)
        rest_frequency, _ = calc_kcorrected_properties(
            frequency=observer_frequency, redshift=self.redshift, time=0.0)
        flux_nu = blackbody_to_flux_density(
            parameters["temperature"], parameters["radius"], self.distance,
            rest_frequency) * (1.0 + self.redshift)
        flux_nu *= self._suppression(nu_to_lambda(rest_frequency), parameters)
        if self.extinction is not None:
            flux_nu *= self.extinction.transmission(observer_wavelength, self.redshift)
        flux_lambda = flux_nu.to(
            uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
            equivalencies=uu.spectral_density(wav=observer_wavelength * uu.Angstrom))
        phases = np.arange(5, dtype=float)
        spectra = np.repeat(flux_lambda.value[None, :], len(phases), axis=0)
        source = RedbackTimeSeriesSource(
            phase=phases, wave=observer_wavelength, flux=spectra)
        magnitudes = source.bandmag(phase=0.0, band=bands, magsys="ab")
        if output_format == "magnitude":
            return magnitudes
        if output_format == "flux":
            return bandpass_magnitude_to_flux(magnitude=magnitudes, bands=bands)
        raise ValueError(f"Bandpass SED fitting does not support data_mode={output_format!r}")

    @staticmethod
    def _dimensionless_fractions(
            temperature, wavelength_range, cutoff_wavelength=None, absorption_index=0.0):
        wavelength_min, wavelength_max = wavelength_range
        if not (
                np.isfinite(temperature) and temperature > 0
                and np.isfinite(wavelength_min) and np.isfinite(wavelength_max)
                and 0 < wavelength_min <= wavelength_max):
            return np.nan, np.nan, np.nan

        scale = constants.planck * constants.speed_of_light / (
            constants.angstrom_cgs * constants.boltzmann_constant * temperature)
        x_min = scale / wavelength_max
        x_max = scale / wavelength_min
        x_cutoff = None if cutoff_wavelength is None else scale / cutoff_wavelength

        def integrand(x):
            if x <= 0.0 or x > 700.0:
                return 0.0
            value = x ** 3 / np.expm1(x)
            if x_cutoff is not None and x > x_cutoff:
                value *= (x_cutoff / x) ** absorption_index
            return value

        total, _ = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1e-8, limit=200)
        infrared, _ = quad(integrand, 0.0, x_min, epsabs=0.0, epsrel=1e-8, limit=200)
        observed, _ = quad(integrand, x_min, x_max, epsabs=0.0, epsrel=1e-8, limit=200)
        ultraviolet = max(total - infrared - observed, 0.0)
        return ultraviolet / total, observed / total, infrared / total

    @classmethod
    def _bolometric_fraction(cls, temperature, cutoff_wavelength, absorption_index):
        scale = constants.planck * constants.speed_of_light / (
            constants.angstrom_cgs * constants.boltzmann_constant * temperature)
        x_cutoff = scale / cutoff_wavelength

        def integrand(x):
            if x <= 0.0 or x > 700.0:
                return 0.0
            value = x ** 3 / np.expm1(x)
            if x > x_cutoff:
                value *= (x_cutoff / x) ** absorption_index
            return value

        integral, _ = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1e-8, limit=200)
        return 15.0 * integral / np.pi ** 4


class CutoffBlackbodySEDModel(BlackbodySEDModel):
    """Blackbody with power-law suppression blueward of a rest wavelength."""

    name = "cutoff_blackbody"

    def __init__(
            self, distance, redshift, cutoff_wavelength=3000.0, absorption_index=1.0,
            fit_cutoff_wavelength=False, fit_absorption_index=False, extinction=None):
        super().__init__(distance=distance, redshift=redshift, extinction=extinction)
        if not np.isfinite(cutoff_wavelength) or cutoff_wavelength <= 0:
            raise ValueError("cutoff_wavelength must be finite and positive")
        if not np.isfinite(absorption_index) or absorption_index < 0:
            raise ValueError("absorption_index must be finite and non-negative")
        self.cutoff_wavelength = float(cutoff_wavelength)
        self.absorption_index = float(absorption_index)
        self.fit_cutoff_wavelength = bool(fit_cutoff_wavelength)
        self.fit_absorption_index = bool(fit_absorption_index)
        names = ["temperature", "radius"]
        if self.fit_cutoff_wavelength:
            names.append("cutoff_wavelength")
        if self.fit_absorption_index:
            names.append("absorption_index")
        self.fit_parameter_names = tuple(names)
        self.parameter_names = (
            "temperature", "radius", "cutoff_wavelength", "absorption_index")

    def fit_setup(self, **kwargs):
        initial, bounds = super().fit_setup(**kwargs)
        lower, upper = [list(item) for item in bounds]
        initial = list(initial)
        if self.fit_cutoff_wavelength:
            cutoff_bounds = kwargs.get("cutoff_wavelength_bounds", (100.0, 100000.0))
            initial.append(np.log10(self.cutoff_wavelength))
            lower.append(np.log10(cutoff_bounds[0]))
            upper.append(np.log10(cutoff_bounds[1]))
        if self.fit_absorption_index:
            index_bounds = kwargs.get("absorption_index_bounds", (0.0, 10.0))
            initial.append(self.absorption_index)
            lower.append(index_bounds[0])
            upper.append(index_bounds[1])
        return np.asarray(initial), (np.asarray(lower), np.asarray(upper))

    def parameters_from_fit(self, values, covariance):
        temperature, radius = np.power(10.0, values[:2])
        parameters = {
            "temperature": temperature,
            "radius": radius,
            "cutoff_wavelength": self.cutoff_wavelength,
            "absorption_index": self.absorption_index,
        }
        transform = [np.log(10.0) * temperature, np.log(10.0) * radius]
        cursor = 2
        if self.fit_cutoff_wavelength:
            parameters["cutoff_wavelength"] = 10.0 ** values[cursor]
            transform.append(np.log(10.0) * parameters["cutoff_wavelength"])
            cursor += 1
        if self.fit_absorption_index:
            parameters["absorption_index"] = values[cursor]
            transform.append(1.0)
        fitted_covariance = np.diag(transform) @ covariance @ np.diag(transform)
        physical_covariance = np.zeros((4, 4))
        fitted_indices = [0, 1]
        if self.fit_cutoff_wavelength:
            fitted_indices.append(2)
        if self.fit_absorption_index:
            fitted_indices.append(3)
        physical_covariance[np.ix_(fitted_indices, fitted_indices)] = fitted_covariance
        return parameters, physical_covariance

    def _suppression(self, rest_wavelength, parameters):
        wavelength = np.asarray(rest_wavelength, dtype=float)
        suppression = np.ones_like(wavelength)
        mask = wavelength < parameters["cutoff_wavelength"]
        suppression[mask] = (
            wavelength[mask] / parameters["cutoff_wavelength"]
        ) ** parameters["absorption_index"]
        return suppression

    def bolometric_luminosity(self, parameters):
        fraction = self._bolometric_fraction(
            parameters["temperature"], parameters["cutoff_wavelength"],
            parameters["absorption_index"])
        return super().bolometric_luminosity(parameters) * fraction

    def luminosity_fractions(self, parameters, wavelength_range):
        return self._dimensionless_fractions(
            parameters["temperature"], wavelength_range,
            cutoff_wavelength=parameters["cutoff_wavelength"],
            absorption_index=parameters["absorption_index"])


class DirectIntegrationSEDModel:
    """Observed-range integration with explicitly selected extrapolation tails."""

    name = "direct_integration"

    def __init__(self, tail_model):
        self.tail_model = tail_model
        self.parameter_names = tail_model.parameter_names if tail_model is not None else ()

    def fit_setup(self, **kwargs):
        if self.tail_model is None:
            return np.array([]), (np.array([]), np.array([]))
        return self.tail_model.fit_setup(**kwargs)

    def parameters_from_fit(self, values, covariance):
        if self.tail_model is None:
            return {}, np.empty((0, 0))
        return self.tail_model.parameters_from_fit(values, covariance)

    def evaluate_fit(self, coordinates, *values, context):
        return self.tail_model.evaluate_fit(coordinates, *values, context=context)

    def evaluate_photometry(self, coordinates, parameters, context):
        if self.tail_model is None:
            raise ValueError("No tail model was fitted for this direct integration")
        return self.tail_model.evaluate_photometry(coordinates, parameters, context)

    def bolometric_luminosity(self, parameters):
        raise RuntimeError("Direct integration luminosity is stored per epoch")

    def luminosity_fractions(self, parameters, wavelength_range):
        raise RuntimeError("Direct integration fractions are stored per epoch")


def estimate_sed(
        transient, method: str | SEDModel = "blackbody", distance: float = 1e27,
        bin_width: float = 1.0, min_filters: int = 3, bandpass: bool = True,
        extinction: ExtinctionConfig | None = None, **kwargs) -> SEDResult:
    """Fit an SED independently in each time bin of an optical transient."""
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance must be finite and positive")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    if not isinstance(min_filters, (int, np.integer)) or min_filters < 1:
        raise ValueError("min_filters must be a positive integer")
    redshift = float(transient.redshift)
    if not np.isfinite(redshift) or redshift <= 0:
        raise ValueError("A finite, positive redshift is required for SED estimation")

    if isinstance(method, str) and method.lower() in {"direct", "direct_integration"}:
        return _estimate_direct_sed(
            transient=transient, distance=distance, redshift=redshift,
            bin_width=bin_width, min_filters=min_filters, extinction=extinction, **kwargs)

    sed_model = _resolve_sed_model(
        method=method, distance=distance, redshift=redshift,
        extinction=extinction, **kwargs)
    time, coordinates, wavelength, observed, observed_error, context = _prepare_photometry(
        transient=transient, redshift=redshift, bandpass=bandpass)
    _validate_photometry(time, wavelength, observed, observed_error)

    order = np.argsort(time)
    time = time[order]
    coordinates = coordinates[order]
    wavelength = wavelength[order]
    observed = observed[order]
    observed_error = observed_error[order]

    edges = _time_bin_edges(time, bin_width)
    epoch_results = []
    initial, bounds = sed_model.fit_setup(**kwargs)
    maxfev = int(kwargs.get("maxfev", 1000))
    for index, (lower_edge, upper_edge) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (time >= lower_edge) & (time <= upper_edge)
        else:
            mask = (time >= lower_edge) & (time < upper_edge)
        if not np.any(mask):
            continue
        epoch_time = float(np.mean(time[mask]))
        epoch_wavelength = wavelength[mask]
        n_filters = len(np.unique(coordinates[mask]))
        common = dict(
            epoch_time=epoch_time, n_filters=n_filters,
            wavelength_min=float(np.min(epoch_wavelength)),
            wavelength_max=float(np.max(epoch_wavelength)),
            coordinates=coordinates[mask].copy(), observed=observed[mask].copy(),
            observed_error=observed_error[mask].copy(), context=context.copy())
        if n_filters < min_filters:
            epoch_results.append(SEDEpochResult(
                success=False, status="insufficient_filters",
                message=f"requires {min_filters} distinct filters", **common))
            continue

        epoch_context = context.copy()
        if context["coordinate_type"] == "band":
            epoch_context["bands"] = coordinates[mask].copy()
            fit_coordinates = np.arange(np.sum(mask), dtype=float)
        else:
            fit_coordinates = np.asarray(coordinates[mask], dtype=float)
        try:
            parameters_fit, covariance_fit = curve_fit(
                lambda x, *values: sed_model.evaluate_fit(
                    x, *values, context=epoch_context),
                fit_coordinates, observed[mask], sigma=observed_error[mask],
                p0=initial, bounds=bounds, absolute_sigma=True, maxfev=maxfev)
            parameters, covariance = sed_model.parameters_from_fit(
                parameters_fit, covariance_fit)
            prediction = sed_model.evaluate_fit(
                fit_coordinates, *parameters_fit, context=epoch_context)
            chi_square = float(np.sum(((observed[mask] - prediction) / observed_error[mask]) ** 2))
            dof = int(np.sum(mask) - len(parameters_fit))
            status = _quality_status(sed_model, parameters, covariance)
            epoch_results.append(SEDEpochResult(
                success=True, status=status, parameters=parameters,
                covariance=covariance, chi_square=chi_square,
                degrees_of_freedom=dof, **common))
        except Exception as exc:
            logger.warning("SED fit failed at epoch %.6g: %s", epoch_time, exc)
            epoch_results.append(SEDEpochResult(
                success=False, status="fit_failed", message=str(exc), **common))

    return SEDResult(
        transient_name=transient.name, method=sed_model.name, model=sed_model,
        epoch_results=epoch_results, redshift=redshift, distance=distance)


def _resolve_sed_model(method, distance, redshift, extinction=None, **kwargs):
    if not isinstance(method, str):
        if not isinstance(method, SEDModel):
            raise TypeError("Custom SED methods must implement the SEDModel protocol")
        return method
    normalized = method.lower()
    if normalized in {"bb", "blackbody"}:
        return BlackbodySEDModel(
            distance=distance, redshift=redshift, extinction=extinction)
    if normalized in {"cutoff", "cutoff_bb", "cutoff_blackbody"}:
        return CutoffBlackbodySEDModel(
            distance=distance, redshift=redshift,
            cutoff_wavelength=kwargs.get("cutoff_wavelength", kwargs.get("lambda_cut", 3000.0)),
            absorption_index=kwargs.get("absorption_index", 1.0),
            fit_cutoff_wavelength=kwargs.get("fit_cutoff_wavelength", False),
            fit_absorption_index=kwargs.get("fit_absorption_index", False),
            extinction=extinction)
    raise ValueError(f"Unknown SED method {method!r}")


def _prepare_photometry(transient, redshift, bandpass):
    time, _, observed, observed_error = transient.get_filtered_data()
    time = np.asarray(time, dtype=float)
    observed = np.asarray(observed, dtype=float)
    observed_error = np.asarray(observed_error, dtype=float)
    if transient.data_mode in {"magnitude", "flux"}:
        bands = np.asarray(transient.filtered_sncosmo_bands)
        observer_frequency = np.asarray(bands_to_frequency(bands), dtype=float)
        rest_frequency, _ = calc_kcorrected_properties(
            frequency=observer_frequency, redshift=redshift, time=0.0)
        wavelength = np.asarray(nu_to_lambda(rest_frequency), dtype=float)
        if bandpass:
            return time, bands, wavelength, observed, observed_error, {
                "coordinate_type": "band", "data_mode": transient.data_mode}
        if transient.data_mode == "magnitude":
            observed, observed_error = abmag_to_flux_density_and_error_inmjy(
                observed, observed_error)
        else:
            widths = bands_to_effective_width(bands)
            observed, observed_error = bandpass_flux_to_flux_density(
                observed, observed_error, widths)
        return time, rest_frequency, wavelength, observed, observed_error, {
            "coordinate_type": "frequency", "data_mode": "flux_density"}

    if transient.data_mode != "flux_density":
        raise ValueError(
            "SED estimation supports magnitude, flux, and flux_density data modes")
    observer_frequency = np.asarray(transient.filtered_frequencies, dtype=float)
    rest_frequency, _ = calc_kcorrected_properties(
        frequency=observer_frequency, redshift=redshift, time=0.0)
    wavelength = np.asarray(nu_to_lambda(rest_frequency), dtype=float)
    return time, rest_frequency, wavelength, observed, observed_error, {
        "coordinate_type": "frequency", "data_mode": "flux_density"}


def _validate_photometry(time, wavelength, observed, observed_error):
    lengths = {len(time), len(wavelength), len(observed), len(observed_error)}
    if len(lengths) != 1:
        raise ValueError("Time, photometry, uncertainty, and spectral coordinates must align")
    if len(time) == 0:
        raise ValueError("Cannot estimate an SED from empty photometry")
    if not np.all(np.isfinite(time)):
        raise ValueError("SED times must be finite")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("SED wavelengths must be finite and positive")
    if not np.all(np.isfinite(observed)):
        raise ValueError("SED photometry must be finite")
    if not np.all(np.isfinite(observed_error)) or np.any(observed_error <= 0):
        raise ValueError("SED uncertainties must be finite and positive")


def _time_bin_edges(time, bin_width):
    start = float(np.min(time))
    stop = float(np.max(time))
    if start == stop:
        return np.array([start, start + bin_width])
    edges = np.arange(start, stop + bin_width, bin_width)
    if edges[-1] < stop:
        edges = np.append(edges, edges[-1] + bin_width)
    return edges


def _quality_status(model, parameters, covariance):
    if covariance is None or not np.all(np.isfinite(covariance)):
        return "invalid_covariance"
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    for name, error in zip(model.parameter_names, errors):
        value = parameters[name]
        if value != 0 and error / abs(value) >= 1:
            return "poorly_constrained"
    return "success"


def _estimate_direct_sed(
        transient, distance, redshift, bin_width, min_filters, uv_tail=None,
        ir_tail=None, extinction=None, **kwargs):
    allowed_tails = {"none", "blackbody", "cutoff_blackbody"}
    if uv_tail not in allowed_tails or ir_tail not in allowed_tails:
        raise ValueError(
            "direct_integration requires explicit uv_tail and ir_tail values from "
            "'none', 'blackbody', or 'cutoff_blackbody'")
    selected = {uv_tail, ir_tail} - {"none"}
    tail_name = "cutoff_blackbody" if "cutoff_blackbody" in selected else (
        "blackbody" if selected else None)
    tail_model = None if tail_name is None else _resolve_sed_model(
        tail_name, distance=distance, redshift=redshift,
        extinction=extinction, **kwargs)
    model = DirectIntegrationSEDModel(tail_model)
    time, frequency, wavelength, observed, observed_error, context = _prepare_photometry(
        transient=transient, redshift=redshift, bandpass=False)
    _validate_photometry(time, wavelength, observed, observed_error)
    order = np.argsort(time)
    time, frequency, wavelength = time[order], frequency[order], wavelength[order]
    observed, observed_error = observed[order], observed_error[order]
    initial, bounds = model.fit_setup(**kwargs)
    maxfev = int(kwargs.get("maxfev", 1000))
    epochs = []
    edges = _time_bin_edges(time, bin_width)
    for index, (lower_edge, upper_edge) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (time >= lower_edge) & (
            (time <= upper_edge) if index == len(edges) - 2 else (time < upper_edge))
        if not np.any(mask):
            continue
        epoch_time = float(np.mean(time[mask]))
        epoch_frequency = np.asarray(frequency[mask], dtype=float)
        epoch_wavelength = np.asarray(wavelength[mask], dtype=float)
        epoch_flux = np.asarray(observed[mask], dtype=float)
        epoch_error = np.asarray(observed_error[mask], dtype=float)
        n_filters = len(np.unique(epoch_frequency))
        common = dict(
            epoch_time=epoch_time, n_filters=n_filters,
            wavelength_min=float(np.min(epoch_wavelength)),
            wavelength_max=float(np.max(epoch_wavelength)),
            coordinates=epoch_frequency.copy(), observed=epoch_flux.copy(),
            observed_error=epoch_error.copy(), context=context.copy())
        if n_filters < min_filters:
            epochs.append(SEDEpochResult(
                success=False, status="insufficient_filters",
                message=f"requires {min_filters} distinct filters", **common))
            continue
        try:
            if tail_model is None:
                parameters, covariance = {}, np.empty((0, 0))
                chi_square, dof = np.nan, 0
            else:
                values, fit_covariance = curve_fit(
                    lambda x, *pars: model.evaluate_fit(x, *pars, context=context),
                    epoch_frequency, epoch_flux, sigma=epoch_error, p0=initial,
                    bounds=bounds, absolute_sigma=True, maxfev=maxfev)
                parameters, covariance = model.parameters_from_fit(values, fit_covariance)
                prediction = model.evaluate_fit(epoch_frequency, *values, context=context)
                chi_square = float(np.sum(((epoch_flux - prediction) / epoch_error) ** 2))
                dof = int(len(epoch_flux) - len(values))

            observed_luminosity, observed_variance = _integrate_observed_flux(
                epoch_frequency, epoch_flux, epoch_error, distance, redshift,
                extinction=extinction)
            ultraviolet = _tail_luminosity(
                uv_tail, "ultraviolet", tail_model, parameters,
                (common["wavelength_min"], common["wavelength_max"]))
            infrared = _tail_luminosity(
                ir_tail, "infrared", tail_model, parameters,
                (common["wavelength_min"], common["wavelength_max"]))
            tail_variance = _tail_variance(
                uv_tail, ir_tail, tail_model, parameters, covariance,
                (common["wavelength_min"], common["wavelength_max"]))
            components = {
                "ultraviolet": ultraviolet,
                "observed": observed_luminosity,
                "infrared": infrared,
            }
            luminosity = sum(components.values())
            epochs.append(SEDEpochResult(
                success=True, status=_quality_status(model, parameters, covariance),
                parameters=parameters, covariance=covariance, chi_square=chi_square,
                degrees_of_freedom=dof, integrated_luminosity=luminosity,
                integrated_luminosity_error=np.sqrt(observed_variance + tail_variance),
                luminosity_components=components, **common))
        except Exception as exc:
            logger.warning("Direct SED integration failed at epoch %.6g: %s", epoch_time, exc)
            epochs.append(SEDEpochResult(
                success=False, status="fit_failed", message=str(exc), **common))
    return SEDResult(
        transient_name=transient.name, method=model.name, model=model,
        epoch_results=epochs, redshift=redshift, distance=distance)


def _integrate_observed_flux(
        frequency, flux_mjy, error_mjy, distance, redshift, extinction=None):
    order = np.argsort(frequency)
    x = frequency[order]
    flux = flux_mjy[order] * 1e-26
    error = error_mjy[order] * 1e-26
    if extinction is not None:
        observer_wavelength = nu_to_lambda(x) * (1.0 + redshift)
        transmission = extinction.transmission(observer_wavelength, redshift)
        flux = flux / transmission
        error = error / transmission
    scale = 4.0 * np.pi * distance ** 2 / (1.0 + redshift)
    luminosity = scale * trapezoid(flux, x=x)
    basis = np.eye(len(x))
    weights = np.asarray([trapezoid(row, x=x) for row in basis])
    variance = scale ** 2 * np.sum((weights * error) ** 2)
    return float(luminosity), float(variance)


def _tail_luminosity(tail_name, region, tail_model, parameters, wavelength_range):
    if tail_name == "none":
        return 0.0
    if tail_name == "blackbody" and isinstance(tail_model, CutoffBlackbodySEDModel):
        integration_model = BlackbodySEDModel(tail_model.distance, tail_model.redshift)
    else:
        integration_model = tail_model
    fractions = integration_model.luminosity_fractions(parameters, wavelength_range)
    fraction = fractions[0] if region == "ultraviolet" else fractions[2]
    return integration_model.bolometric_luminosity(parameters) * fraction


def _tail_variance(uv_tail, ir_tail, tail_model, parameters, covariance, wavelength_range):
    if tail_model is None or covariance is None or covariance.size == 0:
        return 0.0

    def total_tail(values):
        point = dict(parameters)
        point.update(zip(tail_model.parameter_names, values))
        return (
            _tail_luminosity(uv_tail, "ultraviolet", tail_model, point, wavelength_range)
            + _tail_luminosity(ir_tail, "infrared", tail_model, point, wavelength_range))

    values = np.asarray([parameters[name] for name in tail_model.parameter_names])
    gradient = []
    for index, value in enumerate(values):
        step = max(abs(value) * 1e-5, 1e-8)
        lower, upper = values.copy(), values.copy()
        lower[index], upper[index] = value - step, value + step
        gradient.append((total_tail(upper) - total_tail(lower)) / (2.0 * step))
    gradient = np.asarray(gradient)
    variance = float(gradient @ covariance @ gradient)
    return max(variance, 0.0) if np.isfinite(variance) else np.nan
