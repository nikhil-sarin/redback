"""Structured results and interfaces for photometric SED estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    def integrate(self, extinction_magnitude: float = 0.0) -> BolometricResult:
        """Integrate every successful epoch and propagate its full covariance."""
        if not np.isfinite(extinction_magnitude):
            raise ValueError("extinction_magnitude must be finite")
        extinction_factor = 10.0 ** (0.4 * extinction_magnitude)
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
                luminosity = self.model.bolometric_luminosity(epoch.parameters)
                variance = self._luminosity_variance(epoch)
                fractions = self.model.luminosity_fractions(
                    epoch.parameters, (epoch.wavelength_min, epoch.wavelength_max))
                row.update({
                    "lum_bol": luminosity * extinction_factor / 1e50,
                    "lum_bol_err": np.sqrt(variance) * extinction_factor / 1e50,
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
