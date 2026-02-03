from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from redback.utils import logger
from redback.spectral.folding import fold_spectrum
from redback.spectral.conversions import mjy_to_energy_flux_per_keV
from redback.spectral.io import read_pha, OGIPPHASpectrum
from redback.spectral.response import ResponseMatrix, EffectiveArea


@dataclass
class SpectralDataset:
    counts: np.ndarray
    exposure: float
    energy_edges_keV: np.ndarray
    data_mode: str = "spectrum_counts"
    name: str = "spectral_dataset"
    backscale: float = 1.0
    areascal: float = 1.0
    counts_bkg: Optional[np.ndarray] = None
    bkg_exposure: Optional[float] = None
    bkg_backscale: Optional[float] = None
    bkg_areascal: Optional[float] = None
    rmf: Optional[ResponseMatrix] = None
    arf: Optional[EffectiveArea] = None
    quality: Optional[np.ndarray] = None
    grouping: Optional[np.ndarray] = None
    active_energy_min: Optional[float] = None
    active_energy_max: Optional[float] = None

    @property
    def energy_centers_keV(self) -> np.ndarray:
        return 0.5 * (self.energy_edges_keV[:-1] + self.energy_edges_keV[1:])

    @property
    def energy_centers_hz(self) -> np.ndarray:
        keV_to_Hz = 2.417989e17
        return self.energy_centers_keV * keV_to_Hz

    @property
    def background_scale_factor(self) -> float:
        if self.counts_bkg is None:
            return 1.0
        bkg_exposure = self.bkg_exposure if self.bkg_exposure is not None else self.exposure
        bkg_backscale = self.bkg_backscale if self.bkg_backscale is not None else self.backscale
        bkg_areascal = self.bkg_areascal if self.bkg_areascal is not None else self.areascal
        if bkg_exposure <= 0 or bkg_backscale <= 0 or bkg_areascal <= 0:
            return 1.0
        return (self.exposure * self.backscale * self.areascal) / (bkg_exposure * bkg_backscale * bkg_areascal)

    def _get_plot_axis(self):
        if self.rmf is not None and len(self.counts) == len(self.rmf.channel):
            centers = 0.5 * (self.rmf.emin_chan + self.rmf.emax_chan)
            widths = self.rmf.emax_chan - self.rmf.emin_chan
        else:
            centers = self.energy_centers_keV
            widths = self.energy_edges_keV[1:] - self.energy_edges_keV[:-1]
        return centers, widths

    def _group_with_flags(self, counts: np.ndarray, x: np.ndarray, w: np.ndarray):
        """
        Apply OGIP GROUPING to counts. GROUPING convention:
        1 = start of a group, -1 = continuation, 0 = no grouping.
        """
        if self.grouping is None:
            return counts, x, w
        grouping = np.asarray(self.grouping)
        if len(grouping) != len(counts):
            return counts, x, w

        grouped_counts = []
        grouped_x = []
        grouped_w = []
        acc = 0.0
        acc_w = 0.0
        acc_xw = 0.0
        in_group = False
        for g, c, xc, wc in zip(grouping, counts, x, w):
            if g == 1:
                if in_group:
                    grouped_counts.append(acc)
                    grouped_w.append(acc_w)
                    grouped_x.append(acc_xw / acc_w if acc_w > 0 else xc)
                acc = float(c)
                acc_w = float(wc)
                acc_xw = float(xc) * float(wc)
                in_group = True
            elif g == -1 and in_group:
                acc += float(c)
                acc_w += float(wc)
                acc_xw += float(xc) * float(wc)
            else:
                if in_group:
                    grouped_counts.append(acc)
                    grouped_w.append(acc_w)
                    grouped_x.append(acc_xw / acc_w if acc_w > 0 else xc)
                    in_group = False
                grouped_counts.append(float(c))
                grouped_w.append(float(wc))
                grouped_x.append(float(xc))
        if in_group:
            grouped_counts.append(acc)
            grouped_w.append(acc_w)
            grouped_x.append(acc_xw / acc_w if acc_w > 0 else grouped_x[-1])
        return (np.asarray(grouped_counts, dtype=float),
                np.asarray(grouped_x, dtype=float),
                np.asarray(grouped_w, dtype=float))

    def _group_min_counts(self, counts: np.ndarray, x: np.ndarray, w: np.ndarray, min_counts: Optional[int]):
        if min_counts is None or min_counts <= 0:
            return counts, x, w
        grouped_counts = []
        grouped_x = []
        grouped_w = []
        acc = 0.0
        acc_w = 0.0
        acc_xw = 0.0
        for c, xc, wc in zip(counts, x, w):
            acc += float(c)
            acc_w += float(wc)
            acc_xw += float(xc) * float(wc)
            if acc >= min_counts:
                grouped_counts.append(acc)
                grouped_w.append(acc_w)
                grouped_x.append(acc_xw / acc_w if acc_w > 0 else xc)
                acc = 0.0
                acc_w = 0.0
                acc_xw = 0.0
        if acc > 0:
            grouped_counts.append(acc)
            grouped_w.append(acc_w)
            grouped_x.append(acc_xw / acc_w if acc_w > 0 else x[-1])
        return (np.asarray(grouped_counts, dtype=float),
                np.asarray(grouped_x, dtype=float),
                np.asarray(grouped_w, dtype=float))

    def _compute_grouping(self, counts: np.ndarray, x: np.ndarray, w: np.ndarray, min_counts: Optional[int]):
        """
        Compute grouped counts/x/w plus index groups based on OGIP GROUPING and min_counts.
        """
        # Start with per-channel indices
        indices = [np.array([i]) for i in range(len(counts))]

        if self.grouping is not None and len(self.grouping) == len(counts):
            grouped = []
            current = []
            for i, g in enumerate(self.grouping):
                if g == 1:
                    if current:
                        grouped.append(np.array(current))
                    current = [i]
                elif g == -1 and current:
                    current.append(i)
                else:
                    if current:
                        grouped.append(np.array(current))
                        current = []
                    grouped.append(np.array([i]))
            if current:
                grouped.append(np.array(current))
            indices = grouped

        # Apply min_counts grouping on top of existing groups
        if min_counts is not None and min_counts > 0:
            regrouped = []
            acc = []
            acc_counts = 0.0
            for grp in indices:
                acc.append(grp)
                acc_counts += float(np.sum(counts[grp]))
                if acc_counts >= min_counts:
                    regrouped.append(np.concatenate(acc))
                    acc = []
                    acc_counts = 0.0
            if acc:
                regrouped.append(np.concatenate(acc))
            indices = regrouped

        grouped_counts = []
        grouped_x = []
        grouped_w = []
        for grp in indices:
            grp_counts = counts[grp]
            grp_w = w[grp]
            grp_x = x[grp]
            wsum = float(np.sum(grp_w))
            xw = float(np.sum(grp_x * grp_w))
            grouped_counts.append(float(np.sum(grp_counts)))
            grouped_w.append(wsum)
            grouped_x.append(xw / wsum if wsum > 0 else float(np.mean(grp_x)))

        return (np.asarray(grouped_counts, dtype=float),
                np.asarray(grouped_x, dtype=float),
                np.asarray(grouped_w, dtype=float),
                indices)

    @staticmethod
    def _apply_group_indices(values: np.ndarray, groups):
        return np.asarray([np.sum(values[grp]) for grp in groups], dtype=float)

    def mask_valid(self) -> np.ndarray:
        if self.quality is None:
            qual_mask = np.ones_like(self.counts, dtype=bool)
        else:
            qual_mask = np.asarray(self.quality) == 0

        if self.active_energy_min is None and self.active_energy_max is None:
            return qual_mask

        centers, _ = self._get_plot_axis()
        emin = -np.inf if self.active_energy_min is None else self.active_energy_min
        emax = np.inf if self.active_energy_max is None else self.active_energy_max
        energy_mask = (centers >= emin) & (centers <= emax)
        return qual_mask & energy_mask

    def set_active_interval(self, emin_keV: float, emax_keV: float):
        self.active_energy_min = float(emin_keV)
        self.active_energy_max = float(emax_keV)
        logger.info("Set active energy interval: %.3f-%.3f keV", self.active_energy_min, self.active_energy_max)

    def predict_counts(self, model, parameters: dict, model_kwargs: Optional[dict] = None) -> np.ndarray:
        kwargs = {} if model_kwargs is None else dict(model_kwargs)
        kwargs["frequency"] = self.energy_centers_hz
        if isinstance(model, str):
            from redback.model_library import all_models_dict
            model = all_models_dict[model]
        try:
            import inspect
            param_names = list(inspect.signature(model).parameters.keys())
        except Exception:
            param_names = []

        if "energies_keV" in param_names or "energy_keV" in param_names:
            kwargs.pop("frequency", None)
            model_flux_mjy = model(self.energy_centers_keV, **parameters, **kwargs)
        else:
            try:
                model_flux_mjy = model(np.zeros_like(self.energy_centers_hz), **parameters, **kwargs)
            except TypeError:
                kwargs.pop("frequency", None)
                model_flux_mjy = model(self.energy_centers_keV, **parameters, **kwargs)
        return fold_spectrum(
            model_flux_mjy=model_flux_mjy,
            energy_edges_keV=self.energy_edges_keV,
            rmf=self.rmf,
            arf=self.arf,
            exposure=self.exposure,
            areascal=self.areascal,
        )

    def compute_band_flux(
        self,
        model,
        parameters: dict,
        energy_min_keV: float,
        energy_max_keV: float,
        model_kwargs: Optional[dict] = None,
        unabsorbed: bool = False,
        n_grid: int = 400,
    ) -> float:
        """
        Compute band-integrated energy flux in erg/s/cm^2 for the given model.

        :param model: Spectral model (callable or model name) returning flux density in mJy.
        :param parameters: Model parameters.
        :param energy_min_keV: Minimum energy in keV.
        :param energy_max_keV: Maximum energy in keV.
        :param model_kwargs: Optional model kwargs.
        :param unabsorbed: If True, set absorption parameters (nh/lognh) to zero.
        :return: Band-integrated energy flux in erg/s/cm^2.
        """
        if energy_min_keV >= energy_max_keV:
            raise ValueError("energy_min_keV must be < energy_max_keV")

        params = dict(parameters)
        if unabsorbed:
            if "nh" in params:
                params["nh"] = 0.0
            if "lognh" in params:
                params["lognh"] = -np.inf

        if isinstance(model, str):
            from redback.model_library import all_models_dict
            model = all_models_dict[model]

        energies = np.logspace(
            np.log10(energy_min_keV),
            np.log10(energy_max_keV),
            max(n_grid, 10),
        )
        kwargs = {} if model_kwargs is None else dict(model_kwargs)
        kwargs["frequency"] = energies * 2.417989e17
        try:
            import inspect
            param_names = list(inspect.signature(model).parameters.keys())
        except Exception:
            param_names = []

        if "energies_keV" in param_names or "energy_keV" in param_names:
            kwargs.pop("frequency", None)
            mjy = model(energies, **params, **kwargs)
        else:
            try:
                mjy = model(np.zeros_like(energies), **params, **kwargs)
            except TypeError:
                kwargs.pop("frequency", None)
                mjy = model(energies, **params, **kwargs)

        flux_density = mjy_to_energy_flux_per_keV(mjy)
        return float(np.trapz(flux_density, energies))

    def plot_spectrum_data(
        self,
        axes=None,
        filename: Optional[str] = None,
        outdir: Optional[str] = None,
        save: bool = True,
        show: bool = True,
        xscale: str = "log",
        yscale: str = "log",
        rate: bool = True,
        density: bool = True,
        plot_background: bool = True,
        subtract_background: bool = False,
        background_scale: bool = True,
        min_counts: Optional[int] = None,
        color: str = "k",
        marker: str = "o",
        markersize: float = 4.0,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        annotate_min_counts: bool = True,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        close: bool = True,
    ):
        centers, widths = self._get_plot_axis()
        if energy_min is not None or energy_max is not None:
            self.active_energy_min = energy_min if energy_min is not None else self.active_energy_min
            self.active_energy_max = energy_max if energy_max is not None else self.active_energy_max
        mask = self.mask_valid()
        counts = self.counts[mask]
        w = widths[mask]
        x = centers[mask]
        raw_x = x
        raw_w = w

        if energy_min is None or energy_max is None:
            if self.rmf is not None:
                qual = self.quality if self.quality is not None else np.zeros_like(self.counts, dtype=int)
                qual_mask = np.asarray(qual) == 0
                if np.any(qual_mask):
                    emin_vals = self.rmf.emin_chan[qual_mask]
                    emax_vals = self.rmf.emax_chan[qual_mask]
                    if energy_min is None:
                        energy_min = float(np.nanmin(emin_vals))
                    if energy_max is None:
                        energy_max = float(np.nanmax(emax_vals))
        logger.info(
            "Plotting spectrum (counts=%d, min_counts=%s, energy range=%.3f-%.3f keV, bg=%s, subtract_bg=%s)",
            len(self.counts), str(min_counts), float(energy_min) if energy_min is not None else -1.0,
            float(energy_max) if energy_max is not None else -1.0, str(plot_background), str(subtract_background)
        )

        if energy_min is not None or energy_max is not None:
            emin = -np.inf if energy_min is None else energy_min
            emax = np.inf if energy_max is None else energy_max
            energy_mask = (x >= emin) & (x <= emax)
            counts = counts[energy_mask]
            w = w[energy_mask]
            x = x[energy_mask]
            raw_x = raw_x[energy_mask]
            raw_w = raw_w[energy_mask]

        counts, x, w, groups = self._compute_grouping(counts, x, w, min_counts)

        xerr = 0.5 * w
        if rate and density:
            y = counts / self.exposure / w
            yerr = np.sqrt(np.maximum(counts, 0.0)) / self.exposure / w
            ylabel = "Counts/s/keV"
        elif rate:
            y = counts / self.exposure
            yerr = np.sqrt(np.maximum(counts, 0.0)) / self.exposure
            ylabel = "Counts/s"
        elif density:
            y = counts / w
            yerr = np.sqrt(np.maximum(counts, 0.0)) / w
            ylabel = "Counts/keV"
        else:
            y = counts
            yerr = np.sqrt(np.maximum(counts, 0.0))
            ylabel = "Counts"

        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes
        label = "total" if plot_background else "data"

        if plot_background and self.counts_bkg is not None:
            bkg_counts = self.counts_bkg[mask]
            if energy_min is not None or energy_max is not None:
                bkg_counts = bkg_counts[energy_mask]
            # Apply the same grouping indices as data
            bkg_counts_g = np.array([np.sum(bkg_counts[grp]) for grp in groups], dtype=float)
            bx = x
            bw = w
            bkg_counts = bkg_counts_g
            if background_scale:
                bkg_counts = bkg_counts * self.background_scale_factor
            if rate and density:
                bkg = bkg_counts / self.exposure / bw
                bkg_err = np.sqrt(np.maximum(bkg_counts, 0.0)) / self.exposure / bw
            elif rate:
                bkg = bkg_counts / self.exposure
                bkg_err = np.sqrt(np.maximum(bkg_counts, 0.0)) / self.exposure
            elif density:
                bkg = bkg_counts / bw
                bkg_err = np.sqrt(np.maximum(bkg_counts, 0.0)) / bw
            else:
                bkg = bkg_counts
                bkg_err = np.sqrt(np.maximum(bkg_counts, 0.0))
            if subtract_background and len(bkg) == len(y):
                y = y - bkg
                yerr = np.sqrt(np.maximum(yerr, 0.0) ** 2 + np.maximum(bkg_err, 0.0) ** 2)
                label = "source"
            else:
                ax.errorbar(
                    bx, bkg, yerr=bkg_err, xerr=0.5 * bw, fmt=marker, markersize=markersize,
                    color="0.5", elinewidth=1.0, capsize=2, label="background"
                )

        if yscale == "log":
            pos = y > 0
            x = x[pos]
            xerr = xerr[pos]
            y = y[pos]
            yerr = yerr[pos]

        ax.errorbar(
            x, y, yerr=yerr, xerr=xerr, fmt=marker, markersize=markersize,
            color=color, elinewidth=1.0, capsize=2, label=label
        )

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(ylabel)
        ax.legend()

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if annotate_min_counts and min_counts is not None:
            text = f"min counts/bin: {min_counts}"
            ax.text(
                0.02, 0.98, text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none")
            )

        if save and filename is not None:
            path = filename if outdir is None else f"{outdir}/{filename}"
            plt.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        elif axes is None and close:
            plt.close(fig)
        return ax

    def plot_spectrum_fit(
        self,
        model,
        posterior=None,
        parameters: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        random_models: int = 100,
        axes=None,
        filename: Optional[str] = None,
        outdir: Optional[str] = None,
        save: bool = True,
        show: bool = True,
        uncertainty_mode: str = "credible_intervals",
        credible_interval_level: float = 0.68,
        plot_max_likelihood: bool = True,
        max_likelihood_color: str = "tab:red",
        uncertainty_band_alpha: float = 0.25,
        **kwargs,
    ):
        if isinstance(model, str):
            from redback.model_library import all_models_dict
            model = all_models_dict[model]
        ax = self.plot_spectrum_data(axes=axes, save=False, show=False, close=False, **kwargs)

        if parameters is None and posterior is not None:
            if "log_likelihood" in posterior:
                parameters = posterior.loc[posterior["log_likelihood"].idxmax()].to_dict()
            else:
                parameters = posterior.median().to_dict()

        if posterior is not None and uncertainty_mode != "none":
            sample = posterior.sample(n=min(random_models, len(posterior)))
            centers, widths = self._get_plot_axis()
            mask = self.mask_valid()
            base_counts = self.counts[mask]
            base_w = widths[mask]
            base_x = centers[mask]
            _, x, w, groups = self._compute_grouping(base_counts, base_x, base_w, kwargs.get("min_counts", None))
            spectra = []
            for _, row in sample.iterrows():
                model_counts = self.predict_counts(model=model, parameters=row.to_dict(), model_kwargs=model_kwargs)
                y = model_counts[mask]
                y = self._apply_group_indices(y, groups)
                if kwargs.get("rate", True) and kwargs.get("density", True):
                    y = y / self.exposure / w
                elif kwargs.get("rate", True):
                    y = y / self.exposure
                elif kwargs.get("density", True):
                    y = y / w
                spectra.append(y)

            if uncertainty_mode == "random_models":
                for y in spectra:
                    ax.plot(x, y, color="tab:blue", alpha=0.1)
            elif uncertainty_mode == "credible_intervals":
                from redback.utils import calc_credible_intervals
                lower, upper, _ = calc_credible_intervals(samples=spectra, interval=credible_interval_level)
                ax.fill_between(x, lower, upper, alpha=uncertainty_band_alpha, color="tab:blue")

        if plot_max_likelihood and parameters is not None:
            model_counts = self.predict_counts(model=model, parameters=parameters, model_kwargs=model_kwargs)
            centers, widths = self._get_plot_axis()
            mask = self.mask_valid()
            base_counts = self.counts[mask]
            base_w = widths[mask]
            base_x = centers[mask]
            _, x, w, groups = self._compute_grouping(base_counts, base_x, base_w, kwargs.get("min_counts", None))
            y = model_counts[mask]
            y = self._apply_group_indices(y, groups)
            if kwargs.get("rate", True) and kwargs.get("density", True):
                y = y / self.exposure / w
            elif kwargs.get("rate", True):
                y = y / self.exposure
            elif kwargs.get("density", True):
                y = y / w
            ax.plot(x, y, color=max_likelihood_color, linewidth=2, label="max likelihood")
            ax.legend()

        if save and filename is not None:
            path = filename if outdir is None else f"{outdir}/{filename}"
            plt.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        elif axes is None:
            plt.close(ax.figure)
        return ax

    @staticmethod
    def plot_lightcurve(
        time: np.ndarray = None,
        rate: np.ndarray = None,
        error: np.ndarray = None,
        lc=None,
        time_bins: np.ndarray = None,
        axes=None,
        filename: Optional[str] = None,
        outdir: Optional[str] = None,
        save: bool = True,
        show: bool = True,
        xscale: str = "linear",
        yscale: str = "linear",
        color: str = "tab:blue",
        marker: str = "o",
        markersize: float = 4.0,
        min_counts: int = None,
        annotate_min_counts: bool = True,
    ):
        """
        Plot a count-rate lightcurve. Accepts either:
        - lc: OGIPLightCurve object with time/rate/error
        - time/rate/error arrays
        If time_bins is provided, converts rate->counts for ThreeML-style plotting.
        """
        from redback.plotting import plot_binned_count_lightcurve

        if lc is not None:
            time = lc.time
            rate = lc.rate
            error = lc.error

        if time is None or rate is None:
            raise ValueError("Provide lc or time+rate arrays")

        if time_bins is None and lc is not None and lc.timedel is not None:
            dt = lc.timedel
            time_bins = np.concatenate([[time[0] - 0.5 * dt], time + 0.5 * dt])
        elif time_bins is None and len(time) > 1:
            dt = time[1] - time[0]
            time_bins = np.concatenate([[time[0] - 0.5 * dt], time + 0.5 * dt])

        if time_bins is not None:
            dt = time_bins[1:] - time_bins[:-1]
            scale = 1.0
            if lc is not None and lc.fracexp is not None:
                scale = lc.fracexp
            counts = rate * dt * scale
            rate_err = error * scale if error is not None else None
            logger.info("Plotting lightcurve with %d bins (min_counts=%s)", len(dt), str(min_counts))
            return plot_binned_count_lightcurve(
                time_bins=time_bins,
                counts=counts,
                rate=rate,
                error=rate_err,
                axes=axes,
                filename=filename,
                outdir=outdir,
                save=save,
                show=show,
                xscale=xscale,
                yscale=yscale,
                color=color,
                marker=marker,
                markersize=markersize,
                min_counts=min_counts,
                annotate_min_counts=annotate_min_counts,
            )

        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes
        ax.errorbar(
            time, rate, yerr=error, fmt=marker, markersize=markersize,
            color=color, elinewidth=1.0, capsize=2, label="count rate"
        )
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Counts/s")
        ax.legend()

        if save and filename is not None:
            path = filename if outdir is None else f"{outdir}/{filename}"
            plt.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        elif axes is None:
            plt.close(fig)
        return ax

    # Backwards-compatible aliases
    def plot_data(self, *args, **kwargs):
        return self.plot_spectrum_data(*args, **kwargs)

    def plot_fit(self, *args, **kwargs):
        return self.plot_spectrum_fit(*args, **kwargs)

    @classmethod
    def from_ogip(
        cls,
        pha: str,
        rmf: Optional[str] = None,
        arf: Optional[str] = None,
        bkg: Optional[str] = None,
        spectrum_index: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "SpectralDataset":
        import os

        pha_spec = read_pha(pha, spectrum_index=spectrum_index)
        base_dir = os.path.dirname(pha)

        if bkg is None and pha_spec.backfile and pha_spec.backfile.lower() != "none":
            bkg = os.path.join(base_dir, pha_spec.backfile)
        if rmf is None and pha_spec.respfile and pha_spec.respfile.lower() != "none":
            rmf = os.path.join(base_dir, pha_spec.respfile)
        if arf is None and pha_spec.ancrfile and pha_spec.ancrfile.lower() != "none":
            arf = os.path.join(base_dir, pha_spec.ancrfile)

        bkg_counts = None
        bkg_exposure = None
        bkg_backscale = None
        bkg_areascal = None
        if bkg:
            bkg_spec = read_pha(bkg, spectrum_index=spectrum_index)
            bkg_counts = bkg_spec.counts
            bkg_exposure = bkg_spec.exposure
            bkg_backscale = bkg_spec.backscale
            bkg_areascal = bkg_spec.areascal

        rmf_obj = read_rmf(rmf) if rmf is not None else None
        arf_obj = read_arf(arf) if arf is not None else None

        energy_edges = None
        if rmf_obj is not None:
            energy_edges = np.concatenate([rmf_obj.e_min, [rmf_obj.e_max[-1]]])
        elif arf_obj is not None:
            energy_edges = np.concatenate([arf_obj.e_min, [arf_obj.e_max[-1]]])
        else:
            raise ValueError("At least one of RMF or ARF must be provided to determine energy edges")

        return cls(
            counts=pha_spec.counts,
            counts_bkg=bkg_counts,
            bkg_exposure=bkg_exposure,
            bkg_backscale=bkg_backscale,
            bkg_areascal=bkg_areascal,
            exposure=pha_spec.exposure,
            backscale=pha_spec.backscale,
            areascal=pha_spec.areascal,
            energy_edges_keV=energy_edges,
            rmf=rmf_obj,
            arf=arf_obj,
            quality=pha_spec.quality,
            grouping=pha_spec.grouping,
            name=name or "spectral_dataset",
        )

    @classmethod
    def from_ogip_directory(
        cls,
        directory: str,
        pha: Optional[str] = None,
        bkg: Optional[str] = None,
        rmf: Optional[str] = None,
        arf: Optional[str] = None,
        spectrum_index: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "SpectralDataset":
        """
        Convenience loader for standard OGIP directories.

        If file names are not provided, the first matching file is used.
        """
        import os

        def _first_match(ext):
            matches = [f for f in os.listdir(directory) if f.lower().endswith(ext)]
            return os.path.join(directory, matches[0]) if matches else None

        pha_path = pha or _first_match(".pha")
        if pha_path is None:
            raise FileNotFoundError(f"No .pha file found in {directory}")

        rmf_path = rmf or _first_match(".rmf")
        arf_path = arf or _first_match(".arf")
        bkg_path = bkg or _first_match("bk.pha")
        if bkg_path is None:
            # fallback: any other .pha not matching the source pha
            candidates = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(".pha") and os.path.join(directory, f) != pha_path
            ]
            bkg_path = candidates[0] if candidates else None

        return cls.from_ogip(
            pha=pha_path,
            rmf=rmf_path,
            arf=arf_path,
            bkg=bkg_path,
            spectrum_index=spectrum_index,
            name=name,
        )

    @classmethod
    def from_simulator(
        cls,
        sim,
        time_bins: np.ndarray,
    ) -> "SpectralDataset":
        """
        Build a SpectralDataset from SimulateHighEnergyTransient outputs.

        :param sim: SimulateHighEnergyTransient instance (or compatible object).
        :param time_bins: Time bin edges in seconds. Counts are summed over these bins.
        """
        binned = sim.generate_binned_counts(time_bins=time_bins, energy_integrated=False)
        counts = np.zeros(sim.n_energy_bins, dtype=float)

        grouped = binned.groupby("energy_channel")["counts"].sum()
        for ch_idx, val in grouped.items():
            counts[int(ch_idx)] = float(val)

        total_exposure = float(np.sum(np.diff(np.asarray(time_bins, dtype=float))))
        energy_edges = np.asarray(sim.energy_edges, dtype=float)
        energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
        widths = energy_edges[1:] - energy_edges[:-1]

        eff_area = sim.effective_area_func(energy_centers)
        bkg_rate = sim.background_rate_func(energy_centers)
        counts_bkg = bkg_rate * eff_area * widths * total_exposure

        arf = EffectiveArea(
            e_min=energy_edges[:-1],
            e_max=energy_edges[1:],
            area=np.asarray(eff_area, dtype=float),
        )

        return cls(
            counts=counts,
            counts_bkg=np.asarray(counts_bkg, dtype=float),
            bkg_exposure=total_exposure,
            bkg_backscale=1.0,
            bkg_areascal=1.0,
            exposure=total_exposure,
            backscale=1.0,
            areascal=1.0,
            energy_edges_keV=energy_edges,
            rmf=None,
            arf=arf,
            quality=None,
            grouping=None,
        )


def read_rmf(path: str) -> ResponseMatrix:
    from redback.spectral.io import read_rmf as _read_rmf
    return _read_rmf(path)


def read_arf(path: str) -> EffectiveArea:
    from redback.spectral.io import read_arf as _read_arf
    return _read_arf(path)
