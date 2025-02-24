from __future__ import annotations

from os.path import join
from typing import Any, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback
from redback.utils import KwargsAccessorWithDefault

class _FilenameGetter(object):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def __get__(self, instance: Plotter, owner: object) -> str:
        return instance.get_filename(default=f"{instance.transient.name}_{self.suffix}.png")

    def __set__(self, instance: Plotter, value: object) -> None:
        pass


class _FilePathGetter(object):

    def __init__(self, directory_property: str, filename_property: str) -> None:
        self.directory_property = directory_property
        self.filename_property = filename_property

    def __get__(self, instance: Plotter, owner: object) -> str:
        return join(getattr(instance, self.directory_property), getattr(instance, self.filename_property))


class Plotter(object):
    """
    Base class for all lightcurve plotting classes in redback.
    """

    capsize = KwargsAccessorWithDefault("capsize", 0.)
    legend_location = KwargsAccessorWithDefault("legend_location", "best")
    legend_cols = KwargsAccessorWithDefault("legend_cols", 2)
    band_colors = KwargsAccessorWithDefault("band_colors", None)
    color = KwargsAccessorWithDefault("color", "k")
    band_labels = KwargsAccessorWithDefault("band_labels", None)
    band_scaling = KwargsAccessorWithDefault("band_scaling", {})
    dpi = KwargsAccessorWithDefault("dpi", 300)
    elinewidth = KwargsAccessorWithDefault("elinewidth", 2)
    errorbar_fmt = KwargsAccessorWithDefault("errorbar_fmt", "x")
    model = KwargsAccessorWithDefault("model", None)
    ms = KwargsAccessorWithDefault("ms", 1)
    axis_tick_params_pad = KwargsAccessorWithDefault("axis_tick_params_pad", 10)

    max_likelihood_alpha = KwargsAccessorWithDefault("max_likelihood_alpha", 0.65)
    random_sample_alpha = KwargsAccessorWithDefault("random_sample_alpha", 0.05)
    uncertainty_band_alpha = KwargsAccessorWithDefault("uncertainty_band_alpha", 0.4)
    max_likelihood_color = KwargsAccessorWithDefault("max_likelihood_color", "blue")
    random_sample_color = KwargsAccessorWithDefault("random_sample_color", "red")

    bbox_inches = KwargsAccessorWithDefault("bbox_inches", "tight")
    linewidth = KwargsAccessorWithDefault("linewidth", 2)
    zorder = KwargsAccessorWithDefault("zorder", -1)

    xy = KwargsAccessorWithDefault("xy", (0.95, 0.9))
    xycoords = KwargsAccessorWithDefault("xycoords", "axes fraction")
    horizontalalignment = KwargsAccessorWithDefault("horizontalalignment", "right")
    annotation_size = KwargsAccessorWithDefault("annotation_size", 20)

    fontsize_axes = KwargsAccessorWithDefault("fontsize_axes", 18)
    fontsize_figure = KwargsAccessorWithDefault("fontsize_figure", 30)
    fontsize_legend = KwargsAccessorWithDefault("fontsize_legend", 18)
    fontsize_ticks = KwargsAccessorWithDefault("fontsize_ticks", 16)
    hspace = KwargsAccessorWithDefault("hspace", 0.04)
    wspace = KwargsAccessorWithDefault("wspace", 0.15)

    plot_others = KwargsAccessorWithDefault("plot_others", True)
    random_models = KwargsAccessorWithDefault("random_models", 100)
    uncertainty_mode = KwargsAccessorWithDefault("uncertainty_mode", "random_models")
    credible_interval_level = KwargsAccessorWithDefault("credible_interval_level", 0.9)
    plot_max_likelihood = KwargsAccessorWithDefault("plot_max_likelihood", True)
    set_same_color_per_subplot = KwargsAccessorWithDefault("set_same_color_per_subplot", True)

    xlim_high_multiplier = KwargsAccessorWithDefault("xlim_high_multiplier", 2.0)
    xlim_low_multiplier = KwargsAccessorWithDefault("xlim_low_multiplier", 0.5)
    ylim_high_multiplier = KwargsAccessorWithDefault("ylim_high_multiplier", 2.0)
    ylim_low_multiplier = KwargsAccessorWithDefault("ylim_low_multiplier", 0.5)

    def __init__(self, transient: Union[redback.transient.Transient, None], **kwargs) -> None:
        """
        :param transient: An instance of `redback.transient.Transient`. Contains the data to be plotted.
        :param kwargs: Additional kwargs the plotter uses. -------
        :keyword capsize: Same as matplotlib capsize.
        :keyword bands_to_plot: List of bands to plot in plot lightcurve and multiband lightcurve. Default is active bands.
        :keyword legend_location: Same as matplotlib legend location.
        :keyword legend_cols: Same as matplotlib legend columns.
        :keyword color: Color of the data points.
        :keyword band_colors: A dictionary with the colors of the bands.
        :keyword band_labels: List with the names of the bands.
        :keyword band_scaling: Dict with the scaling for each band. First entry should be {type: '+' or 'x'} for different types.
        :keyword dpi: Same as matplotlib dpi.
        :keyword elinewidth: same as matplotlib elinewidth
        :keyword errorbar_fmt: 'fmt' argument of `ax.errorbar`.
        :keyword model: str or callable, the model to plot.
        :keyword ms: Same as matplotlib markersize.
        :keyword axis_tick_params_pad: `pad` argument in calls to `ax.tick_params` when setting the axes.
        :keyword max_likelihood_alpha: `alpha` argument, i.e. transparency, when plotting the max likelihood curve.
        :keyword random_sample_alpha: `alpha` argument, i.e. transparency, when plotting random sample curves.
        :keyword uncertainty_band_alpha: `alpha` argument, i.e. transparency, when plotting a credible band.
        :keyword max_likelihood_color: Color of the maximum likelihood curve.
        :keyword random_sample_color: Color of the random sample curves.
        :keyword bbox_inches: Setting for saving plots. Default is 'tight'.
        :keyword linewidth: Same as matplotlib linewidth
        :keyword zorder: Same as matplotlib zorder
        :keyword xy: For `ax.annotate' x and y coordinates of the point to annotate.
        :keyword xycoords: The coordinate system `xy` is given in. Default is 'axes fraction'
        :keyword horizontalalignment: Horizontal alignment of the annotation. Default is 'right'
        :keyword annotation_size: `size` argument of of `ax.annotate`.
        :keyword fontsize_axes: Font size of the x and y labels.
        :keyword fontsize_legend: Font size of the legend.
        :keyword fontsize_figure: Font size of the figure. Relevant for multiband plots.
                                  Used on `supxlabel` and `supylabel`.
        :keyword fontsize_ticks: Font size of the axis ticks.
        :keyword hspace: Argument for `subplots_adjust`, sets horizontal spacing between panels.
        :keyword wspace: Argument for `subplots_adjust`, sets horizontal spacing between panels.
        :keyword plot_others: Whether to plot additional bands in the data plot, all in the same colors
        :keyword random_models: Number of random draws to use to calculate credible bands or to plot.
        :keyword uncertainty_mode: 'random_models': Plot random draws from the available parameter sets.
                                   'credible_intervals': Plot a credible interval that is calculated based
                                   on the available parameter sets.
        :keyword reference_mjd_date: Date to use as reference point for the x axis.
                                    Default is the first date in the data.
        :keyword credible_interval_level: 0.9: Plot the 90% credible interval.
        :keyword plot_max_likelihood: Plots the draw corresponding to the maximum likelihood. Default is 'True'.
        :keyword set_same_color_per_subplot: Sets the lightcurve to be the same color as the data per subplot. Default is 'True'.
        :keyword xlim_high_multiplier: Adjust the maximum xlim based on available x values.
        :keyword xlim_low_multiplier: Adjust the minimum xlim based on available x values.
        :keyword ylim_high_multiplier: Adjust the maximum ylim based on available x values.
        :keyword ylim_low_multiplier: Adjust the minimum ylim based on available x values.
        """
        self.transient = transient
        self.kwargs = kwargs or dict()
        self._posterior_sorted = False

    keyword_docstring = __init__.__doc__.split("-------")[1]

    def _get_times(self, axes: matplotlib.axes.Axes) -> np.ndarray:
        """
        :param axes: The axes used in the plotting procedure.
        :type axes: matplotlib.axes.Axes

        :return: Linearly or logarithmically scaled time values depending on the y scale used in the plot.
        :rtype: np.ndarray
        """
        if isinstance(axes, np.ndarray):
            ax = axes[0]
        else:
            ax = axes

        if ax.get_yscale() == 'linear':
            times = np.linspace(self._xlim_low, self._xlim_high, 200)
        else:
            times = np.exp(np.linspace(np.log(self._xlim_low), np.log(self._xlim_high), 200))

        if self.transient.use_phase_model:
            times = times + self._reference_mjd_date
        return times

    @property
    def _xlim_low(self) -> float:
        default = self.xlim_low_multiplier * self.transient.x[0]
        if default == 0:
            default += 1e-3
        return self.kwargs.get("xlim_low", default)

    @property
    def _xlim_high(self) -> float:
        if self._x_err is None:
            default = self.xlim_high_multiplier * self.transient.x[-1]
        else:
            default = self.xlim_high_multiplier * (self.transient.x[-1] + self._x_err[1][-1])
        return self.kwargs.get("xlim_high", default)

    @property
    def _ylim_low(self) -> float:
        default = self.ylim_low_multiplier * min(self.transient.y)
        return self.kwargs.get("ylim_low", default)

    @property
    def _ylim_high(self) -> float:
        default = self.ylim_high_multiplier * np.max(self.transient.y)
        return self.kwargs.get("ylim_high", default)

    @property
    def _x_err(self) -> Union[np.ndarray, None]:
        if self.transient.x_err is not None:
            return np.array([np.abs(self.transient.x_err[1, :]), self.transient.x_err[0, :]])
        else:
            return None

    @property
    def _y_err(self) -> np.ndarray:
        if self.transient.y_err.ndim > 1.:
            return np.array([np.abs(self.transient.y_err[1, :]), self.transient.y_err[0, :]])
        else:
            return np.array([np.abs(self.transient.y_err)])
    @property
    def _lightcurve_plot_outdir(self) -> str:
        return self._get_outdir(join(self.transient.directory_structure.directory_path, self.model.__name__))

    @property
    def _data_plot_outdir(self) -> str:
        return self._get_outdir(self.transient.directory_structure.directory_path)

    def _get_outdir(self, default: str) -> str:
        return self._get_kwarg_with_default(kwarg="outdir", default=default)

    def get_filename(self, default: str) -> str:
        return self._get_kwarg_with_default(kwarg="filename", default=default)

    def _get_kwarg_with_default(self, kwarg: str, default: Any) -> Any:
        return self.kwargs.get(kwarg, default) or default

    @property
    def _model_kwargs(self) -> dict:
        return self._get_kwarg_with_default("model_kwargs", dict())

    @property
    def _posterior(self) -> pd.DataFrame:
        posterior = self.kwargs.get("posterior", pd.DataFrame())
        if not self._posterior_sorted and posterior is not None:
            posterior.sort_values(by='log_likelihood', inplace=True)
            self._posterior_sorted = True
        return posterior

    @property
    def _max_like_params(self) -> pd.core.series.Series:
        return self._posterior.iloc[-1]

    def _get_random_parameters(self) -> list[pd.core.series.Series]:
        integers = np.arange(len(self._posterior))
        indices = np.random.choice(integers, size=self.random_models)
        return [self._posterior.iloc[idx] for idx in indices]

    _data_plot_filename = _FilenameGetter(suffix="data")
    _lightcurve_plot_filename = _FilenameGetter(suffix="lightcurve")
    _residual_plot_filename = _FilenameGetter(suffix="residual")
    _multiband_data_plot_filename = _FilenameGetter(suffix="multiband_data")
    _multiband_lightcurve_plot_filename = _FilenameGetter(suffix="multiband_lightcurve")

    _data_plot_filepath = _FilePathGetter(
        directory_property="_data_plot_outdir", filename_property="_data_plot_filename")
    _lightcurve_plot_filepath = _FilePathGetter(
        directory_property="_lightcurve_plot_outdir", filename_property="_lightcurve_plot_filename")
    _residual_plot_filepath = _FilePathGetter(
        directory_property="_lightcurve_plot_outdir", filename_property="_residual_plot_filename")
    _multiband_data_plot_filepath = _FilePathGetter(
        directory_property="_data_plot_outdir", filename_property="_multiband_data_plot_filename")
    _multiband_lightcurve_plot_filepath = _FilePathGetter(
        directory_property="_lightcurve_plot_outdir", filename_property="_multiband_lightcurve_plot_filename")

    def _save_and_show(self, filepath: str, save: bool, show: bool) -> None:
        plt.tight_layout()
        if save:
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches, transparent=False, facecolor='white')
        if show:
            plt.show()

class SpecPlotter(object):
    """
    Base class for all lightcurve plotting classes in redback.
    """

    capsize = KwargsAccessorWithDefault("capsize", 0.)
    elinewidth = KwargsAccessorWithDefault("elinewidth", 2)
    errorbar_fmt = KwargsAccessorWithDefault("errorbar_fmt", "x")
    legend_location = KwargsAccessorWithDefault("legend_location", "best")
    legend_cols = KwargsAccessorWithDefault("legend_cols", 2)
    color = KwargsAccessorWithDefault("color", "k")
    dpi = KwargsAccessorWithDefault("dpi", 300)
    model = KwargsAccessorWithDefault("model", None)
    ms = KwargsAccessorWithDefault("ms", 1)
    axis_tick_params_pad = KwargsAccessorWithDefault("axis_tick_params_pad", 10)

    max_likelihood_alpha = KwargsAccessorWithDefault("max_likelihood_alpha", 0.65)
    random_sample_alpha = KwargsAccessorWithDefault("random_sample_alpha", 0.05)
    uncertainty_band_alpha = KwargsAccessorWithDefault("uncertainty_band_alpha", 0.4)
    max_likelihood_color = KwargsAccessorWithDefault("max_likelihood_color", "blue")
    random_sample_color = KwargsAccessorWithDefault("random_sample_color", "red")

    bbox_inches = KwargsAccessorWithDefault("bbox_inches", "tight")
    linewidth = KwargsAccessorWithDefault("linewidth", 2)
    zorder = KwargsAccessorWithDefault("zorder", -1)
    yscale = KwargsAccessorWithDefault("yscale", "linear")

    xy = KwargsAccessorWithDefault("xy", (0.95, 0.9))
    xycoords = KwargsAccessorWithDefault("xycoords", "axes fraction")
    horizontalalignment = KwargsAccessorWithDefault("horizontalalignment", "right")
    annotation_size = KwargsAccessorWithDefault("annotation_size", 20)

    fontsize_axes = KwargsAccessorWithDefault("fontsize_axes", 18)
    fontsize_figure = KwargsAccessorWithDefault("fontsize_figure", 30)
    fontsize_legend = KwargsAccessorWithDefault("fontsize_legend", 18)
    fontsize_ticks = KwargsAccessorWithDefault("fontsize_ticks", 16)
    hspace = KwargsAccessorWithDefault("hspace", 0.04)
    wspace = KwargsAccessorWithDefault("wspace", 0.15)

    random_models = KwargsAccessorWithDefault("random_models", 100)
    uncertainty_mode = KwargsAccessorWithDefault("uncertainty_mode", "random_models")
    credible_interval_level = KwargsAccessorWithDefault("credible_interval_level", 0.9)
    plot_max_likelihood = KwargsAccessorWithDefault("plot_max_likelihood", True)
    set_same_color_per_subplot = KwargsAccessorWithDefault("set_same_color_per_subplot", True)

    xlim_high_multiplier = KwargsAccessorWithDefault("xlim_high_multiplier", 1.05)
    xlim_low_multiplier = KwargsAccessorWithDefault("xlim_low_multiplier", 0.9)
    ylim_high_multiplier = KwargsAccessorWithDefault("ylim_high_multiplier", 1.1)
    ylim_low_multiplier = KwargsAccessorWithDefault("ylim_low_multiplier", 0.5)

    def __init__(self, spectrum: Union[redback.transient.Spectrum, None], **kwargs) -> None:
        """
        :param spectrum: An instance of `redback.transient.Spectrum`. Contains the data to be plotted.
        :param kwargs: Additional kwargs the plotter uses. -------
        :keyword capsize: Same as matplotlib capsize.
        :keyword elinewidth: same as matplotlib elinewidth
        :keyword errorbar_fmt: 'fmt' argument of `ax.errorbar`.
        :keyword ms: Same as matplotlib markersize.
        :keyword legend_location: Same as matplotlib legend location.
        :keyword legend_cols: Same as matplotlib legend columns.
        :keyword color: Color of the data points.
        :keyword dpi: Same as matplotlib dpi.
        :keyword model: str or callable, the model to plot.
        :keyword ms: Same as matplotlib markersize.
        :keyword axis_tick_params_pad: `pad` argument in calls to `ax.tick_params` when setting the axes.
        :keyword max_likelihood_alpha: `alpha` argument, i.e. transparency, when plotting the max likelihood curve.
        :keyword random_sample_alpha: `alpha` argument, i.e. transparency, when plotting random sample curves.
        :keyword uncertainty_band_alpha: `alpha` argument, i.e. transparency, when plotting a credible band.
        :keyword max_likelihood_color: Color of the maximum likelihood curve.
        :keyword random_sample_color: Color of the random sample curves.
        :keyword bbox_inches: Setting for saving plots. Default is 'tight'.
        :keyword linewidth: Same as matplotlib linewidth
        :keyword zorder: Same as matplotlib zorder
        :keyword yscale: Same as matplotlib yscale, default is linear
        :keyword xy: For `ax.annotate' x and y coordinates of the point to annotate.
        :keyword xycoords: The coordinate system `xy` is given in. Default is 'axes fraction'
        :keyword horizontalalignment: Horizontal alignment of the annotation. Default is 'right'
        :keyword annotation_size: `size` argument of of `ax.annotate`.
        :keyword fontsize_axes: Font size of the x and y labels.
        :keyword fontsize_legend: Font size of the legend.
        :keyword fontsize_figure: Font size of the figure. Relevant for multiband plots.
                                  Used on `supxlabel` and `supylabel`.
        :keyword fontsize_ticks: Font size of the axis ticks.
        :keyword hspace: Argument for `subplots_adjust`, sets horizontal spacing between panels.
        :keyword wspace: Argument for `subplots_adjust`, sets horizontal spacing between panels.
        :keyword plot_others: Whether to plot additional bands in the data plot, all in the same colors
        :keyword random_models: Number of random draws to use to calculate credible bands or to plot.
        :keyword uncertainty_mode: 'random_models': Plot random draws from the available parameter sets.
                                   'credible_intervals': Plot a credible interval that is calculated based
                                   on the available parameter sets.
        :keyword credible_interval_level: 0.9: Plot the 90% credible interval.
        :keyword plot_max_likelihood: Plots the draw corresponding to the maximum likelihood. Default is 'True'.
        :keyword set_same_color_per_subplot: Sets the lightcurve to be the same color as the data per subplot. Default is 'True'.
        :keyword xlim_high_multiplier: Adjust the maximum xlim based on available x values.
        :keyword xlim_low_multiplier: Adjust the minimum xlim based on available x values.
        :keyword ylim_high_multiplier: Adjust the maximum ylim based on available x values.
        :keyword ylim_low_multiplier: Adjust the minimum ylim based on available x values.
        """
        self.transient = spectrum
        self.kwargs = kwargs or dict()
        self._posterior_sorted = False

    keyword_docstring = __init__.__doc__.split("-------")[1]

    def _get_angstroms(self, axes: matplotlib.axes.Axes) -> np.ndarray:
        """
        :param axes: The axes used in the plotting procedure.
        :type axes: matplotlib.axes.Axes

        :return: Linearly or logarithmically scaled angtrom values depending on the y scale used in the plot.
        :rtype: np.ndarray
        """
        if isinstance(axes, np.ndarray):
            ax = axes[0]
        else:
            ax = axes

        if ax.get_yscale() == 'linear':
            angstroms = np.linspace(self._xlim_low, self._xlim_high, 200)
        else:
            angstroms = np.exp(np.linspace(np.log(self._xlim_low), np.log(self._xlim_high), 200))

        return angstroms

    @property
    def _xlim_low(self) -> float:
        default = self.xlim_low_multiplier * self.transient.angstroms[0]
        if default == 0:
            default += 1e-3
        return self.kwargs.get("xlim_low", default)

    @property
    def _xlim_high(self) -> float:
        default = self.xlim_high_multiplier * self.transient.angstroms[-1]
        return self.kwargs.get("xlim_high", default)

    @property
    def _ylim_low(self) -> float:
        default = self.ylim_low_multiplier * min(self.transient.flux_density)
        return self.kwargs.get("ylim_low", default/1e-17)

    @property
    def _ylim_high(self) -> float:
        default = self.ylim_high_multiplier * np.max(self.transient.flux_density)
        return self.kwargs.get("ylim_high", default/1e-17)

    @property
    def _y_err(self) -> np.ndarray:
            return np.array([np.abs(self.transient.y_err)])

    @property
    def _data_plot_outdir(self) -> str:
        return self._get_outdir(self.transient.directory_structure.directory_path)

    def _get_outdir(self, default: str) -> str:
        return self._get_kwarg_with_default(kwarg="outdir", default=default)

    def get_filename(self, default: str) -> str:
        return self._get_kwarg_with_default(kwarg="filename", default=default)

    def _get_kwarg_with_default(self, kwarg: str, default: Any) -> Any:
        return self.kwargs.get(kwarg, default) or default

    @property
    def _model_kwargs(self) -> dict:
        return self._get_kwarg_with_default("model_kwargs", dict())

    @property
    def _posterior(self) -> pd.DataFrame:
        posterior = self.kwargs.get("posterior", pd.DataFrame())
        if not self._posterior_sorted and posterior is not None:
            posterior.sort_values(by='log_likelihood', inplace=True)
            self._posterior_sorted = True
        return posterior

    @property
    def _max_like_params(self) -> pd.core.series.Series:
        return self._posterior.iloc[-1]

    def _get_random_parameters(self) -> list[pd.core.series.Series]:
        integers = np.arange(len(self._posterior))
        indices = np.random.choice(integers, size=self.random_models)
        return [self._posterior.iloc[idx] for idx in indices]

    _data_plot_filename = _FilenameGetter(suffix="data")
    _spectrum_ppd_plot_filename = _FilenameGetter(suffix="spectrum_ppd")
    _residual_plot_filename = _FilenameGetter(suffix="residual")

    _data_plot_filepath = _FilePathGetter(
        directory_property="_data_plot_outdir", filename_property="_data_plot_filename")
    _spectrum_ppd_plot_filepath = _FilePathGetter(
        directory_property="_data_plot_outdir", filename_property="_spectrum_ppd_plot_filename")
    _residual_plot_filepath = _FilePathGetter(
        directory_property="_data_plot_outdir", filename_property="_residual_plot_filename")

    def _save_and_show(self, filepath: str, save: bool, show: bool) -> None:
        plt.tight_layout()
        if save:
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches, transparent=False, facecolor='white')
        if show:
            plt.show()


class IntegratedFluxPlotter(Plotter):

    @property
    def _xlabel(self) -> str:
        return r"Time since burst [s]"

    @property
    def _ylabel(self) -> str:
        return self.transient.ylabel

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the Integrated flux data and returns Axes.

        :param axes: Matplotlib axes to plot the data into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        ax = axes or plt.gca()

        ax.errorbar(self.transient.x, self.transient.y, xerr=self._x_err, yerr=self._y_err,
                    fmt=self.errorbar_fmt, c=self.color, ms=self.ms, elinewidth=self.elinewidth, capsize=self.capsize)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(self._xlim_low, self._xlim_high)
        ax.set_ylim(self._ylim_low, self._ylim_high)
        ax.set_xlabel(self._xlabel, fontsize=self.fontsize_axes)
        ax.set_ylabel(self._ylabel, fontsize=self.fontsize_axes)

        ax.annotate(
            self.transient.name, xy=self.xy, xycoords=self.xycoords,
            horizontalalignment=self.horizontalalignment, size=self.annotation_size)

        ax.tick_params(axis='both', which='both', pad=self.axis_tick_params_pad, labelsize=self.fontsize_ticks)

        self._save_and_show(filepath=self._data_plot_filepath, save=save, show=show)
        return ax

    def plot_lightcurve(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the Integrated flux data and the lightcurve and returns Axes.

        :param axes: Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        
        axes = axes or plt.gca()

        axes = self.plot_data(axes=axes, save=False, show=False)
        times = self._get_times(axes)

        self._plot_lightcurves(axes, times)

        self._save_and_show(filepath=self._lightcurve_plot_filepath, save=save, show=show)
        return axes

    def _plot_lightcurves(self, axes: matplotlib.axes.Axes, times: np.ndarray) -> None:
        if self.plot_max_likelihood:
            ys = self.model(times, **self._max_like_params, **self._model_kwargs)
            axes.plot(times, ys, color=self.max_likelihood_color, alpha=self.max_likelihood_alpha, lw=self.linewidth)

        random_ys_list = [self.model(times, **random_params, **self._model_kwargs)
                          for random_params in self._get_random_parameters()]
        if self.uncertainty_mode == "random_models":
            for ys in random_ys_list:
                axes.plot(times, ys, color=self.random_sample_color, alpha=self.random_sample_alpha, lw=self.linewidth,
                          zorder=self.zorder)
        elif self.uncertainty_mode == "credible_intervals":
            lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=random_ys_list, interval=self.credible_interval_level)
            axes.fill_between(
                times, lower_bound, upper_bound, alpha=self.uncertainty_band_alpha, color=self.max_likelihood_color)

    def _plot_single_lightcurve(self, axes: matplotlib.axes.Axes, times: np.ndarray, params: dict) -> None:
        ys = self.model(times, **params, **self._model_kwargs)
        axes.plot(times, ys, color=self.random_sample_color, alpha=self.random_sample_alpha, lw=self.linewidth,
                  zorder=self.zorder)

    def plot_residuals(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the residual of the Integrated flux data returns Axes.

        :param axes: Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        :param save: Whether to save the plot. (Default value = True)
        :param show: Whether to show the plot. (Default value = True)

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        if axes is None:
            fig, axes = plt.subplots(
                nrows=2, ncols=1, sharex=True, sharey=False, figsize=(10, 8), gridspec_kw=dict(height_ratios=[2, 1]))

        axes[0] = self.plot_lightcurve(axes=axes[0], save=False, show=False)
        axes[1].set_xlabel(axes[0].get_xlabel(), fontsize=self.fontsize_axes)
        axes[0].set_xlabel("")
        ys = self.model(self.transient.x, **self._max_like_params, **self._model_kwargs)
        axes[1].errorbar(
            self.transient.x, self.transient.y - ys, xerr=self._x_err, yerr=self._y_err,
            fmt=self.errorbar_fmt, c=self.color, ms=self.ms, elinewidth=self.elinewidth, capsize=self.capsize)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Residual", fontsize=self.fontsize_axes)
        axes[1].tick_params(axis='both', which='both', pad=self.axis_tick_params_pad, labelsize=self.fontsize_ticks)

        self._save_and_show(filepath=self._residual_plot_filepath, save=save, show=show)
        return axes


class LuminosityPlotter(IntegratedFluxPlotter):
    pass


class MagnitudePlotter(Plotter):

    xlim_low_phase_model_multiplier = KwargsAccessorWithDefault("xlim_low_multiplier", 0.9)
    xlim_high_phase_model_multiplier = KwargsAccessorWithDefault("xlim_high_multiplier", 1.1)
    xlim_high_multiplier = KwargsAccessorWithDefault("xlim_high_multiplier", 1.2)
    ylim_low_magnitude_multiplier = KwargsAccessorWithDefault("ylim_low_multiplier", 0.8)
    ylim_high_magnitude_multiplier = KwargsAccessorWithDefault("ylim_high_multiplier", 1.2)
    ncols = KwargsAccessorWithDefault("ncols", 2)

    @property
    def _colors(self) -> str:
        return self.kwargs.get("colors", self.transient.get_colors(self._filters))

    @property
    def _xlabel(self) -> str:
        if self.transient.use_phase_model:
            default = f"Time since {self._reference_mjd_date} MJD [days]"
        else:
            default = self.transient.xlabel
        return self.kwargs.get("xlabel", default)

    @property
    def _ylabel(self) -> str:
        return self.kwargs.get("ylabel", self.transient.ylabel)

    @property
    def _get_bands_to_plot(self) -> list[str]:
        return self.kwargs.get("bands_to_plot", self.transient.active_bands)

    @property
    def _xlim_low(self) -> float:
        if self.transient.use_phase_model:
            default = (self.transient.x[0] - self._reference_mjd_date) * self.xlim_low_phase_model_multiplier
        else:
            default = self.xlim_low_multiplier * self.transient.x[0]
        if default == 0:
            default += 1e-3
        return self.kwargs.get("xlim_low", default)

    @property
    def _xlim_high(self) -> float:
        if self.transient.use_phase_model:
            default = (self.transient.x[-1] - self._reference_mjd_date) * self.xlim_high_phase_model_multiplier
        else:
            default = self.xlim_high_multiplier * self.transient.x[-1]
        return self.kwargs.get("xlim_high", default)

    @property
    def _ylim_low_magnitude(self) -> float:
        return self.ylim_low_magnitude_multiplier * min(self.transient.y)

    @property
    def _ylim_high_magnitude(self) -> float:
        return self.ylim_high_magnitude_multiplier * np.max(self.transient.y)

    def _get_ylim_low_with_indices(self, indices: list) -> float:
        return self.ylim_low_multiplier * min(self.transient.y[indices])

    def _get_ylim_high_with_indices(self, indices: list) -> float:
        return self.ylim_high_multiplier * np.max(self.transient.y[indices])

    def _get_x_err(self, indices: list) -> np.ndarray:
        return self.transient.x_err[indices] if self.transient.x_err is not None else self.transient.x_err

    def _set_y_axis_data(self, ax: matplotlib.axes.Axes) -> None:
        if self.transient.magnitude_data:
            ax.set_ylim(self._ylim_low_magnitude, self._ylim_high_magnitude)
            ax.invert_yaxis()
        else:
            ax.set_ylim(self._ylim_low, self._ylim_high)
            ax.set_yscale("log")

    def _set_y_axis_multiband_data(self, ax: matplotlib.axes.Axes, indices: list) -> None:
        if self.transient.magnitude_data:
            ax.set_ylim(self._ylim_low_magnitude, self._ylim_high_magnitude)
            ax.invert_yaxis()
        else:
            ax.set_ylim(self._get_ylim_low_with_indices(indices=indices),
                        self._get_ylim_high_with_indices(indices=indices))
            ax.set_yscale("log")

    def _set_x_axis(self, axes: matplotlib.axes.Axes) -> None:
        if self.transient.use_phase_model:
            axes.set_xscale("log")
        axes.set_xlim(self._xlim_low, self._xlim_high)

    @property
    def _nrows(self) -> int:
        default = int(np.ceil(len(self._filters) / 2))
        return self._get_kwarg_with_default("nrows", default=default)

    @property
    def _npanels(self) -> int:
        npanels = self._nrows * self.ncols
        if npanels < len(self._filters):
            raise ValueError(f"Insufficient number of panels. {npanels} panels were given "
                             f"but {len(self._filters)} panels are needed.")
        return npanels

    @property
    def _figsize(self) -> tuple:
        default = (4 + 4 * self.ncols, 2 + 2 * self._nrows)
        return self._get_kwarg_with_default("figsize", default=default)

    @property
    def _reference_mjd_date(self) -> int:
        if self.transient.use_phase_model:
            return self.kwargs.get("reference_mjd_date", int(self.transient.x[0]))
        return 0

    @property
    def band_label_generator(self):
        if self.band_labels is not None:
            return (bl for bl in self.band_labels)

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the Magnitude data and returns Axes.

        :param axes: Matplotlib axes to plot the data into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        ax = axes or plt.gca()

        band_label_generator = self.band_label_generator

        for indices, band in zip(self.transient.list_of_band_indices, self.transient.unique_bands):
            if band in self._filters:
                color = self._colors[list(self._filters).index(band)]
                if band_label_generator is None:
                    if band in self.band_scaling:
                        label = str(self.band_scaling.get(band))  + ' ' + self.band_scaling.get("type") + ' ' + band 
                    else:
                        label = band   
                else:
                    label = next(band_label_generator)
            elif self.plot_others:
                color = "black"
                label = None
            else:
                continue
            if isinstance(label, float):
                label = f"{label:.2e}"
            if self.band_colors is not None:
                color = self.band_colors[band]
            if band in self.band_scaling:
                if self.band_scaling.get("type") == 'x':
                    ax.errorbar(
                        self.transient.x[indices] - self._reference_mjd_date, self.transient.y[indices] * self.band_scaling.get(band),
                        xerr=self._get_x_err(indices), yerr=self.transient.y_err[indices] * self.band_scaling.get(band),
                        fmt=self.errorbar_fmt, ms=self.ms, color=color,
                        elinewidth=self.elinewidth, capsize=self.capsize, label=label)
                elif self.band_scaling.get("type") == '+':
                    ax.errorbar(
                        self.transient.x[indices] - self._reference_mjd_date, self.transient.y[indices] + self.band_scaling.get(band),
                        xerr=self._get_x_err(indices), yerr=self.transient.y_err[indices],
                        fmt=self.errorbar_fmt, ms=self.ms, color=color,
                        elinewidth=self.elinewidth, capsize=self.capsize, label=label)
            else:
                ax.errorbar(
                    self.transient.x[indices] - self._reference_mjd_date, self.transient.y[indices],
                    xerr=self._get_x_err(indices), yerr=self.transient.y_err[indices],
                    fmt=self.errorbar_fmt, ms=self.ms, color=color,
                    elinewidth=self.elinewidth, capsize=self.capsize, label=label)

        self._set_x_axis(axes=ax)
        self._set_y_axis_data(ax)

        ax.set_xlabel(self._xlabel, fontsize=self.fontsize_axes)
        ax.set_ylabel(self._ylabel, fontsize=self.fontsize_axes)

        ax.tick_params(axis='both', which='both', pad=self.axis_tick_params_pad, labelsize=self.fontsize_ticks)
        ax.legend(ncol=self.legend_cols, loc=self.legend_location, fontsize=self.fontsize_legend)

        self._save_and_show(filepath=self._data_plot_filepath, save=save, show=show)
        return ax

    def plot_lightcurve(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True)\
            -> matplotlib.axes.Axes:
        """Plots the Magnitude data and returns Axes.

        :param axes: Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        axes = axes or plt.gca()

        axes = self.plot_data(axes=axes, save=False, show=False)
        axes.set_yscale('log')

        times = self._get_times(axes)
        bands_to_plot = self._get_bands_to_plot

        color_max = self.max_likelihood_color
        color_sample = self.random_sample_color
        for band, color in zip(bands_to_plot, self.transient.get_colors(bands_to_plot)):
            if self.set_same_color_per_subplot is True:
                if self.band_colors is not None:
                    color = self.band_colors[band]
                color_max = color
                color_sample = color
            sn_cosmo_band = redback.utils.sncosmo_bandname_from_band([band])
            self._model_kwargs["bands"] = [sn_cosmo_band[0] for _ in range(len(times))]
            if isinstance(band, str):
                frequency = redback.utils.bands_to_frequency([band])
            else:
                frequency = band
            self._model_kwargs['frequency'] = np.ones(len(times)) * frequency
            if self.plot_max_likelihood:
                ys = self.model(times, **self._max_like_params, **self._model_kwargs)
                if band in self.band_scaling:
                    if self.band_scaling.get("type") == 'x':
                        axes.plot(times - self._reference_mjd_date, ys * self.band_scaling.get(band), color=color_max, alpha=self.max_likelihood_alpha, lw=self.linewidth)
                    elif self.band_scaling.get("type") == '+':
                        axes.plot(times - self._reference_mjd_date, ys + self.band_scaling.get(band), color=color_max, alpha=self.max_likelihood_alpha, lw=self.linewidth)
                else:        
                    axes.plot(times - self._reference_mjd_date, ys, color=color_max, alpha=self.max_likelihood_alpha, lw=self.linewidth)

            random_ys_list = [self.model(times, **random_params, **self._model_kwargs)
                              for random_params in self._get_random_parameters()]
            if self.uncertainty_mode == "random_models":
                for ys in random_ys_list:
                    if band in self.band_scaling:
                        if self.band_scaling.get("type") == 'x':
                            axes.plot(times - self._reference_mjd_date, ys * self.band_scaling.get(band), color=color_sample, alpha=self.random_sample_alpha, lw=self.linewidth, zorder=-1)
                        elif self.band_scaling.get("type") == '+':
                            axes.plot(times - self._reference_mjd_date, ys + self.band_scaling.get(band), color=color_sample, alpha=self.random_sample_alpha, lw=self.linewidth, zorder=-1)
                    else:
                        axes.plot(times - self._reference_mjd_date, ys, color=color_sample, alpha=self.random_sample_alpha, lw=self.linewidth, zorder=-1)
            elif self.uncertainty_mode == "credible_intervals":
                if band in self.band_scaling:
                    if self.band_scaling.get("type") == 'x':
                        lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=np.array(random_ys_list) * self.band_scaling.get(band), interval=self.credible_interval_level)
                    elif self.band_scaling.get("type") == '+':
                        lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=np.array(random_ys_list) + self.band_scaling.get(band), interval=self.credible_interval_level)
                else:
                    lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=np.array(random_ys_list), interval=self.credible_interval_level)
                axes.fill_between(
                    times - self._reference_mjd_date, lower_bound, upper_bound,
                    alpha=self.uncertainty_band_alpha, color=color_sample)

        self._save_and_show(filepath=self._lightcurve_plot_filepath, save=save, show=show)
        return axes

    def _check_valid_multiband_data_mode(self) -> bool:
        if self.transient.luminosity_data:
            redback.utils.logger.warning(
                f"Plotting multiband lightcurve/data not possible for {self.transient.data_mode}. Returning.")
            return False
        return True

    def plot_multiband(
            self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, save: bool = True,
            show: bool = True) -> matplotlib.axes.Axes:
        """Plots the Magnitude multiband data and returns Axes.

        :param figure: Matplotlib figure to plot the data into.
        :type figure: matplotlib.figure.Figure
        :param axes: Matplotlib axes to plot the data into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        if not self._check_valid_multiband_data_mode():
            return

        if figure is None or axes is None:
            figure, axes = plt.subplots(ncols=self.ncols, nrows=self._nrows, sharex='all', figsize=self._figsize)
        axes = axes.ravel()

        band_label_generator = self.band_label_generator

        ii = 0
        for indices, band, freq in zip(
                self.transient.list_of_band_indices, self.transient.unique_bands, self.transient.unique_frequencies):
            if band not in self._filters:
                continue

            x_err = self._get_x_err(indices)
            color = self._colors[list(self._filters).index(band)]
            if self.band_colors is not None:
                color = self.band_colors[band]
            if band_label_generator is None:
                label = self._get_multiband_plot_label(band, freq)
            else:
                label = next(band_label_generator)

            axes[ii].errorbar(
                self.transient.x[indices] - self._reference_mjd_date, self.transient.y[indices], xerr=x_err,
                yerr=self.transient.y_err[indices], fmt=self.errorbar_fmt, ms=self.ms, color=color,
                elinewidth=self.elinewidth, capsize=self.capsize,
                label=label)

            self._set_x_axis(axes[ii])
            self._set_y_axis_multiband_data(axes[ii], indices)
            axes[ii].legend(ncol=self.legend_cols, loc=self.legend_location, fontsize=self.fontsize_legend)
            axes[ii].tick_params(axis='both', which='both', pad=self.axis_tick_params_pad, labelsize=self.fontsize_ticks)
            ii += 1

        figure.supxlabel(self._xlabel, fontsize=self.fontsize_figure)
        figure.supylabel(self._ylabel, fontsize=self.fontsize_figure)
        plt.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

        self._save_and_show(filepath=self._multiband_data_plot_filepath, save=save, show=show)
        return axes

    @staticmethod
    def _get_multiband_plot_label(band: str, freq: float) -> str:
        if isinstance(band, str):
            if 1e10 < float(freq) < 1e16:
                label = band
            else:
                label = f"{freq:.2e}"
        else:
            label = f"{band:.2e}"
        return label

    @property
    def _filters(self) -> list[str]:
        filters = self.kwargs.get("filters", self.transient.active_bands)
        if 'bands_to_plot' in self.kwargs:
            filters = self.kwargs['bands_to_plot']
        if filters is None:
            return self.transient.active_bands
        elif str(filters) == 'default':
            return self.transient.default_filters
        return filters

    def plot_multiband_lightcurve(
        self, figure: matplotlib.figure.Figure = None, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the Magnitude multiband lightcurve and returns Axes.

        :param figure: Matplotlib figure to plot the data into.
        :param axes: Matplotlib axes to plot the data into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        if not self._check_valid_multiband_data_mode():
            return

        if figure is None or axes is None:
            figure, axes = plt.subplots(ncols=self.ncols, nrows=self._nrows, sharex='all', figsize=self._figsize)

        axes = self.plot_multiband(figure=figure, axes=axes, save=False, show=False)
        times = self._get_times(axes)

        ii = 0
        color_max = self.max_likelihood_color
        color_sample = self.random_sample_color
        for band, freq in zip(self.transient.unique_bands, self.transient.unique_frequencies):
            if band not in self._filters:
                continue
            new_model_kwargs = self._model_kwargs.copy()
            new_model_kwargs['frequency'] = freq
            new_model_kwargs['bands'] = band
            
            if self.set_same_color_per_subplot is True:
                color = self._colors[list(self._filters).index(band)]
                if self.band_colors is not None:
                    color = self.band_colors[band]
                color_max = color
                color_sample = color

            if self.plot_max_likelihood:
                ys = self.model(times, **self._max_like_params, **new_model_kwargs)
                axes[ii].plot(
                    times - self._reference_mjd_date, ys, color=color_max,
                    alpha=self.max_likelihood_alpha, lw=self.linewidth)
            random_ys_list = [self.model(times, **random_params, **new_model_kwargs)
                              for random_params in self._get_random_parameters()]
            if self.uncertainty_mode == "random_models":
                for random_ys in random_ys_list:
                    axes[ii].plot(times - self._reference_mjd_date, random_ys, color=color_sample,
                                  alpha=self.random_sample_alpha, lw=self.linewidth, zorder=self.zorder)
            elif self.uncertainty_mode == "credible_intervals":
                lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=random_ys_list, interval=self.credible_interval_level)
                axes[ii].fill_between(
                    times - self._reference_mjd_date, lower_bound, upper_bound,
                    alpha=self.uncertainty_band_alpha, color=color_sample)
            ii += 1

        self._save_and_show(filepath=self._multiband_lightcurve_plot_filepath, save=save, show=show)
        return axes


class FluxDensityPlotter(MagnitudePlotter):
    pass

class IntegratedFluxOpticalPlotter(MagnitudePlotter):
    pass

class SpectrumPlotter(SpecPlotter):
    @property
    def _xlabel(self) -> str:
        return self.transient.xlabel

    @property
    def _ylabel(self) -> str:
        return self.transient.ylabel

    def plot_data(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the spectrum data and returns Axes.

        :param axes: Matplotlib axes to plot the data into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        ax = axes or plt.gca()

        if self.transient.plot_with_time_label:
            label = self.transient.time
        else:
            label = self.transient.name
        ax.plot(self.transient.angstroms, self.transient.flux_density/1e-17, color=self.color,
                lw=self.linewidth)
        ax.set_xscale('linear')
        ax.set_yscale(self.yscale)

        ax.set_xlim(self._xlim_low, self._xlim_high)
        ax.set_ylim(self._ylim_low, self._ylim_high)
        ax.set_xlabel(self._xlabel, fontsize=self.fontsize_axes)
        ax.set_ylabel(self._ylabel, fontsize=self.fontsize_axes)

        ax.annotate(
            label, xy=self.xy, xycoords=self.xycoords,
            horizontalalignment=self.horizontalalignment, size=self.annotation_size)

        ax.tick_params(axis='both', which='both', pad=self.axis_tick_params_pad, labelsize=self.fontsize_ticks)

        self._save_and_show(filepath=self._data_plot_filepath, save=save, show=show)
        return ax

    def plot_spectrum(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the spectrum data and the fit and returns Axes.

        :param axes: Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        :type axes: Union[matplotlib.axes.Axes, None], optional
        :param save: Whether to save the plot. (Default value = True)
        :type save: bool
        :param show: Whether to show the plot. (Default value = True)
        :type show: bool

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """

        axes = axes or plt.gca()

        axes = self.plot_data(axes=axes, save=False, show=False)
        angstroms = self._get_angstroms(axes)

        self._plot_spectrums(axes, angstroms)

        self._save_and_show(filepath=self._spectrum_ppd_plot_filepath, save=save, show=show)
        return axes

    def _plot_spectrums(self, axes: matplotlib.axes.Axes, angstroms: np.ndarray) -> None:
        if self.plot_max_likelihood:
            ys = self.model(angstroms, **self._max_like_params, **self._model_kwargs)
            axes.plot(angstroms, ys/1e-17, color=self.max_likelihood_color, alpha=self.max_likelihood_alpha,
                      lw=self.linewidth)

        random_ys_list = [self.model(angstroms, **random_params, **self._model_kwargs)
                          for random_params in self._get_random_parameters()]
        if self.uncertainty_mode == "random_models":
            for ys in random_ys_list:
                axes.plot(angstroms, ys/1e-17, color=self.random_sample_color, alpha=self.random_sample_alpha,
                          lw=self.linewidth, zorder=self.zorder)
        elif self.uncertainty_mode == "credible_intervals":
            lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=random_ys_list,
                                                                                interval=self.credible_interval_level)
            axes.fill_between(
                angstroms, lower_bound/1e-17, upper_bound/1e-17, alpha=self.uncertainty_band_alpha, color=self.max_likelihood_color)

    def plot_residuals(
            self, axes: matplotlib.axes.Axes = None, save: bool = True, show: bool = True) -> matplotlib.axes.Axes:
        """Plots the residual of the Integrated flux data returns Axes.

        :param axes: Matplotlib axes to plot the lightcurve into. Useful for user specific modifications to the plot.
        :param save: Whether to save the plot. (Default value = True)
        :param show: Whether to show the plot. (Default value = True)

        :return: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        """
        if axes is None:
            fig, axes = plt.subplots(
                nrows=2, ncols=1, sharex=True, sharey=False, figsize=(10, 8), gridspec_kw=dict(height_ratios=[2, 1]))

        axes[0] = self.plot_spectrum(axes=axes[0], save=False, show=False)
        axes[1].set_xlabel(axes[0].get_xlabel(), fontsize=self.fontsize_axes)
        axes[0].set_xlabel("")
        ys = self.model(self.transient.angstroms, **self._max_like_params, **self._model_kwargs)
        axes[1].errorbar(
            self.transient.angstroms, self.transient.flux_density - ys, yerr=self.transient.flux_density_err,
            fmt=self.errorbar_fmt, c=self.color, ms=self.ms, elinewidth=self.elinewidth, capsize=self.capsize)
        axes[1].set_yscale('linear')
        axes[1].set_ylabel("Residual", fontsize=self.fontsize_axes)
        axes[1].tick_params(axis='both', which='both', pad=self.axis_tick_params_pad, labelsize=self.fontsize_ticks)

        self._save_and_show(filepath=self._residual_plot_filepath, save=save, show=show)
        return axes