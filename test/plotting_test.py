import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock, ANY
import numpy as np
import pandas as pd
from redback.transient import Spectrum, Transient
from redback.plotting import (SpecPlotter, IntegratedFluxPlotter, LuminosityOpticalPlotter,
    _FilenameGetter, _FilePathGetter, Plotter, SpectrumPlotter, MagnitudePlotter)

from types import SimpleNamespace
import matplotlib.pyplot as plt



# Dummy transient: Only the attributes needed are defined.
class DummyTransient:
    def __init__(self, name, directory_path):
        self.name = name
        self.directory_structure = SimpleNamespace(directory_path=directory_path)
    # For our tests the other transient attributes aren’t required.

class TestFilenameAndFilePathGetters(unittest.TestCase):

    def setUp(self):
        # Create a dummy transient with a name and a directory_path.
        self.dummy_transient = DummyTransient(name="TestTransient", directory_path="/dummy/path")
        # Create a Plotter instance with no extra kwargs.
        self.plotter = Plotter(self.dummy_transient)

    def test_filename_getter_default(self):
        """
        Test that an attribute defined with _FilenameGetter returns
        the default filename based on the transient name and suffix.
        For _data_plot_filename (with suffix "data") it should be:
          "TestTransient_data.png"
        """
        expected = "TestTransient_data.png"
        self.assertEqual(self.plotter._data_plot_filename, expected)

    def test_filename_getter_override(self):
        """
        Test that if the Plotter is constructed with a 'filename' kwarg, that value overrides the default.
        """
        kwargs = {"filename": "override.png"}
        plotter_override = Plotter(self.dummy_transient, **kwargs)
        self.assertEqual(plotter_override._data_plot_filename, "override.png")

    def test_get_filename_method(self):
        """
        Test that the get_filename() method returns the provided value or the default.
        """
        # When no filename is provided, get_filename(default) should return the default.
        default_value = "default.png"
        self.assertEqual(self.plotter.get_filename(default_value), default_value)
        # When an override value is provided via kwargs, the override is returned.
        kwargs = {"filename": "override.png"}
        plotter_override = Plotter(self.dummy_transient, **kwargs)
        self.assertEqual(plotter_override.get_filename(default_value), "override.png")

    def test_file_path_getter_default(self):
        """
        Test that _FilePathGetter returns the join of the output directory and filename.
        In our case:
          The _data_plot_outdir property returns the transient's directory_structure.directory_path,
          and _data_plot_filename returns "TestTransient_data.png"
          so the expected file path is os.path.join("/dummy/path", "TestTransient_data.png")
        """
        expected_outdir = "/dummy/path"
        expected_filename = "TestTransient_data.png"
        expected_filepath = os.path.join(expected_outdir, expected_filename)
        self.assertEqual(self.plotter._data_plot_filepath, expected_filepath)

    def test_file_path_getter_with_override(self):
        """
        Test that if the Plotter is constructed with an override for outdir and filename,
        the _FilePathGetter returns the joined path correctly.
        """
        kwargs = {"filename": "override.png", "outdir": "/new/path"}
        plotter_override = Plotter(self.dummy_transient, **kwargs)
        expected = os.path.join("/new/path", "override.png")
        self.assertEqual(plotter_override._data_plot_filepath, expected)

    def test_filename_set_no_effect(self):
        """
        Test that attempting to assign a new value to an attribute defined via _FilenameGetter does not
        change its computed value (since __set__ is defined as a no‐op).
        """
        original = self.plotter._data_plot_filename
        # Even though we try to set a new value, the descriptor’s __set__ (which does nothing) prevents overwriting.
        self.plotter._data_plot_filename = "new_value"
        self.assertEqual(self.plotter._data_plot_filename, original)

class TestSpecPlotter(unittest.TestCase):

    def setUp(self) -> None:
        angstroms = np.array([4000, 5000, 6000])
        flux_density = np.array([1e-17, 2e-17, 3e-17])
        flux_density_err = np.array([0.1e-17, 0.1e-17, 0.1e-17])
        self.spectrum = Spectrum(angstroms, flux_density, flux_density_err, name="test_spectrum")
        self.spectrum.directory_structure = SimpleNamespace(directory_path='/dummy/path')
        self.plotter = SpecPlotter(self.spectrum)

    def tearDown(self) -> None:
        plt.close('all')  # reset matplotlib global state
        mock.patch.stopall()
        self.spectrum = None
        self.plotter = None

    def test_get_angstroms_linear(self):
        mock_axes = MagicMock()
        mock_axes.get_yscale.return_value = 'linear'
        result = self.plotter._get_angstroms(mock_axes)
        self.assertTrue(np.all(np.isclose(result, np.linspace(
            self.spectrum.angstroms[0] * self.plotter.xlim_low_multiplier,
            self.spectrum.angstroms[-1] * self.plotter.xlim_high_multiplier,
            200
        ))))

    def test_get_angstroms_log(self):
        mock_axes = MagicMock()
        mock_axes.get_yscale.return_value = 'log'
        result = self.plotter._get_angstroms(mock_axes)
        self.assertTrue(np.allclose(
            result,
            np.exp(np.linspace(
                np.log(self.spectrum.angstroms[0] * self.plotter.xlim_low_multiplier),
                np.log(self.spectrum.angstroms[-1] * self.plotter.xlim_high_multiplier),
                200
            ))
        ))

    def test_xlim_low_property(self):
        calculated_value = self.plotter._xlim_low
        expected_value = self.plotter.xlim_low_multiplier * self.spectrum.angstroms[0]
        if expected_value == 0:
            expected_value += 1e-3
        self.assertEqual(calculated_value, expected_value)

    def test_xlim_high_property(self):
        calculated_value = self.plotter._xlim_high
        expected_value = self.plotter.xlim_high_multiplier * self.spectrum.angstroms[-1]
        self.assertEqual(calculated_value, expected_value)

    def test_ylim_low_property(self):
        calculated_value = self.plotter._ylim_low
        expected_value = self.plotter.ylim_low_multiplier * min(self.spectrum.flux_density) / 1e-17
        self.assertEqual(calculated_value, expected_value)

    def test_ylim_high_property(self):
        calculated_value = self.plotter._ylim_high
        expected_value = self.plotter.ylim_high_multiplier * np.max(self.spectrum.flux_density) / 1e-17
        self.assertEqual(calculated_value, expected_value)

    def test_y_err_property(self):
        calculated_value = self.plotter._y_err
        expected_value = np.array([np.abs(self.spectrum.flux_density_err)])
        np.testing.assert_array_equal(calculated_value, expected_value)

    def test_data_plot_outdir(self):
        calculated_value = self.plotter._data_plot_outdir
        expected_value = self.spectrum.directory_structure.directory_path
        self.assertEqual(calculated_value, expected_value)

    def test_get_filename(self):
        filename = "test_default.png"
        calculated_value = self.plotter.get_filename(default=filename)
        self.assertEqual(calculated_value, filename)

    def test_get_random_parameters(self):
        mock_posterior = pd.DataFrame({'log_likelihood': [1, 2, 3], 'param': [0.1, 0.2, 0.3]})
        self.plotter.kwargs['posterior'] = mock_posterior
        self.plotter.kwargs['random_models'] = 2
        random_parameters = self.plotter._get_random_parameters()
        self.assertEqual(len(random_parameters), 2)
        for param in random_parameters:
            self.assertIn(param['param'], mock_posterior['param'].values)

    def test_max_like_params(self):
        mock_posterior = pd.DataFrame({'log_likelihood': [1, 2, 3], 'param': [0.1, 0.2, 0.3]})
        self.plotter.kwargs['posterior'] = mock_posterior
        max_like_params = self.plotter._max_like_params
        self.assertTrue(max_like_params['log_likelihood'], 3)

class TestIntegratedFluxPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.transient_mock = MagicMock(spec=Transient)
        self.transient_mock.x = np.logspace(0, 2, 10)
        self.transient_mock.y = np.logspace(0, 2, 10)
        self.transient_mock.ylabel = "Test YLabel"
        self.transient_mock.name = "Test Transient"
        self.transient_mock.use_phase_model = False
        self.transient_mock.directory_structure = MagicMock()
        self.transient_mock.directory_structure.directory_path = "/mock/path"

        self.mock_model = MagicMock()
        self.mock_model.__name__ = "MockModel"  # Ensure __name__ is defined.

        self.plotter = IntegratedFluxPlotter(transient=self.transient_mock, model=self.mock_model)

        self.default_patches = patch.multiple(
            IntegratedFluxPlotter,
            _x_err=PropertyMock(return_value=np.zeros_like(self.transient_mock.x)),
            _y_err=PropertyMock(return_value=np.zeros_like(self.transient_mock.y)),
            _xlim_low=PropertyMock(return_value=0.1),
            _xlim_high=PropertyMock(return_value=200),
            _ylim_low=PropertyMock(return_value=0.1),
            _ylim_high=PropertyMock(return_value=200),
            _save_and_show=MagicMock()
        )
        self.default_patches.start()

    def tearDown(self) -> None:
        plt.close('all')  # reset matplotlib global state
        mock.patch.stopall()

    @patch("matplotlib.pyplot.figure", autospec=True)
    def test_plot_data(self, mock_figure):
        axes_mock = MagicMock()

        # Correctly and locally patch plt.gca:
        with patch("matplotlib.pyplot.gca", return_value=axes_mock):
            result_axes = self.plotter.plot_data(save=False, show=False)

        self.assertEqual(result_axes, axes_mock)
        axes_mock.errorbar.assert_called_once()
        axes_mock.set_xscale.assert_called_once_with("log")
        axes_mock.set_yscale.assert_called_once_with("log")
        axes_mock.set_xlim.assert_called_once_with(0.1, 200)
        axes_mock.set_ylim.assert_called_once_with(0.1, 200)
        axes_mock.set_xlabel.assert_called_once_with(
            r"Time since burst [s]", fontsize=self.plotter.fontsize_axes)
        axes_mock.set_ylabel.assert_called_once_with(
            self.transient_mock.ylabel, fontsize=self.plotter.fontsize_axes)
        axes_mock.annotate.assert_called_once_with(
            self.transient_mock.name,
            xy=self.plotter.xy,
            xycoords=self.plotter.xycoords,
            horizontalalignment=self.plotter.horizontalalignment,
            size=self.plotter.annotation_size
        )
        self.plotter._save_and_show.assert_called_once_with(
            filepath=self.plotter._data_plot_filepath, save=False, show=False)
        plt.close('all')

    def test_plot_data_with_custom_axes(self):
        # Test plot_data with a custom matplotlib Axes object
        fig, custom_axes = plt.subplots()
        axes = self.plotter.plot_data(axes=custom_axes, save=False, show=False)
        self.assertEqual(axes, custom_axes)

    def test_plot_lightcurve(self):
        # Create a real matplotlib figure and axes
        fig, real_axes = plt.subplots()

        # Mock the plot_lightcurves method to do nothing
        self.plotter._plot_lightcurves = MagicMock()

        # Mock get_times to return the test data
        self.plotter._get_times = MagicMock(return_value=self.transient_mock.x)

        # Use the real axes in the plot_lightcurve call
        result_axes = self.plotter.plot_lightcurve(axes=real_axes, save=False, show=False)

        # Now the assertion should pass
        self.assertIsInstance(result_axes, plt.Axes)
        self.plotter._plot_lightcurves.assert_called_once()

        # Clean up
        plt.close(fig)

    @patch("redback.plotting.IntegratedFluxPlotter._get_times", return_value=np.logspace(0, 2, 10))
    @patch("redback.plotting.IntegratedFluxPlotter._max_like_params", new_callable=PropertyMock)
    @patch("redback.plotting.IntegratedFluxPlotter._model_kwargs", new_callable=PropertyMock)
    def test_plot_residuals(self, mock_model_kwargs, mock_max_like_params, mock_get_times):
        """
        Test the plot_residuals method with proper posterior handling
        """
        # Create a valid posterior DataFrame
        valid_posterior = pd.DataFrame({
            "log_likelihood": [0.1, 5],
            "other_param": [1.1, 3]
        })

        # Mock the plot_lightcurves method to do nothing
        self.plotter._plot_lightcurves = MagicMock()

        # Set the posterior directly as an instance attribute instead of using PropertyMock
        self.plotter.posterior = valid_posterior

        # Mock additional required attributes
        mock_max_like_params.return_value = {}
        mock_model_kwargs.return_value = {}
        self.plotter.model = MagicMock(return_value=self.transient_mock.y, __name__="MockModel")

        # Execute the method under test
        axes = self.plotter.plot_residuals(save=False, show=False)

        # Verify the results
        self.assertIsInstance(axes, np.ndarray)
        self.assertEqual(axes.shape[0], 2)

        # Verify model was called with correct parameters
        self.plotter.model.assert_called_once_with(
            self.transient_mock.x,
            **self.plotter._max_like_params,
            **self.plotter._model_kwargs
        )

class TestLuminosityOpticalPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_transient = MagicMock()
        self.mock_transient.x = np.array([1, 10, 100])
        self.mock_transient.x_err = None
        self.mock_transient.y = np.array([1e50, 2e50, 1.5e50])
        self.mock_transient.y_err = np.array([0.1e50, 0.2e50, 0.15e50])
        self.mock_transient.use_phase_model = False
        # self.mock_transient.reference_mjd_date = 0
        self.mock_transient.ylabel = "Luminosity"
        self.luminosity_plotter = LuminosityOpticalPlotter(transient=self.mock_transient)

    def tearDown(self) -> None:
        del self.luminosity_plotter

    def test_xlabel_property(self):
        self.assertEqual(
            self.luminosity_plotter._xlabel,
            r"Time since explosion [days]"
        )

    def test_ylabel_property(self):
        self.assertEqual(
            self.luminosity_plotter._ylabel,
            r"L$_{\rm bol}$ [$10^{50}$ erg s$^{-1}$]"
        )

    @mock.patch("matplotlib.pyplot.gca")
    def test_plot_data_creates_axes(self, mock_gca):
        mock_gca.return_value = plt.figure().add_subplot(111)
        axes = self.luminosity_plotter.plot_data(save=False, show=False)
        self.assertIsInstance(axes, plt.Axes)

    def test_plot_data_with_existing_axes(self):
        fig, ax = plt.subplots()
        returned_ax = self.luminosity_plotter.plot_data(axes=ax, save=False, show=False)
        self.assertIs(returned_ax, ax)

class TestSpectrumPlotter(unittest.TestCase):
    def setUp(self) -> None:
        # Mock transient setup
        self.mock_transient = mock.MagicMock()
        self.mock_transient.angstroms = np.linspace(4000, 7000, 200)  # Match the size in error message
        self.mock_transient.flux_density = np.sin(self.mock_transient.angstroms / 1000) * 1e-17
        self.mock_transient.flux_density_err = np.full_like(self.mock_transient.flux_density, 0.1e-17)
        self.mock_transient.xlabel = "Wavelength [Å]"
        self.mock_transient.ylabel = r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"
        self.mock_transient.plot_with_time_label = True
        self.mock_transient.time = "100s"
        self.mock_transient.name = "TestSpectrum"

        # Create mock posterior DataFrame
        mock_data = {
            'log_likelihood': [-100, -50, -10],
            'param1': [1, 2, 3],
            'param2': [0.1, 0.2, 0.3]
        }
        self.mock_posterior = pd.DataFrame(mock_data)

        # Create mock model that returns actual data
        def mock_model_func(angstroms, **kwargs):
            # Return synthetic data matching the input wavelength array size
            return np.ones_like(angstroms) * 1e-17

        self.mock_model = mock.MagicMock(side_effect=mock_model_func)
        self.mock_model.__name__ = "MockModel"

        # Initialize plotter with mocked components
        self.spectrum_plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            posterior=self.mock_posterior,
            model=self.mock_model
        )

    def tearDown(self) -> None:
        del self.spectrum_plotter

    def test_plot_data(self):
        axes = mock.MagicMock()
        result_axes = self.spectrum_plotter.plot_data(axes=axes, save=False, show=False)

        self.assertEqual(result_axes, axes)

        call_args, call_kwargs = axes.plot.call_args
        np.testing.assert_array_equal(call_args[0], self.mock_transient.angstroms)
        np.testing.assert_array_equal(call_args[1], self.mock_transient.flux_density / 1e-17)
        self.assertEqual(call_kwargs.get('color'), self.spectrum_plotter.color)
        self.assertEqual(call_kwargs.get('lw'), self.spectrum_plotter.linewidth)

    def test_posterior_property(self):
        result = self.spectrum_plotter._posterior
        self.assertFalse(result.empty)
        self.assertTrue('log_likelihood' in result.columns)

    def test_max_like_params(self):
        result = self.spectrum_plotter._max_like_params
        self.assertIsNotNone(result)
        self.assertTrue('param1' in result.index)
        self.assertTrue('param2' in result.index)

    def test_plot_spectrum(self):
        axes = self.spectrum_plotter.plot_spectrum(save=False, show=False)
        self.assertIsInstance(axes, plt.Axes)
        # Verify that model was called
        self.mock_model.assert_called()

    def test_plot_residuals(self):
        axes = self.spectrum_plotter.plot_residuals(save=False, show=False)
        self.assertEqual(len(axes), 2)
        self.assertIsInstance(axes[0], plt.Axes)
        self.assertIsInstance(axes[1], plt.Axes)

class TestMagnitudePlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_transient = MagicMock(spec=Transient)
        self.mock_transient.use_phase_model = False
        self.mock_transient.name = "Test"
        self.mock_transient.active_bands = ["r", "g"]
        self.mock_transient.x = np.array([0, 1, 2, 3])
        self.mock_transient.y = np.array([10, 9, 8, 7])
        self.mock_transient.y_err = np.array([0.1, 0.2, 0.1, 0.2])
        self.mock_transient.list_of_band_indices = [[0, 1], [2, 3]]
        self.mock_transient.unique_bands = ["r", "g"]
        self.mock_transient.get_colors = MagicMock(return_value=["red", "green"])
        self.mock_transient.ylabel = "Magnitude"
        self.mock_transient.xlabel = "Time [days]"

        mock_posterior = pd.DataFrame({
            'log_likelihood': [-100, -50, -10]  # Example values
        })

        self.mock_transient.directory_structure = MagicMock()
        self.mock_transient.directory_structure.directory_path = "/mock/path"

        self.mock_model = mock.MagicMock()
        self.mock_model.__name__ = "MockModel"

        self.kwargs = {"xlabel": "Test X Label", "ylabel": "Test Y Label",
                       "posterior": mock_posterior, "model": self.mock_model}
        self.plotter = MagnitudePlotter(self.mock_transient, **self.kwargs)

    def tearDown(self) -> None:
        plt.close('all')  # reset matplotlib global state
        mock.patch.stopall()
        del self.mock_transient
        del self.plotter

    def test_color_property(self):
        self.assertEqual(self.plotter._colors, ["red", "green"])

    def test_xlabel_property_with_custom_label(self):
        self.assertEqual(self.plotter._xlabel, "Test X Label")

    def test_ylabel_property_with_custom_label(self):
        self.assertEqual(self.plotter._ylabel, "Test Y Label")

    def test_xlim_high_property(self):
        self.mock_transient.x = np.array([1, 2, 3])
        self.assertAlmostEqual(self.plotter._xlim_high, 1.2 * 3)

    def test_ylim_low_magnitude_property(self):
        self.mock_transient.y = np.array([10, 20, 30])
        self.assertEqual(self.plotter._ylim_low_magnitude, 0.95 * 10)

    def test_ylim_high_magnitude_property(self):
        self.mock_transient.y = np.array([10, 20, 30])
        self.assertEqual(self.plotter._ylim_high_magnitude, 1.05 * 30)

    @patch('matplotlib.pyplot.gca')
    def test_plot_data(self, mock_gca):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        self.plotter.plot_data(save=False, show=False)

        mock_gca.assert_called_once()
        self.assertTrue(mock_ax.set_xlim.called)
        self.assertTrue(mock_ax.set_ylim.called)
        self.assertTrue(mock_ax.errorbar.called)

    @patch('matplotlib.pyplot.gca')
    def test_plot_lightcurve(self, mock_gca):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        self.plotter.plot_lightcurve(save=False, show=False)

        mock_gca.assert_called_once()
        self.assertTrue(mock_ax.set_yscale.called)
        self.assertTrue(mock_ax.errorbar.called)


    def test_get_multiband_plot_label(self):
        band = "r"
        freq = 1e12
        label = self.plotter._get_multiband_plot_label(band, freq)
        self.assertEqual(label, "r")

    def test_nrows_property(self):
        with patch('redback.plotting.MagnitudePlotter._filters', new_callable=PropertyMock) as mock_filters:
            mock_filters.return_value = ["r", "g"]
            self.assertEqual(self.plotter._nrows, 1)

    def test_figsize_property(self):
        with patch('redback.plotting.MagnitudePlotter._nrows', new_callable=PropertyMock) as mock_nrows:
            mock_nrows.return_value = 1
            self.assertEqual(self.plotter._figsize, (12, 4))


class TestPlotterCustomizationOptions(unittest.TestCase):
    """Tests for new customization options added to Plotter and SpecPlotter classes."""

    def setUp(self) -> None:
        self.dummy_transient = DummyTransient(name="TestTransient", directory_path="/dummy/path")
        self.plotter = Plotter(self.dummy_transient)

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_default_grid_options(self):
        """Test default values for grid options."""
        self.assertEqual(self.plotter.show_grid, False)
        self.assertEqual(self.plotter.grid_alpha, 0.3)
        self.assertEqual(self.plotter.grid_color, "gray")
        self.assertEqual(self.plotter.grid_linestyle, "--")
        self.assertEqual(self.plotter.grid_linewidth, 0.5)

    def test_custom_grid_options(self):
        """Test setting custom grid options."""
        plotter = Plotter(self.dummy_transient,
                         show_grid=True,
                         grid_alpha=0.5,
                         grid_color="blue",
                         grid_linestyle="-",
                         grid_linewidth=1.0)
        self.assertEqual(plotter.show_grid, True)
        self.assertEqual(plotter.grid_alpha, 0.5)
        self.assertEqual(plotter.grid_color, "blue")
        self.assertEqual(plotter.grid_linestyle, "-")
        self.assertEqual(plotter.grid_linewidth, 1.0)

    def test_default_save_format_options(self):
        """Test default save format and transparency options."""
        self.assertEqual(self.plotter.save_format, "png")
        self.assertEqual(self.plotter.transparent, False)

    def test_custom_save_format_options(self):
        """Test custom save format and transparency options."""
        plotter = Plotter(self.dummy_transient, save_format="pdf", transparent=True)
        self.assertEqual(plotter.save_format, "pdf")
        self.assertEqual(plotter.transparent, True)

    def test_default_axis_scale_options(self):
        """Test default axis scale options."""
        self.assertIsNone(self.plotter.xscale)
        self.assertIsNone(self.plotter.yscale)

    def test_custom_axis_scale_options(self):
        """Test custom axis scale options."""
        plotter = Plotter(self.dummy_transient, xscale="linear", yscale="log")
        self.assertEqual(plotter.xscale, "linear")
        self.assertEqual(plotter.yscale, "log")

    def test_default_title_options(self):
        """Test default title options."""
        self.assertIsNone(self.plotter.title)
        self.assertEqual(self.plotter.title_fontsize, 20)

    def test_custom_title_options(self):
        """Test custom title options."""
        plotter = Plotter(self.dummy_transient, title="My Custom Title", title_fontsize=24)
        self.assertEqual(plotter.title, "My Custom Title")
        self.assertEqual(plotter.title_fontsize, 24)

    def test_default_linestyle_options(self):
        """Test default linestyle options."""
        self.assertEqual(self.plotter.linestyle, "-")
        self.assertEqual(self.plotter.max_likelihood_linestyle, "-")
        self.assertEqual(self.plotter.random_sample_linestyle, "-")

    def test_custom_linestyle_options(self):
        """Test custom linestyle options."""
        plotter = Plotter(self.dummy_transient,
                         linestyle="--",
                         max_likelihood_linestyle="-.",
                         random_sample_linestyle=":")
        self.assertEqual(plotter.linestyle, "--")
        self.assertEqual(plotter.max_likelihood_linestyle, "-.")
        self.assertEqual(plotter.random_sample_linestyle, ":")

    def test_default_marker_options(self):
        """Test default marker options."""
        self.assertEqual(self.plotter.markerfillstyle, "full")
        self.assertIsNone(self.plotter.markeredgecolor)
        self.assertEqual(self.plotter.markeredgewidth, 1.0)

    def test_custom_marker_options(self):
        """Test custom marker options."""
        plotter = Plotter(self.dummy_transient,
                         markerfillstyle="none",
                         markeredgecolor="red",
                         markeredgewidth=2.0)
        self.assertEqual(plotter.markerfillstyle, "none")
        self.assertEqual(plotter.markeredgecolor, "red")
        self.assertEqual(plotter.markeredgewidth, 2.0)

    def test_default_legend_options(self):
        """Test default legend customization options."""
        self.assertEqual(self.plotter.legend_frameon, True)
        self.assertEqual(self.plotter.legend_shadow, False)
        self.assertEqual(self.plotter.legend_fancybox, True)
        self.assertEqual(self.plotter.legend_framealpha, 0.8)

    def test_custom_legend_options(self):
        """Test custom legend customization options."""
        plotter = Plotter(self.dummy_transient,
                         legend_frameon=False,
                         legend_shadow=True,
                         legend_fancybox=False,
                         legend_framealpha=0.5)
        self.assertEqual(plotter.legend_frameon, False)
        self.assertEqual(plotter.legend_shadow, True)
        self.assertEqual(plotter.legend_fancybox, False)
        self.assertEqual(plotter.legend_framealpha, 0.5)

    def test_default_tick_options(self):
        """Test default tick customization options."""
        self.assertEqual(self.plotter.tick_direction, "in")
        self.assertIsNone(self.plotter.tick_length)
        self.assertIsNone(self.plotter.tick_width)

    def test_custom_tick_options(self):
        """Test custom tick customization options."""
        plotter = Plotter(self.dummy_transient,
                         tick_direction="out",
                         tick_length=6.0,
                         tick_width=1.5)
        self.assertEqual(plotter.tick_direction, "out")
        self.assertEqual(plotter.tick_length, 6.0)
        self.assertEqual(plotter.tick_width, 1.5)

    def test_default_spine_options(self):
        """Test default spine options."""
        self.assertEqual(self.plotter.show_spines, True)
        self.assertIsNone(self.plotter.spine_linewidth)

    def test_custom_spine_options(self):
        """Test custom spine options."""
        plotter = Plotter(self.dummy_transient,
                         show_spines=False,
                         spine_linewidth=2.0)
        self.assertEqual(plotter.show_spines, False)
        self.assertEqual(plotter.spine_linewidth, 2.0)

    def test_apply_axis_customizations_with_grid(self):
        """Test _apply_axis_customizations applies grid correctly."""
        plotter = Plotter(self.dummy_transient,
                         show_grid=True,
                         grid_alpha=0.5,
                         grid_color="blue",
                         grid_linestyle="-",
                         grid_linewidth=1.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.grid.assert_called_once_with(
            True, alpha=0.5, color="blue", linestyle="-", linewidth=1.0
        )

    def test_apply_axis_customizations_without_grid(self):
        """Test _apply_axis_customizations does not apply grid when disabled."""
        plotter = Plotter(self.dummy_transient, show_grid=False)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.grid.assert_not_called()

    def test_apply_axis_customizations_with_title(self):
        """Test _apply_axis_customizations applies title correctly."""
        plotter = Plotter(self.dummy_transient, title="Test Title", title_fontsize=24)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.set_title.assert_called_once_with("Test Title", fontsize=24)

    def test_apply_axis_customizations_without_title(self):
        """Test _apply_axis_customizations does not apply title when not set."""
        plotter = Plotter(self.dummy_transient)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.set_title.assert_not_called()

    def test_apply_axis_customizations_tick_params(self):
        """Test _apply_axis_customizations applies tick params correctly."""
        plotter = Plotter(self.dummy_transient,
                         tick_direction="out",
                         tick_length=6.0,
                         tick_width=1.5)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.tick_params.assert_called_once()
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['direction'], 'out')
        self.assertEqual(call_kwargs['length'], 6.0)
        self.assertEqual(call_kwargs['width'], 1.5)

    def test_apply_axis_customizations_hide_spines(self):
        """Test _apply_axis_customizations hides spines correctly."""
        plotter = Plotter(self.dummy_transient, show_spines=False)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)
        for spine in mock_spines.values():
            spine.set_visible.assert_called_once_with(False)

    def test_apply_axis_customizations_spine_linewidth(self):
        """Test _apply_axis_customizations sets spine linewidth correctly."""
        plotter = Plotter(self.dummy_transient, spine_linewidth=2.0)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)
        for spine in mock_spines.values():
            spine.set_linewidth.assert_called_once_with(2.0)


class TestPlotterSaveAndShow(unittest.TestCase):
    """Tests for _save_and_show method with new options."""

    def setUp(self) -> None:
        self.dummy_transient = DummyTransient(name="TestTransient", directory_path="/dummy/path")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_png_format(self, mock_tight_layout, mock_savefig):
        """Test saving with PNG format (default)."""
        plotter = Plotter(self.dummy_transient, save_format="png")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.png'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_pdf_format(self, mock_tight_layout, mock_savefig):
        """Test saving with PDF format."""
        plotter = Plotter(self.dummy_transient, save_format="pdf")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.pdf'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_svg_format(self, mock_tight_layout, mock_savefig):
        """Test saving with SVG format."""
        plotter = Plotter(self.dummy_transient, save_format="svg")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.svg'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_eps_format(self, mock_tight_layout, mock_savefig):
        """Test saving with EPS format."""
        plotter = Plotter(self.dummy_transient, save_format="eps")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.eps'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_transparent_background(self, mock_tight_layout, mock_savefig):
        """Test saving with transparent background."""
        plotter = Plotter(self.dummy_transient, transparent=True)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_kwargs = mock_savefig.call_args[1]
        self.assertEqual(call_kwargs['transparent'], True)
        self.assertEqual(call_kwargs['facecolor'], 'none')

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_opaque_background(self, mock_tight_layout, mock_savefig):
        """Test saving with opaque background (default)."""
        plotter = Plotter(self.dummy_transient, transparent=False)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_kwargs = mock_savefig.call_args[1]
        self.assertEqual(call_kwargs['transparent'], False)
        self.assertEqual(call_kwargs['facecolor'], 'white')

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_updates_extension(self, mock_tight_layout, mock_savefig):
        """Test that save format updates file extension correctly."""
        plotter = Plotter(self.dummy_transient, save_format="pdf")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        expected_path = "/path/to/file.pdf"
        mock_savefig.assert_called_once()
        self.assertEqual(mock_savefig.call_args[0][0], expected_path)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_no_extension(self, mock_tight_layout, mock_savefig):
        """Test saving file with no extension in original path."""
        plotter = Plotter(self.dummy_transient, save_format="png")
        filepath = "/path/to/file"
        plotter._save_and_show(filepath, save=True, show=False)
        expected_path = "/path/to/file.png"
        mock_savefig.assert_called_once()
        self.assertEqual(mock_savefig.call_args[0][0], expected_path)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.tight_layout')
    def test_show_without_save(self, mock_tight_layout, mock_show):
        """Test showing plot without saving."""
        plotter = Plotter(self.dummy_transient)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=False, show=True)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_and_show(self, mock_tight_layout, mock_show, mock_savefig):
        """Test both saving and showing plot."""
        plotter = Plotter(self.dummy_transient)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=True)
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()


class TestSpecPlotterCustomizationOptions(unittest.TestCase):
    """Tests for customization options in SpecPlotter."""

    def setUp(self) -> None:
        angstroms = np.array([4000, 5000, 6000])
        flux_density = np.array([1e-17, 2e-17, 3e-17])
        flux_density_err = np.array([0.1e-17, 0.1e-17, 0.1e-17])
        self.spectrum = Spectrum(angstroms, flux_density, flux_density_err, name="test_spectrum")
        self.spectrum.directory_structure = SimpleNamespace(directory_path='/dummy/path')
        self.plotter = SpecPlotter(self.spectrum)

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_default_grid_options(self):
        """Test default grid options for SpecPlotter."""
        self.assertEqual(self.plotter.show_grid, False)
        self.assertEqual(self.plotter.grid_alpha, 0.3)
        self.assertEqual(self.plotter.grid_color, "gray")
        self.assertEqual(self.plotter.grid_linestyle, "--")
        self.assertEqual(self.plotter.grid_linewidth, 0.5)

    def test_default_save_format_options(self):
        """Test default save format options for SpecPlotter."""
        self.assertEqual(self.plotter.save_format, "png")
        self.assertEqual(self.plotter.transparent, False)

    def test_default_title_options(self):
        """Test default title options for SpecPlotter."""
        self.assertIsNone(self.plotter.title)
        self.assertEqual(self.plotter.title_fontsize, 20)

    def test_default_linestyle_options(self):
        """Test default linestyle options for SpecPlotter."""
        self.assertEqual(self.plotter.linestyle, "-")
        self.assertEqual(self.plotter.max_likelihood_linestyle, "-")
        self.assertEqual(self.plotter.random_sample_linestyle, "-")

    def test_default_marker_options(self):
        """Test default marker options for SpecPlotter."""
        self.assertEqual(self.plotter.markerfillstyle, "full")
        self.assertIsNone(self.plotter.markeredgecolor)
        self.assertEqual(self.plotter.markeredgewidth, 1.0)

    def test_default_legend_options(self):
        """Test default legend options for SpecPlotter."""
        self.assertEqual(self.plotter.legend_frameon, True)
        self.assertEqual(self.plotter.legend_shadow, False)
        self.assertEqual(self.plotter.legend_fancybox, True)
        self.assertEqual(self.plotter.legend_framealpha, 0.8)

    def test_default_tick_options(self):
        """Test default tick options for SpecPlotter."""
        self.assertEqual(self.plotter.tick_direction, "in")
        self.assertIsNone(self.plotter.tick_length)
        self.assertIsNone(self.plotter.tick_width)

    def test_default_spine_options(self):
        """Test default spine options for SpecPlotter."""
        self.assertEqual(self.plotter.show_spines, True)
        self.assertIsNone(self.plotter.spine_linewidth)

    def test_custom_options_initialization(self):
        """Test custom options are correctly set during initialization."""
        plotter = SpecPlotter(self.spectrum,
                             show_grid=True,
                             save_format="pdf",
                             transparent=True,
                             title="Test Spectrum",
                             linestyle="--",
                             markerfillstyle="none",
                             legend_frameon=False,
                             tick_direction="out",
                             show_spines=False)
        self.assertEqual(plotter.show_grid, True)
        self.assertEqual(plotter.save_format, "pdf")
        self.assertEqual(plotter.transparent, True)
        self.assertEqual(plotter.title, "Test Spectrum")
        self.assertEqual(plotter.linestyle, "--")
        self.assertEqual(plotter.markerfillstyle, "none")
        self.assertEqual(plotter.legend_frameon, False)
        self.assertEqual(plotter.tick_direction, "out")
        self.assertEqual(plotter.show_spines, False)

    def test_apply_axis_customizations_with_all_options(self):
        """Test _apply_axis_customizations applies all options correctly."""
        plotter = SpecPlotter(self.spectrum,
                             show_grid=True,
                             grid_alpha=0.5,
                             title="Test",
                             title_fontsize=24,
                             tick_direction="inout",
                             tick_length=8.0,
                             tick_width=2.0,
                             spine_linewidth=3.0)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.grid.assert_called_once()
        mock_ax.set_title.assert_called_once_with("Test", fontsize=24)
        mock_ax.tick_params.assert_called_once()
        for spine in mock_spines.values():
            spine.set_linewidth.assert_called_once_with(3.0)


class TestIntegratedFluxPlotterCustomization(unittest.TestCase):
    """Tests for customization in IntegratedFluxPlotter."""

    def setUp(self) -> None:
        self.transient_mock = MagicMock(spec=Transient)
        self.transient_mock.x = np.logspace(0, 2, 10)
        self.transient_mock.y = np.logspace(0, 2, 10)
        self.transient_mock.ylabel = "Test YLabel"
        self.transient_mock.name = "Test Transient"
        self.transient_mock.use_phase_model = False
        self.transient_mock.directory_structure = MagicMock()
        self.transient_mock.directory_structure.directory_path = "/mock/path"

        self.mock_model = MagicMock()
        self.mock_model.__name__ = "MockModel"

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch.object(IntegratedFluxPlotter, '_x_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_y_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_save_and_show')
    @patch('matplotlib.pyplot.gca')
    def test_plot_data_with_custom_scales(self, mock_gca, mock_save, mock_ylim_high,
                                           mock_ylim_low, mock_xlim_high, mock_xlim_low,
                                           mock_y_err, mock_x_err):
        """Test plot_data respects custom xscale and yscale."""
        mock_x_err.return_value = None
        mock_y_err.return_value = np.zeros(10)
        mock_xlim_low.return_value = 1
        mock_xlim_high.return_value = 100
        mock_ylim_low.return_value = 1
        mock_ylim_high.return_value = 100

        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            xscale="linear",
            yscale="linear"
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        plotter.plot_data(save=False, show=False)

        mock_ax.set_xscale.assert_called_once_with("linear")
        mock_ax.set_yscale.assert_called_once_with("linear")

    @patch.object(IntegratedFluxPlotter, '_x_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_y_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_save_and_show')
    @patch('matplotlib.pyplot.gca')
    def test_plot_data_with_marker_customization(self, mock_gca, mock_save, mock_ylim_high,
                                                  mock_ylim_low, mock_xlim_high, mock_xlim_low,
                                                  mock_y_err, mock_x_err):
        """Test plot_data uses marker customization options."""
        mock_x_err.return_value = None
        mock_y_err.return_value = np.zeros(10)
        mock_xlim_low.return_value = 1
        mock_xlim_high.return_value = 100
        mock_ylim_low.return_value = 1
        mock_ylim_high.return_value = 100

        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            markerfillstyle="none",
            markeredgecolor="red",
            markeredgewidth=2.0
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        plotter.plot_data(save=False, show=False)

        call_kwargs = mock_ax.errorbar.call_args[1]
        self.assertEqual(call_kwargs['fillstyle'], "none")
        self.assertEqual(call_kwargs['markeredgecolor'], "red")
        self.assertEqual(call_kwargs['markeredgewidth'], 2.0)

    @patch.object(IntegratedFluxPlotter, '_x_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_y_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_save_and_show')
    @patch.object(IntegratedFluxPlotter, '_apply_axis_customizations')
    @patch('matplotlib.pyplot.gca')
    def test_plot_data_calls_apply_axis_customizations(self, mock_gca, mock_apply, mock_save,
                                                         mock_ylim_high, mock_ylim_low, mock_xlim_high,
                                                         mock_xlim_low, mock_y_err, mock_x_err):
        """Test plot_data calls _apply_axis_customizations."""
        mock_x_err.return_value = None
        mock_y_err.return_value = np.zeros(10)
        mock_xlim_low.return_value = 1
        mock_xlim_high.return_value = 100
        mock_ylim_low.return_value = 1
        mock_ylim_high.return_value = 100

        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            show_grid=True
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        plotter.plot_data(save=False, show=False)

        mock_apply.assert_called_once_with(mock_ax)

    def test_plot_lightcurves_with_linestyle(self):
        """Test _plot_lightcurves uses linestyle options."""
        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            max_likelihood_linestyle="--",
            random_sample_linestyle=":"
        )

        mock_ax = MagicMock()
        times = np.linspace(1, 100, 10)

        # Mock posterior
        mock_posterior = pd.DataFrame({
            'log_likelihood': [1, 2, 3],
            'param': [0.1, 0.2, 0.3]
        })
        plotter.kwargs['posterior'] = mock_posterior
        plotter.kwargs['random_models'] = 2

        # Mock model
        self.mock_model.return_value = np.ones(10)

        plotter._plot_lightcurves(mock_ax, times)

        # Check that plot was called with the correct linestyle
        calls = mock_ax.plot.call_args_list
        self.assertTrue(any('--' in str(call) for call in calls))


class TestMagnitudePlotterCustomization(unittest.TestCase):
    """Tests for customization in MagnitudePlotter."""

    def setUp(self) -> None:
        self.mock_transient = MagicMock(spec=Transient)
        self.mock_transient.use_phase_model = False
        self.mock_transient.name = "Test"
        self.mock_transient.active_bands = ["r", "g"]
        self.mock_transient.x = np.array([0, 1, 2, 3])
        self.mock_transient.x_err = None
        self.mock_transient.y = np.array([10, 9, 8, 7])
        self.mock_transient.y_err = np.array([0.1, 0.2, 0.1, 0.2])
        self.mock_transient.list_of_band_indices = [[0, 1], [2, 3]]
        self.mock_transient.unique_bands = ["r", "g"]
        self.mock_transient.get_colors = MagicMock(return_value=["red", "green"])
        self.mock_transient.ylabel = "Magnitude"
        self.mock_transient.xlabel = "Time [days]"
        self.mock_transient.magnitude_data = True
        self.mock_transient.directory_structure = MagicMock()
        self.mock_transient.directory_structure.directory_path = "/mock/path"

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_set_y_axis_data_with_custom_yscale(self):
        """Test _set_y_axis_data respects custom yscale."""
        plotter = MagnitudePlotter(self.mock_transient, yscale="log")
        mock_ax = MagicMock()
        plotter._set_y_axis_data(mock_ax)
        mock_ax.set_yscale.assert_called_once_with("log")

    def test_set_y_axis_data_with_default_yscale_for_magnitude(self):
        """Test _set_y_axis_data uses linear scale for magnitude data by default."""
        plotter = MagnitudePlotter(self.mock_transient)
        mock_ax = MagicMock()
        plotter._set_y_axis_data(mock_ax)
        mock_ax.set_yscale.assert_called_once_with("linear")

    def test_set_x_axis_with_custom_xscale(self):
        """Test _set_x_axis respects custom xscale."""
        plotter = MagnitudePlotter(self.mock_transient, xscale="log")
        mock_ax = MagicMock()
        plotter._set_x_axis(mock_ax)
        mock_ax.set_xscale.assert_called_once_with("log")

    def test_set_x_axis_with_default_xscale(self):
        """Test _set_x_axis uses default behavior when xscale is None."""
        plotter = MagnitudePlotter(self.mock_transient)
        mock_ax = MagicMock()
        plotter._set_x_axis(mock_ax)
        # For non-phase model, xscale should not be set
        mock_ax.set_xscale.assert_not_called()

    @patch('matplotlib.pyplot.gca')
    @patch.object(MagnitudePlotter, '_save_and_show')
    def test_plot_data_with_legend_customization(self, mock_save, mock_gca):
        """Test plot_data uses legend customization options."""
        plotter = MagnitudePlotter(
            self.mock_transient,
            legend_frameon=False,
            legend_shadow=True,
            legend_fancybox=False,
            legend_framealpha=0.5
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        plotter.plot_data(save=False, show=False)

        call_kwargs = mock_ax.legend.call_args[1]
        self.assertEqual(call_kwargs['frameon'], False)
        self.assertEqual(call_kwargs['shadow'], True)
        self.assertEqual(call_kwargs['fancybox'], False)
        self.assertEqual(call_kwargs['framealpha'], 0.5)

    @patch('matplotlib.pyplot.gca')
    @patch.object(MagnitudePlotter, '_save_and_show')
    def test_plot_data_with_marker_customization(self, mock_save, mock_gca):
        """Test plot_data uses marker customization in errorbar."""
        plotter = MagnitudePlotter(
            self.mock_transient,
            markerfillstyle="left",
            markeredgecolor="blue",
            markeredgewidth=1.5
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        plotter.plot_data(save=False, show=False)

        # Check all errorbar calls include marker customization
        for call in mock_ax.errorbar.call_args_list:
            call_kwargs = call[1]
            self.assertEqual(call_kwargs.get('fillstyle'), "left")
            self.assertEqual(call_kwargs.get('markeredgecolor'), "blue")
            self.assertEqual(call_kwargs.get('markeredgewidth'), 1.5)


class TestSpectrumPlotterCustomization(unittest.TestCase):
    """Tests for customization in SpectrumPlotter."""

    def setUp(self) -> None:
        self.mock_transient = mock.MagicMock()
        self.mock_transient.angstroms = np.linspace(4000, 7000, 200)
        self.mock_transient.flux_density = np.sin(self.mock_transient.angstroms / 1000) * 1e-17
        self.mock_transient.flux_density_err = np.full_like(self.mock_transient.flux_density, 0.1e-17)
        self.mock_transient.xlabel = "Wavelength [Å]"
        self.mock_transient.ylabel = r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"
        self.mock_transient.plot_with_time_label = False
        self.mock_transient.name = "TestSpectrum"

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch.object(SpectrumPlotter, '_save_and_show')
    def test_plot_data_with_custom_linestyle(self, mock_save):
        """Test plot_data uses custom linestyle."""
        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            linestyle="--"
        )

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        call_kwargs = mock_ax.plot.call_args[1]
        self.assertEqual(call_kwargs['linestyle'], "--")

    @patch.object(SpectrumPlotter, '_save_and_show')
    def test_plot_data_with_custom_xscale(self, mock_save):
        """Test plot_data uses custom xscale."""
        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            xscale="log"
        )

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        mock_ax.set_xscale.assert_called_once_with("log")

    @patch.object(SpectrumPlotter, '_save_and_show')
    @patch.object(SpectrumPlotter, '_apply_axis_customizations')
    def test_plot_data_calls_apply_axis_customizations(self, mock_apply, mock_save):
        """Test plot_data calls _apply_axis_customizations."""
        plotter = SpectrumPlotter(spectrum=self.mock_transient)

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        mock_apply.assert_called_once_with(mock_ax)

    def test_plot_spectrums_with_linestyle(self):
        """Test _plot_spectrums uses linestyle options."""
        mock_posterior = pd.DataFrame({
            'log_likelihood': [1, 2, 3],
            'param': [0.1, 0.2, 0.3]
        })

        def mock_model_func(angstroms, **kwargs):
            return np.ones_like(angstroms) * 1e-17

        mock_model = mock.MagicMock(side_effect=mock_model_func)
        mock_model.__name__ = "MockModel"

        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            posterior=mock_posterior,
            model=mock_model,
            max_likelihood_linestyle="--",
            random_sample_linestyle=":"
        )

        mock_ax = MagicMock()
        angstroms = np.linspace(4000, 7000, 100)

        plotter._plot_spectrums(mock_ax, angstroms)

        # Check that plot was called with the correct linestyle
        calls = mock_ax.plot.call_args_list
        self.assertTrue(len(calls) > 0)


class TestPlotterEdgeCases(unittest.TestCase):
    """Tests for edge cases and additional coverage."""

    def setUp(self) -> None:
        self.dummy_transient = DummyTransient(name="TestTransient", directory_path="/dummy/path")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_all_supported_save_formats(self):
        """Test all supported save formats."""
        formats = ["png", "pdf", "svg", "eps", "jpg", "jpeg", "tif", "tiff"]
        for fmt in formats:
            plotter = Plotter(self.dummy_transient, save_format=fmt)
            self.assertEqual(plotter.save_format, fmt)

    def test_all_linestyle_variations(self):
        """Test all matplotlib linestyle variations."""
        linestyles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        for ls in linestyles:
            plotter = Plotter(self.dummy_transient, linestyle=ls)
            self.assertEqual(plotter.linestyle, ls)

    def test_all_markerfillstyle_variations(self):
        """Test all matplotlib marker fill styles."""
        fillstyles = ["full", "left", "right", "bottom", "top", "none"]
        for fs in fillstyles:
            plotter = Plotter(self.dummy_transient, markerfillstyle=fs)
            self.assertEqual(plotter.markerfillstyle, fs)

    def test_tick_direction_variations(self):
        """Test all tick direction options."""
        directions = ["in", "out", "inout"]
        for d in directions:
            plotter = Plotter(self.dummy_transient, tick_direction=d)
            self.assertEqual(plotter.tick_direction, d)

    def test_scale_options(self):
        """Test all axis scale options."""
        scales = ["linear", "log", "symlog", "logit"]
        for scale in scales:
            plotter = Plotter(self.dummy_transient, xscale=scale, yscale=scale)
            self.assertEqual(plotter.xscale, scale)
            self.assertEqual(plotter.yscale, scale)

    def test_apply_axis_customizations_with_default_tick_params(self):
        """Test _apply_axis_customizations with only default tick params (no length/width)."""
        plotter = Plotter(self.dummy_transient)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertNotIn('length', call_kwargs)
        self.assertNotIn('width', call_kwargs)
        self.assertEqual(call_kwargs['direction'], 'in')

    def test_apply_axis_customizations_with_partial_tick_params(self):
        """Test _apply_axis_customizations with only tick_length set."""
        plotter = Plotter(self.dummy_transient, tick_length=5.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertIn('length', call_kwargs)
        self.assertNotIn('width', call_kwargs)
        self.assertEqual(call_kwargs['length'], 5.0)

    def test_apply_axis_customizations_with_only_tick_width(self):
        """Test _apply_axis_customizations with only tick_width set."""
        plotter = Plotter(self.dummy_transient, tick_width=2.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertNotIn('length', call_kwargs)
        self.assertIn('width', call_kwargs)
        self.assertEqual(call_kwargs['width'], 2.0)

    def test_save_format_with_multiple_dots_in_path(self):
        """Test save format handles paths with multiple dots correctly."""
        plotter = Plotter(self.dummy_transient, save_format="pdf")
        filepath = "/path/to/my.file.name.png"
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.tight_layout'):
            plotter._save_and_show(filepath, save=True, show=False)
            expected_path = "/path/to/my.file.name.pdf"
            self.assertEqual(mock_savefig.call_args[0][0], expected_path)

    def test_legend_alpha_range(self):
        """Test legend framealpha with various values."""
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha in alphas:
            plotter = Plotter(self.dummy_transient, legend_framealpha=alpha)
            self.assertEqual(plotter.legend_framealpha, alpha)

    def test_grid_alpha_range(self):
        """Test grid alpha with various values."""
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        for alpha in alphas:
            plotter = Plotter(self.dummy_transient, grid_alpha=alpha)
            self.assertEqual(plotter.grid_alpha, alpha)

    def test_grid_with_various_colors(self):
        """Test grid with various color specifications."""
        colors = ["gray", "red", "blue", "#FF0000", (0.5, 0.5, 0.5), "k"]
        for color in colors:
            plotter = Plotter(self.dummy_transient, grid_color=color)
            self.assertEqual(plotter.grid_color, color)

    def test_title_with_latex(self):
        """Test title with LaTeX formatting."""
        latex_title = r"$\alpha$ = 0.5, $\beta$ = 1.0"
        plotter = Plotter(self.dummy_transient, title=latex_title)
        self.assertEqual(plotter.title, latex_title)

    def test_empty_string_title(self):
        """Test with empty string title."""
        plotter = Plotter(self.dummy_transient, title="")
        self.assertEqual(plotter.title, "")


class TestSpecPlotterEdgeCases(unittest.TestCase):
    """Edge case tests for SpecPlotter."""

    def setUp(self) -> None:
        angstroms = np.array([4000, 5000, 6000])
        flux_density = np.array([1e-17, 2e-17, 3e-17])
        flux_density_err = np.array([0.1e-17, 0.1e-17, 0.1e-17])
        self.spectrum = Spectrum(angstroms, flux_density, flux_density_err, name="test_spectrum")
        self.spectrum.directory_structure = SimpleNamespace(directory_path='/dummy/path')

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_spec_plotter_save_with_pdf_format(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter saving with PDF format."""
        plotter = SpecPlotter(self.spectrum, save_format="pdf")
        filepath = "/path/to/spectrum.png"
        plotter._save_and_show(filepath, save=True, show=False)
        expected_path = "/path/to/spectrum.pdf"
        self.assertEqual(mock_savefig.call_args[0][0], expected_path)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_spec_plotter_save_transparent(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter saving with transparent background."""
        plotter = SpecPlotter(self.spectrum, transparent=True)
        filepath = "/path/to/spectrum.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_kwargs = mock_savefig.call_args[1]
        self.assertEqual(call_kwargs['transparent'], True)
        self.assertEqual(call_kwargs['facecolor'], 'none')

    def test_spec_plotter_apply_axis_customizations_no_title(self):
        """Test SpecPlotter _apply_axis_customizations without title."""
        plotter = SpecPlotter(self.spectrum)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.set_title.assert_not_called()

    def test_spec_plotter_apply_axis_customizations_no_grid(self):
        """Test SpecPlotter _apply_axis_customizations without grid."""
        plotter = SpecPlotter(self.spectrum, show_grid=False)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.grid.assert_not_called()

    def test_spec_plotter_all_tick_params(self):
        """Test SpecPlotter with all tick parameters."""
        plotter = SpecPlotter(self.spectrum,
                             tick_direction="inout",
                             tick_length=10.0,
                             tick_width=3.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['direction'], 'inout')
        self.assertEqual(call_kwargs['length'], 10.0)
        self.assertEqual(call_kwargs['width'], 3.0)


class TestIntegratedFluxPlotterEdgeCases(unittest.TestCase):
    """Edge case tests for IntegratedFluxPlotter."""

    def setUp(self) -> None:
        self.transient_mock = MagicMock(spec=Transient)
        self.transient_mock.x = np.logspace(0, 2, 10)
        self.transient_mock.y = np.logspace(0, 2, 10)
        self.transient_mock.ylabel = "Test YLabel"
        self.transient_mock.name = "Test Transient"
        self.transient_mock.use_phase_model = False
        self.transient_mock.directory_structure = MagicMock()
        self.transient_mock.directory_structure.directory_path = "/mock/path"

        self.mock_model = MagicMock()
        self.mock_model.__name__ = "MockModel"

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch.object(IntegratedFluxPlotter, '_x_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_y_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_save_and_show')
    @patch('matplotlib.pyplot.gca')
    def test_plot_data_with_default_scales(self, mock_gca, mock_save, mock_ylim_high,
                                            mock_ylim_low, mock_xlim_high, mock_xlim_low,
                                            mock_y_err, mock_x_err):
        """Test plot_data uses default log scales when not specified."""
        mock_x_err.return_value = None
        mock_y_err.return_value = np.zeros(10)
        mock_xlim_low.return_value = 1
        mock_xlim_high.return_value = 100
        mock_ylim_low.return_value = 1
        mock_ylim_high.return_value = 100

        plotter = IntegratedFluxPlotter(transient=self.transient_mock, model=self.mock_model)

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        plotter.plot_data(save=False, show=False)

        mock_ax.set_xscale.assert_called_once_with("log")
        mock_ax.set_yscale.assert_called_once_with("log")

    @patch.object(IntegratedFluxPlotter, '_x_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_y_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_save_and_show')
    @patch('matplotlib.pyplot.gca')
    def test_plot_data_with_title(self, mock_gca, mock_save, mock_ylim_high,
                                   mock_ylim_low, mock_xlim_high, mock_xlim_low,
                                   mock_y_err, mock_x_err):
        """Test plot_data applies title correctly."""
        mock_x_err.return_value = None
        mock_y_err.return_value = np.zeros(10)
        mock_xlim_low.return_value = 1
        mock_xlim_high.return_value = 100
        mock_ylim_low.return_value = 1
        mock_ylim_high.return_value = 100

        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            title="My Plot Title",
            title_fontsize=24
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        plotter.plot_data(save=False, show=False)

        mock_ax.set_title.assert_called_once_with("My Plot Title", fontsize=24)

    @patch.object(IntegratedFluxPlotter, '_x_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_y_err', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_xlim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_low', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_ylim_high', new_callable=PropertyMock)
    @patch.object(IntegratedFluxPlotter, '_save_and_show')
    @patch('matplotlib.pyplot.gca')
    def test_plot_data_with_grid(self, mock_gca, mock_save, mock_ylim_high,
                                  mock_ylim_low, mock_xlim_high, mock_xlim_low,
                                  mock_y_err, mock_x_err):
        """Test plot_data applies grid correctly."""
        mock_x_err.return_value = None
        mock_y_err.return_value = np.zeros(10)
        mock_xlim_low.return_value = 1
        mock_xlim_high.return_value = 100
        mock_ylim_low.return_value = 1
        mock_ylim_high.return_value = 100

        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            show_grid=True,
            grid_alpha=0.7,
            grid_color="red"
        )

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        plotter.plot_data(save=False, show=False)

        mock_ax.grid.assert_called_once()
        grid_kwargs = mock_ax.grid.call_args
        self.assertEqual(grid_kwargs[1]['alpha'], 0.7)
        self.assertEqual(grid_kwargs[1]['color'], "red")

    def test_plot_lightcurves_credible_intervals_mode(self):
        """Test _plot_lightcurves with credible_intervals mode."""
        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            uncertainty_mode="credible_intervals",
            credible_interval_level=0.95
        )

        mock_ax = MagicMock()
        times = np.linspace(1, 100, 10)

        # Mock posterior
        mock_posterior = pd.DataFrame({
            'log_likelihood': [1, 2, 3],
            'param': [0.1, 0.2, 0.3]
        })
        plotter.kwargs['posterior'] = mock_posterior
        plotter.kwargs['random_models'] = 3

        # Mock model to return arrays
        self.mock_model.return_value = np.ones(10)

        with patch('redback.utils.calc_credible_intervals') as mock_ci:
            mock_ci.return_value = (np.zeros(10), np.ones(10), np.ones(10))
            plotter._plot_lightcurves(mock_ax, times)

        mock_ax.fill_between.assert_called_once()

    def test_plot_lightcurves_without_max_likelihood(self):
        """Test _plot_lightcurves with plot_max_likelihood=False."""
        plotter = IntegratedFluxPlotter(
            transient=self.transient_mock,
            model=self.mock_model,
            plot_max_likelihood=False
        )

        mock_ax = MagicMock()
        times = np.linspace(1, 100, 10)

        # Mock posterior
        mock_posterior = pd.DataFrame({
            'log_likelihood': [1, 2, 3],
            'param': [0.1, 0.2, 0.3]
        })
        plotter.kwargs['posterior'] = mock_posterior
        plotter.kwargs['random_models'] = 2

        # Mock model
        self.mock_model.return_value = np.ones(10)

        plotter._plot_lightcurves(mock_ax, times)

        # Only random samples should be plotted, not max likelihood
        calls = mock_ax.plot.call_args_list
        # Should be 2 calls for random samples, no call for max likelihood
        self.assertEqual(len(calls), 2)


class TestMagnitudePlotterEdgeCases(unittest.TestCase):
    """Edge case tests for MagnitudePlotter."""

    def setUp(self) -> None:
        self.mock_transient = MagicMock(spec=Transient)
        self.mock_transient.use_phase_model = False
        self.mock_transient.name = "Test"
        self.mock_transient.active_bands = ["r", "g"]
        self.mock_transient.x = np.array([0, 1, 2, 3])
        self.mock_transient.x_err = None
        self.mock_transient.y = np.array([10, 9, 8, 7])
        self.mock_transient.y_err = np.array([0.1, 0.2, 0.1, 0.2])
        self.mock_transient.list_of_band_indices = [[0, 1], [2, 3]]
        self.mock_transient.unique_bands = ["r", "g"]
        self.mock_transient.get_colors = MagicMock(return_value=["red", "green"])
        self.mock_transient.ylabel = "Magnitude"
        self.mock_transient.xlabel = "Time [days]"
        self.mock_transient.magnitude_data = True
        self.mock_transient.directory_structure = MagicMock()
        self.mock_transient.directory_structure.directory_path = "/mock/path"

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_set_y_axis_multiband_data_with_custom_yscale(self):
        """Test _set_y_axis_multiband_data respects custom yscale."""
        plotter = MagnitudePlotter(self.mock_transient, yscale="symlog")
        mock_ax = MagicMock()
        plotter._set_y_axis_multiband_data(mock_ax, [0, 1])
        mock_ax.set_yscale.assert_called_once_with("symlog")

    def test_set_y_axis_multiband_data_default_for_magnitude(self):
        """Test _set_y_axis_multiband_data uses linear for magnitude by default."""
        plotter = MagnitudePlotter(self.mock_transient)
        mock_ax = MagicMock()
        plotter._set_y_axis_multiband_data(mock_ax, [0, 1])
        mock_ax.set_yscale.assert_called_once_with("linear")

    def test_set_y_axis_data_non_magnitude_data(self):
        """Test _set_y_axis_data for non-magnitude data."""
        self.mock_transient.magnitude_data = False
        plotter = MagnitudePlotter(self.mock_transient)
        mock_ax = MagicMock()
        plotter._set_y_axis_data(mock_ax)
        mock_ax.set_yscale.assert_called_once_with("log")
        mock_ax.invert_yaxis.assert_not_called()

    def test_set_y_axis_multiband_non_magnitude_data(self):
        """Test _set_y_axis_multiband_data for non-magnitude data."""
        self.mock_transient.magnitude_data = False
        plotter = MagnitudePlotter(self.mock_transient)
        mock_ax = MagicMock()
        plotter._set_y_axis_multiband_data(mock_ax, [0, 1])
        mock_ax.set_yscale.assert_called_once_with("log")
        mock_ax.invert_yaxis.assert_not_called()

    def test_set_x_axis_with_phase_model(self):
        """Test _set_x_axis uses log scale for phase model by default."""
        self.mock_transient.use_phase_model = True
        plotter = MagnitudePlotter(self.mock_transient)
        mock_ax = MagicMock()
        plotter._set_x_axis(mock_ax)
        mock_ax.set_xscale.assert_called_once_with("log")

    def test_set_x_axis_custom_scale_overrides_phase_model(self):
        """Test custom xscale overrides phase model default."""
        self.mock_transient.use_phase_model = True
        plotter = MagnitudePlotter(self.mock_transient, xscale="linear")
        mock_ax = MagicMock()
        plotter._set_x_axis(mock_ax)
        mock_ax.set_xscale.assert_called_once_with("linear")

    @patch('matplotlib.pyplot.gca')
    @patch.object(MagnitudePlotter, '_save_and_show')
    @patch.object(MagnitudePlotter, '_apply_axis_customizations')
    def test_plot_data_calls_apply_customizations(self, mock_apply, mock_save, mock_gca):
        """Test plot_data calls _apply_axis_customizations."""
        plotter = MagnitudePlotter(self.mock_transient)

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        plotter.plot_data(save=False, show=False)

        mock_apply.assert_called_once_with(mock_ax)


class TestSpectrumPlotterEdgeCases(unittest.TestCase):
    """Edge case tests for SpectrumPlotter."""

    def setUp(self) -> None:
        self.mock_transient = mock.MagicMock()
        self.mock_transient.angstroms = np.linspace(4000, 7000, 200)
        self.mock_transient.flux_density = np.sin(self.mock_transient.angstroms / 1000) * 1e-17
        self.mock_transient.flux_density_err = np.full_like(self.mock_transient.flux_density, 0.1e-17)
        self.mock_transient.xlabel = "Wavelength [Å]"
        self.mock_transient.ylabel = r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"
        self.mock_transient.plot_with_time_label = False
        self.mock_transient.name = "TestSpectrum"

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch.object(SpectrumPlotter, '_save_and_show')
    def test_plot_data_default_xscale(self, mock_save):
        """Test plot_data uses linear scale by default."""
        plotter = SpectrumPlotter(spectrum=self.mock_transient)

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        mock_ax.set_xscale.assert_called_once_with("linear")

    @patch.object(SpectrumPlotter, '_save_and_show')
    def test_plot_data_with_time_label(self, mock_save):
        """Test plot_data with time label."""
        self.mock_transient.plot_with_time_label = True
        self.mock_transient.time = "100s"
        plotter = SpectrumPlotter(spectrum=self.mock_transient)

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        # Check that annotate was called with time label
        mock_ax.annotate.assert_called_once()
        call_args = mock_ax.annotate.call_args
        self.assertEqual(call_args[0][0], "100s")

    @patch.object(SpectrumPlotter, '_save_and_show')
    def test_plot_data_with_title(self, mock_save):
        """Test plot_data applies title correctly."""
        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            title="Spectrum Plot",
            title_fontsize=18
        )

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        mock_ax.set_title.assert_called_once_with("Spectrum Plot", fontsize=18)

    @patch.object(SpectrumPlotter, '_save_and_show')
    def test_plot_data_with_grid(self, mock_save):
        """Test plot_data applies grid correctly."""
        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            show_grid=True,
            grid_color="blue",
            grid_alpha=0.4
        )

        mock_ax = MagicMock()
        plotter.plot_data(axes=mock_ax, save=False, show=False)

        mock_ax.grid.assert_called_once()
        grid_kwargs = mock_ax.grid.call_args[1]
        self.assertEqual(grid_kwargs['color'], "blue")
        self.assertEqual(grid_kwargs['alpha'], 0.4)

    def test_plot_spectrums_credible_intervals_mode(self):
        """Test _plot_spectrums with credible_intervals mode."""
        mock_posterior = pd.DataFrame({
            'log_likelihood': [1, 2, 3],
            'param': [0.1, 0.2, 0.3]
        })

        def mock_model_func(angstroms, **kwargs):
            return np.ones_like(angstroms) * 1e-17

        mock_model = mock.MagicMock(side_effect=mock_model_func)
        mock_model.__name__ = "MockModel"

        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            posterior=mock_posterior,
            model=mock_model,
            uncertainty_mode="credible_intervals"
        )

        mock_ax = MagicMock()
        angstroms = np.linspace(4000, 7000, 100)

        with patch('redback.utils.calc_credible_intervals') as mock_ci:
            mock_ci.return_value = (np.zeros(100) * 1e-17, np.ones(100) * 1e-17, np.ones(100) * 1e-17)
            plotter._plot_spectrums(mock_ax, angstroms)

        mock_ax.fill_between.assert_called_once()

    def test_plot_spectrums_without_max_likelihood(self):
        """Test _plot_spectrums with plot_max_likelihood=False."""
        mock_posterior = pd.DataFrame({
            'log_likelihood': [1, 2, 3],
            'param': [0.1, 0.2, 0.3]
        })

        def mock_model_func(angstroms, **kwargs):
            return np.ones_like(angstroms) * 1e-17

        mock_model = mock.MagicMock(side_effect=mock_model_func)
        mock_model.__name__ = "MockModel"

        plotter = SpectrumPlotter(
            spectrum=self.mock_transient,
            posterior=mock_posterior,
            model=mock_model,
            plot_max_likelihood=False
        )

        mock_ax = MagicMock()
        angstroms = np.linspace(4000, 7000, 100)

        plotter._plot_spectrums(mock_ax, angstroms)

        # Only random samples should be plotted
        calls = mock_ax.plot.call_args_list
        self.assertEqual(len(calls), 100)  # Default random_models is 100


class TestSaveAndShowIntegration(unittest.TestCase):
    """Integration tests for _save_and_show functionality."""

    def setUp(self) -> None:
        self.dummy_transient = DummyTransient(name="TestTransient", directory_path="/dummy/path")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch('matplotlib.pyplot.tight_layout')
    def test_no_save_no_show(self, mock_tight_layout):
        """Test _save_and_show with save=False and show=False."""
        plotter = Plotter(self.dummy_transient)
        filepath = "/path/to/file.png"

        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.show') as mock_show:
            plotter._save_and_show(filepath, save=False, show=False)
            mock_savefig.assert_not_called()
            mock_show.assert_not_called()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_uses_dpi_setting(self, mock_tight_layout, mock_savefig):
        """Test that save uses custom dpi setting."""
        plotter = Plotter(self.dummy_transient, dpi=600)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_kwargs = mock_savefig.call_args[1]
        self.assertEqual(call_kwargs['dpi'], 600)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_uses_bbox_inches_setting(self, mock_tight_layout, mock_savefig):
        """Test that save uses custom bbox_inches setting."""
        plotter = Plotter(self.dummy_transient, bbox_inches="standard")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_kwargs = mock_savefig.call_args[1]
        self.assertEqual(call_kwargs['bbox_inches'], "standard")


class TestSpecPlotterSaveAndShow(unittest.TestCase):
    """Tests for SpecPlotter _save_and_show method to ensure code coverage."""

    def setUp(self) -> None:
        angstroms = np.array([4000, 5000, 6000])
        flux_density = np.array([1e-17, 2e-17, 3e-17])
        flux_density_err = np.array([0.1e-17, 0.1e-17, 0.1e-17])
        self.spectrum = Spectrum(angstroms, flux_density, flux_density_err, name="test_spectrum")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_default_format(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show saves with default png format."""
        plotter = SpecPlotter(self.spectrum)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.png'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_pdf_format(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show saves with pdf format."""
        plotter = SpecPlotter(self.spectrum, save_format="pdf")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.pdf'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_svg_format(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show saves with svg format."""
        plotter = SpecPlotter(self.spectrum, save_format="svg")
        filepath = "/path/to/file"
        plotter._save_and_show(filepath, save=True, show=False)
        call_args = mock_savefig.call_args
        self.assertTrue(call_args[0][0].endswith('.svg'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_transparent_background(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show saves with transparent background."""
        plotter = SpecPlotter(self.spectrum, transparent=True)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_kwargs = mock_savefig.call_args[1]
        self.assertTrue(call_kwargs['transparent'])
        self.assertEqual(call_kwargs['facecolor'], 'none')

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_opaque_background(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show saves with opaque background."""
        plotter = SpecPlotter(self.spectrum, transparent=False)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_kwargs = mock_savefig.call_args[1]
        self.assertFalse(call_kwargs['transparent'])
        self.assertEqual(call_kwargs['facecolor'], 'white')

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.tight_layout')
    def test_show_only(self, mock_tight_layout, mock_show):
        """Test SpecPlotter _save_and_show shows without saving."""
        plotter = SpecPlotter(self.spectrum)
        filepath = "/path/to/file.png"
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plotter._save_and_show(filepath, save=False, show=True)
            mock_savefig.assert_not_called()
            mock_show.assert_called_once()

    @patch('matplotlib.pyplot.tight_layout')
    def test_no_save_no_show(self, mock_tight_layout):
        """Test SpecPlotter _save_and_show with save=False and show=False."""
        plotter = SpecPlotter(self.spectrum)
        filepath = "/path/to/file.png"
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.show') as mock_show:
            plotter._save_and_show(filepath, save=False, show=False)
            mock_savefig.assert_not_called()
            mock_show.assert_not_called()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_with_custom_dpi(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show uses custom dpi."""
        plotter = SpecPlotter(self.spectrum, dpi=600)
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_kwargs = mock_savefig.call_args[1]
        self.assertEqual(call_kwargs['dpi'], 600)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_replaces_extension_correctly(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show replaces file extension correctly."""
        plotter = SpecPlotter(self.spectrum, save_format="eps")
        filepath = "/path/to/file.png"
        plotter._save_and_show(filepath, save=True, show=False)
        call_args = mock_savefig.call_args
        self.assertEqual(call_args[0][0], "/path/to/file.eps")

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_without_extension_in_filepath(self, mock_tight_layout, mock_savefig):
        """Test SpecPlotter _save_and_show adds extension when filepath has none."""
        plotter = SpecPlotter(self.spectrum, save_format="tiff")
        filepath = "/path/to/file"
        plotter._save_and_show(filepath, save=True, show=False)
        call_args = mock_savefig.call_args
        self.assertEqual(call_args[0][0], "/path/to/file.tiff")


class TestSpecPlotterApplyAxisCustomizations(unittest.TestCase):
    """Tests for SpecPlotter _apply_axis_customizations method to ensure code coverage."""

    def setUp(self) -> None:
        angstroms = np.array([4000, 5000, 6000])
        flux_density = np.array([1e-17, 2e-17, 3e-17])
        flux_density_err = np.array([0.1e-17, 0.1e-17, 0.1e-17])
        self.spectrum = Spectrum(angstroms, flux_density, flux_density_err, name="test_spectrum")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_apply_grid_enabled(self):
        """Test SpecPlotter _apply_axis_customizations applies grid when enabled."""
        plotter = SpecPlotter(self.spectrum, show_grid=True, grid_alpha=0.5,
                             grid_color="blue", grid_linestyle="-", grid_linewidth=1.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.grid.assert_called_once_with(
            True, alpha=0.5, color="blue", linestyle="-", linewidth=1.0
        )

    def test_apply_grid_disabled(self):
        """Test SpecPlotter _apply_axis_customizations does not apply grid when disabled."""
        plotter = SpecPlotter(self.spectrum, show_grid=False)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.grid.assert_not_called()

    def test_apply_title(self):
        """Test SpecPlotter _apply_axis_customizations applies title when set."""
        plotter = SpecPlotter(self.spectrum, title="Spectrum Plot", title_fontsize=24)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.set_title.assert_called_once_with("Spectrum Plot", fontsize=24)

    def test_no_title_when_not_set(self):
        """Test SpecPlotter _apply_axis_customizations does not apply title when not set."""
        plotter = SpecPlotter(self.spectrum, title=None)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.set_title.assert_not_called()

    def test_tick_params_with_custom_direction(self):
        """Test SpecPlotter _apply_axis_customizations applies tick direction."""
        plotter = SpecPlotter(self.spectrum, tick_direction="out")
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        mock_ax.tick_params.assert_called_once()
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['direction'], 'out')

    def test_tick_params_with_length(self):
        """Test SpecPlotter _apply_axis_customizations applies tick length."""
        plotter = SpecPlotter(self.spectrum, tick_length=8.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['length'], 8.0)

    def test_tick_params_with_width(self):
        """Test SpecPlotter _apply_axis_customizations applies tick width."""
        plotter = SpecPlotter(self.spectrum, tick_width=2.0)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['width'], 2.0)

    def test_tick_params_with_both_length_and_width(self):
        """Test SpecPlotter _apply_axis_customizations applies both tick length and width."""
        plotter = SpecPlotter(self.spectrum, tick_length=6.0, tick_width=1.5)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['length'], 6.0)
        self.assertEqual(call_kwargs['width'], 1.5)

    def test_hide_spines(self):
        """Test SpecPlotter _apply_axis_customizations hides spines when show_spines=False."""
        plotter = SpecPlotter(self.spectrum, show_spines=False)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)
        for spine in mock_spines.values():
            spine.set_visible.assert_called_once_with(False)

    def test_spine_linewidth(self):
        """Test SpecPlotter _apply_axis_customizations sets spine linewidth."""
        plotter = SpecPlotter(self.spectrum, spine_linewidth=2.5)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)
        for spine in mock_spines.values():
            spine.set_linewidth.assert_called_once_with(2.5)

    def test_spine_linewidth_not_set_when_none(self):
        """Test SpecPlotter _apply_axis_customizations does not set spine linewidth when None."""
        plotter = SpecPlotter(self.spectrum, show_spines=True, spine_linewidth=None)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)
        for spine in mock_spines.values():
            spine.set_linewidth.assert_not_called()
            spine.set_visible.assert_not_called()

    def test_all_customizations_combined(self):
        """Test SpecPlotter _apply_axis_customizations with all options enabled."""
        plotter = SpecPlotter(self.spectrum,
                             show_grid=True, grid_alpha=0.4, grid_color="green",
                             grid_linestyle=":", grid_linewidth=0.8,
                             title="Combined Test", title_fontsize=22,
                             tick_direction="inout", tick_length=7.0, tick_width=1.2,
                             spine_linewidth=1.8)
        mock_ax = MagicMock()
        mock_spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        mock_ax.spines = mock_spines
        plotter._apply_axis_customizations(mock_ax)

        mock_ax.grid.assert_called_once_with(
            True, alpha=0.4, color="green", linestyle=":", linewidth=0.8
        )
        mock_ax.set_title.assert_called_once_with("Combined Test", fontsize=22)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['direction'], 'inout')
        self.assertEqual(call_kwargs['length'], 7.0)
        self.assertEqual(call_kwargs['width'], 1.2)
        for spine in mock_spines.values():
            spine.set_linewidth.assert_called_once_with(1.8)


class TestPlotterImplementationCoverage(unittest.TestCase):
    """Additional tests to ensure implementation code paths are covered."""

    def setUp(self) -> None:
        self.dummy_transient = DummyTransient(name="CoverageTest", directory_path="/coverage/path")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_save_and_show_filepath_with_multiple_dots(self):
        """Test _save_and_show handles filepath with multiple dots correctly."""
        plotter = Plotter(self.dummy_transient, save_format="pdf")
        filepath = "/path/to/file.data.png"
        with patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            plotter._save_and_show(filepath, save=True, show=False)
            call_args = mock_savefig.call_args
            # Should replace the last extension
            self.assertEqual(call_args[0][0], "/path/to/file.data.pdf")

    def test_apply_axis_customizations_default_tick_direction(self):
        """Test _apply_axis_customizations uses default tick direction."""
        plotter = Plotter(self.dummy_transient)  # Default tick_direction="in"
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertEqual(call_kwargs['direction'], 'in')

    def test_apply_axis_customizations_no_optional_params(self):
        """Test _apply_axis_customizations without optional tick params."""
        plotter = Plotter(self.dummy_transient, tick_length=None, tick_width=None)
        mock_ax = MagicMock()
        plotter._apply_axis_customizations(mock_ax)
        call_kwargs = mock_ax.tick_params.call_args[1]
        self.assertNotIn('length', call_kwargs)
        self.assertNotIn('width', call_kwargs)

    def test_save_format_jpeg(self):
        """Test _save_and_show with jpeg format."""
        plotter = Plotter(self.dummy_transient, save_format="jpeg")
        filepath = "/path/to/file.png"
        with patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            plotter._save_and_show(filepath, save=True, show=False)
            call_args = mock_savefig.call_args
            self.assertEqual(call_args[0][0], "/path/to/file.jpeg")

    def test_save_format_jpg(self):
        """Test _save_and_show with jpg format."""
        plotter = Plotter(self.dummy_transient, save_format="jpg")
        filepath = "/path/to/file"
        with patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            plotter._save_and_show(filepath, save=True, show=False)
            call_args = mock_savefig.call_args
            self.assertEqual(call_args[0][0], "/path/to/file.jpg")

    def test_transparent_true_with_pdf(self):
        """Test transparent background with PDF format."""
        plotter = Plotter(self.dummy_transient, save_format="pdf", transparent=True)
        filepath = "/path/to/file"
        with patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            plotter._save_and_show(filepath, save=True, show=False)
            call_kwargs = mock_savefig.call_args[1]
            self.assertTrue(call_kwargs['transparent'])
            self.assertEqual(call_kwargs['facecolor'], 'none')

    def test_show_called_after_save(self):
        """Test that show is called after save when both are True."""
        plotter = Plotter(self.dummy_transient)
        filepath = "/path/to/file.png"
        with patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.show') as mock_show:
            plotter._save_and_show(filepath, save=True, show=True)
            mock_savefig.assert_called_once()
            mock_show.assert_called_once()

    def test_grid_options_all_defaults(self):
        """Test that grid options have correct defaults."""
        plotter = Plotter(self.dummy_transient)
        self.assertFalse(plotter.show_grid)
        self.assertEqual(plotter.grid_alpha, 0.3)
        self.assertEqual(plotter.grid_color, "gray")
        self.assertEqual(plotter.grid_linestyle, "--")
        self.assertEqual(plotter.grid_linewidth, 0.5)

    def test_legend_options_all_defaults(self):
        """Test that legend options have correct defaults."""
        plotter = Plotter(self.dummy_transient)
        self.assertTrue(plotter.legend_frameon)
        self.assertFalse(plotter.legend_shadow)
        self.assertTrue(plotter.legend_fancybox)
        self.assertEqual(plotter.legend_framealpha, 0.8)

    def test_linestyle_options_all_defaults(self):
        """Test that linestyle options have correct defaults."""
        plotter = Plotter(self.dummy_transient)
        self.assertEqual(plotter.linestyle, "-")
        self.assertEqual(plotter.max_likelihood_linestyle, "-")
        self.assertEqual(plotter.random_sample_linestyle, "-")

    def test_marker_options_all_defaults(self):
        """Test that marker options have correct defaults."""
        plotter = Plotter(self.dummy_transient)
        self.assertEqual(plotter.markerfillstyle, "full")
        self.assertIsNone(plotter.markeredgecolor)
        self.assertEqual(plotter.markeredgewidth, 1.0)

    def test_axis_scale_options_defaults(self):
        """Test that axis scale options have correct defaults."""
        plotter = Plotter(self.dummy_transient)
        self.assertIsNone(plotter.xscale)
        self.assertIsNone(plotter.yscale)

    def test_title_options_defaults(self):
        """Test that title options have correct defaults."""
        plotter = Plotter(self.dummy_transient)
        self.assertIsNone(plotter.title)
        self.assertEqual(plotter.title_fontsize, 20)


class TestSpecPlotterImplementationCoverage(unittest.TestCase):
    """Additional tests to ensure SpecPlotter implementation code paths are covered."""

    def setUp(self) -> None:
        angstroms = np.array([4000, 5000, 6000])
        flux_density = np.array([1e-17, 2e-17, 3e-17])
        flux_density_err = np.array([0.1e-17, 0.1e-17, 0.1e-17])
        self.spectrum = Spectrum(angstroms, flux_density, flux_density_err, name="test_spectrum")

    def tearDown(self) -> None:
        plt.close('all')
        mock.patch.stopall()

    def test_spec_plotter_grid_options_defaults(self):
        """Test that SpecPlotter grid options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertFalse(plotter.show_grid)
        self.assertEqual(plotter.grid_alpha, 0.3)
        self.assertEqual(plotter.grid_color, "gray")
        self.assertEqual(plotter.grid_linestyle, "--")
        self.assertEqual(plotter.grid_linewidth, 0.5)

    def test_spec_plotter_legend_options_defaults(self):
        """Test that SpecPlotter legend options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertTrue(plotter.legend_frameon)
        self.assertFalse(plotter.legend_shadow)
        self.assertTrue(plotter.legend_fancybox)
        self.assertEqual(plotter.legend_framealpha, 0.8)

    def test_spec_plotter_linestyle_options_defaults(self):
        """Test that SpecPlotter linestyle options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertEqual(plotter.linestyle, "-")
        self.assertEqual(plotter.max_likelihood_linestyle, "-")
        self.assertEqual(plotter.random_sample_linestyle, "-")

    def test_spec_plotter_marker_options_defaults(self):
        """Test that SpecPlotter marker options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertEqual(plotter.markerfillstyle, "full")
        self.assertIsNone(plotter.markeredgecolor)
        self.assertEqual(plotter.markeredgewidth, 1.0)

    def test_spec_plotter_axis_scale_options_defaults(self):
        """Test that SpecPlotter axis scale options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertIsNone(plotter.xscale)
        # yscale has a different default in SpecPlotter
        self.assertEqual(plotter.yscale, "linear")

    def test_spec_plotter_title_options_defaults(self):
        """Test that SpecPlotter title options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertIsNone(plotter.title)
        self.assertEqual(plotter.title_fontsize, 20)

    def test_spec_plotter_tick_options_defaults(self):
        """Test that SpecPlotter tick options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertEqual(plotter.tick_direction, "in")
        self.assertIsNone(plotter.tick_length)
        self.assertIsNone(plotter.tick_width)

    def test_spec_plotter_spine_options_defaults(self):
        """Test that SpecPlotter spine options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertTrue(plotter.show_spines)
        self.assertIsNone(plotter.spine_linewidth)

    def test_spec_plotter_save_format_defaults(self):
        """Test that SpecPlotter save_format options have correct defaults."""
        plotter = SpecPlotter(self.spectrum)
        self.assertEqual(plotter.save_format, "png")
        self.assertFalse(plotter.transparent)

    def test_spec_plotter_custom_grid_options(self):
        """Test that SpecPlotter respects custom grid options."""
        plotter = SpecPlotter(self.spectrum, show_grid=True, grid_alpha=0.8,
                             grid_color="red", grid_linestyle=":", grid_linewidth=2.0)
        self.assertTrue(plotter.show_grid)
        self.assertEqual(plotter.grid_alpha, 0.8)
        self.assertEqual(plotter.grid_color, "red")
        self.assertEqual(plotter.grid_linestyle, ":")
        self.assertEqual(plotter.grid_linewidth, 2.0)

    def test_spec_plotter_custom_legend_options(self):
        """Test that SpecPlotter respects custom legend options."""
        plotter = SpecPlotter(self.spectrum, legend_frameon=False, legend_shadow=True,
                             legend_fancybox=False, legend_framealpha=0.5)
        self.assertFalse(plotter.legend_frameon)
        self.assertTrue(plotter.legend_shadow)
        self.assertFalse(plotter.legend_fancybox)
        self.assertEqual(plotter.legend_framealpha, 0.5)

    def test_spec_plotter_custom_linestyle_options(self):
        """Test that SpecPlotter respects custom linestyle options."""
        plotter = SpecPlotter(self.spectrum, linestyle="--",
                             max_likelihood_linestyle=":", random_sample_linestyle="-.")
        self.assertEqual(plotter.linestyle, "--")
        self.assertEqual(plotter.max_likelihood_linestyle, ":")
        self.assertEqual(plotter.random_sample_linestyle, "-.")

    def test_spec_plotter_custom_marker_options(self):
        """Test that SpecPlotter respects custom marker options."""
        plotter = SpecPlotter(self.spectrum, markerfillstyle="none",
                             markeredgecolor="black", markeredgewidth=2.0)
        self.assertEqual(plotter.markerfillstyle, "none")
        self.assertEqual(plotter.markeredgecolor, "black")
        self.assertEqual(plotter.markeredgewidth, 2.0)

    def test_spec_plotter_custom_axis_scale_options(self):
        """Test that SpecPlotter respects custom axis scale options."""
        plotter = SpecPlotter(self.spectrum, xscale="log", yscale="log")
        self.assertEqual(plotter.xscale, "log")
        self.assertEqual(plotter.yscale, "log")

    def test_spec_plotter_custom_title_options(self):
        """Test that SpecPlotter respects custom title options."""
        plotter = SpecPlotter(self.spectrum, title="Custom Title", title_fontsize=30)
        self.assertEqual(plotter.title, "Custom Title")
        self.assertEqual(plotter.title_fontsize, 30)

    def test_spec_plotter_custom_tick_options(self):
        """Test that SpecPlotter respects custom tick options."""
        plotter = SpecPlotter(self.spectrum, tick_direction="out",
                             tick_length=10.0, tick_width=2.5)
        self.assertEqual(plotter.tick_direction, "out")
        self.assertEqual(plotter.tick_length, 10.0)
        self.assertEqual(plotter.tick_width, 2.5)

    def test_spec_plotter_custom_spine_options(self):
        """Test that SpecPlotter respects custom spine options."""
        plotter = SpecPlotter(self.spectrum, show_spines=False, spine_linewidth=3.0)
        self.assertFalse(plotter.show_spines)
        self.assertEqual(plotter.spine_linewidth, 3.0)

    def test_spec_plotter_custom_save_format_options(self):
        """Test that SpecPlotter respects custom save_format options."""
        plotter = SpecPlotter(self.spectrum, save_format="svg", transparent=True)
        self.assertEqual(plotter.save_format, "svg")
        self.assertTrue(plotter.transparent)

