import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd
from redback.transient import Spectrum, Transient
from redback.plotting import SpecPlotter, IntegratedFluxPlotter

from types import SimpleNamespace
from redback.plotting import Plotter, _FilenameGetter, _FilePathGetter
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
        self.plotter = SpecPlotter(self.spectrum)

    def tearDown(self) -> None:
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
        patch.stopall()

    @patch("matplotlib.pyplot.figure", autospec=True)
    def test_plot_data(self, mock_figure):

        axes_mock = MagicMock()
        plt.gca = MagicMock(return_value=axes_mock)

        result_axes = self.plotter.plot_data(save=False, show=False)

        self.assertEqual(result_axes, axes_mock)
        axes_mock.errorbar.assert_called_once()
        axes_mock.set_xscale.assert_called_once_with("log")
        axes_mock.set_yscale.assert_called_once_with("log")
        axes_mock.set_xlim.assert_called_once_with(0.1, 200)
        axes_mock.set_ylim.assert_called_once_with(0.1, 200)
        axes_mock.set_xlabel.assert_called_once_with(r"Time since burst [s]", fontsize=self.plotter.fontsize_axes)
        axes_mock.set_ylabel.assert_called_once_with(self.transient_mock.ylabel, fontsize=self.plotter.fontsize_axes)
        axes_mock.annotate.assert_called_once_with(
            self.transient_mock.name,
            xy=self.plotter.xy,
            xycoords=self.plotter.xycoords,
            horizontalalignment=self.plotter.horizontalalignment,
            size=self.plotter.annotation_size
        )
        self.plotter._save_and_show.assert_called_once_with(filepath=self.plotter._data_plot_filepath, save=False,
                                                            show=False)

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

