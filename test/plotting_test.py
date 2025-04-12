import os
import unittest
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