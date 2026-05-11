import pathlib
import runpy
import unittest
import warnings
from unittest import mock

import matplotlib

matplotlib.use("Agg")


EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples"


class ExamplesSmokeTest(unittest.TestCase):

    def test_python_examples_compile(self):
        """Keep example scripts syntactically valid."""
        for path in sorted(EXAMPLES_DIR.glob("*.py")):
            with self.subTest(example=path.name):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    @mock.patch("matplotlib.pyplot.show")
    def test_non_detection_example_runs(self, _mock_show):
        """The lightweight non-detection example should execute without sampling."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            runpy.run_path(str(EXAMPLES_DIR / "non_detection_example.py"), run_name="__main__")

    def test_joint_galaxy_transient_spectrum_example_runs(self):
        """The host-galaxy + transient spectrum example should execute setup code."""
        runpy.run_path(
            str(EXAMPLES_DIR / "joint_galaxy_transient_spectrum_example.py"),
            run_name="__main__"
        )
