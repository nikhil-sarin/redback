import unittest

import matplotlib.pyplot as plt
import numpy as np

from redback.sed_analysis import SEDEpochResult, SEDResult


class _PowerLawLuminositySED:
    name = "test_sed"
    parameter_names = ("temperature", "radius")

    def bolometric_luminosity(self, parameters):
        return parameters["radius"] ** 2 * parameters["temperature"] ** 4

    def luminosity_fractions(self, parameters, wavelength_range):
        return 0.2, 0.7, 0.1

    def evaluate_photometry(self, coordinates, parameters, context):
        return parameters["temperature"] * np.asarray(coordinates)


class TestSEDResult(unittest.TestCase):
    def setUp(self):
        self.model = _PowerLawLuminositySED()
        covariance = np.array([[4.0, -0.5], [-0.5, 0.25]])
        self.success = SEDEpochResult(
            epoch_time=2.0, success=True, status="success",
            parameters={"temperature": 10.0, "radius": 3.0}, covariance=covariance,
            chi_square=4.0, degrees_of_freedom=2, n_filters=4,
            wavelength_min=3000.0, wavelength_max=9000.0,
            coordinates=np.array([1.0, 2.0]), observed=np.array([10.0, 20.0]),
            observed_error=np.ones(2))
        self.failure = SEDEpochResult(
            epoch_time=3.0, success=False, status="fit_failed", message="test failure",
            n_filters=3, wavelength_min=4000.0, wavelength_max=8000.0)
        self.result = SEDResult(
            transient_name="test", method=self.model.name, model=self.model,
            epoch_results=[self.success, self.failure], redshift=1.0, distance=1e27)

    def tearDown(self):
        plt.close("all")

    def test_dataframe_retains_failed_epochs(self):
        frame = self.result.to_dataframe()
        self.assertEqual(2, len(frame))
        self.assertEqual("fit_failed", frame.iloc[1]["status"])
        self.assertEqual(1, len(self.result.to_dataframe(successful_only=True)))

    def test_integration_uses_full_covariance(self):
        bolometric = self.result.integrate()
        frame = bolometric.to_dataframe()
        luminosity = 3.0 ** 2 * 10.0 ** 4
        gradient = np.array([4 * 3.0 ** 2 * 10.0 ** 3, 2 * 3.0 * 10.0 ** 4])
        expected_error = np.sqrt(gradient @ self.success.covariance @ gradient)
        self.assertAlmostEqual(luminosity / 1e50, frame.iloc[0]["lum_bol"])
        self.assertAlmostEqual(expected_error / 1e50, frame.iloc[0]["lum_bol_err"], places=12)
        self.assertTrue(np.isnan(frame.iloc[1]["lum_bol"]))

    def test_integration_records_extrapolated_fractions(self):
        frame = self.result.integrate().to_dataframe(successful_only=True)
        self.assertAlmostEqual(1.0, frame.iloc[0][["uv_fraction", "observed_fraction", "ir_fraction"]].sum())

    def test_diagnostic_plots(self):
        self.assertIsNotNone(self.result.plot_epoch(0))
        self.assertEqual(2, len(self.result.plot_evolution()))
        self.assertIsNotNone(self.result.integrate().plot_evolution())

    def test_failed_epoch_cannot_be_plotted(self):
        with self.assertRaisesRegex(ValueError, "unsuccessful epoch"):
            self.result.plot_epoch(1)


if __name__ == "__main__":
    unittest.main()
