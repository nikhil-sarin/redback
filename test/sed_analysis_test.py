import unittest

import matplotlib.pyplot as plt
import numpy as np

import redback
from redback.sed_analysis import (
    BlackbodySEDModel,
    CutoffBlackbodySEDModel,
    SEDEpochResult,
    SEDResult,
)


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


class TestEstimateSED(unittest.TestCase):
    def _make_transient(self, model_class=BlackbodySEDModel, **model_kwargs):
        redshift = 0.1
        distance = 1e27
        temperature = 11000.0
        radius = 8e14
        wavelengths = np.array([2500.0, 3300.0, 4200.0, 5200.0, 6500.0, 8000.0])
        observer_frequency = redback.utils.lambda_to_nu(wavelengths)
        rest_frequency, _ = redback.utils.calc_kcorrected_properties(
            frequency=observer_frequency, redshift=redshift, time=0.0)
        model = model_class(distance=distance, redshift=redshift, **model_kwargs)
        parameters = {"temperature": temperature, "radius": radius}
        parameters.update(model_kwargs)
        flux_density = model.evaluate_photometry(
            rest_frequency, parameters,
            {"coordinate_type": "frequency", "data_mode": "flux_density"})
        flux_density_err = 0.05 * flux_density
        time = np.array([10.0, 10.05, 10.1, 10.15, 10.2, 10.25])
        transient = redback.transient.OpticalTransient(
            time=time, flux_density=flux_density, flux_density_err=flux_density_err,
            redshift=redshift, data_mode="flux_density", name="TestSED",
            frequency=observer_frequency, use_phase_model=False)
        transient.get_filtered_data = lambda: (
            time, np.zeros(len(time)), flux_density, flux_density_err)
        return transient, distance, parameters

    def test_blackbody_fit_and_integration_recover_inputs(self):
        transient, distance, parameters = self._make_transient()
        result = transient.estimate_sed(distance=distance, bin_width=1.0)
        frame = result.to_dataframe(successful_only=True)
        self.assertEqual("success", frame.iloc[0]["status"])
        np.testing.assert_allclose(frame.iloc[0]["temperature"], parameters["temperature"], rtol=1e-6)
        np.testing.assert_allclose(frame.iloc[0]["radius"], parameters["radius"], rtol=1e-6)
        self.assertNotEqual(0.0, frame.iloc[0]["covariance"][0, 1])

        bolometric = result.integrate().to_dataframe(successful_only=True)
        expected = 4 * np.pi * parameters["radius"] ** 2 * redback.constants.sigma_sb * parameters["temperature"] ** 4
        np.testing.assert_allclose(bolometric.iloc[0]["lum_bol"] * 1e50, expected, rtol=1e-6)
        np.testing.assert_allclose(
            bolometric.iloc[0][["uv_fraction", "observed_fraction", "ir_fraction"]].sum(),
            1.0, rtol=1e-10)

    def test_cutoff_blackbody_fit_owns_its_integral(self):
        cutoff = 4500.0
        index = 2.0
        transient, distance, _ = self._make_transient(
            CutoffBlackbodySEDModel, cutoff_wavelength=cutoff, absorption_index=index)
        result = transient.estimate_sed(
            method="cutoff_blackbody", distance=distance,
            cutoff_wavelength=cutoff, absorption_index=index)
        frame = result.integrate().to_dataframe(successful_only=True)
        self.assertEqual("cutoff_blackbody", frame.iloc[0]["method"])
        self.assertGreater(frame.iloc[0]["lum_bol"], 0.0)
        blackbody = BlackbodySEDModel(distance=distance, redshift=transient.redshift)
        self.assertLess(
            frame.iloc[0]["lum_bol"] * 1e50,
            blackbody.bolometric_luminosity(result.epoch_results[0].parameters))

    def test_insufficient_epoch_is_retained(self):
        transient, distance, _ = self._make_transient()
        original = transient.get_filtered_data()
        time = original[0].copy()
        time[-1] = 12.0
        transient.get_filtered_data = lambda: (time, original[1], original[2], original[3])
        result = transient.estimate_sed(distance=distance, bin_width=1.0, min_filters=3)
        self.assertEqual(2, len(result.epoch_results))
        self.assertEqual("insufficient_filters", result.epoch_results[1].status)

    def test_direct_integration_requires_explicit_tails(self):
        transient, distance, _ = self._make_transient()
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            transient.estimate_sed(method="direct_integration", distance=distance)

    def test_direct_integration_separates_observed_and_tail_luminosity(self):
        transient, distance, _ = self._make_transient()
        result = transient.estimate_sed(
            method="direct_integration", distance=distance,
            uv_tail="cutoff_blackbody", ir_tail="blackbody",
            cutoff_wavelength=4500.0, absorption_index=2.0)
        frame = result.integrate().to_dataframe(successful_only=True)
        self.assertGreater(frame.iloc[0]["lum_ultraviolet"], 0.0)
        self.assertGreater(frame.iloc[0]["lum_observed"], 0.0)
        self.assertGreater(frame.iloc[0]["lum_infrared"], 0.0)
        np.testing.assert_allclose(
            frame.iloc[0]["lum_bol"],
            frame.iloc[0][["lum_ultraviolet", "lum_observed", "lum_infrared"]].sum())
        self.assertGreater(frame.iloc[0]["lum_bol_err"], 0.0)

    def test_direct_integration_can_disable_both_tails(self):
        transient, distance, _ = self._make_transient()
        frame = transient.estimate_sed(
            method="direct_integration", distance=distance,
            uv_tail="none", ir_tail="none").integrate().to_dataframe(successful_only=True)
        self.assertEqual(0.0, frame.iloc[0]["lum_ultraviolet"])
        self.assertEqual(0.0, frame.iloc[0]["lum_infrared"])
        self.assertEqual(frame.iloc[0]["lum_observed"], frame.iloc[0]["lum_bol"])

    def test_direct_integration_recovers_blackbody_with_matching_tails(self):
        transient, distance, parameters = self._make_transient()
        frame = transient.estimate_sed(
            method="direct_integration", distance=distance,
            uv_tail="blackbody", ir_tail="blackbody").integrate().to_dataframe(True)
        expected = (
            4 * np.pi * parameters["radius"] ** 2 * redback.constants.sigma_sb
            * parameters["temperature"] ** 4)
        np.testing.assert_allclose(frame.iloc[0]["lum_bol"] * 1e50, expected, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
