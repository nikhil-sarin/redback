import unittest
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import redback
import redback.sed_analysis as sed_analysis
from redback.sed_analysis import (
    BlackbodySEDModel,
    CutoffBlackbodySEDModel,
    DirectIntegrationSEDModel,
    ExtinctionConfig,
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


class _InvalidLuminositySED(_PowerLawLuminositySED):
    def bolometric_luminosity(self, parameters):
        return np.nan


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
        np.testing.assert_allclose(
            frame.iloc[0]["lum_bol"] * 1e50, luminosity, rtol=1e-10)
        np.testing.assert_allclose(
            frame.iloc[0]["lum_bol_err"] * 1e50, expected_error, rtol=1e-8)
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

    def test_integration_failure_is_retained(self):
        result = SEDResult(
            transient_name="test", method="invalid", model=_InvalidLuminositySED(),
            epoch_results=[self.success], redshift=1.0, distance=1e27)
        frame = result.integrate().to_dataframe()
        self.assertFalse(frame.iloc[0]["success"])
        self.assertEqual("integration_failed", frame.iloc[0]["status"])
        self.assertTrue(np.isnan(frame.iloc[0]["lum_bol"]))


class TestExtinctionConfig(unittest.TestCase):
    def test_supported_extinction_laws(self):
        for law in ("fitzpatrick99", "fm07", "calzetti00", "odonnell94", "ccm89"):
            with self.subTest(law=law):
                config = ExtinctionConfig(host_law=law, mw_law=law)
                self.assertEqual(law, config.host_law)
                self.assertEqual(law, config.mw_law)

    def test_unsupported_extinction_laws(self):
        for field_name in ("host_law", "mw_law"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    ExtinctionConfig(**{field_name: "unsupported"})

    def test_extinction_values_are_validated(self):
        invalid = (
            {"av_host": np.nan}, {"av_mw": np.inf},
            {"av_host": -0.1}, {"av_mw": -0.1},
            {"rv_host": 0.0}, {"rv_mw": -1.0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ExtinctionConfig(**values)


class TestSEDValidationAndFailurePaths(unittest.TestCase):
    def _transient(self, data_mode="flux_density"):
        transient = unittest.mock.MagicMock()
        transient.name = "validation"
        transient.redshift = 0.1
        transient.data_mode = data_mode
        transient.get_filtered_data.return_value = (
            np.array([1.0, 1.1, 1.2]), np.zeros(3),
            np.ones(3), np.full(3, 0.1))
        transient.filtered_frequencies = np.array([3e14, 4e14, 5e14])
        return transient

    def test_top_level_options_are_validated(self):
        transient = self._transient()
        cases = (
            {"distance": 0.0}, {"distance": np.inf},
            {"bin_width": 0.0}, {"bin_width": np.nan},
            {"min_filters": 0}, {"min_filters": 1.5},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                sed_analysis.estimate_sed(transient, **values)
        transient.redshift = -0.1
        with self.assertRaisesRegex(ValueError, "redshift"):
            sed_analysis.estimate_sed(transient)

    def test_model_resolution_rejects_invalid_methods(self):
        with self.assertRaisesRegex(TypeError, "SEDModel protocol"):
            sed_analysis._resolve_sed_model(object(), 1e27, 0.1)
        with self.assertRaisesRegex(ValueError, "Unknown SED method"):
            sed_analysis._resolve_sed_model("not-a-model", 1e27, 0.1)

    def test_photometry_validation_reports_each_contract(self):
        valid = [np.ones(2), np.ones(2), np.ones(2), np.ones(2)]
        cases = (
            [np.ones(1), *valid[1:]],
            [np.array([]), np.array([]), np.array([]), np.array([])],
            [np.array([1.0, np.nan]), *valid[1:]],
            [valid[0], np.array([1.0, 0.0]), *valid[2:]],
            [*valid[:2], np.array([1.0, np.inf]), valid[3]],
            [*valid[:3], np.array([1.0, 0.0])],
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                sed_analysis._validate_photometry(*values)

    def test_unsupported_data_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supports magnitude"):
            sed_analysis._prepare_photometry(self._transient("counts"), 0.1, True)

    def test_non_bandpass_photometry_converts_magnitude_and_flux(self):
        for data_mode in ("magnitude", "flux"):
            transient = self._transient(data_mode)
            transient.filtered_sncosmo_bands = np.array(["ztfg", "ztfr", "ztfi"])
            time, frequency, wavelength, observed, error, context = \
                sed_analysis._prepare_photometry(transient, 0.1, False)
            self.assertEqual("flux_density", context["data_mode"])
            self.assertTrue(np.all(np.isfinite(frequency)))
            self.assertTrue(np.all(np.isfinite(wavelength)))
            self.assertEqual(time.shape, observed.shape)
            self.assertEqual(time.shape, error.shape)

    def test_observed_flux_integration_applies_extinction(self):
        extinction = SimpleNamespace(
            transmission=lambda wavelength, redshift: np.full_like(wavelength, 0.5))
        luminosity, variance = sed_analysis._integrate_observed_flux(
            np.array([3e14, 4e14, 5e14]), np.ones(3), np.full(3, 0.1),
            distance=1e27, redshift=0.1, extinction=extinction)
        self.assertGreater(abs(luminosity), 0.0)
        self.assertGreater(variance, 0.0)

    def test_epoch_diagnostic_and_axes_validation(self):
        epoch = SEDEpochResult(
            epoch_time=1.0, success=True, status="success",
            parameters={"temperature": 1.0, "radius": 1.0},
            covariance=np.eye(2))
        result = SEDResult("test", "test", _PowerLawLuminositySED(), [epoch], 0.1, 1e27)
        with self.assertRaisesRegex(ValueError, "photometry diagnostics"):
            result.plot_epoch(0)
        with self.assertRaisesRegex(ValueError, "Expected 2 axes"):
            result.plot_evolution(axes=[plt.subplots()[1]])

    def test_empty_and_failed_legacy_results_return_none(self):
        empty = SEDResult("test", "test", _PowerLawLuminositySED(), [], 0.1, 1e27)
        self.assertIsNone(empty._legacy_parameter_dataframe())
        self.assertIsNone(empty._legacy_bolometric_dataframe())
        with self.assertRaisesRegex(ValueError, "A_ext"):
            empty._legacy_bolometric_dataframe(A_ext=np.nan)

        poor = SEDEpochResult(
            epoch_time=1.0, success=True, status="poorly_constrained",
            parameters={"temperature": 1.0, "radius": 1.0}, covariance=np.eye(2))
        result = SEDResult("test", "test", _PowerLawLuminositySED(), [poor], 0.1, 1e27)
        self.assertIsNone(result._legacy_parameter_dataframe())

    def test_covariance_helpers_handle_missing_and_invalid_values(self):
        epoch = SEDEpochResult(
            epoch_time=1.0, success=True, status="success",
            parameters={"temperature": 1.0, "radius": 1.0}, covariance=None)
        result = SEDResult("test", "test", _PowerLawLuminositySED(), [epoch], 0.1, 1e27)
        self.assertTrue(np.isnan(result._parameter_error(epoch, "temperature")))
        self.assertTrue(np.isnan(result._luminosity_variance(epoch)))
        self.assertTrue(np.isnan(result._function_variance(epoch, lambda _: 1.0)))

        invalid = np.array([[np.nan, 0.0], [0.0, 1.0]])
        self.assertEqual(
            "invalid_covariance",
            sed_analysis._quality_status(result.model, epoch.parameters, invalid))
        broad = np.diag([4.0, 4.0])
        self.assertEqual(
            "poorly_constrained",
            sed_analysis._quality_status(result.model, epoch.parameters, broad))

    def test_direct_model_without_tail_has_explicit_contract(self):
        model = DirectIntegrationSEDModel(None)
        initial, bounds = model.fit_setup()
        self.assertEqual(0, len(initial))
        self.assertEqual({}, model.parameters_from_fit([], None)[0])
        with self.assertRaisesRegex(ValueError, "No tail model"):
            model.evaluate_photometry([], {}, {})
        with self.assertRaisesRegex(RuntimeError, "stored per epoch"):
            model.bolometric_luminosity({})
        with self.assertRaisesRegex(RuntimeError, "stored per epoch"):
            model.luminosity_fractions({}, (1.0, 2.0))

    def test_fit_failures_are_retained(self):
        transient = self._transient()
        with patch("redback.sed_analysis.curve_fit", side_effect=RuntimeError("failed")):
            result = sed_analysis.estimate_sed(transient, min_filters=1)
        self.assertFalse(result.epoch_results[0].success)
        self.assertEqual("fit_failed", result.epoch_results[0].status)

        with patch("redback.sed_analysis._integrate_observed_flux", side_effect=ValueError("failed")):
            direct = sed_analysis.estimate_sed(
                transient, method="direct_integration", min_filters=1,
                uv_tail="none", ir_tail="none")
        self.assertFalse(direct.epoch_results[0].success)
        self.assertEqual("fit_failed", direct.epoch_results[0].status)


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

    def test_zero_redshift_is_supported(self):
        transient, distance, parameters = self._make_transient()
        transient.redshift = 0.0
        observer_frequency = np.asarray(transient.frequency)
        model = BlackbodySEDModel(distance=distance, redshift=0.0)
        flux = model.evaluate_photometry(
            observer_frequency, parameters,
            {"coordinate_type": "frequency", "data_mode": "flux_density"})
        error = 0.05 * flux
        time = transient.time
        transient.flux_density = flux
        transient.flux_density_err = error
        transient.get_filtered_data = lambda: (time, np.zeros(len(time)), flux, error)

        frame = transient.estimate_sed(distance=distance).to_dataframe(True)
        np.testing.assert_allclose(
            frame.iloc[0]["temperature"], parameters["temperature"], rtol=1e-6)
        np.testing.assert_allclose(frame.iloc[0]["radius"], parameters["radius"], rtol=1e-6)

    def test_custom_sed_model_protocol(self):
        transient, distance, parameters = self._make_transient()
        custom_model = BlackbodySEDModel(distance=distance, redshift=transient.redshift)
        custom_model.name = "custom_blackbody"
        result = transient.estimate_sed(method=custom_model, distance=distance)
        self.assertIs(result.model, custom_model)
        self.assertEqual("custom_blackbody", result.method)
        np.testing.assert_allclose(
            result.epoch_results[0].parameters["temperature"],
            parameters["temperature"], rtol=1e-6)

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
        np.testing.assert_allclose(
            result.epoch_results[0].parameters["temperature"], 11000.0, rtol=1e-6)
        np.testing.assert_allclose(
            result.epoch_results[0].parameters["radius"], 8e14, rtol=1e-6)
        self.assertGreater(frame.iloc[0]["lum_bol"], 0.0)
        blackbody = BlackbodySEDModel(distance=distance, redshift=transient.redshift)
        self.assertLess(
            frame.iloc[0]["lum_bol"] * 1e50,
            blackbody.bolometric_luminosity(result.epoch_results[0].parameters))

    def test_cutoff_parameters_can_be_fitted(self):
        cutoff = 4500.0
        index = 2.0
        transient, distance, parameters = self._make_transient(
            CutoffBlackbodySEDModel, cutoff_wavelength=cutoff, absorption_index=index)
        result = transient.estimate_sed(
            method="cutoff_blackbody", distance=distance,
            cutoff_wavelength=cutoff, absorption_index=index,
            fit_cutoff_wavelength=True, fit_absorption_index=True)
        frame = result.to_dataframe(successful_only=True)
        np.testing.assert_allclose(
            frame.iloc[0]["cutoff_wavelength"], parameters["cutoff_wavelength"], rtol=1e-5)
        np.testing.assert_allclose(
            frame.iloc[0]["absorption_index"], parameters["absorption_index"], rtol=1e-5)
        self.assertEqual((4, 4), frame.iloc[0]["covariance"].shape)

    def test_magnitude_mode_uses_bandpass_photometry(self):
        redshift = 0.1
        distance = 1e27
        bands = np.array(["ztfg", "ztfr", "ztfi", "sdss::u", "sdss::z"])
        model = BlackbodySEDModel(distance=distance, redshift=redshift)
        parameters = {"temperature": 11000.0, "radius": 8e14}
        context = {"coordinate_type": "band", "data_mode": "magnitude"}
        magnitude = model.evaluate_photometry(bands, parameters, context)
        error = np.full(len(bands), 0.05)
        time = np.linspace(10.0, 10.2, len(bands))
        transient = redback.transient.OpticalTransient(
            time=time, magnitude=magnitude, magnitude_err=error,
            redshift=redshift, data_mode="magnitude", name="BandpassSED",
            bands=bands, use_phase_model=False)
        transient.get_filtered_data = lambda: (time, np.zeros(len(time)), magnitude, error)

        frame = transient.estimate_sed(distance=distance).to_dataframe(True)
        np.testing.assert_allclose(frame.iloc[0]["temperature"], parameters["temperature"], rtol=1e-6)
        np.testing.assert_allclose(frame.iloc[0]["radius"], parameters["radius"], rtol=1e-6)

    def test_cutoff_blackbody_uses_bandpass_photometry(self):
        redshift = 0.1
        distance = 1e27
        cutoff = 4500.0
        index = 2.0
        bands = np.array(["sdss::u", "ztfg", "ztfr", "ztfi", "sdss::z"])
        model = CutoffBlackbodySEDModel(
            distance=distance, redshift=redshift,
            cutoff_wavelength=cutoff, absorption_index=index)
        parameters = {
            "temperature": 11000.0, "radius": 8e14,
            "cutoff_wavelength": cutoff, "absorption_index": index}
        context = {"coordinate_type": "band", "data_mode": "magnitude"}
        magnitude = model.evaluate_photometry(bands, parameters, context)
        error = np.full(len(bands), 0.05)
        time = np.linspace(10.0, 10.2, len(bands))
        transient = redback.transient.OpticalTransient(
            time=time, magnitude=magnitude, magnitude_err=error,
            redshift=redshift, data_mode="magnitude", name="BandpassCutoffSED",
            bands=bands, use_phase_model=False)
        transient.get_filtered_data = lambda: (time, np.zeros(len(time)), magnitude, error)

        frame = transient.estimate_sed(
            method="cutoff_blackbody", distance=distance,
            cutoff_wavelength=cutoff, absorption_index=index).to_dataframe(True)
        np.testing.assert_allclose(frame.iloc[0]["temperature"], parameters["temperature"], rtol=1e-6)
        np.testing.assert_allclose(frame.iloc[0]["radius"], parameters["radius"], rtol=1e-6)

    def test_extinction_is_fit_in_forward_photometry(self):
        redshift = 0.1
        distance = 1e27
        extinction = ExtinctionConfig(av_host=0.5, av_mw=0.1)
        model = BlackbodySEDModel(
            distance=distance, redshift=redshift, extinction=extinction)
        parameters = {"temperature": 11000.0, "radius": 8e14}
        observer_frequency = redback.utils.lambda_to_nu(
            np.array([2500.0, 3300.0, 4200.0, 5200.0, 6500.0, 8000.0]))
        rest_frequency, _ = redback.utils.calc_kcorrected_properties(
            frequency=observer_frequency, redshift=redshift, time=0.0)
        flux = model.evaluate_photometry(
            rest_frequency, parameters,
            {"coordinate_type": "frequency", "data_mode": "flux_density"})
        error = 0.05 * flux
        time = np.linspace(10.0, 10.2, len(flux))
        transient = redback.transient.OpticalTransient(
            time=time, flux_density=flux, flux_density_err=error,
            redshift=redshift, data_mode="flux_density", name="ExtinctedSED",
            frequency=observer_frequency, use_phase_model=False)
        transient.get_filtered_data = lambda: (time, np.zeros(len(time)), flux, error)

        frame = transient.estimate_sed(
            distance=distance, extinction=extinction).to_dataframe(True)
        np.testing.assert_allclose(frame.iloc[0]["temperature"], parameters["temperature"], rtol=1e-6)
        np.testing.assert_allclose(frame.iloc[0]["radius"], parameters["radius"], rtol=1e-6)

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

    def test_direct_integration_rejects_unknown_uncertainty_method(self):
        transient, distance, _ = self._make_transient()
        with self.assertRaisesRegex(ValueError, "uncertainty_method"):
            transient.estimate_sed(
                method="direct_integration", distance=distance,
                uv_tail="none", ir_tail="none", uncertainty_method="unknown")

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

    def test_direct_integration_supports_independent_uncertainties(self):
        transient, distance, _ = self._make_transient()
        frame = transient.estimate_sed(
            method="direct_integration", distance=distance,
            uv_tail="none", ir_tail="none",
            uncertainty_method="independent").integrate().to_dataframe(True)
        self.assertGreater(frame.iloc[0]["lum_bol_err"], 0.0)

    def test_direct_integration_recovers_blackbody_with_matching_tails(self):
        transient, distance, parameters = self._make_transient()
        frame = transient.estimate_sed(
            method="direct_integration", distance=distance,
            uv_tail="blackbody", ir_tail="blackbody").integrate().to_dataframe(True)
        expected = (
            4 * np.pi * parameters["radius"] ** 2 * redback.constants.sigma_sb
            * parameters["temperature"] ** 4)
        np.testing.assert_allclose(frame.iloc[0]["lum_bol"] * 1e50, expected, rtol=0.01)

    def test_dataframe_wrapper_preserves_fitted_parameters(self):
        transient, distance, _ = self._make_transient(
            CutoffBlackbodySEDModel, cutoff_wavelength=4500.0, absorption_index=2.0)
        keywords = dict(
            method="cutoff_blackbody", distance=distance,
            cutoff_wavelength=4500.0, absorption_index=2.0)
        current = transient.estimate_bb_params(**keywords)
        structured = transient.estimate_sed(**keywords).to_dataframe(successful_only=True)
        np.testing.assert_allclose(
            current[["temperature", "radius"]],
            structured[["temperature", "radius"]], rtol=1e-6)

    def test_dataframe_bolometric_wrapper_uses_stable_blue_boost(self):
        transient, distance, _ = self._make_transient()
        uncorrected = transient.estimate_bolometric_luminosity(distance=distance)
        corrected = transient.estimate_bolometric_luminosity(
            distance=distance, lambda_cut=4500.0)
        self.assertGreater(corrected.iloc[0]["lum_bol"], uncorrected.iloc[0]["lum_bol"])
        self.assertEqual(corrected.iloc[0]["lum_bol_bb"], uncorrected.iloc[0]["lum_bol_bb"])


if __name__ == "__main__":
    unittest.main()
