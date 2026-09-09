import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from redback.transient_models import phase_models


class TestPhaseModels(unittest.TestCase):

    def test_mask_per_epoch_kwargs_only_masks_matching_arrays(self):
        mask = np.array([True, False, True])
        kwargs = {
            "frequency": np.array([1.0, 2.0, 3.0]),
            "bands": np.array(["g", "r", "i"]),
            "redshift": 0.1,
        }
        result = phase_models._mask_per_epoch_kwargs(kwargs, mask)
        np.testing.assert_array_equal(result["frequency"], [1.0, 3.0])
        np.testing.assert_array_equal(result["bands"], ["g", "i"])
        self.assertEqual(0.1, result["redshift"])
        self.assertEqual(3, len(kwargs["frequency"]))

    def test_t0_base_model_masks_epochs_and_converts_peak_time(self):
        model = unittest.mock.Mock(return_value=np.array([2.0, 3.0]))
        with patch.dict("redback.model_library.all_models_dict", {"phase_test": model}):
            result = phase_models.t0_base_model(
                np.array([59999.0, 60000.0, 60001.0]), t0=60000.0,
                base_model="phase_test", output_format="flux_density",
                peak_time_mjd=60005.0, frequency=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [0.0, 2.0, 3.0])
        np.testing.assert_array_equal(model.call_args.args[0], [0.0, 1.0])
        np.testing.assert_array_equal(model.call_args.kwargs["frequency"], [2.0, 3.0])
        self.assertEqual(5.0, model.call_args.kwargs["peak_time"])

    def test_t0_base_model_uses_faint_magnitude_before_start(self):
        model = unittest.mock.Mock(return_value=np.array([20.0]))
        with patch.dict("redback.model_library.all_models_dict", {"phase_test": model}):
            result = phase_models.t0_base_model(
                np.array([59999.0, 60000.0]), t0=60000.0,
                base_model="phase_test", output_format="magnitude")
        np.testing.assert_array_equal(result, [1000.0, 20.0])

    def test_t0_base_model_sets_expansion_submodel(self):
        model = unittest.mock.Mock(return_value=np.array([1.0]))
        with patch.dict(
                "redback.model_library.all_models_dict",
                {"thin_shell_supernova": model}):
            phase_models.t0_base_model(
                np.array([60000.0]), t0=60000.0,
                base_model="thin_shell_supernova", submodel="arnett_bolometric",
                output_format="flux_density")
        self.assertEqual("arnett_bolometric", model.call_args.kwargs["base_model"])

    def test_extinction_wrappers_select_their_model_type(self):
        wrappers = {
            phase_models.t0_afterglow_extinction: "afterglow",
            phase_models.t0_supernova_extinction: "supernova",
            phase_models.t0_kilonova_extinction: "kilonova",
            phase_models.t0_tde_extinction: "tde",
            phase_models.t0_magnetar_driven_extinction: "magnetar_driven",
            phase_models.t0_shock_powered_extinction: "shock_powered",
        }
        with patch.object(
                phase_models, "_t0_with_extinction",
                return_value=SimpleNamespace(observable=np.array([4.0]))) as helper:
            for function, model_type in wrappers.items():
                kwargs = {"output_format": "flux_density"}
                if model_type == "tde":
                    kwargs["peak_time_mjd"] = 60002.0
                with self.subTest(model_type=model_type):
                    np.testing.assert_array_equal(
                        function(np.array([60001.0]), 60000.0, 0.2, **kwargs), [4.0])
                    self.assertEqual(model_type, helper.call_args.kwargs["model_type"])
                    if model_type == "tde":
                        self.assertEqual(2.0, helper.call_args.kwargs["peak_time"])

    def test_stellar_interaction_extinction_wrapper(self):
        with patch.object(
                phase_models, "_t0_with_extinction",
                return_value=SimpleNamespace(observable=np.array([5.0]))) as helper:
            result = phase_models.t0_stellar_interaction_extinction(
                np.array([60001.0]), 60000.0, 0.3, output_format="flux_density")
        np.testing.assert_array_equal(result, [5.0])
        self.assertEqual("stellar_interaction", helper.call_args.kwargs["model_type"])
        self.assertEqual(0.3, helper.call_args.kwargs["av"])

    def test_t0_with_extinction_masks_values_and_marks_pre_start_epochs(self):
        extinction_model = unittest.mock.Mock(return_value=np.array([2.0, 3.0]))
        with patch.dict(phase_models.extinction_model_functions, {"test": extinction_model}):
            result = phase_models._t0_with_extinction(
                np.array([59999.0, 60000.0, 60001.0]), 60000.0, 0.2,
                model_type="test", output_format="magnitude",
                bands=np.array(["g", "r", "i"]))
        np.testing.assert_array_equal(result.time, [-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result.observable, [5000.0, 2.0, 3.0])
        np.testing.assert_array_equal(extinction_model.call_args.kwargs["bands"], ["r", "i"])

    def test_dust_to_gas_wrapper_supports_flux_and_magnitude(self):
        with patch.object(
                phase_models.extinction_models,
                "extinction_afterglow_galactic_dust_to_gas_ratio",
                return_value=np.array([20.0])):
            magnitude = phase_models.t0_afterglow_extinction_model_d2g(
                np.array([60001.0]), lognh=21.0, factor=1.0,
                t0=60000.0, output_format="magnitude")
            flux = phase_models.t0_afterglow_extinction_model_d2g(
                np.array([60001.0]), lognh=21.0, factor=1.0,
                t0=60000.0, output_format="flux_density")
        np.testing.assert_array_equal(magnitude, [20.0])
        self.assertTrue(np.all(np.isfinite(flux)))

    def test_sampled_peak_predeceleration_supports_both_formats(self):
        with patch.object(
                phase_models, "t0_afterglow_extinction_model_d2g",
                side_effect=[np.array([4.0]), 4.0, np.array([4.0]), 4.0]), \
                patch.object(
                    phase_models.extinction_models, "_extinction_with_predeceleration",
                    side_effect=[np.array([1.0]), np.array([1.0])]):
            kwargs = dict(m=2.0, t0=0.0, output_format="flux_density")
            flux = phase_models._t0_exinction_models_with_sampled_t_peak(
                np.array([1.0, 3.0]), tp=2.0, **kwargs)
            kwargs["output_format"] = "magnitude"
            magnitude = phase_models._t0_exinction_models_with_sampled_t_peak(
                np.array([1.0, 3.0]), tp=2.0, **kwargs)
        np.testing.assert_array_equal(flux, [1.0, 4.0])
        self.assertTrue(np.all(np.isfinite(magnitude)))

    def test_thin_shell_predeceleration_supports_both_formats(self):
        with patch.object(
                phase_models, "t0_afterglow_extinction_model_d2g",
                side_effect=[np.array([4.0]), 4.0, np.array([4.0]), 4.0]), \
                patch.object(
                    phase_models.extinction_models, "_extinction_with_predeceleration",
                    side_effect=[np.array([1.0]), np.array([1.0])]):
            kwargs = dict(
                loge0=50.0, logn0=0.0, g0=100.0, m=2.0, t0=0.0,
                output_format="flux_density")
            time = np.array([0.0, 1e10])
            flux = phase_models._t0_thin_shell_predeceleration(time, **kwargs)
            kwargs["output_format"] = "magnitude"
            magnitude = phase_models._t0_thin_shell_predeceleration(time, **kwargs)
        np.testing.assert_array_equal(flux, [1.0, 4.0])
        self.assertTrue(np.all(np.isfinite(magnitude)))

    def test_afterglow_rate_and_flux_helpers(self):
        with patch.object(
                phase_models.infam, "integrated_flux_rate_model",
                return_value=np.array([7.0, 8.0])) as rate_model:
            rate = phase_models._t0_afterglowpy_rate_model(
                np.array([0.0, 2.0, 3.0]), burst_start=2.0,
                background_rate=3.0, dt=2.0, prefactor=np.ones(3))
        np.testing.assert_array_equal(rate, [6.0, 7.0, 8.0])
        np.testing.assert_array_equal(rate_model.call_args.args[0], [0.0, 1.0])

        with patch.object(
                phase_models.infam, "integrated_flux_afterglowpy_base_model",
                return_value=np.array([9.0])):
            flux, times = phase_models._t0_afterglowpy_flux_model(
                np.array([1.0, 3.0]), burst_start=2.0)
        np.testing.assert_array_equal(flux, [9.0])
        np.testing.assert_array_equal(times, [1.0])

    def test_flux_density_helper_accepts_callable_and_registered_name(self):
        def model(time, **kwargs):
            return np.full(len(time), 5.0)

        flux, times = phase_models._t0_afterglowpy_flux_density_model(
            np.array([1.0, 3.0]), burst_start=2.0, base_model=model)
        np.testing.assert_array_equal(flux, [5.0])
        np.testing.assert_array_equal(times, [1.0])

        with patch.dict(
                "redback.model_library.modules_dict",
                {"afterglow_models": {"registered": model}}):
            phase_models._t0_afterglowpy_flux_density_model(
                np.array([3.0]), burst_start=2.0, base_model="registered")

        with self.assertRaisesRegex(ValueError, "valid base model"):
            phase_models._t0_afterglowpy_flux_density_model(
                np.array([3.0]), burst_start=2.0, base_model=object())


if __name__ == "__main__":
    unittest.main()
