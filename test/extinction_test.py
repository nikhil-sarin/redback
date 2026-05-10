import unittest
from unittest import mock
import numpy as np
import pandas as pd

import redback
from redback.transient_models.extinction_models import (
    _get_correct_function,
    _perform_extinction,
    extinction_with_supernova_base_model,
    extinction_with_kilonova_base_model,
    extinction_with_tde_base_model,
    extinction_with_afterglow_base_model,
    extinction_with_function,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_kwargs(output_format='flux_density', frequency=3e14):
    return dict(
        redshift=0.1,
        f_nickel=0.1,
        mej=1.0,
        output_format=output_format,
        frequency=frequency,
        kappa=0.1,
        kappa_gamma=10.0,
        temperature_floor=3000.0,
        vej=10000.0,
    )


TIME = np.linspace(1, 100, 10)
FREQ = 3e14  # optical Hz


# ---------------------------------------------------------------------------
# _get_correct_function
# ---------------------------------------------------------------------------

class TestGetCorrectFunction(unittest.TestCase):

    def test_function_passthrough(self):
        """A callable is returned as-is."""
        import redback.transient_models.supernova_models as sm
        result = _get_correct_function(sm.arnett, model_type=None)
        self.assertIs(result, sm.arnett)

    def test_string_with_model_type(self):
        """String model name resolved via model_type."""
        f = _get_correct_function('arnett', model_type='supernova')
        self.assertTrue(callable(f))

    def test_string_without_model_type(self):
        """String model name resolved by searching all modules."""
        f = _get_correct_function('arnett', model_type=None)
        self.assertTrue(callable(f))

    def test_string_and_dynamic_return_same_function(self):
        """model_type-specified and dynamic search return the same function."""
        f1 = _get_correct_function('arnett', model_type='supernova')
        f2 = _get_correct_function('arnett', model_type=None)
        self.assertIs(f1, f2)

    def test_bad_model_name_with_model_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            _get_correct_function('nonexistent_model', model_type='supernova')

    def test_bad_model_name_without_model_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            _get_correct_function('nonexistent_model', model_type=None)

    def test_bad_model_type_raises_value_error(self):
        """Invalid model_type must raise ValueError, not KeyError."""
        with self.assertRaises(ValueError):
            _get_correct_function('arnett', model_type='invalid_type')

    def test_non_string_non_function_raises_value_error(self):
        with self.assertRaises(ValueError):
            _get_correct_function(42, model_type='supernova')


# ---------------------------------------------------------------------------
# _perform_extinction
# ---------------------------------------------------------------------------

class TestPerformExtinction(unittest.TestCase):

    def setUp(self):
        # 1 mJy flat spectrum over optical range
        self.angstroms = np.linspace(3000, 8000, 20).astype(float)
        self.flux = np.ones(20)

    def test_zero_av_host_and_mw_unchanged(self):
        result = _perform_extinction(
            self.flux, self.angstroms, av_host=0.0, rv_host=3.1,
            av_mw=0.0, redshift=0.1)
        np.testing.assert_array_almost_equal(result, self.flux)

    def test_positive_av_host_dims_flux(self):
        result = _perform_extinction(
            self.flux, self.angstroms, av_host=1.0, rv_host=3.1,
            av_mw=0.0, redshift=0.1)
        self.assertTrue(np.all(result <= self.flux))
        self.assertTrue(np.any(result < self.flux))

    def test_positive_av_mw_dims_flux(self):
        result = _perform_extinction(
            self.flux, self.angstroms, av_host=0.0, rv_host=3.1,
            av_mw=1.0, redshift=0.1)
        self.assertTrue(np.all(result <= self.flux))
        self.assertTrue(np.any(result < self.flux))

    def test_combined_extinction_greater_than_single(self):
        host_only = _perform_extinction(
            self.flux, self.angstroms, av_host=0.5, rv_host=3.1,
            av_mw=0.0, redshift=0.1)
        both = _perform_extinction(
            self.flux, self.angstroms, av_host=0.5, rv_host=3.1,
            av_mw=0.5, redshift=0.1)
        self.assertTrue(np.all(both <= host_only))

    def test_bad_host_law_raises(self):
        with self.assertRaises(ValueError):
            _perform_extinction(
                self.flux, self.angstroms, av_host=1.0, rv_host=3.1,
                av_mw=0.0, host_law='bad_law', redshift=0.1)

    def test_bad_mw_law_raises(self):
        with self.assertRaises(ValueError):
            _perform_extinction(
                self.flux, self.angstroms, av_host=0.0, rv_host=3.1,
                av_mw=1.0, mw_law='bad_law', redshift=0.1)

    def test_does_not_mutate_input(self):
        flux_copy = self.flux.copy()
        _perform_extinction(
            self.flux, self.angstroms, av_host=1.0, rv_host=3.1,
            av_mw=0.0, redshift=0.1)
        np.testing.assert_array_equal(self.flux, flux_copy)


# ---------------------------------------------------------------------------
# extinction_with_supernova_base_model
# ---------------------------------------------------------------------------

class TestExtinctionWithSupernovaBaseModel(unittest.TestCase):

    def test_zero_extinction_matches_base_model(self):
        """av_host=0 must reproduce the bare arnett output."""
        import redback.transient_models.supernova_models as sm
        kwargs = _base_kwargs()
        result_ext = extinction_with_supernova_base_model(
            TIME, av_host=0.0, base_model='arnett', **kwargs)
        result_base = sm.arnett(TIME, **{k: v for k, v in kwargs.items()
                                         if k != 'base_model'})
        np.testing.assert_allclose(result_ext, result_base, rtol=1e-10)

    def test_positive_extinction_dims_flux(self):
        kwargs = _base_kwargs()
        bright = extinction_with_supernova_base_model(
            TIME, av_host=0.0, base_model='arnett', **kwargs)
        dim = extinction_with_supernova_base_model(
            TIME, av_host=1.0, base_model='arnett', **kwargs)
        self.assertTrue(np.all(dim <= bright))

    def test_output_shape(self):
        kwargs = _base_kwargs()
        result = extinction_with_supernova_base_model(
            TIME, av_host=0.5, base_model='arnett', **kwargs)
        self.assertEqual(result.shape, TIME.shape)

    def test_output_is_finite(self):
        kwargs = _base_kwargs()
        result = extinction_with_supernova_base_model(
            TIME, av_host=0.5, base_model='arnett', **kwargs)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_bad_base_model_raises(self):
        kwargs = _base_kwargs()
        with self.assertRaises(ValueError):
            extinction_with_supernova_base_model(
                TIME, av_host=0.5, base_model='not_a_model', **kwargs)


# ---------------------------------------------------------------------------
# extinction_with_function
# ---------------------------------------------------------------------------

class TestExtinctionWithFunction(unittest.TestCase):

    def test_function_passthrough_zero_extinction(self):
        """Passing a callable with av_host=0 must match bare model output."""
        import redback.transient_models.supernova_models as sm
        kwargs = _base_kwargs()
        result_fn = extinction_with_function(
            TIME, av_host=0.0, base_model=sm.arnett, **kwargs)
        result_base = sm.arnett(TIME, **kwargs)
        np.testing.assert_allclose(result_fn, result_base, rtol=1e-10)

    def test_function_passthrough_with_extinction(self):
        """Callable with av_host>0 must dim the output vs av_host=0."""
        import redback.transient_models.supernova_models as sm
        kwargs = _base_kwargs()
        bright = extinction_with_function(TIME, av_host=0.0, base_model=sm.arnett, **kwargs)
        dim = extinction_with_function(TIME, av_host=1.0, base_model=sm.arnett, **kwargs)
        self.assertTrue(np.all(dim <= bright))


# ---------------------------------------------------------------------------
# No duplicate stellar_interaction function
# ---------------------------------------------------------------------------

class TestNoDuplicateFunction(unittest.TestCase):

    def test_stellar_interaction_in_model_library(self):
        """extinction_with_stellar_interaction_base_model must exist and be callable."""
        from redback.transient_models import extinction_models
        self.assertTrue(hasattr(extinction_models, 'extinction_with_stellar_interaction_base_model'))
        self.assertTrue(callable(extinction_models.extinction_with_stellar_interaction_base_model))

    def test_general_synchrotron_still_present(self):
        from redback.transient_models import extinction_models
        self.assertTrue(hasattr(extinction_models, 'extinction_with_general_synchrotron_base_model'))


# ---------------------------------------------------------------------------
# from_simulated_optical_data with upper limits
# ---------------------------------------------------------------------------

def _make_mock_df(include_nan_mag=True):
    """Build a mock DataFrame matching the redback simulated CSV format."""
    data = {
        "time (days)": [1.0, 2.0, 3.0, 4.0, 5.0],
        "time": [60500.0, 60501.0, 60502.0, 60503.0, 60504.0],
        "magnitude": [22.0, 23.0, np.nan if include_nan_mag else 24.5, 24.0, 25.0],
        "e_magnitude": [0.1, 0.2, 0.3, 0.4, 0.5],
        "band": ["g", "r", "g", "r", "g"],
        "bands": ["g", "r", "g", "r", "g"],
        "wavelength [Hz]": [6e14, 5e14, 6e14, 5e14, 6e14],
        "sncosmo_name": ["ztfg", "ztfr", "ztfg", "ztfr", "ztfg"],
        "flux(erg/cm2/s)": [1e-15, 2e-15, 0.0, 4e-15, 5e-15],
        "flux_error": [1e-16, 2e-16, 0.0, 4e-16, 5e-16],
        "flux_density(mjy)": [0.001, 0.002, 0.0, 0.004, 0.005],
        "flux_density_error": [0.0001, 0.0002, 0.0, 0.0004, 0.0005],
        "detected": [1, 1, 0, 1, 0],
        "limiting_magnitude": [25.0, 25.0, 24.5, 25.0, 26.0],
    }
    return pd.DataFrame(data)


class TestFromSimulatedOpticalDataUpperLimits(unittest.TestCase):

    @mock.patch("pandas.read_csv")
    def test_include_upper_limits_false_filters_nondetections(self, mock_read_csv):
        """Default (include_upper_limits=False) must filter out non-detections."""
        mock_read_csv.return_value = _make_mock_df()
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=False)
        self.assertEqual(len(instance.time), 3)  # only the 3 detected rows
        self.assertFalse(instance.has_upper_limits)

    @mock.patch("pandas.read_csv")
    def test_include_upper_limits_true_keeps_all(self, mock_read_csv):
        mock_read_csv.return_value = _make_mock_df()
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=True)
        self.assertEqual(len(instance.time), 5)
        self.assertTrue(instance.has_upper_limits)

    @mock.patch("pandas.read_csv")
    def test_detections_array_correct(self, mock_read_csv):
        mock_read_csv.return_value = _make_mock_df()
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=True)
        expected = np.array([True, True, False, True, False])
        np.testing.assert_array_equal(instance.detections, expected)

    @mock.patch("pandas.read_csv")
    def test_nan_magnitude_substituted_with_limiting_magnitude(self, mock_read_csv):
        """NaN mag for non-detections must be replaced by limiting_magnitude."""
        mock_read_csv.return_value = _make_mock_df(include_nan_mag=True)
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=True)
        # Row 2 had NaN magnitude, limiting_magnitude=24.5
        self.assertTrue(np.isfinite(instance.magnitude[2]))
        self.assertAlmostEqual(instance.magnitude[2], 24.5)

    @mock.patch("pandas.read_csv")
    def test_no_nan_upper_limits_after_fix(self, mock_read_csv):
        """After loading with include_upper_limits=True there must be no NaN y-values."""
        mock_read_csv.return_value = _make_mock_df(include_nan_mag=True)
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=True)
        ul_y = instance.y[instance.upper_limits]
        self.assertTrue(np.all(np.isfinite(ul_y)),
                        f"Upper limit y-values contain NaN: {ul_y}")

    @mock.patch("pandas.read_csv")
    def test_no_nan_substitution_when_magnitude_not_nan(self, mock_read_csv):
        """When magnitude is not NaN for non-detections, original value is kept."""
        mock_read_csv.return_value = _make_mock_df(include_nan_mag=False)
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=True)
        self.assertAlmostEqual(instance.magnitude[2], 24.5)

    @mock.patch("pandas.read_csv")
    def test_upper_limits_property_is_inverse_of_detections(self, mock_read_csv):
        mock_read_csv.return_value = _make_mock_df()
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=True)
        np.testing.assert_array_equal(instance.upper_limits, ~instance.detections)

    @mock.patch("pandas.read_csv")
    def test_no_detections_array_when_upper_limits_false(self, mock_read_csv):
        mock_read_csv.return_value = _make_mock_df()
        instance = redback.transient.Transient.from_simulated_optical_data(
            name="test", data_mode="magnitude", include_upper_limits=False)
        self.assertIsNone(instance._detections)


# ---------------------------------------------------------------------------
# Transient upper limit properties
# ---------------------------------------------------------------------------

class TestTransientUpperLimitProperties(unittest.TestCase):

    def _make_transient(self, detections=None):
        time = np.array([1.0, 2.0, 3.0, 4.0])
        magnitude = np.array([22.0, 23.0, 24.5, 25.0])
        magnitude_err = np.array([0.1, 0.2, 0.3, 0.4])
        bands = np.array(["lsstz", "lsstz", "lsstz", "lsstz"])
        return redback.transient.Transient(
            name="test", data_mode="magnitude",
            time=time, time_err=None, time_mjd=time + 60000,
            magnitude=magnitude, magnitude_err=magnitude_err,
            bands=bands, active_bands="all", optical_data=True,
            detections=detections)

    def test_has_upper_limits_false_when_no_detections_array(self):
        t = self._make_transient(detections=None)
        self.assertFalse(t.has_upper_limits)

    def test_has_upper_limits_false_when_all_detections(self):
        t = self._make_transient(detections=np.array([True, True, True, True]))
        self.assertFalse(t.has_upper_limits)

    def test_has_upper_limits_true_when_some_non_detections(self):
        t = self._make_transient(detections=np.array([True, True, False, False]))
        self.assertTrue(t.has_upper_limits)

    def test_upper_limits_is_complement_of_detections(self):
        det = np.array([True, False, True, False])
        t = self._make_transient(detections=det)
        np.testing.assert_array_equal(t.upper_limits, ~det)

    def test_detections_setter_wrong_length_raises(self):
        t = self._make_transient()
        with self.assertRaises(ValueError):
            t.detections = np.array([True, False])  # wrong length


if __name__ == '__main__':
    unittest.main()
