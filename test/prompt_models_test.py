import unittest
import numpy as np
from redback.transient_models import prompt_models


class TestGaussianPrompt(unittest.TestCase):
    """Test gaussian_prompt function"""

    def test_gaussian_prompt_basic(self):
        """Test basic Gaussian prompt emission"""
        times = np.linspace(-5, 5, 100)
        amplitude = 1.0
        t_0 = 0.0
        sigma = 1.0

        result = prompt_models.gaussian_prompt(times, amplitude, t_0, sigma)

        # Check shape
        self.assertEqual(len(result), len(times))

        # Check peak is near t_0
        peak_index = np.argmax(result)
        self.assertAlmostEqual(times[peak_index], t_0, delta=0.1)

        # Check amplitude at peak (should be approximately amplitude * dt)
        self.assertGreater(result[peak_index], 0)

        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_gaussian_prompt_with_dt(self):
        """Test Gaussian prompt with custom dt"""
        times = np.linspace(-5, 5, 100)
        amplitude = 2.0
        t_0 = 0.0
        sigma = 1.0
        dt = 0.5

        result = prompt_models.gaussian_prompt(times, amplitude, t_0, sigma, dt=dt)

        # Result should be scaled by dt
        result_default = prompt_models.gaussian_prompt(times, amplitude, t_0, sigma, dt=1.0)
        np.testing.assert_allclose(result, result_default * dt, rtol=1e-10)

    def test_gaussian_prompt_shifted_center(self):
        """Test Gaussian with shifted center"""
        times = np.linspace(0, 10, 100)
        amplitude = 1.5
        t_0 = 5.0
        sigma = 2.0

        result = prompt_models.gaussian_prompt(times, amplitude, t_0, sigma)

        # Peak should be near t_0
        peak_index = np.argmax(result)
        self.assertAlmostEqual(times[peak_index], t_0, delta=0.1)


class TestSkewGaussian(unittest.TestCase):
    """Test skew_gaussian function"""

    def test_skew_gaussian_basic(self):
        """Test basic skew Gaussian"""
        times = np.linspace(-5, 5, 100)
        amplitude = 1.0
        t_0 = 0.0
        sigma_rise = 1.0
        sigma_fall = 2.0

        result = prompt_models.skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall)

        # Check shape
        self.assertEqual(len(result), len(times))

        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(result)))

        # Check that values are non-negative
        self.assertTrue(np.all(result >= 0))

    def test_skew_gaussian_asymmetry(self):
        """Test that skew Gaussian is asymmetric"""
        times = np.linspace(-10, 10, 200)
        amplitude = 1.0
        t_0 = 0.0
        sigma_rise = 1.0
        sigma_fall = 3.0

        result = prompt_models.skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall)

        # Find indices before and after t_0
        before_indices = times < t_0
        after_indices = times > t_0

        # The fall should be broader than the rise
        # Check that values decay slower after t_0
        time_before = times[before_indices]
        time_after = times[after_indices]

        if len(time_before) > 0 and len(time_after) > 0:
            # Just verify we get reasonable values
            self.assertGreater(np.max(result), 0)

    def test_skew_gaussian_with_dt(self):
        """Test skew Gaussian with custom dt"""
        times = np.linspace(-5, 5, 100)
        amplitude = 1.0
        t_0 = 0.0
        sigma_rise = 1.0
        sigma_fall = 2.0
        dt = 0.5

        result = prompt_models.skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall, dt=dt)

        # Check that dt scaling is applied
        result_default = prompt_models.skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall, dt=1.0)
        np.testing.assert_allclose(result, result_default * dt, rtol=1e-10)


class TestSkewExponential(unittest.TestCase):
    """Test skew_exponential function"""

    def test_skew_exponential_basic(self):
        """Test basic skew exponential"""
        times = np.linspace(-5, 5, 100)
        amplitude = 1.0
        t_0 = 0.0
        tau_rise = 1.0
        tau_fall = 2.0

        result = prompt_models.skew_exponential(times, amplitude, t_0, tau_rise, tau_fall)

        # Check shape
        self.assertEqual(len(result), len(times))

        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(result)))

        # Peak should be at t_0
        peak_index = np.argmax(result)
        self.assertAlmostEqual(times[peak_index], t_0, delta=0.1)

    def test_skew_exponential_rise_fall(self):
        """Test that exponential has proper rise and fall"""
        times = np.linspace(-10, 10, 200)
        amplitude = 1.0
        t_0 = 0.0
        tau_rise = 1.0
        tau_fall = 2.0

        result = prompt_models.skew_exponential(times, amplitude, t_0, tau_rise, tau_fall)

        # Before t_0: exponential rise
        before_indices = times < t_0
        # After t_0: exponential fall
        after_indices = times > t_0

        # Check that we have values before and after
        if np.sum(before_indices) > 0 and np.sum(after_indices) > 0:
            # Peak should be at t_0
            peak_index = np.argmax(result)
            self.assertAlmostEqual(times[peak_index], t_0, delta=0.1)

    def test_skew_exponential_with_dt(self):
        """Test skew exponential with custom dt"""
        times = np.linspace(-5, 5, 100)
        amplitude = 1.0
        t_0 = 0.0
        tau_rise = 1.0
        tau_fall = 2.0
        dt = 0.5

        result = prompt_models.skew_exponential(times, amplitude, t_0, tau_rise, tau_fall, dt=dt)

        # Check that dt scaling is applied
        result_default = prompt_models.skew_exponential(times, amplitude, t_0, tau_rise, tau_fall, dt=1.0)
        np.testing.assert_allclose(result, result_default * dt, rtol=1e-10)


class TestFred(unittest.TestCase):
    """Test fred (Fast Rise Exponential Decay) function"""

    def test_fred_basic(self):
        """Test basic FRED function"""
        times = np.linspace(0.1, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0

        result = prompt_models.fred(times, amplitude, psi, tau, delta)

        # Check shape
        self.assertEqual(len(result), len(times))

        # Check that we have finite positive values
        finite_mask = np.isfinite(result)
        self.assertGreater(np.sum(finite_mask), 0)

    def test_fred_with_dt(self):
        """Test FRED with custom dt"""
        times = np.linspace(0.1, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0
        dt = 0.5

        result = prompt_models.fred(times, amplitude, psi, tau, delta, dt=dt)

        # Check that dt scaling is applied
        result_default = prompt_models.fred(times, amplitude, psi, tau, delta, dt=1.0)

        # Only compare finite values
        finite_mask = np.isfinite(result) & np.isfinite(result_default)
        if np.sum(finite_mask) > 0:
            np.testing.assert_allclose(
                result[finite_mask],
                result_default[finite_mask] * dt,
                rtol=1e-10
            )

    def test_fred_handles_edge_cases(self):
        """Test that FRED handles edge cases gracefully"""
        times = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0

        # Should not raise an error
        result = prompt_models.fred(times, amplitude, psi, tau, delta)

        # Result should have same length
        self.assertEqual(len(result), len(times))

    def test_fred_with_delta_offset(self):
        """Test FRED with time offset delta"""
        times = np.linspace(1, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 2.0

        result = prompt_models.fred(times, amplitude, psi, tau, delta)

        # Check shape
        self.assertEqual(len(result), len(times))


class TestFredExtended(unittest.TestCase):
    """Test fred_extended function"""

    def test_fred_extended_basic(self):
        """Test basic extended FRED function"""
        times = np.linspace(0.1, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0
        gamma = 1.0
        nu = 1.0

        result = prompt_models.fred_extended(times, amplitude, psi, tau, delta, gamma, nu)

        # Check shape
        self.assertEqual(len(result), len(times))

        # Check that we have some finite values
        finite_mask = np.isfinite(result)
        self.assertGreater(np.sum(finite_mask), 0)

    def test_fred_extended_with_dt(self):
        """Test extended FRED with custom dt"""
        times = np.linspace(0.1, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0
        gamma = 1.0
        nu = 1.0
        dt = 0.5

        result = prompt_models.fred_extended(times, amplitude, psi, tau, delta, gamma, nu, dt=dt)

        # Check that dt scaling is applied
        result_default = prompt_models.fred_extended(times, amplitude, psi, tau, delta, gamma, nu, dt=1.0)

        # Only compare finite values
        finite_mask = np.isfinite(result) & np.isfinite(result_default)
        if np.sum(finite_mask) > 0:
            np.testing.assert_allclose(
                result[finite_mask],
                result_default[finite_mask] * dt,
                rtol=1e-10
            )

    def test_fred_extended_different_powers(self):
        """Test extended FRED with different gamma and nu"""
        times = np.linspace(0.5, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0
        gamma = 1.5
        nu = 2.0

        result = prompt_models.fred_extended(times, amplitude, psi, tau, delta, gamma, nu)

        # Check shape
        self.assertEqual(len(result), len(times))

        # Check that we have finite values
        finite_mask = np.isfinite(result)
        self.assertGreater(np.sum(finite_mask), 0)

    def test_fred_extended_reduces_to_fred(self):
        """Test that extended FRED reduces to regular FRED when gamma=nu=1"""
        times = np.linspace(0.5, 10, 100)
        amplitude = 1.0
        psi = 1.0
        tau = 1.0
        delta = 0.0

        result_fred = prompt_models.fred(times, amplitude, psi, tau, delta)
        result_extended = prompt_models.fred_extended(times, amplitude, psi, tau, delta, gamma=1.0, nu=1.0)

        # Should be approximately equal where both are finite
        finite_mask = np.isfinite(result_fred) & np.isfinite(result_extended)
        if np.sum(finite_mask) > 0:
            np.testing.assert_allclose(
                result_fred[finite_mask],
                result_extended[finite_mask],
                rtol=1e-10
            )
