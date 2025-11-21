import unittest
import numpy as np
from redback.transient_models import fireball_models


class TestPredeceleration(unittest.TestCase):
    """Test predeceleration function"""

    def test_predeceleration_basic(self):
        """Test basic predeceleration power law"""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aa = 1.0
        mm = 3.0
        t0 = 0.0

        result = fireball_models.predeceleration(time, aa, mm, t0)

        # Check shape
        self.assertEqual(len(result), len(time))

        # Check power law behavior: (t-t0)^mm
        expected = aa * (time - t0)**mm
        np.testing.assert_allclose(result, expected)

    def test_predeceleration_with_t0_offset(self):
        """Test predeceleration with time offset"""
        time = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        aa = 2.0
        mm = 3.0
        t0 = 1.0

        result = fireball_models.predeceleration(time, aa, mm, t0)

        # Check power law behavior with offset
        expected = aa * (time - t0)**mm
        np.testing.assert_allclose(result, expected)

    def test_predeceleration_different_powers(self):
        """Test predeceleration with different power law indices"""
        time = np.linspace(1, 10, 50)
        aa = 1.5
        t0 = 0.0

        # Test different values of mm
        for mm in [1.0, 2.0, 3.0, 4.0, -2.0]:
            result = fireball_models.predeceleration(time, aa, mm, t0)
            expected = aa * (time - t0)**mm
            np.testing.assert_allclose(result, expected)

    def test_predeceleration_with_kwargs(self):
        """Test that predeceleration accepts kwargs"""
        time = np.array([1.0, 2.0, 3.0])
        aa = 1.0
        mm = 3.0
        t0 = 0.0

        # Should accept additional kwargs without error
        result = fireball_models.predeceleration(time, aa, mm, t0, extra_param=1.0)

        expected = aa * (time - t0)**mm
        np.testing.assert_allclose(result, expected)

    def test_predeceleration_negative_mm(self):
        """Test predeceleration with negative power law index"""
        time = np.array([1.0, 2.0, 3.0, 4.0])
        aa = 1.0
        mm = -2.0
        t0 = 0.0

        result = fireball_models.predeceleration(time, aa, mm, t0)

        # Should decay as (t-t0)^(-2)
        expected = aa * (time - t0)**mm
        np.testing.assert_allclose(result, expected)


class TestOneComponentFireballModel(unittest.TestCase):
    """Test one_component_fireball_model function"""

    def test_one_component_basic(self):
        """Test basic one component fireball model"""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a_1 = 1.0
        alpha_1 = 2.0

        result = fireball_models.one_component_fireball_model(time, a_1, alpha_1)

        # Check shape
        self.assertEqual(len(result), len(time))

        # Check power law behavior: a_1 * t^alpha_1
        expected = a_1 * time**alpha_1
        np.testing.assert_allclose(result, expected)

    def test_one_component_different_amplitudes(self):
        """Test with different amplitudes"""
        time = np.linspace(1, 10, 50)
        alpha_1 = -1.2

        for a_1 in [0.5, 1.0, 2.0, 5.0]:
            result = fireball_models.one_component_fireball_model(time, a_1, alpha_1)
            expected = a_1 * time**alpha_1
            np.testing.assert_allclose(result, expected)

    def test_one_component_different_exponents(self):
        """Test with different power law exponents"""
        time = np.linspace(1, 10, 50)
        a_1 = 1.5

        for alpha_1 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            result = fireball_models.one_component_fireball_model(time, a_1, alpha_1)
            expected = a_1 * time**alpha_1
            np.testing.assert_allclose(result, expected)

    def test_one_component_decay(self):
        """Test decay behavior with negative exponent"""
        time = np.array([1.0, 2.0, 4.0, 8.0])
        a_1 = 10.0
        alpha_1 = -1.0

        result = fireball_models.one_component_fireball_model(time, a_1, alpha_1)

        # With negative exponent, should decay
        # Values should decrease
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])
        self.assertGreater(result[2], result[3])

    def test_one_component_growth(self):
        """Test growth behavior with positive exponent"""
        time = np.array([1.0, 2.0, 4.0, 8.0])
        a_1 = 1.0
        alpha_1 = 2.0

        result = fireball_models.one_component_fireball_model(time, a_1, alpha_1)

        # With positive exponent, should grow
        # Values should increase
        self.assertLess(result[0], result[1])
        self.assertLess(result[1], result[2])
        self.assertLess(result[2], result[3])

    def test_one_component_with_kwargs(self):
        """Test that one_component_fireball_model accepts kwargs"""
        time = np.array([1.0, 2.0, 3.0])
        a_1 = 1.0
        alpha_1 = 2.0

        # Should accept additional kwargs without error
        result = fireball_models.one_component_fireball_model(time, a_1, alpha_1, extra_param=1.0)

        expected = a_1 * time**alpha_1
        np.testing.assert_allclose(result, expected)

    def test_one_component_zero_exponent(self):
        """Test with zero exponent (constant output)"""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a_1 = 3.0
        alpha_1 = 0.0

        result = fireball_models.one_component_fireball_model(time, a_1, alpha_1)

        # With alpha_1 = 0, t^0 = 1, so result should be a_1
        expected = np.full_like(time, a_1)
        np.testing.assert_allclose(result, expected)
