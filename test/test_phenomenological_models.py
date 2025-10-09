"""
Tests for redback.transient_models.phenomenological_models module.
"""
import functools
import jax.numpy as jnp
import numpy as np
import pytest

from jax import jit

from redback.transient_models import phenomenological_models


class TestPhenomenologicalModelsModule:
    """Test class for phenomenological_models module functionality."""

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_smooth_exponential_powerlaw(self, backend):
        """Test the smooth_exponential_powerlaw function."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        if backend == "jax":
            func = jit(functools.partial(phenomenological_models.smooth_exponential_powerlaw, xnp=jnp))
            time = jnp.array(time)
        else:
            func = phenomenological_models.smooth_exponential_powerlaw

        # Test a few different settings against ground truth from the non-JAX implementation.
        result = func(
            time, 
            1.0,  # a_1
            3.0,  # tpeak
            2.0,  # alpha_1
            -1.0,  # alpha_2
            1.0,  # smoothing_factor
        )
        assert np.allclose(result, np.array([0.4, 0.76923077, 1.0, 1.12, 1.17647059]), atol=1e-5)

        # Test a few different settings against ground truth from the non-JAX implementation.
        result = func(
            time, 
            2.0,  # a_1
            2.5,  # tpeak
            1.5,  # alpha_1
            -1.2,  # alpha_2
            5.0,  # smoothing_factor
        )
        assert np.allclose(result, np.array([0.50654, 1.54579, 1.74911, 1.16408, 0.87522]), atol=1e-5)

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_exp_rise_powerlaw_decline(self, backend):
        """Test the exp_rise_powerlaw_decline function."""
        time = np.array([1.0, 5.0, 10.0, 20.0])
        m_peak = np.array([10.0, 15.0, 20.0])

        if backend == "jax":
            func = jit(functools.partial(phenomenological_models.exp_rise_powerlaw_decline, xnp=jnp))
            time = jnp.array(time)
            m_peak = jnp.array(m_peak)
        else:
            func = phenomenological_models.exp_rise_powerlaw_decline

        # Test a few different settings against ground truth from the non-JAX implementation.
        result = func(
            time,
            0.0,  # t0
            18.0,  # m_peak (single value)
            3.0,  # tau_rise
            1.5,  # alpha
            8.0,  # t_peak
            delta=0.25,
        )
        expected = np.array([20.52861, 18.99819, 18.23379, 19.49224])
        assert np.allclose(result, expected, atol=1e-5)

        result = func(
            time, 
            0.0,  # t0
            m_peak,  # m_peak (array)
            2.0,  # tau_rise
            1.0,  # alpha
            10.0,  # t_peak
        )
        expected = np.array(
            [
                [14.69053, 19.69053, 24.69053],
                [12.30165, 17.30165, 22.30165],
                [10.0, 15.0, 20.0],
                [10.64137, 15.64137, 20.64137],
            ]
        )
        assert np.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_bazin_sne(self, backend):
        """Test the bazin_sne function."""
        time = np.array([1.0, 5.0, 10.0, 20.0])

        if backend == "jax":
            func = jit(functools.partial(phenomenological_models.bazin_sne, xnp=jnp))
            time = jnp.array(time)
        else:
            func = phenomenological_models.bazin_sne

        # Test a few different settings against ground truth from the non-JAX implementation.
        result = func(
            time,
            1.0,  # aa normalization
            2.0,  # bb additive constant
            0.0,  # t0
            4.0,  # tau_rise
            10.0,  # tau_fall
        )
        expected = np.array([2.50867833, 2.4714562 , 2.33997278, 2.1344295 ])
        assert np.allclose(result, expected, atol=1e-5)

        aa = np.array([0.0, 1.0, 2.0])
        bb = np.array([5.0, 6.0, 7.0])

        result = phenomenological_models.bazin_sne(
            time,
            aa,
            bb,
            0.0,  # t0
            3.0,  # tau_rise
            12.0,  # tau_fall
        )
        expected = np.array(
            [
                [5.0, 5.0, 5.0, 5.0],
                [6.53599, 6.55451, 6.41963, 6.18864],
                [8.07198, 8.10902, 7.83926, 7.37727],
            ]
        )
        assert np.allclose(result, expected, atol=1e-5)
