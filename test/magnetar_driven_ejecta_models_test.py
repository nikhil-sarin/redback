import unittest
from unittest.mock import MagicMock

import astropy.units as uu
import numpy as np

from redback.constants import day_to_s, speed_of_light
from redback.transient_models.magnetar_driven_ejecta_models import (
    _ejecta_dynamics_and_interaction,
)
from redback.transient_models.magnetar_models import magnetar_only
from redback.utils import velocity_from_lorentz_factor
from redback.transient_models.supernova_models import (
    general_magnetar_driven_supernova,
    general_magnetar_driven_supernova_bolometric,
)


class TestEjectaDynamics(unittest.TestCase):

    parameters = dict(
        mej=5.0,
        beta=0.02,
        ejecta_radius=1.0e11,
        kappa=0.1,
        n_ism=1.0e-5,
        pair_cascade_switch=False,
        use_gamma_ray_opacity=True,
        kappa_gamma=0.1,
        use_r_process=False,
        f_nickel=0.1,
    )

    def evaluate(self, time, **overrides):
        parameters = self.parameters | overrides
        return _ejecta_dynamics_and_interaction(
            time=time,
            magnetar_luminosity=np.full_like(time, 1.0e45),
            **parameters,
        )

    def test_returns_result_instance_with_finite_arrays(self):
        time = np.geomspace(1.0, 1.0e7, 100)
        result = self.evaluate(time)

        self.assertIsInstance(result, tuple)
        self.assertEqual(result._fields[-1], 'r_photosphere')
        for field in result._fields:
            value = getattr(result, field)
            self.assertTrue(np.all(np.isfinite(value)), field)
        np.testing.assert_array_equal(result.r_photosphere, result.radius)

    def test_solution_is_independent_of_output_grid(self):
        coarse_time = np.geomspace(1.0, 1.0e7, 80)
        fine_time = np.geomspace(1.0, 1.0e7, 4 * (len(coarse_time) - 1) + 1)
        coarse = self.evaluate(coarse_time)
        fine = self.evaluate(fine_time)

        np.testing.assert_allclose(
            coarse.bolometric_luminosity, fine.bolometric_luminosity[::4],
            rtol=2e-5)
        np.testing.assert_allclose(coarse.radius, fine.radius[::4], rtol=2e-5)

    def test_magnetar_source_grid_is_converged(self):
        coarse_time = np.geomspace(1.0, 1.0e8, 500)
        fine_time = np.geomspace(1.0, 1.0e8, 2000)
        coarse = _ejecta_dynamics_and_interaction(
            time=coarse_time,
            magnetar_luminosity=magnetar_only(
                coarse_time, l0=1.0e46, tau=1.0e6, nn=3.0),
            **self.parameters,
        )
        fine = _ejecta_dynamics_and_interaction(
            time=fine_time,
            magnetar_luminosity=magnetar_only(
                fine_time, l0=1.0e46, tau=1.0e6, nn=3.0),
            **self.parameters,
        )
        fine_luminosity = np.interp(
            coarse_time, fine_time, fine.bolometric_luminosity)
        luminous = fine_luminosity > 1.0e-4 * np.max(fine_luminosity)

        np.testing.assert_allclose(
            coarse.bolometric_luminosity[luminous],
            fine_luminosity[luminous], rtol=1e-3)

    def test_pair_cascade_uses_physical_ejecta_velocity(self):
        time = np.geomspace(day_to_s, 100 * day_to_s, 60)
        without_cascade = self.evaluate(time)
        with_cascade = self.evaluate(time, pair_cascade_switch=True)
        velocity = velocity_from_lorentz_factor(with_cascade.lorentz_factor)
        tlife_t = (
            (0.6 / (1 - 0.5)) * (0.05 / 0.1) ** 0.5
            * (1.0e45 / 1.0e45) ** 0.5
            * (velocity / (0.3 * speed_of_light)) ** 0.5
            * (time / day_to_s) ** -0.5
        )

        np.testing.assert_allclose(
            with_cascade.bolometric_luminosity,
            without_cascade.bolometric_luminosity / (1 + tlife_t),
            rtol=2e-5)

    def test_rejects_superluminal_initial_velocity(self):
        time = np.geomspace(1.0, 100.0, 10)
        with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
            self.evaluate(time, beta=1.0)


class TestGeneralMagnetarDrivenSupernova(unittest.TestCase):

    parameters = dict(
        mej=5.0,
        E_sn=1.0e51,
        kappa=0.1,
        l0=1.0e45,
        tau_sd=1.0e6,
        nn=3.0,
        kappa_gamma=0.1,
        f_nickel=0.1,
    )

    def test_bolometric_dynamics_output_is_an_instance(self):
        result = general_magnetar_driven_supernova_bolometric(
            np.array([1.0, 10.0]), output_format='dynamics_output',
            **self.parameters)

        self.assertIsInstance(result, tuple)
        self.assertGreater(len(result.time), 500)
        self.assertEqual(result.v_ej.shape, result.time.shape)
        self.assertEqual(result.radius.shape, result.time.shape)
        self.assertTrue(np.isfinite(result.erad_total))

    def test_initial_radius_is_configurable(self):
        result = general_magnetar_driven_supernova_bolometric(
            np.array([1.0, 10.0]), output_format='dynamics_output',
            ejecta_radius=3.0e12, **self.parameters)

        self.assertEqual(result.radius[0], 3.0e12)

    def test_rejects_too_small_dynamics_grid(self):
        with self.assertRaisesRegex(ValueError, "at least 100"):
            general_magnetar_driven_supernova_bolometric(
                np.array([1.0, 10.0]), output_format='dynamics_output',
                dynamics_resolution=99, **self.parameters)

    def test_rejects_non_integral_dynamics_grid(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            general_magnetar_driven_supernova_bolometric(
                np.array([1.0, 10.0]), output_format='dynamics_output',
                dynamics_resolution=1000.5, **self.parameters)

    def test_photometric_model_uses_custom_cosmology_and_evolved_radius(self):
        cosmology = MagicMock()
        cosmology.luminosity_distance.return_value = 1.0e27 * uu.cm
        result = general_magnetar_driven_supernova(
            np.array([1.0, 10.0]), redshift=0.1,
            output_format='dynamics_output', temperature_floor=1000.0,
            cosmology=cosmology, **self.parameters)

        cosmology.luminosity_distance.assert_called_once_with(0.1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result.photosphere_temperature.shape, result.time.shape)
        np.testing.assert_allclose(result.r_photosphere[:100], result.radius[:100])


if __name__ == '__main__':
    unittest.main()
