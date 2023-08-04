import numpy as np
import unittest

import redback


class TestTemperatureFloor(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.luminosity = np.array([1, 2, 3]) * 2e17
        self.vej = np.array([1, 2, 3])
        self.temperature_floor = 1
        self.temperature_floor_instance = redback.photosphere.TemperatureFloor(
            time=self.time, luminosity=self.luminosity, vej=self.vej, temperature_floor=self.temperature_floor)

    def tearDown(self) -> None:
        del self.time
        del self.luminosity
        del self.vej
        del self.temperature_floor
        del self.temperature_floor_instance

    def test_calculate_photosphere_properties(self):
        temperatures, r_photosphere = self.temperature_floor_instance.calculate_photosphere_properties()
        expected_temperatures = np.array([1.39250008, 1., 1.])
        expected_r_photosphere = np.array([8640000000.0, 2.36929531e10, 29017822836.350456])
        self.assertTrue(np.allclose(expected_temperatures, temperatures))
        self.assertTrue(np.allclose(expected_r_photosphere, r_photosphere))


class TestTDEPhotosphere(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.luminosity = np.array([1, 2, 3]) * 2e17
        self.mass_bh = 5
        self.mass_star = 2
        self.star_radius = 1
        self.tpeak = 1
        self.beta = 2
        self.rph_0 = 2
        self.lphoto = 2e17
        self.photosphere = redback.photosphere.TDEPhotosphere(
            time=self.time, luminosity=self.luminosity, mass_bh=self.mass_bh, mass_star=self.mass_star,
            star_radius=self.star_radius, tpeak=self.tpeak, beta=self.beta, rph_0=self.rph_0, lphoto=self.lphoto)

    def tearDown(self) -> None:
        del self.time
        del self.luminosity
        del self.mass_bh
        del self.mass_star
        del self.star_radius
        del self.tpeak
        del self.beta
        del self.rph_0
        del self.lphoto
        del self.photosphere

    def test_photosphere_properties(self):
        self.photosphere.calculate_photosphere_properties()
        expected_temperature = np.array([61.49734369, 73.13307867, 80.9350559])
        expected_r_photosphere = np.array([4429875.11415037, 4429875.11415037, 4429875.11415037])
        expected_rp = 47210508396.62691
        self.assertTrue(np.allclose(expected_temperature, self.photosphere.photosphere_temperature))
        self.assertTrue(np.allclose(expected_r_photosphere, self.photosphere.r_photosphere))
        self.assertEqual(expected_rp, self.photosphere.rp)


class TestDenseCore(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.luminosity = np.array([1, 2, 3]) * 2e17
        self.mej = 1
        self.vej = 100
        self.kappa = 10
        self.envelope_slope = 10
        self.dense_core = redback.photosphere.DenseCore(
            time=self.time, luminosity=self.luminosity, mej=self.mej,
            vej=self.vej, kappa=self.kappa, envelope_slope=self.envelope_slope)

    def tearDown(self) -> None:
        del self.time
        del self.luminosity
        del self.mej
        del self.vej
        del self.kappa
        del self.envelope_slope

    def test_calculate_photosphere_properties(self):
        expected_temperature = np.array([1.00000000e+05, 3.98642285e-02, 3.76813318e-02])
        expected_r_photosphere = np.array([1.67534478e+00, 1.49091357e+13, 2.04367737e+13])
        self.assertTrue(np.allclose(expected_temperature, self.dense_core.photosphere_temperature))
        self.assertTrue(np.allclose(expected_r_photosphere, self.dense_core.r_photosphere))
