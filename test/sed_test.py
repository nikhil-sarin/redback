import unittest
from unittest.mock import MagicMock

import astropy.units as uu
import astropy
import numpy as np

import redback.sed


class TestCutoffBlackbody(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.luminosity = np.array([1, 2, 3]) * 2e17
        self.temperature = np.array([3000, 2000, 1000])
        self.r_photosphere = np.array([1, 2, 3]) * 1e10
        self.frequency = np.array([1, 2, 3])
        self.luminosity_distance = 1e15
        self.cutoff_wavelength = 1e26
        self.sed = redback.sed.CutoffBlackbody(
            time=self.time, temperature=self.temperature, luminosity=self.luminosity, r_photosphere=self.r_photosphere,
            frequency=self.frequency, luminosity_distance=self.luminosity_distance,
            cutoff_wavelength=self.cutoff_wavelength)

    def tearDown(self) -> None:
        del self.time
        del self.luminosity
        del self.temperature
        del self.r_photosphere
        del self.frequency
        del self.luminosity_distance
        del self.cutoff_wavelength
        del self.sed

    def test_flux_density(self):
        expected_flux_density = np.array([-1.89280803e-61,  3.02849285e-60, -1.02140923e-59])
        actual_flux_density = np.array([q.value for q in self.sed.flux_density])
        self.assertTrue(np.allclose(expected_flux_density, actual_flux_density, atol=1e-80))

    def test_flux_density_units(self):
        self.assertEqual(uu.mJy, self.sed.flux_density.unit)

    def test_sed(self):
        expected_sed = np.array([-7.93406459e-75, 5.07780134e-73, -3.85328781e-72])
        self.assertTrue(np.allclose(expected_sed, self.sed.sed, atol=1e-90))


class TestBlackBody(unittest.TestCase):

    def setUp(self) -> None:
        self.temperature = np.array([3000, 2000, 1000])
        self.r_photosphere = np.array([1, 2, 3]) * 1e10
        self.frequency = np.array([1, 2, 3])
        self.luminosity_distance = 1e15
        self.sed = redback.sed.Blackbody(
            temperature=self.temperature, r_photosphere=self.r_photosphere,
            frequency=self.frequency, luminosity_distance=self.luminosity_distance)

    def tearDown(self) -> None:
        del self.temperature
        del self.r_photosphere
        del self.frequency
        del self.luminosity_distance
        del self.sed

    def test_flux_density(self):
        expected_flux_density = np.array([2.89562955e-43, 3.08867152e-42, 7.81819978e-42])
        actual_flux_density = np.array([q.value for q in self.sed.flux_density])
        self.assertTrue(np.allclose(expected_flux_density, actual_flux_density, atol=1e-50))

    def test_flux_density_units(self):
        self.assertEqual(uu.erg * uu.Hz**3 * uu.s**3 / uu.cm**2, self.sed.flux_density.unit)


class TestSynchrotron(unittest.TestCase):

    def setUp(self) -> None:
        self.frequency = np.array([1, 2, 3])
        self.luminosity_distance = 1e15
        self.pp = 3
        self.nu_max = 2.5
        self.source_radius = 1
        self.f0 = 1.5
        self.sed = redback.sed.Synchrotron(
            frequency=self.frequency, luminosity_distance=self.luminosity_distance, pp=self.pp, nu_max=self.nu_max,
            source_radius=self.source_radius, f0=self.f0)

    def tearDown(self) -> None:
        del self.frequency
        del self.luminosity_distance
        del self.pp
        del self.nu_max
        del self.source_radius
        del self.f0
        del self.sed

    def test_flux_density(self):
        expected_flux_density = np.array([8.05267394e-33, 4.55528028e-32, 6.55328283e-31])
        actual_flux_density = np.array([q.value for q in self.sed.flux_density])
        self.assertTrue(np.allclose(expected_flux_density, actual_flux_density, atol=1e-40))

    def test_flux_density_units(self):
        self.assertEqual(uu.mJy, self.sed.flux_density.unit)

    def test_sed(self):
        expected_sed = np.array([3.37543132e-46, 7.63772920e-45, 2.47223973e-43])
        self.assertTrue(np.allclose(expected_sed, self.sed.sed, atol=1e-50))
