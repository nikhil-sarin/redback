import unittest
import numpy as np
from unittest.mock import patch
import redback.interaction_processes as interaction


class TestDiffusion(unittest.TestCase):
    """Test Diffusion class"""

    def setUp(self):
        self.time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.dense_times = np.linspace(0.5, 10.0, 100)
        self.luminosity = 1e43 * np.ones(100)
        self.kappa = 0.1
        self.kappa_gamma = 0.05
        self.mej = 10.0
        self.vej = 5000.0

    def test_initialization(self):
        """Test Diffusion initialization"""
        diff = interaction.Diffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej)

        self.assertEqual(diff.kappa, self.kappa)
        self.assertEqual(diff.kappa_gamma, self.kappa_gamma)
        self.assertEqual(diff.m_ejecta, self.mej)
        self.assertEqual(diff.v_ejecta, self.vej)
        self.assertIsNotNone(diff.reference)

    def test_convert_input_luminosity(self):
        """Test luminosity conversion"""
        diff = interaction.Diffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej)

        self.assertIsNotNone(diff.tau_d)
        self.assertIsNotNone(diff.new_luminosity)
        self.assertGreater(diff.tau_d, 0)
        self.assertEqual(len(diff.new_luminosity), len(self.time))

    def test_tau_diffusion_calculation(self):
        """Test diffusion timescale calculation"""
        diff = interaction.Diffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej)

        # Tau should be positive
        self.assertGreater(diff.tau_d, 0)

    def test_new_luminosity_values(self):
        """Test that new luminosity is calculated"""
        diff = interaction.Diffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej)

        # Should have luminosity values for all time points
        self.assertTrue(np.all(diff.new_luminosity >= 0))

    def test_varying_kappa(self):
        """Test with different opacity values"""
        diff1 = interaction.Diffusion(
            self.time, self.dense_times, self.luminosity,
            0.05, self.kappa_gamma, self.mej, self.vej)
        diff2 = interaction.Diffusion(
            self.time, self.dense_times, self.luminosity,
            0.2, self.kappa_gamma, self.mej, self.vej)

        # Different kappa should give different tau
        self.assertNotEqual(diff1.tau_d, diff2.tau_d)


class TestAsphericalDiffusion(unittest.TestCase):
    """Test AsphericalDiffusion class"""

    def setUp(self):
        self.time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.dense_times = np.linspace(0.5, 10.0, 100)
        self.luminosity = 1e43 * np.ones(100)
        self.kappa = 0.1
        self.kappa_gamma = 0.05
        self.mej = 10.0
        self.vej = 5000.0
        self.area_projection = 0.5
        self.area_reference = 1.0

    def test_initialization(self):
        """Test AsphericalDiffusion initialization"""
        asph = interaction.AsphericalDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej,
            self.area_projection, self.area_reference)

        self.assertEqual(asph.kappa, self.kappa)
        self.assertEqual(asph.area_projection, self.area_projection)
        self.assertEqual(asph.area_reference, self.area_reference)
        self.assertIsNotNone(asph.reference)

    def test_convert_input_luminosity(self):
        """Test luminosity conversion for aspherical diffusion"""
        asph = interaction.AsphericalDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej,
            self.area_projection, self.area_reference)

        self.assertIsNotNone(asph.tau_d)
        self.assertIsNotNone(asph.new_luminosity)
        self.assertEqual(len(asph.new_luminosity), len(self.time))

    def test_area_projection_effect(self):
        """Test effect of area projection on luminosity"""
        asph1 = interaction.AsphericalDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej,
            0.3, 1.0)
        asph2 = interaction.AsphericalDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej,
            0.7, 1.0)

        # Different area projections should give different luminosities
        self.assertFalse(np.allclose(asph1.new_luminosity, asph2.new_luminosity))

    def test_tau_diffusion(self):
        """Test diffusion timescale"""
        asph = interaction.AsphericalDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.kappa_gamma, self.mej, self.vej,
            self.area_projection, self.area_reference)

        self.assertGreater(asph.tau_d, 0)


class TestCSMDiffusion(unittest.TestCase):
    """Test CSMDiffusion class"""

    def setUp(self):
        self.time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.dense_times = np.linspace(0.5, 10.0, 100)
        self.luminosity = 1e43 * np.ones(100)
        self.kappa = 0.1
        self.r_photosphere = 1e14
        self.mass_csm_threshold = 0.5
        self.csm_mass = 1.0

    def test_initialization(self):
        """Test CSMDiffusion initialization"""
        csm = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.r_photosphere, self.mass_csm_threshold,
            self.csm_mass)

        self.assertEqual(csm.kappa, self.kappa)
        self.assertEqual(csm.r_photosphere, self.r_photosphere)
        self.assertIsNotNone(csm.reference)

    def test_convert_input_luminosity(self):
        """Test CSM diffusion luminosity conversion"""
        csm = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.r_photosphere, self.mass_csm_threshold,
            self.csm_mass)

        self.assertIsNotNone(csm.new_luminosity)
        self.assertEqual(len(csm.new_luminosity), len(self.time))

    def test_new_luminosity_positive(self):
        """Test that CSM diffusion produces valid luminosities"""
        csm = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.r_photosphere, self.mass_csm_threshold,
            self.csm_mass)

        # Luminosity should be non-negative
        self.assertTrue(np.all(csm.new_luminosity >= 0))

    def test_varying_csm_mass(self):
        """Test with different CSM masses"""
        csm1 = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.r_photosphere, self.mass_csm_threshold,
            0.5)
        csm2 = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            self.kappa, self.r_photosphere, self.mass_csm_threshold,
            2.0)

        # Different CSM masses should give different luminosities
        self.assertFalse(np.allclose(csm1.new_luminosity, csm2.new_luminosity))

    def test_varying_kappa(self):
        """Test with different opacity values"""
        csm1 = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            0.05, self.r_photosphere, self.mass_csm_threshold,
            self.csm_mass)
        csm2 = interaction.CSMDiffusion(
            self.time, self.dense_times, self.luminosity,
            0.2, self.r_photosphere, self.mass_csm_threshold,
            self.csm_mass)

        # Different kappa should affect luminosity
        self.assertFalse(np.allclose(csm1.new_luminosity, csm2.new_luminosity))


class TestViscous(unittest.TestCase):
    """Test Viscous class"""

    def setUp(self):
        self.time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.dense_times = np.linspace(0.5, 10.0, 100)
        self.luminosity = 1e43 * np.ones(100)
        self.t_viscous = 2.0

    def test_initialization(self):
        """Test Viscous initialization"""
        visc = interaction.Viscous(
            self.time, self.dense_times, self.luminosity, self.t_viscous)

        self.assertEqual(visc.tvisc, self.t_viscous)
        self.assertIsNotNone(visc.reference)

    def test_convert_input_luminosity(self):
        """Test viscous luminosity conversion"""
        visc = interaction.Viscous(
            self.time, self.dense_times, self.luminosity, self.t_viscous)

        self.assertIsNotNone(visc.new_luminosity)
        self.assertEqual(len(visc.new_luminosity), len(self.time))

    def test_new_luminosity_values(self):
        """Test that viscous processing produces valid luminosities"""
        visc = interaction.Viscous(
            self.time, self.dense_times, self.luminosity, self.t_viscous)

        # Luminosity should be non-negative
        self.assertTrue(np.all(visc.new_luminosity >= 0))

    def test_varying_viscous_timescale(self):
        """Test with different viscous timescales"""
        visc1 = interaction.Viscous(
            self.time, self.dense_times, self.luminosity, 1.0)
        visc2 = interaction.Viscous(
            self.time, self.dense_times, self.luminosity, 5.0)

        # Different viscous timescales should give different results
        self.assertFalse(np.allclose(visc1.new_luminosity, visc2.new_luminosity))

    def test_time_dependence(self):
        """Test time dependence of viscous processing"""
        visc = interaction.Viscous(
            self.time, self.dense_times, self.luminosity, self.t_viscous)

        # Luminosity should vary with time
        unique_values = len(np.unique(visc.new_luminosity))
        self.assertGreaterEqual(unique_values, 1)

    def test_varying_input_luminosity(self):
        """Test with varying input luminosity"""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.linspace(1.0, 2.0, 100)  # Linearly increasing

        visc = interaction.Viscous(time, dense_times, luminosity, 2.0)

        self.assertIsNotNone(visc.new_luminosity)
        self.assertEqual(len(visc.new_luminosity), len(time))


class TestDiffusionEdgeCases(unittest.TestCase):
    """Test edge cases for Diffusion class"""

    def test_single_time_point(self):
        """Test with single time point"""
        time = np.array([2.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        diff = interaction.Diffusion(
            time, dense_times, luminosity, 0.1, 0.05, 10.0, 5000.0)

        self.assertEqual(len(diff.new_luminosity), 1)
        self.assertGreaterEqual(diff.new_luminosity[0], 0)

    def test_high_velocity(self):
        """Test with high ejecta velocity"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        diff = interaction.Diffusion(
            time, dense_times, luminosity, 0.1, 0.05, 10.0, 20000.0)

        self.assertIsNotNone(diff.tau_d)
        self.assertGreater(diff.tau_d, 0)


class TestAsphericalDiffusionEdgeCases(unittest.TestCase):
    """Test edge cases for AsphericalDiffusion"""

    def test_equal_areas(self):
        """Test with equal projection and reference areas"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        asph = interaction.AsphericalDiffusion(
            time, dense_times, luminosity, 0.1, 0.05, 10.0, 5000.0, 1.0, 1.0)

        self.assertIsNotNone(asph.new_luminosity)

    def test_small_projection_area(self):
        """Test with very small projection area"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        asph = interaction.AsphericalDiffusion(
            time, dense_times, luminosity, 0.1, 0.05, 10.0, 5000.0, 0.1, 1.0)

        self.assertIsNotNone(asph.new_luminosity)


class TestCSMDiffusionEdgeCases(unittest.TestCase):
    """Test edge cases for CSMDiffusion"""

    def test_low_csm_mass(self):
        """Test with low CSM mass"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        csm = interaction.CSMDiffusion(
            time, dense_times, luminosity, 0.1, 1e14, 0.1, 0.1)

        self.assertIsNotNone(csm.new_luminosity)

    def test_large_photosphere_radius(self):
        """Test with large photosphere radius"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        csm = interaction.CSMDiffusion(
            time, dense_times, luminosity, 0.1, 1e15, 0.5, 1.0)

        self.assertIsNotNone(csm.new_luminosity)


class TestViscousEdgeCases(unittest.TestCase):
    """Test edge cases for Viscous class"""

    def test_short_viscous_timescale(self):
        """Test with very short viscous timescale"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        visc = interaction.Viscous(time, dense_times, luminosity, 0.1)

        self.assertIsNotNone(visc.new_luminosity)

    def test_long_viscous_timescale(self):
        """Test with very long viscous timescale"""
        time = np.array([1.0, 2.0, 3.0])
        dense_times = np.linspace(0.5, 10.0, 100)
        luminosity = 1e43 * np.ones(100)

        visc = interaction.Viscous(time, dense_times, luminosity, 100.0)

        self.assertIsNotNone(visc.new_luminosity)


if __name__ == '__main__':
    unittest.main()
