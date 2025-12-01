import unittest
import numpy as np
from unittest.mock import patch
import redback.ejecta_relations as ejecta


class TestCalcCompactnessFromLambda(unittest.TestCase):
    """Test calc_compactness_from_lambda function"""

    def test_compactness_calculation(self):
        """Test compactness calculation from lambda"""
        lambda_val = 400.0
        result = ejecta.calc_compactness_from_lambda(lambda_val)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)  # Compactness should be between 0 and 1

    def test_compactness_array(self):
        """Test with array of lambda values"""
        lambda_vals = np.array([100, 400, 1000])
        result = ejecta.calc_compactness_from_lambda(lambda_vals)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))


class TestCalcCompactness(unittest.TestCase):
    """Test calc_compactness function"""

    def test_compactness_from_mass_radius(self):
        """Test compactness from mass and radius"""
        mass = 1.4  # solar masses
        radius = 12000.0  # meters
        result = ejecta.calc_compactness(mass, radius)
        self.assertGreater(result, 0)
        self.assertLess(result, 0.5)  # Typical NS compactness


class TestCalcBaryonicMass(unittest.TestCase):
    """Test calc_baryonic_mass function"""

    def test_baryonic_mass_calculation(self):
        """Test baryonic mass calculation"""
        mass = 1.4
        compactness = 0.15
        result = ejecta.calc_baryonic_mass(mass, compactness)
        self.assertGreater(result, mass)  # Baryonic mass should be larger

    def test_baryonic_mass_high_compactness(self):
        """Test with higher compactness"""
        mass = 1.4
        low_c = 0.10
        high_c = 0.20
        result_low = ejecta.calc_baryonic_mass(mass, low_c)
        result_high = ejecta.calc_baryonic_mass(mass, high_c)
        self.assertGreater(result_high, result_low)


class TestCalcBaryonicMassEOSInsensitive(unittest.TestCase):
    """Test calc_baryonic_mass_eos_insensitive function"""

    def test_eos_insensitive_baryonic_mass(self):
        """Test EOS-insensitive baryonic mass"""
        mass_g = 1.4
        radius_14 = 12000.0
        result = ejecta.calc_baryonic_mass_eos_insensitive(mass_g, radius_14)
        self.assertGreater(result, mass_g)


class TestCalcVrho(unittest.TestCase):
    """Test calc_vrho function"""

    def test_vrho_calculation(self):
        """Test average velocity in orbital plane"""
        mass_1 = 1.4
        mass_2 = 1.3
        lambda_1 = 400.0
        lambda_2 = 600.0
        result = ejecta.calc_vrho(mass_1, mass_2, lambda_1, lambda_2)
        self.assertGreater(result, 0)
        self.assertLess(result, 1.0)  # Velocity in c

    def test_vrho_has_citation(self):
        """Test that calc_vrho has citation attribute"""
        self.assertTrue(hasattr(ejecta.calc_vrho, 'citation'))


class TestCalcVz(unittest.TestCase):
    """Test calc_vz function"""

    def test_vz_calculation(self):
        """Test velocity orthogonal to orbital plane"""
        mass_1 = 1.4
        mass_2 = 1.3
        lambda_1 = 400.0
        lambda_2 = 600.0
        result = ejecta.calc_vz(mass_1, mass_2, lambda_1, lambda_2)
        self.assertGreater(result, 0)
        self.assertLess(result, 1.0)  # Velocity in c

    def test_vz_has_citation(self):
        """Test that calc_vz has citation attribute"""
        self.assertTrue(hasattr(ejecta.calc_vz, 'citation'))


class TestOneComponentBNSNoProjection(unittest.TestCase):
    """Test OneComponentBNSNoProjection class"""

    def setUp(self):
        self.mass_1 = 1.4
        self.mass_2 = 1.3
        self.lambda_1 = 400.0
        self.lambda_2 = 600.0

    def test_initialization(self):
        """Test class initialization"""
        obj = ejecta.OneComponentBNSNoProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)

        self.assertEqual(obj.mass_1, self.mass_1)
        self.assertEqual(obj.mass_2, self.mass_2)
        self.assertEqual(obj.lambda_1, self.lambda_1)
        self.assertEqual(obj.lambda_2, self.lambda_2)
        self.assertIsNotNone(obj.c1)
        self.assertIsNotNone(obj.c2)
        self.assertIsNotNone(obj.reference)

    def test_ejecta_velocity(self):
        """Test ejecta velocity calculation"""
        obj = ejecta.OneComponentBNSNoProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        vel = obj.ejecta_velocity
        self.assertGreater(vel, 0)
        self.assertLess(vel, 1.0)  # Should be in units of c

    def test_ejecta_mass(self):
        """Test ejecta mass calculation"""
        obj = ejecta.OneComponentBNSNoProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        mass = obj.ejecta_mass
        self.assertGreater(mass, 0)
        self.assertLess(mass, 0.1)  # Typical ejecta mass in solar masses

    def test_qej(self):
        """Test polar opening angle"""
        obj = ejecta.OneComponentBNSNoProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        qej = obj.qej
        self.assertGreater(qej, 0)

    def test_phej(self):
        """Test azimuthal opening angle"""
        obj = ejecta.OneComponentBNSNoProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        phej = obj.phej
        self.assertGreater(phej, 0)


class TestOneComponentBNSProjection(unittest.TestCase):
    """Test OneComponentBNSProjection class"""

    def setUp(self):
        self.mass_1 = 1.4
        self.mass_2 = 1.3
        self.lambda_1 = 400.0
        self.lambda_2 = 600.0

    def test_initialization(self):
        """Test class initialization"""
        obj = ejecta.OneComponentBNSProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)

        self.assertEqual(obj.mass_1, self.mass_1)
        self.assertEqual(obj.mass_2, self.mass_2)
        self.assertIsNotNone(obj.reference)

    def test_ejecta_velocity_with_projection(self):
        """Test ejecta velocity with projection"""
        obj = ejecta.OneComponentBNSProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        vel = obj.ejecta_velocity
        self.assertGreater(vel, 0)
        self.assertLess(vel, 1.0)

    def test_ejecta_mass_with_projection(self):
        """Test ejecta mass with projection"""
        obj = ejecta.OneComponentBNSProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        mass = obj.ejecta_mass
        self.assertGreater(mass, 0)

    def test_qej_with_projection(self):
        """Test polar opening angle with projection"""
        obj = ejecta.OneComponentBNSProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        qej = obj.qej
        self.assertGreater(qej, 0)

    def test_phej_with_projection(self):
        """Test azimuthal opening angle with projection"""
        obj = ejecta.OneComponentBNSProjection(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2)
        phej = obj.phej
        self.assertGreater(phej, 0)


class TestTwoComponentBNS(unittest.TestCase):
    """Test TwoComponentBNS class"""

    def setUp(self):
        self.mass_1 = 1.4
        self.mass_2 = 1.3
        self.lambda_1 = 400.0
        self.lambda_2 = 600.0
        self.mtov = 2.1
        self.zeta = 0.1

    def test_initialization(self):
        """Test class initialization"""
        obj = ejecta.TwoComponentBNS(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2, self.mtov, self.zeta)

        self.assertEqual(obj.mass_1, self.mass_1)
        self.assertEqual(obj.mass_2, self.mass_2)
        self.assertEqual(obj.mtov, self.mtov)
        self.assertEqual(obj.zeta, self.zeta)
        self.assertIsNotNone(obj.reference)

    def test_dynamical_ejecta_mass(self):
        """Test dynamical ejecta mass calculation"""
        obj = ejecta.TwoComponentBNS(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2, self.mtov, self.zeta)
        mass = obj.dynamical_mej
        self.assertGreater(mass, 0)

    def test_disk_wind_mass(self):
        """Test disk wind mass calculation"""
        obj = ejecta.TwoComponentBNS(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2, self.mtov, self.zeta)
        mass = obj.disk_wind_mej
        self.assertGreater(mass, 0)

    def test_ejecta_velocity(self):
        """Test ejecta velocity for two component"""
        obj = ejecta.TwoComponentBNS(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2, self.mtov, self.zeta)
        vel = obj.ejecta_velocity
        self.assertGreater(vel, 0)
        self.assertLess(vel, 1.0)

    def test_qej(self):
        """Test polar opening angle for two component"""
        obj = ejecta.TwoComponentBNS(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2, self.mtov, self.zeta)
        qej = obj.qej
        self.assertGreater(qej, 0)

    def test_phej(self):
        """Test azimuthal opening angle for two component"""
        obj = ejecta.TwoComponentBNS(
            self.mass_1, self.mass_2, self.lambda_1, self.lambda_2, self.mtov, self.zeta)
        phej = obj.phej
        self.assertGreater(phej, 0)


class TestTwoComponentNSBH(unittest.TestCase):
    """Test TwoComponentNSBH class"""

    def setUp(self):
        self.mass_bh = 5.0
        self.mass_ns = 1.4
        self.chi_bh = 0.5
        self.lambda_ns = 400.0
        self.zeta = 0.1

    def test_initialization(self):
        """Test class initialization"""
        obj = ejecta.TwoComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns, self.zeta)

        self.assertEqual(obj.mass_bh, self.mass_bh)
        self.assertEqual(obj.mass_ns, self.mass_ns)
        self.assertEqual(obj.chi_bh, self.chi_bh)
        self.assertEqual(obj.lambda_ns, self.lambda_ns)
        self.assertEqual(obj.zeta, self.zeta)
        self.assertIsInstance(obj.reference, list)

    def test_isco_radius(self):
        """Test ISCO radius calculation"""
        obj = ejecta.TwoComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns, self.zeta)
        risco = obj.risco
        self.assertGreater(risco, 0)
        self.assertLess(risco, 10)  # Typical ISCO radius

    def test_ejecta_velocity(self):
        """Test ejecta velocity for NSBH"""
        obj = ejecta.TwoComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns, self.zeta)
        vel = obj.ejecta_velocity
        self.assertGreater(vel, 0)
        self.assertLess(vel, 1.0)

    def test_dynamical_ejecta_mass(self):
        """Test dynamical ejecta mass for NSBH"""
        obj = ejecta.TwoComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns, self.zeta)
        mass = obj.dynamical_mej
        self.assertGreaterEqual(mass, 0)  # Can be zero for some parameters

    def test_disk_wind_mass(self):
        """Test disk wind mass for NSBH"""
        obj = ejecta.TwoComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns, self.zeta)
        mass = obj.disk_wind_mej
        self.assertGreaterEqual(mass, 0)

    def test_negative_spin(self):
        """Test with negative BH spin"""
        obj = ejecta.TwoComponentNSBH(
            self.mass_bh, self.mass_ns, -0.5, self.lambda_ns, self.zeta)
        risco = obj.risco
        self.assertGreater(risco, 0)


class TestOneComponentNSBH(unittest.TestCase):
    """Test OneComponentNSBH class"""

    def setUp(self):
        self.mass_bh = 5.0
        self.mass_ns = 1.4
        self.chi_bh = 0.5
        self.lambda_ns = 400.0

    def test_initialization(self):
        """Test class initialization"""
        obj = ejecta.OneComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns)

        self.assertEqual(obj.mass_bh, self.mass_bh)
        self.assertEqual(obj.mass_ns, self.mass_ns)
        self.assertEqual(obj.chi_bh, self.chi_bh)
        self.assertEqual(obj.lambda_ns, self.lambda_ns)
        self.assertIsNotNone(obj.reference)

    def test_isco_radius(self):
        """Test ISCO radius for one component"""
        obj = ejecta.OneComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns)
        risco = obj.risco
        self.assertGreater(risco, 0)
        self.assertLess(risco, 10)

    def test_ejecta_velocity(self):
        """Test ejecta velocity for one component NSBH"""
        obj = ejecta.OneComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns)
        vel = obj.ejecta_velocity
        self.assertGreater(vel, 0)
        self.assertLess(vel, 1.0)

    def test_ejecta_mass(self):
        """Test ejecta mass for one component NSBH"""
        obj = ejecta.OneComponentNSBH(
            self.mass_bh, self.mass_ns, self.chi_bh, self.lambda_ns)
        mass = obj.ejecta_mass
        self.assertGreaterEqual(mass, 0)

    def test_high_mass_ratio(self):
        """Test with high mass ratio"""
        obj = ejecta.OneComponentNSBH(
            10.0, 1.4, 0.5, 400.0)
        mass = obj.ejecta_mass
        self.assertGreaterEqual(mass, 0)

    def test_zero_spin(self):
        """Test with zero BH spin"""
        obj = ejecta.OneComponentNSBH(
            self.mass_bh, self.mass_ns, 0.0, self.lambda_ns)
        risco = obj.risco
        self.assertGreater(risco, 0)


if __name__ == '__main__':
    unittest.main()
