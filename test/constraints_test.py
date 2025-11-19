import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import redback.constraints as constraints


class TestSLSNConstraint(unittest.TestCase):
    """Test slsn_constraint function"""

    def test_slsn_constraint_basic(self):
        """Test basic slsn constraint calculation"""
        parameters = {
            'mej': 10.0,  # solar masses
            'vej': 5000.0,  # km/s
            'kappa': 0.1,
            'mass_ns': 1.4,
            'p0': 0.001  # initial period in seconds
        }
        result = constraints.slsn_constraint(parameters)

        # Check that original parameters are preserved
        for key in parameters.keys():
            self.assertEqual(result[key], parameters[key])

        # Check that constraint parameters were added
        self.assertIn('erot_constraint', result)
        self.assertIn('t_nebula_min', result)

        # Check that erot_constraint is positive
        self.assertGreater(result['erot_constraint'], 0)

    def test_slsn_constraint_preserves_input(self):
        """Test that input parameters are not modified"""
        parameters = {
            'mej': 5.0,
            'vej': 10000.0,
            'kappa': 0.2,
            'mass_ns': 1.6,
            'p0': 0.002
        }
        original = parameters.copy()
        result = constraints.slsn_constraint(parameters)

        # Original dict should be unchanged
        self.assertEqual(parameters, original)
        # Result should be different object
        self.assertIsNot(result, parameters)


class TestBasicMagnetarPoweredSNConstraints(unittest.TestCase):
    """Test basic_magnetar_powered_sn_constraints function"""

    def test_basic_constraint(self):
        """Test basic magnetar constraint calculation"""
        parameters = {
            'mej': 10.0,
            'vej': 5000.0,
            'mass_ns': 1.4,
            'p0': 0.001
        }
        result = constraints.basic_magnetar_powered_sn_constraints(parameters)

        self.assertIn('erot_constraint', result)
        self.assertGreater(result['erot_constraint'], 0)

    def test_high_mass_ratio(self):
        """Test with high ejecta mass"""
        parameters = {
            'mej': 20.0,
            'vej': 5000.0,
            'mass_ns': 1.4,
            'p0': 0.001
        }
        result = constraints.basic_magnetar_powered_sn_constraints(parameters)
        self.assertGreater(result['erot_constraint'], 0)


class TestGeneralMagnetarPoweredSNConstraints(unittest.TestCase):
    """Test general_magnetar_powered_sn_constraints function"""

    def test_general_constraint(self):
        """Test general magnetar constraint"""
        parameters = {
            'mej': 10.0,
            'vej': 5000.0,
            'l0': 1e43,
            'tsd': 1e5
        }
        result = constraints.general_magnetar_powered_sn_constraints(parameters)

        self.assertIn('erot_constraint', result)
        self.assertGreater(result['erot_constraint'], 0)

    def test_varying_l0(self):
        """Test with different l0 values"""
        params1 = {'mej': 10.0, 'vej': 5000.0, 'l0': 1e42, 'tsd': 1e5}
        params2 = {'mej': 10.0, 'vej': 5000.0, 'l0': 1e44, 'tsd': 1e5}

        result1 = constraints.general_magnetar_powered_sn_constraints(params1)
        result2 = constraints.general_magnetar_powered_sn_constraints(params2)

        # Higher l0 should give different constraint
        self.assertNotEqual(result1['erot_constraint'], result2['erot_constraint'])


class TestVacuumDipoleMagnetarConstraints(unittest.TestCase):
    """Test vacuum_dipole_magnetar_powered_supernova_constraints"""

    def test_vacuum_dipole_constraint(self):
        """Test vacuum dipole constraint"""
        parameters = {
            'l0': 1e43,
            'tau_sd': 1e5
        }
        result = constraints.vacuum_dipole_magnetar_powered_supernova_constraints(parameters)

        self.assertIn('erot_constraint', result)
        # Constraint should be rotational_energy / 1e53
        expected = (parameters['l0'] * parameters['tau_sd']) / 1e53
        self.assertAlmostEqual(result['erot_constraint'], expected)


class TestGeneralMagnetarPoweredSupernovaConstraints(unittest.TestCase):
    """Test general_magnetar_powered_supernova_constraints"""

    def test_general_supernova_constraint(self):
        """Test general supernova constraint"""
        parameters = {
            'l0': 1e43,
            'tau_sd': 1e5,
            'nn': 3.0
        }
        result = constraints.general_magnetar_powered_supernova_constraints(parameters)

        self.assertIn('erot_constraint', result)
        self.assertGreater(result['erot_constraint'], 0)

    def test_different_nn_values(self):
        """Test with different braking indices"""
        params1 = {'l0': 1e43, 'tau_sd': 1e5, 'nn': 2.0}
        params2 = {'l0': 1e43, 'tau_sd': 1e5, 'nn': 5.0}

        result1 = constraints.general_magnetar_powered_supernova_constraints(params1)
        result2 = constraints.general_magnetar_powered_supernova_constraints(params2)

        self.assertNotEqual(result1['erot_constraint'], result2['erot_constraint'])


class TestTDEConstraints(unittest.TestCase):
    """Test tde_constraints function"""

    def test_tde_constraint(self):
        """Test TDE constraint calculation"""
        parameters = {
            'pericenter_radius': 100.0,  # AU
            'mass_bh': 1e6  # solar masses
        }
        result = constraints.tde_constraints(parameters)

        self.assertIn('disruption_radius', result)
        self.assertGreater(result['disruption_radius'], 0)

    def test_large_pericenter(self):
        """Test with large pericenter radius"""
        parameters = {
            'pericenter_radius': 1000.0,
            'mass_bh': 1e6
        }
        result = constraints.tde_constraints(parameters)
        # Larger pericenter should give smaller constraint ratio
        self.assertLess(result['disruption_radius'], 1.0)


class TestGaussianriseTDEConstraints(unittest.TestCase):
    """Test gaussianrise_tde_constraints function"""

    @patch('redback.constraints.calc_tfb')
    def test_gaussianrise_constraint(self, mock_calc_tfb):
        """Test gaussian rise TDE constraint"""
        mock_calc_tfb.return_value = 86400 * 30  # 30 days in seconds

        parameters = {
            'stellar_mass': 1.0,
            'mbh_6': 1.0,
            'beta': 2.0,
            'peak_time': 50.0,
            'redshift': 0.1
        }
        result = constraints.gaussianrise_tde_constraints(parameters)

        self.assertIn('beta_high', result)
        self.assertIn('tfb_max', result)
        mock_calc_tfb.assert_called_once()


class TestNuclearBurningConstraints(unittest.TestCase):
    """Test nuclear_burning_constraints function"""

    def test_nuclear_burning_constraint(self):
        """Test nuclear burning constraint"""
        parameters = {
            'mej': 10.0,
            'vej': 5000.0,
            'f_nickel': 0.1
        }
        result = constraints.nuclear_burning_constraints(parameters)

        self.assertIn('emax_constraint', result)
        self.assertGreater(result['emax_constraint'], 0)

    def test_high_nickel_fraction(self):
        """Test with high nickel fraction"""
        params_low = {'mej': 10.0, 'vej': 5000.0, 'f_nickel': 0.01}
        params_high = {'mej': 10.0, 'vej': 5000.0, 'f_nickel': 0.5}

        result_low = constraints.nuclear_burning_constraints(params_low)
        result_high = constraints.nuclear_burning_constraints(params_high)

        # Higher nickel fraction should give lower constraint (more energy available)
        self.assertGreater(result_low['emax_constraint'], result_high['emax_constraint'])


class TestSimpleFallbackConstraints(unittest.TestCase):
    """Test simple_fallback_constraints function"""

    def test_simple_fallback(self):
        """Test simple fallback constraint"""
        parameters = {
            'mej': 10.0,
            'vej': 5000.0,
            'kappa': 0.1,
            'l0': 1e43,
            't_0_turn': 10.0
        }
        result = constraints.simple_fallback_constraints(parameters)

        self.assertIn('en_constraint', result)
        self.assertIn('t_nebula_min', result)
        self.assertGreater(result['en_constraint'], 0)


class TestCSMConstraints(unittest.TestCase):
    """Test csm_constraints function"""

    def test_csm_constraint_scalar(self):
        """Test CSM constraint with scalar parameters"""
        parameters = {
            'mej': 10.0,
            'csm_mass': 1.0,
            'kappa': 0.1,
            'r0': 100.0,
            'vej': 5000.0,
            'eta': 0.5,
            'rho': 1e-14
        }
        result = constraints.csm_constraints(parameters)

        self.assertIn('shock_time', result)
        self.assertIn('photosphere_constraint_1', result)
        self.assertIn('photosphere_constraint_2', result)

    def test_csm_constraint_with_nn_delta(self):
        """Test CSM constraint with nn and delta parameters"""
        parameters = {
            'mej': 10.0,
            'csm_mass': 1.0,
            'kappa': 0.1,
            'r0': 100.0,
            'vej': 5000.0,
            'eta': 0.5,
            'rho': 1e-14,
            'nn': 10.0,
            'delta': 0.1
        }
        result = constraints.csm_constraints(parameters)

        self.assertIn('shock_time', result)
        self.assertIn('photosphere_constraint_1', result)
        self.assertIn('photosphere_constraint_2', result)

    def test_csm_constraint_array(self):
        """Test CSM constraint with array inputs"""
        parameters = {
            'mej': np.array([10.0, 15.0]),
            'csm_mass': np.array([1.0, 2.0]),
            'kappa': 0.1,
            'r0': 100.0,
            'vej': 5000.0,
            'eta': 0.5,
            'rho': 1e-14
        }
        result = constraints.csm_constraints(parameters)

        # Should handle arrays
        self.assertTrue(hasattr(result['shock_time'], '__len__') or
                       isinstance(result['shock_time'], (int, float, np.number)))


class TestPiecewisePolytropeEOSConstraints(unittest.TestCase):
    """Test piecewise_polytrope_eos_constraints function"""

    @patch('redback.constraints.calc_max_mass')
    @patch('redback.constraints.calc_speed_of_sound')
    def test_piecewise_polytrope_constraint(self, mock_calc_sos, mock_calc_max_mass):
        """Test piecewise polytrope EOS constraint"""
        mock_calc_max_mass.return_value = 2.1
        mock_calc_sos.return_value = 0.8

        parameters = {
            'log_p': 34.0,
            'gamma_1': 3.0,
            'gamma_2': 2.5,
            'gamma_3': 2.0
        }
        result = constraints.piecewise_polytrope_eos_constraints(parameters)

        self.assertIn('maximum_eos_mass', result)
        self.assertIn('maximum_speed_of_sound', result)
        self.assertEqual(result['maximum_eos_mass'], 2.1)
        self.assertEqual(result['maximum_speed_of_sound'], 0.8)


class TestCalcMaxMass(unittest.TestCase):
    """Test calc_max_mass function"""

    @patch('redback.constraints.eos.PiecewisePolytrope')
    def test_calc_max_mass_scalar(self, mock_polytrope_class):
        """Test calc_max_mass with scalar inputs"""
        mock_polytrope = MagicMock()
        mock_polytrope.maximum_mass.return_value = 2.2
        mock_polytrope_class.return_value = mock_polytrope

        # calc_max_mass is vectorized, so we get an array back
        result = constraints.calc_max_mass(34.0, 3.0, 2.5, 2.0)
        # For scalar input to vectorized function, result is still scalar-like
        self.assertIsNotNone(result)
        mock_polytrope_class.assert_called_once_with(
            log_p=34.0, gamma_1=3.0, gamma_2=2.5, gamma_3=2.0)


class TestCalcSpeedOfSound(unittest.TestCase):
    """Test calc_speed_of_sound function"""

    @patch('redback.constraints.eos.PiecewisePolytrope')
    def test_calc_speed_of_sound_scalar(self, mock_polytrope_class):
        """Test calc_speed_of_sound with scalar inputs"""
        mock_polytrope = MagicMock()
        mock_polytrope.maximum_speed_of_sound.return_value = 0.85
        mock_polytrope_class.return_value = mock_polytrope

        # calc_speed_of_sound is vectorized, so handle accordingly
        result = constraints.calc_speed_of_sound(34.0, 3.0, 2.5, 2.0)
        self.assertIsNotNone(result)
        mock_polytrope_class.assert_called_once_with(
            log_p=34.0, gamma_1=3.0, gamma_2=2.5, gamma_3=2.0)


if __name__ == '__main__':
    unittest.main()
