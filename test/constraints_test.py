import unittest
import builtins
import numpy as np
from unittest.mock import patch, MagicMock
import warnings
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
        self.assertIn('e_rot_constraint', result)
        self.assertIn('t_nebula_min', result)

        # Check that e_rot_constraint is positive
        self.assertGreater(result['e_rot_constraint'], 0)

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


class TestEOSImports(unittest.TestCase):
    """Test optional EOS dependency imports."""

    def test_lalsimulation_import_suppresses_swiglal_redirect_warning(self):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lalsimulation":
                warnings.warn("Wswiglal-redir-stdio: redirection enabled", UserWarning)
                return MagicMock()
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                constraints.eos._import_lalsimulation()

        self.assertFalse(any("Wswiglal-redir-stdio" in str(w.message) for w in caught))


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

    def test_csm_constraint_scalar_and_length_one_array_match_defaults(self):
        """The CSM constraint defaults should not depend on sample shape."""
        parameters = {
            'mej': 10.0,
            'csm_mass': 1.0,
            'kappa': 0.1,
            'r0': 100.0,
            'vej': 5000.0,
            'eta': 0.5,
            'rho': 1e-14
        }
        scalar_result = constraints.csm_constraints(parameters)
        array_parameters = {key: np.array([value]) for key, value in parameters.items()}
        array_result = constraints.csm_constraints(array_parameters)

        for key in ['shock_time', 'photosphere_constraint_1', 'photosphere_constraint_2']:
            np.testing.assert_allclose(scalar_result[key], array_result[key][0])

    def test_csm_constraint_broadcasts_fixed_values_with_sampled_nn_delta(self):
        """Fixed scalar kappa/r0 values should broadcast across sampled CSM arrays."""
        parameters = {
            'mej': np.array([10.0, 15.0]),
            'csm_mass': np.array([1.0, 2.0]),
            'kappa': 0.34,
            'r0': 100.0,
            'vej': np.array([5000.0, 8000.0]),
            'eta': np.array([0.5, 1.0]),
            'rho': np.array([1e-14, 5e-14]),
            'nn': np.array([8.0, 12.0]),
            'delta': np.array([0.0, 1.0])
        }
        result = constraints.csm_constraints(parameters)

        for key in ['shock_time', 'photosphere_constraint_1', 'photosphere_constraint_2']:
            self.assertEqual(np.shape(result[key]), (2,))
            self.assertTrue(np.all(np.isfinite(result[key])))

    def test_csm_constraint_returns_rejectable_values_for_invalid_samples(self):
        """Pathological CSM samples should not produce NaN constraint values."""
        parameters = {
            'mej': 0.0030976313516303715,
            'csm_mass': 0.00023473720624880357,
            'ek': 7.867879885236966e51,
            'eta': 0.31834817913487723,
            'rho': 1.2431552383024176e-15,
            'r0': 71.17425373342279,
            'kappa': 0.26256530921692656
        }
        result = constraints.csm_constraints(parameters)

        for key in ['shock_time', 'photosphere_constraint_1', 'photosphere_constraint_2']:
            self.assertFalse(np.any(np.isnan(result[key])))


class TestKilonovaEjectaRelationConstraints(unittest.TestCase):
    """Test kilonova ejecta-relation domain constraints."""

    def test_one_component_nsbh_ejecta_relation_constraints(self):
        parameters = {
            'mass_bh': 5.0,
            'mass_ns': 1.4,
            'chi_bh': 0.9,
            'lambda_ns': 800.0
        }
        result = constraints.one_component_nsbh_ejecta_relation_constraints(parameters)

        for key in ['ejecta_mej_min', 'ejecta_mej_max', 'ejecta_vej_min', 'ejecta_vej_max']:
            self.assertIn(key, result)
            self.assertTrue(np.all(np.isfinite(result[key])))

    def test_two_component_bns_ejecta_relation_constraints(self):
        parameters = {
            'mass_1': 1.35,
            'mass_2': 1.30,
            'lambda_1': 600.0,
            'lambda_2': 700.0,
            'mtov': 2.2,
            'zeta': 0.2,
            'vej_2': 0.2
        }
        result = constraints.two_component_bns_ejecta_relation_constraints(parameters)

        for key in [
            'dynamical_ejecta_mej_min', 'dynamical_ejecta_mej_max',
            'dynamical_ejecta_vej_min', 'dynamical_ejecta_vej_max',
            'disk_wind_ejecta_mej_min', 'disk_wind_ejecta_mej_max',
            'disk_wind_ejecta_vej_min', 'disk_wind_ejecta_vej_max'
        ]:
            self.assertIn(key, result)
            self.assertTrue(np.all(np.isfinite(result[key])))

    @patch("redback.constraints.ejr.OneComponentBNSNoProjection")
    def test_one_component_bns_constraints(self, relation_class):
        relation_class.return_value.ejecta_mass = 0.02
        relation_class.return_value.ejecta_velocity = 0.2
        parameters = dict(mass_1=1.4, mass_2=1.3, lambda_1=500.0, lambda_2=600.0)
        result = constraints.one_component_bns_ejecta_relation_constraints(parameters)
        self.assertIn("ejecta_mej_min", result)
        self.assertIn("ejecta_vej_max", result)

    @patch("redback.constraints.ejr.OneComponentBNSProjection")
    def test_projected_one_component_bns_constraints(self, relation_class):
        relation_class.return_value.ejecta_mass = 0.02
        relation_class.return_value.ejecta_velocity = 0.2
        parameters = dict(mass_1=1.4, mass_2=1.3, lambda_1=500.0, lambda_2=600.0)
        result = constraints.one_component_bns_ejecta_relation_projection_constraints(parameters)
        self.assertIn("ejecta_mej_max", result)
        self.assertIn("ejecta_vej_min", result)

    @patch("redback.constraints.ejr.TwoComponentNSBH")
    def test_two_component_nsbh_constraints(self, relation_class):
        relation_class.return_value.dynamical_mej = 0.02
        relation_class.return_value.disk_wind_mej = 0.03
        relation_class.return_value.ejecta_velocity = 0.2
        parameters = dict(
            mass_bh=5.0, mass_ns=1.4, chi_bh=0.8, lambda_ns=500.0,
            zeta=0.2, vej_2=0.1)
        result = constraints.two_component_nsbh_ejecta_relation_constraints(parameters)
        self.assertIn("dynamical_ejecta_mej_min", result)
        self.assertIn("disk_wind_ejecta_vej_max", result)

    def test_safe_constraint_ratio_rejects_nonfinite_values(self):
        result = constraints._safe_constraint_ratio(
            np.array([1.0, 1.0, np.inf]), np.array([0.0, np.nan, 1.0]))
        self.assertTrue(np.all(np.isinf(result)))

    def test_maybe_scalar_returns_python_scalar(self):
        self.assertEqual(constraints._maybe_scalar(np.array(3.0)), 3.0)

    @patch("redback.constraints.ejr.TwoComponentBNS")
    @patch("redback.constraints.eos.PiecewisePolytrope")
    def test_polytrope_ejecta_quantities_support_arrays(self, eos_class, relation_class):
        eos_class.return_value.maximum_mass.return_value = 2.2
        eos_class.return_value.lambda_of_mass.side_effect = [(500.0, 0.0), (600.0, 0.0)] * 2
        relation_class.return_value.dynamical_mej = 0.02
        relation_class.return_value.disk_wind_mej = 0.03
        relation_class.return_value.ejecta_velocity = 0.2
        values = constraints._polytrope_two_component_bns_ejecta_quantities(
            mass_1=np.array([1.4, 1.5]), mass_2=np.array([1.3, 1.2]),
            log_p=34.0, gamma_1=3.0, gamma_2=2.5, gamma_3=2.0, zeta=0.2)
        for value in values:
            np.testing.assert_array_equal(value, [value[0], value[0]])

    @patch("redback.constraints.eos.PiecewisePolytrope", side_effect=ValueError("bad EOS"))
    def test_polytrope_ejecta_quantities_return_nan_on_invalid_eos(self, eos_class):
        values = constraints._polytrope_two_component_bns_ejecta_quantities(
            mass_1=1.4, mass_2=1.3, log_p=34.0,
            gamma_1=3.0, gamma_2=2.5, gamma_3=2.0, zeta=0.2)
        self.assertTrue(all(np.isnan(value) for value in values))
        eos_class.assert_called_once()

    @patch("redback.constraints._polytrope_two_component_bns_ejecta_quantities")
    @patch("redback.constraints.piecewise_polytrope_eos_constraints")
    def test_polytrope_constraint_adds_both_ejecta_components(self, eos_constraints, ejecta):
        parameters = dict(
            mass_1=1.4, mass_2=1.3, log_p=34.0, gamma_1=3.0,
            gamma_2=2.5, gamma_3=2.0, zeta=0.2, vej_2=0.1)
        eos_constraints.return_value = parameters.copy()
        ejecta.return_value = (0.02, 0.03, 0.2)
        result = constraints.polytrope_eos_two_component_bns_constraints(parameters)
        self.assertIn("dynamical_ejecta_mej_min", result)
        self.assertIn("disk_wind_ejecta_mej_max", result)


class TestCoolingEnvelopeConstraints(unittest.TestCase):
    """Test cooling-envelope TDE constraints."""

    def test_cooling_envelope_constraints_add_domain_ratios(self):
        parameters = {
            'stellar_mass': 1.0,
            'mbh_6': 1.0,
            'eta': 0.1,
            'beta': 2.0
        }
        result = constraints.cooling_envelope_constraints(parameters)

        self.assertIn('eta_min_ratio', result)
        self.assertIn('beta_max_ratio', result)
        self.assertTrue(np.isfinite(result['eta_min_ratio']))
        self.assertTrue(np.isfinite(result['beta_max_ratio']))

    def test_gaussian_tail_constraint_rejects_underflowing_stitching(self):
        parameters = {
            'redshift': 2.567877768220058,
            'peak_time': 38.9806277500574,
            'sigma_t': 19.491419465518803,
            'mbh_6': 9.331707719561335,
            'stellar_mass': 4.881164962346675,
            'eta': 0.04933967484280144,
            'alpha': 0.11303522640434999,
            'beta': 1.0793171463784845
        }
        result = constraints.cooling_envelope_constraints(parameters)

        self.assertIn('gaussian_stitching_tail', result)
        self.assertGreater(result['gaussian_stitching_tail'], 35)


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
        # Vectorized function may call multiple times, just check it was called
        self.assertTrue(mock_polytrope_class.called)


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
        # Vectorized function may call multiple times, just check it was called
        self.assertTrue(mock_polytrope_class.called)


if __name__ == '__main__':
    unittest.main()
