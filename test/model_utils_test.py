"""
Unit tests for the model_utils module.
"""

import unittest
import numpy as np
from astropy.cosmology import Planck18 as cosmo

import redback.interaction_processes as ip
import redback.photosphere as photosphere
import redback.sed as sed
from redback.model_utils import (
    setup_optical_depth_defaults,
    get_cosmology_defaults,
    setup_photosphere_sed_defaults,
    compute_photosphere_and_sed
)


class TestModelUtils(unittest.TestCase):
    """Test the model utility functions."""

    def test_setup_optical_depth_defaults(self):
        """Test that setup_optical_depth_defaults sets correct defaults."""
        kwargs = {}
        setup_optical_depth_defaults(kwargs)

        self.assertEqual(kwargs['interaction_process'], ip.Diffusion)
        self.assertEqual(kwargs['photosphere'], photosphere.TemperatureFloor)
        self.assertEqual(kwargs['sed'], sed.Blackbody)

    def test_setup_optical_depth_defaults_preserves_existing(self):
        """Test that existing values are not overwritten."""
        kwargs = {
            'interaction_process': ip.AngularReprocessing,
            'photosphere': photosphere.Photosphere,
            'sed': sed.CutoffBlackbody
        }
        setup_optical_depth_defaults(kwargs)

        self.assertEqual(kwargs['interaction_process'], ip.AngularReprocessing)
        self.assertEqual(kwargs['photosphere'], photosphere.Photosphere)
        self.assertEqual(kwargs['sed'], sed.CutoffBlackbody)

    def test_get_cosmology_defaults(self):
        """Test cosmology and luminosity distance calculation."""
        redshift = 0.1
        kwargs = {}

        cosmology_result, dl = get_cosmology_defaults(redshift, kwargs)

        self.assertEqual(cosmology_result, cosmo)
        # Check that dl is a positive number
        self.assertGreater(dl, 0)
        # Check that dl matches expected value from Planck18
        expected_dl = cosmo.luminosity_distance(redshift).cgs.value
        self.assertAlmostEqual(dl, expected_dl)

    def test_get_cosmology_defaults_custom_cosmology(self):
        """Test with custom cosmology."""
        from astropy.cosmology import FlatLambdaCDM
        custom_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        redshift = 0.1
        kwargs = {'cosmology': custom_cosmo}

        cosmology_result, dl = get_cosmology_defaults(redshift, kwargs)

        self.assertEqual(cosmology_result, custom_cosmo)
        expected_dl = custom_cosmo.luminosity_distance(redshift).cgs.value
        self.assertAlmostEqual(dl, expected_dl)

    def test_setup_photosphere_sed_defaults(self):
        """Test photosphere and SED defaults."""
        kwargs = {}

        photosphere_class, sed_class = setup_photosphere_sed_defaults(kwargs)

        self.assertEqual(photosphere_class, photosphere.TemperatureFloor)
        self.assertEqual(sed_class, sed.Blackbody)

    def test_setup_photosphere_sed_defaults_custom(self):
        """Test with custom photosphere and SED."""
        kwargs = {
            'photosphere': photosphere.Photosphere,
            'sed': sed.CutoffBlackbody
        }

        photosphere_class, sed_class = setup_photosphere_sed_defaults(kwargs)

        self.assertEqual(photosphere_class, photosphere.Photosphere)
        self.assertEqual(sed_class, sed.CutoffBlackbody)

    def test_compute_photosphere_and_sed(self):
        """Test photosphere and SED computation."""
        # Setup simple test case
        time = np.array([1.0, 2.0, 3.0])
        lbol = np.array([1e43, 2e43, 3e43])  # ergs/s
        frequency = np.array([1e15])  # Hz
        dl = 1e28  # cm

        kwargs = {'vej': 1e4, 'kappa': 0.1, 'kappa_gamma': 0.01,
                  'temperature_floor': 3000}

        photo, sed_obj = compute_photosphere_and_sed(
            time, lbol, frequency,
            photosphere.TemperatureFloor,
            sed.Blackbody,
            dl, **kwargs)

        # Check that objects were created
        self.assertIsNotNone(photo)
        self.assertIsNotNone(sed_obj)

        # Check that photosphere has expected attributes
        self.assertTrue(hasattr(photo, 'photosphere_temperature'))
        self.assertTrue(hasattr(photo, 'r_photosphere'))

        # Check that SED has flux_density attribute
        self.assertTrue(hasattr(sed_obj, 'flux_density'))


if __name__ == '__main__':
    unittest.main()
