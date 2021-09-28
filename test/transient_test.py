import unittest
import numpy as np
import shutil

from redback.redback.transient.transient import Transient
from redback.redback.transient.afterglow import Afterglow, SGRB, LGRB
from redback.redback.transient.prompt import PromptTimeSeries
from redback.redback.getdata import get_afterglow_data_from_swift

class TestTransient(unittest.TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3])
        self.time_err = np.array([0.2, 0.3, 0.4])
        self.y = np.array([3, 4, 2])
        self.y_err = np.sqrt(self.y)
        self.redshift = 0.75
        self.data_mode = 'counts'
        self.name = "GRB123456"
        self.path = '.'
        self.photon_index = 2
        self.transient = Transient(time=self.time, time_err=self.time_err, y=self.y, y_err=self.y_err,
                                   redshift=self.redshift, data_mode=self.data_mode, name=self.name, path=self.path,
                                   photon_index=self.photon_index)

    def tearDown(self) -> None:
        del self.time
        del self.time_err
        del self.y
        del self.y_err
        del self.redshift
        del self.data_mode
        del self.name
        del self.path
        del self.photon_index
        del self.transient

    def test_data_mode_switches(self):
        self.assertTrue(self.transient.counts_data)
        self.assertFalse(self.transient.luminosity_data)
        self.assertFalse(self.transient.flux_data)
        self.assertFalse(self.transient.flux_density_data)
        self.assertFalse(self.transient.photometry_data)
        self.assertFalse(self.transient.tte_data)

    def test_set_data_mode_switch(self):
        self.transient.flux_data = True
        self.assertTrue(self.transient.flux_data)
        self.assertFalse(self.transient.counts_data)

    def test_get_time_via_x(self):
        self.assertTrue(np.array_equal(self.time, self.transient.x))
        self.assertTrue(np.array_equal(self.time_err, self.transient.x_err))

    def test_get_time_via_x_luminosity_data(self):
        new_times = np.array([1, 2, 3])
        new_time_errs = np.array([0.1, 0.2, 0.3])
        self.transient.time_rest_frame = new_times
        self.transient.time_rest_frame_err = new_time_errs
        self.transient.data_mode = "luminosity"
        self.assertTrue(np.array_equal(new_times, self.transient.x))
        self.assertTrue(np.array_equal(new_time_errs, self.transient.x_err))

    def test_x_same_as_time(self):
        self.assertTrue(np.array_equal(self.transient.x, self.transient.time))

    def test_xerr_same_as_time_err(self):
        self.assertTrue(np.array_equal(self.transient.x_err, self.transient.time_err))

    def test_set_x(self):
        new_x = np.array([2, 3, 4])
        self.transient.x = new_x
        self.assertTrue(np.array_equal(new_x, self.transient.x))
        self.assertTrue(np.array_equal(new_x, self.transient.time))

    def test_set_x_err(self):
        new_x_err = np.array([3, 4, 5])
        self.transient.x_err = new_x_err
        self.assertTrue(np.array_equal(new_x_err, self.transient.x_err))
        self.assertTrue(np.array_equal(new_x_err, self.transient.time_err))

    def test_y_same_as_counts(self):
        self.assertTrue(np.array_equal(self.transient.y, self.transient.counts))

    def test_yerr_same_as_counts(self):
        self.assertTrue(np.array_equal(self.transient.y_err, self.transient.counts_err))

    def test_redshift(self):
        self.assertEqual(self.redshift, self.transient.redshift)

    def test_get_data_mode(self):
        self.assertEqual(self.data_mode, self.transient.data_mode)

    def test_set_data_mode(self):
        new_data_mode = "luminosity"
        self.transient.data_mode = new_data_mode
        self.assertEqual(new_data_mode, self.transient.data_mode)

    def test_set_illegal_data_mode(self):
        with self.assertRaises(ValueError):
            self.transient.data_mode = "abc"

    def test_path(self):
        self.assertEqual(self.path, self.transient.path)

    def test_plot_lightcurve(self):
        self.transient.plot_lightcurve(model=None)


class TestAfterglow(unittest.TestCase):

    def setUp(self) -> None:
        self.redshift = 0.75
        self.data_mode = 'flux'
        self.name = "170728A"
        get_afterglow_data_from_swift(self.name, data_mode='flux')
        self.transient = SGRB(name=self.name, data_mode=self.data_mode)

    def tearDown(self) -> None:
        del self.redshift
        del self.data_mode
        del self.name
        del self.transient

    def test_analytical_flux_to_luminosity_illegal_data_mode(self):
        self.transient.data_mode = "flux_density"
        self.transient.analytical_flux_to_luminosity()
        self.assertEqual("flux_density", self.transient.data_mode)
        self.assertTrue(len(self.transient.time_rest_frame) == 0)
        self.assertTrue(len(self.transient.time_rest_frame_err) == 0)
        self.assertTrue(len(self.transient.Lum50) == 0)
        self.assertTrue(len(self.transient.Lum50_err) == 0)

    def test_analytical_flux_to_luminosity(self):
        self.transient.analytical_flux_to_luminosity()
        self.assertEqual("luminosity", self.transient.data_mode)
        self.assertFalse(len(self.transient.time_rest_frame) == 0)
        self.assertFalse(len(self.transient.time_rest_frame_err) == 0)
        self.assertFalse(len(self.transient.Lum50) == 0)
        self.assertFalse(len(self.transient.Lum50_err) == 0)

    def test_numerical_flux_to_luminosity_illegal_data_mode(self):
        self.transient.data_mode = "flux_density"
        self.transient.numerical_flux_to_luminosity(counts_to_flux_absorbed=1, counts_to_flux_unabsorbed=1)
        self.assertEqual("flux_density", self.transient.data_mode)
        self.assertTrue(len(self.transient.time_rest_frame) == 0)
        self.assertTrue(len(self.transient.time_rest_frame_err) == 0)
        self.assertTrue(len(self.transient.Lum50) == 0)
        self.assertTrue(len(self.transient.Lum50_err) == 0)

    def test_numerical_flux_to_luminosity(self):
        self.transient.numerical_flux_to_luminosity(counts_to_flux_absorbed=1, counts_to_flux_unabsorbed=1)
        self.assertEqual("luminosity", self.transient.data_mode)
        self.assertFalse(len(self.transient.time_rest_frame) == 0)
        self.assertFalse(len(self.transient.time_rest_frame_err) == 0)
        self.assertFalse(len(self.transient.Lum50) == 0)
        self.assertFalse(len(self.transient.Lum50_err) == 0)
