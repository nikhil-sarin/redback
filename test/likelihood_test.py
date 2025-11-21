import bilby.core.prior
import numpy as np
import unittest
from unittest import mock

from redback import likelihoods


class GaussianLikelihoodTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1, 2])
        self.sigma = 1

        def func(x, param_1, param_2, **kwargs):
            return x

        self.function = func
        self.kwargs = dict(kwarg_1='test_kwarg')
        self.likelihood = likelihoods.GaussianLikelihood(
            x=self.x, y=self.y, sigma=self.sigma, function=self.function, kwargs=self.kwargs)

    def tearDown(self):
        del self.x
        del self.y
        del self.sigma
        del self.function
        del self.kwargs
        del self.likelihood

    def test_set_x(self):
        self.assertTrue(np.array_equal(self.x, self.likelihood.x))

    def test_set_y(self):
        self.assertTrue(np.array_equal(self.y, self.likelihood.y))

    def test_set_sigma_float(self):
        self.assertEqual(self.sigma, self.likelihood.sigma)

    def test_set_function(self):
        self.assertEqual(self.function, self.likelihood.function)

    def test_set_kwargs(self):
        self.assertDictEqual(self.kwargs, self.likelihood.kwargs)

    def test_set_kwargs_None(self):
        self.likelihood.kwargs = None
        self.assertDictEqual(dict(), self.likelihood.kwargs)

    def test_set_likelihood_parameters(self):
        self.assertListEqual(['param_1', 'param_2'], list(self.likelihood.parameters.keys()))

    def test_set_sigma_parameter(self):
        self.likelihood = likelihoods.GaussianLikelihood(
            x=self.x, y=self.y, sigma=None, function=self.function, kwargs=self.kwargs)
        self.assertListEqual(sorted(['param_1', 'param_2', 'sigma']),
                             sorted(list(self.likelihood.parameters.keys())))

    def test_N(self):
        self.assertEqual(3, self.likelihood.n)

    def test_log_l_value(self):
        expected = -3 * np.log(2 * np.pi * self.sigma ** 2) / 2
        self.assertEqual(expected, self.likelihood.log_likelihood())

    def test_noise_log_l_value(self):
        expected = np.sum(- (self.y / self.sigma) ** 2 / 2 - np.log(2 * np.pi * self.sigma ** 2) / 2)
        self.assertEqual(expected, self.likelihood.noise_log_likelihood())

    def test_residual(self):
        expected = self.x - self.y
        self.assertTrue(np.array_equal(expected, self.likelihood.residual))


class GaussianLikelihoodUniformXErrorsTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1, 2])
        self.bin_size = 1
        self.sigma = 1

        def func(x, param_1, param_2, **kwargs):
            return x

        self.function = func
        self.kwargs = dict(kwarg_1='test_kwarg')
        self.likelihood = likelihoods.GaussianLikelihoodUniformXErrors(
            x=self.x, y=self.y, sigma=self.sigma, bin_size=self.bin_size, function=self.function, kwargs=self.kwargs)

    def tearDown(self):
        del self.x
        del self.y
        del self.sigma
        del self.bin_size
        del self.function
        del self.kwargs
        del self.likelihood

    def test_xerr(self):
        expected = np.array([1, 1, 1])
        self.assertTrue(np.array_equal(expected, self.likelihood.xerr))

    def test_log_l_value(self):
        expected_x = 0
        expected_y = -3 * np.log(2 * np.pi * self.sigma ** 2) / 2
        expected = expected_x + expected_y
        self.assertEqual(expected, self.likelihood.log_likelihood())
        self.assertEqual(expected_x, self.likelihood.log_likelihood_x())
        self.assertEqual(expected_y, self.likelihood.log_likelihood_y())

    def test_noise_log_l_value(self):
        with mock.patch("redback.likelihoods.GaussianLikelihood._gaussian_log_likelihood") as m:
            m.return_value = 0
            self.assertEqual(0, self.likelihood.noise_log_likelihood())


class GaussianLikelihoodQuadratureNoiseTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1, 2])
        self.bin_size = 1
        self.sigma_i = 1

        def func(x, param_1, param_2, **kwargs):
            return x

        self.function = func
        self.kwargs = dict(kwarg_1='test_kwarg')
        self.likelihood = likelihoods.GaussianLikelihoodQuadratureNoise(
            x=self.x, y=self.y, sigma_i=self.sigma_i, function=self.function, kwargs=self.kwargs)
        self.likelihood.parameters['sigma'] = 1

    def tearDown(self):
        del self.x
        del self.y
        del self.sigma_i
        del self.bin_size
        del self.function
        del self.kwargs
        del self.likelihood

    def test_sigma_i(self):
        self.assertEqual(self.sigma_i, self.likelihood.sigma_i)

    def test_full_sigma(self):
        self.assertEqual(np.sqrt(2), self.likelihood.full_sigma)

    def test_log_l_value(self):
        expected = -3 * np.log(2 * np.pi * np.sqrt(2) ** 2) / 2
        self.assertEqual(expected, self.likelihood.log_likelihood())

    def test_noise_log_l_value(self):
        with mock.patch("redback.likelihoods.GaussianLikelihood._gaussian_log_likelihood") as m:
            m.return_value = 0
            self.assertEqual(0, self.likelihood.noise_log_likelihood())


class GaussianLikelihoodQuadratureNoiseNonDetectionsTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1, 2])
        self.bin_size = 1
        self.sigma_i = 1
        self.upperlimit_kwargs = dict(upperlimit_kwarg_1='test', flux=1)

        def func(x, param_1, param_2, **kwargs):
            return x

        self.function = func
        self.kwargs = dict(kwarg_1='test_kwarg')
        self.likelihood = likelihoods.GaussianLikelihoodQuadratureNoiseNonDetections(
            x=self.x, y=self.y, sigma_i=self.sigma_i, function=self.function,
            kwargs=self.kwargs, upperlimit_kwargs=self.upperlimit_kwargs)
        self.likelihood.parameters['sigma'] = 1

    def tearDown(self):
        del self.x
        del self.y
        del self.sigma_i
        del self.bin_size
        del self.function
        del self.kwargs
        del self.likelihood

    def test_upper_limit_kwargs(self):
        self.assertDictEqual(self.upperlimit_kwargs, self.likelihood.upperlimit_kwargs)

    def test_upperlimit_flux(self):
        self.assertEqual(self.upperlimit_kwargs['flux'], self.likelihood.upperlimit_flux)

    def test_log_likelihood_y_value(self):
        expected = -3 * np.log(2 * np.pi * np.sqrt(2) ** 2) / 2
        self.assertEqual(expected, self.likelihood.log_likelihood_y())

    def test_log_likelihood_upper_limit_exceed(self):
        self.assertEqual(np.nan_to_num(-np.inf), self.likelihood.log_likelihood_upper_limit())

    def test_log_likelihood_upper_limit(self):
        new_upper_limit = 5
        self.likelihood.upperlimit_kwargs['flux'] = new_upper_limit
        expected = -3*np.log(new_upper_limit)
        self.assertEqual(expected, self.likelihood.log_likelihood_upper_limit())

    def test_log_likelihood_exceed(self):
        self.assertEqual(np.nan_to_num(-np.inf), self.likelihood.log_likelihood())

    def test_log_likelihood(self):
        new_upper_limit = 5
        self.likelihood.upperlimit_kwargs['flux'] = new_upper_limit
        expected_y = -3 * np.log(2 * np.pi * np.sqrt(2) ** 2) / 2
        expected_upper_limit = -3*np.log(new_upper_limit)
        expected = expected_y + expected_upper_limit
        self.assertEqual(expected, self.likelihood.log_likelihood())


class GRBGaussianLikelihoodTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1, 2])
        self.sigma = 1

        def func(x, param_1, param_2, **kwargs):
            return x

        self.function = func
        self.kwargs = dict(kwarg_1='test_kwarg')
        self.likelihood = likelihoods.GRBGaussianLikelihood(
            x=self.x, y=self.y, sigma=self.sigma, function=self.function, kwargs=self.kwargs)

    def tearDown(self):
        del self.x
        del self.y
        del self.sigma
        del self.function
        del self.kwargs
        del self.likelihood


class PoissonLikelihoodTest(unittest.TestCase):

    def setUp(self):
        self.time = np.array([1, 2, 3])
        self.counts = np.array([1, 2, 3])

        def func(x, param_1, param_2, **kwargs):
            return x

        self.function = func
        self.dt = 3
        self.kwargs = dict(dt=self.dt)
        self.integrated_rate_function = True
        self.likelihood = likelihoods.PoissonLikelihood(
            time=self.time, counts=self.counts, function=self.function,
            integrated_rate_function=self.integrated_rate_function, dt=self.dt, kwargs=self.kwargs)

    def tearDown(self):
        del self.time
        del self.counts
        del self.function
        del self.dt
        del self.kwargs
        del self.integrated_rate_function
        del self.likelihood

    def test_set_time(self):
        self.assertTrue(np.array_equal(self.time, self.likelihood.time))

    def test_counts(self):
        self.assertTrue(np.array_equal(self.counts, self.likelihood.counts))

    def test_dt(self):
        self.assertEqual(self.dt, self.likelihood.dt)

    def test_dt_no_dt_given(self):
        self.likelihood.dt = None
        expected = self.time[1] - self.time[0]
        self.assertEqual(expected, self.likelihood.dt)

    def test_background_rate(self):
        self.assertEqual(0, self.likelihood.background_rate)

    def test_noise_log_likelihood(self):
        with mock.patch("redback.likelihoods.PoissonLikelihood._poisson_log_likelihood") as m:
            expected = 0
            m.return_value = expected
            actual = self.likelihood.noise_log_likelihood()
            self.assertEqual(expected, actual)
            m.assert_called_with(rate=0)

    def test_log_likelihood_integrated_rate_function(self):
        with mock.patch("redback.likelihoods.PoissonLikelihood._poisson_log_likelihood") as m:
            expected = 0
            m.return_value = expected
            actual = self.likelihood.log_likelihood()
            self.assertEqual(expected, actual)
            self.assertTrue(np.array_equal(np.array([1, 2, 3]), m.call_args[1]['rate']))

    def test_log_likelihood_not_integrated_rate_function(self):
        self.likelihood.integrated_rate_function = False
        with mock.patch("redback.likelihoods.PoissonLikelihood._poisson_log_likelihood") as m:
            expected = 0
            m.return_value = expected
            actual = self.likelihood.log_likelihood()
            self.assertEqual(expected, actual)
            self.assertTrue(np.array_equal(np.array([3, 6, 9]), m.call_args[1]['rate']))

    def test_log_likelihood_value(self):
        expected = -6 + np.log(9)
        actual = self.likelihood.log_likelihood()
        self.assertEqual(expected, actual)

class MaximumLikelihoodTest(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 10, 50)

        def func(x, m, c, **kwargs):
            return m*x + c

        self.m = 2
        self.c = 3
        ytrue = func(self.x, self.m, self.c)
        self.sigma = 0.2
        noise = np.random.normal(0, self.sigma, len(self.x))
        self.yobs = ytrue + noise
        self.function = func
        self.kwargs = dict(kwarg_1='test_kwarg')

        priors = bilby.core.prior.PriorDict()
        priors['m'] = bilby.core.prior.Uniform(minimum=0, maximum=10, name='m')
        priors['c'] = bilby.core.prior.Uniform(minimum=0, maximum=10, name='c')
        fid = priors.sample()

        self.likelihood = likelihoods.GaussianLikelihood(
            x=self.x, y=self.yobs, sigma=self.sigma, function=self.function, kwargs=self.kwargs, priors=priors,
            fiducial_parameters=fid)

    def tearDown(self):
        del self.x
        del self.yobs
        del self.sigma
        del self.function
        del self.kwargs
        del self.m
        del self.c
        del self.likelihood

    def test_maximum_likelihood(self):
        maxl_parameters = self.likelihood.find_maximum_likelihood_parameters()
        self.assertAlmostEqual(maxl_parameters['m'], self.m, places=0)
        self.assertAlmostEqual(maxl_parameters['c'], self.c, places=0)

class GaussianLikelihoodWithUpperLimitsTest(unittest.TestCase):
    """Test GaussianLikelihoodWithUpperLimits class"""

    def setUp(self):
        self.x = np.array([0, 1, 2, 3, 4])
        self.y = np.array([0.5, 1.2, 2.1, 3.5, 4.8])
        self.sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        self.detections = np.array([True, True, True, False, False])
        self.upper_limit_sigma = 2.0

        def func(x, param_1, **kwargs):
            return x * param_1

        self.function = func
        self.kwargs = {}
        self.likelihood = likelihoods.GaussianLikelihoodWithUpperLimits(
            x=self.x, y=self.y, sigma=self.sigma, function=self.function,
            detections=self.detections, upper_limit_sigma=self.upper_limit_sigma,
            kwargs=self.kwargs)

    def test_initialization(self):
        """Test basic initialization"""
        self.assertEqual(len(self.likelihood.x), 5)
        self.assertTrue(hasattr(self.likelihood, 'detections'))
        self.assertEqual(self.likelihood.upper_limit_sigma, 2.0)

    def test_detections_setter_validation(self):
        """Test that detections setter validates length"""
        with self.assertRaises(ValueError):
            self.likelihood.detections = np.array([True, False])  # Wrong length

    def test_upper_limit_sigma_setter_scalar(self):
        """Test upper_limit_sigma setter with scalar"""
        self.likelihood.upper_limit_sigma = 3.0
        result = self.likelihood.get_upper_limit_sigma_values()
        np.testing.assert_array_equal(result, np.array([3.0, 3.0]))

    def test_upper_limit_sigma_setter_array(self):
        """Test upper_limit_sigma setter with array"""
        self.likelihood.upper_limit_sigma = np.array([2.0, 3.0])
        result = self.likelihood.get_upper_limit_sigma_values()
        np.testing.assert_array_equal(result, np.array([2.0, 3.0]))

    def test_upper_limit_sigma_wrong_length(self):
        """Test upper_limit_sigma with wrong length raises error"""
        with self.assertRaises(ValueError):
            self.likelihood.upper_limit_sigma = np.array([1.0, 2.0, 3.0])


class MixtureGaussianLikelihoodTest(unittest.TestCase):
    """Test MixtureGaussianLikelihood class for outlier detection"""

    def setUp(self):
        self.x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Add one outlier at index 5
        self.y = np.array([0.1, 1.0, 2.1, 2.9, 4.1, 15.0, 6.0, 7.1, 7.9, 9.0])
        self.sigma = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        def func(x, param_1, **kwargs):
            return x * param_1

        self.function = func
        self.kwargs = {}
        self.likelihood = likelihoods.MixtureGaussianLikelihood(
            x=self.x, y=self.y, sigma=self.sigma, function=self.function,
            kwargs=self.kwargs)

    def test_initialization(self):
        """Test that object can be instantiated"""
        self.assertEqual(len(self.likelihood.x), 10)
        # Test that p_in and p_out methods exist
        self.assertTrue(hasattr(self.likelihood, 'p_in'))
        self.assertTrue(hasattr(self.likelihood, 'p_out'))


class GaussianLikelihoodWithFractionalNoiseTest(unittest.TestCase):
    """Test GaussianLikelihoodWithFractionalNoise class"""

    def setUp(self):
        self.x = np.array([0, 1, 2, 3, 4])
        self.y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        self.sigma_i = np.array([0.1, 0.2, 0.4, 0.8, 1.6])

        def func(x, param_1, **kwargs):
            return 2.0 ** x

        self.function = func
        self.kwargs = {}
        self.likelihood = likelihoods.GaussianLikelihoodWithFractionalNoise(
            x=self.x, y=self.y, sigma_i=self.sigma_i, function=self.function,
            kwargs=self.kwargs)

    def test_initialization_has_sigma_i(self):
        """Test that sigma_i is stored"""
        self.assertTrue(hasattr(self.likelihood, 'sigma_i'))
        np.testing.assert_array_equal(self.likelihood.sigma_i, self.sigma_i)


class GaussianLikelihoodWithSystematicNoiseTest(unittest.TestCase):
    """Test GaussianLikelihoodWithSystematicNoise class"""

    def setUp(self):
        self.x = np.array([0, 1, 2, 3, 4])
        self.y = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        self.sigma_i = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        def func(x, param_1, **kwargs):
            return x + param_1

        self.function = func
        self.kwargs = {}
        self.likelihood = likelihoods.GaussianLikelihoodWithSystematicNoise(
            x=self.x, y=self.y, sigma_i=self.sigma_i, function=self.function,
            kwargs=self.kwargs)

    def test_initialization_has_sigma_i(self):
        """Test that sigma_i is stored"""
        self.assertTrue(hasattr(self.likelihood, 'sigma_i'))
        np.testing.assert_array_equal(self.likelihood.sigma_i, self.sigma_i)


class GaussianLikelihoodValidationTest(unittest.TestCase):
    """Test validation and error handling in likelihood classes"""

    def setUp(self):
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1, 2])

        def func(x, param_1, **kwargs):
            return x * param_1

        self.function = func
        self.kwargs = {}

    def test_sigma_2d_wrong_shape(self):
        """Test that 2D sigma with wrong shape raises ValueError"""
        wrong_sigma = np.array([[1, 0], [0, 1]])  # 2x2 instead of 3x3
        with self.assertRaises(ValueError):
            likelihood = likelihoods.GaussianLikelihood(
                x=self.x, y=self.y, sigma=wrong_sigma, function=self.function,
                kwargs=self.kwargs)

    def test_model_output_property(self):
        """Test model_output property"""
        likelihood = likelihoods.GaussianLikelihood(
            x=self.x, y=self.y, sigma=1.0, function=self.function,
            kwargs=self.kwargs)
        
        # Set parameters
        likelihood.parameters = {'param_1': 2.0}
        model_output = likelihood.model_output
        expected = self.x * 2.0
        np.testing.assert_array_almost_equal(model_output, expected)


if __name__ == '__main__':
    unittest.main()
