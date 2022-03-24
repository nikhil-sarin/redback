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
