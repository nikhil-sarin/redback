import numpy as np
import unittest

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
        self.assertEqual(3, self.likelihood.N)

    def test_log_l_value(self):
        expected = -3 * np.log(2 * np.pi * self.sigma ** 2) / 2
        self.assertEqual(expected, self.likelihood.log_likelihood())

    def test_noise_log_l_value(self):
        expected = np.sum(- (self.y / self.sigma) ** 2 / 2 - np.log(2 * np.pi * self.sigma ** 2) / 2)
        self.assertEqual(expected, self.likelihood.noise_log_likelihood())

    def test_residual(self):
        expected = self.x - self.y
        self.assertTrue(np.array_equal(expected, self.likelihood.residual))
