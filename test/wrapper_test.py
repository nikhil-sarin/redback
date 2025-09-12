import unittest
import warnings

from redback.wrappers import cond_jit


class ConditionalJITTest(unittest.TestCase):

    def test_wrapper_no_args_implicit(self):
        """Check that we can define a conditional JIT wrapper (with
        no arguments) and still evaluate the function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Leaving out the parentheses will pass the function itself
            # as the first argument to the decorator.
            @cond_jit
            def _my_func(x):
                return x + 1

        self.assertEqual(_my_func(1), 2)

    def test_wrapper_no_args_explicit(self):
        """Check that we can define a conditional JIT wrapper (with
        no arguments) and still evaluate the function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Including empty parentheses will pass NO arguments
            # (including the function itself).
            @cond_jit()
            def _my_func(x):
                return x + 1

        self.assertEqual(_my_func(1), 2)

    def test_wrapper_args(self):
        """Check that we can define a conditional JIT wrapper (with
        arguments) and still evaluate the function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Only the included arguments will be passed
            # (not the function itself).
            @cond_jit(nopython=True, fastmath=True, cache=True)
            def _my_func(x):
                return x + 1

        self.assertEqual(_my_func(1), 2)
