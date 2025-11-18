import unittest
from unittest.mock import patch, MagicMock
import redback.wrappers as wrappers


class TestCondJit(unittest.TestCase):
    """Test cond_jit decorator function"""

    def test_cond_jit_with_numba_available(self):
        """Test cond_jit when numba is available"""
        # Define a simple function to decorate
        @wrappers.cond_jit
        def simple_function(x):
            return x * 2

        # Should work normally
        result = simple_function(5)
        self.assertEqual(result, 10)

    def test_cond_jit_with_kwargs_numba_available(self):
        """Test cond_jit with kwargs when numba is available"""
        # Define a function with explicit kwargs
        @wrappers.cond_jit(nopython=True)
        def simple_function(x):
            return x + 1

        # Should work normally
        result = simple_function(5)
        self.assertEqual(result, 6)

    @patch('redback.wrappers.logger')
    def test_cond_jit_without_numba(self, mock_logger):
        """Test cond_jit when numba is not available"""
        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'numba': None}):
            # Force reimport to trigger ImportError path
            import importlib
            importlib.reload(wrappers)

            # Define a function
            @wrappers.cond_jit
            def simple_function(x):
                return x * 3

            # Should still work, just without JIT compilation
            result = simple_function(7)
            self.assertEqual(result, 21)

            # Reload wrappers to restore normal state
            importlib.reload(wrappers)

    @patch('redback.wrappers.logger')
    def test_cond_jit_import_error_with_kwargs(self, mock_logger):
        """Test cond_jit with kwargs when numba import fails"""
        import builtins
        original_import = builtins.__import__

        # Create a mock that raises ImportError
        def mock_import(name, *args, **kwargs):
            if name == 'numba':
                raise ImportError("Mocked numba import failure")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            import importlib
            importlib.reload(wrappers)

            # Define a function with kwargs
            @wrappers.cond_jit(nopython=True)
            def test_func(x):
                return x * 2

            # Should work as no-op decorator
            result = test_func(4)
            self.assertEqual(result, 8)

            # Reload to restore
            importlib.reload(wrappers)

    def test_cond_jit_returns_callable(self):
        """Test that cond_jit returns a callable"""
        decorator = wrappers.cond_jit(nopython=True)
        self.assertTrue(callable(decorator))

    def test_cond_jit_preserves_function_behavior(self):
        """Test that decorated function preserves original behavior"""
        def original_func(a, b):
            return a + b

        decorated_func = wrappers.cond_jit(original_func)

        # Should produce same results
        self.assertEqual(original_func(3, 4), decorated_func(3, 4))
        self.assertEqual(original_func(10, 20), decorated_func(10, 20))

    def test_cond_jit_with_multiple_arguments(self):
        """Test cond_jit with function that takes multiple arguments"""
        @wrappers.cond_jit
        def multi_arg_func(a, b, c):
            return a * b + c

        result = multi_arg_func(2, 3, 4)
        self.assertEqual(result, 10)

    def test_cond_jit_with_return_types(self):
        """Test cond_jit with different return types"""
        @wrappers.cond_jit
        def return_tuple(x):
            return (x, x * 2)

        result = return_tuple(5)
        self.assertEqual(result, (5, 10))

        @wrappers.cond_jit
        def return_list(x):
            return [x, x + 1, x + 2]

        result = return_list(1)
        self.assertEqual(result, [1, 2, 3])
