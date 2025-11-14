"""
Unit tests for the Redback plugin system.

Tests the entry point based plugin discovery and loading mechanism.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestPluginDiscovery(unittest.TestCase):
    """Test plugin discovery functionality."""

    def test_discover_model_plugins_exists(self):
        """Test that discover_model_plugins function exists."""
        from redback.model_library import discover_model_plugins
        self.assertTrue(callable(discover_model_plugins))

    def test_discover_base_model_plugins_exists(self):
        """Test that discover_base_model_plugins function exists."""
        from redback.model_library import discover_base_model_plugins
        self.assertTrue(callable(discover_base_model_plugins))

    def test_model_dictionaries_exist(self):
        """Test that model dictionaries are created."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_models_dict, dict)
        self.assertIsInstance(ml.base_models_dict, dict)
        self.assertIsInstance(ml.modules_dict, dict)

    def test_model_dictionaries_populated(self):
        """Test that model dictionaries contain models."""
        import redback.model_library as ml
        self.assertGreater(len(ml.all_models_dict), 0,
                          "all_models_dict should contain models")
        self.assertGreater(len(ml.base_models_dict), 0,
                          "base_models_dict should contain base models")
        self.assertGreater(len(ml.modules_dict), 0,
                          "modules_dict should contain modules")

    def test_built_in_models_loaded(self):
        """Test that built-in models are loaded."""
        import redback.model_library as ml

        # Check for some known built-in model modules
        module_names = [name for name in ml.modules_dict.keys()]

        # Should have common model types
        known_modules = [
            'kilonova_models',
            'supernova_models',
            'afterglow_models',
        ]

        for module in known_modules:
            self.assertIn(module, module_names,
                         f"Built-in module {module} should be loaded")

    @patch('redback.model_library.entry_points')
    def test_plugin_discovery_with_mock_entry_point(self, mock_entry_points):
        """Test plugin discovery with a mocked entry point."""
        # Create a mock entry point
        mock_ep = Mock()
        mock_ep.name = 'test_plugin'
        mock_ep.value = 'test_module'

        # Create a mock module with a test function
        mock_module = Mock()
        mock_module.__name__ = 'test_module'

        def test_model(time, param1, **kwargs):
            return time * param1

        mock_module.test_model = test_model

        mock_ep.load.return_value = mock_module

        # Mock entry_points to return our mock entry point
        mock_eps = Mock()
        if sys.version_info >= (3, 10):
            mock_eps.select.return_value = [mock_ep]
        else:
            mock_eps.get.return_value = [mock_ep]

        mock_entry_points.return_value = mock_eps

        # Import the function and test it
        from redback.model_library import discover_model_plugins

        # Reload to pick up the mock
        plugins = discover_model_plugins()

        # Should have loaded the mock module
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0], mock_module)

    @patch('redback.model_library.entry_points')
    def test_plugin_discovery_handles_errors(self, mock_entry_points):
        """Test that plugin discovery handles errors gracefully."""
        # Create a mock entry point that raises an error
        mock_ep = Mock()
        mock_ep.name = 'broken_plugin'
        mock_ep.value = 'broken_module'
        mock_ep.load.side_effect = ImportError("Module not found")

        mock_eps = Mock()
        if sys.version_info >= (3, 10):
            mock_eps.select.return_value = [mock_ep]
        else:
            mock_eps.get.return_value = [mock_ep]

        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins

        # Should not raise an exception, but return empty list
        with self.assertWarns(RuntimeWarning):
            plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 0)

    def test_models_are_callable(self):
        """Test that models in the dictionary are callable."""
        import redback.model_library as ml

        # Sample some models and check they're callable
        sample_size = min(10, len(ml.all_models_dict))
        for model_name in list(ml.all_models_dict.keys())[:sample_size]:
            model = ml.all_models_dict[model_name]
            self.assertTrue(callable(model),
                          f"Model {model_name} should be callable")


class TestPluginSystemIntegration(unittest.TestCase):
    """Integration tests for the plugin system."""

    def test_models_dict_type_consistency(self):
        """Test that all entries in model dictionaries are of consistent types."""
        import redback.model_library as ml

        # All keys should be strings
        for key in ml.all_models_dict.keys():
            self.assertIsInstance(key, str)

        # All values should be callable
        for value in ml.all_models_dict.values():
            self.assertTrue(callable(value))

    def test_base_models_subset_of_all_models(self):
        """Test that base models are a subset of all models."""
        import redback.model_library as ml

        for base_model_name in ml.base_models_dict.keys():
            self.assertIn(base_model_name, ml.all_models_dict,
                         f"Base model {base_model_name} should be in all_models_dict")

    def test_modules_dict_structure(self):
        """Test that modules_dict has the expected structure."""
        import redback.model_library as ml

        # modules_dict should have module names as keys
        # and dictionaries of models as values
        for module_name, models in ml.modules_dict.items():
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(models, dict)

            # Each model should be callable
            for model_name, model_func in models.items():
                self.assertIsInstance(model_name, str)
                self.assertTrue(callable(model_func))


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the plugin system maintains backward compatibility."""

    def test_existing_imports_still_work(self):
        """Test that existing import patterns still work."""
        # These should not raise ImportError
        from redback.model_library import all_models_dict
        from redback.model_library import base_models_dict
        from redback.model_library import modules_dict

        self.assertIsNotNone(all_models_dict)
        self.assertIsNotNone(base_models_dict)
        self.assertIsNotNone(modules_dict)

    def test_model_access_by_string_name(self):
        """Test that models can still be accessed by string name."""
        import redback.model_library as ml

        # Pick a model if available
        if len(ml.all_models_dict) > 0:
            model_name = list(ml.all_models_dict.keys())[0]
            model = ml.all_models_dict[model_name]

            self.assertTrue(callable(model))
            self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
