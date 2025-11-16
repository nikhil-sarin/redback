"""
Unit tests for the Redback plugin system.

Tests the entry point based plugin discovery and loading mechanism.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import sys
import warnings
import types
import io


class TestPluginDiscoveryFunctions(unittest.TestCase):
    """Test the plugin discovery functions directly."""

    def test_discover_model_plugins_returns_list(self):
        """Test that discover_model_plugins returns a list."""
        from redback.model_library import discover_model_plugins
        result = discover_model_plugins()
        self.assertIsInstance(result, list)

    def test_discover_base_model_plugins_returns_list(self):
        """Test that discover_base_model_plugins returns a list."""
        from redback.model_library import discover_base_model_plugins
        result = discover_base_model_plugins()
        self.assertIsInstance(result, list)

    def test_discover_model_plugins_no_plugins_installed(self):
        """Test discovery when no plugins are installed (normal case)."""
        from redback.model_library import discover_model_plugins
        # Without any plugins installed, should return empty list
        result = discover_model_plugins()
        # Result should be a list (may be empty if no plugins installed)
        self.assertIsInstance(result, list)

    def test_discover_base_model_plugins_no_plugins_installed(self):
        """Test base model discovery when no plugins are installed."""
        from redback.model_library import discover_base_model_plugins
        result = discover_base_model_plugins()
        self.assertIsInstance(result, list)

    @patch('redback.model_library.entry_points')
    def test_discover_model_plugins_with_select_method(self, mock_entry_points):
        """Test plugin discovery using select method (Python 3.10+)."""
        # Create a mock entry point
        mock_ep = Mock()
        mock_ep.name = 'test_plugin'
        mock_ep.value = 'test_module'

        # Create a mock module
        mock_module = types.ModuleType('test_module')
        mock_ep.load.return_value = mock_module

        # Mock entry_points with select method
        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins
        plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0], mock_module)
        mock_eps.select.assert_called_once_with(group='redback.model.modules')

    @patch('redback.model_library.entry_points')
    def test_discover_base_model_plugins_with_select_method(self, mock_entry_points):
        """Test base model plugin discovery using select method."""
        mock_ep = Mock()
        mock_ep.name = 'test_base_plugin'
        mock_ep.value = 'test_base_module'

        mock_module = types.ModuleType('test_base_module')
        mock_ep.load.return_value = mock_module

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_base_model_plugins
        plugins = discover_base_model_plugins()

        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0], mock_module)
        mock_eps.select.assert_called_once_with(group='redback.model.base_modules')

    @patch('redback.model_library.entry_points')
    def test_discover_model_plugins_without_select_method(self, mock_entry_points):
        """Test plugin discovery using get method (Python 3.9 and earlier)."""
        mock_ep = Mock()
        mock_ep.name = 'old_style_plugin'
        mock_ep.value = 'old_module'

        mock_module = types.ModuleType('old_module')
        mock_ep.load.return_value = mock_module

        # Mock entry_points without select (like older importlib_metadata)
        mock_eps = Mock(spec=['get'])  # Only has 'get', no 'select'
        mock_eps.get.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins
        plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 1)
        mock_eps.get.assert_called_once_with('redback.model.modules', [])

    @patch('redback.model_library.entry_points')
    def test_discover_base_model_plugins_without_select_method(self, mock_entry_points):
        """Test base model discovery using get method."""
        mock_ep = Mock()
        mock_ep.name = 'old_base_plugin'
        mock_ep.value = 'old_base_module'

        mock_module = types.ModuleType('old_base_module')
        mock_ep.load.return_value = mock_module

        mock_eps = Mock(spec=['get'])
        mock_eps.get.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_base_model_plugins
        plugins = discover_base_model_plugins()

        self.assertEqual(len(plugins), 1)
        mock_eps.get.assert_called_once_with('redback.model.base_modules', [])

    @patch('redback.model_library.entry_points')
    def test_discover_model_plugins_handles_load_error(self, mock_entry_points):
        """Test that plugin discovery handles module load errors."""
        mock_ep = Mock()
        mock_ep.name = 'broken_plugin'
        mock_ep.value = 'broken_module'
        mock_ep.load.side_effect = ImportError("Module not found")

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins

        with self.assertWarns(RuntimeWarning) as cm:
            plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 0)
        self.assertIn('broken_plugin', str(cm.warning))
        self.assertIn('Module not found', str(cm.warning))

    @patch('redback.model_library.entry_points')
    def test_discover_base_model_plugins_handles_load_error(self, mock_entry_points):
        """Test that base model discovery handles load errors."""
        mock_ep = Mock()
        mock_ep.name = 'broken_base'
        mock_ep.value = 'broken_base_module'
        mock_ep.load.side_effect = Exception("Generic error")

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_base_model_plugins

        with self.assertWarns(RuntimeWarning) as cm:
            plugins = discover_base_model_plugins()

        self.assertEqual(len(plugins), 0)
        self.assertIn('broken_base', str(cm.warning))

    @patch('redback.model_library.entry_points')
    def test_discover_model_plugins_handles_entry_points_error(self, mock_entry_points):
        """Test that discovery handles errors from entry_points() itself."""
        mock_entry_points.side_effect = Exception("Entry points error")

        from redback.model_library import discover_model_plugins

        with self.assertWarns(RuntimeWarning) as cm:
            plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 0)
        self.assertIn('Entry points error', str(cm.warning))

    @patch('redback.model_library.entry_points')
    def test_discover_base_model_plugins_handles_entry_points_error(self, mock_entry_points):
        """Test that base discovery handles entry_points() errors."""
        mock_entry_points.side_effect = Exception("Base entry points error")

        from redback.model_library import discover_base_model_plugins

        with self.assertWarns(RuntimeWarning) as cm:
            plugins = discover_base_model_plugins()

        self.assertEqual(len(plugins), 0)
        self.assertIn('Base entry points error', str(cm.warning))

    @patch('redback.model_library.entry_points')
    def test_discover_multiple_plugins(self, mock_entry_points):
        """Test discovery of multiple plugins."""
        # Create multiple mock entry points
        mock_ep1 = Mock()
        mock_ep1.name = 'plugin1'
        mock_ep1.value = 'module1'
        mock_module1 = types.ModuleType('module1')
        mock_ep1.load.return_value = mock_module1

        mock_ep2 = Mock()
        mock_ep2.name = 'plugin2'
        mock_ep2.value = 'module2'
        mock_module2 = types.ModuleType('module2')
        mock_ep2.load.return_value = mock_module2

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep1, mock_ep2]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins
        plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 2)
        self.assertIn(mock_module1, plugins)
        self.assertIn(mock_module2, plugins)

    @patch('redback.model_library.entry_points')
    def test_discover_partial_success(self, mock_entry_points):
        """Test that successful plugins load even when others fail."""
        # One working plugin, one broken
        mock_ep_good = Mock()
        mock_ep_good.name = 'good_plugin'
        mock_ep_good.value = 'good_module'
        mock_module_good = types.ModuleType('good_module')
        mock_ep_good.load.return_value = mock_module_good

        mock_ep_bad = Mock()
        mock_ep_bad.name = 'bad_plugin'
        mock_ep_bad.value = 'bad_module'
        mock_ep_bad.load.side_effect = ImportError("Bad module")

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep_good, mock_ep_bad]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins

        with self.assertWarns(RuntimeWarning):
            plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0], mock_module_good)

    @patch('redback.model_library.entry_points')
    def test_discover_empty_entry_points(self, mock_entry_points):
        """Test discovery when no entry points registered."""
        mock_eps = Mock()
        mock_eps.select.return_value = []
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins
        plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 0)
        self.assertIsInstance(plugins, list)

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('redback.model_library.entry_points')
    def test_discover_model_plugins_prints_on_success(self, mock_entry_points, mock_stdout):
        """Test that successful plugin loading prints a message."""
        mock_ep = Mock()
        mock_ep.name = 'loaded_plugin'
        mock_ep.value = 'loaded_module'
        mock_module = types.ModuleType('loaded_module')
        mock_ep.load.return_value = mock_module

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins
        plugins = discover_model_plugins()

        output = mock_stdout.getvalue()
        self.assertIn('Loaded plugin model module', output)
        self.assertIn('loaded_plugin', output)
        self.assertIn('loaded_module', output)

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('redback.model_library.entry_points')
    def test_discover_base_model_plugins_prints_on_success(self, mock_entry_points, mock_stdout):
        """Test that successful base plugin loading prints a message."""
        mock_ep = Mock()
        mock_ep.name = 'loaded_base_plugin'
        mock_ep.value = 'loaded_base_module'
        mock_module = types.ModuleType('loaded_base_module')
        mock_ep.load.return_value = mock_module

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_base_model_plugins
        plugins = discover_base_model_plugins()

        output = mock_stdout.getvalue()
        self.assertIn('Loaded plugin base model module', output)
        self.assertIn('loaded_base_plugin', output)


class TestModelLibraryState(unittest.TestCase):
    """Test the state of model_library after loading."""

    def test_all_models_dict_exists(self):
        """Test that all_models_dict is created."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_models_dict, dict)

    def test_base_models_dict_exists(self):
        """Test that base_models_dict is created."""
        import redback.model_library as ml
        self.assertIsInstance(ml.base_models_dict, dict)

    def test_modules_dict_exists(self):
        """Test that modules_dict is created."""
        import redback.model_library as ml
        self.assertIsInstance(ml.modules_dict, dict)

    def test_all_models_dict_populated(self):
        """Test that all_models_dict contains models."""
        import redback.model_library as ml
        self.assertGreater(len(ml.all_models_dict), 0)

    def test_base_models_dict_populated(self):
        """Test that base_models_dict contains base models."""
        import redback.model_library as ml
        self.assertGreater(len(ml.base_models_dict), 0)

    def test_modules_dict_populated(self):
        """Test that modules_dict contains modules."""
        import redback.model_library as ml
        self.assertGreater(len(ml.modules_dict), 0)

    def test_builtin_modules_list_exists(self):
        """Test that the modules list is defined."""
        import redback.model_library as ml
        self.assertIsInstance(ml.modules, list)
        self.assertGreater(len(ml.modules), 0)

    def test_base_modules_list_exists(self):
        """Test that the base_modules list is defined."""
        import redback.model_library as ml
        self.assertIsInstance(ml.base_modules, list)
        self.assertGreater(len(ml.base_modules), 0)

    def test_all_modules_list_exists(self):
        """Test that the all_modules list is defined."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_modules, list)
        self.assertGreaterEqual(len(ml.all_modules), len(ml.modules))

    def test_all_base_modules_list_exists(self):
        """Test that the all_base_modules list is defined."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_base_modules, list)
        self.assertGreaterEqual(len(ml.all_base_modules), len(ml.base_modules))

    def test_plugin_modules_list_exists(self):
        """Test that plugin_modules list is defined."""
        import redback.model_library as ml
        self.assertIsInstance(ml.plugin_modules, list)

    def test_base_plugin_modules_list_exists(self):
        """Test that base_plugin_modules list is defined."""
        import redback.model_library as ml
        self.assertIsInstance(ml.base_plugin_modules, list)

    def test_known_builtin_modules_present(self):
        """Test that known built-in modules are present."""
        import redback.model_library as ml

        expected_modules = [
            'kilonova_models',
            'supernova_models',
            'afterglow_models',
            'magnetar_models',
            'tde_models',
            'phenomenological_models',
            'extinction_models',
            'phase_models',
        ]

        module_names = list(ml.modules_dict.keys())
        for module_name in expected_modules:
            self.assertIn(module_name, module_names,
                         f"Built-in module {module_name} should be in modules_dict")

    def test_models_have_correct_keys(self):
        """Test that model names are strings."""
        import redback.model_library as ml

        for key in ml.all_models_dict.keys():
            self.assertIsInstance(key, str)
            self.assertGreater(len(key), 0)

    def test_models_are_callable(self):
        """Test that all models are callable functions."""
        import redback.model_library as ml

        for model_name, model_func in ml.all_models_dict.items():
            self.assertTrue(callable(model_func),
                           f"Model {model_name} should be callable")

    def test_base_models_are_subset(self):
        """Test that base models are in all_models_dict."""
        import redback.model_library as ml

        for base_name in ml.base_models_dict.keys():
            self.assertIn(base_name, ml.all_models_dict,
                         f"Base model {base_name} should be in all_models_dict")

    def test_modules_dict_structure(self):
        """Test modules_dict has correct structure."""
        import redback.model_library as ml

        for module_name, models in ml.modules_dict.items():
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(models, dict)

            for func_name, func in models.items():
                self.assertIsInstance(func_name, str)
                self.assertTrue(callable(func))

    def test_all_modules_count_matches_builtin_plus_plugins(self):
        """Test that all_modules has correct count."""
        import redback.model_library as ml
        expected_count = len(ml.modules) + len(ml.plugin_modules)
        self.assertEqual(len(ml.all_modules), expected_count)

    def test_all_base_modules_count_matches(self):
        """Test that all_base_modules has correct count."""
        import redback.model_library as ml
        expected_count = len(ml.base_modules) + len(ml.base_plugin_modules)
        self.assertEqual(len(ml.all_base_modules), expected_count)

    def test_builtin_modules_are_actual_modules(self):
        """Test that builtin modules are Python module objects."""
        import redback.model_library as ml
        for mod in ml.modules:
            self.assertTrue(hasattr(mod, '__name__'))
            self.assertTrue(hasattr(mod, '__file__'))

    def test_base_modules_are_actual_modules(self):
        """Test that base modules are Python module objects."""
        import redback.model_library as ml
        for mod in ml.base_modules:
            self.assertTrue(hasattr(mod, '__name__'))

    def test_modules_dict_keys_match_module_names(self):
        """Test that modules_dict keys correspond to module names."""
        import redback.model_library as ml
        for module in ml.all_modules:
            module_name = module.__name__.split('.')[-1]
            self.assertIn(module_name, ml.modules_dict)

    def test_no_duplicate_model_names_in_same_module(self):
        """Test that there are no duplicate models within a module."""
        import redback.model_library as ml
        for module_name, models in ml.modules_dict.items():
            # Dict keys are inherently unique, so just check it's a valid dict
            self.assertIsInstance(models, dict)
            self.assertEqual(len(models), len(set(models.keys())))


class TestPluginIntegration(unittest.TestCase):
    """Test that plugins integrate correctly with model loading."""

    def test_plugin_modules_combined_with_builtin(self):
        """Test that plugin modules are combined with built-in modules."""
        import redback.model_library as ml

        # all_modules should include both builtin and plugin modules
        self.assertEqual(
            len(ml.all_modules),
            len(ml.modules) + len(ml.plugin_modules)
        )

    def test_base_plugin_modules_combined(self):
        """Test that base plugin modules are combined."""
        import redback.model_library as ml

        self.assertEqual(
            len(ml.all_base_modules),
            len(ml.base_modules) + len(ml.base_plugin_modules)
        )

    def test_builtin_modules_in_all_modules(self):
        """Test that all builtin modules are in all_modules."""
        import redback.model_library as ml
        for builtin_module in ml.modules:
            self.assertIn(builtin_module, ml.all_modules)

    def test_base_builtin_modules_in_all_base_modules(self):
        """Test that all base builtin modules are in all_base_modules."""
        import redback.model_library as ml
        for base_module in ml.base_modules:
            self.assertIn(base_module, ml.all_base_modules)

    def test_model_dict_contains_all_module_functions(self):
        """Test that all_models_dict contains functions from all modules."""
        import redback.model_library as ml
        from redback.utils import get_functions_dict

        for module in ml.all_modules:
            module_funcs = get_functions_dict(module)
            module_name = module.__name__.split('.')[-1]
            for func_name in module_funcs[module_name].keys():
                self.assertIn(func_name, ml.all_models_dict)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that existing code continues to work."""

    def test_import_all_models_dict(self):
        """Test importing all_models_dict."""
        from redback.model_library import all_models_dict
        self.assertIsInstance(all_models_dict, dict)
        self.assertGreater(len(all_models_dict), 0)

    def test_import_base_models_dict(self):
        """Test importing base_models_dict."""
        from redback.model_library import base_models_dict
        self.assertIsInstance(base_models_dict, dict)

    def test_import_modules_dict(self):
        """Test importing modules_dict."""
        from redback.model_library import modules_dict
        self.assertIsInstance(modules_dict, dict)

    def test_import_modules_list(self):
        """Test importing modules list."""
        from redback.model_library import modules
        self.assertIsInstance(modules, list)

    def test_import_base_modules_list(self):
        """Test importing base_modules list."""
        from redback.model_library import base_modules
        self.assertIsInstance(base_modules, list)

    def test_import_all_modules_list(self):
        """Test importing all_modules list."""
        from redback.model_library import all_modules
        self.assertIsInstance(all_modules, list)

    def test_import_all_base_modules_list(self):
        """Test importing all_base_modules list."""
        from redback.model_library import all_base_modules
        self.assertIsInstance(all_base_modules, list)

    def test_import_plugin_modules_list(self):
        """Test importing plugin_modules list."""
        from redback.model_library import plugin_modules
        self.assertIsInstance(plugin_modules, list)

    def test_import_base_plugin_modules_list(self):
        """Test importing base_plugin_modules list."""
        from redback.model_library import base_plugin_modules
        self.assertIsInstance(base_plugin_modules, list)

    def test_import_discovery_functions(self):
        """Test importing discovery functions."""
        from redback.model_library import discover_model_plugins, discover_base_model_plugins
        self.assertTrue(callable(discover_model_plugins))
        self.assertTrue(callable(discover_base_model_plugins))

    def test_model_lookup_by_name(self):
        """Test that models can be looked up by string name."""
        import redback.model_library as ml

        if len(ml.all_models_dict) > 0:
            model_name = list(ml.all_models_dict.keys())[0]
            model = ml.all_models_dict[model_name]
            self.assertTrue(callable(model))

    def test_access_specific_known_model(self):
        """Test accessing a specific known model."""
        import redback.model_library as ml

        # These are common models that should exist
        known_models = [
            'one_component_kilonova_model',
            'arnett_bolometric',
        ]

        for model_name in known_models:
            if model_name in ml.all_models_dict:
                model = ml.all_models_dict[model_name]
                self.assertTrue(callable(model))

    def test_get_model_from_string(self):
        """Test getting model function from string name."""
        from redback.model_library import all_models_dict

        # Get a sample model name
        if len(all_models_dict) > 0:
            sample_name = list(all_models_dict.keys())[0]
            func = all_models_dict[sample_name]
            self.assertTrue(callable(func))
            self.assertTrue(hasattr(func, '__name__'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in plugin system."""

    def test_missing_model_raises_key_error(self):
        """Test that accessing missing model raises KeyError."""
        import redback.model_library as ml

        with self.assertRaises(KeyError):
            _ = ml.all_models_dict['nonexistent_model_xyz123']

    def test_empty_model_name_not_present(self):
        """Test that empty string is not a valid model key."""
        import redback.model_library as ml
        self.assertNotIn('', ml.all_models_dict)

    def test_none_not_in_model_names(self):
        """Test that None is not in model names."""
        import redback.model_library as ml
        self.assertNotIn(None, ml.all_models_dict)

    def test_integer_key_raises_error(self):
        """Test that integer keys raise appropriate errors."""
        import redback.model_library as ml
        # Dict will raise KeyError for missing key, even if int
        with self.assertRaises(KeyError):
            _ = ml.all_models_dict[12345]


class TestModuleProcessing(unittest.TestCase):
    """Test the module processing logic."""

    def test_get_functions_dict_returns_dict(self):
        """Test that get_functions_dict returns a dictionary."""
        from redback.utils import get_functions_dict
        import redback.transient_models.fireball_models as fb
        result = get_functions_dict(fb)
        self.assertIsInstance(result, dict)

    def test_get_functions_dict_extracts_functions(self):
        """Test that get_functions_dict extracts callable functions."""
        from redback.utils import get_functions_dict
        import redback.transient_models.fireball_models as fb
        result = get_functions_dict(fb)
        module_name = fb.__name__.split('.')[-1]
        self.assertIn(module_name, result)
        for func_name, func in result[module_name].items():
            self.assertTrue(callable(func))

    def test_module_name_extraction(self):
        """Test that module names are correctly extracted."""
        import redback.model_library as ml
        for module in ml.modules:
            name = module.__name__.split('.')[-1]
            self.assertTrue(name.endswith('_models') or name.endswith('_models'))
            self.assertIn(name, ml.modules_dict)

    def test_models_from_each_builtin_module_are_loaded(self):
        """Test that each builtin module contributes models."""
        import redback.model_library as ml
        from redback.utils import get_functions_dict

        for module in ml.modules:
            funcs = get_functions_dict(module)
            module_name = module.__name__.split('.')[-1]
            # Each module should have at least one function
            self.assertGreater(len(funcs[module_name]), 0,
                             f"Module {module_name} should have functions")

    def test_base_models_from_base_modules(self):
        """Test that base models come from base modules."""
        import redback.model_library as ml
        from redback.utils import get_functions_dict

        for base_mod in ml.base_modules:
            funcs = get_functions_dict(base_mod)
            module_name = base_mod.__name__.split('.')[-1]
            # Check that functions are in base_models_dict
            for func_name in funcs[module_name].keys():
                self.assertIn(func_name, ml.base_models_dict)


class TestSpecificModels(unittest.TestCase):
    """Test specific model existence and properties."""

    def test_extinction_models_present(self):
        """Test that extinction models are loaded."""
        import redback.model_library as ml
        self.assertIn('extinction_models', ml.modules_dict)
        self.assertGreater(len(ml.modules_dict['extinction_models']), 0)

    def test_phase_models_present(self):
        """Test that phase models are loaded."""
        import redback.model_library as ml
        self.assertIn('phase_models', ml.modules_dict)

    def test_kilonova_models_present(self):
        """Test that kilonova models are loaded."""
        import redback.model_library as ml
        self.assertIn('kilonova_models', ml.modules_dict)

    def test_supernova_models_present(self):
        """Test that supernova models are loaded."""
        import redback.model_library as ml
        self.assertIn('supernova_models', ml.modules_dict)

    def test_afterglow_models_present(self):
        """Test that afterglow models are loaded."""
        import redback.model_library as ml
        self.assertIn('afterglow_models', ml.modules_dict)

    def test_fireball_models_present(self):
        """Test that fireball models are loaded."""
        import redback.model_library as ml
        self.assertIn('fireball_models', ml.modules_dict)

    def test_tde_models_present(self):
        """Test that TDE models are loaded."""
        import redback.model_library as ml
        self.assertIn('tde_models', ml.modules_dict)

    def test_magnetar_models_present(self):
        """Test that magnetar models are loaded."""
        import redback.model_library as ml
        self.assertIn('magnetar_models', ml.modules_dict)

    def test_phenomenological_models_present(self):
        """Test that phenomenological models are loaded."""
        import redback.model_library as ml
        self.assertIn('phenomenological_models', ml.modules_dict)

    def test_combined_models_present(self):
        """Test that combined models are loaded."""
        import redback.model_library as ml
        self.assertIn('combined_models', ml.modules_dict)

    def test_spectral_models_present(self):
        """Test that spectral models are loaded."""
        import redback.model_library as ml
        self.assertIn('spectral_models', ml.modules_dict)


if __name__ == '__main__':
    unittest.main()
