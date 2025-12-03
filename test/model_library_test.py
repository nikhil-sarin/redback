"""
Unit tests for the Redback plugin system and model_library.

Tests the entry point based plugin discovery and loading mechanism.
These tests exercise the actual code paths in model_library.py to ensure
proper test coverage.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import warnings
import types
import io


class TestPluginDiscoveryRealExecution(unittest.TestCase):
    """Test the plugin discovery functions with real execution."""

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

    def test_discover_model_plugins_executes_entry_points(self):
        """Test that discovery calls entry_points and checks for select."""
        from redback.model_library import discover_model_plugins
        # This should execute the hasattr check and either select or get path
        result = discover_model_plugins()
        self.assertIsInstance(result, list)

    def test_discover_base_model_plugins_executes_entry_points(self):
        """Test that base discovery calls entry_points and checks for select."""
        from redback.model_library import discover_base_model_plugins
        result = discover_base_model_plugins()
        self.assertIsInstance(result, list)

    def test_discover_model_plugins_multiple_calls(self):
        """Test that discovery can be called multiple times."""
        from redback.model_library import discover_model_plugins
        result1 = discover_model_plugins()
        result2 = discover_model_plugins()
        self.assertEqual(result1, result2)

    def test_discover_base_model_plugins_multiple_calls(self):
        """Test that base discovery can be called multiple times."""
        from redback.model_library import discover_base_model_plugins
        result1 = discover_base_model_plugins()
        result2 = discover_base_model_plugins()
        self.assertEqual(result1, result2)


class TestPluginDiscoveryWithMocks(unittest.TestCase):
    """Test plugin discovery with controlled mock scenarios."""

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

    @patch('redback.model_library.entry_points')
    def test_discover_without_select_method(self, mock_entry_points):
        """Test discovery with entry_points that only has get method."""
        mock_ep = Mock()
        mock_ep.name = 'old_plugin'
        mock_ep.value = 'old_module'
        mock_module = types.ModuleType('old_module')
        mock_ep.load.return_value = mock_module

        mock_eps = Mock(spec=['get'])
        mock_eps.get.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        from redback.model_library import discover_model_plugins
        plugins = discover_model_plugins()

        self.assertEqual(len(plugins), 1)
        mock_eps.get.assert_called_once_with('redback.model.modules', [])

    @patch('redback.model_library.entry_points')
    def test_discover_base_without_select_method(self, mock_entry_points):
        """Test base discovery with entry_points that only has get method."""
        mock_ep = Mock()
        mock_ep.name = 'old_base'
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


class TestModelLibraryModuleState(unittest.TestCase):
    """Test the actual module-level state of model_library."""

    def test_all_models_dict_is_dict(self):
        """Test all_models_dict type."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_models_dict, dict)

    def test_base_models_dict_is_dict(self):
        """Test base_models_dict type."""
        import redback.model_library as ml
        self.assertIsInstance(ml.base_models_dict, dict)

    def test_modules_dict_is_dict(self):
        """Test modules_dict type."""
        import redback.model_library as ml
        self.assertIsInstance(ml.modules_dict, dict)

    def test_all_models_dict_has_models(self):
        """Test all_models_dict is populated."""
        import redback.model_library as ml
        self.assertGreater(len(ml.all_models_dict), 0)

    def test_base_models_dict_has_models(self):
        """Test base_models_dict is populated."""
        import redback.model_library as ml
        self.assertGreater(len(ml.base_models_dict), 0)

    def test_modules_dict_has_modules(self):
        """Test modules_dict is populated."""
        import redback.model_library as ml
        self.assertGreater(len(ml.modules_dict), 0)

    def test_modules_is_list(self):
        """Test modules is a list."""
        import redback.model_library as ml
        self.assertIsInstance(ml.modules, list)
        self.assertGreater(len(ml.modules), 0)

    def test_base_modules_is_list(self):
        """Test base_modules is a list."""
        import redback.model_library as ml
        self.assertIsInstance(ml.base_modules, list)
        self.assertGreater(len(ml.base_modules), 0)

    def test_all_modules_is_list(self):
        """Test all_modules is a list."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_modules, list)
        self.assertGreaterEqual(len(ml.all_modules), len(ml.modules))

    def test_all_base_modules_is_list(self):
        """Test all_base_modules is a list."""
        import redback.model_library as ml
        self.assertIsInstance(ml.all_base_modules, list)

    def test_plugin_modules_is_list(self):
        """Test plugin_modules is a list."""
        import redback.model_library as ml
        self.assertIsInstance(ml.plugin_modules, list)

    def test_base_plugin_modules_is_list(self):
        """Test base_plugin_modules is a list."""
        import redback.model_library as ml
        self.assertIsInstance(ml.base_plugin_modules, list)

    def test_kilonova_models_in_modules_dict(self):
        """Test kilonova_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('kilonova_models', ml.modules_dict)

    def test_supernova_models_in_modules_dict(self):
        """Test supernova_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('supernova_models', ml.modules_dict)

    def test_afterglow_models_in_modules_dict(self):
        """Test afterglow_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('afterglow_models', ml.modules_dict)

    def test_magnetar_models_in_modules_dict(self):
        """Test magnetar_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('magnetar_models', ml.modules_dict)

    def test_tde_models_in_modules_dict(self):
        """Test tde_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('tde_models', ml.modules_dict)

    def test_phenomenological_models_in_modules_dict(self):
        """Test phenomenological_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('phenomenological_models', ml.modules_dict)

    def test_extinction_models_in_modules_dict(self):
        """Test extinction_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('extinction_models', ml.modules_dict)

    def test_phase_models_in_modules_dict(self):
        """Test phase_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('phase_models', ml.modules_dict)

    def test_fireball_models_in_modules_dict(self):
        """Test fireball_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('fireball_models', ml.modules_dict)

    def test_combined_models_in_modules_dict(self):
        """Test combined_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('combined_models', ml.modules_dict)

    def test_spectral_models_in_modules_dict(self):
        """Test spectral_models module is loaded."""
        import redback.model_library as ml
        self.assertIn('spectral_models', ml.modules_dict)

    def test_model_keys_are_strings(self):
        """Test that all model keys are strings."""
        import redback.model_library as ml
        for key in ml.all_models_dict.keys():
            self.assertIsInstance(key, str)
            self.assertGreater(len(key), 0)

    def test_model_values_are_callable(self):
        """Test that all model values are callable."""
        import redback.model_library as ml
        for model_name, model_func in ml.all_models_dict.items():
            self.assertTrue(callable(model_func), f"{model_name} should be callable")

    def test_base_models_in_all_models(self):
        """Test that all base models are in all_models_dict."""
        import redback.model_library as ml
        for base_name in ml.base_models_dict.keys():
            self.assertIn(base_name, ml.all_models_dict)

    def test_modules_dict_values_are_dicts(self):
        """Test modules_dict values are dictionaries."""
        import redback.model_library as ml
        for module_name, models in ml.modules_dict.items():
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(models, dict)

    def test_modules_dict_functions_are_callable(self):
        """Test that functions in modules_dict are callable."""
        import redback.model_library as ml
        for module_name, models in ml.modules_dict.items():
            for func_name, func in models.items():
                self.assertIsInstance(func_name, str)
                self.assertTrue(callable(func))

    def test_all_modules_count(self):
        """Test all_modules has correct count."""
        import redback.model_library as ml
        expected_count = len(ml.modules) + len(ml.plugin_modules)
        self.assertEqual(len(ml.all_modules), expected_count)

    def test_all_base_modules_count(self):
        """Test all_base_modules has correct count."""
        import redback.model_library as ml
        expected_count = len(ml.base_modules) + len(ml.base_plugin_modules)
        self.assertEqual(len(ml.all_base_modules), expected_count)

    def test_builtin_modules_have_names(self):
        """Test builtin modules have __name__ attribute."""
        import redback.model_library as ml
        for mod in ml.modules:
            self.assertTrue(hasattr(mod, '__name__'))

    def test_builtin_modules_have_files(self):
        """Test builtin modules have __file__ attribute."""
        import redback.model_library as ml
        for mod in ml.modules:
            self.assertTrue(hasattr(mod, '__file__'))

    def test_module_names_match_dict_keys(self):
        """Test module names match modules_dict keys."""
        import redback.model_library as ml
        for module in ml.all_modules:
            module_name = module.__name__.split('.')[-1]
            self.assertIn(module_name, ml.modules_dict)

    def test_builtin_modules_in_all_modules(self):
        """Test all builtin modules are in all_modules."""
        import redback.model_library as ml
        for builtin_module in ml.modules:
            self.assertIn(builtin_module, ml.all_modules)

    def test_base_modules_in_all_base_modules(self):
        """Test all base modules are in all_base_modules."""
        import redback.model_library as ml
        for base_module in ml.base_modules:
            self.assertIn(base_module, ml.all_base_modules)


class TestModuleProcessingLogic(unittest.TestCase):
    """Test the module processing logic."""

    def test_get_functions_dict_with_fireball_models(self):
        """Test get_functions_dict extracts functions."""
        from redback.utils import get_functions_dict
        import redback.transient_models.fireball_models as fb
        result = get_functions_dict(fb)
        self.assertIsInstance(result, dict)
        module_name = fb.__name__.split('.')[-1]
        self.assertIn(module_name, result)

    def test_get_functions_dict_functions_are_callable(self):
        """Test extracted functions are callable."""
        from redback.utils import get_functions_dict
        import redback.transient_models.fireball_models as fb
        result = get_functions_dict(fb)
        module_name = fb.__name__.split('.')[-1]
        for func_name, func in result[module_name].items():
            self.assertTrue(callable(func))

    def test_module_name_extraction_logic(self):
        """Test the __name__.split('.')[-1] logic."""
        import redback.model_library as ml
        for module in ml.modules:
            name = module.__name__.split('.')[-1]
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)
            self.assertIn(name, ml.modules_dict)

    def test_each_module_contributes_functions(self):
        """Test each module has functions."""
        import redback.model_library as ml
        from redback.utils import get_functions_dict
        for module in ml.modules:
            funcs = get_functions_dict(module)
            module_name = module.__name__.split('.')[-1]
            self.assertGreater(len(funcs[module_name]), 0)

    def test_base_module_functions_in_base_dict(self):
        """Test base module functions are in base_models_dict."""
        import redback.model_library as ml
        from redback.utils import get_functions_dict
        for base_mod in ml.base_modules:
            funcs = get_functions_dict(base_mod)
            module_name = base_mod.__name__.split('.')[-1]
            for func_name in funcs[module_name].keys():
                self.assertIn(func_name, ml.base_models_dict)

    def test_all_module_functions_in_all_models_dict(self):
        """Test all module functions are in all_models_dict."""
        import redback.model_library as ml
        from redback.utils import get_functions_dict
        for module in ml.all_modules:
            module_funcs = get_functions_dict(module)
            module_name = module.__name__.split('.')[-1]
            for func_name in module_funcs[module_name].keys():
                self.assertIn(func_name, ml.all_models_dict)


class TestBackwardCompatibilityImports(unittest.TestCase):
    """Test backward compatibility of imports."""

    def test_import_all_models_dict(self):
        """Test importing all_models_dict."""
        from redback.model_library import all_models_dict
        self.assertIsInstance(all_models_dict, dict)

    def test_import_base_models_dict(self):
        """Test importing base_models_dict."""
        from redback.model_library import base_models_dict
        self.assertIsInstance(base_models_dict, dict)

    def test_import_modules_dict(self):
        """Test importing modules_dict."""
        from redback.model_library import modules_dict
        self.assertIsInstance(modules_dict, dict)

    def test_import_modules(self):
        """Test importing modules."""
        from redback.model_library import modules
        self.assertIsInstance(modules, list)

    def test_import_base_modules(self):
        """Test importing base_modules."""
        from redback.model_library import base_modules
        self.assertIsInstance(base_modules, list)

    def test_import_all_modules(self):
        """Test importing all_modules."""
        from redback.model_library import all_modules
        self.assertIsInstance(all_modules, list)

    def test_import_all_base_modules(self):
        """Test importing all_base_modules."""
        from redback.model_library import all_base_modules
        self.assertIsInstance(all_base_modules, list)

    def test_import_plugin_modules(self):
        """Test importing plugin_modules."""
        from redback.model_library import plugin_modules
        self.assertIsInstance(plugin_modules, list)

    def test_import_base_plugin_modules(self):
        """Test importing base_plugin_modules."""
        from redback.model_library import base_plugin_modules
        self.assertIsInstance(base_plugin_modules, list)

    def test_import_discover_functions(self):
        """Test importing discovery functions."""
        from redback.model_library import discover_model_plugins, discover_base_model_plugins
        self.assertTrue(callable(discover_model_plugins))
        self.assertTrue(callable(discover_base_model_plugins))

    def test_model_lookup_by_name(self):
        """Test model lookup by string name."""
        import redback.model_library as ml
        if len(ml.all_models_dict) > 0:
            model_name = list(ml.all_models_dict.keys())[0]
            model = ml.all_models_dict[model_name]
            self.assertTrue(callable(model))

    def test_model_has_name_attribute(self):
        """Test model function has __name__."""
        from redback.model_library import all_models_dict
        if len(all_models_dict) > 0:
            sample_name = list(all_models_dict.keys())[0]
            func = all_models_dict[sample_name]
            self.assertTrue(hasattr(func, '__name__'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling."""

    def test_missing_model_key_error(self):
        """Test KeyError for missing model."""
        import redback.model_library as ml
        with self.assertRaises(KeyError):
            _ = ml.all_models_dict['nonexistent_model_xyz123']

    def test_empty_string_not_in_models(self):
        """Test empty string not a valid key."""
        import redback.model_library as ml
        self.assertNotIn('', ml.all_models_dict)

    def test_none_not_in_models(self):
        """Test None not a valid key."""
        import redback.model_library as ml
        self.assertNotIn(None, ml.all_models_dict)

    def test_integer_key_error(self):
        """Test integer key raises error."""
        import redback.model_library as ml
        with self.assertRaises(KeyError):
            _ = ml.all_models_dict[12345]


if __name__ == '__main__':
    unittest.main()
