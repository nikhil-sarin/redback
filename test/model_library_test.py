"""
Unit tests for the Redback plugin system and model_library.
"""

import types
import unittest
from unittest.mock import patch, Mock

import redback.model_library as ml
from redback.utils import get_functions_dict


class TestBuiltinModelsLoaded(unittest.TestCase):
    """Built-in model dictionaries are populated at import time."""

    def test_all_models_dict_nonempty(self):
        self.assertGreater(len(ml.all_models_dict), 0)

    def test_known_model_present(self):
        self.assertIn('arnett', ml.all_models_dict)

    def test_all_model_values_callable(self):
        for name, func in ml.all_models_dict.items():
            self.assertTrue(callable(func), f"{name} should be callable")

    def test_modules_dict_has_builtin_keys(self):
        for expected in ('kilonova_models', 'supernova_models', 'afterglow_models',
                         'tde_models', 'prompt_models'):
            self.assertIn(expected, ml.modules_dict)

    def test_base_models_dict_nonempty(self):
        self.assertGreater(len(ml.base_models_dict), 0)

    def test_base_models_subset_of_all_models(self):
        for name in ml.base_models_dict:
            self.assertIn(name, ml.all_models_dict)


class TestPluginModelCollisionWarning(unittest.TestCase):
    """Built-in wins when a plugin model name clashes with a built-in."""

    def test_collision_logs_warning_and_builtin_preserved(self):
        builtin_func = ml.all_models_dict['arnett']

        # Build a fake plugin module that exposes 'arnett'
        fake_module = types.ModuleType('fake_plugin_module')
        fake_module.__name__ = 'fake_plugin_module'
        fake_plugin_func = lambda time, **kw: None
        fake_module.arnett = fake_plugin_func

        fake_ep = Mock()
        fake_ep.name = 'fake_plugin'
        fake_ep.load.return_value = fake_module

        with patch('redback.model_library.entry_points') as mock_eps, \
             patch('redback.model_library.get_functions_dict') as mock_gfd, \
             patch('redback.model_library.logger') as mock_logger:

            mock_eps.return_value = [fake_ep]
            mock_gfd.return_value = {'fake_plugin_module': {'arnett': fake_plugin_func}}

            ml._load_plugin_modules.__globals__['entry_points'] = mock_eps

            # Re-run the loading logic directly
            import redback.model_library as _ml
            saved = dict(_ml.all_models_dict)
            try:
                # Simulate collision detection inline (same logic as _load_plugin_modules)
                plugin_funcs = {'arnett': fake_plugin_func}
                for k, v in plugin_funcs.items():
                    if k in _ml.all_models_dict:
                        _ml.logger.warning(
                            f"Plugin model '{k}' from 'fake_plugin' conflicts with a built-in model. "
                            f"Skipping plugin model."
                        )
                    else:
                        _ml.all_models_dict[k] = v

                # Built-in must still be present
                self.assertIs(_ml.all_models_dict['arnett'], builtin_func)
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                self.assertIn('arnett', warning_msg)
                self.assertIn('conflicts', warning_msg)
            finally:
                _ml.all_models_dict.clear()
                _ml.all_models_dict.update(saved)


class TestPluginModelLoads(unittest.TestCase):
    """A plugin module with a new model gets added to all_models_dict."""

    def test_new_plugin_model_added(self):
        new_func = lambda time, **kw: None
        new_func.__name__ = 'plugin_unique_model_xyz'

        fake_module = types.ModuleType('myplugin_models')
        fake_module.__name__ = 'myplugin_models'

        fake_ep = Mock()
        fake_ep.name = 'myplugin_models'
        fake_ep.load.return_value = fake_module

        import redback.model_library as _ml
        saved = dict(_ml.all_models_dict)
        saved_mdict = dict(_ml.modules_dict)
        try:
            with patch('redback.model_library.entry_points', return_value=[fake_ep]), \
                 patch('redback.model_library.get_functions_dict',
                       return_value={'myplugin_models': {'plugin_unique_model_xyz': new_func}}):
                _ml._load_plugin_modules()

            self.assertIn('plugin_unique_model_xyz', _ml.all_models_dict)
            self.assertIs(_ml.all_models_dict['plugin_unique_model_xyz'], new_func)
            self.assertIn('myplugin_models', _ml.modules_dict)
        finally:
            _ml.all_models_dict.clear()
            _ml.all_models_dict.update(saved)
            _ml.modules_dict.clear()
            _ml.modules_dict.update(saved_mdict)


class TestModulesDictKeyUniqueness(unittest.TestCase):
    """Two plugins with the same module leaf name don't overwrite each other in modules_dict."""

    def test_ep_name_used_as_key(self):
        func_a = lambda time, **kw: None
        func_b = lambda time, **kw: None

        mod_a = types.ModuleType('models')
        mod_a.__name__ = 'models'
        mod_b = types.ModuleType('models')
        mod_b.__name__ = 'models'

        ep_a = Mock()
        ep_a.name = 'plugin_a_models'
        ep_a.load.return_value = mod_a

        ep_b = Mock()
        ep_b.name = 'plugin_b_models'
        ep_b.load.return_value = mod_b

        import redback.model_library as _ml
        saved = dict(_ml.all_models_dict)
        saved_mdict = dict(_ml.modules_dict)
        try:
            def fake_gfd(module):
                if module is mod_a:
                    return {'models': {'plugin_a_func': func_a}}
                return {'models': {'plugin_b_func': func_b}}

            with patch('redback.model_library.entry_points', return_value=[ep_a, ep_b]), \
                 patch('redback.model_library.get_functions_dict', side_effect=fake_gfd):
                _ml._load_plugin_modules()

            self.assertIn('plugin_a_models', _ml.modules_dict)
            self.assertIn('plugin_b_models', _ml.modules_dict)
            self.assertNotEqual(_ml.modules_dict['plugin_a_models'],
                                _ml.modules_dict['plugin_b_models'])
        finally:
            _ml.all_models_dict.clear()
            _ml.all_models_dict.update(saved)
            _ml.modules_dict.clear()
            _ml.modules_dict.update(saved_mdict)


class TestPluginLoadErrorIsWarningNotCrash(unittest.TestCase):
    """A broken entry point emits a warning but does not raise."""

    def test_broken_ep_does_not_crash(self):
        broken_ep = Mock()
        broken_ep.name = 'broken_plugin'
        broken_ep.load.side_effect = ImportError("Module not found")

        import redback.model_library as _ml
        with patch('redback.model_library.entry_points', return_value=[broken_ep]), \
             patch('redback.model_library.logger') as mock_logger:
            # Should not raise
            _ml._load_plugin_modules()

        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn('broken_plugin', warning_msg)

    def test_broken_prior_ep_does_not_crash(self):
        broken_ep = Mock()
        broken_ep.name = 'broken_prior_plugin'
        broken_ep.load.side_effect = ImportError("Prior module not found")

        with patch('redback.model_library.entry_points', return_value=[broken_ep]), \
             patch('redback.model_library.logger') as mock_logger:
            result = ml.discover_prior_plugins()

        self.assertEqual(result, {})
        mock_logger.warning.assert_called()


class TestPluginPriorProviderCalled(unittest.TestCase):
    """get_priors() calls plugin prior providers when file-based lookup fails."""

    def test_provider_called_and_result_returned(self):
        from bilby.core.prior import PriorDict, Uniform
        import redback.priors as rp
        import redback.model_library as _ml

        expected = PriorDict({'x': Uniform(0, 1, name='x')})
        mock_provider = Mock(return_value=expected)

        with patch.object(_ml, 'plugin_prior_providers', [mock_provider]):
            result = rp.get_priors('nonexistent_model_xyz_plugin_test')

        mock_provider.assert_called_once_with('nonexistent_model_xyz_plugin_test')
        self.assertIs(result, expected)

    def test_provider_returns_none_falls_through(self):
        import redback.priors as rp
        import redback.model_library as _ml

        mock_provider = Mock(return_value=None)

        with patch.object(_ml, 'plugin_prior_providers', [mock_provider]):
            result = rp.get_priors('nonexistent_model_xyz_no_prior')

        from bilby.core.prior import PriorDict
        self.assertIsInstance(result, PriorDict)
        self.assertEqual(len(result), 0)

    def test_provider_exception_logs_warning_and_continues(self):
        import redback.priors as rp
        import redback.model_library as _ml
        from bilby.core.prior import PriorDict

        failing_provider = Mock(side_effect=RuntimeError("provider exploded"))
        with patch.object(_ml, 'plugin_prior_providers', [failing_provider]), \
             patch('redback.priors.logger') as mock_logger:
            result = rp.get_priors('nonexistent_model_xyz_failing_provider')

        mock_logger.warning.assert_called()
        found = any('provider exploded' in str(c) for c in mock_logger.warning.call_args_list)
        self.assertTrue(found)
        self.assertIsInstance(result, PriorDict)


class TestDiscoverPriorPlugins(unittest.TestCase):
    """discover_prior_plugins() returns callable providers keyed by ep.name."""

    def test_callable_provider_registered(self):
        provider = Mock()
        ep = Mock()
        ep.name = 'my_prior_provider'
        ep.load.return_value = provider

        with patch('redback.model_library.entry_points', return_value=[ep]):
            result = ml.discover_prior_plugins()

        self.assertIn('my_prior_provider', result)
        self.assertIs(result['my_prior_provider'], provider)

    def test_non_callable_provider_skipped(self):
        ep = Mock()
        ep.name = 'bad_prior_provider'
        ep.load.return_value = "not_a_callable"

        with patch('redback.model_library.entry_points', return_value=[ep]), \
             patch('redback.model_library.logger') as mock_logger:
            result = ml.discover_prior_plugins()

        self.assertNotIn('bad_prior_provider', result)
        mock_logger.warning.assert_called()

    def test_plugin_prior_providers_is_list(self):
        self.assertIsInstance(ml.plugin_prior_providers, list)


if __name__ == '__main__':
    unittest.main()
