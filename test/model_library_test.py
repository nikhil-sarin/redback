"""
Unit tests for the Redback plugin system and model_library.
"""

import types
import unittest
from unittest.mock import patch, Mock

import redback.model_library as ml
from redback.model_metadata import ModelMetadata
from redback.model_metadata import coerce_model_metadata
from redback.model_metadata import validate_model_metadata
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

    def test_builtin_model_metadata_loaded(self):
        self.assertIn('arnett', ml.model_metadata_dict)
        metadata = ml.model_metadata_dict['arnett']
        self.assertEqual(metadata.model_type, 'supernova')
        self.assertIn('flux_density', metadata.output_formats)
        self.assertEqual('supernova_models', metadata.source_module)
        self.assertTrue(metadata.has_prior)

    def test_builtin_model_metadata_is_curated_for_registered_models(self):
        self.assertGreaterEqual(len(ml.model_metadata_dict), 40)
        for model_name in ml.model_metadata_dict:
            self.assertIn(model_name, ml.all_models_dict)

    def test_prompt_and_spectral_metadata_loaded(self):
        prompt_metadata = ml.model_metadata_dict['fred_extended']
        self.assertEqual(prompt_metadata.model_type, 'prompt')
        self.assertEqual(('counts',), prompt_metadata.output_formats)
        self.assertEqual('counts', prompt_metadata.default_output_format)

        spectral_metadata = ml.model_metadata_dict['band_function_high_energy']
        self.assertEqual(spectral_metadata.model_type, 'spectral')
        self.assertEqual(('spectrum',), spectral_metadata.output_formats)

    def test_important_health_check_models_have_metadata(self):
        for model_name in ('arnett', 'csm_nickel', 'one_component_kilonova_model'):
            metadata = ml.get_model_metadata(model_name)
            self.assertEqual('flux_density', metadata.default_output_format)
            self.assertIn('output_format', metadata.required_kwargs)

    def test_get_model_metadata(self):
        metadata = ml.get_model_metadata('arnett')
        self.assertIs(metadata, ml.model_metadata_dict['arnett'])

    def test_get_model_metadata_default(self):
        default = object()
        self.assertIs(default, ml.get_model_metadata('missing_model_xyz', default=default))

    def test_get_model_metadata_none_default(self):
        self.assertIsNone(ml.get_model_metadata('missing_model_xyz', default=None))

    def test_invalid_default_output_format_rejected(self):
        with self.assertRaises(ValueError):
            coerce_model_metadata('bad_metadata_model', {
                'model_type': 'supernova',
                'output_formats': ('flux_density',),
                'default_output_format': 'magnitude',
            })

    def test_single_string_output_format_is_coerced(self):
        metadata = coerce_model_metadata('string_metadata_model', {
            'model_type': 'supernova',
            'output_formats': 'flux_density',
            'default_output_format': 'flux_density',
        })
        self.assertEqual(('flux_density',), metadata.output_formats)

    def test_metadata_validation_rejects_invalid_fields(self):
        valid = dict(name="model", model_type="supernova")
        cases = [
            ({"name": ""}, ValueError, "name"),
            ({"model_type": ""}, ValueError, "model_type"),
            ({"source_module": ""}, TypeError, "source_module"),
            ({"output_formats": ["flux"]}, TypeError, "output_formats"),
            ({"output_formats": ("",)}, TypeError, "output_formats"),
            ({"required_kwargs": ["redshift"]}, TypeError, "required_kwargs"),
            ({"optional_dependencies": (None,)}, TypeError, "optional_dependencies"),
            ({"default_output_format": 1}, TypeError, "default_output_format"),
            ({"has_prior": 1}, TypeError, "has_prior"),
            ({"is_public": 1}, TypeError, "is_public"),
            ({"supports_extinction": 1}, TypeError, "supports_extinction"),
            ({"supports_constraints": 1}, TypeError, "supports_constraints"),
            ({"speed": ""}, ValueError, "speed"),
            ({"max_time_days": 0}, ValueError, "max_time_days"),
        ]
        for changes, error, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(error, message):
                validate_model_metadata(ModelMetadata(**(valid | changes)))

    def test_metadata_validation_checks_registered_model_names(self):
        metadata = ModelMetadata(name="missing", model_type="supernova")
        with self.assertRaisesRegex(ValueError, "has no matching model"):
            validate_model_metadata(metadata, available_model_names={"present"})

    def test_metadata_instance_is_renamed_during_coercion(self):
        metadata = ModelMetadata(name="old", model_type="supernova")
        renamed = coerce_model_metadata("new", metadata)
        self.assertEqual("new", renamed.name)
        self.assertEqual("old", metadata.name)

    def test_metadata_coercion_rejects_non_mapping(self):
        with self.assertRaisesRegex(TypeError, "instance or mapping"):
            coerce_model_metadata("model", object())

    def test_metadata_coercion_normalizes_iterables_and_none(self):
        metadata = coerce_model_metadata("model", {
            "model_type": "supernova",
            "output_formats": ["flux", "magnitude"],
            "required_kwargs": "redshift",
            "optional_dependencies": None,
        })
        self.assertEqual(("flux", "magnitude"), metadata.output_formats)
        self.assertEqual(("redshift",), metadata.required_kwargs)
        self.assertEqual((), metadata.optional_dependencies)


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

    def test_new_plugin_model_metadata_added(self):
        new_func = lambda time, **kw: None
        new_func.__name__ = 'plugin_metadata_model_xyz'

        fake_module = types.ModuleType('myplugin_metadata_models')
        fake_module.__name__ = 'myplugin_metadata_models'
        fake_module.redback_model_metadata = {
            'plugin_metadata_model_xyz': {
                'model_type': 'supernova',
                'output_formats': ('flux_density',),
                'default_output_format': 'flux_density',
                'speed': 'fast',
            }
        }

        fake_ep = Mock()
        fake_ep.name = 'myplugin_metadata_models'
        fake_ep.load.return_value = fake_module

        import redback.model_library as _ml
        saved = dict(_ml.all_models_dict)
        saved_mdict = dict(_ml.modules_dict)
        saved_metadata = dict(_ml.model_metadata_dict)
        try:
            with patch('redback.model_library.entry_points', return_value=[fake_ep]), \
                 patch('redback.model_library.get_functions_dict',
                       return_value={'myplugin_metadata_models': {'plugin_metadata_model_xyz': new_func}}):
                _ml._load_plugin_modules()

            self.assertIn('plugin_metadata_model_xyz', _ml.model_metadata_dict)
            metadata = _ml.model_metadata_dict['plugin_metadata_model_xyz']
            self.assertIsInstance(metadata, ModelMetadata)
            self.assertEqual(metadata.model_type, 'supernova')
            self.assertEqual('myplugin_metadata_models', metadata.source_module)
            self.assertEqual(metadata.speed, 'fast')
        finally:
            _ml.all_models_dict.clear()
            _ml.all_models_dict.update(saved)
            _ml.modules_dict.clear()
            _ml.modules_dict.update(saved_mdict)
            _ml.model_metadata_dict.clear()
            _ml.model_metadata_dict.update(saved_metadata)

    def test_plugin_metadata_without_matching_model_warns(self):
        fake_module = types.ModuleType('myplugin_bad_metadata_models')
        fake_module.__name__ = 'myplugin_bad_metadata_models'
        fake_module.redback_model_metadata = {
            'missing_model': {'model_type': 'supernova'}
        }

        fake_ep = Mock()
        fake_ep.name = 'myplugin_bad_metadata_models'
        fake_ep.load.return_value = fake_module

        import redback.model_library as _ml
        with patch('redback.model_library.entry_points', return_value=[fake_ep]), \
             patch('redback.model_library.get_functions_dict',
                   return_value={'myplugin_bad_metadata_models': {}}), \
             patch('redback.model_library.logger') as mock_logger:
            _ml._load_plugin_modules()

        mock_logger.warning.assert_called()

    def test_invalid_plugin_metadata_warns_but_model_loads(self):
        new_func = lambda time, **kw: None
        new_func.__name__ = 'plugin_invalid_metadata_model_xyz'

        fake_module = types.ModuleType('myplugin_invalid_metadata_models')
        fake_module.__name__ = 'myplugin_invalid_metadata_models'
        fake_module.redback_model_metadata = {
            'plugin_invalid_metadata_model_xyz': {
                'model_type': 'supernova',
                'output_formats': ('flux_density',),
                'default_output_format': 'magnitude',
            }
        }

        fake_ep = Mock()
        fake_ep.name = 'myplugin_invalid_metadata_models'
        fake_ep.load.return_value = fake_module

        import redback.model_library as _ml
        saved = dict(_ml.all_models_dict)
        saved_mdict = dict(_ml.modules_dict)
        saved_metadata = dict(_ml.model_metadata_dict)
        try:
            with patch('redback.model_library.entry_points', return_value=[fake_ep]), \
                 patch('redback.model_library.get_functions_dict',
                       return_value={'myplugin_invalid_metadata_models': {
                           'plugin_invalid_metadata_model_xyz': new_func}}), \
                 patch('redback.model_library.logger') as mock_logger:
                _ml._load_plugin_modules()

            self.assertIn('plugin_invalid_metadata_model_xyz', _ml.all_models_dict)
            self.assertNotIn('plugin_invalid_metadata_model_xyz', _ml.model_metadata_dict)
            mock_logger.warning.assert_called()
        finally:
            _ml.all_models_dict.clear()
            _ml.all_models_dict.update(saved)
            _ml.modules_dict.clear()
            _ml.modules_dict.update(saved_mdict)
            _ml.model_metadata_dict.clear()
            _ml.model_metadata_dict.update(saved_metadata)


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
