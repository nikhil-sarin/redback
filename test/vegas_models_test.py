"""
Test VegasAfterglow model integration
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call


class TestAfterglowModelsInit:
    """Test afterglow_models __init__.py imports and fallbacks"""
    
    def test_base_models_imported(self):
        """Test that base models are imported"""
        from redback.transient_models import afterglow_models
        
        # Check some key base models are available
        assert hasattr(afterglow_models, 'tophat')
        assert hasattr(afterglow_models, 'gaussian_redback')
        assert hasattr(afterglow_models, 'jetsimpy_tophat')
        
        # Actually call them to ensure they're imported
        assert callable(afterglow_models.tophat)
        assert callable(afterglow_models.gaussian_redback)
        assert callable(afterglow_models.jetsimpy_tophat)
        
    def test_private_functions_imported(self):
        """Test that private functions are available"""
        from redback.transient_models import afterglow_models
        
        assert hasattr(afterglow_models, '_get_kn_dynamics')
        assert hasattr(afterglow_models, '_pnu_synchrotron')
        assert callable(afterglow_models._get_kn_dynamics)
        assert callable(afterglow_models._pnu_synchrotron)
    
    def test_vegas_models_imported(self):
        """Test that vegas models are available"""
        from redback.transient_models import afterglow_models
        
        vegas_models = [
            'vegas_tophat', 'vegas_gaussian', 'vegas_powerlaw',
            'vegas_powerlaw_wing', 'vegas_two_component', 'vegas_step_powerlaw'
        ]
        
        for model in vegas_models:
            assert hasattr(afterglow_models, model)
            func = getattr(afterglow_models, model)
            assert callable(func)
    
    def test_all_public_base_models_accessible(self):
        """Test that all public base models are accessible through __init__"""
        from redback.transient_models import afterglow_models
        from redback.transient_models.afterglow_models import base_models
        
        # Get all public functions from base_models
        base_model_functions = [name for name in dir(base_models) 
                               if not name.startswith('_') and callable(getattr(base_models, name))]
        
        # Check a sample of them are accessible
        sample_models = ['tophat', 'gaussian_redback', 'cocoon', 'cone_afterglow',
                        'gaussian', 'kn_afterglow', 'kilonova_afterglow_redback']
        
        for model in sample_models:
            if model in base_model_functions:
                assert hasattr(afterglow_models, model), f"{model} not accessible from __init__"
                assert callable(getattr(afterglow_models, model))


class TestVegasModelsInterface:
    """Test Vegas models interface and parameter handling"""
    
    def test_all_vegas_models_callable(self):
        """Test all 6 Vegas model variants are callable"""
        from redback.transient_models.afterglow_models import (
            vegas_tophat, vegas_gaussian, vegas_powerlaw,
            vegas_powerlaw_wing, vegas_two_component, vegas_step_powerlaw
        )
        
        models = [vegas_tophat, vegas_gaussian, vegas_powerlaw,
                  vegas_powerlaw_wing, vegas_two_component, vegas_step_powerlaw]
        
        for model in models:
            assert callable(model)
            assert model.__doc__ is not None
            assert 'VegasAfterglow' in model.__doc__
    
    def test_vegas_models_parameter_signatures(self):
        """Test Vegas models have correct parameter signatures"""
        from redback.transient_models.afterglow_models import vegas_tophat
        import inspect
        
        sig = inspect.signature(vegas_tophat)
        params = list(sig.parameters.keys())
        
        required_params = ['time', 'redshift', 'thv', 'loge0', 'thc', 
                          'lognism', 'loga', 'p', 'logepse', 'logepsb', 'g0']
        
        for param in required_params:
            assert param in params, f"Missing required parameter: {param}"
    
    def test_vegas_tophat_docstring_parameters(self):
        """Test vegas_tophat docstring documents all parameters"""
        from redback.transient_models.afterglow_models import vegas_tophat
        
        doc = vegas_tophat.__doc__
        required_params = ['time', 'redshift', 'thv', 'loge0', 'thc', 
                          'lognism', 'loga', 'p', 'logepse', 'logepsb', 'g0']
        
        for param in required_params:
            assert f':param {param}:' in doc, f"Parameter {param} not documented"
        
        assert ':return:' in doc, "Return value not documented"
        assert 'mJy' in doc, "Units not documented in docstring"
    
    def test_vegas_tophat_raises_without_vegasafterglow(self):
        """Test vegas_tophat raises clear error when VegasAfterglow not installed"""
        from redback.transient_models.afterglow_models.vegas_models import VEGASAFTERGLOW_AVAILABLE
        
        # Only test if VegasAfterglow is actually not available
        if not VEGASAFTERGLOW_AVAILABLE:
            from redback.transient_models.afterglow_models import vegas_tophat
            
            with pytest.raises(ImportError, match="VegasAfterglow is not installed"):
                vegas_tophat(
                    time=np.array([1.0]), redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                    lognism=0.0, loga=-np.inf, p=2.3, logepse=-1.0, 
                    logepsb=-3.0, g0=300,
                    frequency=np.array([1e14]), output_format='flux_density'
                )
        else:
            pytest.skip("VegasAfterglow is installed")
    
    def test_all_vegas_models_have_consistent_docstrings(self):
        """Test all Vegas models have consistent, complete docstrings"""
        from redback.transient_models import afterglow_models
        
        vegas_models = [
            'vegas_tophat', 'vegas_gaussian', 'vegas_powerlaw',
            'vegas_powerlaw_wing', 'vegas_two_component', 'vegas_step_powerlaw'
        ]
        
        for model_name in vegas_models:
            model = getattr(afterglow_models, model_name)
            doc = model.__doc__
            
            assert doc is not None, f"{model_name} missing docstring"
            assert len(doc) > 100, f"{model_name} docstring too short"
            assert ':param time:' in doc, f"{model_name} missing time parameter doc"
            assert ':param redshift:' in doc, f"{model_name} missing redshift parameter doc"
            assert ':return:' in doc, f"{model_name} missing return doc"
            assert 'VegasAfterglow' in doc, f"{model_name} doesn't mention VegasAfterglow"
            assert 'mJy' in doc, f"{model_name} doesn't document units"


class TestVegasModelsRegistration:
    """Test that Vegas models are properly registered"""
    
    def test_vegas_models_in_library(self):
        """Test all Vegas models are in model library"""
        from redback.model_library import all_models_dict
        
        expected_models = [
            'vegas_tophat',
            'vegas_gaussian', 
            'vegas_powerlaw',
            'vegas_powerlaw_wing',
            'vegas_two_component',
            'vegas_step_powerlaw'
        ]
        
        for model_name in expected_models:
            assert model_name in all_models_dict, f"{model_name} not in model library"
    
    def test_vegas_models_callable_from_library(self):
        """Test Vegas models from library are callable"""
        from redback.model_library import all_models_dict
        
        vegas_models = [k for k in all_models_dict.keys() if k.startswith('vegas_')]
        
        assert len(vegas_models) == 6, f"Expected 6 vegas models, found {len(vegas_models)}"
        
        for model_name in vegas_models:
            model_func = all_models_dict[model_name]
            assert callable(model_func), f"{model_name} is not callable"
            assert model_func.__doc__ is not None, f"{model_name} has no docstring"