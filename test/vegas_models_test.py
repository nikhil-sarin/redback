"""
Test VegasAfterglow model integration
"""
import pytest
import numpy as np


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
    
    def test_vegas_models_callable(self):
        """Test Vegas models are callable functions"""
        from redback.model_library import all_models_dict
        
        vegas_models = [k for k in all_models_dict.keys() if k.startswith('vegas_')]
        
        for model_name in vegas_models:
            model_func = all_models_dict[model_name]
            assert callable(model_func), f"{model_name} is not callable"
            assert model_func.__doc__ is not None, f"{model_name} has no docstring"


@pytest.mark.skipif(True, reason="Requires VegasAfterglow installation")
class TestVegasModelsExecution:
    """Test Vegas models execute correctly (requires VegasAfterglow)"""
    
    def test_vegas_tophat_ism(self):
        """Test vegas_tophat with pure ISM medium"""
        from redback.model_library import all_models_dict
        
        model = all_models_dict['vegas_tophat']
        
        time = np.logspace(0, 2, 10)  # 1 to 100 days
        kwargs = {
            'frequency': np.full_like(time, 1e14),  # Optical
            'output_format': 'flux_density'
        }
        
        # Pure ISM: lognism=0.0, loga=-inf
        flux = model(
            time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
            lognism=0.0, loga=-np.inf, 
            p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
            **kwargs
        )
        
        assert len(flux) == len(time)
        assert np.all(flux > 0)
        assert np.all(np.isfinite(flux))
    
    def test_vegas_tophat_wind(self):
        """Test vegas_tophat with pure Wind medium"""
        from redback.model_library import all_models_dict
        
        model = all_models_dict['vegas_tophat']
        
        time = np.logspace(0, 2, 10)
        kwargs = {
            'frequency': np.full_like(time, 1e14),
            'output_format': 'flux_density'
        }
        
        # Pure Wind: lognism=-inf, loga=11.0
        flux = model(
            time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
            lognism=-np.inf, loga=11.0,
            p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
            **kwargs
        )
        
        assert len(flux) == len(time)
        assert np.all(flux > 0)
        assert np.all(np.isfinite(flux))
    
    def test_vegas_tophat_hybrid(self):
        """Test vegas_tophat with Hybrid medium (wind + ISM floor)"""
        from redback.model_library import all_models_dict
        
        model = all_models_dict['vegas_tophat']
        
        time = np.logspace(0, 2, 10)
        kwargs = {
            'frequency': np.full_like(time, 1e14),
            'output_format': 'flux_density'
        }
        
        # Hybrid: lognism=0.0, loga=11.0
        flux = model(
            time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
            lognism=0.0, loga=11.0,
            p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
            **kwargs
        )
        
        assert len(flux) == len(time)
        assert np.all(flux > 0)
        assert np.all(np.isfinite(flux))
