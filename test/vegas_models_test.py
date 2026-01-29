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


class TestVegasModelsInterface:
    """Test Vegas models interface with mocked VegasAfterglow"""
    
    def setup_method(self):
        """Setup mock VegasAfterglow for each test"""
        self.mock_vegas = MagicMock()
        self.mock_cosmology = MagicMock()
        self.mock_dl = MagicMock()
        self.mock_dl.cgs.value = 1e28  # Mock luminosity distance
        self.mock_cosmology.luminosity_distance.return_value = self.mock_dl
        
        # Mock return values in CGS (erg/s/cm^2/Hz)
        self.mock_flux_cgs = np.array([1e-26, 2e-26, 3e-26])
        
    def test_vegas_tophat_flux_density_mode(self):
        """Test vegas_tophat calls VegasAfterglow correctly in flux_density mode"""
        from redback.transient_models.afterglow_models import vegas_tophat
        
        self.mock_vegas.LightCurve.return_value = self.mock_flux_cgs
        
        time = np.array([1.0, 10.0, 100.0])
        frequency = np.array([1e14, 1e14, 1e14])
        
        with patch.dict('sys.modules', {'VegasAfterglow': self.mock_vegas}), \
             patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', True), \
             patch('redback.transient_models.afterglow_models.vegas_models.cosmo', self.mock_cosmology):
            
            # Force reimport to pick up mocked module
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            result = vm.vegas_tophat(
                time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                lognism=0.0, loga=-np.inf, p=2.3, logepse=-1.0, 
                logepsb=-3.0, g0=300,
                frequency=frequency, output_format='flux_density'
            )
        
        # Check VegasAfterglow was called
        self.mock_vegas.LightCurve.assert_called_once()
        
        # Check result is converted from CGS to mJy
        expected_mjy = self.mock_flux_cgs / 1e-26
        np.testing.assert_array_almost_equal(result, expected_mjy)
    
    def test_vegas_tophat_magnitude_mode(self):
        """Test vegas_tophat handles magnitude output correctly"""
        from redback.transient_models.afterglow_models import vegas_tophat
        
        self.mock_vegas.LightCurve.return_value = self.mock_flux_cgs
        mock_frequency = 5e14
        
        time = np.array([1.0, 10.0])
        
        with patch.dict('sys.modules', {'VegasAfterglow': self.mock_vegas}), \
             patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', True), \
             patch('redback.transient_models.afterglow_models.vegas_models.cosmo', self.mock_cosmology), \
             patch('redback.transient_models.afterglow_models.vegas_models.bands_to_frequency', return_value=mock_frequency):
            
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            result = vm.vegas_tophat(
                time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                lognism=0.0, loga=-np.inf, p=2.3, logepse=-1.0, 
                logepsb=-3.0, g0=300,
                bands=['g', 'r'], output_format='magnitude'
            )
        
        # Should return magnitude values
        assert isinstance(result, np.ndarray)
        assert len(result) == len(time)
    
    def test_vegas_tophat_ism_medium(self):
        """Test vegas_tophat with pure ISM medium parameters"""
        self.mock_vegas.LightCurve.return_value = self.mock_flux_cgs
        
        with patch.dict('sys.modules', {'VegasAfterglow': self.mock_vegas}), \
             patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', True), \
             patch('redback.transient_models.afterglow_models.vegas_models.cosmo', self.mock_cosmology):
            
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            time = np.array([1.0])
            vm.vegas_tophat(
                time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                lognism=0.0, loga=-np.inf,  # Pure ISM
                p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
                frequency=np.array([1e14]), output_format='flux_density'
            )
        
        # Verify VegasAfterglow called with correct medium parameters
        call_args = self.mock_vegas.LightCurve.call_args
        assert call_args is not None
        
    def test_vegas_tophat_wind_medium(self):
        """Test vegas_tophat with pure Wind medium parameters"""
        self.mock_vegas.LightCurve.return_value = self.mock_flux_cgs
        
        with patch.dict('sys.modules', {'VegasAfterglow': self.mock_vegas}), \
             patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', True), \
             patch('redback.transient_models.afterglow_models.vegas_models.cosmo', self.mock_cosmology):
            
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            time = np.array([1.0])
            vm.vegas_tophat(
                time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                lognism=-np.inf, loga=11.0,  # Pure Wind
                p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
                frequency=np.array([1e14]), output_format='flux_density'
            )
        
        self.mock_vegas.LightCurve.assert_called_once()
    
    def test_vegas_tophat_hybrid_medium(self):
        """Test vegas_tophat with hybrid medium (wind + ISM floor)"""
        self.mock_vegas.LightCurve.return_value = self.mock_flux_cgs
        
        with patch.dict('sys.modules', {'VegasAfterglow': self.mock_vegas}), \
             patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', True), \
             patch('redback.transient_models.afterglow_models.vegas_models.cosmo', self.mock_cosmology):
            
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            time = np.array([1.0])
            vm.vegas_tophat(
                time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                lognism=0.0, loga=11.0,  # Hybrid
                p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
                frequency=np.array([1e14]), output_format='flux_density'
            )
        
        self.mock_vegas.LightCurve.assert_called_once()
    
    def test_vegas_tophat_optional_parameters(self):
        """Test vegas_tophat passes optional parameters correctly"""
        self.mock_vegas.LightCurve.return_value = self.mock_flux_cgs
        
        with patch.dict('sys.modules', {'VegasAfterglow': self.mock_vegas}), \
             patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', True), \
             patch('redback.transient_models.afterglow_models.vegas_models.cosmo', self.mock_cosmology):
            
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            time = np.array([1.0])
            vm.vegas_tophat(
                time=time, redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                lognism=0.0, loga=-np.inf, p=2.3, logepse=-1.0, logepsb=-3.0, g0=300,
                frequency=np.array([1e14]), output_format='flux_density',
                # Optional parameters
                spreading=True,
                reverse_shock=True,
                ssc=True,
                magnetar_L0=1e50,
                magnetar_t0=1000.0,
                magnetar_q=3.0
            )
        
        self.mock_vegas.LightCurve.assert_called_once()
    
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
    
    def test_vegas_tophat_raises_without_vegasafterglow(self):
        """Test vegas_tophat raises clear error when VegasAfterglow not installed"""
        with patch('redback.transient_models.afterglow_models.vegas_models.VEGASAFTERGLOW_AVAILABLE', False):
            import importlib
            import redback.transient_models.afterglow_models.vegas_models as vm
            importlib.reload(vm)
            
            with pytest.raises(ImportError, match="VegasAfterglow is not installed"):
                vm.vegas_tophat(
                    time=np.array([1.0]), redshift=0.1, thv=0.0, loge0=52.0, thc=0.1,
                    lognism=0.0, loga=-np.inf, p=2.3, logepse=-1.0, 
                    logepsb=-3.0, g0=300,
                    frequency=np.array([1e14]), output_format='flux_density'
                )


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