"""
Afterglow Models Subpackage
============================

GRB afterglow models organized by implementation backend.

Structure:
- base_models.py: Original redback afterglow models (afterglowpy wrappers, native implementations)
- vegas_models.py: VegasAfterglow C++ high-performance models

This subpackage maintains backward compatibility - all models can be imported from
redback.transient_models.afterglow_models as before.
"""

# Import everything from base_models (original afterglow_models.py)
try:
    from redback.transient_models.afterglow_models.base_models import *
    
    # Explicitly import private functions that are used by other modules
    from redback.transient_models.afterglow_models.base_models import (
        _get_kn_dynamics,
        _pnu_synchrotron
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import base afterglow models: {e}")
    # Define minimal stubs
    def _get_kn_dynamics(*args, **kwargs):
        raise ImportError("Base afterglow models not available")
    def _pnu_synchrotron(*args, **kwargs):
        raise ImportError("Base afterglow models not available")

# Import VegasAfterglow models
try:
    from redback.transient_models.afterglow_models.vegas_models import (
        vegas_tophat,
        vegas_gaussian,
        vegas_powerlaw,
        vegas_powerlaw_wing,
        vegas_two_component,
        vegas_step_powerlaw,
    )
except ImportError as e:
    # If VegasAfterglow is not installed, create placeholder functions for documentation
    import warnings
    warnings.warn(f"Could not import VegasAfterglow models: {e}")
    
    # Define placeholder functions with proper signatures so docs can build
    def vegas_tophat(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, **kwargs):
        """
        VegasAfterglow tophat jet with unified medium (ISM/Wind/Hybrid)
        
        Note: This is a placeholder. VegasAfterglow must be installed to use this model.
        Install with: pip install VegasAfterglow
        """
        raise ImportError("VegasAfterglow is not installed. Install with: pip install VegasAfterglow")
    
    def vegas_gaussian(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, **kwargs):
        """
        VegasAfterglow Gaussian jet with unified medium (ISM/Wind/Hybrid)
        
        Note: This is a placeholder. VegasAfterglow must be installed to use this model.
        Install with: pip install VegasAfterglow
        """
        raise ImportError("VegasAfterglow is not installed. Install with: pip install VegasAfterglow")
    
    def vegas_powerlaw(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, ke, kg, **kwargs):
        """
        VegasAfterglow power-law jet with unified medium (ISM/Wind/Hybrid)
        
        Note: This is a placeholder. VegasAfterglow must be installed to use this model.
        Install with: pip install VegasAfterglow
        """
        raise ImportError("VegasAfterglow is not installed. Install with: pip install VegasAfterglow")
    
    def vegas_powerlaw_wing(time, redshift, thv, loge0_w, thc, lognism, loga, p, logepse, logepsb, g0_w, ke, kg, **kwargs):
        """
        VegasAfterglow power-law wing jet with unified medium (ISM/Wind/Hybrid)
        
        Note: This is a placeholder. VegasAfterglow must be installed to use this model.
        Install with: pip install VegasAfterglow
        """
        raise ImportError("VegasAfterglow is not installed. Install with: pip install VegasAfterglow")
    
    def vegas_two_component(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, 
                            theta_w, loge0_w, g0_w, **kwargs):
        """
        VegasAfterglow two-component jet with unified medium (ISM/Wind/Hybrid)
        
        Note: This is a placeholder. VegasAfterglow must be installed to use this model.
        Install with: pip install VegasAfterglow
        """
        raise ImportError("VegasAfterglow is not installed. Install with: pip install VegasAfterglow")
    
    def vegas_step_powerlaw(time, redshift, thv, loge0, thc, lognism, loga, p, logepse, logepsb, g0, 
                            loge0_w, g0_w, ke, kg, **kwargs):
        """
        VegasAfterglow step power-law jet with unified medium (ISM/Wind/Hybrid)
        
        Note: This is a placeholder. VegasAfterglow must be installed to use this model.
        Install with: pip install VegasAfterglow
        """
        raise ImportError("VegasAfterglow is not installed. Install with: pip install VegasAfterglow")
