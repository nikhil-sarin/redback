"""
Afterglow Models Subpackage
============================

GRB afterglow models organized by implementation backend.

This subpackage currently serves as a compatibility layer that:
1. Re-exports everything from the original afterglow_models.py file
2. Adds new VegasAfterglow models
"""

# Import the original afterglow_models.py using relative import from parent package
# We need to import it as a sibling module
import importlib
import sys

# Load the original afterglow_models.py file as a module
original_module_name = 'redback.transient_models._afterglow_models_original'
original_file = __file__.replace('afterglow_models/__init__.py', 'afterglow_models.py')

spec = importlib.util.spec_from_file_location(original_module_name, original_file)
_original = importlib.util.module_from_spec(spec)
sys.modules[original_module_name] = _original
spec.loader.exec_module(_original)

# Re-export everything (including private functions)
for name in dir(_original):
    if name != '__builtins__':
        globals()[name] = getattr(_original, name)

# Now add VegasAfterglow models
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
    # If VegasAfterglow is not installed, models won't be available
    import warnings
    warnings.warn(f"Could not import VegasAfterglow models: {e}")


