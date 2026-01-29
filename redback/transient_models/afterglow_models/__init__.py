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
from redback.transient_models.afterglow_models.base_models import *

# Explicitly import private functions that are used by other modules
from redback.transient_models.afterglow_models.base_models import (
    _get_kn_dynamics,
    _pnu_synchrotron
)

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
    # If VegasAfterglow is not installed, models won't be available
    import warnings
    warnings.warn(f"Could not import VegasAfterglow models: {e}")


