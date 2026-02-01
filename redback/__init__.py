from redback import analysis, constants, get_data, redback_errors, priors, result, sampler, transient, \
    transient_models, utils, photosphere, sed, interaction_processes, constraints, plotting, model_library, \
    simulate_transients
from redback.transient import afterglow, kilonova, prompt, supernova, tde
from redback.sampler import fit_model
from redback.utils import setup_logger

# Read version from setup.py to maintain single source of truth
import re
import os

def _get_version():
    """Extract version from setup.py"""
    setup_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup.py')
    try:
        with open(setup_path, 'r') as f:
            content = f.read()
            match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
            if match:
                return match.group(1)
    except (FileNotFoundError, IOError):
        pass
    return "unknown"

__version__ = _get_version()
setup_logger(log_level='info')