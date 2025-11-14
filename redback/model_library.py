import importlib
import sys
import warnings
from typing import Dict, List, Any

from redback.transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, magnetar_driven_ejecta_models, phase_models, phenomenological_models, \
    prompt_models, shock_powered_models, supernova_models, tde_models, integrated_flux_afterglow_models, combined_models, \
    general_synchrotron_models, spectral_models, stellar_interaction_models

from redback.utils import get_functions_dict

# For Python 3.10+, use importlib.metadata
if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points


def discover_model_plugins() -> List[Any]:
    """
    Discover model modules from installed packages via entry points.

    External packages can register their model modules by adding an entry point
    in their setup.py or setup.cfg:

    setup(
        ...
        entry_points={
            'redback.model.modules': [
                'my_models = my_package.my_models_module',
            ]
        }
    )

    Returns:
        List of loaded model modules from plugins
    """
    plugin_modules = []

    try:
        # Get all entry points in the 'redback.model.modules' group
        eps = entry_points()

        # Handle different return types based on Python version
        if hasattr(eps, 'select'):
            # Python 3.10+
            model_eps = eps.select(group='redback.model.modules')
        else:
            # Python 3.9 and earlier (with importlib_metadata)
            model_eps = eps.get('redback.model.modules', [])

        for ep in model_eps:
            try:
                # Load the module from the entry point
                module = ep.load()
                plugin_modules.append(module)
                print(f"Loaded plugin model module: {ep.name} from {ep.value}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load plugin model module '{ep.name}': {str(e)}",
                    RuntimeWarning
                )
    except Exception as e:
        warnings.warn(
            f"Error discovering model plugins: {str(e)}",
            RuntimeWarning
        )

    return plugin_modules


def discover_base_model_plugins() -> List[Any]:
    """
    Discover base model modules from installed packages via entry points.

    Base models are used as building blocks for other models.

    External packages can register their base model modules by adding an entry point:

    setup(
        ...
        entry_points={
            'redback.model.base_modules': [
                'my_base_models = my_package.my_base_models_module',
            ]
        }
    )

    Returns:
        List of loaded base model modules from plugins
    """
    plugin_modules = []

    try:
        eps = entry_points()

        # Handle different return types based on Python version
        if hasattr(eps, 'select'):
            # Python 3.10+
            model_eps = eps.select(group='redback.model.base_modules')
        else:
            # Python 3.9 and earlier
            model_eps = eps.get('redback.model.base_modules', [])

        for ep in model_eps:
            try:
                module = ep.load()
                plugin_modules.append(module)
                print(f"Loaded plugin base model module: {ep.name} from {ep.value}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load plugin base model module '{ep.name}': {str(e)}",
                    RuntimeWarning
                )
    except Exception as e:
        warnings.warn(
            f"Error discovering base model plugins: {str(e)}",
            RuntimeWarning
        )

    return plugin_modules


# Built-in model modules
modules = [afterglow_models, extinction_models, fireball_models,
           gaussianprocess_models,  integrated_flux_afterglow_models, kilonova_models,
           magnetar_models, magnetar_driven_ejecta_models,
           phase_models, phenomenological_models, prompt_models, shock_powered_models, supernova_models,
           tde_models, combined_models, general_synchrotron_models, spectral_models, stellar_interaction_models]

base_modules = [extinction_models, phase_models]

# Discover and load plugin modules
plugin_modules = discover_model_plugins()
base_plugin_modules = discover_base_model_plugins()

# Combine built-in and plugin modules
all_modules = modules + plugin_modules
all_base_modules = base_modules + base_plugin_modules

# Build model dictionaries
all_models_dict = dict()
base_models_dict = dict()
modules_dict = dict()

# Process all model modules (built-in + plugins)
for module in all_modules:
    models_dict = get_functions_dict(module)
    modules_dict.update(models_dict)
    for k, v in models_dict[module.__name__.split('.')[-1]].items():
        all_models_dict[k] = v

# Process base model modules (built-in + plugins)
for mod in all_base_modules:
    models_dict = get_functions_dict(mod)
    for k, v in models_dict[mod.__name__.split('.')[-1]].items():
        base_models_dict[k] = v