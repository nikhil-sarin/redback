from dataclasses import replace
from importlib.metadata import entry_points

from redback.transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, magnetar_driven_ejecta_models, phase_models, phenomenological_models, \
    prompt_models, shock_powered_models, supernova_models, tde_models, integrated_flux_afterglow_models, combined_models, \
    general_synchrotron_models, spectral_models, stellar_interaction_models

from redback.model_metadata import BUILTIN_MODEL_METADATA, coerce_model_metadata
from redback.utils import get_functions_dict, logger

modules = [afterglow_models, extinction_models, fireball_models,
           gaussianprocess_models,  integrated_flux_afterglow_models, kilonova_models,
           magnetar_models, magnetar_driven_ejecta_models,
           phase_models, phenomenological_models, prompt_models, shock_powered_models, supernova_models,
           tde_models, combined_models, general_synchrotron_models, spectral_models, stellar_interaction_models]

base_modules = [extinction_models, phase_models]

all_models_dict = dict()
base_models_dict = dict()
modules_dict = dict()
plugin_module_model_types = dict()
model_metadata_dict = dict(BUILTIN_MODEL_METADATA)
builtin_module_names = {module.__name__.split('.')[-1] for module in modules}
_DEFAULT_METADATA_SENTINEL = object()
for module in modules:
    models_dict = get_functions_dict(module, public_only=True)
    modules_dict.update(models_dict)
    for k, v in models_dict[module.__name__.split('.')[-1]].items():
        all_models_dict[k] = v
for mod in base_modules:
    models_dict = get_functions_dict(mod, public_only=True)
    for k, v in models_dict[mod.__name__.split('.')[-1]].items():
        base_models_dict[k] = v


def _get_plugin_model_types(module):
    """Return model-type metadata declared by a plugin module."""
    raw_model_types = getattr(module, 'redback_model_types', None)
    if raw_model_types is None:
        raw_model_types = getattr(module, 'redback_model_type', None)
    if raw_model_types is None:
        return set()
    if isinstance(raw_model_types, str):
        return {raw_model_types}
    try:
        return set(raw_model_types)
    except TypeError:
        return {raw_model_types}


def _get_plugin_model_metadata(module):
    """Return model metadata declared by a plugin module."""
    raw_metadata = getattr(module, 'redback_model_metadata', None)
    if raw_metadata is None:
        return {}
    if not isinstance(raw_metadata, dict):
        raise TypeError("redback_model_metadata must be a dictionary keyed by model name.")
    return {
        name: coerce_model_metadata(name, metadata)
        for name, metadata in raw_metadata.items()
    }


def get_model_metadata(model_name, default=_DEFAULT_METADATA_SENTINEL):
    """Return structured metadata for a model function."""
    try:
        return model_metadata_dict[model_name]
    except KeyError:
        if default is not _DEFAULT_METADATA_SENTINEL:
            return default
        raise


def _load_plugin_modules():
    """Load plugin model modules registered via entry points."""
    for group, is_base in (('redback.model.modules', False), ('redback.model.base_modules', True)):
        eps = entry_points(group=group)
        for ep in eps:
            try:
                module = ep.load()
            except Exception as e:
                logger.warning(f"Failed to load plugin module '{ep.name}' from group '{group}': {e}")
                continue

            try:
                leaf = module.__name__.split('.')[-1]
                plugin_models = get_functions_dict(module)
                plugin_funcs = plugin_models.get(leaf, {})

                # Built-in models win on collision
                loaded_plugin_model_names = set()
                for k, v in plugin_funcs.items():
                    if k in all_models_dict:
                        logger.warning(
                            f"Plugin model '{k}' from '{ep.name}' conflicts with a built-in model. "
                            f"Skipping plugin model."
                        )
                    else:
                        all_models_dict[k] = v
                        loaded_plugin_model_names.add(k)
                        if is_base:
                            base_models_dict[k] = v

                try:
                    plugin_metadata = _get_plugin_model_metadata(module)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Invalid plugin metadata from '{ep.name}'. Skipping metadata: {e}")
                    plugin_metadata = {}
                for k, metadata in plugin_metadata.items():
                    if k in loaded_plugin_model_names:
                        if metadata.source_module is None:
                            metadata = replace(metadata, source_module=ep.name)
                        model_metadata_dict[k] = metadata
                    elif k not in plugin_funcs:
                        logger.warning(
                            f"Plugin metadata for '{k}' from '{ep.name}' has no matching plugin model. "
                            f"Skipping metadata."
                        )

                # Key by ep.name to avoid collisions between plugins with the same module leaf name
                modules_dict[ep.name] = plugin_funcs
                plugin_module_model_types[ep.name] = _get_plugin_model_types(module)
                logger.info(f"Loaded plugin module '{ep.name}' with {len(plugin_funcs)} model(s).")
            except Exception as e:
                logger.warning(f"Failed to process plugin module '{ep.name}': {e}")


_load_plugin_modules()


def discover_prior_plugins() -> dict:
    """
    Returns dict mapping entry_point_name -> callable(model_name) -> PriorDict or None.

    Plugins register via entry point group 'redback.model.priors'.
    The registered object must be a callable that accepts a model name string
    and returns a bilby PriorDict or None if the model is not known to that plugin.
    """
    providers = {}
    eps = entry_points(group='redback.model.priors')
    for ep in eps:
        try:
            provider = ep.load()
            if not callable(provider):
                logger.warning(f"Plugin prior provider '{ep.name}' is not callable. Skipping.")
                continue
            providers[ep.name] = provider
            logger.info(f"Loaded plugin prior provider '{ep.name}'.")
        except Exception as e:
            logger.warning(f"Failed to load plugin prior provider '{ep.name}': {e}")
    return providers


# List of plugin prior provider callables, each taking a model name and returning a PriorDict or None
plugin_prior_providers = list(discover_prior_plugins().values())
