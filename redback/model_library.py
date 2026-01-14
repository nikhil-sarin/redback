from redback.transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, magnetar_driven_ejecta_models, phase_models, phenomenological_models, \
    prompt_models, shock_powered_models, supernova_models, tde_models, integrated_flux_afterglow_models, combined_models, \
    general_synchrotron_models, spectral_models, stellar_interaction_models

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

logger.debug("Building model library from transient model modules")
for module in modules:
    try:
        models_dict = get_functions_dict(module)
        modules_dict.update(models_dict)
        module_name = module.__name__.split('.')[-1]
        num_models = len(models_dict.get(module_name, {}))
        logger.debug(f"Loaded {num_models} models from {module_name}")
        for k, v in models_dict[module_name].items():
            all_models_dict[k] = v
    except Exception as e:
        logger.error(f"Failed to load models from {module.__name__}: {e}")
        raise

for mod in base_modules:
    try:
        models_dict = get_functions_dict(mod)
        module_name = mod.__name__.split('.')[-1]
        num_models = len(models_dict.get(module_name, {}))
        logger.debug(f"Loaded {num_models} base models from {module_name}")
        for k, v in models_dict[module_name].items():
            base_models_dict[k] = v
    except Exception as e:
        logger.error(f"Failed to load base models from {mod.__name__}: {e}")
        raise

logger.info(f"Model library initialized with {len(all_models_dict)} total models")