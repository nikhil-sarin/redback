from redback.transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, magnetar_driven_ejecta_models, phase_models, phenomenological_models, \
    prompt_models, shock_powered_models, supernova_models, tde_models, integrated_flux_afterglow_models, combined_models, \
    general_synchrotron_models, spectral_models

from redback.utils import get_functions_dict

modules = [afterglow_models, extinction_models, fireball_models,
           gaussianprocess_models,  integrated_flux_afterglow_models, kilonova_models,
           magnetar_models, magnetar_driven_ejecta_models,
           phase_models, phenomenological_models, prompt_models, shock_powered_models, supernova_models,
           tde_models, combined_models, general_synchrotron_models, spectral_models]

all_models_dict = dict()
modules_dict = dict()
for module in modules:
    models_dict = get_functions_dict(module)
    modules_dict.update(models_dict)
    for k, v in models_dict[module.__name__.split('.')[-1]].items():
        all_models_dict[k] = v
