from .transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, mergernova_models, phase_models, prompt_models, \
    supernova_models, tde_models, integrated_flux_afterglow_models
from inspect import getmembers, isfunction

modules = [afterglow_models, extinction_models, kilonova_models, fireball_models,
           gaussianprocess_models, magnetar_models, mergernova_models,
           phase_models, prompt_models, supernova_models, tde_models, integrated_flux_afterglow_models]

all_models_dict = {}

for module in modules:
    _functions_list = [o for o in getmembers(module) if isfunction(o[1])]
    _functions_dict = {f[0]: f[1] for f in _functions_list}
    all_models_dict.update(_functions_dict)