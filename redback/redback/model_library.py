from .transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, mergernova_models, phase_models, prompt_models, \
    supernova_models, tde_models
from .utils import get_functions_dict

modules = [afterglow_models,extinction_models, kilonova_models, fireball_models,
          gaussianprocess_models, magnetar_models, mergernova_models,
          phase_models, prompt_models,supernova_models, tde_models]

all_models_dict = []
modules_dict = {}
for module in modules:
    all_modules, modules_list = get_functions_dict(module)
    all_models_dict.append(all_modules)
    modules_dict.update(modules_list)
