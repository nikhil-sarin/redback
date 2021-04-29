from inspect import getmembers, isfunction
from .transient_models import afterglow_models, \
    extinction_models, kilonova_models, fireball_models, \
    gaussianprocess_models, magnetar_models, mergernova_models, phase_models, prompt_models, \
    supernova_models, tde_models

modules = [afterglow_models,extinction_models, kilonova_models, fireball_models,
          gaussianprocess_models, magnetar_models, mergernova_models,
          phase_models, prompt_models,supernova_models, tde_models]

all_models_dict = []
modules_dict = {}
for module in modules:
    _functions_list = [o for o in getmembers(module) if isfunction(o[1])]
    _functions_dict = {f[0]: f[1] for f in _functions_list}
    all_models_dict.append(_functions_dict)
    modules_dict[module] = _functions_dict

print(all_models_dict)
print('hahah')
print(modules_dict)