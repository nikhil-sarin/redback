# Redback Example Plugin

A minimal working example of a redback plugin. It demonstrates every piece a plugin author needs to provide: model functions, prior files, a prior loader, and the `setup.py` entry points that wire them into redback automatically.

After `pip install -e .`, the plugin's models and priors are available in redback with no further configuration.

---

## Quick start

```bash
cd examples/example_plugin
pip install -e .
```

```python
import redback

# Models are immediately available alongside all built-in models
print('example_exponential_model' in redback.model_library.all_models_dict)  # True

# Priors are immediately available via get_priors
priors = redback.priors.get_priors('example_exponential_model')
print(list(priors.keys()))  # ['amplitude', 'decay_time']

# Use the model exactly like any built-in redback model
import numpy as np
model = redback.model_library.all_models_dict['example_exponential_model']
flux = model(np.linspace(0, 10, 100), amplitude=10.0, decay_time=2.0)
```

---

## Directory structure explained

```
example_plugin/
├── setup.py                                   # Package config + entry points (see below)
└── redback_example_plugin/
    ├── __init__.py                            # Standard Python package init
    ├── models.py                              # Model functions — one function per model
    ├── priors.py                              # Prior loader: get_prior(model_name) -> PriorDict or None
    └── priors/                                # One .prior file per model
        ├── example_exponential_model.prior
        ├── example_gaussian_model.prior
        └── example_combined_model.prior
```

### `models.py` — model functions

Each model is a plain Python function. The only requirement is that `time` is the first positional argument and the function accepts `**kwargs` (so redback can pass extra arguments through without breaking your function).

```python
import numpy as np

def example_exponential_model(time, amplitude, decay_time, **kwargs):
    return amplitude * np.exp(-time / decay_time)
```

There are no decorators required. Redback discovers the functions automatically from the module.

### `priors/` — one `.prior` file per model

Each file is named `{model_name}.prior` and uses bilby's PriorDict file format: one parameter per line.

```
# example_exponential_model.prior
amplitude  = LogUniform(minimum=1e-3, maximum=1e3, name='amplitude', latex_label='$A$')
decay_time = LogUniform(minimum=0.1,  maximum=1000, name='decay_time', latex_label='$\tau$')
```

Supported prior types are anything bilby understands: `Uniform`, `LogUniform`, `Gaussian`, `LogNormal`, `PowerLaw`, `DeltaFunction`, `Constraint`, etc.

### `priors.py` — prior loader

This is the function that redback calls when it cannot find a prior file in its own built-in directories. It receives the model name as a string and must return a `bilby.core.prior.PriorDict` if it knows that model, or `None` to let redback continue searching.

```python
import os
from bilby.core.prior import PriorDict

PRIOR_DIR = os.path.join(os.path.dirname(__file__), 'priors')

def get_prior(model_name):
    path = os.path.join(PRIOR_DIR, f'{model_name}.prior')
    if not os.path.exists(path):
        return None
    p = PriorDict()
    p.from_file(path)
    return p
```

This implementation is completely generic — it works for any number of models as long as each has a corresponding `.prior` file in the `priors/` directory. You can copy it verbatim.

### `setup.py` — entry points

This is how redback discovers the plugin. Two entry point groups are used:

```python
entry_points={
    # Tell redback which module contains your model functions.
    # Format: 'any_unique_name = your_package.your_module'
    'redback.model.modules': [
        'example_models = redback_example_plugin.models',
    ],

    # Tell redback which callable to use for prior lookup.
    # Format: 'any_unique_name = your_package.your_module:callable_name'
    'redback.model.priors': [
        'example_priors = redback_example_plugin.priors:get_prior',
    ],
},
```

- **`redback.model.modules`** — redback imports the module and registers every function in it as an available model (added to `redback.model_library.all_models_dict`).
- **`redback.model.priors`** — redback calls `get_prior(model_name)` as a fallback after checking its own built-in prior directories. Return a `PriorDict` if you know the model, `None` otherwise.

The names on the left side of `=` (e.g. `example_models`, `example_priors`) are arbitrary unique identifiers — they just need to be unique across all installed plugins.

---

## Models included in this example

| Model | Parameters | Description |
|-------|-----------|-------------|
| `example_exponential_model` | `amplitude`, `decay_time` | Exponential decay `A * exp(-t/τ)` |
| `example_gaussian_model` | `peak_time`, `peak_amplitude`, `width` | Gaussian profile |
| `example_combined_model` | `amp_exp`, `decay_time`, `amp_gauss`, `peak_time`, `width` | Sum of exponential and Gaussian |

---

## Creating your own plugin from this template

1. Copy this directory and rename `redback_example_plugin` to your package name (e.g. `mypackage_redback`).
2. Update `setup.py`: change `name`, `description`, and the entry point strings to match your package name.
3. Replace the functions in `models.py` with your own model functions. Remember: first argument must be `time`, and the function must accept `**kwargs`.
4. For each model, add a `{model_name}.prior` file in `priors/`. The filename must exactly match the function name.
5. `priors.py` is generic and needs no changes unless you want custom prior-loading logic.
6. Run `pip install -e .` from the plugin directory, then `import redback` — your models and priors are live.

### Name collision behaviour

If a plugin model has the same name as a built-in redback model, the built-in always wins and redback logs a warning. Rename your model to avoid the conflict.

### Base models

If your models are intended to be used as `base_model` arguments in redback's extinction or phase wrappers, register the module under `redback.model.base_modules` instead of (or in addition to) `redback.model.modules`:

```python
entry_points={
    'redback.model.base_modules': [
        'example_base_models = redback_example_plugin.models',
    ],
    ...
}
```

### X-ray / high-energy spectral models

X-ray spectral models work through exactly the same entry point (`redback.model.modules`) — no separate group is needed. The only difference is the first argument must be `energies_keV` (or `energy_keV`) instead of `time`. Redback validates this at fit time and raises a helpful error if the signature is wrong.

```python
import numpy as np

def my_powerlaw_xray(energies_keV, norm, photon_index, **kwargs):
    """
    Simple power-law X-ray spectrum.

    :param energies_keV: Energy array in keV
    :param norm: Normalisation
    :param photon_index: Photon spectral index
    :return: Photon flux in photons/s/cm^2/keV
    """
    return norm * energies_keV ** (-photon_index)
```

Register it identically to any other model:

```python
entry_points={
    'redback.model.modules': [
        'my_xray_models = my_plugin.xray_models',
    ],
    'redback.model.priors': [
        'my_xray_priors = my_plugin.priors:get_prior',
    ],
},
```

Then fit as usual with a `CountsSpectrumTransient` and redback will pick up the model from `all_models_dict` and route it through the spectral fitting path automatically.
