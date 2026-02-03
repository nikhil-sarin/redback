# Redback Example Plugin

This is a minimal example plugin demonstrating how to extend Redback with custom models using the plugin system.

## Installation

To test this example plugin:

```bash
cd examples/example_plugin
pip install -e .
```

## Models Included

This example plugin provides three simple models:

1. **example_exponential_model**: Simple exponential decay
2. **example_gaussian_model**: Gaussian profile
3. **example_combined_model**: Combination of exponential and Gaussian

## Usage

After installation, the models are automatically available in Redback:

```python
import redback

# Check that plugin models are loaded
print('example_exponential_model' in redback.model_library.all_models_dict)

# Use the model
from redback.model_library import all_models_dict
import numpy as np

model = all_models_dict['example_exponential_model']
time = np.linspace(0, 10, 100)
flux = model(time, amplitude=10.0, decay_time=2.0)
```

## Plugin Structure

```
redback-example-plugin/
├── redback_example_plugin/
│   ├── __init__.py       # Package initialization
│   └── models.py         # Model functions
├── setup.py              # Package configuration with entry points
└── README.md            # This file
```

## Key Components

### Entry Points

In `setup.py`, the plugin is registered using:

```python
entry_points={
    'redback.model.modules': [
        'example_models = redback_example_plugin.models',
    ],
}
```

This tells Redback to load `redback_example_plugin.models` and make all its model functions available.

### Model Functions

Models are simple Python functions decorated with `@citation_wrapper`:

```python
from redback.utils import citation_wrapper
import numpy as np

@citation_wrapper('redback')
def example_exponential_model(time, amplitude, decay_time, **kwargs):
    return amplitude * np.exp(-time / decay_time)
```

## Creating Your Own Plugin

Use this example as a template:

1. Copy this directory structure
2. Rename `redback_example_plugin` to your plugin name
3. Update `setup.py` with your package name and details
4. Add your model functions to `models.py`
5. Install with `pip install -e .`

See `PLUGIN_GUIDE.md` in the main Redback repository for detailed instructions.
