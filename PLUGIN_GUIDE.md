# Redback Plugin System Guide

Redback supports a plugin system that allows external packages to register custom model modules using Python entry points. This enables developers to extend Redback with new models without modifying the core codebase.

## Overview

The plugin system uses Python's setuptools entry points mechanism to discover and load model modules from installed packages. When Redback is imported, it automatically discovers and loads all registered plugin models, making them available in the `all_models_dict` alongside built-in models.

## Entry Point Groups

Redback defines two entry point groups for plugins:

1. **`redback.model.modules`**: For registering standard model modules
2. **`redback.model.base_modules`**: For registering base model modules (building blocks used by other models)

## Creating a Plugin Package

### Step 1: Create Your Model Module

Create a Python module with your model functions. Models should be simple functions that accept parameters and return results. Use the `@citation_wrapper` decorator if you want to track citations.

Example: `my_redback_plugin/models.py`

```python
from redback.utils import citation_wrapper
import numpy as np

@citation_wrapper('https://doi.org/10.1234/example')
def custom_kilonova_model(time, redshift, mass, velocity, **kwargs):
    """
    A custom kilonova model.

    :param time: Time array in days (observer frame)
    :param redshift: Source redshift
    :param mass: Ejecta mass in solar masses
    :param velocity: Ejecta velocity in c
    :param kwargs: Additional keyword arguments
    :return: Model output (flux, magnitude, etc. depending on output_format)
    """
    # Your model implementation here
    output = mass * velocity * np.exp(-time / (1 + redshift))
    return output


@citation_wrapper('https://doi.org/10.5678/another')
def custom_supernova_model(time, peak_magnitude, rise_time, **kwargs):
    """
    A custom supernova light curve model.

    :param time: Time array in days
    :param peak_magnitude: Peak magnitude
    :param rise_time: Rise time to peak in days
    :param kwargs: Additional keyword arguments
    :return: Magnitude as a function of time
    """
    # Your model implementation
    magnitude = peak_magnitude + 2.5 * np.log10(time / rise_time)
    return magnitude
```

### Step 2: Create Your Package Structure

Organize your package with the following structure:

```
my_redback_plugin/
├── my_redback_plugin/
│   ├── __init__.py
│   └── models.py          # Your model functions
├── setup.py               # Or pyproject.toml
└── README.md
```

### Step 3: Register Your Plugin

In your `setup.py`, register your model module using the `entry_points` parameter:

```python
from setuptools import setup

setup(
    name='my-redback-plugin',
    version='0.1.0',
    description='Custom models for Redback',
    packages=['my_redback_plugin'],
    install_requires=[
        'redback>=1.12.0',
        'numpy',
        'scipy',
    ],
    entry_points={
        'redback.model.modules': [
            'my_custom_models = my_redback_plugin.models',
        ],
    },
    python_requires='>=3.10',
)
```

Alternatively, using `pyproject.toml`:

```toml
[project]
name = "my-redback-plugin"
version = "0.1.0"
description = "Custom models for Redback"
requires-python = ">=3.10"
dependencies = [
    "redback>=1.12.0",
    "numpy",
    "scipy",
]

[project.entry-points."redback.model.modules"]
my_custom_models = "my_redback_plugin.models"
```

### Step 4: Install Your Plugin

Install your plugin package in the same environment as Redback:

```bash
pip install -e .  # For development
# or
pip install my-redback-plugin  # From PyPI or other source
```

## Using Plugin Models

Once installed, your plugin models are automatically available in Redback:

```python
import redback

# Your custom models are now in all_models_dict
print('custom_kilonova_model' in redback.model_library.all_models_dict)  # True

# Use them like any built-in model
from redback.sampler import fit_model
from redback.transient import OpticalTransient

transient = OpticalTransient.from_open_transient_catalog_data('SN2011fe')

# Fit using your custom model by name
result = fit_model(transient, model='custom_kilonova_model')

# Or use the function directly
from redback.model_library import all_models_dict
custom_model = all_models_dict['custom_kilonova_model']
output = custom_model(time=[1, 2, 3], redshift=0.01, mass=0.05, velocity=0.1)
```

## Base Models

If you're creating models that are meant to be used as building blocks (like extinction or phase models), register them as base models:

```python
setup(
    ...
    entry_points={
        'redback.model.base_modules': [
            'my_base_models = my_redback_plugin.base_models',
        ],
    },
)
```

Base models will be available in both `all_models_dict` and `base_models_dict`.

## Model Requirements

Your model functions should follow these conventions:

1. **Function Signature**: Accept `time` as the first parameter (or appropriate independent variable)
2. **Parameters**: Clearly document all parameters in the docstring
3. **Kwargs**: Accept `**kwargs` to handle additional arguments passed by the framework
4. **Return Value**: Return numerical data compatible with Redback's analysis pipeline
5. **Documentation**: Provide comprehensive docstrings
6. **Citations**: Use `@citation_wrapper` decorator to track relevant publications

## Example: Complete Plugin Package

Here's a complete minimal example:

**setup.py:**
```python
from setuptools import setup

setup(
    name='redback-exotic-models',
    version='1.0.0',
    description='Exotic transient models for Redback',
    author='Your Name',
    packages=['redback_exotic_models'],
    install_requires=['redback>=1.12.0', 'numpy'],
    entry_points={
        'redback.model.modules': [
            'exotic = redback_exotic_models.models',
        ],
    },
)
```

**redback_exotic_models/\_\_init\_\_.py:**
```python
__version__ = '1.0.0'
```

**redback_exotic_models/models.py:**
```python
import numpy as np
from redback.utils import citation_wrapper

@citation_wrapper('redback')
def exotic_powerlaw(time, amplitude, index, **kwargs):
    """
    Simple power law model.

    :param time: Time array
    :param amplitude: Amplitude
    :param index: Power law index
    :return: Power law curve
    """
    return amplitude * time ** index
```

After installing this package, the `exotic_powerlaw` model will be automatically available in Redback.

## Troubleshooting

### Plugin Not Loading

If your plugin doesn't appear:

1. Verify installation: `pip list | grep my-redback-plugin`
2. Check entry points: `python -m pip show -f my-redback-plugin`
3. Test import: `python -c "from my_redback_plugin import models"`
4. Check for errors: Look for warnings when importing redback

### Import Errors

Make sure all dependencies are installed and your module can be imported independently before registering it as a plugin.

### Naming Conflicts

If your model has the same name as a built-in model, the built-in model will be overwritten. Choose unique names for your models to avoid conflicts.

## Best Practices

1. **Versioning**: Keep your plugin version compatible with Redback versions
2. **Testing**: Include tests for your models
3. **Documentation**: Provide clear documentation and examples
4. **Dependencies**: Minimize additional dependencies where possible
5. **Performance**: Use NumPy/Numba for computational efficiency
6. **Validation**: Validate input parameters and provide helpful error messages

## Advanced: Multiple Modules

You can register multiple model modules from a single plugin:

```python
entry_points={
    'redback.model.modules': [
        'kilonova_variants = my_plugin.kilonova_models',
        'supernova_variants = my_plugin.supernova_models',
        'tde_variants = my_plugin.tde_models',
    ],
    'redback.model.base_modules': [
        'custom_extinction = my_plugin.extinction_models',
    ],
}
```

## Support

For questions or issues with the plugin system:
- Check the Redback documentation
- Open an issue on GitHub: https://github.com/nikhil-sarin/redback/issues
- Consult existing plugin examples in the ecosystem
