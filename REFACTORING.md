# Redback Refactoring Guide

This document describes the refactoring work done to improve code quality and maintainability in the redback package.

## Overview

A comprehensive analysis of the redback codebase identified several opportunities for refactoring to reduce code duplication and improve maintainability:

- **196 duplicated output format handling blocks** (~2,000 lines of duplication)
- **554 scattered parameter extraction calls**
- **11 paired bolometric/wrapper function pairs** (~300 lines)
- **6 complex functions >200 lines**
- Various naming inconsistencies and code organization issues

## New Utilities Added

### `redback/model_utils.py`

A new utility module has been added to provide common functions used across multiple model files. This helps maintain consistency and reduces code duplication.

#### Available Functions

##### `setup_optical_depth_defaults(kwargs)`

Reduces the repetition of these 3 lines that appear in ~100+ model functions:

```python
kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
```

**Usage:**

```python
from redback.model_utils import setup_optical_depth_defaults

def my_model(time, redshift, param1, param2, **kwargs):
    # Instead of manually setting defaults:
    # kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    # kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    # kwargs['sed'] = kwargs.get("sed", sed.Blackbody)

    # Use the utility:
    setup_optical_depth_defaults(kwargs)

    # Now kwargs has the defaults set
    cosmology, dl = get_cosmology_defaults(redshift, kwargs)
    # ... rest of model
```

##### `get_cosmology_defaults(redshift, kwargs)`

Extracts cosmology from kwargs and computes luminosity distance. This pattern appears in ~100+ model functions.

**Returns:** `(cosmology_object, luminosity_distance_in_cgs)`

**Usage:**

```python
from redback.model_utils import get_cosmology_defaults

cosmology, dl = get_cosmology_defaults(redshift, kwargs)
# Now you have:
# - cosmology: astropy cosmology object (defaults to Planck18)
# - dl: luminosity distance in cgs units
```

##### `setup_photosphere_sed_defaults(kwargs)`

Setup default photosphere and SED classes. Returns tuple of `(photosphere_class, sed_class)`.

**Usage:**

```python
from redback.model_utils import setup_photosphere_sed_defaults

photosphere_class, sed_class = setup_photosphere_sed_defaults(kwargs)
```

##### `compute_photosphere_and_sed(time, lbol, frequency, photosphere_class, sed_class, dl, **kwargs)`

Common pattern for computing photosphere properties and SED. This 3-line pattern appears in almost every model function:

```python
photo = photosphere_class(time=time, luminosity=lbol, **kwargs)
sed_obj = sed_class(temperature=photo.photosphere_temperature,
                   r_photosphere=photo.r_photosphere,
                   frequency=frequency, luminosity_distance=dl)
```

**Returns:** `(photosphere_object, sed_object)`

**Usage:**

```python
from redback.model_utils import compute_photosphere_and_sed

photo, sed_obj = compute_photosphere_and_sed(
    time, lbol, frequency, photosphere_class, sed_class, dl, **kwargs)

# Now you can use:
# - photo.photosphere_temperature
# - photo.r_photosphere
# - sed_obj.flux_density
```

## Migration Guide for Model Functions

Here's an example of how to refactor a model function to use the new utilities:

### Before:

```python
def arnett(time, redshift, f_nickel, mej, **kwargs):
    # Manual parameter setup (4 lines)
    kwargs['interaction_process'] = kwargs.get("interaction_process", ip.Diffusion)
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, **kwargs)

        # Manual photosphere and SED computation (3 lines)
        photo = kwargs['photosphere'](time=time, luminosity=lbol, **kwargs)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature,
                             r_photosphere=photo.r_photosphere,
                             frequency=frequency, luminosity_distance=dl)
        flux_density = sed_1.flux_density
        return flux_density.to(uu.mJy).value / (1 + redshift)
    # ... rest of function
```

### After:

```python
from redback.model_utils import (setup_optical_depth_defaults, get_cosmology_defaults,
                                  compute_photosphere_and_sed)

def arnett(time, redshift, f_nickel, mej, **kwargs):
    # Cleaner parameter setup (2 lines instead of 5)
    setup_optical_depth_defaults(kwargs)
    cosmology, dl = get_cosmology_defaults(redshift, kwargs)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej, **kwargs)

        # Cleaner photosphere and SED computation (1 line instead of 3)
        photo, sed_obj = compute_photosphere_and_sed(
            time, lbol, frequency, kwargs['photosphere'], kwargs['sed'], dl, **kwargs)

        return sed_obj.flux_density.to(uu.mJy).value / (1 + redshift)
    # ... rest of function
```

### Benefits:
- **7 lines reduced to 3 lines** in this example
- More readable and maintainable
- Consistent behavior across all model functions
- Easier to update defaults in one place

## Future Refactoring Opportunities

The comprehensive analysis (see `CODEBASE_ANALYSIS.md`) identified additional refactoring opportunities:

### High Priority (High Impact, Medium Effort)

1. **Create decorator pattern for bolometric/wrapper functions** - Would eliminate ~300 lines of duplicated wrapper logic
2. **Extract output format handling into a handler class** - Would address the 196 duplicated output format blocks
3. **Split large model files** - `supernova_models.py` (2,849 lines) could be split into 4 files

### Medium Priority (Medium Impact, Low Effort)

1. **Consolidate K-correction pattern** - 48 scattered calls could be unified
2. **Create photosphere/SED selector utilities** - Consolidate setup code

### Lower Priority (Good Practice)

1. **Extract intermediate variables in long functions** - Improve readability of 6 functions >200 lines
2. **Reduce parameter lists using dataclasses** - Functions with 10+ parameters could use dataclasses

## Code Quality Improvements

### Metrics

| Metric | Before | Target | Status |
|--------|---------|--------|--------|
| Duplicated parameter setup lines | 554 | <100 | **In Progress** |
| Average function length | 64 lines | <50 lines | Future work |
| Longest function | 350 lines | <200 lines | Future work |
| Code duplication | ~2,500 lines | <500 lines | Future work |

## Testing

Unit tests for the new utilities are located in `test/model_utils_test.py`. These tests cover:

- Default parameter setup
- Cosmology and luminosity distance calculation
- Photosphere and SED class selection
- Photosphere and SED object computation

Run tests with:
```bash
python -m pytest test/model_utils_test.py -v
```

## Contributing

When adding new model functions, please:

1. **Use the utility functions** from `model_utils.py` instead of duplicating code
2. **Follow consistent patterns** as shown in the migration guide above
3. **Add tests** for any new functionality
4. **Keep functions focused** - aim for <100 lines per function
5. **Document parameters** clearly in docstrings

## References

- See `CODEBASE_ANALYSIS.md` for detailed analysis of refactoring opportunities
- See `REFACTORING_EXAMPLES.md` for concrete code examples with line numbers
