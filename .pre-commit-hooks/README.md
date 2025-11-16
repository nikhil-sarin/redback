# Pre-commit Hooks

This directory contains custom pre-commit hooks to maintain code quality and prevent common errors.

## Installation

To use these pre-commit hooks, first install pre-commit:

```bash
pip install pre-commit
```

Then install the git hooks:

```bash
pre-commit install
```

## Custom Hooks

### check_duplicates.py

Detects duplicate class and function definitions in Python files. This prevents bugs like:
- Multiple class definitions with the same name
- Duplicate function definitions
- Duplicate constant assignments

**Example issues prevented:**
- `MixtureGaussianLikelihood` class defined twice
- `StudentTLikelihood` class defined twice
- `solar_radius` constant assigned twice

### check_nan_comparison.py

Detects incorrect NaN comparisons that will never work as expected. In Python/NumPy, you cannot compare with `np.nan` using `==` or `!=` because NaN is not equal to anything, including itself.

**Incorrect:**
```python
mask = bandflux == np.nan  # This NEVER works!
```

**Correct:**
```python
mask = np.isnan(bandflux)
```

## Running Manually

You can run the pre-commit hooks manually on all files:

```bash
pre-commit run --all-files
```

Or on specific files:

```bash
pre-commit run --files redback/likelihoods.py
```

## Bypassing Hooks (Not Recommended)

In rare cases where you need to bypass the hooks:

```bash
git commit --no-verify
```

However, this is strongly discouraged as these hooks catch serious bugs.
